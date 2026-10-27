//! SSE and SSE2 for the interpreter (the JIT runs these through `helper_exec`).
//!
//! Floating-point arithmetic and conversions never touch the host FPU: x86 NaN selection, the
//! default NaN, DAZ/FTZ, MXCSR.RC, the six exception flags and the unmasked-exception fault have to
//! be identical on aarch64 and x86_64 hosts, so a small soft-float engine on raw bits does the work
//! and the instruction arms only pick lanes. An unmasked exception sets its MXCSR flag, then
//! faults (#XM, or #UD without CR4.OSXMMEXCPT) before the destination is written.
//!
//! Known limits:
//! - rcpps/rsqrtps (and the scalar forms) return the correctly rounded result, where hardware
//!   returns an approximation with ~12 bits, so low mantissa bits differ.
//! - Tininess is detected after rounding to the subnormal grid (see `round_pack`), which differs
//!   from x86 only for values within half a subnormal ulp of the smallest normal number.
//! - A subnormal operand raises the denormal flag even when the other operand is a NaN, as in
//!   QEMU.
//! - SSE3 and later live in `sse_ext.rs` (a child module). No maskmovdqu.
//! - MMX: mmN is the low 64 bits of physical x87 register N (`Cpu::st`); see `Cpu::mmx`. Pending
//!   x87 exceptions (#MF) are not modelled.

use super::decode::*;
use super::exception::Exception;
use super::flags::*;
use super::sys::{CR0_EM, CR0_TS, CR4_OSFXSR, CR4_OSXMMEXCPT};
use super::*;

/// A vector register viewed as `N` lanes of this type; lane 0 is the least significant.
trait Lane: Copy {
    const N: usize;
    fn get(v: u128, i: usize) -> Self;
    /// ORs into the lane, so `v` must start zeroed (or the lane must be known to be clear).
    fn put(v: &mut u128, i: usize, x: Self);
}

macro_rules! lane {
    ($t:ty, $n:expr) => {
        impl Lane for $t {
            const N: usize = $n;
            fn get(v: u128, i: usize) -> Self {
                (v >> (i * (128 / $n))) as $t
            }
            fn put(v: &mut u128, i: usize, x: Self) {
                *v |= (x as u128) << (i * (128 / $n));
            }
        }
    };
}
lane!(u8, 16);
lane!(u16, 8);
lane!(u32, 4);
lane!(u64, 2);

fn zip_lanes<T: Lane>(a: u128, b: u128, f: impl Fn(T, T) -> T) -> u128 {
    let mut r = 0;
    for i in 0..T::N {
        T::put(&mut r, i, f(T::get(a, i), T::get(b, i)));
    }
    r
}

/// punpckl*/punpckh*: alternates lanes of `a` and `b` taken from the low or high half of each.
fn interleave<T: Lane>(a: u128, b: u128, hi: bool) -> u128 {
    let mut r = 0;
    let off = if hi { T::N / 2 } else { 0 };
    for i in 0..T::N / 2 {
        T::put(&mut r, 2 * i, T::get(a, off + i));
        T::put(&mut r, 2 * i + 1, T::get(b, off + i));
    }
    r
}

/// packsswb/packuswb (`source_bits` 16) and packssdw (`source_bits` 32): saturating narrow of the
/// signed lanes of `a` into the low half of the result, then those of `b` into the high half.
fn pack_sat(a: u128, b: u128, source_bits: usize, signed_out: bool) -> u128 {
    let mut r = 0u128;
    let get = |v: u128, i: usize| -> i32 {
        if source_bits == 16 {
            u16::get(v, i) as i16 as i32
        } else {
            u32::get(v, i) as i32
        }
    };
    let n = 128 / source_bits;
    for k in 0..2 * n {
        let x = get(if k < n { a } else { b }, k % n);
        if source_bits == 16 {
            let y = if signed_out {
                x.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8 as u8
            } else {
                x.clamp(0, i32::from(u8::MAX)) as u8
            };
            u8::put(&mut r, k, y);
        } else {
            let y = if signed_out {
                x.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16 as u16
            } else {
                x.clamp(0, i32::from(u16::MAX)) as u16
            };
            u16::put(&mut r, k, y);
        }
    }
    r
}

/// MXCSR exception flag bits (0-5); the matching mask bits are the same positions shifted by
/// `MXCSR_MASK_SHIFT`.
const FLAG_INVALID: u32 = 1 << 0;
const FLAG_DENORMAL: u32 = 1 << 1;
const FLAG_DIVIDE_BY_ZERO: u32 = 1 << 2;
const FLAG_OVERFLOW: u32 = 1 << 3;
const FLAG_UNDERFLOW: u32 = 1 << 4;
const FLAG_PRECISION: u32 = 1 << 5;
const MXCSR_EXCEPTION_FLAGS: u32 = 0x3f;
const MXCSR_MASK_SHIFT: u32 = 7;
const MXCSR_DAZ: u32 = 1 << 6;
const MXCSR_UNDERFLOW_MASK: u32 = 1 << 11;
const MXCSR_FTZ: u32 = 1 << 15;
/// MXCSR.RC field, bits 14:13.
const MXCSR_RC_SHIFT: u32 = 13;
const MXCSR_RC_MASK: u32 = 3;
const ROUND_NEAREST_EVEN: u32 = 0;
const ROUND_DOWN: u32 = 1;
const ROUND_UP: u32 = 2;
const ROUND_TOWARD_ZERO: u32 = 3;
/// #XM, raised for an unmasked SIMD floating-point exception when CR4.OSXMMEXCPT is set.
const VECTOR_SIMD_FLOATING_POINT: u8 = 19;

/// Rounding mode, DAZ/FTZ and the exception flags one instruction raises. The flags are merged
/// into MXCSR (and checked against the masks) by `Cpu::fp_commit` only after every lane is done,
/// so an unmasked exception leaves the destination untouched.
struct FpEnv {
    rounding: u32,
    denormals_are_zero: bool,
    flush_to_zero: bool,
    underflow_masked: bool,
    flags: u32,
}

impl FpEnv {
    fn new(mxcsr: u32) -> Self {
        FpEnv {
            rounding: (mxcsr >> MXCSR_RC_SHIFT) & MXCSR_RC_MASK,
            denormals_are_zero: mxcsr & MXCSR_DAZ != 0,
            flush_to_zero: mxcsr & MXCSR_FTZ != 0,
            underflow_masked: mxcsr & MXCSR_UNDERFLOW_MASK != 0,
            flags: 0,
        }
    }
}

/// An IEEE binary format, parameterised so one set of arithmetic serves f32 and f64. Values travel
/// as raw bits in a u64: the host FPU never sees them, because x86 NaN selection, the default NaN,
/// DAZ/FTZ, directed rounding and the exception flags cannot be taken from it portably.
#[derive(Clone, Copy)]
struct Format {
    exponent_bits: u32,
    fraction_bits: u32,
}

/// Width of an XMM register.
const VECTOR_BITS: u32 = 128;
/// SSE memory operands of 16 bytes or more must be aligned to this many bytes (#GP(0) otherwise).
const VECTOR_ALIGNMENT: u64 = 16;

const SINGLE: Format = Format {
    exponent_bits: 8,
    fraction_bits: 23,
};
const DOUBLE: Format = Format {
    exponent_bits: 11,
    fraction_bits: 52,
};

impl Format {
    fn width(self) -> u32 {
        1 + self.exponent_bits + self.fraction_bits
    }
    fn bytes(self) -> u8 {
        (self.width() / 8) as u8
    }
    fn lanes(self) -> usize {
        (VECTOR_BITS / self.width()) as usize
    }
    fn exponent_mask(self) -> u64 {
        (1 << self.exponent_bits) - 1
    }
    fn fraction_mask(self) -> u64 {
        (1 << self.fraction_bits) - 1
    }
    fn lane_mask(self) -> u128 {
        (1u128 << self.width()) - 1
    }
    fn bias(self) -> i32 {
        (1 << (self.exponent_bits - 1)) - 1
    }
    /// Unbiased exponent of the smallest normal number.
    fn min_exponent(self) -> i32 {
        1 - self.bias()
    }
    /// Unbiased exponent of the largest finite number.
    fn max_exponent(self) -> i32 {
        self.bias()
    }
    fn quiet_bit(self) -> u64 {
        1 << (self.fraction_bits - 1)
    }
    fn sign_bit(self) -> u64 {
        1 << (self.exponent_bits + self.fraction_bits)
    }
    fn infinity(self, sign: bool) -> u64 {
        (self.exponent_mask() << self.fraction_bits) | if sign { self.sign_bit() } else { 0 }
    }
    fn zero(self, sign: bool) -> u64 {
        if sign { self.sign_bit() } else { 0 }
    }
    fn max_finite(self, sign: bool) -> u64 {
        ((self.exponent_mask() - 1) << self.fraction_bits)
            | self.fraction_mask()
            | if sign { self.sign_bit() } else { 0 }
    }
    /// The "real indefinite" QNaN that invalid operations produce: negative, quiet, empty
    /// payload (0xFFC00000 / 0xFFF8000000000000).
    fn indefinite(self) -> u64 {
        self.infinity(true) | self.quiet_bit()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Zero,
    Finite,
    Infinity,
    QuietNan,
    SignalingNan,
}

/// A decoded operand. Finite values (subnormals included) are normalised: `significand` has its
/// leading one at bit `fraction_bits` and `exponent` is the unbiased exponent of that bit.
/// `bits` is the operand as the instruction sees it, i.e. with DAZ applied.
#[derive(Clone, Copy)]
struct Unpacked {
    sign: bool,
    kind: Kind,
    exponent: i32,
    significand: u64,
    bits: u64,
}

impl Unpacked {
    fn is_nan(&self) -> bool {
        matches!(self.kind, Kind::QuietNan | Kind::SignalingNan)
    }
    fn is_zero(&self) -> bool {
        self.kind == Kind::Zero
    }
}

/// Decodes `bits`. A subnormal source raises the denormal-operand flag, unless DAZ turns it into a
/// zero of the same sign.
fn unpack(env: &mut FpEnv, format: Format, bits: u64) -> Unpacked {
    let sign = bits & format.sign_bit() != 0;
    let exponent_field = (bits >> format.fraction_bits) & format.exponent_mask();
    let fraction = bits & format.fraction_mask();
    let mut value = Unpacked {
        sign,
        kind: Kind::Finite,
        exponent: 0,
        significand: 0,
        bits,
    };
    if exponent_field == format.exponent_mask() {
        value.kind = if fraction == 0 {
            Kind::Infinity
        } else if fraction & format.quiet_bit() != 0 {
            Kind::QuietNan
        } else {
            Kind::SignalingNan
        };
    } else if exponent_field == 0 {
        if fraction == 0 || env.denormals_are_zero {
            value.kind = Kind::Zero;
            value.bits = format.zero(sign);
        } else {
            env.flags |= FLAG_DENORMAL;
            let shift = format.fraction_bits - (u64::BITS - 1 - fraction.leading_zeros());
            value.significand = fraction << shift;
            value.exponent = format.min_exponent() - shift as i32;
        }
    } else {
        value.significand = fraction | (1 << format.fraction_bits);
        value.exponent = exponent_field as i32 - format.bias();
    }
    value
}

/// Width of the intermediate significand every operation hands to `round_pack`.
const WIDE_BITS: u32 = 128;
/// Leading-one position of that intermediate: one below the top so that adding two aligned
/// significands cannot overflow.
const WIDE_LEADING_BIT: u32 = WIDE_BITS - 2;
/// Where the leading one of the integer square root's radicand is placed: high enough that the
/// root has well over the 55 bits the widest format needs for round and sticky.
const RADICAND_LEADING_BIT: u32 = 120;

/// Position of the highest set bit of a nonzero value.
fn leading_bit(value: u128) -> u32 {
    WIDE_BITS - 1 - value.leading_zeros()
}

fn widen(format: Format, value: &Unpacked) -> u128 {
    if value.kind == Kind::Finite {
        (value.significand as u128) << (WIDE_LEADING_BIT - format.fraction_bits)
    } else {
        0
    }
}

/// Right shift that ORs everything shifted out into bit 0, so later rounding still sees it.
fn shift_right_sticky(value: u128, count: u32) -> u128 {
    if count == 0 {
        value
    } else if count >= WIDE_BITS {
        (value != 0) as u128
    } else {
        (value >> count) | ((value & ((1u128 << count) - 1) != 0) as u128)
    }
}

/// Whether magnitude `kept` (with `remainder` of `half * 2` dropped below it) rounds up.
fn rounds_up(rounding: u32, sign: bool, kept: u128, remainder: u128, half: u128) -> bool {
    match rounding {
        ROUND_NEAREST_EVEN => remainder > half || (remainder == half && kept & 1 == 1),
        ROUND_DOWN => sign && remainder != 0,
        ROUND_UP => !sign && remainder != 0,
        _ => false,
    }
}

/// Splits `significand` (a value in [2^126, 2^127)) at `shift` bits, giving the kept part, the
/// dropped part and half of one kept-part unit.
fn split_significand(significand: u128, shift: u32) -> (u128, u128, u128) {
    if shift >= WIDE_BITS {
        (0, significand, 1 << (WIDE_BITS - 1))
    } else {
        (
            significand >> shift,
            significand & ((1u128 << shift) - 1),
            1 << (shift - 1),
        )
    }
}

/// Rounds `significand` x 2^(exponent - 126) to `format` under the current rounding mode and
/// returns the packed result, raising overflow, underflow and precision as needed. The caller
/// guarantees bit 126 is set and that bit 0 carries the sticky of anything below it.
/// Tininess is detected after rounding as on x86, with one simplification: rounding is done to the
/// subnormal grid, so a value that only rounds up to the smallest normal number there is treated
/// as not tiny. FTZ (with underflow masked) replaces any tiny result, even an exact one, by a
/// signed zero and reports underflow and precision, as hardware does.
fn round_pack(
    env: &mut FpEnv,
    format: Format,
    sign: bool,
    exponent: i32,
    significand: u128,
) -> u64 {
    let fraction_bits = format.fraction_bits;
    let in_subnormal_range = exponent < format.min_exponent();
    let mut shift = WIDE_LEADING_BIT - fraction_bits;
    if in_subnormal_range {
        shift = shift.saturating_add((format.min_exponent() - exponent) as u32);
    }
    let (mut kept, remainder, half) = split_significand(significand, shift);
    let inexact = remainder != 0;
    if rounds_up(env.rounding, sign, kept, remainder, half) {
        kept += 1;
    }
    let mut kept = kept as u64;
    let mut exponent = exponent;
    let sign_bits = if sign { format.sign_bit() } else { 0 };
    if inexact {
        env.flags |= FLAG_PRECISION;
    }
    if in_subnormal_range {
        let tiny = kept < (1 << fraction_bits);
        if tiny {
            if env.flush_to_zero && env.underflow_masked {
                env.flags |= FLAG_UNDERFLOW | FLAG_PRECISION;
                return format.zero(sign);
            }
            if inexact || !env.underflow_masked {
                env.flags |= FLAG_UNDERFLOW;
            }
        }
        // kept == 2^fraction_bits is the smallest normal number: the same bits as exponent field 1.
        return sign_bits | kept;
    }
    if kept >> (fraction_bits + 1) != 0 {
        kept >>= 1;
        exponent += 1;
    }
    if exponent > format.max_exponent() {
        env.flags |= FLAG_OVERFLOW | FLAG_PRECISION;
        let to_infinity = match env.rounding {
            ROUND_NEAREST_EVEN => true,
            ROUND_DOWN => sign,
            ROUND_UP => !sign,
            _ => false,
        };
        return if to_infinity {
            format.infinity(sign)
        } else {
            format.max_finite(sign)
        };
    }
    let exponent_field = (exponent + format.bias()) as u64;
    sign_bits | (exponent_field << fraction_bits) | (kept & format.fraction_mask())
}

fn invalid(env: &mut FpEnv, format: Format) -> u64 {
    env.flags |= FLAG_INVALID;
    format.indefinite()
}

/// x86 NaN selection for a two-operand instruction: the destination operand's NaN wins when both
/// are NaN; the result is always quieted, and any signaling NaN raises invalid.
fn propagate_nan(
    env: &mut FpEnv,
    format: Format,
    dest: &Unpacked,
    source: &Unpacked,
) -> Option<u64> {
    if !dest.is_nan() && !source.is_nan() {
        return None;
    }
    if dest.kind == Kind::SignalingNan || source.kind == Kind::SignalingNan {
        env.flags |= FLAG_INVALID;
    }
    let chosen = if dest.is_nan() { dest } else { source };
    Some(chosen.bits | format.quiet_bit())
}

#[derive(Clone, Copy, PartialEq)]
enum Arithmetic {
    Add,
    Subtract,
    Multiply,
    Divide,
    SquareRoot,
}

/// One lane of add/sub/mul/div/sqrt, `dest` being the destination operand and `source` the
/// ModRM.rm one (sqrt only reads `source`).
fn arithmetic(
    env: &mut FpEnv,
    format: Format,
    operation: Arithmetic,
    dest: u64,
    source: u64,
) -> u64 {
    let b = unpack(env, format, source);
    if operation == Arithmetic::SquareRoot {
        if b.is_nan() {
            return propagate_nan(env, format, &b, &b).unwrap();
        }
        return square_root(env, format, &b);
    }
    let a = unpack(env, format, dest);
    if let Some(nan) = propagate_nan(env, format, &a, &b) {
        return nan;
    }
    match operation {
        Arithmetic::Add => add(env, format, &a, &b, false),
        Arithmetic::Subtract => add(env, format, &a, &b, true),
        Arithmetic::Multiply => multiply(env, format, &a, &b),
        _ => divide(env, format, &a, &b),
    }
}

fn add(env: &mut FpEnv, format: Format, a: &Unpacked, b: &Unpacked, subtract: bool) -> u64 {
    let b_sign = b.sign != subtract;
    match (a.kind, b.kind) {
        (Kind::Infinity, Kind::Infinity) => {
            return if a.sign == b_sign {
                format.infinity(a.sign)
            } else {
                invalid(env, format)
            };
        }
        (Kind::Infinity, _) => return format.infinity(a.sign),
        (_, Kind::Infinity) => return format.infinity(b_sign),
        // Exact zero sum: -0 only when both are -0, or in round-down mode for opposite signs.
        (Kind::Zero, Kind::Zero) => {
            let sign = if a.sign == b_sign {
                a.sign
            } else {
                env.rounding == ROUND_DOWN
            };
            return format.zero(sign);
        }
        _ => {}
    }
    // Zero operands get an exponent below every finite one, so the ordering swap below puts the
    // zero second.
    let term = |value: &Unpacked, sign: bool| {
        let exponent = if value.is_zero() {
            i32::MIN
        } else {
            value.exponent
        };
        (sign, exponent, widen(format, value))
    };
    let mut x = term(a, a.sign);
    let mut y = term(b, b_sign);
    if (y.1, y.2) > (x.1, x.2) {
        core::mem::swap(&mut x, &mut y);
    }
    if y.2 == 0 {
        return round_pack(env, format, x.0, x.1, x.2);
    }
    let aligned = shift_right_sticky(y.2, (x.1 - y.1) as u32);
    if x.0 == y.0 {
        let mut sum = x.2 + aligned;
        let mut exponent = x.1;
        if sum >> (WIDE_BITS - 1) != 0 {
            sum = shift_right_sticky(sum, 1);
            exponent += 1;
        }
        round_pack(env, format, x.0, exponent, sum)
    } else {
        let difference = x.2 - aligned;
        if difference == 0 {
            return format.zero(env.rounding == ROUND_DOWN);
        }
        let shift = WIDE_LEADING_BIT - leading_bit(difference);
        round_pack(env, format, x.0, x.1 - shift as i32, difference << shift)
    }
}

fn multiply(env: &mut FpEnv, format: Format, a: &Unpacked, b: &Unpacked) -> u64 {
    let sign = a.sign != b.sign;
    match (a.kind, b.kind) {
        (Kind::Infinity, Kind::Zero) | (Kind::Zero, Kind::Infinity) => {
            return invalid(env, format);
        }
        (Kind::Infinity, _) | (_, Kind::Infinity) => return format.infinity(sign),
        (Kind::Zero, _) | (_, Kind::Zero) => return format.zero(sign),
        _ => {}
    }
    let product = a.significand as u128 * b.significand as u128;
    let leading = leading_bit(product);
    // The product of two significands in [1, 2) in 2 x fraction_bits fixed point is in [1, 4).
    let exponent = a.exponent + b.exponent + leading as i32 - 2 * format.fraction_bits as i32;
    round_pack(
        env,
        format,
        sign,
        exponent,
        product << (WIDE_LEADING_BIT - leading),
    )
}

/// Fractional bits of the quotient computed by `divide`; far more than any format needs, so the
/// remainder only has to supply the sticky bit.
const QUOTIENT_FRACTION_BITS: u32 = 64;

fn divide(env: &mut FpEnv, format: Format, a: &Unpacked, b: &Unpacked) -> u64 {
    let sign = a.sign != b.sign;
    match (a.kind, b.kind) {
        (Kind::Infinity, Kind::Infinity) | (Kind::Zero, Kind::Zero) => {
            return invalid(env, format);
        }
        (Kind::Infinity, _) => return format.infinity(sign),
        (_, Kind::Infinity) => return format.zero(sign),
        (Kind::Zero, _) => return format.zero(sign),
        (_, Kind::Zero) => {
            env.flags |= FLAG_DIVIDE_BY_ZERO;
            return format.infinity(sign);
        }
        _ => {}
    }
    let numerator = (a.significand as u128) << QUOTIENT_FRACTION_BITS;
    let divisor = b.significand as u128;
    let quotient = numerator / divisor;
    let sticky = !numerator.is_multiple_of(divisor) as u128;
    let leading = leading_bit(quotient);
    let exponent = a.exponent - b.exponent + leading as i32 - QUOTIENT_FRACTION_BITS as i32;
    round_pack(
        env,
        format,
        sign,
        exponent,
        (quotient << (WIDE_LEADING_BIT - leading)) | sticky,
    )
}

/// Floor of the square root of `n`.
fn integer_square_root(mut n: u128) -> u128 {
    let mut root = 0u128;
    let mut bit = 1u128 << WIDE_LEADING_BIT;
    while bit > n {
        bit >>= 2;
    }
    while bit != 0 {
        if n >= root + bit {
            n -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

fn square_root(env: &mut FpEnv, format: Format, value: &Unpacked) -> u64 {
    match value.kind {
        Kind::Zero => return value.bits,
        Kind::Infinity if !value.sign => return format.infinity(false),
        _ if value.sign => return invalid(env, format),
        _ => {}
    }
    // value = m * 2^e with e even, then scaled up so the integer root has plenty of bits.
    let mut m = value.significand as u128;
    let mut e = value.exponent - format.fraction_bits as i32;
    if e & 1 != 0 {
        m <<= 1;
        e -= 1;
    }
    let scale = (RADICAND_LEADING_BIT - leading_bit(m)) & !1;
    let radicand = m << scale;
    let root = integer_square_root(radicand);
    let sticky = (root * root != radicand) as u128;
    let leading = leading_bit(root);
    let exponent = (e - scale as i32) / 2 + leading as i32;
    round_pack(
        env,
        format,
        false,
        exponent,
        (root << (WIDE_LEADING_BIT - leading)) | sticky,
    )
}

/// Orders two non-NaN operands numerically; zeros of either sign are equal.
fn order(a: &Unpacked, b: &Unpacked) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    let negative = |v: &Unpacked| v.sign && !v.is_zero();
    let magnitude = |v: &Unpacked| match v.kind {
        Kind::Zero => (0, 0, 0),
        Kind::Infinity => (2, 0, 0),
        _ => (1, v.exponent, v.significand),
    };
    match (negative(a), negative(b)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => magnitude(b).cmp(&magnitude(a)),
        (false, false) => magnitude(a).cmp(&magnitude(b)),
    }
}

/// minps/maxps and the scalar forms: when either operand is a NaN (any kind, which raises invalid)
/// or both are zero, the second (source) operand is returned unchanged.
fn minimum_maximum(env: &mut FpEnv, format: Format, maximum: bool, dest: u64, source: u64) -> u64 {
    let a = unpack(env, format, dest);
    let b = unpack(env, format, source);
    if a.is_nan() || b.is_nan() {
        env.flags |= FLAG_INVALID;
        return source;
    }
    if a.is_zero() && b.is_zero() {
        return b.bits;
    }
    let ordering = order(&a, &b);
    let first = if maximum {
        ordering == core::cmp::Ordering::Greater
    } else {
        ordering == core::cmp::Ordering::Less
    };
    if first { a.bits } else { b.bits }
}

/// Orders two operands for the compare instructions, `None` when unordered. `signal_quiet_nan`
/// selects whether a QNaN (not just an SNaN) raises invalid.
fn compare(
    env: &mut FpEnv,
    format: Format,
    dest: u64,
    source: u64,
    signal_quiet_nan: bool,
) -> Option<core::cmp::Ordering> {
    let a = unpack(env, format, dest);
    let b = unpack(env, format, source);
    if a.is_nan() || b.is_nan() {
        let signaling = a.kind == Kind::SignalingNan || b.kind == Kind::SignalingNan;
        if signaling || signal_quiet_nan {
            env.flags |= FLAG_INVALID;
        }
        return None;
    }
    Some(order(&a, &b))
}

/// cmpps/cmppd/cmpss/cmpsd with imm8 predicates 0-7 (eq, lt, le, unord, neq, nlt, nle, ord; only
/// the low three bits exist before AVX). lt, le, nlt and nle also raise invalid for a QNaN.
fn compare_predicate(
    env: &mut FpEnv,
    format: Format,
    predicate: u8,
    dest: u64,
    source: u64,
) -> bool {
    use core::cmp::Ordering::{Equal, Less};
    let predicate = predicate & 7;
    let signaling = matches!(predicate, 1 | 2 | 5 | 6);
    let ordering = compare(env, format, dest, source, signaling);
    match predicate {
        0 => ordering == Some(Equal),
        1 => ordering == Some(Less),
        2 => matches!(ordering, Some(Less | Equal)),
        3 => ordering.is_none(),
        4 => ordering != Some(Equal),
        5 => ordering != Some(Less),
        6 => !matches!(ordering, Some(Less | Equal)),
        _ => ordering.is_some(),
    }
}

/// (u)comiss/(u)comisd result: ZF, PF, CF encode unordered / less / equal; the other status
/// flags are cleared.
fn compare_eflags(ordering: Option<core::cmp::Ordering>) -> u64 {
    use core::cmp::Ordering::{Equal, Greater, Less};
    match ordering {
        None => ZF | PF | CF,
        Some(Greater) => 0,
        Some(Less) => CF,
        Some(Equal) => ZF,
    }
}

/// Signed integer to float (cvtsi2ss/sd, cvtdq2ps/pd), rounded once under the current mode.
fn int_to_float(env: &mut FpEnv, format: Format, value: i64) -> u64 {
    if value == 0 {
        return 0;
    }
    let magnitude = value.unsigned_abs() as u128;
    let leading = leading_bit(magnitude);
    round_pack(
        env,
        format,
        value < 0,
        leading as i32,
        magnitude << (WIDE_LEADING_BIT - leading),
    )
}

/// cvt(t)ss2si/cvt(t)sd2si/cvt(t)ps2dq/cvt(t)pd2dq to a `width`-bit signed result (returned as
/// raw bits, zero-extended to 64). `truncate` selects the cvtt* forms; the others round under
/// MXCSR.RC. NaN, infinity and out-of-range values give the "integer indefinite" (the minimum)
/// and raise invalid; an inexact conversion raises precision.
fn float_to_int(env: &mut FpEnv, format: Format, bits: u64, width: u32, truncate: bool) -> u64 {
    let indefinite = 1u64 << (width - 1);
    let value = unpack(env, format, bits);
    match value.kind {
        Kind::Zero => return 0,
        Kind::Finite => {}
        _ => {
            env.flags |= FLAG_INVALID;
            return indefinite;
        }
    }
    // Values of 2^64 or more are out of range for either width; this also keeps the shift below
    // positive.
    if value.exponent >= 64 {
        env.flags |= FLAG_INVALID;
        return indefinite;
    }
    let rounding = if truncate {
        ROUND_TOWARD_ZERO
    } else {
        env.rounding
    };
    let (mut magnitude, remainder, half) = split_significand(
        widen(format, &value),
        (WIDE_LEADING_BIT as i32 - value.exponent) as u32,
    );
    if rounds_up(rounding, value.sign, magnitude, remainder, half) {
        magnitude += 1;
    }
    let limit = if value.sign {
        1u128 << (width - 1)
    } else {
        (1u128 << (width - 1)) - 1
    };
    if magnitude > limit {
        env.flags |= FLAG_INVALID;
        return indefinite;
    }
    if remainder != 0 {
        env.flags |= FLAG_PRECISION;
    }
    let result = if value.sign {
        (magnitude as u64).wrapping_neg()
    } else {
        magnitude as u64
    };
    if width == 32 {
        result & 0xffff_ffff
    } else {
        result
    }
}

/// cvtss2sd, cvtsd2ss, cvtps2pd, cvtpd2ps for one lane. NaNs keep their payload (shifted to the
/// new width) and are quieted.
fn convert_float(env: &mut FpEnv, from: Format, to: Format, bits: u64) -> u64 {
    let value = unpack(env, from, bits);
    match value.kind {
        Kind::QuietNan | Kind::SignalingNan => {
            if value.kind == Kind::SignalingNan {
                env.flags |= FLAG_INVALID;
            }
            let payload = bits & from.fraction_mask();
            let payload = if to.fraction_bits > from.fraction_bits {
                payload << (to.fraction_bits - from.fraction_bits)
            } else {
                payload >> (from.fraction_bits - to.fraction_bits)
            };
            let sign = if value.sign { to.sign_bit() } else { 0 };
            sign | to.infinity(false) | to.quiet_bit() | payload
        }
        Kind::Infinity => to.infinity(value.sign),
        Kind::Zero => to.zero(value.sign),
        Kind::Finite => round_pack(env, to, value.sign, value.exponent, widen(from, &value)),
    }
}

/// rcpps/rcpss and rsqrtps/rsqrtss. Hardware returns a ~12-bit approximation; this computes the
/// correctly rounded reciprocal (of the square root), so the low mantissa bits differ. These
/// instructions never touch MXCSR flags and always treat subnormal inputs and tiny results as
/// zero, which a scratch environment with DAZ and FTZ forced on reproduces.
fn reciprocal(format: Format, bits: u64, square_root: bool) -> u64 {
    let mut scratch = FpEnv {
        rounding: ROUND_NEAREST_EVEN,
        denormals_are_zero: true,
        flush_to_zero: true,
        underflow_masked: true,
        flags: 0,
    };
    let one = (format.bias() as u64) << format.fraction_bits;
    let operand = if square_root {
        arithmetic(&mut scratch, format, Arithmetic::SquareRoot, 0, bits)
    } else {
        bits
    };
    arithmetic(&mut scratch, format, Arithmetic::Divide, one, operand)
}

fn lane_get(value: u128, format: Format, lane: usize) -> u64 {
    (value >> (lane as u32 * format.width())) as u64 & format.lane_mask() as u64
}

fn lane_put(result: &mut u128, format: Format, lane: usize, bits: u64) {
    *result |= (bits as u128) << (lane as u32 * format.width());
}

/// One lane of the sqrt/add/mul/sub/min/div/max group, selected by opcode.
fn binary_operation(env: &mut FpEnv, format: Format, opcode: u8, dest: u64, source: u64) -> u64 {
    let operation = match opcode {
        0x51 => Arithmetic::SquareRoot,
        0x58 => Arithmetic::Add,
        0x59 => Arithmetic::Multiply,
        0x5c => Arithmetic::Subtract,
        0x5e => Arithmetic::Divide,
        _ => return minimum_maximum(env, format, opcode == 0x5f, dest, source),
    };
    arithmetic(env, format, operation, dest, source)
}

/// The mandatory prefix that selects between the packed-single, packed-double, scalar-single and
/// scalar-double (or MMX vs. XMM integer) forms of an 0F opcode. `Np` = none, `P66` = operand-size
/// prefix, `F3`/`F2` = rep/repne; if several are present F3/F2 win (see `Insn::rep`).
#[derive(Clone, Copy, PartialEq)]
enum Prefix {
    Np,
    P66,
    F3,
    F2,
}

/// `Insn::op` is `(map << 8) | opcode_byte`; map 1 is the `0f xx` two-byte map.
const OPCODE_MAP_0F: u16 = 1;
const MODRM_MOD_REGISTER: u8 = 3;
/// MMX registers mm0-mm7 are the low 64 bits of the *physical* x87 registers, whatever TOP is.
const MMX_REGISTER_MASK: u8 = 7;
const MMX_QWORD_BYTES: usize = 8;
/// The sign and exponent field an MMX write leaves in the x87 register: all ones.
const MMX_EXPONENT_FIELD: u16 = 0xffff;
/// FSW.TOP, bits 13:11.
const FSW_TOP_MASK: u16 = 7 << 11;
/// `Cpu::ftw` holds one bit per physical x87 register, set when it is not empty.
const ALL_X87_REGISTERS_VALID: u16 = 0xff;
const MMX_WORD_COUNT: usize = 4;
const GPR_RDI: usize = 7;
const LOW_32: u128 = 0xffff_ffff;
const LOW_64: u128 = u64::MAX as u128;

impl Cpu {
    /// SDM SSE exception rules: #UD if CR0.EM is set or the OS has not enabled SSE (CR4.OSFXSR),
    /// #NM if CR0.TS is set (lazy FPU switching). #UD takes priority over #NM.
    pub(super) fn sse_gate(&self) -> ExecResult<()> {
        if self.cr0 & CR0_EM != 0 || self.cr4 & CR4_OSFXSR == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.cr0 & CR0_TS != 0 {
            return Err(Exception::device_not_available());
        }
        Ok(())
    }

    /// Merges the exception flags an instruction raised into MXCSR. An unmasked one is a fault
    /// (#XM, or #UD when the OS has not set CR4.OSXMMEXCPT) and the caller must not have written
    /// its destination yet. The flags stay set in MXCSR either way, as on hardware.
    fn fp_commit(&mut self, env: &FpEnv) -> ExecResult<()> {
        self.mxcsr |= env.flags;
        let unmasked = env.flags & !(self.mxcsr >> MXCSR_MASK_SHIFT) & MXCSR_EXCEPTION_FLAGS;
        if unmasked == 0 {
            Ok(())
        } else if self.cr4 & CR4_OSXMMEXCPT != 0 {
            Err(Exception::fault(VECTOR_SIMD_FLOATING_POINT, None))
        } else {
            Err(Exception::invalid_opcode())
        }
    }

    fn mmx_enter(&mut self) {
        self.fsw &= !FSW_TOP_MASK;
        self.ftw = ALL_X87_REGISTERS_VALID;
    }

    fn mm_get(&self, index: u8) -> u64 {
        let raw = &self.st[(index & MMX_REGISTER_MASK) as usize];
        u64::from_le_bytes(raw[..MMX_QWORD_BYTES].try_into().unwrap())
    }

    fn mm_put(&mut self, index: u8, value: u64) {
        let raw = &mut self.st[(index & MMX_REGISTER_MASK) as usize];
        raw[..MMX_QWORD_BYTES].copy_from_slice(&value.to_le_bytes());
        raw[MMX_QWORD_BYTES..].copy_from_slice(&MMX_EXPONENT_FIELD.to_le_bytes());
    }

    /// The MMX instructions (see `mmx_form`). Gating: CR0.EM is #UD and CR0.TS is #NM; CR4.OSFXSR
    /// only matters for the forms that also touch an xmm register. Any of them except emms puts
    /// the x87 stack in MMX state (TOP = 0, every tag valid) and a write to mmN also fills the
    /// register's sign/exponent field with ones. Memory operands are read before any state
    /// changes, so a fault leaves the machine as it was.
    fn mmx(&mut self, i: &Insn, next: u64, opcode: u8, prefix: Prefix) -> ExecResult<()> {
        if matches!(opcode, 0x2a | 0x2c | 0x2d | 0xd6) {
            self.sse_gate()?;
        } else {
            if self.cr0 & CR0_EM != 0 {
                return Err(Exception::invalid_opcode());
            }
            self.fpu_check()?;
        }
        let reg = i.reg;
        let rm = i.rm;
        let mem_addr = if i.md == MODRM_MOD_REGISTER {
            None
        } else {
            Some(self.linear_address(i, self.effective_address(i, next)))
        };
        let operand_size = if i.opsize == 8 { 8 } else { 4 };
        match opcode {
            // movd/movq mm, r/m
            0x6e => {
                let v = match mem_addr {
                    None => self.read_gpr(rm, operand_size, true),
                    Some(addr) => self.read(addr, operand_size)?,
                } & mask(operand_size);
                self.mmx_enter();
                self.mm_put(reg, v);
            }
            // movd/movq r/m, mm
            0x7e => {
                let v = self.mm_get(reg) & mask(operand_size);
                match mem_addr {
                    None => self.write_gpr(rm, operand_size, true, v),
                    Some(addr) => self.write(addr, operand_size, v)?,
                }
                self.mmx_enter();
            }
            // movq mm, mm/m64
            0x6f => {
                let v = self.mm_or_m64(rm, mem_addr)?;
                self.mmx_enter();
                self.mm_put(reg, v);
            }
            // movq mm/m64, mm
            0x7f => {
                let v = self.mm_get(reg);
                match mem_addr {
                    None => {
                        self.mmx_enter();
                        self.mm_put(rm, v);
                    }
                    Some(addr) => {
                        self.write(addr, MMX_QWORD_BYTES as u8, v)?;
                        self.mmx_enter();
                    }
                }
            }
            // movntq m64, mm (the memory-only form was checked by `form_valid`)
            0xe7 => {
                let v = self.mm_get(reg);
                if let Some(addr) = mem_addr {
                    self.write(addr, MMX_QWORD_BYTES as u8, v)?;
                }
                self.mmx_enter();
            }
            // pshufw: the pshuflw word selection applied to the whole register
            0x70 => {
                let source = self.mm_or_m64(rm, mem_addr)?;
                let imm8 = i.imm as u8;
                let mut r = 0u64;
                for k in 0..MMX_WORD_COUNT {
                    let word = (source >> (16 * ((imm8 >> (2 * k)) & 3))) & 0xffff;
                    r |= word << (16 * k);
                }
                self.mmx_enter();
                self.mm_put(reg, r);
            }
            // shifts by immediate; the register is ModRM.rm
            0x71..=0x73 => {
                let x = self.mm_get(rm) as u128;
                let count = i.imm as u8 as u64;
                let Some(r) = shift_immediate(opcode, i.reg & 7, x, count) else {
                    return Err(self.unimplemented_opcode(i));
                };
                self.mmx_enter();
                self.mm_put(rm, r as u64);
            }
            // pinsrw mm, r32/m16: imm8 & 3 selects the word
            0xc4 => {
                let word = match mem_addr {
                    None => self.read_gpr(rm, 2, true),
                    Some(addr) => self.read(addr, 2)?,
                } & 0xffff;
                let shift = 16 * (i.imm as u64 & (MMX_WORD_COUNT as u64 - 1));
                let r = (self.mm_get(reg) & !(0xffff << shift)) | (word << shift);
                self.mmx_enter();
                self.mm_put(reg, r);
            }
            // pextrw r32, mm, imm8
            0xc5 => {
                let shift = 16 * (i.imm as u64 & (MMX_WORD_COUNT as u64 - 1));
                self.gpr[reg as usize] = (self.mm_get(rm) >> shift) & 0xffff;
                self.mmx_enter();
            }
            // pmovmskb r32, mm
            0xd7 => {
                let x = self.mm_get(rm);
                self.gpr[reg as usize] =
                    (0..MMX_QWORD_BYTES).fold(0, |m, k| m | (((x >> (8 * k + 7)) & 1) << k));
                self.mmx_enter();
            }
            // maskmovq mm1, mm2: stores the bytes of mm1 whose mm2 byte has its top bit set at
            // DS:rdi. Bytes that are not selected are not accessed.
            0xf7 => {
                let data = self.mm_get(reg);
                let selector = self.mm_get(rm);
                for k in 0..MMX_QWORD_BYTES as u64 {
                    if (selector >> (8 * k + 7)) & 1 != 0 {
                        let addr = self.linear_address(i, self.gpr[GPR_RDI].wrapping_add(k));
                        self.write(addr, 1, (data >> (8 * k)) & 0xff)?;
                    }
                }
                self.mmx_enter();
            }
            // cvtpi2ps xmm, mm/m64 (Np) and cvtpi2pd xmm, mm/m64 (66)
            0x2a => {
                let source = self.mm_or_m64(rm, mem_addr)?;
                let mut fp = FpEnv::new(self.mxcsr);
                let dst = self.xmm[reg as usize];
                let int = |k: u32| (source >> (32 * k)) as u32 as i32 as i64;
                let r = if prefix == Prefix::Np {
                    let mut r = 0;
                    for k in 0..2 {
                        lane_put(
                            &mut r,
                            SINGLE,
                            k,
                            int_to_float(&mut fp, SINGLE, int(k as u32)),
                        );
                    }
                    (dst & !LOW_64) | r
                } else {
                    let mut r = 0;
                    for k in 0..2 {
                        lane_put(
                            &mut r,
                            DOUBLE,
                            k,
                            int_to_float(&mut fp, DOUBLE, int(k as u32)),
                        );
                    }
                    r
                };
                self.fp_commit(&fp)?;
                self.mmx_enter();
                self.xmm[reg as usize] = r;
            }
            // cvtps2pi/cvttps2pi mm, xmm/m64 (Np) and cvtpd2pi/cvttpd2pi mm, xmm/m128 (66)
            0x2c | 0x2d => {
                let truncate = opcode == 0x2c;
                let mut fp = FpEnv::new(self.mxcsr);
                let format = if prefix == Prefix::Np { SINGLE } else { DOUBLE };
                let source = match mem_addr {
                    None => self.xmm[rm as usize],
                    Some(addr) if prefix == Prefix::Np => self.read(addr, 8)? as u128,
                    Some(addr) => {
                        if addr & (VECTOR_ALIGNMENT - 1) != 0 {
                            return Err(Exception::general_protection(0));
                        }
                        self.read128(addr)?
                    }
                };
                let mut r = 0u64;
                for k in 0..2 {
                    let lane = lane_get(source, format, k);
                    let integer = float_to_int(&mut fp, format, lane, 32, truncate);
                    r |= integer << (32 * k);
                }
                self.fp_commit(&fp)?;
                self.mmx_enter();
                self.mm_put(reg, r);
            }
            // F3: movq2dq xmm, mm; F2: movdq2q mm, xmm (register forms only)
            0xd6 => {
                if prefix == Prefix::F3 {
                    let v = self.mm_get(rm);
                    self.mmx_enter();
                    self.xmm[reg as usize] = v as u128;
                } else {
                    let v = self.xmm[rm as usize] as u64;
                    self.mmx_enter();
                    self.mm_put(reg, v);
                }
            }
            // everything else is a two-operand packed integer operation on mm and mm/m64
            _ => {
                // punpckl* read only 32 bits of a memory operand (mm/m32)
                let size = if matches!(opcode, 0x60..=0x62) {
                    4
                } else {
                    MMX_QWORD_BYTES as u8
                };
                let source = match mem_addr {
                    None => self.mm_get(rm),
                    Some(addr) => self.read(addr, size)?,
                };
                let r = mmx_binary(opcode, self.mm_get(reg), source);
                self.mmx_enter();
                self.mm_put(reg, r);
            }
        }
        Ok(())
    }

    /// The mm/m64 operand: register mm`rm`, or 8 bytes at `mem_addr`.
    fn mm_or_m64(&mut self, rm: u8, mem_addr: Option<u64>) -> ExecResult<u64> {
        match mem_addr {
            None => Ok(self.mm_get(rm)),
            Some(addr) => self.read(addr, MMX_QWORD_BYTES as u8),
        }
    }

    /// Executes an SSE/SSE2 instruction from the `0f xx` map. Returns `Ok(false)` for anything
    /// else, which the caller turns into #UD; on `Ok(true)` the caller sets rip to `next`.
    /// CPUID (`cpu/sys.rs`) advertises MMX, SSE and SSE2 and this covers those (the MMX register
    /// forms are in `mmx`). SSE3 and later, which have their own opcodes and mandatory prefixes,
    /// are in `sse_ext`; there is no VEX.
    ///
    /// Vector registers are `u128` in `self.xmm`. The `src!` operand is the whole register in the
    /// register form, so scalar arms truncate it to the lane they use. Every instruction that can
    /// raise an FP exception collects its flags in `fp` and commits them (possibly faulting)
    /// before the destination is written.
    pub fn sse(&mut self, i: &Insn, next: u64) -> ExecResult<bool> {
        let map = i.op >> 8;
        if map != OPCODE_MAP_0F {
            return self.sse_ext(i, next);
        }
        let opcode = (i.op & 0xff) as u8;
        let prefix = match (i.rep, i.p66) {
            (1, _) => Prefix::F3,
            (2, _) => Prefix::F2,
            (_, true) => Prefix::P66,
            _ => Prefix::Np,
        };
        if !allowed(opcode, prefix) {
            return self.sse_ext(i, next);
        }
        // An encoding that is invalid for this ModRM form is #UD even with CR0.TS set (#UD has
        // priority over #NM, as in QEMU).
        if !form_valid(opcode, prefix, i.md == MODRM_MOD_REGISTER, i.reg & 7) {
            return Err(Exception::invalid_opcode());
        }
        // emms is an MMX instruction: CR0.EM is #UD and CR0.TS is #NM, and CR4.OSFXSR does not
        // matter. ftw is the abridged tag word of FXSAVE: 0 marks all eight registers empty.
        if opcode == 0x77 {
            if self.cr0 & CR0_EM != 0 {
                return Err(Exception::invalid_opcode());
            }
            self.fpu_check()?;
            self.ftw = 0;
            return Ok(true);
        }
        if mmx_form(opcode, prefix) {
            self.mmx(i, next, opcode, prefix)?;
            return Ok(true);
        }
        self.sse_gate()?;
        // ModRM.reg and .rm already include REX.R/REX.B. `dst_reg` is a general register for the
        // instructions that write one (cvt*2si, movmskp*, pmovmskb, pextrw).
        let dst_reg = i.reg as usize;
        let rm_reg = i.rm as usize;
        // Linear address of the memory operand, `None` for the register form (ModRM.mod == 3).
        let mem_addr = if i.md == MODRM_MOD_REGISTER {
            None
        } else {
            Some(self.linear_address(i, self.effective_address(i, next)))
        };
        // Every 128-bit memory operand of a legacy-SSE instruction must be 16-byte aligned (#GP(0))
        // except for the explicitly unaligned forms: movups, movupd and movdqu loads.
        let unaligned_ok = matches!(
            (opcode, prefix),
            (0x10, Prefix::Np | Prefix::P66) | (0x6f, Prefix::F3)
        );
        macro_rules! src {
            ($size:expr) => {
                match mem_addr {
                    None => self.xmm[rm_reg],
                    Some(addr) => {
                        if $size == 16 {
                            if addr & (VECTOR_ALIGNMENT - 1) != 0 && !unaligned_ok {
                                return Err(Exception::general_protection(0));
                            }
                            self.read128(addr)?
                        } else {
                            self.read(addr, $size)? as u128
                        }
                    }
                }
            };
        }
        // #GP(0) for a misaligned 16-byte store destination; loads check in `src!`.
        macro_rules! aligned {
            () => {
                if let Some(addr) = mem_addr {
                    if addr & (VECTOR_ALIGNMENT - 1) != 0 {
                        return Err(Exception::general_protection(0));
                    }
                }
            };
        }
        let dst = self.xmm[dst_reg];
        let mut fp = FpEnv::new(self.mxcsr);
        let result: Option<u128> = match opcode {
            // movups, movupd, movss, movsd load. The memory form of movss/movsd zeroes the rest of
            // the register; the register form merges into the low lane.
            0x10 => Some(match prefix {
                Prefix::Np | Prefix::P66 => src!(16),
                Prefix::F3 => {
                    if mem_addr.is_some() {
                        src!(4)
                    } else {
                        (dst & !LOW_32) | (self.xmm[rm_reg] & LOW_32)
                    }
                }
                Prefix::F2 => {
                    if mem_addr.is_some() {
                        src!(8)
                    } else {
                        (dst & !LOW_64) | (self.xmm[rm_reg] & LOW_64)
                    }
                }
            }),
            // stores: movups/movupd/movss/movsd (0x11), movaps/movapd (0x29), movntps/movntpd
            // (0x2b), movdqa/movdqu (0x7f), movntdq (0xe7). Scalar register destinations merge into
            // the low lane.
            0x11 | 0x29 | 0x2b | 0x7f | 0xe7 => {
                if matches!(opcode, 0x29 | 0x2b)
                    || (opcode == 0x7f && prefix == Prefix::P66)
                    || opcode == 0xe7
                {
                    aligned!();
                }
                let size = match (opcode, prefix) {
                    (0x11, Prefix::F3) => 4,
                    (0x11, Prefix::F2) => 8,
                    _ => 16,
                };
                match mem_addr {
                    Some(addr) => match size {
                        16 => self.write128(addr, dst)?,
                        s => self.write(addr, s, dst as u64)?,
                    },
                    None => {
                        let cur = self.xmm[rm_reg];
                        self.xmm[rm_reg] = match size {
                            4 => (cur & !LOW_32) | (dst & LOW_32),
                            8 => (cur & !LOW_64) | (dst & LOW_64),
                            _ => dst,
                        };
                    }
                }
                None
            }
            // movlps/movlpd (0x12) and movhps/movhpd (0x16) loads; the register forms are movhlps
            // and movlhps.
            0x12 | 0x16 => {
                let low_half = opcode == 0x12;
                let v = if mem_addr.is_none() {
                    if low_half {
                        self.xmm[rm_reg] >> 64
                    } else {
                        self.xmm[rm_reg] & LOW_64
                    }
                } else {
                    src!(8)
                } & LOW_64;
                Some(if low_half {
                    (dst & !LOW_64) | v
                } else {
                    (dst & LOW_64) | (v << 64)
                })
            }
            // movlps/movlpd and movhps/movhpd stores; memory destination only.
            0x13 | 0x17 => {
                let addr = mem_addr.ok_or_else(Exception::invalid_opcode)?;
                let v = if opcode == 0x13 {
                    dst as u64
                } else {
                    (dst >> 64) as u64
                };
                self.write(addr, 8, v)?;
                None
            }
            // unpcklps/unpcklpd (0x14), unpckhps/unpckhpd (0x15)
            0x14 | 0x15 => {
                let s = src!(16);
                let hi = opcode == 0x15;
                Some(if prefix == Prefix::Np {
                    interleave::<u32>(dst, s, hi)
                } else {
                    interleave::<u64>(dst, s, hi)
                })
            }
            // movaps/movapd load
            0x28 => Some(src!(16)),
            // cvtsi2ss/cvtsi2sd from a 32-bit or (REX.W) 64-bit integer, rounded once under MXCSR.RC
            0x2a => {
                let size = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, size, true),
                    Some(addr) => self.read(addr, size)?,
                };
                let value = if size == 8 {
                    v as i64
                } else {
                    v as u32 as i32 as i64
                };
                let format = if prefix == Prefix::F3 { SINGLE } else { DOUBLE };
                let converted = int_to_float(&mut fp, format, value) as u128;
                Some((dst & !format.lane_mask()) | converted)
            }
            // cvttss2si/cvttsd2si (0x2c, truncates) and cvtss2si/cvtsd2si (0x2d, rounds per
            // MXCSR.RC)
            0x2c | 0x2d => {
                let format = if prefix == Prefix::F3 { SINGLE } else { DOUBLE };
                let size = if prefix == Prefix::F3 { 4 } else { 8 };
                let v = lane_get(src!(size), format, 0);
                let width = if i.opsize == 8 { 64 } else { 32 };
                let converted = float_to_int(&mut fp, format, v, width, opcode == 0x2c);
                self.fp_commit(&fp)?;
                self.gpr[dst_reg] = converted;
                None
            }
            // ucomiss/ucomisd (0x2e) and comiss/comisd (0x2f). comis also raises invalid for a
            // QNaN.
            0x2e | 0x2f => {
                let (format, size) = if prefix == Prefix::Np {
                    (SINGLE, 4)
                } else {
                    (DOUBLE, 8)
                };
                let s = lane_get(src!(size), format, 0);
                let ordering =
                    compare(&mut fp, format, lane_get(dst, format, 0), s, opcode == 0x2f);
                self.fp_commit(&fp)?;
                self.set_eflags_arith(compare_eflags(ordering));
                None
            }
            // movmskps/movmskpd: sign bits into a general register
            0x50 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let x = self.xmm[rm_reg];
                self.gpr[dst_reg] = if prefix == Prefix::Np {
                    (0..4).fold(0, |m, k| m | (((u32::get(x, k) >> 31) as u64) << k))
                } else {
                    (0..2).fold(0, |m, k| m | ((u64::get(x, k) >> 63) << k))
                };
                None
            }
            // sqrt, add, mul, sub, min, div, max in packed and scalar forms; scalar forms merge
            // into the low lane.
            0x51 | 0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f => {
                let format = if matches!(prefix, Prefix::Np | Prefix::F3) {
                    SINGLE
                } else {
                    DOUBLE
                };
                if matches!(prefix, Prefix::Np | Prefix::P66) {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..format.lanes() {
                        let lane = binary_operation(
                            &mut fp,
                            format,
                            opcode,
                            lane_get(dst, format, k),
                            lane_get(s, format, k),
                        );
                        lane_put(&mut r, format, k, lane);
                    }
                    Some(r)
                } else {
                    let s = lane_get(src!(format.bytes()), format, 0);
                    let lane =
                        binary_operation(&mut fp, format, opcode, lane_get(dst, format, 0), s);
                    Some((dst & !format.lane_mask()) | lane as u128)
                }
            }
            // rsqrtps/rsqrtss (0x52), rcpps/rcpss (0x53)
            0x52 | 0x53 => {
                let square_root = opcode == 0x52;
                Some(if prefix == Prefix::F3 {
                    let s = lane_get(src!(4), SINGLE, 0);
                    (dst & !SINGLE.lane_mask()) | reciprocal(SINGLE, s, square_root) as u128
                } else {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..SINGLE.lanes() {
                        lane_put(
                            &mut r,
                            SINGLE,
                            k,
                            reciprocal(SINGLE, lane_get(s, SINGLE, k), square_root),
                        );
                    }
                    r
                })
            }
            // andps/andpd, andnps/andnpd, orps/orpd, xorps/xorpd
            0x54..=0x57 => {
                let s = src!(16);
                Some(match opcode {
                    0x54 => dst & s,
                    0x55 => !dst & s,
                    0x56 => dst | s,
                    _ => dst ^ s,
                })
            }
            // cvtps2pd, cvtpd2ps, cvtss2sd, cvtsd2ss
            0x5a => Some(match prefix {
                Prefix::Np => {
                    let s = src!(8);
                    let mut r = 0;
                    for k in 0..DOUBLE.lanes() {
                        let lane = convert_float(&mut fp, SINGLE, DOUBLE, lane_get(s, SINGLE, k));
                        lane_put(&mut r, DOUBLE, k, lane);
                    }
                    r
                }
                Prefix::P66 => {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..DOUBLE.lanes() {
                        let lane = convert_float(&mut fp, DOUBLE, SINGLE, lane_get(s, DOUBLE, k));
                        lane_put(&mut r, SINGLE, k, lane);
                    }
                    r
                }
                Prefix::F3 => {
                    let s = lane_get(src!(4), SINGLE, 0);
                    (dst & !DOUBLE.lane_mask()) | convert_float(&mut fp, SINGLE, DOUBLE, s) as u128
                }
                Prefix::F2 => {
                    let s = lane_get(src!(8), DOUBLE, 0);
                    (dst & !SINGLE.lane_mask()) | convert_float(&mut fp, DOUBLE, SINGLE, s) as u128
                }
            }),
            // cvtdq2ps, cvtps2dq (rounds per MXCSR.RC), cvttps2dq (truncates)
            0x5b => {
                let s = src!(16);
                let mut r = 0;
                for k in 0..SINGLE.lanes() {
                    let lane = lane_get(s, SINGLE, k);
                    let converted = if prefix == Prefix::Np {
                        int_to_float(&mut fp, SINGLE, lane as u32 as i32 as i64)
                    } else {
                        float_to_int(&mut fp, SINGLE, lane, 32, prefix == Prefix::F3)
                    };
                    lane_put(&mut r, SINGLE, k, converted);
                }
                Some(r)
            }
            // 0x60-0x6d unpack/pack/compare, 0x74-0x76 pcmpeq, shifts by an xmm/m128 count and
            // the rest of the SSE2 integer arithmetic and logic: see `integer_op`.
            0x60..=0x6d
            | 0x74..=0x76
            | 0xd1..=0xd5
            | 0xd8..=0xdf
            | 0xe0..=0xe5
            | 0xe8..=0xef
            | 0xf1..=0xf6
            | 0xf8..=0xfe => Some(integer_op(opcode, dst, src!(16))),
            // movd/movq xmm, r/m; zero-extends into the whole register
            0x6e => {
                let size = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, size, true),
                    Some(addr) => self.read(addr, size)?,
                };
                Some(v as u128)
            }
            // movdqa and movdqu load
            0x6f => Some(src!(16)),
            // pshufd, pshuflw, pshufhw: imm8 holds four 2-bit source lane indices, lowest field
            // first; pshuflw/pshufhw copy the other half of the register unchanged.
            0x70 => {
                let s = src!(16);
                let imm8 = i.imm as u8;
                Some(match prefix {
                    Prefix::P66 => {
                        let mut r = 0;
                        for k in 0..4 {
                            u32::put(&mut r, k, u32::get(s, ((imm8 >> (2 * k)) & 3) as usize));
                        }
                        r
                    }
                    Prefix::F2 => {
                        let mut r = s & !LOW_64;
                        for k in 0..4 {
                            u16::put(&mut r, k, u16::get(s, ((imm8 >> (2 * k)) & 3) as usize));
                        }
                        r
                    }
                    _ => {
                        let mut r = s & LOW_64;
                        for k in 0..4 {
                            u16::put(
                                &mut r,
                                4 + k,
                                u16::get(s, 4 + ((imm8 >> (2 * k)) & 3) as usize),
                            );
                        }
                        r
                    }
                })
            }
            // Shifts by immediate; ModRM.reg selects the operation (see `shift_immediate`)
            0x71..=0x73 => {
                let count = i.imm as u8 as u64;
                let Some(r) = shift_immediate(opcode, i.reg & 7, self.xmm[rm_reg], count) else {
                    // unassigned /reg values were already rejected by `form_valid`
                    return Err(self.unimplemented_opcode(i));
                };
                self.xmm[rm_reg] = r;
                None
            }
            // movq xmm, xmm/m64 (F3, zero-extends) and movd/movq r/m, xmm (66)
            0x7e => {
                if prefix == Prefix::F3 {
                    Some(src!(8) & LOW_64)
                } else {
                    let size = if i.opsize == 8 { 8 } else { 4 };
                    let v = dst as u64 & if size == 8 { u64::MAX } else { 0xffff_ffff };
                    match mem_addr {
                        None => self.write_gpr(i.rm, size, true, v),
                        Some(addr) => self.write(addr, size, v)?,
                    }
                    None
                }
            }
            // cmpps/cmppd/cmpss/cmpsd: all ones or zero per lane
            0xc2 => {
                let predicate = i.imm as u8;
                let format = if matches!(prefix, Prefix::Np | Prefix::F3) {
                    SINGLE
                } else {
                    DOUBLE
                };
                if matches!(prefix, Prefix::Np | Prefix::P66) {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..format.lanes() {
                        let (a, b) = (lane_get(dst, format, k), lane_get(s, format, k));
                        if compare_predicate(&mut fp, format, predicate, a, b) {
                            lane_put(&mut r, format, k, format.lane_mask() as u64);
                        }
                    }
                    Some(r)
                } else {
                    let s = lane_get(src!(format.bytes()), format, 0);
                    let a = lane_get(dst, format, 0);
                    let matched = compare_predicate(&mut fp, format, predicate, a, s);
                    Some((dst & !format.lane_mask()) | if matched { format.lane_mask() } else { 0 })
                }
            }
            // pinsrw: imm8 & 7 selects the word
            0xc4 => {
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, 2, true),
                    Some(addr) => self.read(addr, 2)?,
                } as u16;
                let k = (i.imm as usize) & 7;
                let sh = k * 16;
                Some((dst & !(0xffffu128 << sh)) | ((v as u128) << sh))
            }
            // pextrw: register source only; zero-extends into the general register
            0xc5 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let k = (i.imm as usize) & 7;
                self.gpr[dst_reg] = u16::get(self.xmm[rm_reg], k) as u64;
                None
            }
            // shufps: result lanes 0-1 come from dst, 2-3 from the source. shufpd: imm8 bit 0 picks
            // the low lane from dst, bit 1 the high lane from the source.
            0xc6 => {
                let s = src!(16);
                let imm8 = i.imm as u8 as usize;
                Some(if prefix == Prefix::Np {
                    let mut r = 0;
                    u32::put(&mut r, 0, u32::get(dst, imm8 & 3));
                    u32::put(&mut r, 1, u32::get(dst, (imm8 >> 2) & 3));
                    u32::put(&mut r, 2, u32::get(s, (imm8 >> 4) & 3));
                    u32::put(&mut r, 3, u32::get(s, (imm8 >> 6) & 3));
                    r
                } else {
                    u64::get(dst, imm8 & 1) as u128 | ((u64::get(s, (imm8 >> 1) & 1) as u128) << 64)
                })
            }
            // movq xmm/m64, xmm; the register form zeroes the upper half
            0xd6 => {
                let v = dst & LOW_64;
                match mem_addr {
                    None => self.xmm[rm_reg] = v,
                    Some(addr) => self.write(addr, 8, v as u64)?,
                }
                None
            }
            // pmovmskb
            0xd7 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let x = self.xmm[rm_reg];
                self.gpr[dst_reg] =
                    (0..16).fold(0, |m, k| m | (((u8::get(x, k) >> 7) as u64) << k));
                None
            }
            // cvtdq2pd (F3), cvtpd2dq (F2, MXCSR.RC), cvttpd2dq (66, truncates). The integer forms
            // fill only the low two dwords and zero the upper half.
            0xe6 => {
                let truncate = prefix == Prefix::P66;
                let s = src!(if prefix == Prefix::F3 { 8 } else { 16 });
                let mut r = 0;
                for k in 0..DOUBLE.lanes() {
                    if prefix == Prefix::F3 {
                        let integer = lane_get(s, SINGLE, k) as u32 as i32 as i64;
                        lane_put(&mut r, DOUBLE, k, int_to_float(&mut fp, DOUBLE, integer));
                    } else {
                        let lane =
                            float_to_int(&mut fp, DOUBLE, lane_get(s, DOUBLE, k), 32, truncate);
                        lane_put(&mut r, SINGLE, k, lane);
                    }
                }
                Some(r)
            }
            _ => return Ok(false),
        };
        if fp.flags != 0 {
            self.fp_commit(&fp)?;
        }
        if let Some(v) = result {
            self.xmm[dst_reg] = v;
        }
        Ok(true)
    }
}

/// Opcodes (with this prefix) whose operands are MMX registers, or for the conversions and
/// movq2dq/movdq2q a mix of MMX and xmm registers. The same opcodes with 66 are the xmm forms.
fn mmx_form(opcode: u8, prefix: Prefix) -> bool {
    match opcode {
        0x2a | 0x2c | 0x2d => matches!(prefix, Prefix::Np | Prefix::P66),
        0xd6 => matches!(prefix, Prefix::F3 | Prefix::F2),
        0x60..=0x6b
        | 0x6e
        | 0x6f
        | 0x70
        | 0x71..=0x76
        | 0x7e
        | 0x7f
        | 0xc4
        | 0xc5
        | 0xd1..=0xd5
        | 0xd7..=0xdf
        | 0xe0..=0xe5
        | 0xe7..=0xef
        | 0xf1..=0xf7
        | 0xf8..=0xfe => prefix == Prefix::Np,
        _ => false,
    }
}

/// The packed integer instructions on 64-bit mm operands. Most are `integer_op` on a register
/// whose upper half is zero; unpack-high, pack and the other cross-lane ones need their inputs
/// rearranged because an mm register has half as many lanes as an xmm one.
fn mmx_binary(opcode: u8, dst: u64, s: u64) -> u64 {
    let (a, b) = (dst as u128, s as u128);
    let r = match opcode {
        // packs read the low half of dst then the low half of s, the layout of xmm lanes 0-7
        0x63 | 0x67 => pack_sat(a | (b << 64), 0, 16, opcode == 0x63),
        0x6b => pack_sat(a | (b << 64), 0, 32, true),
        // punpckh* of an mm register interleaves its upper 32 bits: the low-unpack of those
        0x68..=0x6a => integer_op(opcode - 8, a >> 32, b >> 32),
        _ => integer_op(opcode, a, b),
    };
    r as u64
}

/// The 0F 71/72/73 shift-by-immediate groups: /2 psrl, /4 psra, /6 psll, and for 0x73 /3 psrldq
/// and /7 pslldq (whole-register byte shifts; a count above 15 clears the register). `None` for an
/// unassigned (opcode, /reg).
fn shift_immediate(opcode: u8, group: u8, x: u128, count: u64) -> Option<u128> {
    Some(match (opcode, group) {
        (0x71, 2) => shift_logical::<u16>(x, count, false, 16),
        (0x71, 4) => sra::<u16>(x, count, 16),
        (0x71, 6) => shift_logical::<u16>(x, count, true, 16),
        (0x72, 2) => shift_logical::<u32>(x, count, false, 32),
        (0x72, 4) => sra::<u32>(x, count, 32),
        (0x72, 6) => shift_logical::<u32>(x, count, true, 32),
        (0x73, 2) => shift_logical::<u64>(x, count, false, 64),
        (0x73, 6) => shift_logical::<u64>(x, count, true, 64),
        (0x73, 3) => {
            if count > 15 {
                0
            } else {
                x >> (count * 8)
            }
        }
        (0x73, 7) => {
            if count > 15 {
                0
            } else {
                x << (count * 8)
            }
        }
        _ => return None,
    })
}

/// The two-operand packed integer instructions shared by the xmm (66) forms: unpack, pack,
/// compare, shifts by a register count (only the low 64 bits of `s` count) and arithmetic.
/// `opcode` is the 0F-map opcode byte.
fn integer_op(opcode: u8, dst: u128, s: u128) -> u128 {
    // shift count: the low quadword only
    let count = s as u64;
    match opcode {
        0x60 => interleave::<u8>(dst, s, false),
        0x61 => interleave::<u16>(dst, s, false),
        0x62 => interleave::<u32>(dst, s, false),
        0x63 => pack_sat(dst, s, 16, true),
        0x64 => zip_lanes::<u8>(dst, s, |a, b| if (a as i8) > (b as i8) { 0xff } else { 0 }),
        0x65 => zip_lanes::<u16>(
            dst,
            s,
            |a, b| if (a as i16) > (b as i16) { 0xffff } else { 0 },
        ),
        0x66 => zip_lanes::<u32>(
            dst,
            s,
            |a, b| if (a as i32) > (b as i32) { u32::MAX } else { 0 },
        ),
        0x67 => pack_sat(dst, s, 16, false),
        0x68 => interleave::<u8>(dst, s, true),
        0x69 => interleave::<u16>(dst, s, true),
        0x6a => interleave::<u32>(dst, s, true),
        0x6b => pack_sat(dst, s, 32, true),
        0x6c => interleave::<u64>(dst, s, false),
        0x6d => interleave::<u64>(dst, s, true),
        0x74 => zip_lanes::<u8>(dst, s, |a, b| if a == b { 0xff } else { 0 }),
        0x75 => zip_lanes::<u16>(dst, s, |a, b| if a == b { 0xffff } else { 0 }),
        0x76 => zip_lanes::<u32>(dst, s, |a, b| if a == b { u32::MAX } else { 0 }),
        0xd1 => shift_logical::<u16>(dst, count, false, 16),
        0xd2 => shift_logical::<u32>(dst, count, false, 32),
        0xd3 => shift_logical::<u64>(dst, count, false, 64),
        0xe1 => sra::<u16>(dst, count, 16),
        0xe2 => sra::<u32>(dst, count, 32),
        0xf1 => shift_logical::<u16>(dst, count, true, 16),
        0xf2 => shift_logical::<u32>(dst, count, true, 32),
        0xf3 => shift_logical::<u64>(dst, count, true, 64),
        0xd4 => zip_lanes::<u64>(dst, s, |a, b| a.wrapping_add(b)),
        0xd5 => zip_lanes::<u16>(dst, s, |a, b| (a as i16).wrapping_mul(b as i16) as u16),
        0xd8 => zip_lanes::<u8>(dst, s, |a, b| a.saturating_sub(b)),
        0xd9 => zip_lanes::<u16>(dst, s, |a, b| a.saturating_sub(b)),
        0xda => zip_lanes::<u8>(dst, s, |a, b| a.min(b)),
        0xdb => dst & s,
        0xdc => zip_lanes::<u8>(dst, s, |a, b| a.saturating_add(b)),
        0xdd => zip_lanes::<u16>(dst, s, |a, b| a.saturating_add(b)),
        0xde => zip_lanes::<u8>(dst, s, |a, b| a.max(b)),
        0xdf => !dst & s,
        0xe0 => zip_lanes::<u8>(dst, s, |a, b| ((a as u16 + b as u16 + 1) >> 1) as u8),
        0xe3 => zip_lanes::<u16>(dst, s, |a, b| ((a as u32 + b as u32 + 1) >> 1) as u16),
        0xe4 => zip_lanes::<u16>(dst, s, |a, b| ((a as u32 * b as u32) >> 16) as u16),
        0xe5 => zip_lanes::<u16>(dst, s, |a, b| {
            ((a as i16 as i32 * b as i16 as i32) >> 16) as u16
        }),
        0xe8 => zip_lanes::<u8>(dst, s, |a, b| (a as i8).saturating_sub(b as i8) as u8),
        0xe9 => zip_lanes::<u16>(dst, s, |a, b| (a as i16).saturating_sub(b as i16) as u16),
        0xea => zip_lanes::<u16>(dst, s, |a, b| (a as i16).min(b as i16) as u16),
        0xeb => dst | s,
        0xec => zip_lanes::<u8>(dst, s, |a, b| (a as i8).saturating_add(b as i8) as u8),
        0xed => zip_lanes::<u16>(dst, s, |a, b| (a as i16).saturating_add(b as i16) as u16),
        0xee => zip_lanes::<u16>(dst, s, |a, b| (a as i16).max(b as i16) as u16),
        0xef => dst ^ s,
        0xf4 => {
            let mut r = 0;
            for k in 0..2 {
                u64::put(
                    &mut r,
                    k,
                    (u32::get(dst, 2 * k) as u64) * (u32::get(s, 2 * k) as u64),
                );
            }
            r
        }
        0xf5 => {
            let mut r = 0;
            for k in 0..4 {
                let lo = (u16::get(dst, 2 * k) as i16 as i32) * (u16::get(s, 2 * k) as i16 as i32);
                let hi = (u16::get(dst, 2 * k + 1) as i16 as i32)
                    * (u16::get(s, 2 * k + 1) as i16 as i32);
                u32::put(&mut r, k, lo.wrapping_add(hi) as u32);
            }
            r
        }
        0xf6 => {
            let mut r = 0;
            for k in 0..2 {
                let sum: u64 = (0..8)
                    .map(|j| {
                        (u8::get(dst, 8 * k + j) as i32 - u8::get(s, 8 * k + j) as i32)
                            .unsigned_abs() as u64
                    })
                    .sum();
                u64::put(&mut r, k, sum);
            }
            r
        }
        0xf8 => zip_lanes::<u8>(dst, s, |a, b| a.wrapping_sub(b)),
        0xf9 => zip_lanes::<u16>(dst, s, |a, b| a.wrapping_sub(b)),
        0xfa => zip_lanes::<u32>(dst, s, |a, b| a.wrapping_sub(b)),
        0xfb => zip_lanes::<u64>(dst, s, |a, b| a.wrapping_sub(b)),
        0xfc => zip_lanes::<u8>(dst, s, |a, b| a.wrapping_add(b)),
        0xfd => zip_lanes::<u16>(dst, s, |a, b| a.wrapping_add(b)),
        _ => zip_lanes::<u32>(dst, s, |a, b| a.wrapping_add(b)),
    }
}

/// ModRM forms that are invalid encodings: memory-only instructions with mod == 3 and
/// register-only ones with a memory operand, plus the unassigned /reg values of the shift groups
/// 0x71-0x73. `group` is ModRM.reg & 7.
fn form_valid(opcode: u8, prefix: Prefix, is_register: bool, group: u8) -> bool {
    match opcode {
        0x13 | 0x17 | 0x2b | 0xe7 => !is_register,
        // movlps/movhps have register forms (movhlps/movlhps); movlpd/movhpd do not
        0x12 | 0x16 => prefix == Prefix::Np || !is_register,
        0x50 | 0xc5 | 0xd7 | 0xf7 => is_register,
        // movq2dq/movdq2q exist only as register forms; 66 0F D6 is movq xmm/m64, xmm
        0xd6 => prefix == Prefix::P66 || is_register,
        0x71 | 0x72 => is_register && matches!(group, 2 | 4 | 6),
        // the byte shifts /3 and /7 exist only for xmm registers
        0x73 => {
            is_register
                && if prefix == Prefix::Np {
                    matches!(group, 2 | 6)
                } else {
                    matches!(group, 2 | 3 | 6 | 7)
                }
        }
        _ => true,
    }
}

/// Which mandatory prefixes each implemented opcode accepts. This is the only prefix validation:
/// `sse` assumes any (opcode, prefix) pair that gets past here is handled by its arm.
fn allowed(opcode: u8, prefix: Prefix) -> bool {
    use Prefix::*;
    if mmx_form(opcode, prefix) {
        return true;
    }
    match opcode {
        // movups/movss/movupd/movsd (load and store), sqrt, add, mul, cvtps2pd family, sub, min,
        // div, max, cmp: all four prefixes
        0x10 | 0x11 | 0x51 | 0x58 | 0x59 | 0x5a | 0x5c..=0x5f | 0xc2 => true,
        // movlps/movlpd, movhps/movhpd (and movhlps/movlhps), unpck, movaps/movapd, movnt,
        // (u)comiss/sd, movmsk, and/andn/or/xor, shufps/shufpd: packed only
        0x12
        | 0x13
        | 0x14
        | 0x15
        | 0x16
        | 0x17
        | 0x28
        | 0x29
        | 0x2b
        | 0x2e
        | 0x2f
        | 0x50
        | 0x54..=0x57
        | 0xc6 => {
            matches!(prefix, Np | P66)
        }
        // cvtsi2ss/sd, cvttss2si/cvttsd2si, cvtss2si/cvtsd2si: scalar only
        0x2a | 0x2c | 0x2d => matches!(prefix, F3 | F2),
        // rsqrtps/rsqrtss, rcpps/rcpss
        0x52 | 0x53 => matches!(prefix, Np | F3),
        // cvtdq2ps, cvtps2dq, cvttps2dq
        0x5b => prefix != F2,
        // movdqa/movdqu load, movdqa/movdqu store, movd/movq store and movq load
        0x6f | 0x7f | 0x7e => matches!(prefix, P66 | F3),
        // pshufd/pshufhw/pshuflw, cvtdq2pd/cvtpd2dq/cvttpd2dq
        0x70 | 0xe6 => prefix != Np,
        // emms (the only MMX instruction implemented)
        0x77 => prefix == Np,
        // SSE2 integer instructions, 66-prefixed XMM forms only (the unprefixed ones are MMX)
        0x60..=0x6e
        | 0x71..=0x76
        | 0xc4
        | 0xc5
        | 0xd1..=0xd6
        | 0xd7..=0xdf
        | 0xe0..=0xe5
        | 0xe7..=0xef
        | 0xf1..=0xf6
        | 0xf8..=0xfe => prefix == P66,
        _ => false,
    }
}

/// psrl*/psll* on every lane of type `T` (`width` bits). Counts of `width` or more clear the
/// register, as the SDM specifies; Rust's own shifts would panic or wrap there.
fn shift_logical<T: Lane>(x: u128, count: u64, left: bool, width: u32) -> u128 {
    if count >= width as u64 {
        return 0;
    }
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        let shifted = if left { v << count } else { v >> count };
        put_u64::<T>(&mut r, k, shifted, width);
    }
    r
}

/// psra*: arithmetic right shift. Counts of `width` or more behave as `width - 1` (all sign bits).
fn sra<T: Lane>(x: u128, count: u64, width: u32) -> u128 {
    let count = count.min(width as u64 - 1);
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        // Left-align the lane in an i64 so the arithmetic shift sign-extends it.
        let align = 64 - width;
        let shifted = (((v << align) as i64) >> align) >> count;
        put_u64::<T>(&mut r, k, shifted as u64, width);
    }
    r
}

/// Lane `k` zero-extended to u64; `Lane::get` returns the lane type itself, which the generic
/// shift code cannot do arithmetic on.
fn lane_u64<T: Lane>(x: u128, k: usize) -> u64 {
    let w = 128 / T::N;
    ((x >> (k * w)) as u64) & if w == 64 { u64::MAX } else { (1u64 << w) - 1 }
}

/// ORs the low `width` bits of `v` into lane `k` of `r`.
fn put_u64<T: Lane>(r: &mut u128, k: usize, v: u64, width: u32) {
    let m = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    *r |= ((v & m) as u128) << (k * width as usize);
}

#[path = "sse_ext.rs"]
mod ext;

#[cfg(test)]
#[path = "sse_tests.rs"]
mod tests;
