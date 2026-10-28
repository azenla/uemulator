//! EVEX floating-point instructions that are not an SSE kernel on four lanes: compares into
//! opmasks, FMA, the AVX-512 arithmetic helpers (reciprocal approximations, getexp, getmant,
//! scalef, fixupimm, rndscale, reduce, range, fpclass) and every conversion. All of it computes
//! through the soft-float engine of `sse.rs`; masked-out elements are never computed, so they
//! cannot raise flags or faults.

use super::super::fma::{FmaForm, FmaSign, fused_lane, half_to_single, single_to_half};
use super::*;

const OPCODE_MAP_0F: u16 = 1;
const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;

/// vcmp predicates and vfpclass categories use the low bits of the immediate.
const COMPARE_PREDICATE_MASK: i64 = 0x1f;
const FPCLASS_QNAN: u8 = 1 << 0;
const FPCLASS_POSITIVE_ZERO: u8 = 1 << 1;
const FPCLASS_NEGATIVE_ZERO: u8 = 1 << 2;
const FPCLASS_POSITIVE_INFINITY: u8 = 1 << 3;
const FPCLASS_NEGATIVE_INFINITY: u8 = 1 << 4;
const FPCLASS_DENORMAL: u8 = 1 << 5;
const FPCLASS_NEGATIVE_FINITE: u8 = 1 << 6;
const FPCLASS_SNAN: u8 = 1 << 7;

/// vrndscale/vreduce immediate: bits 7:4 scale (round to a multiple of 2^-M), bit 3 suppresses the
/// precision exception, bit 2 selects MXCSR.RC, bits 1:0 the rounding mode.
const SCALE_SHIFT: u32 = 4;
const SCALE_MASK: u8 = 0xf;
const RNDSCALE_SUPPRESS_PRECISION: u8 = 1 << 3;
const RNDSCALE_USE_MXCSR: u8 = 1 << 2;
const RNDSCALE_MODE_MASK: u8 = 3;

/// vgetmant immediate: bits 1:0 interval, bits 3:2 sign control.
const GETMANT_INTERVAL_MASK: u8 = 3;
const GETMANT_SIGN_SHIFT: u32 = 2;
const GETMANT_SIGN_MASK: u8 = 3;
const INTERVAL_ONE_TO_TWO: u8 = 0;
const INTERVAL_HALF_TO_TWO: u8 = 1;
const INTERVAL_HALF_TO_ONE: u8 = 2;
const SIGN_FROM_SOURCE: u8 = 0;

/// vfixupimm: 3-bit token classes and the 4-bit responses stored in the table operand.
const TOKEN_QNAN: u32 = 0;
const TOKEN_SNAN: u32 = 1;
const TOKEN_ZERO: u32 = 2;
const TOKEN_ONE: u32 = 3;
const TOKEN_NEGATIVE_INFINITY: u32 = 4;
const TOKEN_POSITIVE_INFINITY: u32 = 5;
const TOKEN_NEGATIVE_VALUE: u32 = 6;
const TOKEN_POSITIVE_VALUE: u32 = 7;
const RESPONSE_BITS: u32 = 4;
/// Largest scale count vscalef needs: beyond this every finite result overflows or underflows.
const SCALEF_COUNT_LIMIT: i64 = 1 << 14;
const FIXUPIMM_NINETY: u64 = 90;

fn format_of(wide: bool) -> Format {
    if wide { DOUBLE } else { SINGLE }
}

/// Unsigned integer to float, rounded once under the current mode.
fn uint_to_float(env: &mut FpEnv, format: Format, value: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    let magnitude = value as u128;
    let leading = leading_bit(magnitude);
    round_pack(
        env,
        format,
        false,
        leading as i32,
        magnitude << (WIDE_LEADING_BIT - leading),
    )
}

/// Float to an unsigned `width`-bit integer. A NaN, an infinity or a value that does not fit gives
/// all ones and raises invalid; a negative value that rounds to zero gives 0 (inexact).
fn float_to_uint(env: &mut FpEnv, format: Format, bits: u64, width: u32, truncate: bool) -> u64 {
    let all_ones = if width == 64 {
        u64::MAX
    } else {
        (1 << width) - 1
    };
    let value = unpack(env, format, bits);
    match value.kind {
        Kind::Zero => return 0,
        Kind::Finite => {}
        _ => {
            env.flags |= FLAG_INVALID;
            return all_ones;
        }
    }
    if value.exponent >= 64 {
        env.flags |= FLAG_INVALID;
        return all_ones;
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
    if (value.sign && magnitude != 0) || magnitude > all_ones as u128 {
        env.flags |= FLAG_INVALID;
        return all_ones;
    }
    if remainder != 0 {
        env.flags |= FLAG_PRECISION;
    }
    magnitude as u64
}

/// vgetexp: floor(log2(|x|)) as a float.
fn get_exponent(env: &mut FpEnv, format: Format, bits: u64) -> u64 {
    let value = unpack(env, format, bits);
    match value.kind {
        Kind::SignalingNan => {
            env.flags |= FLAG_INVALID;
            value.bits | format.quiet_bit()
        }
        Kind::QuietNan => value.bits,
        Kind::Infinity => format.infinity(false),
        Kind::Zero => format.infinity(true),
        Kind::Finite => int_to_float(env, format, value.exponent as i64),
    }
}

/// vgetmant: the mantissa of `bits` normalised to the interval and sign `control` selects.
fn get_mantissa(env: &mut FpEnv, format: Format, bits: u64, control: u8) -> u64 {
    let value = unpack(env, format, bits);
    let interval = control & GETMANT_INTERVAL_MASK;
    let sign_control = control >> GETMANT_SIGN_SHIFT & GETMANT_SIGN_MASK;
    match value.kind {
        Kind::SignalingNan => {
            env.flags |= FLAG_INVALID;
            return value.bits | format.quiet_bit();
        }
        Kind::QuietNan => return value.bits,
        _ => {}
    }
    // Sign control 2 turns a negative input (other than -0) into the indefinite.
    if sign_control == 2 && value.sign && value.kind != Kind::Zero {
        return invalid(env, format);
    }
    let sign = sign_control == SIGN_FROM_SOURCE && value.sign;
    let sign_bit = if sign { format.sign_bit() } else { 0 };
    if value.kind == Kind::Zero {
        return format.zero(sign);
    }
    let one = (format.bias() as u64) << format.fraction_bits;
    if value.kind == Kind::Infinity {
        return sign_bit | one;
    }
    // Normalised fraction of the significand with exponent 0 is in [1, 2).
    let fraction = value.significand & format.fraction_mask();
    let half_or_more = |exponent: i32| -> u64 {
        ((exponent + format.bias()) as u64) << format.fraction_bits | fraction
    };
    let exponent = match interval {
        INTERVAL_ONE_TO_TWO => 0,
        INTERVAL_HALF_TO_TWO => {
            // [0.5, 2): values in [1, 2) with the leading fraction bit set map to [0.5, 1)
            if fraction >> (format.fraction_bits - 1) != 0 {
                -1
            } else {
                0
            }
        }
        INTERVAL_HALF_TO_ONE => -1,
        _ => {
            // [0.75, 1.5): significands of 1.5 or more halve
            let three_halves = 1u64 << (format.fraction_bits - 1);
            if fraction >= three_halves { -1 } else { 0 }
        }
    };
    sign_bit | half_or_more(exponent)
}

/// x * 2^floor(y), the semantics of vscalef including the special-value table.
fn scale_float(env: &mut FpEnv, format: Format, a: u64, b: u64) -> u64 {
    let x = unpack(env, format, a);
    let y = unpack(env, format, b);
    if x.is_nan() || y.is_nan() {
        if x.kind == Kind::SignalingNan || y.kind == Kind::SignalingNan {
            env.flags |= FLAG_INVALID;
        }
        let chosen = if x.is_nan() { &x } else { &y };
        return chosen.bits | format.quiet_bit();
    }
    match (x.kind, y.kind) {
        (Kind::Infinity, Kind::Infinity) if y.sign => return invalid(env, format),
        (Kind::Infinity, _) => return x.bits,
        (Kind::Zero, Kind::Infinity) if !y.sign => return invalid(env, format),
        (Kind::Zero, _) => return x.bits,
        (_, Kind::Infinity) => {
            return if y.sign {
                format.zero(x.sign)
            } else {
                format.infinity(x.sign)
            };
        }
        _ => {}
    }
    // floor(y) as a clamped integer
    let count: i64 = if y.kind == Kind::Zero {
        0
    } else if y.exponent >= 40 {
        if y.sign {
            -SCALEF_COUNT_LIMIT
        } else {
            SCALEF_COUNT_LIMIT
        }
    } else if y.exponent < 0 {
        // |y| < 1: floor is 0 for positive values and -1 for negative ones
        if y.sign { -1 } else { 0 }
    } else {
        let shift = format.fraction_bits as i32 - y.exponent;
        let integer = (y.significand >> shift) as i64;
        let fractional = y.significand & ((1u64 << shift) - 1) != 0;
        if y.sign {
            -(integer + fractional as i64)
        } else {
            integer
        }
    };
    let count = count.clamp(-SCALEF_COUNT_LIMIT, SCALEF_COUNT_LIMIT) as i32;
    round_pack(env, format, x.sign, x.exponent + count, widen(format, &x))
}

/// vrndscale: rounds `bits` to a multiple of 2^-scale under the rounding `immediate` selects.
fn round_scaled(env: &mut FpEnv, format: Format, bits: u64, immediate: u8) -> u64 {
    let scale = (immediate >> SCALE_SHIFT & SCALE_MASK) as i32;
    let rounding = if immediate & RNDSCALE_USE_MXCSR != 0 {
        env.rounding
    } else {
        (immediate & RNDSCALE_MODE_MASK) as u32
    };
    let quiet_precision = immediate & RNDSCALE_SUPPRESS_PRECISION != 0;
    let denormal_before = env.flags & FLAG_DENORMAL;
    let value = unpack(env, format, bits);
    env.flags = (env.flags & !FLAG_DENORMAL) | denormal_before;
    match value.kind {
        Kind::SignalingNan => {
            env.flags |= FLAG_INVALID;
            return value.bits | format.quiet_bit();
        }
        Kind::QuietNan | Kind::Infinity | Kind::Zero => return value.bits,
        Kind::Finite => {}
    }
    // Once the scaled magnitude has no fractional bits the value is already a multiple.
    if value.exponent + scale >= format.fraction_bits as i32 {
        return value.bits;
    }
    let shift = (format.fraction_bits as i32 - value.exponent - scale) as u32;
    const NEGLIGIBLE_SHIFT: u32 = 100;
    let significand = value.significand as u128;
    let (mut kept, remainder, half) = if shift > NEGLIGIBLE_SHIFT {
        (0, 1, 2)
    } else {
        (
            significand >> shift,
            significand & ((1u128 << shift) - 1),
            1u128 << (shift - 1),
        )
    };
    if rounds_up(rounding, value.sign, kept, remainder, half) {
        kept += 1;
    }
    if remainder != 0 && !quiet_precision {
        env.flags |= FLAG_PRECISION;
    }
    if kept == 0 {
        return format.zero(value.sign);
    }
    let leading = leading_bit(kept);
    let sign_bits = if value.sign { format.sign_bit() } else { 0 };
    let exponent_field = (leading as i32 - scale + format.bias()) as u64;
    let fraction = ((kept << (format.fraction_bits - leading)) as u64) & format.fraction_mask();
    sign_bits | (exponent_field << format.fraction_bits) | fraction
}

/// vfpclass: whether the value falls into any class `selector` names.
fn classify(format: Format, bits: u64, selector: u8) -> bool {
    // Classification looks at the raw encoding: DAZ does not apply.
    let exponent = (bits >> format.fraction_bits) & format.exponent_mask();
    let fraction = bits & format.fraction_mask();
    let negative = bits & format.sign_bit() != 0;
    let mut classes = 0u8;
    if exponent == format.exponent_mask() {
        if fraction == 0 {
            classes |= if negative {
                FPCLASS_NEGATIVE_INFINITY
            } else {
                FPCLASS_POSITIVE_INFINITY
            };
        } else if fraction & format.quiet_bit() != 0 {
            classes |= FPCLASS_QNAN;
        } else {
            classes |= FPCLASS_SNAN;
        }
    } else if exponent == 0 {
        if fraction == 0 {
            classes |= if negative {
                FPCLASS_NEGATIVE_ZERO
            } else {
                FPCLASS_POSITIVE_ZERO
            };
        } else {
            classes |= FPCLASS_DENORMAL;
            if negative {
                classes |= FPCLASS_NEGATIVE_FINITE;
            }
        }
    } else if negative {
        classes |= FPCLASS_NEGATIVE_FINITE;
    }
    classes & selector != 0
}

/// One element of vfixupimm: replaces `source` (vvvv) by the table response for its class.
fn fixup_element(
    env: &mut FpEnv,
    format: Format,
    old: u64,
    source: u64,
    table: u64,
    immediate: u8,
) -> u64 {
    let value = unpack(env, format, source);
    let one = (format.bias() as u64) << format.fraction_bits;
    let token = match value.kind {
        Kind::QuietNan => TOKEN_QNAN,
        Kind::SignalingNan => TOKEN_SNAN,
        Kind::Zero => TOKEN_ZERO,
        Kind::Infinity if value.sign => TOKEN_NEGATIVE_INFINITY,
        Kind::Infinity => TOKEN_POSITIVE_INFINITY,
        Kind::Finite if value.bits == one => TOKEN_ONE,
        Kind::Finite if value.sign => TOKEN_NEGATIVE_VALUE,
        Kind::Finite => TOKEN_POSITIVE_VALUE,
    };
    let signal = |env: &mut FpEnv, flag: u32, bit: u32| {
        if immediate >> bit & 1 == 1 {
            env.flags |= flag;
        }
    };
    match token {
        TOKEN_ZERO => {
            signal(env, FLAG_DIVIDE_BY_ZERO, 0);
            signal(env, FLAG_INVALID, 1);
        }
        TOKEN_ONE => {
            signal(env, FLAG_DIVIDE_BY_ZERO, 2);
            signal(env, FLAG_INVALID, 3);
        }
        TOKEN_SNAN => signal(env, FLAG_INVALID, 4),
        TOKEN_NEGATIVE_INFINITY => signal(env, FLAG_INVALID, 5),
        TOKEN_NEGATIVE_VALUE => signal(env, FLAG_INVALID, 6),
        TOKEN_POSITIVE_INFINITY => signal(env, FLAG_INVALID, 7),
        _ => {}
    }
    let response = (table >> (RESPONSE_BITS * token)) & 0xf;
    let max_finite = format.max_finite(false);
    let pi_over_two = if format.width() == 32 {
        core::f32::consts::FRAC_PI_2.to_bits() as u64
    } else {
        core::f64::consts::FRAC_PI_2.to_bits()
    };
    let ninety = if format.width() == 32 {
        (FIXUPIMM_NINETY as f32).to_bits() as u64
    } else {
        (FIXUPIMM_NINETY as f64).to_bits()
    };
    let half = ((format.bias() - 1) as u64) << format.fraction_bits;
    match response {
        0 => old,
        1 => value.bits,
        2 => value.bits | format.quiet_bit(),
        3 => format.indefinite(),
        4 => format.infinity(true),
        5 => format.infinity(false),
        6 => format.infinity(value.sign),
        7 => format.zero(true),
        8 => format.zero(false),
        9 => one | format.sign_bit(),
        10 => one,
        11 => half,
        12 => ninety,
        13 => pi_over_two,
        14 => max_finite,
        _ => max_finite | format.sign_bit(),
    }
}

/// vrange: min/max (optionally by magnitude) with a sign control.
fn range_element(env: &mut FpEnv, format: Format, a: u64, b: u64, immediate: u8) -> u64 {
    let x = unpack(env, format, a);
    let y = unpack(env, format, b);
    if x.is_nan() || y.is_nan() {
        if x.kind == Kind::SignalingNan || y.kind == Kind::SignalingNan {
            env.flags |= FLAG_INVALID;
        }
        let chosen = if x.is_nan() { &x } else { &y };
        return chosen.bits | format.quiet_bit();
    }
    let by_magnitude = immediate & 2 != 0;
    let want_max = immediate & 1 != 0;
    let (ax, ay) = (Unpacked { sign: false, ..x }, Unpacked { sign: false, ..y });
    let ordering = if by_magnitude {
        order(&ax, &ay)
    } else {
        order(&x, &y)
    };
    // Ties go to the first operand.
    let pick_first = !matches!(
        (want_max, ordering),
        (true, core::cmp::Ordering::Less) | (false, core::cmp::Ordering::Greater)
    );
    let chosen = if pick_first { &x } else { &y };
    let sign = match immediate >> 2 & 3 {
        0 => chosen.sign,
        1 => x.sign,
        2 => false,
        _ => true,
    };
    let magnitude = chosen.bits & !format.sign_bit();
    magnitude | if sign { format.sign_bit() } else { 0 }
}

impl Cpu {
    pub(super) fn exec_evex_fp(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        let format = format_of(wide);
        let element = format.bytes() as usize;
        match (map, opcode, prefix) {
            (OPCODE_MAP_0F, 0xc2, Np | P66 | F3 | F2) => {
                let (format, scalar) = match prefix {
                    Np => (SINGLE, false),
                    P66 => (DOUBLE, false),
                    F3 => (SINGLE, true),
                    _ => (DOUBLE, true),
                };
                self.fp_compare(i, next, format, scalar)?;
            }
            (OPCODE_MAP_0F, 0x2e | 0x2f, Np | P66) => {
                let element = if prefix == Np { 4 } else { 8 };
                let shape = RmShape {
                    bytes: element,
                    element,
                    broadcast: false,
                    suppress: None,
                };
                let source = self.fetch_rm(i, next, shape)?[0];
                let old = self.zmm(i.reg as usize)[0];
                self.with_static_rounding(i, Rounding::Sae, |cpu| {
                    cpu.run_legacy_registers(i, old, source)
                })?;
            }
            (OPCODE_MAP_0F38, 0x96..=0x9f | 0xa6..=0xaf | 0xb6..=0xbf, P66) => {
                self.fma(i, next, opcode, format)?;
            }
            (OPCODE_MAP_0F38, 0x4c | 0x4e, P66) => {
                let square_root = opcode == 0x4e;
                self.fp_packed(
                    i,
                    next,
                    OpShape::elementwise(element).unary(),
                    format,
                    move |_, _, b, _| reciprocal(format, b, square_root),
                )?;
            }
            (OPCODE_MAP_0F38, 0x4d | 0x4f, P66) => {
                let square_root = opcode == 0x4f;
                self.fp_scalar(i, next, format, Rounding::None, false, move |_, _, b, _| {
                    reciprocal(format, b, square_root)
                })?;
            }
            (OPCODE_MAP_0F38, 0x42, P66) => {
                let mut shape = OpShape::elementwise(element).unary();
                shape.rounding = Rounding::Sae;
                self.fp_packed(i, next, shape, format, move |env, _, b, _| {
                    get_exponent(env, format, b)
                })?;
            }
            (OPCODE_MAP_0F38, 0x43, P66) => {
                self.fp_scalar(
                    i,
                    next,
                    format,
                    Rounding::Sae,
                    false,
                    move |env, _, b, _| get_exponent(env, format, b),
                )?;
            }
            (OPCODE_MAP_0F3A, 0x26, P66) => {
                let control = i.imm as u8;
                let mut shape = OpShape::elementwise(element).unary();
                shape.rounding = Rounding::Sae;
                self.fp_packed(i, next, shape, format, move |env, _, b, _| {
                    get_mantissa(env, format, b, control)
                })?;
            }
            (OPCODE_MAP_0F3A, 0x27, P66) => {
                let control = i.imm as u8;
                self.fp_scalar(
                    i,
                    next,
                    format,
                    Rounding::Sae,
                    false,
                    move |env, _, b, _| get_mantissa(env, format, b, control),
                )?;
            }
            (OPCODE_MAP_0F38, 0x2c, P66) => {
                let mut shape = OpShape::elementwise(element);
                shape.rounding = Rounding::Embedded;
                self.fp_packed(i, next, shape, format, move |env, a, b, _| {
                    scale_float(env, format, a, b)
                })?;
            }
            (OPCODE_MAP_0F38, 0x2d, P66) => {
                self.fp_scalar(
                    i,
                    next,
                    format,
                    Rounding::Embedded,
                    false,
                    move |env, a, b, _| scale_float(env, format, a, b),
                )?;
            }
            (OPCODE_MAP_0F3A, 0x54, P66) => {
                let immediate = i.imm as u8;
                let mut shape = OpShape::elementwise(element);
                shape.rounding = Rounding::Sae;
                // the table operand is an integer vector of the same element size
                self.fp_packed(i, next, shape, format, move |env, source, table, old| {
                    fixup_element(env, format, old, source, table, immediate)
                })?;
            }
            (OPCODE_MAP_0F3A, 0x55, P66) => {
                let immediate = i.imm as u8;
                self.fp_scalar(
                    i,
                    next,
                    format,
                    Rounding::Sae,
                    true,
                    move |env, source, table, old| {
                        fixup_element(env, format, old, source, table, immediate)
                    },
                )?;
            }
            (OPCODE_MAP_0F3A, 0x08 | 0x09, P66) => {
                let immediate = i.imm as u8;
                let mut shape = OpShape::elementwise(if opcode == 0x08 { 4 } else { 8 }).unary();
                shape.rounding = Rounding::Sae;
                let format = if opcode == 0x08 { SINGLE } else { DOUBLE };
                self.fp_packed(i, next, shape, format, move |env, _, b, _| {
                    round_scaled(env, format, b, immediate)
                })?;
            }
            (OPCODE_MAP_0F3A, 0x0a | 0x0b, P66) => {
                let immediate = i.imm as u8;
                let format = if opcode == 0x0a { SINGLE } else { DOUBLE };
                self.fp_scalar(
                    i,
                    next,
                    format,
                    Rounding::Sae,
                    false,
                    move |env, _, b, _| round_scaled(env, format, b, immediate),
                )?;
            }
            (OPCODE_MAP_0F3A, 0x56, P66) => {
                let immediate = i.imm as u8;
                let mut shape = OpShape::elementwise(element).unary();
                shape.rounding = Rounding::Sae;
                self.fp_packed(i, next, shape, format, move |env, _, b, _| {
                    let rounded = round_scaled(env, format, b, immediate);
                    arithmetic(env, format, Arithmetic::Subtract, b, rounded)
                })?;
            }
            (OPCODE_MAP_0F3A, 0x57, P66) => {
                let immediate = i.imm as u8;
                self.fp_scalar(
                    i,
                    next,
                    format,
                    Rounding::Sae,
                    false,
                    move |env, _, b, _| {
                        let rounded = round_scaled(env, format, b, immediate);
                        arithmetic(env, format, Arithmetic::Subtract, b, rounded)
                    },
                )?;
            }
            (OPCODE_MAP_0F3A, 0x50, P66) => {
                let immediate = i.imm as u8;
                let mut shape = OpShape::elementwise(element);
                shape.rounding = Rounding::Sae;
                self.fp_packed(i, next, shape, format, move |env, a, b, _| {
                    range_element(env, format, a, b, immediate)
                })?;
            }
            (OPCODE_MAP_0F3A, 0x51, P66) => {
                let immediate = i.imm as u8;
                self.fp_scalar(
                    i,
                    next,
                    format,
                    Rounding::Sae,
                    false,
                    move |env, a, b, _| range_element(env, format, a, b, immediate),
                )?;
            }
            (OPCODE_MAP_0F3A, 0x66 | 0x67, P66) => {
                let selector = i.imm as u8;
                let scalar = opcode == 0x67;
                self.classify_to_mask(i, next, format, scalar, selector)?;
            }
            (OPCODE_MAP_0F38, 0x13, P66) if !wide => self.convert_half_to_single(i, next)?,
            (OPCODE_MAP_0F3A, 0x1d, P66) if !wide => self.convert_single_to_half(i, next)?,
            _ => return self.exec_evex_convert(i, next, map, opcode, prefix, wide),
        }
        Ok(true)
    }

    /// Element-wise packed FP operation: `f(env, a, b, old)` per unmasked element, flags committed
    /// (and unmasked exceptions raised) before anything is written.
    fn fp_packed(
        &mut self,
        i: &Insn,
        next: u64,
        shape: OpShape,
        format: Format,
        f: impl Fn(&mut FpEnv, u64, u64, u64) -> u64,
    ) -> ExecResult<()> {
        let element = format.bytes() as usize;
        self.exec_vector_op(i, next, shape, |cpu, old, a, b, bytes| {
            let mask = cpu.write_mask(i);
            let mut env = FpEnv::new(cpu.mxcsr);
            let out = build(bytes, element, |k| {
                if mask >> k & 1 == 1 {
                    f(
                        &mut env,
                        super::element(a, element, k),
                        super::element(b, element, k),
                        super::element(old, element, k),
                    )
                } else {
                    0
                }
            });
            cpu.fp_commit(&env)?;
            Ok(out)
        })
    }

    /// Scalar FP operation on element 0: the rest of the low lane comes from vvvv (or from the
    /// old destination for the fused and fixup forms), everything above is zeroed.
    fn fp_scalar(
        &mut self,
        i: &Insn,
        next: u64,
        format: Format,
        rounding: Rounding,
        upper_from_destination: bool,
        f: impl Fn(&mut FpEnv, u64, u64, u64) -> u64,
    ) -> ExecResult<()> {
        let element = format.bytes() as usize;
        if i.evex_b && i.md == 3 && rounding == Rounding::None {
            return Err(Exception::invalid_opcode());
        }
        let shape = RmShape {
            bytes: element,
            element,
            broadcast: false,
            suppress: None,
        };
        let b = self.fetch_rm(i, next, shape)?[0] as u64 & size_mask(element);
        let first = self.zmm(i.vex_v as usize)[0];
        let old = self.zmm(i.reg as usize)[0];
        let base = if upper_from_destination { old } else { first };
        let enabled = i.evex_mask == 0 || self.opmask[i.evex_mask as usize] & 1 == 1;
        let result = if enabled {
            self.with_static_rounding(i, rounding, |cpu| {
                let mut env = FpEnv::new(cpu.mxcsr);
                let bits = f(
                    &mut env,
                    first as u64 & size_mask(element),
                    b,
                    old as u64 & size_mask(element),
                );
                cpu.fp_commit(&env)?;
                Ok(bits)
            })?
        } else {
            0
        };
        let lane = base & !(size_mask(element) as u128) | result as u128;
        self.commit_scalar(i, i.reg as usize, element, lane);
        Ok(())
    }

    fn fp_compare(&mut self, i: &Insn, next: u64, format: Format, scalar: bool) -> ExecResult<()> {
        let element = format.bytes() as usize;
        let bytes = if scalar {
            element
        } else {
            self.vector_bytes(i, Rounding::Sae)?
        };
        let mask = self.write_mask(i);
        let shape = if scalar {
            RmShape {
                bytes: element,
                element,
                broadcast: false,
                suppress: None,
            }
        } else {
            RmShape {
                suppress: (i.evex_mask != 0).then_some(mask),
                ..RmShape::full(bytes, element)
            }
        };
        let second = self.fetch_rm(i, next, shape)?;
        let first = self.zmm(i.vex_v as usize);
        let predicate = (i.imm & COMPARE_PREDICATE_MASK) as u8;
        let bits = self.with_static_rounding(i, Rounding::Sae, |cpu| {
            let mut env = FpEnv::new(cpu.mxcsr);
            let mut bits = 0u64;
            for k in 0..bytes / element {
                if mask >> k & 1 == 1
                    && fma::compare_predicate_avx(
                        &mut env,
                        format,
                        predicate,
                        super::element(&first, element, k),
                        super::element(&second, element, k),
                    )
                {
                    bits |= 1 << k;
                }
            }
            cpu.fp_commit(&env)?;
            Ok(bits)
        })?;
        self.opmask[(i.reg & 7) as usize] = bits;
        Ok(())
    }

    fn classify_to_mask(
        &mut self,
        i: &Insn,
        next: u64,
        format: Format,
        scalar: bool,
        selector: u8,
    ) -> ExecResult<()> {
        let element = format.bytes() as usize;
        let bytes = if scalar {
            element
        } else {
            self.vector_bytes(i, Rounding::None)?
        };
        let mask = self.write_mask(i);
        let shape = RmShape {
            suppress: (i.evex_mask != 0).then_some(mask),
            broadcast: !scalar,
            ..RmShape::full(bytes, element)
        };
        let source = self.fetch_rm(i, next, shape)?;
        let mut bits = 0u64;
        for k in 0..bytes / element {
            if mask >> k & 1 == 1 && classify(format, super::element(&source, element, k), selector)
            {
                bits |= 1 << k;
            }
        }
        self.opmask[(i.reg & 7) as usize] = bits;
        Ok(())
    }

    /// vfmadd/vfmsub/vfnmadd/vfnmsub (132/213/231), vfmaddsub and vfmsubadd.
    fn fma(&mut self, i: &Insn, next: u64, opcode: u8, format: Format) -> ExecResult<()> {
        let form = match opcode >> 4 {
            0x9 => FmaForm::Form132,
            0xa => FmaForm::Form213,
            _ => FmaForm::Form231,
        };
        let kind = opcode & 0xf;
        let scalar = matches!(kind, 0x9 | 0xb | 0xd | 0xf);
        let negate_product = matches!(kind, 0xc..=0xf);
        let subtract = matches!(kind, 0xa | 0xb | 0xe | 0xf);
        let alternate = matches!(kind, 0x6 | 0x7);
        let element = format.bytes() as usize;
        let operation = move |env: &mut FpEnv, lane: usize, [op1, op2, op3]: [u64; 3]| {
            let sign = if alternate {
                // vfmaddsub: even lanes subtract; vfmsubadd: even lanes add
                if lane.is_multiple_of(2) == (kind == 0x6) {
                    FmaSign::Subtract
                } else {
                    FmaSign::Add
                }
            } else if subtract {
                FmaSign::Subtract
            } else {
                FmaSign::Add
            };
            fused_lane(env, format, [op1, op2, op3], form, negate_product, sign)
        };
        if scalar {
            let shape = RmShape {
                bytes: element,
                element,
                broadcast: false,
                suppress: None,
            };
            let op3 = self.fetch_rm(i, next, shape)?[0] as u64 & size_mask(element);
            let op2 = self.zmm(i.vex_v as usize)[0] as u64 & size_mask(element);
            let old = self.zmm(i.reg as usize)[0];
            let op1 = old as u64 & size_mask(element);
            let enabled = i.evex_mask == 0 || self.opmask[i.evex_mask as usize] & 1 == 1;
            let result = if enabled {
                self.with_static_rounding(i, Rounding::Embedded, |cpu| {
                    let mut env = FpEnv::new(cpu.mxcsr);
                    let bits = operation(&mut env, 0, [op1, op2, op3]);
                    cpu.fp_commit(&env)?;
                    Ok(bits)
                })?
            } else {
                0
            };
            let lane = old & !(size_mask(element) as u128) | result as u128;
            self.commit_scalar(i, i.reg as usize, element, lane);
            return Ok(());
        }
        let mut shape = OpShape::elementwise(element);
        shape.rounding = Rounding::Embedded;
        self.exec_vector_op(i, next, shape, |cpu, old, op2, op3, bytes| {
            let mask = cpu.write_mask(i);
            let mut env = FpEnv::new(cpu.mxcsr);
            let out = build(bytes, element, |k| {
                if mask >> k & 1 == 1 {
                    operation(
                        &mut env,
                        k,
                        [
                            super::element(old, element, k),
                            super::element(op2, element, k),
                            super::element(op3, element, k),
                        ],
                    )
                } else {
                    0
                }
            });
            cpu.fp_commit(&env)?;
            Ok(out)
        })
    }

    /// vcvtph2ps: half-precision elements of the r/m operand widened to singles.
    fn convert_half_to_single(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let mut shape = OpShape::elementwise(4).unary();
        shape.rm_element = Some(2);
        shape.rm_bytes = Some(2);
        shape.broadcast = false;
        shape.rounding = Rounding::Sae;
        self.exec_vector_op(i, next, shape, |cpu, _, _, source, bytes| {
            let mask = cpu.write_mask(i);
            let mut env = FpEnv::new(cpu.mxcsr);
            let out = build(bytes, 4, |k| {
                if mask >> k & 1 == 0 {
                    return 0;
                }
                let (single, signaling) = half_to_single(element(source, 2, k) as u16);
                if signaling {
                    env.flags |= FLAG_INVALID;
                }
                single as u64
            });
            cpu.fp_commit(&env)?;
            Ok(out)
        })
    }

    /// vcvtps2ph: singles of ModRM.reg narrowed to halves in r/m.
    fn convert_single_to_half(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::Sae)?;
        let count = bytes / 4;
        let selector = i.imm as u8;
        let source = self.zmm(i.reg as usize);
        let mask = self.write_mask(i);
        let packed = self.with_static_rounding(i, Rounding::Sae, |cpu| {
            let mut env = FpEnv::new(cpu.mxcsr);
            let rounding = if selector & 4 != 0 {
                env.rounding
            } else {
                (selector & 3) as u32
            };
            let mut flags = 0;
            let out = build(count * 2, 2, |k| {
                if mask >> k & 1 == 0 {
                    return 0;
                }
                let (half, raised) = single_to_half(element(&source, 4, k) as u32, rounding);
                flags |= raised;
                half as u64
            });
            env.flags = flags;
            cpu.fp_commit(&env)?;
            Ok(out)
        })?;
        if i.md == 3 {
            self.commit_masked(i, i.rm as usize, 2, count * 2, &packed);
            return Ok(());
        }
        if i.evex_zeroing {
            return Err(Exception::invalid_opcode());
        }
        let address = self.evex_address(i, next, count * 2);
        self.store_vector(address, &packed, count * 2, 2, mask)
    }
}

#[path = "avx512_convert.rs"]
mod convert;
