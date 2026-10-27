use super::decode::*;
use super::exception::Exception;
use super::exec::Loc;
use super::flags::*;
use super::sys::{CR0_EM, CR0_TS};
use super::*;
#[cfg(target_os = "uefi")]
use crate::math::{Float64Ext, FloatExt};

// x87 registers are kept as raw 80-bit values (`Cpu::st`, 10 bytes little-endian: 64-bit
// significand with an explicit integer bit, then 15-bit exponent and sign), but every
// arithmetic operation converts to f64, computes there and converts back. Results therefore
// have 53-bit precision regardless of FCW.PC, and rounding control only affects frndint and
// the fist family. Only moves (fld/fst m80, fxch, fld st(i), fxsave) preserve all 64 bits.
// No x87 exception is ever raised: FSW exception flags are never set and FCW masks are ignored.

/// Number of x87 registers, and the mask that wraps a stack position into them.
const STACK_DEPTH: usize = 8;
const STACK_MASK: usize = STACK_DEPTH - 1;

const EXT_EXPONENT_BIAS: i32 = 16383;
/// Exponent field value for infinity and NaN.
const EXT_EXPONENT_SPECIAL: u16 = 0x7fff;
const EXT_SIGN: u16 = 0x8000;
/// Index of the byte holding the sign and the high exponent bits, and the sign bit within it.
const EXT_SIGN_BYTE: usize = 9;
const EXT_SIGN_BYTE_BIT: u8 = 0x80;
/// Bit 63 of the significand: the explicit integer bit, set in every normalised value.
const EXT_INTEGER_BIT: u64 = 1 << 63;
/// Distance from the f64 fraction (52 bits) up to the extended fraction (63 bits).
const F64_FRACTION_TO_EXT_SHIFT: u32 = 11;
const F64_EXPONENT_BIAS: i32 = 1023;
const F64_EXPONENT_SPECIAL: i32 = 0x7ff;
const F64_FRACTION_BITS: u32 = 52;

/// FCW layout: exception masks in bits 0-5, precision control in 8-9, rounding control in 10-11.
/// The power-on/FINIT value is `FCW_RESET` in cpu/mod.rs.
const FCW_EXCEPTION_MASKS: u16 = 0x3f;
const FCW_ROUNDING_CONTROL_SHIFT: u32 = 10;
const ROUNDING_CONTROL_MASK: u16 = 3;
const ROUND_NEAREST_EVEN: u16 = 0;
const ROUND_DOWN: u16 = 1;
const ROUND_UP: u16 = 2;

/// FSW: TOP in bits 11-13, condition codes C0 (8), C1 (9), C2 (10), C3 (14).
const FSW_TOP_SHIFT: u32 = 11;
const FSW_TOP_MASK: u16 = 7 << FSW_TOP_SHIFT;
const FSW_C0: u16 = 1 << 8;
const FSW_C1: u16 = 1 << 9;
const FSW_C2: u16 = 1 << 10;
const FSW_C3: u16 = 1 << 14;
/// C1 is excluded: fxam and fprem set it themselves and other compares leave it alone.
const FSW_CONDITION_MASK: u16 = FSW_C0 | FSW_C2 | FSW_C3;
/// fnclex keeps TOP and C0-C3; it clears the exception flags (0-7) and the busy bit (15).
const FSW_KEPT_BY_FNCLEX: u16 = 0x7f00;

/// Two-bit tag word values, as stored by fnstenv.
const TAG_VALID: u16 = 0;
const TAG_ZERO: u16 = 1;
const TAG_SPECIAL: u16 = 2;
const TAG_EMPTY: u16 = 3;

/// 32-bit protected-mode FNSTENV/FLDENV image: FCW, FSW, FTW as 32-bit words, then the
/// FIP/FCS/FDP/FDS pointer words, which are not tracked and always written as zero.
/// FNSAVE/FRSTOR append the eight 10-byte registers in stack order right after it. The 16-bit
/// formats selected by a 66 prefix or real mode are not implemented.
const ENV_FSW_OFFSET: u64 = 4;
const ENV_FTW_OFFSET: u64 = 8;
const ENV_POINTER_FIRST_WORD: u64 = 3;
const ENV_POINTER_END_WORD: u64 = 7;
const ENV_SIZE: u64 = 28;
const EXT_REG_SIZE: u64 = 10;

/// FXSAVE image (SDM layout, 512 bytes). FOP/FIP/FDP at 6..24 are not written.
const FXSAVE_AREA_SIZE: u64 = 512;
/// The area must be 16-byte aligned or the instruction is #GP(0).
const FXSAVE_ALIGNMENT: u64 = 16;
const FXSAVE_FSW_OFFSET: u64 = 2;
/// Abridged tag word: one bit per physical register, 1 = not empty.
const FXSAVE_FTW_OFFSET: u64 = 4;
const FXSAVE_MXCSR_OFFSET: u64 = 24;
const FXSAVE_MXCSR_MASK_OFFSET: u64 = 28;
const FXSAVE_ST_OFFSET: u64 = 32;
const FXSAVE_ST_STRIDE: u64 = 16;
const FXSAVE_XMM_OFFSET: u64 = 160;
const FXSAVE_XMM_STRIDE: u64 = 16;

/// MXCSR bits that may be set by ldmxcsr/fxrstor; writing any of 31:16 is #GP. Also reported
/// as MXCSR_MASK in the FXSAVE image.
pub(crate) const MXCSR_VALID_BITS: u32 = 0xffff;

pub(crate) fn ext_to_f64(b: [u8; 10]) -> f64 {
    let significand = u64::from_le_bytes(b[..8].try_into().unwrap());
    let sign_exponent = u16::from_le_bytes([b[8], b[9]]);
    let neg = sign_exponent & EXT_SIGN != 0;
    let exponent = (sign_exponent & !EXT_SIGN) as i32;
    let v = if exponent == EXT_EXPONENT_SPECIAL as i32 {
        // integer bit is ignored when telling infinity from NaN
        if significand << 1 == 0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    } else if significand == 0 {
        0.0
    } else {
        // value = significand * 2^(exponent - bias - 63); split the power so each half stays
        // representable when the full scale factor would overflow or underflow f64
        let scale = exponent - EXT_EXPONENT_BIAS - 63;
        let (first, second) = (scale / 2, scale - scale / 2);
        (significand as f64) * 2f64.powi(first) * 2f64.powi(second)
    };
    if neg { -v } else { v }
}

pub(crate) fn f64_to_ext(v: f64) -> [u8; 10] {
    let bits = v.to_bits();
    let neg = bits >> 63 != 0;
    let exponent = ((bits >> F64_FRACTION_BITS) & F64_EXPONENT_SPECIAL as u64) as i32;
    let fraction = bits & ((1 << F64_FRACTION_BITS) - 1);
    let (sign_exponent, significand): (u16, u64) = if exponent == F64_EXPONENT_SPECIAL {
        (
            EXT_EXPONENT_SPECIAL,
            EXT_INTEGER_BIT | (fraction << F64_FRACTION_TO_EXT_SHIFT),
        )
    } else if exponent == 0 {
        if fraction == 0 {
            (0, 0)
        } else {
            // f64 subnormals are normalised: extended has the range to hold them
            let shift = fraction.leading_zeros();
            (
                ((EXT_EXPONENT_BIAS - F64_EXPONENT_BIAS + 1
                    - (shift as i32 - F64_FRACTION_TO_EXT_SHIFT as i32)) as u16),
                fraction << shift,
            )
        }
    } else {
        (
            (exponent - F64_EXPONENT_BIAS + EXT_EXPONENT_BIAS) as u16,
            EXT_INTEGER_BIT | (fraction << F64_FRACTION_TO_EXT_SHIFT),
        )
    };
    let mut out = [0u8; 10];
    out[..8].copy_from_slice(&significand.to_le_bytes());
    out[8..].copy_from_slice(&(sign_exponent | if neg { EXT_SIGN } else { 0 }).to_le_bytes());
    out
}

/// What an 80-bit register holds, as FXAM reports it and as the tag word records it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExtClass {
    /// Unnormals, pseudo-infinities and pseudo-NaNs: formats the 387 and later reject.
    Unsupported,
    Nan,
    Normal,
    Infinity,
    Zero,
    /// Exponent 0 with a nonzero significand. A pseudo-denormal (integer bit set) is included.
    Denormal,
}

/// Classifies the raw bits, not an f64 conversion: an extended denormal underflows f64 to zero
/// and would otherwise look like one.
fn classify(raw: [u8; 10]) -> ExtClass {
    let significand = u64::from_le_bytes(raw[..8].try_into().unwrap());
    let exponent = u16::from_le_bytes([raw[8], raw[9]]) & !EXT_SIGN;
    let integer_bit_set = significand & EXT_INTEGER_BIT != 0;
    match exponent {
        0 if significand == 0 => ExtClass::Zero,
        0 => ExtClass::Denormal,
        EXT_EXPONENT_SPECIAL if !integer_bit_set => ExtClass::Unsupported,
        EXT_EXPONENT_SPECIAL if significand << 1 == 0 => ExtClass::Infinity,
        EXT_EXPONENT_SPECIAL => ExtClass::Nan,
        _ if integer_bit_set => ExtClass::Normal,
        _ => ExtClass::Unsupported,
    }
}

/// Rounds to an integral value according to an FCW.RC field value (bits above 1:0 are ignored).
fn round_to_integral(v: f64, rounding_control: u16) -> f64 {
    match rounding_control & ROUNDING_CONTROL_MASK {
        ROUND_NEAREST_EVEN => v.round_ties_even(),
        ROUND_DOWN => v.floor(),
        ROUND_UP => v.ceil(),
        _ => v.trunc(),
    }
}

impl Cpu {
    /// #NM when CR0.EM or CR0.TS is set (the kernel's lazy FPU switching relies on the TS case).
    pub fn fpu_check(&self) -> ExecResult<()> {
        if self.cr0 & (CR0_EM | CR0_TS) != 0 {
            return Err(Exception::device_not_available());
        }
        Ok(())
    }

    fn top(&self) -> usize {
        ((self.fsw & FSW_TOP_MASK) >> FSW_TOP_SHIFT) as usize
    }

    fn set_top(&mut self, top: usize) {
        self.fsw = (self.fsw & !FSW_TOP_MASK) | ((top as u16 & STACK_MASK as u16) << FSW_TOP_SHIFT);
    }

    /// Raw contents of ST(i); `i` is relative to TOP.
    fn st_raw(&self, i: usize) -> [u8; 10] {
        self.st[(self.top() + i) & STACK_MASK]
    }

    fn st_val(&self, i: usize) -> f64 {
        ext_to_f64(self.st_raw(i))
    }

    /// Writes ST(i) and marks the physical register non-empty. `ftw` is indexed by physical
    /// register (like the FXSAVE abridged tag), not by stack position.
    fn set_st_raw(&mut self, i: usize, v: [u8; 10]) {
        let physical = (self.top() + i) & STACK_MASK;
        self.st[physical] = v;
        self.ftw |= 1 << physical;
    }

    fn set_st(&mut self, i: usize, v: f64) {
        self.set_st_raw(i, f64_to_ext(v));
    }

    fn fpush_raw(&mut self, v: [u8; 10]) {
        let new_top = self.top().wrapping_sub(1) & STACK_MASK;
        self.set_top(new_top);
        self.st[new_top] = v;
        self.ftw |= 1 << new_top;
    }

    fn fpush(&mut self, v: f64) {
        self.fpush_raw(f64_to_ext(v));
    }

    fn fpop(&mut self) {
        let top = self.top();
        self.ftw &= !(1 << top);
        self.set_top((top + 1) & STACK_MASK);
    }

    fn set_condition_codes(&mut self, c0: bool, c2: bool, c3: bool) {
        self.fsw = (self.fsw & !FSW_CONDITION_MASK)
            | (c0 as u16 * FSW_C0)
            | (c2 as u16 * FSW_C2)
            | (c3 as u16 * FSW_C3);
    }

    /// fcom/fucom: C3,C2,C0 = 111 unordered, 000 greater, 001 less, 100 equal. fucom is not
    /// distinguished from fcom (no invalid-operation signalling either way).
    fn fcom_val(&mut self, a: f64, b: f64) {
        if a.is_nan() || b.is_nan() {
            self.set_condition_codes(true, true, true);
        } else if a > b {
            self.set_condition_codes(false, false, false);
        } else if a < b {
            self.set_condition_codes(true, false, false);
        } else {
            self.set_condition_codes(false, false, true);
        }
    }

    /// fcomi/fucomi: same outcomes as fcom but reported in ZF, PF, CF.
    fn fcomi_val(&mut self, a: f64, b: f64) {
        let flags = if a.is_nan() || b.is_nan() {
            ZF | PF | CF
        } else if a > b {
            0
        } else if a < b {
            CF
        } else {
            ZF
        };
        let current = self.eflags_full();
        self.set_eflags_arith((current & !(ZF | PF | CF)) | flags);
    }

    /// Arithmetic selected by ModRM.reg of the D8/DC/DE/DA groups: 0 add, 1 mul, 4 sub, 5 subr,
    /// 6 div, 7 divr (2 and 3 are the compares, handled by callers).
    fn arith(reg: u8, a: f64, b: f64) -> f64 {
        match reg & 7 {
            0 => a + b,
            1 => a * b,
            4 => a - b,
            5 => b - a,
            6 => a / b,
            _ => b / a,
        }
    }

    /// Converts to a signed integer of `bits` width using FCW.RC. Out-of-range and NaN give the
    /// "integer indefinite" value (only the sign bit set) without signalling #IA.
    fn to_int(&self, v: f64, bits: u32) -> i64 {
        let rounded = round_to_integral(v, self.fcw >> FCW_ROUNDING_CONTROL_SHIFT);
        let limit = (1u128 << (bits - 1)) as f64;
        if rounded.is_nan() || rounded >= limit || rounded < -limit {
            i64::MIN >> (64 - bits)
        } else {
            rounded as i64
        }
    }

    /// Opcodes d8-df. Register forms (mod == 3) are selected by opcode and the whole ModRM
    /// byte or ModRM.reg; memory forms by opcode and ModRM.reg.
    pub fn x87(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        self.fpu_check()?;
        let op = (i.op & 0xff) as u8;
        let r = i.reg & 7;
        let sti = (i.rm & 7) as usize;
        if i.md == 3 {
            match op {
                0xd8 => {
                    // st0 = st0 op st(i); /2 fcom, /3 fcomp
                    if r == 2 || r == 3 {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcom_val(a, b);
                        if r == 3 {
                            self.fpop();
                        }
                    } else {
                        let v = Self::arith(r, self.st_val(0), self.st_val(sti));
                        self.set_st(0, v);
                    }
                }
                0xd9 => match i.modrm {
                    // fld st(i)
                    0xc0..=0xc7 => {
                        let v = self.st_raw(sti);
                        self.fpush_raw(v);
                    }
                    // fxch
                    0xc8..=0xcf => {
                        let (a, b) = (self.st_raw(0), self.st_raw(sti));
                        self.set_st_raw(0, b);
                        self.set_st_raw(sti, a);
                    }
                    // fnop
                    0xd0 => {}
                    // fchs
                    0xe0 => {
                        let mut v = self.st_raw(0);
                        v[EXT_SIGN_BYTE] ^= EXT_SIGN_BYTE_BIT;
                        self.set_st_raw(0, v);
                    }
                    // fabs
                    0xe1 => {
                        let mut v = self.st_raw(0);
                        v[EXT_SIGN_BYTE] &= !EXT_SIGN_BYTE_BIT;
                        self.set_st_raw(0, v);
                    }
                    // ftst
                    0xe4 => {
                        let a = self.st_val(0);
                        self.fcom_val(a, 0.0);
                    }
                    // fxam: C3,C2,C0 = 101 empty, 001 NaN, 010 normal, 011 infinity, 100 zero,
                    // 110 denormal, 000 unsupported format. C1 is the sign of ST(0), even when
                    // the register is empty.
                    0xe5 => {
                        let raw = self.st_raw(0);
                        let neg = raw[EXT_SIGN_BYTE] & EXT_SIGN_BYTE_BIT != 0;
                        let empty = self.ftw & (1 << self.top()) == 0;
                        let (c3, c2, c0) = if empty {
                            (true, false, true)
                        } else {
                            match classify(raw) {
                                ExtClass::Unsupported => (false, false, false),
                                ExtClass::Nan => (false, false, true),
                                ExtClass::Normal => (false, true, false),
                                ExtClass::Infinity => (false, true, true),
                                ExtClass::Zero => (true, false, false),
                                ExtClass::Denormal => (true, true, false),
                            }
                        };
                        self.set_condition_codes(c0, c2, c3);
                        self.fsw = (self.fsw & !FSW_C1) | (neg as u16 * FSW_C1);
                    }
                    // Constants are rounded to f64, not to 64 significand bits.
                    // fld1, fldl2t, fldl2e, fldpi, fldlg2, fldln2, fldz
                    0xe8 => self.fpush(1.0),
                    0xe9 => self.fpush(core::f64::consts::LOG2_10),
                    0xea => self.fpush(core::f64::consts::LOG2_E),
                    0xeb => self.fpush(core::f64::consts::PI),
                    0xec => self.fpush(core::f64::consts::LOG10_2),
                    0xed => self.fpush(core::f64::consts::LN_2),
                    0xee => self.fpush(0.0),
                    // f2xm1
                    0xf0 => {
                        let x = self.st_val(0);
                        self.set_st(0, (x * core::f64::consts::LN_2).exp_m1());
                    }
                    // fyl2x
                    0xf1 => {
                        let v = self.st_val(1) * self.st_val(0).log2();
                        self.set_st(1, v);
                        self.fpop();
                    }
                    // fptan: C2 = 0 reports the operand as in range
                    0xf2 => {
                        let x = self.st_val(0).tan();
                        self.set_st(0, x);
                        self.fpush(1.0);
                        self.set_condition_codes(false, false, false);
                    }
                    // fpatan
                    0xf3 => {
                        let v = self.st_val(1).atan2(self.st_val(0));
                        self.set_st(1, v);
                        self.fpop();
                    }
                    // fprem1 (quotient rounded to nearest) and fprem (truncated). The remainder is
                    // always computed in one step, so C2 stays clear; C0,C3,C1 = quotient bits 2,1,0.
                    0xf5 | 0xf8 => {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        let q = if i.modrm == 0xf8 {
                            (a / b).trunc()
                        } else {
                            (a / b).round_ties_even()
                        };
                        self.set_st(0, a - q * b);
                        let qi = q.abs() as u64;
                        self.set_condition_codes(qi & 4 != 0, false, qi & 2 != 0);
                        self.fsw = (self.fsw & !FSW_C1) | ((qi & 1) as u16 * FSW_C1);
                    }
                    // fdecstp, fincstp: move TOP only, tags are left alone
                    0xf6 => {
                        let top = self.top();
                        self.set_top(top.wrapping_sub(1) & STACK_MASK);
                    }
                    0xf7 => {
                        let top = self.top();
                        self.set_top((top + 1) & STACK_MASK);
                    }
                    // fyl2xp1
                    0xf9 => {
                        let v = self.st_val(1) * (1.0 + self.st_val(0)).log2();
                        self.set_st(1, v);
                        self.fpop();
                    }
                    // fsqrt
                    0xfa => {
                        let v = self.st_val(0).sqrt();
                        self.set_st(0, v);
                    }
                    // fsincos
                    0xfb => {
                        let x = self.st_val(0);
                        self.set_st(0, x.sin());
                        self.fpush(x.cos());
                        self.set_condition_codes(false, false, false);
                    }
                    // frndint
                    0xfc => {
                        let v = round_to_integral(
                            self.st_val(0),
                            self.fcw >> FCW_ROUNDING_CONTROL_SHIFT,
                        );
                        self.set_st(0, v);
                    }
                    // fscale
                    0xfd => {
                        let v = self.st_val(0) * 2f64.powf(self.st_val(1).trunc());
                        self.set_st(0, v);
                    }
                    // fsin, fcos
                    0xfe => {
                        let v = self.st_val(0).sin();
                        self.set_st(0, v);
                        self.set_condition_codes(false, false, false);
                    }
                    0xff => {
                        let v = self.st_val(0).cos();
                        self.set_st(0, v);
                        self.set_condition_codes(false, false, false);
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                0xda => match i.modrm {
                    // fcmovb, fcmove, fcmovbe, fcmovu
                    0xc0..=0xdf => {
                        let flags = self.eflags_full();
                        let condition = match r {
                            0 => flags & CF != 0,
                            1 => flags & ZF != 0,
                            2 => flags & (CF | ZF) != 0,
                            _ => flags & PF != 0,
                        };
                        if condition {
                            let v = self.st_raw(sti);
                            self.set_st_raw(0, v);
                        }
                    }
                    // fucompp
                    0xe9 => {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        self.fcom_val(a, b);
                        self.fpop();
                        self.fpop();
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                0xdb => match i.modrm {
                    // fcmovnb, fcmovne, fcmovnbe, fcmovnu
                    0xc0..=0xdf => {
                        let flags = self.eflags_full();
                        let condition = match r {
                            0 => flags & CF == 0,
                            1 => flags & ZF == 0,
                            2 => flags & (CF | ZF) == 0,
                            _ => flags & PF == 0,
                        };
                        if condition {
                            let v = self.st_raw(sti);
                            self.set_st_raw(0, v);
                        }
                    }
                    // fnclex
                    0xe2 => self.fsw &= FSW_KEPT_BY_FNCLEX,
                    // fninit
                    0xe3 => {
                        self.fcw = FCW_RESET;
                        self.fsw = 0;
                        self.ftw = 0;
                    }
                    // fucomi (e8-ef), fcomi (f0-f7)
                    0xe8..=0xf7 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcomi_val(a, b);
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                // st(i) = st(i) op st0. The sub/subr and div/divr encodings are swapped relative
                // to d8, which is why /4 computes b - a and /5 computes a - b below.
                0xdc => {
                    if r == 2 || r == 3 {
                        return Err(self.unimplemented_opcode(i));
                    }
                    let (a, b) = (self.st_val(sti), self.st_val(0));
                    let v = match r {
                        4 => b - a,
                        5 => a - b,
                        6 => b / a,
                        7 => a / b,
                        _ => Self::arith(r, a, b),
                    };
                    self.set_st(sti, v);
                }
                0xdd => match r {
                    // ffree st(i)
                    0 => self.ftw &= !(1 << ((self.top() + sti) & STACK_MASK)),
                    // fst/fstp st(i)
                    2 | 3 => {
                        let v = self.st_raw(0);
                        self.set_st_raw(sti, v);
                        if r == 3 {
                            self.fpop();
                        }
                    }
                    // fucom/fucomp
                    4 | 5 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcom_val(a, b);
                        if r == 5 {
                            self.fpop();
                        }
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                // faddp/fmulp/fsubrp/fsubp/fdivrp/fdivp (same swapped encodings as dc), fcompp
                0xde => {
                    if i.modrm == 0xd9 {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        self.fcom_val(a, b);
                        self.fpop();
                    } else if r == 2 || r == 3 {
                        return Err(self.unimplemented_opcode(i));
                    } else {
                        let (a, b) = (self.st_val(sti), self.st_val(0));
                        let v = match r {
                            4 => b - a,
                            5 => a - b,
                            6 => b / a,
                            7 => a / b,
                            _ => Self::arith(r, a, b),
                        };
                        self.set_st(sti, v);
                    }
                    self.fpop();
                }
                // df
                _ => match i.modrm {
                    // fnstsw ax
                    0xe0 => {
                        let v = self.fsw as u64;
                        self.write_gpr(0, 2, true, v);
                    }
                    // fucomip (e8-ef), fcomip (f0-f7)
                    0xe8..=0xf7 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcomi_val(a, b);
                        self.fpop();
                    }
                    // ffreep
                    0xc0..=0xc7 => {
                        self.ftw &= !(1 << ((self.top() + sti) & STACK_MASK));
                        self.fpop();
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
            }
            self.rip = next;
            return Ok(());
        }
        let a = match self.modrm_location(i, next) {
            Loc::Mem(a) => a,
            _ => unreachable!(),
        };
        match op {
            0xd8 | 0xdc => {
                let v = if op == 0xd8 {
                    f32::from_bits(self.read(a, 4)? as u32) as f64
                } else {
                    f64::from_bits(self.read(a, 8)?)
                };
                self.arith_mem(r, v);
            }
            0xda | 0xde => {
                let v = if op == 0xda {
                    self.read(a, 4)? as u32 as i32 as f64
                } else {
                    self.read(a, 2)? as u16 as i16 as f64
                };
                self.arith_mem(r, v);
            }
            0xd9 => match r {
                0 => {
                    let v = f32::from_bits(self.read(a, 4)? as u32) as f64;
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.st_val(0) as f32;
                    self.write(a, 4, v.to_bits() as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                4 => self.fldenv(a)?,
                // fldcw: no exception unmasking is modelled
                5 => self.fcw = self.read(a, 2)? as u16,
                6 => self.fnstenv(a)?,
                7 => self.write(a, 2, self.fcw as u64)?,
                // /1 is not an instruction
                _ => return Err(Exception::invalid_opcode()),
            },
            0xdb => match r {
                0 => {
                    let v = self.read(a, 4)? as u32 as i32 as f64;
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.to_int(self.st_val(0), 32) as u32;
                    self.write(a, 4, v as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                // fld m80 / fstp m80
                5 => {
                    let mut raw = [0u8; 10];
                    for (offset, byte) in raw.iter_mut().enumerate() {
                        *byte = self.read(a + offset as u64, 1)? as u8;
                    }
                    self.fpush_raw(raw);
                }
                7 => {
                    let raw = self.st_raw(0);
                    for (offset, byte) in raw.iter().enumerate() {
                        self.write(a + offset as u64, 1, *byte as u64)?;
                    }
                    self.fpop();
                }
                _ => return Err(self.unimplemented_opcode(i)),
            },
            0xdd => match r {
                0 => {
                    let v = f64::from_bits(self.read(a, 8)?);
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.st_val(0);
                    self.write(a, 8, v.to_bits())?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                4 => self.frstor(a)?,
                6 => self.fnsave(a)?,
                7 => self.write(a, 2, self.fsw as u64)?,
                _ => return Err(self.unimplemented_opcode(i)),
            },
            // df: fild/fist/fistp m16, fild/fistp m64 (bcd forms are not implemented)
            _ => match r {
                0 => {
                    let v = self.read(a, 2)? as u16 as i16 as f64;
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.to_int(self.st_val(0), 16) as u16;
                    self.write(a, 2, v as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                5 => {
                    let v = self.read(a, 8)? as i64 as f64;
                    self.fpush(v);
                }
                7 => {
                    let v = self.to_int(self.st_val(0), 64);
                    self.write(a, 8, v as u64)?;
                    self.fpop();
                }
                _ => return Err(self.unimplemented_opcode(i)),
            },
        }
        self.rip = next;
        Ok(())
    }

    /// Memory-operand forms of the d8/da/dc/de groups: st0 = st0 op v, or compare with v.
    fn arith_mem(&mut self, reg: u8, v: f64) {
        let a = self.st_val(0);
        if reg == 2 || reg == 3 {
            self.fcom_val(a, v);
            if reg == 3 {
                self.fpop();
            }
        } else {
            let res = Self::arith(reg, a, v);
            self.set_st(0, res);
        }
    }

    /// Builds the two-bit-per-register tag word for fnstenv from `ftw` and the register
    /// contents. Tags are indexed by physical register.
    fn full_tag(&self) -> u16 {
        let mut tag_word = 0u16;
        for physical in 0..STACK_DEPTH {
            let tag = if self.ftw & (1 << physical) == 0 {
                TAG_EMPTY
            } else {
                match classify(self.st[physical]) {
                    ExtClass::Normal => TAG_VALID,
                    ExtClass::Zero => TAG_ZERO,
                    ExtClass::Nan
                    | ExtClass::Infinity
                    | ExtClass::Denormal
                    | ExtClass::Unsupported => TAG_SPECIAL,
                }
            };
            tag_word |= tag << (2 * physical);
        }
        tag_word
    }

    fn fnstenv(&mut self, a: u64) -> ExecResult<()> {
        self.write(a, 4, self.fcw as u64)?;
        self.write(a + ENV_FSW_OFFSET, 4, self.fsw as u64)?;
        let tag_word = self.full_tag();
        self.write(a + ENV_FTW_OFFSET, 4, tag_word as u64)?;
        for word in ENV_POINTER_FIRST_WORD..ENV_POINTER_END_WORD {
            self.write(a + word * 4, 4, 0)?;
        }
        // fnstenv masks all exceptions after saving the environment
        self.fcw |= FCW_EXCEPTION_MASKS;
        Ok(())
    }

    /// Only a 2-bit "empty or not" distinction survives into `ftw`.
    fn fldenv(&mut self, a: u64) -> ExecResult<()> {
        self.fcw = self.read(a, 2)? as u16;
        self.fsw = self.read(a + ENV_FSW_OFFSET, 2)? as u16;
        let tag_word = self.read(a + ENV_FTW_OFFSET, 2)? as u16;
        self.ftw = (0..STACK_DEPTH)
            .filter(|physical| (tag_word >> (2 * physical)) & 3 != TAG_EMPTY)
            .fold(0, |mask, physical| mask | (1 << physical));
        Ok(())
    }

    fn fnsave(&mut self, a: u64) -> ExecResult<()> {
        self.fnstenv(a)?;
        for stack_pos in 0..STACK_DEPTH {
            let raw = self.st_raw(stack_pos);
            for (offset, byte) in raw.iter().enumerate() {
                let at = a + ENV_SIZE + stack_pos as u64 * EXT_REG_SIZE + offset as u64;
                self.write(at, 1, *byte as u64)?;
            }
        }
        // fnsave reinitialises the FPU, same state as fninit
        self.fcw = FCW_RESET;
        self.fsw = 0;
        self.ftw = 0;
        Ok(())
    }

    fn frstor(&mut self, a: u64) -> ExecResult<()> {
        self.fldenv(a)?;
        for stack_pos in 0..STACK_DEPTH {
            let mut raw = [0u8; 10];
            for (offset, byte) in raw.iter_mut().enumerate() {
                let at = a + ENV_SIZE + stack_pos as u64 * EXT_REG_SIZE + offset as u64;
                *byte = self.read(at, 1)? as u8;
            }
            let physical = (self.top() + stack_pos) & STACK_MASK;
            self.st[physical] = raw;
        }
        Ok(())
    }

    /// #GP(0) unless the 512-byte area is 16-byte aligned, checked on the linear address before
    /// anything is read or written.
    fn check_fxsave_alignment(&self, a: u64) -> ExecResult<()> {
        if a & (FXSAVE_ALIGNMENT - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    pub fn fxsave(&mut self, a: u64) -> ExecResult<()> {
        if self.cr0 & CR0_EM != 0 {
            return Err(Exception::invalid_opcode());
        }
        self.fpu_check()?;
        self.check_fxsave_alignment(a)?;
        // Probe both ends first so a page fault is raised before any byte is written.
        self.translate(a, mem::AccessKind::Write, false)?;
        self.translate(a + FXSAVE_AREA_SIZE - 1, mem::AccessKind::Write, false)?;
        self.write(a, 2, self.fcw as u64)?;
        self.write(a + FXSAVE_FSW_OFFSET, 2, self.fsw as u64)?;
        self.write(a + FXSAVE_FTW_OFFSET, 1, self.ftw as u64 & 0xff)?;
        self.write(a + FXSAVE_MXCSR_OFFSET, 4, self.mxcsr as u64)?;
        self.write(a + FXSAVE_MXCSR_MASK_OFFSET, 4, MXCSR_VALID_BITS as u64)?;
        for stack_pos in 0..STACK_DEPTH {
            let raw = self.st_raw(stack_pos);
            for (offset, byte) in raw.iter().enumerate() {
                let at = a + FXSAVE_ST_OFFSET + stack_pos as u64 * FXSAVE_ST_STRIDE + offset as u64;
                self.write(at, 1, *byte as u64)?;
            }
        }
        for k in 0..self.xmm.len() {
            self.write128(
                a + FXSAVE_XMM_OFFSET + k as u64 * FXSAVE_XMM_STRIDE,
                self.xmm[k],
            )?;
        }
        Ok(())
    }

    /// Reads and validates the whole image before changing any register, so a fault or a
    /// reserved MXCSR bit leaves the x87 and SSE state exactly as it was.
    pub fn fxrstor(&mut self, a: u64) -> ExecResult<()> {
        if self.cr0 & CR0_EM != 0 {
            return Err(Exception::invalid_opcode());
        }
        self.fpu_check()?;
        self.check_fxsave_alignment(a)?;
        let fcw = self.read(a, 2)? as u16;
        let fsw = self.read(a + FXSAVE_FSW_OFFSET, 2)? as u16;
        let ftw = self.read(a + FXSAVE_FTW_OFFSET, 1)? as u16;
        let mxcsr = self.read(a + FXSAVE_MXCSR_OFFSET, 4)? as u32;
        let mut st = [[0u8; 10]; STACK_DEPTH];
        let top = ((fsw & FSW_TOP_MASK) >> FSW_TOP_SHIFT) as usize;
        // the image stores registers in stack order; place them by physical register
        for stack_pos in 0..STACK_DEPTH {
            for (offset, byte) in st[(top + stack_pos) & STACK_MASK].iter_mut().enumerate() {
                let at = a + FXSAVE_ST_OFFSET + stack_pos as u64 * FXSAVE_ST_STRIDE + offset as u64;
                *byte = self.read(at, 1)? as u8;
            }
        }
        let mut xmm = [0u128; 16];
        for (k, register) in xmm.iter_mut().enumerate() {
            *register = self.read128(a + FXSAVE_XMM_OFFSET + k as u64 * FXSAVE_XMM_STRIDE)?;
        }
        if mxcsr & !MXCSR_VALID_BITS != 0 {
            return Err(Exception::general_protection(0));
        }
        self.st = st;
        self.xmm = xmm;
        self.fcw = fcw;
        self.fsw = fsw;
        self.ftw = ftw;
        self.mxcsr = mxcsr;
        Ok(())
    }
}

#[cfg(test)]
#[path = "fpu_tests.rs"]
mod fpu_tests;
