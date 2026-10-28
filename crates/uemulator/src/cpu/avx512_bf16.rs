//! AVX512_BF16: vcvtneps2bf16, vcvtne2ps2bf16 and vdpbf16ps (Zen 4 has them; the guest kernels and
//! libraries that use them are the ML ones, but a CPUID bit that is set promises all three).
//!
//! The SDM defines these without MXCSR: rounding is always to nearest even, denormal inputs read as
//! zero and denormal results flush to zero whatever MXCSR.DAZ/FTZ say, and no exception flag is
//! raised or consulted. So each instruction runs on a private `FpEnv` whose flags are dropped
//! instead of committed.

use super::*;

const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_CVTNEPS2BF16_AND_CVTNE2PS2BF16: u8 = 0x72;
const OPCODE_DPBF16PS: u8 = 0x52;

const SINGLE_BYTES: usize = 4;
const BFLOAT16_BYTES: usize = 2;
/// A bfloat16 is the top half of the binary32 with the same sign, exponent and leading fraction.
const BFLOAT16_SHIFT: u32 = 16;
const BFLOAT16_QUIET_BIT: u16 = 0x0040;
const SINGLE_EXPONENT_SHIFT: u32 = 23;
const SINGLE_EXPONENT_MASK: u32 = 0xff;
const SINGLE_FRACTION_MASK: u32 = (1 << SINGLE_EXPONENT_SHIFT) - 1;
const SINGLE_SIGN_IN_BFLOAT16: u16 = 0x8000;
/// Adding this to the bits of a binary32 (plus the lowest kept bit) and truncating rounds to
/// nearest even; the carry out of the fraction bumps the exponent, which gives infinity for the
/// largest finite values.
const ROUND_TO_NEAREST_EVEN_BIAS: u32 = 0x7fff;
/// MXCSR as these instructions behave: nearest-even rounding, DAZ, FTZ and underflow masked.
const FIXED_ENVIRONMENT: u32 = MXCSR_DAZ | MXCSR_FTZ | MXCSR_UNDERFLOW_MASK;

/// SDM `convert_fp32_to_bfloat16`: NaNs keep their top payload bits and become quiet, denormals
/// become a signed zero, everything else rounds to nearest even.
fn single_to_bfloat16(bits: u32) -> u16 {
    let exponent = bits >> SINGLE_EXPONENT_SHIFT & SINGLE_EXPONENT_MASK;
    let fraction = bits & SINGLE_FRACTION_MASK;
    let high_half = (bits >> BFLOAT16_SHIFT) as u16;
    if exponent == SINGLE_EXPONENT_MASK && fraction != 0 {
        return high_half | BFLOAT16_QUIET_BIT;
    }
    if exponent == 0 {
        return high_half & SINGLE_SIGN_IN_BFLOAT16;
    }
    let lowest_kept_bit = bits >> BFLOAT16_SHIFT & 1;
    (bits.wrapping_add(ROUND_TO_NEAREST_EVEN_BIAS + lowest_kept_bit) >> BFLOAT16_SHIFT) as u16
}

/// One accumulator dword of vdpbf16ps: the odd bfloat16 pair is multiplied and added first, then
/// the even pair, each as a separate binary32 operation (so the intermediate sum rounds).
fn dot_product_accumulate(accumulator: u32, first: u32, second: u32) -> u32 {
    let mut env = FpEnv::new(FIXED_ENVIRONMENT);
    let widen =
        |pair: u32, half: u32| ((pair >> (BFLOAT16_SHIFT * half)) & 0xffff) << BFLOAT16_SHIFT;
    let mut sum = accumulator as u64;
    for half in [1, 0] {
        let product = arithmetic(
            &mut env,
            SINGLE,
            Arithmetic::Multiply,
            widen(first, half) as u64,
            widen(second, half) as u64,
        );
        sum = arithmetic(&mut env, SINGLE, Arithmetic::Add, sum, product);
    }
    sum as u32
}

impl Cpu {
    pub(super) fn exec_evex_bfloat16(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        match (map, opcode, prefix) {
            // vcvtne2ps2bf16: the low half of the result comes from the r/m source, the high half
            // from vvvv. The mask applies per bfloat16 while the broadcast element is a dword.
            (OPCODE_MAP_0F38, OPCODE_CVTNEPS2BF16_AND_CVTNE2PS2BF16, F2) if !wide => {
                let mut shape = OpShape::elementwise(BFLOAT16_BYTES).crossing();
                shape.rm_element = Some(SINGLE_BYTES);
                self.exec_vector_op(i, next, shape, |_, _, first, second, bytes| {
                    let half = bytes / SINGLE_BYTES;
                    Ok(build(bytes, BFLOAT16_BYTES, |k| {
                        let (source, index) = if k < half {
                            (second, k)
                        } else {
                            (first, k - half)
                        };
                        single_to_bfloat16(element(source, SINGLE_BYTES, index) as u32) as u64
                    }))
                })?;
            }
            // vcvtneps2bf16: half-width result; the same machinery as vcvtpd2ps
            (OPCODE_MAP_0F38, OPCODE_CVTNEPS2BF16_AND_CVTNE2PS2BF16, F3) if !wide => {
                self.convert_packed(
                    i,
                    next,
                    SINGLE_BYTES,
                    BFLOAT16_BYTES,
                    Rounding::None,
                    |env, bits| {
                        env.flags = 0;
                        single_to_bfloat16(bits as u32) as u64
                    },
                )?;
            }
            (OPCODE_MAP_0F38, OPCODE_DPBF16PS, F3) if !wide => {
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(SINGLE_BYTES),
                    |_, old, first, second, bytes| {
                        Ok(build(bytes, SINGLE_BYTES, |k| {
                            dot_product_accumulate(
                                element(old, SINGLE_BYTES, k) as u32,
                                element(first, SINGLE_BYTES, k) as u32,
                                element(second, SINGLE_BYTES, k) as u32,
                            ) as u64
                        }))
                    },
                )?;
            }
            _ => return self.exec_evex_convert(i, next, map, opcode, prefix, wide),
        }
        Ok(true)
    }
}
