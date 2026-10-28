//! The AVX-512 extensions built on integer vectors: conflict detection (CD), IFMA, VNNI, BITALG,
//! VBMI, VBMI2 and GFNI. The floating-point instructions are in `avx512_fp.rs`.

use super::*;

const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;
/// IFMA operates on the low 52 bits of each quadword.
const IFMA_BITS: u32 = 52;
const IFMA_MASK: u64 = (1 << IFMA_BITS) - 1;
/// The GF(2^8) reduction polynomial x^8 + x^4 + x^3 + x + 1 (without the x^8 term).
const GF_POLYNOMIAL: u8 = 0x1b;
/// Order of the multiplicative group of GF(2^8): a^254 is the inverse of a.
const GF_INVERSE_EXPONENT: u32 = 254;

pub(in crate::cpu) fn gf_multiply(a: u8, b: u8) -> u8 {
    let (mut a, mut b, mut product) = (a, b, 0u8);
    for _ in 0..8 {
        if b & 1 == 1 {
            product ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= GF_POLYNOMIAL;
        }
        b >>= 1;
    }
    product
}

/// Multiplicative inverse in GF(2^8); 0 maps to 0.
fn gf_inverse(a: u8) -> u8 {
    let (mut result, mut power) = (1u8, a);
    let mut exponent = GF_INVERSE_EXPONENT;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = gf_multiply(result, power);
        }
        power = gf_multiply(power, power);
        exponent >>= 1;
    }
    if a == 0 { 0 } else { result }
}

/// One byte of vgf2p8affineqb: bit `n` of the result is the parity of (matrix byte 7-n AND x),
/// XOR bit `n` of the immediate.
fn affine_byte(matrix: u64, x: u8, immediate: u8) -> u8 {
    let mut result = 0u8;
    for n in 0..8 {
        let row = (matrix >> (8 * (7 - n))) as u8;
        let bit = (row & x).count_ones() as u8 & 1;
        result |= (bit ^ (immediate >> n & 1)) << n;
    }
    result
}

impl Cpu {
    pub(super) fn exec_evex_misc(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        if prefix != P66 {
            return self.exec_evex_fp(i, next, map, opcode, prefix, wide);
        }
        let size = if wide { 8 } else { 4 };
        match (map, opcode) {
            (OPCODE_MAP_0F38, 0xc4) => {
                // vpconflictd/q: for each element, the earlier elements equal to it as a bit set
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size).unary().crossing(),
                    |_, _, _, source, bytes| {
                        Ok(build(bytes, size, |k| {
                            let value = element(source, size, k);
                            (0..k)
                                .filter(|&j| element(source, size, j) == value)
                                .fold(0, |bits, j| bits | 1 << j)
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F38, 0xb4 | 0xb5) if wide => {
                let high = opcode == 0xb5;
                self.exec_vector_op(i, next, OpShape::elementwise(8), |_, old, a, b, bytes| {
                    Ok(build(bytes, 8, |k| {
                        let product = (element(a, 8, k) & IFMA_MASK) as u128
                            * (element(b, 8, k) & IFMA_MASK) as u128;
                        let part =
                            if high { product >> IFMA_BITS } else { product } as u64 & IFMA_MASK;
                        element(old, 8, k).wrapping_add(part)
                    }))
                })?;
            }
            // vpdpbusd(s) and vpdpwssd(s): dot products accumulated into the destination
            (OPCODE_MAP_0F38, 0x50..=0x53) if !wide => {
                let saturating = opcode & 1 == 1;
                let words = opcode >= 0x52;
                self.exec_vector_op(i, next, OpShape::elementwise(4), |_, old, a, b, bytes| {
                    Ok(build(bytes, 4, |k| {
                        let sum: i64 = if words {
                            (0..2)
                                .map(|j| {
                                    signed(element(a, 2, 2 * k + j), 2)
                                        * signed(element(b, 2, 2 * k + j), 2)
                                })
                                .sum()
                        } else {
                            (0..4)
                                .map(|j| {
                                    element(a, 1, 4 * k + j) as i64
                                        * signed(element(b, 1, 4 * k + j), 1)
                                })
                                .sum()
                        };
                        let total = signed(element(old, 4, k), 4) + sum;
                        if saturating {
                            total.clamp(i32::MIN as i64, i32::MAX as i64) as u64
                        } else {
                            total as u64
                        }
                    }))
                })?;
            }
            (OPCODE_MAP_0F38, 0x8f) if !wide => self.shuffle_bits_to_mask(i, next)?,
            (OPCODE_MAP_0F38, 0x83) if wide => {
                let mut shape = OpShape::elementwise(1).crossing();
                shape.rm_element = Some(8);
                self.exec_vector_op(i, next, shape, |_, _, control, data, bytes| {
                    Ok(build(bytes, 1, |k| {
                        let quadword = element(data, 8, k / 8);
                        let start = element(control, 1, k) as u32 & 63;
                        quadword.rotate_right(start) & 0xff
                    }))
                })?;
            }
            // vpshldv/vpshrdv: concatenate-and-shift by per-element counts (VBMI2)
            (OPCODE_MAP_0F38, 0x70..=0x73) => {
                let element_bytes = if opcode & 1 == 0 { 2 } else { size };
                if opcode & 1 == 0 && !wide {
                    return Err(Exception::invalid_opcode());
                }
                let left = opcode < 0x72;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(element_bytes),
                    |_, old, counts, other, bytes| {
                        Ok(build(bytes, element_bytes, |k| {
                            let bits = element_bytes as u32 * 8;
                            let n = element(counts, element_bytes, k) as u32 % bits;
                            funnel(
                                left,
                                element(old, element_bytes, k),
                                element(other, element_bytes, k),
                                n,
                                bits,
                            )
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F3A, 0x70..=0x73) => {
                let element_bytes = if opcode & 1 == 0 { 2 } else { size };
                if opcode & 1 == 0 && !wide {
                    return Err(Exception::invalid_opcode());
                }
                let left = opcode < 0x72;
                let imm = i.imm as u32;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(element_bytes),
                    |_, _, a, b, bytes| {
                        Ok(build(bytes, element_bytes, |k| {
                            let bits = element_bytes as u32 * 8;
                            funnel(
                                left,
                                element(a, element_bytes, k),
                                element(b, element_bytes, k),
                                imm % bits,
                                bits,
                            )
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F38, 0xcf) if !wide => {
                let mut shape = OpShape::elementwise(1);
                shape.broadcast = false;
                self.exec_vector_op(i, next, shape, |_, _, a, b, bytes| {
                    Ok(build(bytes, 1, |k| {
                        gf_multiply(element(a, 1, k) as u8, element(b, 1, k) as u8) as u64
                    }))
                })?;
            }
            (OPCODE_MAP_0F3A, 0xce | 0xcf) if wide => {
                let inverse = opcode == 0xcf;
                let immediate = i.imm as u8;
                let mut shape = OpShape::elementwise(1);
                shape.rm_element = Some(8);
                shape.suppress = false;
                self.exec_vector_op(i, next, shape, |_, _, data, matrix, bytes| {
                    Ok(build(bytes, 1, |k| {
                        let x = element(data, 1, k) as u8;
                        let x = if inverse { gf_inverse(x) } else { x };
                        affine_byte(element(matrix, 8, k / 8), x, immediate) as u64
                    }))
                })?;
            }
            _ => return self.exec_evex_fp(i, next, map, opcode, prefix, wide),
        }
        Ok(true)
    }

    /// vpshufbitqmb: bit `k` of the result is bit `index[k] & 63` of the quadword of vvvv that byte
    /// `k` belongs to.
    fn shuffle_bits_to_mask(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let mask = self.write_mask(i);
        let shape = RmShape {
            suppress: (i.evex_mask != 0).then_some(mask),
            broadcast: false,
            ..RmShape::full(bytes, 1)
        };
        let indices = self.fetch_rm(i, next, shape)?;
        let data = self.zmm(i.vex_v as usize);
        let mut bits = 0u64;
        for k in 0..bytes {
            let bit = element(&data, 8, k / 8) >> (element(&indices, 1, k) & 63) & 1;
            if mask >> k & 1 == 1 {
                bits |= bit << k;
            }
        }
        self.opmask[(i.reg & 7) as usize] = bits;
        Ok(())
    }
}

/// One element of a double-width shift: `left` is vpshld (the high half of (a:b) << n), the
/// other vpshrd (the low half of (b:a) >> n).
fn funnel(left: bool, a: u64, b: u64, n: u32, bits: u32) -> u64 {
    if n == 0 {
        return a;
    }
    let (a, b) = (
        a & size_mask(bits as usize / 8),
        b & size_mask(bits as usize / 8),
    );
    let value = if left {
        a << n | b >> (bits - n)
    } else {
        a >> n | b << (bits - n)
    };
    value & size_mask(bits as usize / 8)
}
