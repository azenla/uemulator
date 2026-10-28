//! EVEX permutes, blends, widening and narrowing moves, compress/expand and the conversions
//! between opmasks and vectors.

use super::*;

const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;

/// Opcode groups of the pmovsx/pmovzx/pmov down-convert family: the low three bits pick the
/// element-size pair, bits 5:4 the saturation (1 = unsigned saturating, 2 = signed saturating,
/// 3 = truncating for the down-converts; 2 = sign extending, 3 = zero extending for the widening
/// ones).
const CONVERT_UNSIGNED_SATURATE: u8 = 1;
const CONVERT_SIGNED_SATURATE: u8 = 2;
const CONVERT_PLAIN: u8 = 3;

/// (source element bytes, destination element bytes) for the six pairs bw, bd, bq, wd, wq, dq.
const WIDEN_PAIRS: [(usize, usize); 6] = [(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 8)];

fn saturate(value: i128, bytes: usize, signed_result: bool) -> u64 {
    let bits = bytes as u32 * 8;
    let (low, high) = if signed_result {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    };
    value.clamp(low, high) as u64 & size_mask(bytes)
}

impl Cpu {
    pub(super) fn exec_evex_permute(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        let size = if wide { 8 } else { 4 };
        match (map, opcode, prefix) {
            // vpermd/vpermq and vpermps/vpermpd: index vector in vvvv, table in r/m
            (OPCODE_MAP_0F38, 0x36 | 0x16, P66) => {
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size).crossing(),
                    |_, _, index, table, bytes| {
                        let count = bytes / size;
                        Ok(build(bytes, size, |k| {
                            element(table, size, element(index, size, k) as usize & (count - 1))
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F38, 0x8d, P66) => {
                let size = if wide { 2 } else { 1 };
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size).crossing(),
                    |_, _, index, table, bytes| {
                        let count = bytes / size;
                        Ok(build(bytes, size, |k| {
                            element(table, size, element(index, size, k) as usize & (count - 1))
                        }))
                    },
                )?;
            }
            // vpermq/vpermpd by immediate: within each 256-bit half
            (OPCODE_MAP_0F3A, 0x00 | 0x01, P66) if wide => {
                let imm = i.imm as usize;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(8).crossing().unary(),
                    |_, _, _, source, bytes| {
                        Ok(build(bytes, 8, |k| {
                            element(source, 8, k / 4 * 4 + (imm >> (2 * (k % 4)) & 3))
                        }))
                    },
                )?;
            }
            // vpermi2*: the destination holds the indices; vvvv and r/m are the two tables
            (OPCODE_MAP_0F38, 0x75..=0x77, P66) => {
                let size = match (opcode, wide) {
                    (0x75, false) => 1,
                    (0x75, true) => 2,
                    (_, false) => 4,
                    (_, true) => 8,
                };
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size).crossing(),
                    |_, index, first, second, bytes| {
                        let count = bytes / size;
                        Ok(build(bytes, size, |k| {
                            let selector = element(index, size, k) as usize;
                            let table = if selector & count == 0 { first } else { second };
                            element(table, size, selector & (count - 1))
                        }))
                    },
                )?;
            }
            // vpermt2*: the destination is the first table, vvvv the indices, r/m the second table
            (OPCODE_MAP_0F38, 0x7d..=0x7f, P66) => {
                let size = match (opcode, wide) {
                    (0x7d, false) => 1,
                    (0x7d, true) => 2,
                    (_, false) => 4,
                    (_, true) => 8,
                };
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size).crossing(),
                    |_, first, index, second, bytes| {
                        let count = bytes / size;
                        Ok(build(bytes, size, |k| {
                            let selector = element(index, size, k) as usize;
                            let table = if selector & count == 0 { first } else { second };
                            element(table, size, selector & (count - 1))
                        }))
                    },
                )?;
            }
            // vpermilps/vpermilpd with a variable control (0C, 0D) or an immediate (3A 04, 05)
            (OPCODE_MAP_0F38, 0x0c | 0x0d, P66) => {
                let size = if opcode == 0x0c { 4 } else { 8 };
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size),
                    |_, _, source, control, bytes| {
                        let per_lane = LANE_BYTES / size;
                        Ok(build(bytes, size, |k| {
                            let control = element(control, size, k);
                            // vpermilpd uses bit 1 of each control element, vpermilps bits 1:0
                            let pick = if size == 4 {
                                control & 3
                            } else {
                                control >> 1 & 1
                            } as usize;
                            element(source, size, k / per_lane * per_lane + pick)
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F3A, 0x04 | 0x05, P66) => {
                let size = if opcode == 0x04 { 4 } else { 8 };
                let imm = i.imm as usize;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size).unary(),
                    |_, _, _, source, bytes| {
                        let per_lane = LANE_BYTES / size;
                        Ok(build(bytes, size, |k| {
                            let pick = if size == 4 {
                                imm >> (2 * (k % 4)) & 3
                            } else {
                                imm >> k & 1
                            };
                            element(source, size, k / per_lane * per_lane + pick)
                        }))
                    },
                )?;
            }
            // vpblendm*/vblendmp*: masked select between vvvv and r/m; merging keeps vvvv
            (OPCODE_MAP_0F38, 0x64..=0x66, P66) => {
                let size = match (opcode, wide) {
                    (0x66, false) => 1,
                    (0x66, true) => 2,
                    (_, false) => 4,
                    (_, true) => 8,
                };
                self.blend_with_mask(i, next, size)?;
            }
            (OPCODE_MAP_0F38, 0x20..=0x25 | 0x30..=0x35, P66) => {
                let (source, destination) = WIDEN_PAIRS[(opcode & 7) as usize];
                let sign_extend = opcode & 0x10 == 0;
                let mut shape = OpShape::elementwise(destination).unary();
                shape.rm_element = Some(source);
                shape.rm_bytes = Some(destination / source);
                shape.broadcast = false;
                self.exec_vector_op(i, next, shape, |_, _, _, value, bytes| {
                    Ok(build(bytes, destination, |k| {
                        let v = element(value, source, k);
                        if sign_extend {
                            signed(v, source) as u64
                        } else {
                            v
                        }
                    }))
                })?;
            }
            (OPCODE_MAP_0F38, 0x10..=0x15 | 0x20..=0x25 | 0x30..=0x35, F3) => {
                self.narrow(i, next, opcode)?;
            }
            (OPCODE_MAP_0F38, 0x28 | 0x38, F3) => {
                // vpmovm2b/w and vpmovm2d/q: every element is all ones or all zeros
                let size = match (opcode, wide) {
                    (0x28, false) => 1,
                    (0x28, true) => 2,
                    (_, false) => 4,
                    (_, true) => 8,
                };
                if i.vex_v != 0 || i.md != 3 || i.evex_mask != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let bytes = self.vector_bytes(i, Rounding::None)?;
                let bits = self.opmask[(i.rm & 7) as usize];
                let value = build(bytes, size, |k| {
                    if bits >> k & 1 == 1 {
                        size_mask(size)
                    } else {
                        0
                    }
                });
                self.set_zmm(i.reg as usize, value);
            }
            (OPCODE_MAP_0F38, 0x29 | 0x39, F3) => {
                // vpmovb2m/w2m and vpmovd2m/q2m: the top bit of every element
                let size = match (opcode, wide) {
                    (0x29, false) => 1,
                    (0x29, true) => 2,
                    (_, false) => 4,
                    (_, true) => 8,
                };
                if i.vex_v != 0 || i.md != 3 || i.evex_mask != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let bytes = self.vector_bytes(i, Rounding::None)?;
                let source = self.zmm(i.rm as usize);
                let mut bits = 0u64;
                for k in 0..bytes / size {
                    bits |= (element(&source, size, k) >> (size * 8 - 1) & 1) << k;
                }
                self.opmask[(i.reg & 7) as usize] = bits;
            }
            // vpbroadcastmb2q / vpbroadcastmw2d (AVX512CD): the low byte / word of an opmask
            // register repeated over every qword / dword; no write mask
            (OPCODE_MAP_0F38, 0x2a | 0x3a, F3) if wide == (opcode == 0x2a) => {
                if i.vex_v != 0 || i.md != 3 || i.evex_mask != 0 || i.evex_zeroing {
                    return Err(Exception::invalid_opcode());
                }
                let bytes = self.vector_bytes(i, Rounding::None)?;
                let (size, bits) = if opcode == 0x2a {
                    (8, 0xff)
                } else {
                    (4, 0xffff)
                };
                let value = self.opmask[(i.rm & 7) as usize] & bits;
                self.set_zmm(i.reg as usize, build(bytes, size, |_| value));
            }
            (OPCODE_MAP_0F38, 0x88 | 0x89 | 0x62, P66) => self.expand(i, next, opcode, wide)?,
            (OPCODE_MAP_0F38, 0x8a | 0x8b | 0x63, P66) => self.compress(i, next, opcode, wide)?,
            _ => return self.exec_evex_misc(i, next, map, opcode, prefix, wide),
        }
        Ok(true)
    }

    /// vpblendm*: result element = mask ? r/m : vvvv. Without a mask the whole result is r/m.
    fn blend_with_mask(&mut self, i: &Insn, next: u64, size: usize) -> ExecResult<()> {
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let mask = self.write_mask(i);
        let shape = RmShape {
            suppress: (i.evex_mask != 0).then_some(mask),
            broadcast: size >= 4,
            ..RmShape::full(bytes, size)
        };
        let second = self.fetch_rm(i, next, shape)?;
        let first = self.zmm(i.vex_v as usize);
        let result = build(bytes, size, |k| {
            if mask >> k & 1 == 1 {
                element(&second, size, k)
            } else if i.evex_zeroing {
                0
            } else {
                element(&first, size, k)
            }
        });
        self.set_zmm(i.reg as usize, result);
        Ok(())
    }

    /// vpmov{,s,us}{wb,db,qb,dw,qw,qd}: narrows every element of ModRM.reg into r/m.
    fn narrow(&mut self, i: &Insn, next: u64, opcode: u8) -> ExecResult<()> {
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        // the table lists (small, large) pairs for bw, bd, bq, wd, wq, dq
        let (destination, source) = WIDEN_PAIRS[(opcode & 7) as usize];
        let kind = opcode >> 4;
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let count = bytes / source;
        let result_bytes = count * destination;
        let value = self.zmm(i.reg as usize);
        let narrowed = build(result_bytes, destination, |k| {
            let v = element(&value, source, k);
            match kind {
                CONVERT_UNSIGNED_SATURATE => saturate(v as i128, destination, false),
                CONVERT_SIGNED_SATURATE => saturate(signed(v, source) as i128, destination, true),
                _ => v & size_mask(destination),
            }
        });
        debug_assert!(matches!(kind, CONVERT_UNSIGNED_SATURATE..=CONVERT_PLAIN));
        if i.md == 3 {
            self.commit_masked(i, i.rm as usize, destination, result_bytes, &narrowed);
            return Ok(());
        }
        if i.evex_zeroing {
            return Err(Exception::invalid_opcode());
        }
        let address = self.evex_address(i, next, result_bytes)?;
        let mask = self.write_mask(i);
        self.store_vector(address, &narrowed, result_bytes, destination, mask)
    }

    /// vcompressps/pd, vpcompressd/q and vpcompressb/w: the selected elements of ModRM.reg packed
    /// into the low end of r/m.
    fn compress(&mut self, i: &Insn, next: u64, opcode: u8, wide: bool) -> ExecResult<()> {
        let size = match (opcode, wide) {
            (0x63, false) => 1,
            (0x63, true) => 2,
            (_, false) => 4,
            (_, true) => 8,
        };
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let mask = self.write_mask(i);
        let source = self.zmm(i.reg as usize);
        let mut packed = [0u128; LANES];
        let mut count = 0;
        for k in 0..bytes / size {
            if mask >> k & 1 == 1 {
                set_element(&mut packed, size, count, element(&source, size, k));
                count += 1;
            }
        }
        if i.md == 3 {
            let old = self.zmm(i.rm as usize);
            let result = build(bytes, size, |k| {
                if k < count {
                    element(&packed, size, k)
                } else if i.evex_zeroing {
                    0
                } else {
                    element(&old, size, k)
                }
            });
            self.set_zmm(i.rm as usize, result);
            return Ok(());
        }
        if i.evex_zeroing {
            return Err(Exception::invalid_opcode());
        }
        let address = self.evex_address(i, next, size)?;
        self.store_vector(address, &packed, count * size, size, u64::MAX)
    }

    /// vexpandps/pd, vpexpandd/q and vpexpandb/w: consecutive elements of r/m spread to the
    /// positions the mask selects.
    fn expand(&mut self, i: &Insn, next: u64, opcode: u8, wide: bool) -> ExecResult<()> {
        let size = match (opcode, wide) {
            (0x62, false) => 1,
            (0x62, true) => 2,
            (_, false) => 4,
            (_, true) => 8,
        };
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let elements = bytes / size;
        let mask = self.write_mask(i);
        let selected = if elements >= 64 {
            mask
        } else {
            mask & ((1 << elements) - 1)
        };
        let count = selected.count_ones() as usize;
        let packed = if i.md == 3 {
            self.zmm(i.rm as usize)
        } else {
            let address = self.evex_address(i, next, size)?;
            self.load_vector(address, count * size, size, None)?
        };
        let old = self.zmm(i.reg as usize);
        let mut taken = 0;
        let result = build(bytes, size, |k| {
            if selected >> k & 1 == 1 {
                taken += 1;
                element(&packed, size, taken - 1)
            } else if i.evex_zeroing {
                0
            } else {
                element(&old, size, k)
            }
        });
        // The merge/zero was handled above, so the write must not mask again.
        self.set_zmm(i.reg as usize, result);
        Ok(())
    }
}
