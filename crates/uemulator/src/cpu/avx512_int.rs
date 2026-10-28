//! EVEX integer instructions that are not an SSE kernel on four lanes: 64-bit and variable shifts,
//! rotates, counts, compares into opmasks, ternary logic, permutes, blends, widening and
//! narrowing moves, compress/expand, conflict detection and the AVX-512 extensions (VBMI, VBMI2,
//! IFMA, VNNI, BITALG, VPOPCNTDQ, GFNI).

use super::*;

const OPCODE_MAP_0F: u16 = 1;
const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;

/// Integer-compare predicates of vpcmp[u]* (imm8 bits 2:0).
const PREDICATE_EQUAL: u8 = 0;
const PREDICATE_LESS: u8 = 1;
const PREDICATE_LESS_EQUAL: u8 = 2;
const PREDICATE_FALSE: u8 = 3;
const PREDICATE_NOT_EQUAL: u8 = 4;
const PREDICATE_NOT_LESS: u8 = 5;
const PREDICATE_NOT_LESS_EQUAL: u8 = 6;
const PREDICATE_MASK: i64 = 7;

/// Shift-by-immediate groups (opcodes 71, 72, 73): ModRM.reg selects the operation, vvvv is the
/// destination.
const GROUP_SHIFT_WORD: u8 = 0x71;
const GROUP_SHIFT_DWORD: u8 = 0x72;
const GROUP_SHIFT_QWORD: u8 = 0x73;
const SHIFT_RIGHT_LOGICAL: u8 = 2;
const SHIFT_RIGHT_BYTES: u8 = 3;
const SHIFT_RIGHT_ARITHMETIC: u8 = 4;
const SHIFT_LEFT_LOGICAL: u8 = 6;
const SHIFT_LEFT_BYTES: u8 = 7;
const ROTATE_RIGHT: u8 = 0;
const ROTATE_LEFT: u8 = 1;

type Binary = fn(u64, u64) -> u64;

/// The element-wise integer operations: (element bytes, unary, function). `a` is the vvvv
/// element, `b` the r/m element (the only one for unary operations).
fn elementwise(map: u16, opcode: u8, prefix: Prefix, wide: bool) -> Option<(usize, bool, Binary)> {
    use Prefix::P66;
    if prefix != P66 {
        return None;
    }
    let size = if wide { 8 } else { 4 };
    // A shift count of a variable shift is the whole element: counts at or above the width
    // shift everything out (or fill with the sign for the arithmetic one).
    let shift_left: Binary = if wide {
        |a, b| if b >= 64 { 0 } else { a << b }
    } else {
        |a, b| if b >= 32 { 0 } else { (a << b) & 0xffff_ffff }
    };
    let shift_right: Binary = if wide {
        |a, b| if b >= 64 { 0 } else { a >> b }
    } else {
        |a, b| if b >= 32 { 0 } else { a >> b }
    };
    let shift_arithmetic: Binary = if wide {
        |a, b| ((a as i64) >> b.min(63)) as u64
    } else {
        |a, b| (((a as u32 as i32) >> b.min(31)) as u32) as u64
    };
    let rotate_left: Binary = if wide {
        |a, b| a.rotate_left((b & 63) as u32)
    } else {
        |a, b| (a as u32).rotate_left((b & 31) as u32) as u64
    };
    let rotate_right: Binary = if wide {
        |a, b| a.rotate_right((b & 63) as u32)
    } else {
        |a, b| (a as u32).rotate_right((b & 31) as u32) as u64
    };
    Some(match (map, opcode) {
        (OPCODE_MAP_0F38, 0x39) if wide => (8, false, |a, b| (a as i64).min(b as i64) as u64),
        (OPCODE_MAP_0F38, 0x3b) if wide => (8, false, |a, b| a.min(b)),
        (OPCODE_MAP_0F38, 0x3d) if wide => (8, false, |a, b| (a as i64).max(b as i64) as u64),
        (OPCODE_MAP_0F38, 0x3f) if wide => (8, false, |a, b| a.max(b)),
        (OPCODE_MAP_0F38, 0x40) if wide => (8, false, |a, b| a.wrapping_mul(b)),
        (OPCODE_MAP_0F38, 0x1f) if wide => (8, true, |_, b| (b as i64).unsigned_abs()),
        (OPCODE_MAP_0F38, 0x45) => (size, false, shift_right),
        (OPCODE_MAP_0F38, 0x46) => (size, false, shift_arithmetic),
        (OPCODE_MAP_0F38, 0x47) => (size, false, shift_left),
        (OPCODE_MAP_0F38, 0x10) if wide => (2, false, |a, b| if b >= 16 { 0 } else { a >> b }),
        (OPCODE_MAP_0F38, 0x11) if wide => (2, false, |a, b| {
            (((a as u16 as i16) >> b.min(15)) as u16) as u64
        }),
        (OPCODE_MAP_0F38, 0x12) if wide => {
            (2, false, |a, b| if b >= 16 { 0 } else { (a << b) & 0xffff })
        }
        (OPCODE_MAP_0F38, 0x14) => (size, false, rotate_right),
        (OPCODE_MAP_0F38, 0x15) => (size, false, rotate_left),
        (OPCODE_MAP_0F38, 0x44) if wide => (8, true, |_, b| b.leading_zeros() as u64),
        (OPCODE_MAP_0F38, 0x44) => (4, true, |_, b| (b as u32).leading_zeros() as u64),
        (OPCODE_MAP_0F38, 0x54) if !wide => (1, true, |_, b| (b as u8).count_ones() as u64),
        (OPCODE_MAP_0F38, 0x54) => (2, true, |_, b| (b as u16).count_ones() as u64),
        (OPCODE_MAP_0F38, 0x55) if wide => (8, true, |_, b| b.count_ones() as u64),
        (OPCODE_MAP_0F38, 0x55) => (4, true, |_, b| (b as u32).count_ones() as u64),
        _ => return None,
    })
}

/// Shift and rotate by a register count that is not an SSE kernel: arithmetic quadword shift.
fn shift_by_count_quadword_arithmetic(count: u128, value: u64) -> u64 {
    ((value as i64) >> (count as u64).min(63)) as u64
}

/// The integer comparison `predicate` of `a` and `b`, element size `size`, signed or not.
fn integer_predicate(predicate: u8, a: u64, b: u64, size: usize, signed_compare: bool) -> bool {
    let (less, equal) = if signed_compare {
        (
            signed(a, size) < signed(b, size),
            signed(a, size) == signed(b, size),
        )
    } else {
        let mask = size_mask(size);
        ((a & mask) < (b & mask), a & mask == b & mask)
    };
    match predicate {
        PREDICATE_EQUAL => equal,
        PREDICATE_LESS => less,
        PREDICATE_LESS_EQUAL => less || equal,
        PREDICATE_FALSE => false,
        PREDICATE_NOT_EQUAL => !equal,
        PREDICATE_NOT_LESS => !less,
        PREDICATE_NOT_LESS_EQUAL => !(less || equal),
        _ => true,
    }
}

impl Cpu {
    pub(super) fn exec_evex_int(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        if let Some((element, unary, f)) = elementwise(map, opcode, prefix, wide) {
            let mut shape = OpShape::elementwise(element);
            shape.unary = unary;
            self.exec_vector_op(i, next, shape, |_, _, a, b, bytes| {
                Ok(build(bytes, element, |k| {
                    f(super::element(a, element, k), super::element(b, element, k))
                }))
            })?;
            return Ok(true);
        }
        match (map, opcode, prefix) {
            // vpsraq by a register count: the one 64-bit arithmetic shift the SSE kernels lack
            (OPCODE_MAP_0F, 0xe2, P66) if wide => {
                let count = self.fetch_count(i, next)?;
                let shape = OpShape::elementwise(8);
                self.exec_vector_op_with(i, next, shape, |_, _, a, _, bytes| {
                    Ok(build(bytes, 8, |k| {
                        shift_by_count_quadword_arithmetic(count, super::element(a, 8, k))
                    }))
                })?;
            }
            (OPCODE_MAP_0F, GROUP_SHIFT_WORD..=GROUP_SHIFT_QWORD, P66) => {
                self.shift_by_immediate(i, next, opcode, wide)?;
            }
            (OPCODE_MAP_0F, 0x74..=0x76 | 0x64..=0x66, P66) => {
                // b, w, d in opcode order; 74-76 compare for equality, 64-66 for signed greater
                let element = 1 << (opcode & 3);
                let equal = opcode >= 0x74;
                self.compare_to_mask(i, next, element, move |a, b| {
                    if equal {
                        integer_predicate(PREDICATE_EQUAL, a, b, element, true)
                    } else {
                        integer_predicate(PREDICATE_LESS, b, a, element, true)
                    }
                })?;
            }
            (OPCODE_MAP_0F38, 0x29 | 0x37, P66) if wide => {
                let equal = opcode == 0x29;
                self.compare_to_mask(i, next, 8, move |a, b| {
                    if equal {
                        a == b
                    } else {
                        (a as i64) > (b as i64)
                    }
                })?;
            }
            (OPCODE_MAP_0F3A, 0x1e | 0x1f | 0x3e | 0x3f, P66) => {
                let element = match (opcode & 0x20 != 0, wide) {
                    (true, false) => 1,
                    (true, true) => 2,
                    (false, false) => 4,
                    (false, true) => 8,
                };
                let signed_compare = opcode & 1 == 1;
                let predicate = (i.imm & PREDICATE_MASK) as u8;
                self.compare_to_mask(i, next, element, move |a, b| {
                    integer_predicate(predicate, a, b, element, signed_compare)
                })?;
            }
            (OPCODE_MAP_0F38, 0x26 | 0x27, P66 | F3) => {
                let element = match (opcode, wide) {
                    (0x26, false) => 1,
                    (0x26, true) => 2,
                    (_, false) => 4,
                    (_, true) => 8,
                };
                let nonzero_wanted = prefix == P66;
                self.compare_to_mask(i, next, element, move |a, b| {
                    (a & b & size_mask(element) != 0) == nonzero_wanted
                })?;
            }
            (OPCODE_MAP_0F3A, 0x25, P66) => {
                let element = if wide { 8 } else { 4 };
                let table = i.imm as u8;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(element),
                    |_, old, a, b, bytes| {
                        // result bit = table[(A << 2) | (B << 1) | C] with A the old destination,
                        // B the vvvv operand and C the r/m operand
                        let mut out = [0u128; LANES];
                        for lane in 0..bytes / LANE_BYTES {
                            for pattern in 0..8u32 {
                                if table >> pattern & 1 == 0 {
                                    continue;
                                }
                                let select = |bit: u32, value: u128| {
                                    if pattern >> bit & 1 == 1 {
                                        value
                                    } else {
                                        !value
                                    }
                                };
                                out[lane] |=
                                    select(2, old[lane]) & select(1, a[lane]) & select(0, b[lane]);
                            }
                        }
                        Ok(out)
                    },
                )?;
            }
            (OPCODE_MAP_0F3A, 0x03, P66) => {
                let element = if wide { 8 } else { 4 };
                let imm = i.imm as usize;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(element).crossing(),
                    |_, _, high, low, bytes| {
                        let count = bytes / element;
                        // (high:low) shifted right by imm elements
                        Ok(build(bytes, element, |k| {
                            let at = k + (imm & (count - 1));
                            if at < count {
                                super::element(low, element, at)
                            } else {
                                super::element(high, element, at - count)
                            }
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F3A, 0x23 | 0x43, P66) => {
                let element = if wide { 8 } else { 4 };
                let imm = i.imm as usize;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(element).crossing(),
                    |_, _, a, b, bytes| {
                        let mut out = [0u128; LANES];
                        if bytes == VECTOR_BYTES {
                            out[0] = a[imm & 3];
                            out[1] = a[imm >> 2 & 3];
                            out[2] = b[imm >> 4 & 3];
                            out[3] = b[imm >> 6 & 3];
                        } else {
                            out[0] = a[imm & 1];
                            out[1] = b[imm >> 1 & 1];
                        }
                        Ok(out)
                    },
                )?;
            }
            _ => return self.exec_evex_permute(i, next, map, opcode, prefix, wide),
        }
        Ok(true)
    }

    /// The 128-bit shift count operand of a shift: low 128 bits of a register or 16 bytes.
    fn fetch_count(&mut self, i: &Insn, next: u64) -> ExecResult<u128> {
        let shape = RmShape {
            bytes: LANE_BYTES,
            element: LANE_BYTES,
            broadcast: false,
            suppress: None,
        };
        Ok(self.fetch_rm(i, next, shape)?[0])
    }

    /// Like `exec_vector_op` for instructions whose r/m operand was already consumed (the count):
    /// the source is vvvv, the destination ModRM.reg.
    fn exec_vector_op_with(
        &mut self,
        i: &Insn,
        _next: u64,
        shape: OpShape,
        body: impl FnOnce(&mut Cpu, &Vector, &Vector, &Vector, usize) -> ExecResult<Vector>,
    ) -> ExecResult<()> {
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let first = self.zmm(i.vex_v as usize);
        let old = self.zmm(i.reg as usize);
        let result = body(self, &old, &first, &[0; LANES], bytes)?;
        self.commit_masked(i, i.reg as usize, shape.element, bytes, &result);
        Ok(())
    }

    /// vpsrlw/vpsraw/... with an immediate count: the destination is vvvv, the source r/m.
    fn shift_by_immediate(&mut self, i: &Insn, next: u64, group: u8, wide: bool) -> ExecResult<()> {
        let operation = i.reg & 7;
        let count = i.imm as u8 as u64;
        let (element, byte_shift) = match (group, operation) {
            (
                GROUP_SHIFT_WORD,
                SHIFT_RIGHT_LOGICAL | SHIFT_RIGHT_ARITHMETIC | SHIFT_LEFT_LOGICAL,
            ) => (2, false),
            (GROUP_SHIFT_DWORD, ROTATE_RIGHT | ROTATE_LEFT) => (if wide { 8 } else { 4 }, false),
            (
                GROUP_SHIFT_DWORD,
                SHIFT_RIGHT_LOGICAL | SHIFT_RIGHT_ARITHMETIC | SHIFT_LEFT_LOGICAL,
            ) => (if wide { 8 } else { 4 }, false),
            (GROUP_SHIFT_QWORD, SHIFT_RIGHT_LOGICAL | SHIFT_LEFT_LOGICAL) => (8, false),
            (GROUP_SHIFT_QWORD, SHIFT_RIGHT_BYTES | SHIFT_LEFT_BYTES) => (1, true),
            _ => return Err(self.unimplemented_opcode(i)),
        };
        let bits = element as u64 * 8;
        let wide_element = element == 8;
        let apply = move |value: u64| -> u64 {
            let value = value & size_mask(element);
            match (group, operation) {
                (_, ROTATE_RIGHT) => {
                    if wide_element {
                        value.rotate_right((count & 63) as u32)
                    } else {
                        (value as u32).rotate_right((count & 31) as u32) as u64
                    }
                }
                (_, ROTATE_LEFT) => {
                    if wide_element {
                        value.rotate_left((count & 63) as u32)
                    } else {
                        (value as u32).rotate_left((count & 31) as u32) as u64
                    }
                }
                (_, SHIFT_RIGHT_LOGICAL) => {
                    if count >= bits {
                        0
                    } else {
                        value >> count
                    }
                }
                (_, SHIFT_LEFT_LOGICAL) => {
                    if count >= bits {
                        0
                    } else {
                        (value << count) & size_mask(element)
                    }
                }
                _ => (signed(value, element) >> count.min(bits - 1)) as u64,
            }
        };
        // The source is the r/m operand and vvvv is the destination, so swap the operand roles.
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let mask = self.write_mask(i);
        if byte_shift {
            // vpslldq/vpsrldq: per 128-bit lane byte shifts, never masked
            let source = self.fetch_rm(
                i,
                next,
                RmShape {
                    broadcast: false,
                    suppress: None,
                    ..RmShape::full(bytes, 1)
                },
            )?;
            let mut out = [0u128; LANES];
            for lane in 0..bytes / LANE_BYTES {
                out[lane] = if count > 15 {
                    0
                } else if operation == SHIFT_LEFT_BYTES {
                    source[lane] << (count * 8)
                } else {
                    source[lane] >> (count * 8)
                };
            }
            self.set_zmm(i.vex_v as usize, out);
            return Ok(());
        }
        let shape_rm = RmShape {
            suppress: (i.evex_mask != 0).then_some(mask),
            broadcast: element >= 4,
            ..RmShape::full(bytes, element)
        };
        let source = self.fetch_rm(i, next, shape_rm)?;
        let result = build(bytes, element, |k| {
            apply(super::element(&source, element, k))
        });
        self.commit_masked(i, i.vex_v as usize, element, bytes, &result);
        Ok(())
    }

    /// Compares element-wise and writes one bit per element to the opmask register ModRM.reg,
    /// ANDed with the write mask (a masked-out element compares as 0). Bits beyond the vector
    /// length are cleared.
    pub(in crate::cpu) fn compare_to_mask(
        &mut self,
        i: &Insn,
        next: u64,
        element: usize,
        mut predicate: impl FnMut(u64, u64) -> bool,
    ) -> ExecResult<()> {
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let mask = self.write_mask(i);
        let shape = RmShape {
            suppress: (i.evex_mask != 0).then_some(mask),
            broadcast: element >= 4,
            ..RmShape::full(bytes, element)
        };
        let second = self.fetch_rm(i, next, shape)?;
        let first = self.zmm(i.vex_v as usize);
        let mut bits = 0u64;
        for k in 0..bytes / element {
            if mask >> k & 1 == 1
                && predicate(
                    super::element(&first, element, k),
                    super::element(&second, element, k),
                )
            {
                bits |= 1 << k;
            }
        }
        self.opmask[(i.reg & 7) as usize] = bits;
        Ok(())
    }
}

#[path = "avx512_perm.rs"]
mod perm;
