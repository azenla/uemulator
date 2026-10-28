//! The opmask instructions (VEX-encoded, AVX-512 state): kmov, the bitwise operations, kadd,
//! kunpck, kshift, kortest and ktest. The operand width comes from pp and VEX.W:
//! b (8 bits) is 66/W0, w (16) is none/W0, d (32) is 66/W1 and q (64) is none/W1.

use super::*;
use crate::cpu::flags::{CF, ZF};

const OPMASK_COUNT: usize = 8;
const OPCODE_MAP_0F: u16 = 1;
const OPCODE_MAP_0F3A: u16 = 3;
const OP_KAND: u8 = 0x41;
const OP_KANDN: u8 = 0x42;
const OP_KNOT: u8 = 0x44;
const OP_KOR: u8 = 0x45;
const OP_KXNOR: u8 = 0x46;
const OP_KXOR: u8 = 0x47;
const OP_KADD: u8 = 0x4a;
const OP_KUNPCK: u8 = 0x4b;
const OP_KMOV_LOAD: u8 = 0x90;
const OP_KMOV_STORE: u8 = 0x91;
const OP_KMOV_FROM_GENERAL: u8 = 0x92;
const OP_KMOV_TO_GENERAL: u8 = 0x93;
const OP_KORTEST: u8 = 0x98;
const OP_KTEST: u8 = 0x99;
const OP_KSHIFTR_BYTE_WORD: u8 = 0x30;
const OP_KSHIFTR_DWORD_QWORD: u8 = 0x31;
const OP_KSHIFTL_BYTE_WORD: u8 = 0x32;
const OP_KSHIFTL_DWORD_QWORD: u8 = 0x33;

/// Operand width in bits for a k instruction, from the implied prefix and VEX.W.
fn width_bits(prefix: Prefix, wide: bool) -> Option<u32> {
    match (prefix, wide) {
        (Prefix::P66, false) => Some(8),
        (Prefix::Np, false) => Some(16),
        (Prefix::P66, true) => Some(32),
        (Prefix::Np, true) => Some(64),
        _ => None,
    }
}

fn width_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

impl Cpu {
    /// Handles the VEX-encoded opmask instructions; false if `i` is not one of them.
    pub(in crate::cpu) fn exec_opmask(&mut self, i: &Insn, next: u64) -> ExecResult<bool> {
        let op = i.op & !VEX_OP_FLAG;
        let (map, opcode) = (op >> 8, (op & 0xff) as u8);
        let prefix = prefix_of(i);
        let wide = i.opsize == 8;
        let is_kmov_family = matches!(
            opcode,
            OP_KMOV_LOAD | OP_KMOV_STORE | OP_KMOV_FROM_GENERAL | OP_KMOV_TO_GENERAL
        );
        let is_logic = matches!(
            opcode,
            OP_KAND | OP_KANDN | OP_KOR | OP_KXNOR | OP_KXOR | OP_KADD | OP_KUNPCK
        );
        let is_unary = matches!(opcode, OP_KNOT | OP_KORTEST | OP_KTEST);
        let is_shift = map == OPCODE_MAP_0F3A
            && matches!(
                opcode,
                OP_KSHIFTR_BYTE_WORD
                    | OP_KSHIFTR_DWORD_QWORD
                    | OP_KSHIFTL_BYTE_WORD
                    | OP_KSHIFTL_DWORD_QWORD
            );
        if !(map == OPCODE_MAP_0F && (is_kmov_family || is_logic || is_unary) || is_shift) {
            return Ok(false);
        }
        self.evex_gate()?;
        let reg = (i.reg & 7) as usize;
        let rm = (i.rm & 7) as usize;
        let vvvv = (i.vex_v & 7) as usize;
        debug_assert!(reg < OPMASK_COUNT);
        if is_kmov_family {
            self.exec_kmov(i, next, opcode, prefix, wide)?;
        } else if is_shift {
            if prefix != Prefix::P66 || i.vex_l || i.vex_v != 0 || i.md != 3 {
                return Err(Exception::invalid_opcode());
            }
            let bits = match (opcode, wide) {
                (OP_KSHIFTR_BYTE_WORD | OP_KSHIFTL_BYTE_WORD, false) => 8,
                (OP_KSHIFTR_BYTE_WORD | OP_KSHIFTL_BYTE_WORD, true) => 16,
                (_, false) => 32,
                (_, true) => 64,
            };
            let count = i.imm as u8 as u32;
            let left = matches!(opcode, OP_KSHIFTL_BYTE_WORD | OP_KSHIFTL_DWORD_QWORD);
            let source = self.opmask[rm] & width_mask(bits);
            let shifted = match (count < bits, left) {
                (false, _) => 0,
                (true, true) => source << count,
                (true, false) => source >> count,
            };
            self.opmask[reg] = shifted & width_mask(bits);
        } else if is_unary {
            let Some(bits) = width_bits(prefix, wide) else {
                return Err(Exception::invalid_opcode());
            };
            if i.vex_l || i.md != 3 || (opcode == OP_KNOT && i.vex_v != 0) {
                return Err(Exception::invalid_opcode());
            }
            let full = width_mask(bits);
            let source = self.opmask[rm] & full;
            match opcode {
                OP_KNOT => self.opmask[reg] = !source & full,
                OP_KORTEST => {
                    let combined = (self.opmask[reg] | source) & full;
                    let flags =
                        if combined == 0 { ZF } else { 0 } | if combined == full { CF } else { 0 };
                    self.set_eflags_arith(flags);
                }
                _ => {
                    let first = self.opmask[reg] & full;
                    let flags = if first & source == 0 { ZF } else { 0 }
                        | if !first & source & full == 0 { CF } else { 0 };
                    self.set_eflags_arith(flags);
                }
            }
        } else {
            // Binary: dest = reg, sources = vvvv and rm, all registers, VEX.L = 1.
            if !i.vex_l || i.md != 3 {
                return Err(Exception::invalid_opcode());
            }
            let (a, b) = (self.opmask[vvvv], self.opmask[rm]);
            let result = if opcode == OP_KUNPCK {
                // kunpckbw (66/W0), kunpckwd (none/W0), kunpckdq (none/W1): the first source
                // becomes the high half, the second the low half.
                let half = match (prefix, wide) {
                    (Prefix::P66, false) => 8,
                    (Prefix::Np, false) => 16,
                    (Prefix::Np, true) => 32,
                    _ => return Err(Exception::invalid_opcode()),
                };
                (a & width_mask(half)) << half | b & width_mask(half)
            } else {
                let Some(bits) = width_bits(prefix, wide) else {
                    return Err(Exception::invalid_opcode());
                };
                let full = width_mask(bits);
                (match opcode {
                    OP_KAND => a & b,
                    OP_KANDN => !a & b,
                    OP_KOR => a | b,
                    OP_KXNOR => !(a ^ b),
                    OP_KXOR => a ^ b,
                    _ => a.wrapping_add(b),
                }) & full
            };
            self.opmask[reg] = result;
        }
        self.rip = next;
        Ok(true)
    }

    fn exec_kmov(
        &mut self,
        i: &Insn,
        next: u64,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<()> {
        if i.vex_l || i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let reg = (i.reg & 7) as usize;
        let rm = (i.rm & 7) as usize;
        match opcode {
            OP_KMOV_LOAD | OP_KMOV_STORE => {
                let Some(bits) = width_bits(prefix, wide) else {
                    return Err(Exception::invalid_opcode());
                };
                let bytes = (bits / 8) as u8;
                if opcode == OP_KMOV_LOAD {
                    self.opmask[reg] = if i.md == 3 {
                        self.opmask[rm] & width_mask(bits)
                    } else {
                        let address = self.vex_address(i, next)?;
                        self.read(address, bytes)?
                    };
                } else {
                    if i.md == 3 {
                        return Err(Exception::invalid_opcode());
                    }
                    let address = self.vex_address(i, next)?;
                    self.write(address, bytes, self.opmask[reg])?;
                }
            }
            _ => {
                if i.md != 3 {
                    return Err(Exception::invalid_opcode());
                }
                // 92/93 select the width by a different prefix set: none = w, 66 = b, F2 = d/q.
                let bits = match (prefix, wide) {
                    (Prefix::Np, false) => 16,
                    (Prefix::P66, false) => 8,
                    (Prefix::F2, false) => 32,
                    (Prefix::F2, true) => 64,
                    _ => return Err(Exception::invalid_opcode()),
                };
                if opcode == OP_KMOV_FROM_GENERAL {
                    let value = self.read_gpr(i.rm & 0xf, if bits == 64 { 8 } else { 4 }, true);
                    self.opmask[reg] = value & width_mask(bits);
                } else {
                    let value = self.opmask[rm] & width_mask(bits);
                    self.write_gpr(i.reg & 0xf, if bits == 64 { 8 } else { 4 }, true, value);
                }
            }
        }
        Ok(())
    }
}
