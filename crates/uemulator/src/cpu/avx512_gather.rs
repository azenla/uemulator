//! EVEX gathers and scatters. The index vector comes from the VSIB register (`Insn::index`, up to
//! 31), the write mask is an opmask register that is consumed: each completed element clears its
//! bit, so a faulting instruction can be restarted where it stopped.

use super::*;
use crate::cpu::decode::NO_REG;

const OPCODE_MAP_0F38: u16 = 2;
const GATHER_FIRST: u8 = 0x90;
const GATHER_LAST: u8 = 0x93;
const SCATTER_FIRST: u8 = 0xa0;
const SCATTER_LAST: u8 = 0xa3;

impl Cpu {
    /// vpgather*/vgather* (90-93) and vpscatter*/vscatter* (a0-a3). Returns false for any other
    /// opcode.
    pub(super) fn exec_evex_gather(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        wide: bool,
    ) -> ExecResult<bool> {
        let is_gather = (GATHER_FIRST..=GATHER_LAST).contains(&opcode);
        let is_scatter = (SCATTER_FIRST..=SCATTER_LAST).contains(&opcode);
        if map != OPCODE_MAP_0F38 || !(is_gather || is_scatter) {
            return Ok(false);
        }
        // 90/92 and a0/a2 use dword indices, 91/93 and a1/a3 qword indices
        let index_bytes = if opcode & 1 == 0 { 4 } else { 8 };
        let data_bytes = if wide { 8 } else { 4 };
        let register = i.reg as usize;
        let index_register = i.index as usize;
        // A gather needs memory, a VSIB index and a real mask; its destination must not be the
        // index register. Scatters have no destination register but the same other rules.
        if i.md == 3
            || i.index == NO_REG
            || i.evex_mask == 0
            || i.evex_zeroing
            || (is_gather && register == index_register)
        {
            return Err(Exception::invalid_opcode());
        }
        let length = self.vector_bytes(i, Rounding::None)?;
        let count = length / data_bytes.max(index_bytes);
        let indices = self.zmm(index_register);
        let mut scaled = *i;
        if i.evex_disp8 {
            scaled.disp = i.disp.wrapping_mul(data_bytes as i64);
        }
        let base = if i.rip_rel {
            next
        } else if i.base != NO_REG {
            self.gpr[i.base as usize]
        } else {
            0
        }
        .wrapping_add(scaled.disp as u64);
        let mask_register = i.evex_mask as usize;
        let mut mask = self.opmask[mask_register];
        let mut data = self.zmm(register);
        let source = self.zmm(register);
        let mut failure = None;
        for k in 0..count {
            if mask >> k & 1 == 0 {
                continue;
            }
            let raw = element(&indices, index_bytes, k);
            let index = signed(raw, index_bytes);
            let mut address = base.wrapping_add((index << i.scale) as u64);
            if i.addr32 {
                address &= 0xffff_ffff;
            }
            let linear = self.linear_address(i, address)?;
            let outcome = if is_gather {
                self.read(linear, data_bytes as u8).map(|value| {
                    set_element(&mut data, data_bytes, k, value);
                })
            } else {
                self.write(linear, data_bytes as u8, element(&source, data_bytes, k))
            };
            match outcome {
                Ok(()) => mask &= !(1 << k),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        if failure.is_none() {
            mask = 0;
        }
        self.opmask[mask_register] = mask;
        if is_gather {
            if failure.is_none() {
                // the destination holds `count` elements; everything above is zeroed
                let kept = build(count * data_bytes, data_bytes, |k| {
                    element(&data, data_bytes, k)
                });
                self.set_zmm(register, kept);
            } else {
                self.set_zmm(register, data);
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(true),
        }
    }
}
