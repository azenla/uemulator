//! EVEX data movement: full-vector and scalar moves, vmovd/vmovq, non-temporal stores.

use super::*;

const OPCODE_MAP_0F: u16 = 1;

fn alignment_fault<T>() -> ExecResult<T> {
    Err(Exception::general_protection(0))
}

impl Cpu {
    pub(super) fn exec_evex_data(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        if map != OPCODE_MAP_0F {
            return self.exec_evex_shuffle_data(i, next, map, opcode, prefix, wide);
        }
        let packed_element = if wide { 8 } else { 4 };
        match (opcode, prefix) {
            (0x10 | 0x28, Np | P66) => {
                let element = if prefix == Np { 4 } else { 8 };
                self.vector_load(i, next, element, opcode == 0x28)?;
            }
            (0x11 | 0x29, Np | P66) => {
                let element = if prefix == Np { 4 } else { 8 };
                self.vector_store(i, next, element, opcode == 0x29, true)?;
            }
            (0x6f, P66) => self.vector_load(i, next, packed_element, true)?,
            (0x6f, F3) => self.vector_load(i, next, packed_element, false)?,
            (0x6f, F2) => self.vector_load(i, next, if wide { 2 } else { 1 }, false)?,
            (0x7f, P66) => self.vector_store(i, next, packed_element, true, true)?,
            (0x7f, F3) => self.vector_store(i, next, packed_element, false, true)?,
            (0x7f, F2) => self.vector_store(i, next, if wide { 2 } else { 1 }, false, true)?,
            // Non-temporal stores: aligned, memory only, never masked.
            (0x2b, Np | P66) | (0xe7, P66) => {
                if i.md == 3 || i.evex_mask != 0 || i.evex_zeroing {
                    return Err(Exception::invalid_opcode());
                }
                let element = if opcode == 0x2b && prefix == Np { 4 } else { 8 };
                self.vector_store(i, next, element, true, false)?;
            }
            (0x10, F3 | F2) => {
                let element = if prefix == F3 { 4 } else { 8 };
                self.scalar_move_load(i, next, element)?;
            }
            (0x11, F3 | F2) => {
                let element = if prefix == F3 { 4 } else { 8 };
                self.scalar_move_store(i, next, element)?;
            }
            (0x6e, P66) => self.move_from_general(i, next, wide)?,
            (0x7e, P66) => self.move_to_general(i, next, wide)?,
            (0x7e, F3) => self.move_quad_load(i, next)?,
            (0xd6, P66) => self.move_quad_store(i, next)?,
            (0x12 | 0x13 | 0x16 | 0x17, Np | P66) => self.move_half_lanes(i, next, opcode)?,
            (0xc4, P66) => self.insert_general(i, next, 2, i.imm as usize & 7)?,
            (0xc5, P66) => self.extract_word_to_register(i)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(super) fn check_move_alignment(&self, i: &Insn, next: u64, bytes: usize) -> ExecResult<()> {
        if i.md != 3 {
            let address = self.evex_address(i, next, bytes);
            if address & (bytes as u64 - 1) != 0 {
                return alignment_fault();
            }
        }
        Ok(())
    }

    /// vmovups/vmovdqu*/vmovdqa* loads (reg <- reg/mem) with merge or zero masking.
    fn vector_load(
        &mut self,
        i: &Insn,
        next: u64,
        element: usize,
        aligned: bool,
    ) -> ExecResult<()> {
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        if aligned {
            self.check_move_alignment(i, next, bytes)?;
        }
        let mask = self.write_mask(i);
        let shape = RmShape {
            bytes,
            element,
            broadcast: false,
            suppress: (i.evex_mask != 0).then_some(mask),
        };
        let value = self.fetch_rm(i, next, shape)?;
        self.commit_masked(i, i.reg as usize, element, bytes, &value);
        Ok(())
    }

    /// Stores (rm <- reg). A register destination merges under the mask like any other write;
    /// memory writes only the selected elements (and EVEX.z is #UD there).
    fn vector_store(
        &mut self,
        i: &Insn,
        next: u64,
        element: usize,
        aligned: bool,
        maskable: bool,
    ) -> ExecResult<()> {
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let value = self.zmm(i.reg as usize);
        if i.md == 3 {
            self.commit_masked(i, i.rm as usize, element, bytes, &value);
            return Ok(());
        }
        if i.evex_zeroing && maskable {
            return Err(Exception::invalid_opcode());
        }
        if aligned {
            self.check_move_alignment(i, next, bytes)?;
        }
        let address = self.evex_address(i, next, bytes);
        let mask = self.write_mask(i);
        self.store_vector(address, &value, bytes, element, mask)
    }

    /// vmovss/vmovsd: the register form merges the upper elements from vvvv, the memory form
    /// loads one element and zeroes the rest.
    fn scalar_move_load(&mut self, i: &Insn, next: u64, element: usize) -> ExecResult<()> {
        let elements_mask = size_mask(element) as u128;
        let result = if i.md == 3 {
            let low = self.zmm(i.rm as usize)[0] & elements_mask;
            self.zmm(i.vex_v as usize)[0] & !elements_mask | low
        } else {
            if i.vex_v != 0 {
                return Err(Exception::invalid_opcode());
            }
            // The load itself is suppressed when the mask bit is clear.

            if i.evex_mask == 0 || self.opmask[i.evex_mask as usize] & 1 == 1 {
                let address = self.evex_address(i, next, element);
                self.read(address, element as u8)? as u128
            } else {
                0
            }
        };
        self.commit_scalar(i, i.reg as usize, element, result);
        Ok(())
    }

    fn scalar_move_store(&mut self, i: &Insn, next: u64, element: usize) -> ExecResult<()> {
        let value = self.zmm(i.reg as usize)[0];
        if i.md == 3 {
            let elements_mask = size_mask(element) as u128;
            let result = self.zmm(i.vex_v as usize)[0] & !elements_mask | value & elements_mask;
            self.commit_scalar(i, i.rm as usize, element, result);
            return Ok(());
        }
        if i.vex_v != 0 || i.evex_zeroing {
            return Err(Exception::invalid_opcode());
        }
        if i.evex_mask == 0 || self.opmask[i.evex_mask as usize] & 1 == 1 {
            let address = self.evex_address(i, next, element);
            self.write(address, element as u8, value as u64)?;
        }
        Ok(())
    }

    /// vmovd/vmovq xmm <- r/m32 or r/m64. No masking; the rest of the register is zeroed.
    fn move_from_general(&mut self, i: &Insn, next: u64, wide: bool) -> ExecResult<()> {
        self.check_general_move(i)?;
        let size = if wide { 8 } else { 4 };
        let value = if i.md == 3 {
            self.read_gpr(i.rm & 0xf, size, true)
        } else {
            let address = self.evex_address(i, next, size as usize);
            self.read(address, size)?
        };
        self.set_zmm(i.reg as usize, [value as u128, 0, 0, 0]);
        Ok(())
    }

    fn move_to_general(&mut self, i: &Insn, next: u64, wide: bool) -> ExecResult<()> {
        self.check_general_move(i)?;
        let size = if wide { 8 } else { 4 };
        let value = self.zmm(i.reg as usize)[0] as u64;
        if i.md == 3 {
            self.write_gpr(i.rm & 0xf, size, true, value);
        } else {
            let address = self.evex_address(i, next, size as usize);
            self.write(address, size, value)?;
        }
        Ok(())
    }

    /// vmovd/vmovq take no mask, no vvvv and a 128-bit length.
    pub(super) fn check_general_move(&self, i: &Insn) -> ExecResult<()> {
        if i.vex_v != 0 || i.evex_mask != 0 || i.evex_zeroing || i.evex_b || i.evex_ll != LENGTH_128
        {
            return Err(Exception::invalid_opcode());
        }
        Ok(())
    }

    /// F3 0F 7E: vmovq xmm <- xmm/m64.
    fn move_quad_load(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        self.check_general_move(i)?;
        let value = if i.md == 3 {
            self.zmm(i.rm as usize)[0] as u64
        } else {
            let address = self.evex_address(i, next, 8);
            self.read(address, 8)?
        };
        self.set_zmm(i.reg as usize, [value as u128, 0, 0, 0]);
        Ok(())
    }

    /// 66 0F D6: vmovq xmm/m64 <- xmm.
    fn move_quad_store(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        self.check_general_move(i)?;
        let value = self.zmm(i.reg as usize)[0] as u64;
        if i.md == 3 {
            self.set_zmm(i.rm as usize, [value as u128, 0, 0, 0]);
        } else {
            let address = self.evex_address(i, next, 8);
            self.write(address, 8, value)?;
        }
        Ok(())
    }
}
