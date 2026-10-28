//! EVEX broadcasts, 128/256-bit inserts and extracts, element inserts and extracts, and the
//! half-lane moves: the data movement of maps 0F38 and 0F3A and a few 0F leftovers.

use super::*;

const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;
const BROADCAST_BLOCK_128: usize = 16;
const BROADCAST_BLOCK_256: usize = 32;

fn size_mask_wide(bytes: usize) -> u128 {
    if bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (bytes * 8)) - 1
    }
}

impl Cpu {
    pub(super) fn exec_evex_shuffle_data(
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
            return Ok(false);
        }
        let dword_or_qword = if wide { 8 } else { 4 };
        match (map, opcode) {
            // vbroadcastss/sd and vpbroadcastb/w/d/q from a vector register or memory
            (OPCODE_MAP_0F38, 0x18 | 0x58) if !wide => self.broadcast(i, next, 4, 4)?,
            (OPCODE_MAP_0F38, 0x19 | 0x59) if wide => self.broadcast(i, next, 8, 8)?,
            // the 32x2 forms: a pair of dwords repeated, masked per dword
            (OPCODE_MAP_0F38, 0x19 | 0x59) => self.broadcast(i, next, 8, 4)?,
            (OPCODE_MAP_0F38, 0x78) if !wide => self.broadcast(i, next, 1, 1)?,
            (OPCODE_MAP_0F38, 0x79) if !wide => self.broadcast(i, next, 2, 2)?,
            // vbroadcast{f,i}32x4 / 64x2 and 32x8 / 64x4: memory only
            (OPCODE_MAP_0F38, 0x1a | 0x5a) => {
                self.broadcast_block(i, next, BROADCAST_BLOCK_128, dword_or_qword)?
            }
            (OPCODE_MAP_0F38, 0x1b | 0x5b) => {
                self.broadcast_block(i, next, BROADCAST_BLOCK_256, dword_or_qword)?
            }
            (OPCODE_MAP_0F38, 0x7a..=0x7c) => {
                let size = match (opcode, wide) {
                    (0x7a, false) => 1,
                    (0x7b, false) => 2,
                    (0x7c, false) => 4,
                    (0x7c, true) => 8,
                    _ => return Err(Exception::invalid_opcode()),
                };
                self.broadcast_general(i, size)?;
            }
            (OPCODE_MAP_0F38, 0x2a) if !wide => {
                if i.md == 3 || i.evex_mask != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let bytes = self.vector_bytes(i, Rounding::None)?;
                self.check_move_alignment(i, next, bytes)?;
                let shape = RmShape {
                    broadcast: false,
                    suppress: None,
                    ..RmShape::full(bytes, 8)
                };
                let value = self.fetch_rm(i, next, shape)?;
                self.commit_masked(i, i.reg as usize, 8, bytes, &value);
            }
            (OPCODE_MAP_0F3A, 0x18 | 0x38) => {
                self.insert_block(i, next, BROADCAST_BLOCK_128, dword_or_qword)?
            }
            (OPCODE_MAP_0F3A, 0x1a | 0x3a) => {
                self.insert_block(i, next, BROADCAST_BLOCK_256, dword_or_qword)?
            }
            (OPCODE_MAP_0F3A, 0x19 | 0x39) => {
                self.extract_block(i, next, BROADCAST_BLOCK_128, dword_or_qword)?
            }
            (OPCODE_MAP_0F3A, 0x1b | 0x3b) => {
                self.extract_block(i, next, BROADCAST_BLOCK_256, dword_or_qword)?
            }
            (OPCODE_MAP_0F3A, 0x14) => self.extract_general(i, next, 1, i.imm as usize & 15)?,
            (OPCODE_MAP_0F3A, 0x15) => self.extract_general(i, next, 2, i.imm as usize & 7)?,
            (OPCODE_MAP_0F3A, 0x16) => {
                let size = if wide { 8 } else { 4 };
                self.extract_general(i, next, size, i.imm as usize & (16 / size - 1))?;
            }
            (OPCODE_MAP_0F3A, 0x17) if !wide => {
                self.extract_general(i, next, 4, i.imm as usize & 3)?
            }
            (OPCODE_MAP_0F3A, 0x20) => self.insert_general(i, next, 1, i.imm as usize & 15)?,
            (OPCODE_MAP_0F3A, 0x22) => {
                let size = if wide { 8 } else { 4 };
                self.insert_general(i, next, size, i.imm as usize & (16 / size - 1))?;
            }
            (OPCODE_MAP_0F3A, 0x21) if !wide => self.insert_single(i, next)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// vbroadcast*/vpbroadcast*: `source_bytes` from the r/m operand repeated over the vector.
    /// `element` is the mask granularity (it differs from `source_bytes` for the 32x2 forms).
    fn broadcast(
        &mut self,
        i: &Insn,
        next: u64,
        source_bytes: usize,
        element: usize,
    ) -> ExecResult<()> {
        if i.vex_v != 0 || i.evex_b {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let chunk = if i.md == 3 {
            self.zmm(i.rm as usize)[0] & size_mask_wide(source_bytes)
        } else {
            let address = self.evex_address(i, next, source_bytes);
            self.read(address, source_bytes as u8)? as u128
        };
        let mut value = [0u128; LANES];
        for k in 0..bytes / source_bytes {
            let bit = k * source_bytes * 8;
            value[bit / 128] |= chunk << (bit % 128);
        }
        self.commit_masked(i, i.reg as usize, element, bytes, &value);
        Ok(())
    }

    /// vbroadcast{f,i}32x4/64x2 (`block` 16) and the x8/x4 forms (`block` 32): a memory block
    /// repeated across the vector.
    fn broadcast_block(
        &mut self,
        i: &Insn,
        next: u64,
        block: usize,
        element: usize,
    ) -> ExecResult<()> {
        if i.vex_v != 0 || i.md == 3 || i.evex_b {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        // the destination must be wider than the block
        if bytes <= block {
            return Err(Exception::invalid_opcode());
        }
        let address = self.evex_address(i, next, block);
        let chunk = self.load_vector(address, block, element, None)?;
        let lanes_per_block = block / LANE_BYTES;
        let mut value = [0u128; LANES];
        for (lane, slot) in value.iter_mut().enumerate().take(bytes / LANE_BYTES) {
            *slot = chunk[lane % lanes_per_block];
        }
        self.commit_masked(i, i.reg as usize, element, bytes, &value);
        Ok(())
    }

    /// vpbroadcastb/w/d/q from a general register (EVEX only).
    fn broadcast_general(&mut self, i: &Insn, size: usize) -> ExecResult<()> {
        if i.vex_v != 0 || i.md != 3 || i.evex_b {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let value = self.read_gpr(i.rm & 0xf, size.max(4) as u8, true) & size_mask(size);
        let result = build(bytes, size, |_| value);
        self.commit_masked(i, i.reg as usize, size, bytes, &result);
        Ok(())
    }

    /// vinsert{f,i}{32x4,64x2} (`block` 16) and {32x8,64x4} (`block` 32): vvvv with one block
    /// replaced by the r/m operand, selected by the immediate.
    fn insert_block(
        &mut self,
        i: &Insn,
        next: u64,
        block: usize,
        element: usize,
    ) -> ExecResult<()> {
        let bytes = self.vector_bytes(i, Rounding::None)?;
        if bytes <= block {
            return Err(Exception::invalid_opcode());
        }
        let shape = RmShape {
            broadcast: false,
            suppress: None,
            ..RmShape::full(block, element)
        };
        let inserted = self.fetch_rm(i, next, shape)?;
        let mut value = self.zmm(i.vex_v as usize);
        let slot = (i.imm as usize & (bytes / block - 1)) * (block / LANE_BYTES);
        value[slot..slot + block / LANE_BYTES].copy_from_slice(&inserted[..block / LANE_BYTES]);
        self.commit_masked(i, i.reg as usize, element, bytes, &value);
        Ok(())
    }

    /// vextract{f,i}{32x4,64x2} and {32x8,64x4}: one block of ModRM.reg into r/m, a register
    /// (with masking) or memory (masked store).
    fn extract_block(
        &mut self,
        i: &Insn,
        next: u64,
        block: usize,
        element: usize,
    ) -> ExecResult<()> {
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, Rounding::None)?;
        if bytes <= block {
            return Err(Exception::invalid_opcode());
        }
        let source = self.zmm(i.reg as usize);
        let slot = (i.imm as usize & (bytes / block - 1)) * (block / LANE_BYTES);
        let mut part = [0u128; LANES];
        part[..block / LANE_BYTES].copy_from_slice(&source[slot..slot + block / LANE_BYTES]);
        if i.md == 3 {
            self.commit_masked(i, i.rm as usize, element, block, &part);
            return Ok(());
        }
        if i.evex_zeroing {
            return Err(Exception::invalid_opcode());
        }
        let address = self.evex_address(i, next, block);
        let mask = self.write_mask(i);
        self.store_vector(address, &part, block, element, mask)
    }

    /// vpextrb/w/d/q and vextractps: one element of ModRM.reg to a general register or memory.
    pub(super) fn extract_general(
        &mut self,
        i: &Insn,
        next: u64,
        size: usize,
        index: usize,
    ) -> ExecResult<()> {
        self.check_general_move(i)?;
        let value = element(&self.zmm(i.reg as usize), size, index);
        if i.md == 3 {
            self.write_gpr(i.rm & 0xf, size.max(4) as u8, true, value);
            return Ok(());
        }
        let address = self.evex_address(i, next, size);
        self.write(address, size as u8, value)
    }

    /// 0F C5 vpextrw: register form only, the general register is ModRM.reg.
    pub(super) fn extract_word_to_register(&mut self, i: &Insn) -> ExecResult<()> {
        self.check_general_move(i)?;
        if i.md != 3 {
            return Err(Exception::invalid_opcode());
        }
        let value = element(&self.zmm(i.rm as usize), 2, i.imm as usize & 7);
        self.write_gpr(i.reg & 0xf, 4, true, value);
        Ok(())
    }

    /// vpinsrb/w/d/q: a general register or memory element inserted into vvvv's copy.
    pub(super) fn insert_general(
        &mut self,
        i: &Insn,
        next: u64,
        size: usize,
        index: usize,
    ) -> ExecResult<()> {
        if i.evex_mask != 0 || i.evex_zeroing || i.evex_b || i.evex_ll != LENGTH_128 {
            return Err(Exception::invalid_opcode());
        }
        let value = if i.md == 3 {
            self.read_gpr(i.rm & 0xf, size.max(4) as u8, true)
        } else {
            let address = self.evex_address(i, next, size);
            self.read(address, size as u8)?
        };
        let mut result = [self.zmm(i.vex_v as usize)[0], 0, 0, 0];
        set_element(&mut result, size, index, value);
        self.set_zmm(i.reg as usize, result);
        Ok(())
    }

    /// vinsertps: an element of the source (or the m32 operand) into vvvv's copy, then zeroing
    /// the elements the immediate names.
    fn insert_single(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        if i.evex_mask != 0 || i.evex_zeroing || i.evex_b || i.evex_ll != LENGTH_128 {
            return Err(Exception::invalid_opcode());
        }
        let selector = i.imm as usize;
        let value = if i.md == 3 {
            element(&self.zmm(i.rm as usize), 4, selector >> 6 & 3)
        } else {
            let address = self.evex_address(i, next, 4);
            self.read(address, 4)?
        };
        let mut result = [self.zmm(i.vex_v as usize)[0], 0, 0, 0];
        set_element(&mut result, 4, selector >> 4 & 3, value);
        for k in 0..4 {
            if selector >> k & 1 == 1 {
                set_element(&mut result, 4, k, 0);
            }
        }
        self.set_zmm(i.reg as usize, result);
        Ok(())
    }

    /// vmovlps/vmovlpd/vmovhps/vmovhpd/vmovhlps/vmovlhps: 128-bit, no masks.
    pub(super) fn move_half_lanes(&mut self, i: &Insn, next: u64, opcode: u8) -> ExecResult<()> {
        if i.evex_mask != 0 || i.evex_zeroing || i.evex_b || i.evex_ll != LENGTH_128 {
            return Err(Exception::invalid_opcode());
        }
        let low_half = u64::MAX as u128;
        match opcode {
            // 12: vmovlps/vmovlpd load, or vmovhlps (register); 16: vmovhps/vmovhpd load, or
            // vmovlhps (register)
            0x12 | 0x16 => {
                let high_half = opcode == 0x16;
                let source = if i.md == 3 {
                    let rm = self.zmm(i.rm as usize)[0];
                    if high_half { rm & low_half } else { rm >> 64 }
                } else {
                    let address = self.evex_address(i, next, 8);
                    self.read(address, 8)? as u128
                };
                let base = self.zmm(i.vex_v as usize)[0];
                let result = if high_half {
                    base & low_half | source << 64
                } else {
                    base & !low_half | source
                };
                self.set_zmm(i.reg as usize, [result, 0, 0, 0]);
            }
            // 13 and 17: stores of the low and high quadword
            _ => {
                if i.md == 3 || i.vex_v != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let value = self.zmm(i.reg as usize)[0];
                let quadword = if opcode == 0x17 {
                    value >> 64
                } else {
                    value & low_half
                };
                let address = self.evex_address(i, next, 8);
                self.write(address, 8, quadword as u64)?;
            }
        }
        Ok(())
    }
}
