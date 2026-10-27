//! Minimal aarch64 assembler: just the instruction forms `emit.rs` and the prologue in
//! `arch_a64.rs` use. Each opcode literal is the fixed bits of the instruction with every
//! register and immediate field zero; the fields are or-ed in at the `*_SHIFT` positions below.
//! `is64 = false` selects the W-register (32-bit) form by leaving `SF` clear.
//!
//! Register arguments are plain numbers 0..=31. Number 31 is `xzr` or `sp` depending on the
//! instruction, which is why `mov_sp` exists next to `mov`.

#![allow(dead_code)]

pub(crate) const XZR: u32 = 31;
pub(crate) const SP: u32 = 31;

/// A64 condition field values. Flipping bit 0 inverts a condition (`cset` relies on this).
pub(crate) const EQ: u32 = 0;
pub(crate) const NE: u32 = 1;
pub(crate) const CS: u32 = 2;
pub(crate) const CC: u32 = 3;
pub(crate) const MI: u32 = 4;
pub(crate) const PL: u32 = 5;
pub(crate) const VS: u32 = 6;
pub(crate) const VC: u32 = 7;
pub(crate) const HI: u32 = 8;
pub(crate) const LS: u32 = 9;
pub(crate) const GE: u32 = 10;
pub(crate) const LT: u32 = 11;
pub(crate) const GT: u32 = 12;
pub(crate) const LE: u32 = 13;

/// Instruction-word field positions.
const RN_SHIFT: u32 = 5;
const RM_SHIFT: u32 = 16;
/// Bit 31: 64-bit (X register) form.
const SF: u32 = 1 << 31;
const COND_SHIFT: u32 = 12;
const IMM12_SHIFT: u32 = 10;
/// Shift amount of the shifted-register add.
const IMM6_SHIFT: u32 = 10;
/// MOVZ/MOVK/MOVN: 16-bit immediate, and which halfword it replaces (`hw` * 16 bits).
const IMM16_SHIFT: u32 = 5;
const HW_SHIFT: u32 = 21;
/// Bitfield moves: rotate amount and index of the last bit of the field.
const IMMR_SHIFT: u32 = 16;
const IMMS_SHIFT: u32 = 10;
/// N:immr:imms of a logical (bitmask) immediate sits here as one 13-bit field.
const BITMASK_IMM_SHIFT: u32 = 10;
/// Load/store pair: second register and the signed 7-bit offset, scaled by 8.
const RT2_SHIFT: u32 = 10;
const IMM7_SHIFT: u32 = 15;
const IMM7_MASK: u32 = 0x7f;
const PAIR_OFFSET_SCALE: i32 = 8;

/// PC-relative branch immediates count instructions, not bytes.
const IMM26_MASK: u32 = 0x03ff_ffff;
const IMM19_MASK: u32 = 0x7_ffff;
const IMM19_SHIFT: u32 = 5;
const IMM26_SIGNED_LIMIT: i64 = 1 << 25;

/// Unsigned 12-bit immediates of add/sub and of the scaled load/store offset.
pub(crate) const IMM12_LIMIT: u32 = 1 << 12;
const XREG_BITS: u32 = 64;
const WREG_BITS: u32 = 32;

/// Index into `Asm::labels`.
pub(crate) type Label = usize;

/// Which immediate field of a branch `Asm::finish` patches once the label's position is known.
#[derive(Clone, Copy)]
enum Fixup {
    /// `b`: 26-bit offset from bit 0.
    Imm26,
    /// `b.cond`, `cbz`, `cbnz`: 19-bit offset from bit 5.
    Imm19,
}

pub(crate) struct Asm {
    pub code: Vec<u32>,
    /// Address the code will run at. Must be where the words are finally written, because
    /// `here` and `b_abs` compute pc-relative offsets against it.
    base: u64,
    /// Position (in words) each label was bound to.
    labels: Vec<Option<usize>>,
    /// Branches emitted before their target was bound: (word index, label, field to patch).
    fixes: Vec<(usize, Label, Fixup)>,
}

/// Encodes `val` as a logical immediate (the 13-bit N:immr:imms field) for `and`/`orr`/`eor`/
/// `ands`, or `None` if it is not representable. Representable values are a run of ones,
/// rotated, replicated across the register in 2, 4, ..., 64-bit elements. Zero and all-ones
/// are never encodable. For `!is64` only the low 32 bits count.
pub(crate) fn encode_bitmask(val: u64, is64: bool) -> Option<u32> {
    let register_bits: u32 = if is64 { XREG_BITS } else { WREG_BITS };
    let mut pattern = if is64 { val } else { val & 0xffff_ffff };
    if pattern == 0 || pattern == if is64 { u64::MAX } else { 0xffff_ffff } {
        return None;
    }
    if !is64 {
        pattern |= pattern << 32;
    }
    // Smallest element size whose halves are equal all the way down.
    let mut element_size = register_bits;
    loop {
        element_size /= 2;
        let half_mask = (1u64 << element_size) - 1;
        if (pattern & half_mask) != ((pattern >> element_size) & half_mask) {
            element_size *= 2;
            break;
        }
        if element_size <= 2 {
            break;
        }
    }
    let element_mask = u64::MAX >> (XREG_BITS - element_size);
    pattern &= element_mask;
    // True if `v` is one contiguous run of ones (possibly shifted up from bit 0).
    let is_run_of_ones = |v: u64| {
        v != 0 && {
            let filled = (v - 1) | v;
            filled & filled.wrapping_add(1) == 0
        }
    };
    let (rotation_source, run_length);
    if is_run_of_ones(pattern) {
        rotation_source = pattern.trailing_zeros();
        run_length = (pattern >> rotation_source).trailing_ones();
    } else {
        // The run wraps around the element: extend with ones above it and look at the zeros.
        pattern |= !element_mask;
        if !is_run_of_ones(!pattern) {
            return None;
        }
        let leading_ones = pattern.leading_ones();
        rotation_source = XREG_BITS - leading_ones;
        run_length = leading_ones + pattern.trailing_ones() - (XREG_BITS - element_size);
    }
    let immr = (element_size - rotation_source) & (element_size - 1);
    // imms holds run_length - 1 in its low bits; the bits above encode the element size
    // (0xxxxx for 32-bit elements, 10xxxx for 16, ...). N is set only for 64-bit elements,
    // which is the inverse of bit 6 of this value.
    let mut size_prefixed_imms = !(element_size - 1) << 1;
    size_prefixed_imms |= run_length - 1;
    let n = ((size_prefixed_imms >> 6) & 1) ^ 1;
    Some((n << 12) | (immr << 6) | (size_prefixed_imms & 0x3f))
}

impl Asm {
    pub fn new(base: u64) -> Asm {
        Asm {
            code: Vec::new(),
            base,
            labels: Vec::new(),
            fixes: Vec::new(),
        }
    }

    /// Address of the next instruction to be emitted.
    pub fn here(&self) -> u64 {
        self.base + self.code.len() as u64 * 4
    }

    pub fn emit(&mut self, word: u32) {
        self.code.push(word);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        self.labels.len() - 1
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label] = Some(self.code.len());
    }

    /// Patches every branch to a label. The offsets are not range-checked: `b.cond`/`cbz` reach
    /// only +-1 MiB and `b` +-128 MiB, far more than one block (at most `MAX_INSNS` instructions
    /// plus their out-of-line slow paths) ever spans. Panics on a label that was never bound.
    pub fn finish(&mut self) {
        for &(at, label, fixup) in &self.fixes {
            let offset = self.labels[label].expect("unbound label") as i64 - at as i64;
            match fixup {
                Fixup::Imm26 => self.code[at] |= (offset as u32) & IMM26_MASK,
                Fixup::Imm19 => self.code[at] |= ((offset as u32) & IMM19_MASK) << IMM19_SHIFT,
            }
        }
        self.fixes.clear();
    }

    pub fn b(&mut self, label: Label) {
        self.fixes.push((self.code.len(), label, Fixup::Imm26));
        self.emit(0x1400_0000);
    }

    pub fn b_cond(&mut self, cond: u32, label: Label) {
        self.fixes.push((self.code.len(), label, Fixup::Imm19));
        self.emit(0x5400_0000 | cond);
    }

    pub fn cbz(&mut self, rt: u32, label: Label, is64: bool) {
        self.fixes.push((self.code.len(), label, Fixup::Imm19));
        self.emit(if is64 { 0xb400_0000 } else { 0x3400_0000 } | rt);
    }

    pub fn cbnz(&mut self, rt: u32, label: Label, is64: bool) {
        self.fixes.push((self.code.len(), label, Fixup::Imm19));
        self.emit(if is64 { 0xb500_0000 } else { 0x3500_0000 } | rt);
    }

    /// Direct branch to an absolute address outside this `Asm` (the exit stubs in the prologue).
    /// Reach is +-128 MiB; the JIT code buffer (`CODE_SIZE` in `native.rs`) is exactly that big,
    /// so a block can always reach the stubs at its start.
    pub fn b_abs(&mut self, target: u64) {
        let offset = (target as i64 - self.here() as i64) / 4;
        assert!(offset.abs() < IMM26_SIGNED_LIMIT, "branch out of range");
        self.emit(0x1400_0000 | (offset as u32 & IMM26_MASK));
    }

    pub fn ret(&mut self) {
        self.emit(0xd65f_03c0);
    }

    pub fn blr(&mut self, rn: u32) {
        self.emit(0xd63f_0000 | rn << RN_SHIFT);
    }

    pub fn br(&mut self, rn: u32) {
        self.emit(0xd61f_0000 | rn << RN_SHIFT);
    }

    /// `hw` is the halfword index 0..=3 the 16 bits land in.
    pub fn movz(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0xd280_0000 | hw << HW_SHIFT | (imm & 0xffff) << IMM16_SHIFT | rd);
    }

    pub fn movk(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0xf280_0000 | hw << HW_SHIFT | (imm & 0xffff) << IMM16_SHIFT | rd);
    }

    /// Writes the bitwise NOT of the shifted immediate.
    pub fn movn(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0x9280_0000 | hw << HW_SHIFT | (imm & 0xffff) << IMM16_SHIFT | rd);
    }

    /// Loads any 64-bit constant: one `movn` when the top 48 bits are all ones, otherwise a
    /// `movz` for the first non-zero halfword and a `movk` for each further one.
    pub fn mov_imm(&mut self, rd: u32, v: u64) {
        if v == 0 {
            self.movz(rd, 0, 0);
            return;
        }
        if !v >> 16 == 0 {
            self.movn(rd, (!v) as u32, 0);
            return;
        }
        let mut first = true;
        for hw in 0..4 {
            let part = ((v >> (16 * hw)) & 0xffff) as u32;
            if part != 0 {
                if first {
                    self.movz(rd, part, hw);
                    first = false;
                } else {
                    self.movk(rd, part, hw);
                }
            }
        }
    }

    /// `orr rd, xzr, rm`. Register 31 here is `xzr`, so this cannot move from or to `sp`.
    pub fn mov(&mut self, rd: u32, rm: u32) {
        self.emit(0xaa00_03e0 | rm << RM_SHIFT | rd);
    }

    /// `add rd, rn, #0`, the form that treats register 31 as `sp`.
    pub fn mov_sp(&mut self, rd: u32, rn: u32) {
        self.emit(0x9100_0000 | rn << RN_SHIFT | rd);
    }

    /// Unshifted 12-bit immediate only; callers must range-check (see `Gen::effective_address`).
    fn addsub_imm(&mut self, op: u32, rd: u32, rn: u32, imm: u32, is64: bool) {
        let sf = if is64 { SF } else { 0 };
        assert!(imm < IMM12_LIMIT);
        self.emit(sf | op | imm << IMM12_SHIFT | rn << RN_SHIFT | rd);
    }

    pub fn add_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x1100_0000, rd, rn, imm, is64);
    }
    pub fn sub_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x5100_0000, rd, rn, imm, is64);
    }
    pub fn adds_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x3100_0000, rd, rn, imm, is64);
    }
    pub fn subs_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x7100_0000, rd, rn, imm, is64);
    }

    fn reg3(&mut self, op: u32, rd: u32, rn: u32, rm: u32, is64: bool) {
        let sf = if is64 { SF } else { 0 };
        self.emit(sf | op | rm << RM_SHIFT | rn << RN_SHIFT | rd);
    }

    pub fn add(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x0b00_0000, rd, rn, rm, is64);
    }
    pub fn adds(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x2b00_0000, rd, rn, rm, is64);
    }
    pub fn sub(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x4b00_0000, rd, rn, rm, is64);
    }
    pub fn subs(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x6b00_0000, rd, rn, rm, is64);
    }
    pub fn and(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x0a00_0000, rd, rn, rm, is64);
    }
    pub fn ands(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x6a00_0000, rd, rn, rm, is64);
    }
    pub fn orr(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x2a00_0000, rd, rn, rm, is64);
    }
    pub fn eor(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x4a00_0000, rd, rn, rm, is64);
    }
    /// High 64 bits of the signed 128-bit product.
    pub fn smulh(&mut self, rd: u32, rn: u32, rm: u32) {
        self.reg3(0x9b40_7c00, rd, rn, rm, false);
    }
    pub fn mul(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1b00_7c00, rd, rn, rm, is64);
    }
    pub fn lslv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2000, rd, rn, rm, is64);
    }
    pub fn lsrv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2400, rd, rn, rm, is64);
    }
    pub fn asrv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2800, rd, rn, rm, is64);
    }

    /// 64-bit `add rd, rn, rm, lsl #shift`.
    pub fn add_lsl(&mut self, rd: u32, rn: u32, rm: u32, shift: u32) {
        self.emit(0x8b00_0000 | rm << RM_SHIFT | shift << IMM6_SHIFT | rn << RN_SHIFT | rd);
    }

    pub fn cmp(&mut self, rn: u32, rm: u32, is64: bool) {
        self.subs(XZR, rn, rm, is64);
    }
    pub fn cmp_imm(&mut self, rn: u32, imm: u32, is64: bool) {
        self.subs_imm(XZR, rn, imm, is64);
    }
    pub fn tst(&mut self, rn: u32, rm: u32, is64: bool) {
        self.ands(XZR, rn, rm, is64);
    }

    /// Returns false, emitting nothing, when `imm` is not a valid bitmask immediate; the
    /// caller must then materialise the constant in a register itself.
    #[must_use]
    fn logic_imm(&mut self, op: u32, rd: u32, rn: u32, imm: u64, is64: bool) -> bool {
        match encode_bitmask(imm, is64) {
            Some(bitmask) => {
                let sf = if is64 { SF } else { 0 };
                self.emit(sf | op | bitmask << BITMASK_IMM_SHIFT | rn << RN_SHIFT | rd);
                true
            }
            None => false,
        }
    }

    /// See `logic_imm` for the meaning of the result.
    #[must_use]
    pub fn and_imm(&mut self, rd: u32, rn: u32, imm: u64, is64: bool) -> bool {
        self.logic_imm(0x1200_0000, rd, rn, imm, is64)
    }

    // The shifts and extensions below are all aliases of these three bitfield moves, with
    // `immr` the rotate amount and `imms` the index of the field's last bit.
    pub fn ubfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let opcode = if is64 { 0xd340_0000 } else { 0x5300_0000 };
        self.emit(opcode | immr << IMMR_SHIFT | imms << IMMS_SHIFT | rn << RN_SHIFT | rd);
    }
    pub fn sbfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let opcode = if is64 { 0x9340_0000 } else { 0x1300_0000 };
        self.emit(opcode | immr << IMMR_SHIFT | imms << IMMS_SHIFT | rn << RN_SHIFT | rd);
    }
    pub fn bfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let opcode = if is64 { 0xb340_0000 } else { 0x3300_0000 };
        self.emit(opcode | immr << IMMR_SHIFT | imms << IMMS_SHIFT | rn << RN_SHIFT | rd);
    }

    /// 64-bit immediate shifts.
    pub fn lsl_imm(&mut self, rd: u32, rn: u32, shift: u32) {
        self.ubfm(
            rd,
            rn,
            (XREG_BITS - shift) & (XREG_BITS - 1),
            XREG_BITS - 1 - shift,
            true,
        );
    }
    pub fn lsr_imm(&mut self, rd: u32, rn: u32, shift: u32) {
        self.ubfm(rd, rn, shift, XREG_BITS - 1, true);
    }
    pub fn asr_imm(&mut self, rd: u32, rn: u32, shift: u32) {
        self.sbfm(rd, rn, shift, XREG_BITS - 1, true);
    }
    /// `ubfx rd, rn, #lsb, #width`: the field zero-extended into `rd`.
    pub fn ubfx(&mut self, rd: u32, rn: u32, lsb: u32, width: u32, is64: bool) {
        self.ubfm(rd, rn, lsb, lsb + width - 1, is64);
    }

    /// Immediate shifts at either register width (the 64-bit-only `lsl_imm` family above is the
    /// common case; these exist for 8/16/32-bit operand sizes done in W registers).
    pub fn lsl_imm_sized(&mut self, rd: u32, rn: u32, shift: u32, is64: bool) {
        let bits = if is64 { XREG_BITS } else { WREG_BITS };
        self.ubfm(rd, rn, (bits - shift) & (bits - 1), bits - 1 - shift, is64);
    }
    pub fn lsr_imm_sized(&mut self, rd: u32, rn: u32, shift: u32, is64: bool) {
        let bits = if is64 { XREG_BITS } else { WREG_BITS };
        self.ubfm(rd, rn, shift, bits - 1, is64);
    }
    pub fn asr_imm_sized(&mut self, rd: u32, rn: u32, shift: u32, is64: bool) {
        let bits = if is64 { XREG_BITS } else { WREG_BITS };
        self.sbfm(rd, rn, shift, bits - 1, is64);
    }
    pub fn uxtb(&mut self, rd: u32, rn: u32) {
        self.ubfm(rd, rn, 0, 7, false);
    }
    pub fn uxth(&mut self, rd: u32, rn: u32) {
        self.ubfm(rd, rn, 0, 15, false);
    }
    /// 32-bit `mov`, which zeroes the upper half of the destination.
    pub fn uxtw(&mut self, rd: u32, rn: u32) {
        self.emit(0x2a00_03e0 | rn << RM_SHIFT | rd);
    }
    pub fn sxtb(&mut self, rd: u32, rn: u32, is64: bool) {
        self.sbfm(rd, rn, 0, 7, is64);
    }
    pub fn sxth(&mut self, rd: u32, rn: u32, is64: bool) {
        self.sbfm(rd, rn, 0, 15, is64);
    }
    pub fn sxtw(&mut self, rd: u32, rn: u32) {
        self.sbfm(rd, rn, 0, 31, true);
    }

    /// Inserts the low `width` bits of `rn` into `rd` at bit `lsb`, keeping the other bits.
    pub fn bfi(&mut self, rd: u32, rn: u32, lsb: u32, width: u32, is64: bool) {
        let last_bit = if is64 { XREG_BITS - 1 } else { WREG_BITS - 1 };
        self.bfm(rd, rn, (last_bit + 1 - lsb) & last_bit, width - 1, is64);
    }

    /// `csinc rd, xzr, xzr, !cond`: rd = cond ? 1 : 0.
    pub fn cset(&mut self, rd: u32, cond: u32) {
        self.emit(0x9a9f_07e0 | ((cond ^ 1) << COND_SHIFT) | rd);
    }

    pub fn csel(&mut self, rd: u32, rn: u32, rm: u32, cond: u32) {
        self.emit(0x9a80_0000 | rm << RM_SHIFT | cond << COND_SHIFT | rn << RN_SHIFT | rd);
    }

    /// Unsigned-offset load/store: the encoded immediate is `off >> scale`, so `off` must be a
    /// multiple of the access size (`1 << scale`) and below `4096 << scale`.
    fn ldst_imm(&mut self, base: u32, scale: u32, rt: u32, rn: u32, off: u32) {
        assert!(
            off.is_multiple_of(1 << scale) && (off >> scale) < IMM12_LIMIT,
            "bad ldst offset {off}"
        );
        self.emit(base | (off >> scale) << IMM12_SHIFT | rn << RN_SHIFT | rt);
    }

    pub fn ldr(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xf940_0000, 3, rt, rn, off);
    }
    pub fn str(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xf900_0000, 3, rt, rn, off);
    }
    pub fn ldr_w(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xb940_0000, 2, rt, rn, off);
    }
    pub fn str_w(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xb900_0000, 2, rt, rn, off);
    }
    pub fn ldrh(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x7940_0000, 1, rt, rn, off);
    }
    pub fn strh(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x7900_0000, 1, rt, rn, off);
    }
    pub fn ldrb(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x3940_0000, 0, rt, rn, off);
    }
    pub fn strb(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x3900_0000, 0, rt, rn, off);
    }

    /// Register-offset form `[rn, rm]`: no shift, no extend.
    fn ldst_reg(&mut self, base: u32, rt: u32, rn: u32, rm: u32) {
        self.emit(base | rm << RM_SHIFT | rn << RN_SHIFT | rt);
    }

    /// Loads `size` bytes from `[rn + rm]`. Unsigned loads zero-extend; signed ones sign-extend
    /// to 64 bits (`ldrsb`/`ldrsh`/`ldrsw` into an X register). A signed 8-byte load is a plain
    /// `ldr`, and a signed size other than 1 or 2 is treated as 4.
    pub fn ldr_r(&mut self, rt: u32, rn: u32, rm: u32, size: u8, signed: bool) {
        let base = match (size, signed) {
            (1, false) => 0x3860_6800,
            (2, false) => 0x7860_6800,
            (4, false) => 0xb860_6800,
            (8, _) => 0xf860_6800,
            (1, true) => 0x38a0_6800,
            (2, true) => 0x78a0_6800,
            _ => 0xb8a0_6800,
        };
        self.ldst_reg(base, rt, rn, rm);
    }

    /// Stores the low `size` bytes of `rt` to `[rn + rm]`; any size other than 1, 2 or 4
    /// stores 8.
    pub fn str_r(&mut self, rt: u32, rn: u32, rm: u32, size: u8) {
        let base = match size {
            1 => 0x3820_6800,
            2 => 0x7820_6800,
            4 => 0xb820_6800,
            _ => 0xf820_6800,
        };
        self.ldst_reg(base, rt, rn, rm);
    }

    /// 64-bit `stp`/`ldp`. `imm` is a byte offset, a multiple of 8 in -512..=504.
    fn pair_imm7(imm: i32) -> u32 {
        ((imm / PAIR_OFFSET_SCALE) as u32 & IMM7_MASK) << IMM7_SHIFT
    }

    /// Pre-indexed: the base register is updated before the store.
    pub fn stp_pre(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa980_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }
    pub fn stp_off(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa900_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }
    pub fn ldp_off(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa940_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }
    /// `csinc rd, rn, rn, ne`: rd = eq ? rn + 1 : rn.
    pub fn cinc_eq(&mut self, rd: u32, rn: u32) {
        self.emit(0x9a80_0400 | rn << RM_SHIFT | (NE << COND_SHIFT) | rn << RN_SHIFT | rd);
    }
    /// Read/write the NZCV condition flags (system register S3_3_C4_C2_0).
    pub fn mrs_nzcv(&mut self, rt: u32) {
        self.emit(0xd53b_4200 | rt);
    }
    pub fn msr_nzcv(&mut self, rt: u32) {
        self.emit(0xd51b_4200 | rt);
    }
    /// Post-indexed: the base register is updated after the load.
    pub fn ldp_post(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa8c0_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::sys::ExecMem;

    fn run(f: impl FnOnce(&mut Asm), a0: u64, a1: u64) -> u64 {
        let _guard = crate::jit::serialize_jit_tests();
        let m = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(m.base());
        f(&mut a);
        a.ret();
        a.finish();
        m.write(0, &a.code);
        let fun: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(m.base() as usize) };
        fun(a0, a1)
    }

    #[test]
    fn arithmetic() {
        assert_eq!(run(|a| a.add(0, 0, 1, true), 40, 2), 42);
        assert_eq!(run(|a| a.sub(0, 0, 1, true), 40, 2), 38);
        assert_eq!(
            run(|a| a.mov_imm(0, 0x1234_5678_9abc_def0), 0, 0),
            0x1234_5678_9abc_def0
        );
        assert_eq!(run(|a| a.mov_imm(0, u64::MAX - 5), 0, 0), u64::MAX - 5);
        assert_eq!(
            run(|a| a.mov_imm(0, 0xffff_0000_0000_1234), 0, 0),
            0xffff_0000_0000_1234
        );
        assert_eq!(run(|a| a.lsl_imm(0, 0, 4), 3, 0), 48);
        assert_eq!(run(|a| a.lsr_imm(0, 0, 4), 0xf0, 0), 0xf);
        assert_eq!(
            run(|a| a.asr_imm(0, 0, 4), (-32i64) as u64, 0),
            (-2i64) as u64
        );
        assert_eq!(run(|a| a.mul(0, 0, 1, true), 6, 7), 42);
    }

    #[test]
    fn logic_immediates() {
        for &v in &[
            0xffu64,
            0x3ff,
            0xffff_ffff,
            !0xfffu64,
            0xff00,
            0x5555_5555_5555_5555,
            0x8000_0000_0000_0001,
        ] {
            let got = run(
                |a| {
                    assert!(a.and_imm(0, 0, v, true), "{v:#x}");
                },
                u64::MAX,
                0,
            );
            assert_eq!(got, v, "{v:#x}");
        }
        assert!(encode_bitmask(0x1234, true).is_none());
        assert!(encode_bitmask(0, true).is_none());
    }

    #[test]
    fn branches_and_flags() {
        let r = run(
            |a| {
                let l = a.label();
                a.cmp(0, 1, true);
                a.mov_imm(0, 1);
                a.b_cond(LT, l);
                a.mov_imm(0, 2);
                a.bind(l);
            },
            1,
            5,
        );
        assert_eq!(r, 1);
        assert_eq!(
            run(
                |a| {
                    a.adds(2, 0, 1, true);
                    a.cset(0, CS);
                },
                u64::MAX,
                1
            ),
            1
        );
    }

    #[test]
    fn loads_stores() {
        let mut buf = [0u64; 4];
        let p = buf.as_mut_ptr() as u64;
        run(
            |a| {
                a.mov_imm(2, 0xdead_beef_1234_5678);
                a.str(2, 0, 8);
            },
            p,
            0,
        );
        assert_eq!(buf[1], 0xdead_beef_1234_5678);
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 8);
                    a.ldr_r(0, 0, 2, 4, false);
                },
                p,
                0
            ),
            0x1234_5678
        );
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 8);
                    a.ldr_r(0, 0, 2, 1, true);
                },
                p,
                0
            ),
            0x78
        );
        run(
            |a| {
                a.mov_imm(2, 0xab);
                a.mov_imm(3, 16);
                a.str_r(2, 0, 3, 1);
            },
            p,
            0,
        );
        assert_eq!(buf[2] & 0xff, 0xab);
    }
}

#[cfg(test)]
#[path = "a64_tests.rs"]
mod encoding_tests;
