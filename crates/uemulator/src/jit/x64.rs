//! Minimal x86-64 assembler for the JIT backend.
//!
//! Only the instruction forms `emit_x64.rs` needs are encoded. Register operands are the
//! architectural 0..=15 numbers; the low three bits go into ModRM/SIB and bit 3 into a REX prefix.

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;
pub const R12: u8 = 12;
pub const R13: u8 = 13;
pub const R14: u8 = 14;
pub const R15: u8 = 15;

/// Integer argument registers in order. Windows x64 and System V differ only here (and in the
/// shadow space the caller must reserve on Windows, see `emit_x64::FRAME`); the generated code
/// never passes more than four arguments, so the stack-passed tail never comes up.
#[cfg(windows)]
pub const ARGS: [u8; 4] = [RCX, RDX, R8, R9];
#[cfg(not(windows))]
pub const ARGS: [u8; 4] = [RDI, RSI, RDX, RCX];

/// Group-1 ALU operation numbers. They are both the ModRM.reg opcode extension of the immediate
/// forms (`0x81`/`0x83`) and bits 5:3 of the `op r/m, r` opcodes (`0x01`, `0x09`, ...). ADC (2)
/// and SBB (3) fill the gaps and are not used.
pub const ADD: u8 = 0;
pub const OR: u8 = 1;
pub const AND: u8 = 4;
pub const SUB: u8 = 5;
pub const XOR: u8 = 6;
pub const CMP: u8 = 7;

/// Shift operation numbers: ModRM.reg opcode extension of `0xc1`/`0xd3`.
pub const SHL: u8 = 4;
pub const SHR: u8 = 5;
pub const SAR: u8 = 7;

/// Condition codes as encoded in `0f 8x` (Jcc rel32), `0f 9x` (setcc) and `0f 4x` (cmovcc). The
/// same numbers are the low nibble of the guest `70..7f` / `0f 80..8f` opcodes, so the JIT passes
/// guest condition codes straight through. Only the ones the generated code names itself are
/// listed.
pub const B: u8 = 2;
pub const AE: u8 = 3;
pub const E: u8 = 4;
pub const NE: u8 = 5;
pub const O: u8 = 0;

const REGISTER_COUNT: u8 = 16;

const REX_BASE: u8 = 0x40;
const REX_W: u8 = 1 << 3;
const REX_R: u8 = 1 << 2;
const REX_X: u8 = 1 << 1;
const REX_B: u8 = 1;

const OPERAND_SIZE_PREFIX: u8 = 0x66;

/// ModRM.mod values.
const MOD_NO_DISPLACEMENT: u8 = 0;
const MOD_DISP8: u8 = 1;
const MOD_DISP32: u8 = 2;
const MOD_REGISTER: u8 = 3;

/// ModRM.rm (or SIB.base, low three bits) value 4 means "a SIB byte follows", so rsp and r12
/// as base always need one.
const RM_NEEDS_SIB: u8 = 4;
/// With mod=00, base/rm value 5 means disp32 with no base (RIP-relative as rm), so rbp and r13
/// as base cannot use mod=00 and get a zero disp8 instead.
const BASE_DISP32_ONLY: u8 = 5;
/// SIB.index value 4 (without REX.X) means "no index".
const SIB_NO_INDEX: u8 = 4;

/// Low bit of an opcode pair: clear for the byte form, set for the 16/32/64-bit form.
const WIDE_FORM: u8 = 1;
/// Low bits of the `op r/m, r` ALU opcodes: the 64/32-bit "r/m is the destination" form.
const ALU_RM_REG: u8 = WIDE_FORM;

/// ModRM.reg extensions of the unary group (`f6`/`f7`) and the inc/dec group (`fe`/`ff`).
const UNARY_TEST: u8 = 0;
const UNARY_NOT: u8 = 2;
const UNARY_NEG: u8 = 3;
const UNARY_INC: u8 = 0;
const UNARY_DEC: u8 = 1;
/// ModRM.reg extension of `bt r/m, imm8` (`0f ba`).
const BT_IMMEDIATE: u8 = 4;

/// A memory operand `[base + index << scale + disp]`.
#[derive(Clone, Copy)]
pub struct Mem {
    base: u8,
    /// `(register, log2 scale)`.
    index: Option<(u8, u8)>,
    disp: i32,
}

pub fn mem(base: u8, disp: i32) -> Mem {
    Mem {
        base,
        index: None,
        disp,
    }
}

/// `[base + index << scale + disp]`; `scale` is log2 (0..=3) and `index` must not be RSP, whose
/// SIB encoding means "no index" (R12 is fine, REX.X distinguishes it).
pub fn mem_idx(base: u8, index: u8, scale: u8, disp: i32) -> Mem {
    debug_assert!(index != RSP && scale <= 3);
    Mem {
        base,
        index: Some((index, scale)),
        disp,
    }
}

#[derive(Clone, Copy)]
pub struct Label(usize);

pub struct Asm {
    /// Address the finished code will execute at: `jmp_abs` and `here` are relative to it, so it
    /// must be the final location, not a staging buffer.
    base: u64,
    pub code: Vec<u8>,
    /// Bound offset into `code` per label.
    labels: Vec<Option<usize>>,
    /// `(offset of a rel32 field, label)`; resolved by `finish`.
    fixups: Vec<(usize, usize)>,
}

impl Asm {
    pub fn new(base: u64) -> Asm {
        Asm {
            base,
            code: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
        }
    }

    pub fn here(&self) -> u64 {
        self.base + self.code.len() as u64
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.code.len());
    }

    /// Patches every label reference. All labels that were jumped to must be bound by now.
    pub fn finish(&mut self) {
        for &(pos, label) in &self.fixups {
            let target = self.labels[label].expect("unbound label");
            // rel32 is relative to the end of the instruction, and the field is always last.
            let rel = target as i64 - (pos as i64 + 4);
            self.code[pos..pos + 4].copy_from_slice(&(rel as i32).to_le_bytes());
        }
        self.fixups.clear();
    }

    /// Emits a REX prefix only when it carries a bit. `force_rex` is for byte operations on a
    /// register numbered 4..=7: without any REX they address ah/ch/dh/bh instead of spl/bpl/sil/dil.
    fn rex(&mut self, w: bool, reg: u8, index: u8, base: u8, force_rex: bool) {
        // Every encoder funnels through here (`mov_imm`, `alu_ri`, ...); a number of 16 or more
        // would silently alias register `n & 15` because only bit 3 is kept.
        debug_assert!(reg < REGISTER_COUNT && index < REGISTER_COUNT && base < REGISTER_COUNT);
        let mut rex = REX_BASE;
        if w {
            rex |= REX_W;
        }
        if reg & 8 != 0 {
            rex |= REX_R;
        }
        if index & 8 != 0 {
            rex |= REX_X;
        }
        if base & 8 != 0 {
            rex |= REX_B;
        }
        if rex != REX_BASE || force_rex {
            self.code.push(rex);
        }
    }

    fn modrm_mem(&mut self, reg: u8, m: Mem) {
        let base = m.base & 7;
        let sib = m.index.is_some() || base == RM_NEEDS_SIB;
        let mode = if m.disp == 0 && base != BASE_DISP32_ONLY {
            MOD_NO_DISPLACEMENT
        } else if i8::try_from(m.disp).is_ok() {
            MOD_DISP8
        } else {
            MOD_DISP32
        };
        self.code
            .push(mode << 6 | (reg & 7) << 3 | if sib { RM_NEEDS_SIB } else { base });
        if sib {
            let (index, scale) = m.index.map_or((SIB_NO_INDEX, 0), |(i, s)| (i & 7, s));
            self.code.push(scale << 6 | index << 3 | base);
        }
        match mode {
            MOD_DISP8 => self.code.push(m.disp as i8 as u8),
            MOD_DISP32 => self.code.extend_from_slice(&m.disp.to_le_bytes()),
            _ => {}
        }
    }

    /// `w` is REX.W (64-bit operand size). `reg` is the ModRM.reg field: a register or an
    /// opcode extension digit.
    fn op_rm(
        &mut self,
        operand_size_prefix: bool,
        w: bool,
        force_rex: bool,
        opcode: &[u8],
        reg: u8,
        m: Mem,
    ) {
        if operand_size_prefix {
            self.code.push(OPERAND_SIZE_PREFIX);
        }
        self.rex(w, reg, m.index.map_or(0, |i| i.0), m.base, force_rex);
        self.code.extend_from_slice(opcode);
        self.modrm_mem(reg, m);
    }

    fn op_rr(
        &mut self,
        operand_size_prefix: bool,
        w: bool,
        force_rex: bool,
        opcode: &[u8],
        reg: u8,
        rm: u8,
    ) {
        if operand_size_prefix {
            self.code.push(OPERAND_SIZE_PREFIX);
        }
        self.rex(w, reg, 0, rm, force_rex);
        self.code.extend_from_slice(opcode);
        self.code
            .push(MOD_REGISTER << 6 | (reg & 7) << 3 | (rm & 7));
    }

    /// With `w == false` this is a 32-bit move that zero-extends into the upper half, which is
    /// how the JIT truncates a value to 32 bits (also with `d == s`).
    pub fn mov_rr(&mut self, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x89], s, d);
    }

    /// Picks the shortest encoding. Deliberately a `mov` even for zero (never `xor`): emitted
    /// blocks load constants while host flags are still live for a following `jcc`.
    pub fn mov_imm(&mut self, r: u8, v: u64) {
        if v <= u32::MAX as u64 {
            // mov r32, imm32 zero-extends.
            self.rex(false, 0, 0, r, false);
            self.code.push(0xb8 | (r & 7));
            self.code.extend_from_slice(&(v as u32).to_le_bytes());
        } else if i32::try_from(v as i64).is_ok() {
            // mov r/m64, imm32 sign-extends: covers the negative values 32-bit mov cannot.
            self.op_rr(false, true, false, &[0xc7], 0, r);
            self.code.extend_from_slice(&(v as u32).to_le_bytes());
        } else {
            // movabs r64, imm64
            self.rex(true, 0, 0, r, false);
            self.code.push(0xb8 | (r & 7));
            self.code.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// 64-bit load.
    pub fn ld(&mut self, d: u8, m: Mem) {
        self.op_rm(false, true, false, &[0x8b], d, m);
    }

    /// Load `size` bytes, zero- or sign-extended to 64 bits. The 32-bit zero-extending form
    /// relies on the CPU clearing the upper half; `size` outside 1/2/4 means 8.
    pub fn ld_ext(&mut self, d: u8, m: Mem, size: u8, signed: bool) {
        match (size, signed) {
            (1, false) => self.op_rm(false, false, false, &[0x0f, 0xb6], d, m),
            (1, true) => self.op_rm(false, true, false, &[0x0f, 0xbe], d, m),
            (2, false) => self.op_rm(false, false, false, &[0x0f, 0xb7], d, m),
            (2, true) => self.op_rm(false, true, false, &[0x0f, 0xbf], d, m),
            (4, false) => self.op_rm(false, false, false, &[0x8b], d, m),
            // movsxd
            (4, true) => self.op_rm(false, true, false, &[0x63], d, m),
            _ => self.op_rm(false, true, false, &[0x8b], d, m),
        }
    }

    /// Register-to-register counterpart of `ld_ext`. For `size == 1` the source is a byte
    /// register, so REX is forced to reach sil/dil.
    pub fn ext_rr(&mut self, d: u8, s: u8, size: u8, signed: bool) {
        match (size, signed) {
            (1, false) => self.op_rr(false, false, true, &[0x0f, 0xb6], d, s),
            (1, true) => self.op_rr(false, true, true, &[0x0f, 0xbe], d, s),
            (2, false) => self.op_rr(false, false, false, &[0x0f, 0xb7], d, s),
            (2, true) => self.op_rr(false, true, false, &[0x0f, 0xbf], d, s),
            (4, false) => self.mov_rr(d, s, false),
            (4, true) => self.op_rr(false, true, false, &[0x63], d, s),
            _ => self.mov_rr(d, s, true),
        }
    }

    /// Store the low `size` bytes of `s`; `size` outside 1/2/4 means 8. The byte form forces REX
    /// so that `s` = rsi/rdi means sil/dil rather than dh/bh.
    pub fn st(&mut self, m: Mem, s: u8, size: u8) {
        match size {
            1 => self.op_rm(false, false, true, &[0x88], s, m),
            2 => self.op_rm(true, false, false, &[0x89], s, m),
            4 => self.op_rm(false, false, false, &[0x89], s, m),
            _ => self.op_rm(false, true, false, &[0x89], s, m),
        }
    }

    pub fn st_imm16(&mut self, m: Mem, v: u16) {
        self.op_rm(true, false, false, &[0xc7], 0, m);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// Does not touch host flags, which is why the block generator computes effective addresses
    /// with it.
    pub fn lea(&mut self, d: u8, m: Mem) {
        self.op_rm(false, true, false, &[0x8d], d, m);
    }

    /// `op d, s` for ADD/OR/AND/SUB/XOR/CMP (the `op r/m, r` forms).
    pub fn alu_rr(&mut self, op: u8, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[op << 3 | ALU_RM_REG], s, d);
    }

    pub fn test_rr(&mut self, a: u8, b: u8, w: bool) {
        self.op_rr(false, w, false, &[0x85], b, a);
    }

    /// A register-to-register form that exists as an opcode pair (byte form `opcode`, wider form
    /// `opcode | 1`) at any operand `size`. The byte form forces REX so that rsi/rdi mean
    /// sil/dil; the 16-bit form carries the operand-size prefix.
    fn op_rr_sized(&mut self, size: u8, opcode: u8, reg: u8, rm: u8) {
        match size {
            1 => self.op_rr(false, false, true, &[opcode], reg, rm),
            2 => self.op_rr(true, false, false, &[opcode | WIDE_FORM], reg, rm),
            4 => self.op_rr(false, false, false, &[opcode | WIDE_FORM], reg, rm),
            _ => self.op_rr(false, true, false, &[opcode | WIDE_FORM], reg, rm),
        }
    }

    /// `op d, s` at operand `size` (1, 2, 4 or 8) for ADD/OR/AND/SUB/XOR/CMP. Only the low
    /// `size` bytes of the registers take part; a 4-byte result zero-extends, narrower ones
    /// leave the upper bits of `d` alone. The flags are those of an operation of that width.
    pub fn alu_rr_sized(&mut self, op: u8, d: u8, s: u8, size: u8) {
        self.op_rr_sized(size, op << 3, s, d);
    }

    /// `test r, imm32` (sign-extended with `w`) at 32 or 64 bits.
    pub fn test_ri(&mut self, r: u8, imm: i32, w: bool) {
        self.op_rr(false, w, false, &[0xf7], UNARY_TEST, r);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    pub fn not_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xf6, UNARY_NOT, r);
    }

    pub fn neg_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xf6, UNARY_NEG, r);
    }

    /// Leaves CF alone, like the guest instruction.
    pub fn inc_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xfe, UNARY_INC, r);
    }

    pub fn dec_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xfe, UNARY_DEC, r);
    }

    /// `bt r, imm8`: CF = bit `bit` of `r`. The other flags are undefined.
    pub fn bt_ri(&mut self, r: u8, bit: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0xba], BT_IMMEDIATE, r);
        self.code.push(bit);
    }

    /// `d = d * s` truncated, signed. CF and OF are set when the full product does not fit; the
    /// other flags are undefined.
    pub fn imul_rr(&mut self, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0xaf], d, s);
    }

    /// `if cc { d = s }`. With a 32-bit operand `d` is zero-extended whether or not it moved.
    pub fn cmov_rr(&mut self, cc: u8, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0x40 | cc], d, s);
    }

    /// Shift `r` by `cl`; the CPU masks the count to 5 bits, or 6 with `w`.
    pub fn shift_cl(&mut self, op: u8, r: u8, w: bool) {
        self.op_rr(false, w, false, &[0xd3], op, r);
    }

    /// `edx:eax` sign extension (`cdq`) or `rdx:rax` (`cqo`). Does not touch flags.
    pub fn cdq(&mut self, w: bool) {
        if w {
            self.code.push(REX_BASE | REX_W);
        }
        self.code.push(0x99);
    }

    /// Uses the sign-extended imm8 form when the value fits. With `w` the imm32 is
    /// sign-extended to 64 bits.
    pub fn alu_ri(&mut self, op: u8, r: u8, imm: i32, w: bool) {
        if let Ok(imm8) = i8::try_from(imm) {
            self.op_rr(false, w, false, &[0x83], op, r);
            self.code.push(imm8 as u8);
        } else {
            self.op_rr(false, w, false, &[0x81], op, r);
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
    }

    pub fn shift_ri(&mut self, op: u8, r: u8, count: u8, w: bool) {
        self.op_rr(false, w, false, &[0xc1], op, r);
        self.code.push(count);
    }

    /// 64-bit `cmp [m], r`.
    pub fn cmp_mr(&mut self, m: Mem, r: u8) {
        self.op_rm(false, true, false, &[0x39], r, m);
    }

    pub fn cmp_m8_imm(&mut self, m: Mem, v: u8) {
        self.op_rm(false, false, false, &[0x80], CMP, m);
        self.code.push(v);
    }

    /// Byte register `r` is set to 1 if condition `cc` holds, else 0; the rest of the register is
    /// left alone, and so are the flags. REX is forced for sil/dil.
    pub fn setcc(&mut self, cc: u8, r: u8) {
        self.op_rr(false, false, true, &[0x0f, 0x90 | cc], 0, r);
    }

    pub fn sete(&mut self, r: u8) {
        self.setcc(E, r);
    }

    /// Always the rel32 form so label distances never need relaxation.
    pub fn jcc(&mut self, cc: u8, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x80 | cc]);
        self.fixups.push((self.code.len(), label.0));
        self.code.extend_from_slice(&[0; 4]);
    }

    pub fn jmp(&mut self, label: Label) {
        self.code.push(0xe9);
        self.fixups.push((self.code.len(), label.0));
        self.code.extend_from_slice(&[0; 4]);
    }

    /// Jump to an absolute address within rel32 range (the code buffer is far smaller than 2 GiB).
    pub fn jmp_abs(&mut self, target: u64) {
        let rel = target as i64 - (self.here() as i64 + 5);
        self.code.push(0xe9);
        self.code
            .extend_from_slice(&i32::try_from(rel).expect("jump out of range").to_le_bytes());
    }

    /// REX.B (if `r >= 8`) is the only prefix needed: indirect call/jump default to 64-bit.
    pub fn call_reg(&mut self, r: u8) {
        self.op_rr(false, false, false, &[0xff], 2, r);
    }

    pub fn jmp_reg(&mut self, r: u8) {
        self.op_rr(false, false, false, &[0xff], 4, r);
    }

    pub fn push(&mut self, r: u8) {
        self.rex(false, 0, 0, r, false);
        self.code.push(0x50 | (r & 7));
    }

    pub fn pop(&mut self, r: u8) {
        self.rex(false, 0, 0, r, false);
        self.code.push(0x58 | (r & 7));
    }

    pub fn ret(&mut self) {
        self.code.push(0xc3);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::sys::ExecMem;

    fn run(f: impl FnOnce(&mut Asm), a0: u64, a1: u64) -> u64 {
        let _guard = crate::jit::serialize_jit_tests();
        let m = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(m.base());
        a.mov_rr(RAX, ARGS[0], true);
        a.mov_rr(RCX, ARGS[1], true);
        f(&mut a);
        a.ret();
        a.finish();
        m.write_bytes(0, &a.code);
        let fun: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(m.base() as usize) };
        fun(a0, a1)
    }

    #[test]
    fn arithmetic() {
        assert_eq!(run(|a| a.alu_rr(ADD, RAX, RCX, true), 40, 2), 42);
        assert_eq!(run(|a| a.alu_rr(SUB, RAX, RCX, true), 40, 2), 38);
        assert_eq!(run(|a| a.alu_rr(AND, RAX, RCX, true), 0xff0, 0x0ff), 0x0f0);
        assert_eq!(run(|a| a.alu_rr(OR, RAX, RCX, true), 0xf0, 0x0f), 0xff);
        assert_eq!(run(|a| a.alu_rr(XOR, RAX, RCX, true), 0xff, 0x0f), 0xf0);
        assert_eq!(
            run(|a| a.alu_rr(ADD, RAX, RCX, false), 0xffff_ffff, 2),
            1,
            "32-bit ops zero-extend"
        );
        assert_eq!(run(|a| a.alu_ri(ADD, RAX, 5, true), 1, 0), 6);
        assert_eq!(run(|a| a.alu_ri(SUB, RAX, 0x1000, true), 0x1001, 0), 1);
        assert_eq!(run(|a| a.shift_ri(SHL, RAX, 4, true), 3, 0), 48);
        assert_eq!(run(|a| a.shift_ri(SHR, RAX, 4, true), 0xf0, 0), 0xf);
        assert_eq!(run(|a| a.lea(RAX, mem_idx(RAX, RCX, 3, -2)), 100, 5), 138);
    }

    #[test]
    fn immediates_and_high_registers() {
        for &v in &[
            0u64,
            1,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0xffff_ffff_ffff_fff0,
            0x1234_5678_9abc_def0,
        ] {
            assert_eq!(run(|a| a.mov_imm(RAX, v), 0, 0), v, "{v:#x}");
            assert_eq!(
                run(
                    |a| {
                        a.mov_imm(R9, v);
                        a.mov_rr(RAX, R9, true);
                    },
                    0,
                    0
                ),
                v,
                "{v:#x} via r9"
            );
            assert_eq!(
                run(
                    |a| {
                        a.push(R13);
                        a.mov_imm(R13, v);
                        a.mov_rr(RAX, R13, true);
                        a.pop(R13);
                    },
                    0,
                    0
                ),
                v,
                "{v:#x} via r13"
            );
        }
    }

    #[test]
    fn loads_and_stores() {
        let mut buf = [0u64; 4];
        let p = buf.as_mut_ptr() as u64;
        run(
            |a| {
                a.mov_imm(RDX, 0xdead_beef_1234_5678);
                a.st(mem(RAX, 8), RDX, 8);
            },
            p,
            0,
        );
        assert_eq!(buf[1], 0xdead_beef_1234_5678);
        let ld = |size, signed, off: i32| {
            run(
                move |a| {
                    a.mov_imm(RDX, off as u64);
                    a.ld_ext(RAX, mem_idx(RAX, RDX, 0, 0), size, signed);
                },
                p,
                0,
            )
        };
        assert_eq!(ld(8, false, 8), 0xdead_beef_1234_5678);
        assert_eq!(ld(4, false, 8), 0x1234_5678);
        assert_eq!(ld(4, true, 12), 0xffff_ffff_dead_beef);
        assert_eq!(ld(2, false, 8), 0x5678);
        assert_eq!(ld(2, true, 12), 0xffff_ffff_ffff_beef);
        assert_eq!(ld(1, false, 8), 0x78);
        assert_eq!(ld(1, true, 15), 0xffff_ffff_ffff_ffde);
        for (size, mask) in [(1u8, 0xffu64), (2, 0xffff), (4, 0xffff_ffff)] {
            buf[2] = !0;
            run(
                |a| {
                    a.mov_imm(R9, 0x1111_2222_3333_4444);
                    a.mov_imm(RDI, 16);
                    a.st(mem_idx(RAX, RDI, 0, 0), R9, size);
                },
                p,
                0,
            );
            assert_eq!(
                buf[2],
                !mask | (0x1111_2222_3333_4444 & mask),
                "size {size}"
            );
        }
    }

    #[test]
    fn base_register_encodings() {
        let mut buf = [0u64; 4];
        buf[1] = 0x42;
        let p = buf.as_mut_ptr() as u64;
        for base in [RBX, RBP, RSI, R8, R12, R13, R15] {
            let got = run(
                |a| {
                    a.push(base);
                    a.mov_rr(base, RAX, true);
                    a.ld(RAX, mem(base, 8));
                    a.pop(base);
                },
                p,
                0,
            );
            assert_eq!(got, 0x42, "base {base}");
        }
        assert_eq!(
            run(
                |a| {
                    a.alu_ri(SUB, RSP, 16, true);
                    a.st(mem(RSP, 8), RAX, 8);
                    a.ld(RAX, mem(RSP, 8));
                    a.alu_ri(ADD, RSP, 16, true);
                },
                7,
                0
            ),
            7
        );
    }

    #[test]
    fn extensions() {
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 1, true), 0, 0x80), !0x7f);
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 1, false), !0, 0x1ff), 0xff);
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 2, true), 0, 0x8000), !0x7fff);
        assert_eq!(
            run(|a| a.ext_rr(RAX, RCX, 4, true), 0, 0x8000_0000),
            !0x7fff_ffff
        );
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 4, false), 0, !0), 0xffff_ffff);
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(RSI, 0x1ff);
                    a.ext_rr(RAX, RSI, 1, false);
                },
                0,
                0
            ),
            0xff
        );
    }

    #[test]
    fn branches_and_flags() {
        const COND_L: u8 = 0xc;
        let r = run(
            |a| {
                let l = a.label();
                a.alu_rr(CMP, RAX, RCX, true);
                a.mov_imm(RAX, 1);
                a.jcc(COND_L, l);
                a.mov_imm(RAX, 2);
                a.bind(l);
            },
            1,
            5,
        );
        assert_eq!(r, 1);
        let r = run(
            |a| {
                a.alu_rr(ADD, RAX, RCX, true);
                a.sete(RDX);
                a.mov_imm(RAX, 0);
                a.ext_rr(RAX, RDX, 1, false);
            },
            u64::MAX,
            1,
        );
        assert_eq!(r, 1);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "x64_tests.rs"]
mod encoding_tests;
