//! x86-64 block generator. Mirrors `emit.rs`; the guest flags map straight onto host flags, so a
//! conditional jump right after an inlined ALU op tests them directly. Anything that clobbers host
//! flags (TLB probes, helper calls) ends that window.
//!
//! Instructions without an inline lowering call `helper_exec` (the interpreter). Every way out of a
//! block jumps to a shared stub in `arch_x64::prologue` with the exit code described in
//! `helpers.rs`: `exit_ok` with rax = `retired << EXIT_COUNT_SHIFT`, or `exit_fail` with rax = the
//! helper status and rcx = index of the instruction that stopped the block.

use std::mem::{offset_of, size_of};

use super::helpers::*;
use super::x64::*;
use crate::cpu::decode::{Insn, NO_REG, NO_SEG};
use crate::cpu::flags::LazyFlagsOp;
use crate::cpu::{Cpu, SEG_FS, SEG_GS, Seg, TLB_ENTRIES_PER_MODE, TlbEntry};

// Offsets into `Cpu` baked into generated code. They come from `offset_of!`, so they always match
// the compiled layout; what has to stay true is the shape of the `#[repr(C)]` `Cpu` and `TlbEntry`:
// `gpr` is indexed by guest register number, `rip`/`icount`/`prepaid_icount`/`lazy_dst`/`lazy_src`
// are u64, `block_exit_requested` is a bool (tested as a byte), and `lazy_size` immediately follows
// the one-byte `lazy_op` (written together by `store_lazy_op`).
const GPR: i32 = offset_of!(Cpu, gpr) as i32;
const RIP: i32 = offset_of!(Cpu, rip) as i32;
const ICOUNT: i32 = offset_of!(Cpu, icount) as i32;
const BLOCK_EXIT_REQUESTED: i32 = offset_of!(Cpu, block_exit_requested) as i32;
const PREPAID_ICOUNT: i32 = offset_of!(Cpu, prepaid_icount) as i32;
const LAZY_OP: i32 = offset_of!(Cpu, lazy_op) as i32;
const LAZY_DST: i32 = offset_of!(Cpu, lazy_dst) as i32;
const LAZY_SRC: i32 = offset_of!(Cpu, lazy_src) as i32;
// `Seg` is not `repr(C)`; `offset_of!` and `size_of` still report the real layout.
const SEG_BASE: i32 = (offset_of!(Cpu, seg) + offset_of!(Seg, base)) as i32;
const SEG_SIZE: i32 = size_of::<Seg>() as i32;

const _: () = assert!(size_of::<LazyFlagsOp>() == 1);
const _: () = assert!(offset_of!(Cpu, lazy_size) == offset_of!(Cpu, lazy_op) + 1);

// `Cpu::translate` indexes `tlb` as `[user << TLB_INDEX_BITS | page_number & (ENTRIES - 1)]`; the
// inline probe gets the user/supervisor half from `Cpu::tlb_active_base` and only adds the
// in-half index, scaled by the size of a `TlbEntry` (32 bytes; `Cpu::set_cpl` uses the same
// scale for its half offset).
const TLB_TAG_READ: i32 = offset_of!(TlbEntry, tag_r) as i32;
const TLB_TAG_WRITE: i32 = offset_of!(TlbEntry, tag_w) as i32;
const TLB_ADDEND: i32 = offset_of!(TlbEntry, addend) as i32;
const TLB_ENTRY_SHIFT: u8 = size_of::<TlbEntry>().trailing_zeros() as u8;
const _: () = assert!(size_of::<TlbEntry>().is_power_of_two());

const PAGE_SHIFT: u8 = 12;
/// Sign-extends to `!0xfff` in a 64-bit `and`.
const PAGE_MASK: i32 = !((1 << PAGE_SHIFT) - 1);

// Host register roles. `CPU` and `TLB_BASE` live for the whole block. `ACC`..`SCRATCH_D` and
// `VALUE` are working registers that any helper call clobbers; the slow paths therefore stash
// `ADDR` and `VALUE` in `SAVED_ADDR`/`SAVED_VALUE`. RBX, RBP, R12 and R13 are callee-saved on
// both ABIs, and the prologue saves them (and the Win64 callee-saved RSI/RDI/R14/R15 too).
const CPU: u8 = RBX;
/// `Cpu::tlb_active_base`, loaded once at block entry. CPL only changes inside block-terminating
/// instructions, so it cannot go stale within a block.
const TLB_BASE: u8 = RBP;
/// First ALU operand and result; also the destination of a loaded first operand.
const ACC: u8 = RAX;
/// Second ALU operand.
const OPERAND: u8 = RCX;
/// Guest linear address of the memory operand being accessed.
const ADDR: u8 = RDX;
/// Scratch: TLB entry pointer during a probe, index/segment base in `ea`.
const SCRATCH_A: u8 = RSI;
/// Scratch: the TLB addend while completing an access.
const SCRATCH_B: u8 = RDI;
/// Scratch: the page tag compared against the TLB entry.
const SCRATCH_C: u8 = R8;
/// Scratch for 64-bit constants stored to `Cpu` (rip, icount).
const SCRATCH_D: u8 = R10;
/// Value being stored to memory or just loaded; moves between a guest register and memory.
const VALUE: u8 = R9;
/// Holds the helper address for `call`; helpers are too far away for a rel32 call.
const CALL_TARGET: u8 = R11;
const SAVED_ADDR: u8 = R12;
const SAVED_VALUE: u8 = R13;

const GUEST_RAX: u8 = 0;
const GUEST_RSP: u8 = 4;
/// `Insn::rex` bit B, which extends the register number encoded in the opcode.
const GUEST_REX_B: u8 = 1;
/// Without a REX prefix, byte-register numbers 4..8 mean ah/ch/dh/bh (bits 8..16 of guest
/// registers 0..4), not the low byte of rsp/rbp/rsi/rdi.
const HIGH_BYTE_REGS: std::ops::Range<u8> = 4..8;

/// Stack frame the prologue reserves below its saved registers. The eight pushes leave rsp at
/// 8 mod 16, so the frame must be 8 mod 16 for calls from blocks to see rsp 16-byte aligned (both
/// ABIs require it). It also has to hold the 32 bytes of Win64 shadow space at the bottom, with
/// the slow-path load result slot (`OUT`) above it. Shared with `arch_x64`.
pub(crate) const FRAME: i32 = 56;
/// Offset from rsp of the 8-byte slot `helper_load_via_out` writes its result to. Above the shadow space so the
/// callee cannot spill its register arguments over it.
const OUT: i32 = 32;

/// Group-1 ALU operation numbers (the x86 opcode extension), plus a pseudo operation.
const ALU_ADD: u8 = 0;
const ALU_OR: u8 = 1;
const ALU_ADC: u8 = 2;
const ALU_SBB: u8 = 3;
const ALU_AND: u8 = 4;
const ALU_SUB: u8 = 5;
const ALU_CMP: u8 = 7;
/// `test`: and without writeback. Not a real group-1 number; must stay outside 0..=7.
const ALU_TEST: u8 = 8;

// Opcodes that read the guest clock (`Insn::op`; 0x100 and up are `0f xx`).
const OP_RDTSC: u16 = 0x131;
const OP_IN_OUT_IMM8: std::ops::RangeInclusive<u16> = 0xe4..=0xe7;
const OP_IN_OUT_DX: std::ops::RangeInclusive<u16> = 0xec..=0xef;

/// A TLB-miss path, emitted out of line after the block body.
enum SlowKind {
    Ld { dst: u8, size: u8, signed: bool },
    St { size: u8 },
}

struct Slow {
    /// Taken from the inline TLB probe on a miss.
    slow: Label,
    /// Where the inline path continues after the access completes.
    resume: Label,
    kind: SlowKind,
    /// Block index of the accessing instruction, reported to `exit_fail` if the access faults.
    insn_index: u32,
    /// Guest rip of that instruction, stored to `Cpu::rip` before the helper runs because
    /// `helper_load_via_out`/`helper_store_x64` do not set it.
    rip: u64,
}

#[derive(Clone, Copy)]
enum Dst {
    Reg(u8),
    Mem,
}

#[derive(Clone, Copy)]
enum Src {
    Reg(u8),
    Mem,
    Imm(u64),
}

/// Absolute addresses of the helpers and of the shared exit stubs, loaded into registers at the
/// call sites. Built once per `Jit` by `arch_x64::addrs`.
pub(crate) struct Addrs {
    pub exec: u64,
    pub ld: u64,
    pub st: u64,
    pub cond: u64,
    pub exit_ok: u64,
    pub exit_fail: u64,
}

pub(crate) struct Gen<'a> {
    pub a: Asm,
    addrs: &'a Addrs,
    /// Generated code embeds raw pointers into this slice (for `helper_exec`), so the caller must keep
    /// it alive as long as the block.
    insns: &'a [Insn],
    slows: Vec<Slow>,
    /// Out-of-line `(label, insn_index, next rip)` exits taken when a store requested a block exit.
    smc: Vec<(Label, u32, u64)>,
    /// Host flags currently equal the guest flags (set by the last inlined ALU op).
    live: bool,
    /// Number of guest instructions lowered inline (not through `helper_exec`), for statistics.
    pub inlined: u32,
}

impl<'a> Gen<'a> {
    pub fn new(base: u64, addrs: &'a Addrs, insns: &'a [Insn]) -> Self {
        Gen {
            a: Asm::new(base),
            addrs,
            insns,
            slows: Vec::new(),
            smc: Vec::new(),
            live: false,
            inlined: 0,
        }
    }

    fn ld_gpr(&mut self, rd: u8, r: u8) {
        self.a.ld(rd, mem(CPU, GPR + 8 * r as i32));
    }

    fn st_gpr(&mut self, r: u8, rs: u8) {
        self.a.st(mem(CPU, GPR + 8 * r as i32), rs, 8);
    }

    fn call(&mut self, addr: u64) {
        self.a.mov_imm(CALL_TARGET, addr);
        self.a.call_reg(CALL_TARGET);
    }

    fn set_rip(&mut self, rip: u64) {
        self.a.mov_imm(SCRATCH_D, rip);
        self.a.st(mem(CPU, RIP), SCRATCH_D, 8);
    }

    /// Leave the block normally having retired `retired` instructions.
    fn exit_ok(&mut self, retired: u32) {
        self.a.mov_imm(ACC, (retired as u64) << EXIT_COUNT_SHIFT);
        self.a.jmp_abs(self.addrs.exit_ok);
    }

    /// Leave the block because the status in rax is non-zero; the epilogue wants the failing
    /// instruction's block index in rcx.
    fn fail_exit(&mut self, insn_index: u32) {
        self.a.mov_imm(OPERAND, insn_index as u64);
        self.a.jmp_abs(self.addrs.exit_fail);
    }

    /// Run instruction `insn_index` through the interpreter. `terminal` ends the block after it.
    fn helper(&mut self, insn_index: usize, rip: u64, terminal: bool) {
        let insn = &self.insns[insn_index] as *const Insn as u64;
        // The run loop already counted one instruction for the block. These instructions see guest
        // time, which derives from `icount`, so add the ones before this now and tell
        // `step_block` through `prepaid_icount` not to count them again.
        if insn_index > 0 && reads_guest_clock(self.insns[insn_index].op) {
            self.a.ld(SCRATCH_D, mem(CPU, ICOUNT));
            self.a.alu_ri(ADD, SCRATCH_D, insn_index as i32, true);
            self.a.st(mem(CPU, ICOUNT), SCRATCH_D, 8);
            self.a.mov_imm(SCRATCH_D, insn_index as u64);
            self.a.st(mem(CPU, PREPAID_ICOUNT), SCRATCH_D, 8);
        }
        self.a.mov_rr(ARGS[0], CPU, true);
        self.a.mov_imm(ARGS[1], insn);
        self.a.mov_imm(ARGS[2], rip);
        self.call(self.addrs.exec);
        let ok = self.a.label();
        self.a.test_rr(ACC, ACC, true);
        self.a.jcc(E, ok);
        self.fail_exit(insn_index as u32);
        self.a.bind(ok);
        if terminal {
            self.exit_ok(insn_index as u32 + 1);
        }
        self.live = false;
    }

    /// Guest linear address of `insn`'s memory operand into `ADDR`. Only lea and moves here: the
    /// effective address must not disturb live host flags.
    fn ea(&mut self, insn: &Insn, next: u64) {
        let disp = insn.disp as i32;
        if insn.rip_rel {
            // Blocks are keyed by guest rip, so this is a translation-time constant.
            self.a.mov_imm(ADDR, next.wrapping_add(insn.disp as u64));
        } else if insn.index != NO_REG {
            if insn.base != NO_REG {
                self.ld_gpr(ADDR, insn.base);
            } else {
                self.a.mov_imm(ADDR, 0);
            }
            self.ld_gpr(SCRATCH_A, insn.index);
            self.a.lea(ADDR, mem_idx(ADDR, SCRATCH_A, insn.scale, disp));
        } else if insn.base != NO_REG {
            self.ld_gpr(ADDR, insn.base);
            if disp != 0 {
                self.a.lea(ADDR, mem(ADDR, disp));
            }
        } else {
            self.a.mov_imm(ADDR, insn.disp as u64);
        }
        // In 64-bit mode only the FS and GS bases apply; other overrides are ignored.
        if has_fs_gs_override(insn) {
            self.a
                .ld(SCRATCH_A, mem(CPU, SEG_BASE + SEG_SIZE * insn.seg as i32));
            self.a.lea(ADDR, mem_idx(ADDR, SCRATCH_A, 0, 0));
        }
    }

    /// Inline mirror of the fast path in `Cpu::mem_read`/`mem_write`. Jumps to `slow` on a miss;
    /// on a hit falls through with `SCRATCH_A` pointing at the TLB entry. Clobbers host flags.
    fn tlb_probe(&mut self, size: u8, tag_offset: i32, slow: Label) {
        self.live = false;
        self.a.mov_rr(SCRATCH_A, ADDR, true);
        self.a.shift_ri(SHR, SCRATCH_A, PAGE_SHIFT, true);
        self.a
            .alu_ri(AND, SCRATCH_A, (TLB_ENTRIES_PER_MODE - 1) as i32, true);
        self.a.shift_ri(SHL, SCRATCH_A, TLB_ENTRY_SHIFT, true);
        self.a.lea(SCRATCH_A, mem_idx(TLB_BASE, SCRATCH_A, 0, 0));
        // Compare the page of the access's last byte against the tag. The entry was picked by the
        // first byte's page, so an access that crosses a page can never match (the next page
        // maps to a different slot) and takes the slow path, which splits it. This replaces the
        // interpreter's separate `(la & 0xfff) + size <= 4096` check. TLB_TAG_INVALID has low
        // bits set, so it never equals a page-aligned value.
        if size == 1 {
            self.a.mov_rr(SCRATCH_C, ADDR, true);
        } else {
            self.a.lea(SCRATCH_C, mem(ADDR, size as i32 - 1));
        }
        self.a.alu_ri(AND, SCRATCH_C, PAGE_MASK, true);
        self.a.cmp_mr(mem(SCRATCH_A, tag_offset), SCRATCH_C);
        self.a.jcc(NE, slow);
    }

    /// Load `size` bytes at `ADDR` into `dst`. A TLB miss runs the helper out of line and resumes
    /// after the load with `ADDR` intact.
    fn mem_load(&mut self, dst: u8, size: u8, signed: bool, insn_index: u32, rip: u64) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, TLB_TAG_READ, slow);
        // The addend turns a guest linear address into a host pointer: host = la + addend.
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        self.a
            .ld_ext(dst, mem_idx(ADDR, SCRATCH_B, 0, 0), size, signed);
        self.a.bind(resume);
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Ld { dst, size, signed },
            insn_index,
            rip,
        });
    }

    /// Store the low `size` bytes of `VALUE` at `ADDR`. The write tag is invalidated for pages
    /// holding translated code, so stores to code always miss and reach `Cpu::code_written`.
    fn mem_store(&mut self, size: u8, insn_index: u32, rip: u64) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, TLB_TAG_WRITE, slow);
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        self.a.st(mem_idx(ADDR, SCRATCH_B, 0, 0), VALUE, size);
        self.a.bind(resume);
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::St { size },
            insn_index,
            rip,
        });
    }

    /// After a store: if it invalidated translated code (`Cpu::code_written` sets
    /// `block_exit_requested`), leave the block so the rest of it does not run from stale code.
    fn smc_check(&mut self, insn_index: u32, next: u64) {
        let exit = self.a.label();
        self.a.cmp_m8_imm(mem(CPU, BLOCK_EXIT_REQUESTED), 0);
        self.a.jcc(NE, exit);
        self.smc.push((exit, insn_index, next));
    }

    /// Emit everything deferred out of line, then resolve labels. Must be the last thing emitted
    /// for a block.
    ///
    /// Slow paths call `helper_load_via_out`/`helper_store_x64` with the guest address saved in `SAVED_ADDR` (and a
    /// store's value in `SAVED_VALUE`), because the call clobbers every other working register.
    /// They restore `ADDR` before jumping back, so the inline code may only rely on `ADDR` (and
    /// on `dst`, for loads) surviving a memory access.
    fn emit_slows(&mut self) {
        let smc = std::mem::take(&mut self.smc);
        for (label, insn_index, next) in smc {
            self.a.bind(label);
            self.set_rip(next);
            self.exit_ok(insn_index + 1);
        }
        let slows = std::mem::take(&mut self.slows);
        for slow in slows {
            self.a.bind(slow.slow);
            self.a.mov_rr(SAVED_ADDR, ADDR, true);
            self.set_rip(slow.rip);
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_rr(ARGS[1], SAVED_ADDR, true);
            let fail = self.a.label();
            match slow.kind {
                SlowKind::Ld { dst, size, signed } => {
                    self.a.mov_imm(ARGS[2], size as u64);
                    self.a.lea(ARGS[3], mem(RSP, OUT));
                    self.call(self.addrs.ld);
                    self.a.test_rr(ACC, ACC, true);
                    self.a.jcc(NE, fail);
                    self.a.ld_ext(dst, mem(RSP, OUT), size, signed);
                }
                SlowKind::St { size } => {
                    self.a.mov_rr(SAVED_VALUE, VALUE, true);
                    self.a.mov_imm(ARGS[2], size as u64);
                    self.a.mov_rr(ARGS[3], SAVED_VALUE, true);
                    self.call(self.addrs.st);
                    self.a.test_rr(ACC, ACC, true);
                    self.a.jcc(NE, fail);
                }
            }
            self.a.mov_rr(ADDR, SAVED_ADDR, true);
            self.a.jmp(slow.resume);
            self.a.bind(fail);
            self.fail_exit(slow.insn_index);
        }
        self.a.finish();
    }

    /// Record the just-emitted ALU op in the lazy flags: result in `ACC`, second operand in
    /// `OPERAND`. `lazy_op` and `lazy_size` are adjacent bytes, written with one 16-bit store.
    fn store_lazy_op(&mut self, op: LazyFlagsOp, osz: u8) {
        self.a
            .st_imm16(mem(CPU, LAZY_OP), (op as u32 | (osz as u32) << 8) as u16);
        self.a.st(mem(CPU, LAZY_DST), ACC, 8);
        self.a.st(mem(CPU, LAZY_SRC), OPERAND, 8);
        self.live = true;
    }

    /// Memory operands the inline path handles: 64-bit addressing, no lock prefix.
    fn mem_ok(insn: &Insn) -> bool {
        !insn.addr32 && (insn.seg == NO_SEG || insn.seg as usize <= SEG_GS) && !insn.lock
    }

    /// `alu_op` is a group-1 number or `ALU_TEST`. Loads both operands (the memory operand first,
    /// so a slow-path call cannot clobber the register one), runs the op on `ACC`/`OPERAND`, and
    /// writes the result back to a register destination except for cmp and test. A memory
    /// destination is therefore only used by cmp/test, which never store.
    #[allow(clippy::too_many_arguments)]
    fn alu(
        &mut self,
        insn: &Insn,
        alu_op: u8,
        dst: Dst,
        src: Src,
        osz: u8,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) {
        let w = osz == 8;
        match (dst, src) {
            (Dst::Mem, Src::Reg(s)) => {
                self.ea(insn, next);
                self.mem_load(ACC, osz, false, insn_index, rip);
                self.ld_gpr(OPERAND, s);
            }
            (Dst::Mem, Src::Imm(v)) => {
                self.ea(insn, next);
                self.mem_load(ACC, osz, false, insn_index, rip);
                self.a.mov_imm(OPERAND, v);
            }
            (Dst::Reg(d), Src::Mem) => {
                self.ea(insn, next);
                self.mem_load(OPERAND, osz, false, insn_index, rip);
                self.ld_gpr(ACC, d);
            }
            (Dst::Reg(d), Src::Reg(s)) => {
                self.ld_gpr(ACC, d);
                self.ld_gpr(OPERAND, s);
            }
            (Dst::Reg(d), Src::Imm(v)) => {
                self.ld_gpr(ACC, d);
                self.a.mov_imm(OPERAND, v);
            }
            (Dst::Mem, Src::Mem) => unreachable!(),
        }
        // cmp and test run as sub and and on the scratch copy: the result feeds the lazy flags
        let (host_op, lazy_op) = match alu_op {
            ALU_ADD => (ADD, LazyFlagsOp::Add),
            ALU_OR => (OR, LazyFlagsOp::Logic),
            ALU_AND | ALU_TEST => (AND, LazyFlagsOp::Logic),
            ALU_SUB | ALU_CMP => (SUB, LazyFlagsOp::Sub),
            _ => (XOR, LazyFlagsOp::Logic),
        };
        self.a.alu_rr(host_op, ACC, OPERAND, w);
        if alu_op != ALU_CMP
            && alu_op != ALU_TEST
            && let Dst::Reg(d) = dst
        {
            self.st_gpr(d, ACC);
        }
        self.store_lazy_op(lazy_op, osz);
    }

    /// Immediate as the host op should see it: sign-extended to 64 bits for 64-bit operations,
    /// truncated to 32 bits otherwise.
    fn imm_for(osz: u8, imm: i64) -> u64 {
        if osz == 8 {
            imm as u64
        } else {
            imm as u32 as u64
        }
    }

    /// Conditional branch ending the block. With `live` host flags the guest condition code is
    /// also the host one; otherwise `helper_cond` evaluates it from the lazy flags.
    fn jcc(&mut self, cc: u8, target: u64, next: u64, retired: u32) {
        let taken = self.a.label();
        if self.live {
            self.a.jcc(cc & 15, taken);
        } else {
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_imm(ARGS[1], cc as u64);
            self.call(self.addrs.cond);
            self.a.test_rr(ACC, ACC, true);
            self.a.jcc(NE, taken);
        }
        self.set_rip(next);
        self.exit_ok(retired);
        self.a.bind(taken);
        self.set_rip(target);
        self.exit_ok(retired);
    }

    /// Lower instruction `index` inline if it is a supported form. Returns false with nothing
    /// emitted otherwise (callers fall back to `helper_exec`), so every `return false` below must come
    /// before the first emit for that arm.
    fn try_inline(&mut self, index: usize, rip: u64, next: u64) -> bool {
        let insn: Insn = self.insns[index];
        let insn_index = index as u32;
        let retired = insn_index + 1;
        let osz = insn.opsize;
        // 32/64-bit operation without lock, 66 or 67 prefixes (`opsize` alone cannot tell: REX.W
        // overrides 66).
        let plain = !insn.addr32 && !insn.lock && !insn.p66 && (osz == 4 || osz == 8);
        let is_mem = insn.md != 3;
        let mem_ok = Self::mem_ok(&insn);
        let rex = insn.rex_present;
        let mut store = false;
        match insn.op {
            // ALU grid 00..3f: bits 5:3 pick the operation, bits 2:0 the form (1 = r/m, r;
            // 3 = r, r/m; 5 = eax, imm32; the byte forms 0/2/4 are not handled). 6 and 7 are
            // push/pop of segment registers and prefixes, excluded by `< 6`.
            0x00..=0x3f if insn.op & 7 < 6 && plain => {
                let alu_op = ((insn.op >> 3) & 7) as u8;
                // adc and sbb need the guest carry, which is not in the host flags.
                if alu_op == ALU_ADC || alu_op == ALU_SBB {
                    return false;
                }
                match insn.op & 7 {
                    1 => {
                        // A memory destination is only for cmp (no writeback, so no SMC check).
                        if is_mem && (alu_op != ALU_CMP || !mem_ok) {
                            return false;
                        }
                        let dst = if is_mem { Dst::Mem } else { Dst::Reg(insn.rm) };
                        self.alu(
                            &insn,
                            alu_op,
                            dst,
                            Src::Reg(insn.reg),
                            osz,
                            next,
                            insn_index,
                            rip,
                        );
                    }
                    3 => {
                        if is_mem && !mem_ok {
                            return false;
                        }
                        let src = if is_mem { Src::Mem } else { Src::Reg(insn.rm) };
                        self.alu(
                            &insn,
                            alu_op,
                            Dst::Reg(insn.reg),
                            src,
                            osz,
                            next,
                            insn_index,
                            rip,
                        );
                    }
                    5 => {
                        let v = Self::imm_for(osz, insn.imm);
                        self.alu(
                            &insn,
                            alu_op,
                            Dst::Reg(GUEST_RAX),
                            Src::Imm(v),
                            osz,
                            next,
                            insn_index,
                            rip,
                        );
                    }
                    _ => return false,
                }
            }
            // group 1: r/m, imm8 (sign-extended by the decoder) or imm32
            0x81 | 0x83 if plain => {
                let alu_op = insn.reg & 7;
                if alu_op == ALU_ADC
                    || alu_op == ALU_SBB
                    || (is_mem && (alu_op != ALU_CMP || !mem_ok))
                {
                    return false;
                }
                let v = Self::imm_for(osz, insn.imm);
                let dst = if is_mem { Dst::Mem } else { Dst::Reg(insn.rm) };
                self.alu(&insn, alu_op, dst, Src::Imm(v), osz, next, insn_index, rip);
            }
            // test r/m, r
            0x85 if plain => {
                if is_mem && !mem_ok {
                    return false;
                }
                let dst = if is_mem { Dst::Mem } else { Dst::Reg(insn.rm) };
                self.alu(
                    &insn,
                    ALU_TEST,
                    dst,
                    Src::Reg(insn.reg),
                    osz,
                    next,
                    insn_index,
                    rip,
                );
            }
            // test eax, imm32
            0xa9 if plain => {
                let v = Self::imm_for(osz, insn.imm);
                self.alu(
                    &insn,
                    ALU_TEST,
                    Dst::Reg(GUEST_RAX),
                    Src::Imm(v),
                    osz,
                    next,
                    insn_index,
                    rip,
                );
            }
            // group 3 /0: test r/m, imm32
            0xf7 if plain && insn.reg & 7 == 0 => {
                if is_mem && !mem_ok {
                    return false;
                }
                let v = Self::imm_for(osz, insn.imm);
                let dst = if is_mem { Dst::Mem } else { Dst::Reg(insn.rm) };
                self.alu(
                    &insn,
                    ALU_TEST,
                    dst,
                    Src::Imm(v),
                    osz,
                    next,
                    insn_index,
                    rip,
                );
            }
            // mov r/m, r
            0x89 if plain => {
                if is_mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&insn, next);
                    self.ld_gpr(VALUE, insn.reg);
                    self.mem_store(osz, insn_index, rip);
                    store = true;
                } else {
                    self.ld_gpr(VALUE, insn.reg);
                    // A 32-bit register write clears the upper half.
                    if osz == 4 {
                        self.a.mov_rr(VALUE, VALUE, false);
                    }
                    self.st_gpr(insn.rm, VALUE);
                }
            }
            // mov r, r/m
            0x8b if plain => {
                if is_mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&insn, next);
                    self.mem_load(VALUE, osz, false, insn_index, rip);
                } else {
                    self.ld_gpr(VALUE, insn.rm);
                    if osz == 4 {
                        self.a.mov_rr(VALUE, VALUE, false);
                    }
                }
                self.st_gpr(insn.reg, VALUE);
            }
            // mov m8, r8
            0x88 if is_mem && mem_ok && !insn.p66 => {
                if !rex && HIGH_BYTE_REGS.contains(&insn.reg) {
                    self.ld_gpr(VALUE, insn.reg - HIGH_BYTE_REGS.start);
                    self.a.shift_ri(SHR, VALUE, 8, true);
                } else {
                    self.ld_gpr(VALUE, insn.reg);
                }
                self.ea(&insn, next);
                self.mem_store(1, insn_index, rip);
                store = true;
            }
            // group 11 /0: mov r/m, imm32
            0xc7 if plain && insn.reg & 7 == 0 => {
                let v = Self::imm_for(osz, insn.imm);
                if is_mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&insn, next);
                    self.a.mov_imm(VALUE, v);
                    self.mem_store(osz, insn_index, rip);
                    store = true;
                } else {
                    self.a.mov_imm(VALUE, v);
                    self.st_gpr(insn.rm, VALUE);
                }
            }
            // mov r, imm (register in the opcode, extended by REX.B)
            0xb8..=0xbf if plain => {
                let r = (insn.op as u8 & 7) | ((insn.rex & GUEST_REX_B) << 3);
                self.a.mov_imm(VALUE, Self::imm_for(osz, insn.imm));
                self.st_gpr(r, VALUE);
            }
            // lea; `ea` would add a segment base, which lea must not
            0x8d if plain && is_mem => {
                if has_fs_gs_override(&insn) {
                    return false;
                }
                self.ea(&insn, next);
                if osz == 4 {
                    self.a.mov_rr(ADDR, ADDR, false);
                }
                self.st_gpr(insn.reg, ADDR);
            }
            // movsxd r64, r/m32
            0x63 if osz == 8 && !insn.addr32 && !insn.lock && !insn.p66 && (!is_mem || mem_ok) => {
                if is_mem {
                    self.ea(&insn, next);
                    self.mem_load(VALUE, 4, true, insn_index, rip);
                } else {
                    self.ld_gpr(VALUE, insn.rm);
                    self.a.ext_rr(VALUE, VALUE, 4, true);
                }
                self.st_gpr(insn.reg, VALUE);
            }
            // movzx (0f b6, b7) / movsx (0f be, bf) from a byte (even opcode) or word source
            0x1b6 | 0x1b7 | 0x1be | 0x1bf if plain && (!is_mem || mem_ok) => {
                let source_size: u8 = if insn.op & 1 == 0 { 1 } else { 2 };
                let signed = insn.op >= 0x1b8;
                if is_mem {
                    self.ea(&insn, next);
                    self.mem_load(VALUE, source_size, signed, insn_index, rip);
                } else {
                    // ah/ch/dh/bh source: not worth a special case.
                    if source_size == 1 && !rex && HIGH_BYTE_REGS.contains(&insn.rm) {
                        return false;
                    }
                    self.ld_gpr(VALUE, insn.rm);
                    self.a.ext_rr(VALUE, VALUE, source_size, signed);
                }
                if osz == 4 {
                    self.a.mov_rr(VALUE, VALUE, false);
                }
                self.st_gpr(insn.reg, VALUE);
            }
            // push r64. rsp is updated only after the store succeeded, and the register is read
            // after the new rsp is computed but before it is stored, so `push rsp` pushes the
            // old value.
            0x50..=0x57 if !insn.p66 => {
                let r = (insn.op as u8 & 7) | ((insn.rex & GUEST_REX_B) << 3);
                self.ld_gpr(ADDR, GUEST_RSP);
                self.a.lea(ADDR, mem(ADDR, -8));
                self.ld_gpr(VALUE, r);
                self.mem_store(8, insn_index, rip);
                self.st_gpr(GUEST_RSP, ADDR);
                store = true;
            }
            // pop r64. The register is written last so `pop rsp` ends up with the popped value.
            0x58..=0x5f if !insn.p66 => {
                let r = (insn.op as u8 & 7) | ((insn.rex & GUEST_REX_B) << 3);
                self.ld_gpr(ADDR, GUEST_RSP);
                self.mem_load(VALUE, 8, false, insn_index, rip);
                self.a.lea(ADDR, mem(ADDR, 8));
                self.st_gpr(GUEST_RSP, ADDR);
                self.st_gpr(r, VALUE);
            }
            // nop, prefetchw (0f 0d), and the 0f 18..1f hint/nop group: nothing to emit. 0x90
            // with REX.B is xchg r8, rax.
            0x90 | 0x10d | 0x118..=0x11f => {
                if insn.op == 0x90 && insn.rex & GUEST_REX_B != 0 {
                    return false;
                }
            }
            // jcc rel8 / rel32
            0x70..=0x7f | 0x180..=0x18f => {
                let target = next.wrapping_add(insn.imm as u64);
                self.jcc((insn.op & 15) as u8, target, next, retired);
                self.inlined += 1;
                return true;
            }
            // jmp rel32 / rel8
            0xe9 | 0xeb => {
                self.set_rip(next.wrapping_add(insn.imm as u64));
                self.exit_ok(retired);
                self.inlined += 1;
                return true;
            }
            // call rel32
            0xe8 => {
                self.ld_gpr(ADDR, GUEST_RSP);
                self.a.lea(ADDR, mem(ADDR, -8));
                self.a.mov_imm(VALUE, next);
                self.mem_store(8, insn_index, rip);
                self.st_gpr(GUEST_RSP, ADDR);
                self.set_rip(next.wrapping_add(insn.imm as u64));
                self.exit_ok(retired);
                self.inlined += 1;
                return true;
            }
            // ret
            0xc3 => {
                self.ld_gpr(ADDR, GUEST_RSP);
                self.mem_load(VALUE, 8, false, insn_index, rip);
                self.a.lea(ADDR, mem(ADDR, 8));
                self.st_gpr(GUEST_RSP, ADDR);
                self.a.st(mem(CPU, RIP), VALUE, 8);
                self.exit_ok(retired);
                self.inlined += 1;
                return true;
            }
            _ => return false,
        }
        if store {
            self.smc_check(insn_index, next);
        }
        self.inlined += 1;
        true
    }

    /// `rips[k]` is the guest rip of instruction `k`; `end_rip` is the rip after the last one.
    pub fn compile(&mut self, rips: &[u64], end_rip: u64) {
        let count = self.insns.len();
        for index in 0..count {
            let rip = rips[index];
            let next = if index + 1 < count {
                rips[index + 1]
            } else {
                end_rip
            };
            let terminal = ends_block(&self.insns[index]);
            if self.try_inline(index, rip, next) {
                if is_inline_term(&self.insns[index]) {
                    self.emit_slows();
                    return;
                }
                continue;
            }
            self.helper(index, rip, terminal);
            if terminal {
                self.emit_slows();
                return;
            }
        }
        self.set_rip(end_rip);
        self.exit_ok(count as u32);
        self.emit_slows();
    }
}

fn has_fs_gs_override(insn: &Insn) -> bool {
    matches!(insn.seg as usize, SEG_FS | SEG_GS)
}

/// Instructions whose result depends on guest time (`Cpu::now_ns` is derived from `icount`).
fn reads_guest_clock(op: u16) -> bool {
    op == OP_RDTSC || OP_IN_OUT_IMM8.contains(&op) || OP_IN_OUT_DX.contains(&op)
}

/// Instructions `try_inline` lowers that end the block themselves: jcc, jmp, call, ret. Keep in
/// step with the terminal arms there.
fn is_inline_term(insn: &Insn) -> bool {
    matches!(
        insn.op,
        0x70..=0x7f | 0x180..=0x18f | 0xe9 | 0xeb | 0xe8 | 0xc3
    )
}
