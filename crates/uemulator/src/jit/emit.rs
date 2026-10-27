//! aarch64 block compiler. Lowers the decoded instructions of one translated block (see
//! `Jit::translate` in `native.rs`) to machine code. Common integer instructions (ALU, mov,
//! lea, movzx/movsx, push/pop, near jmp/jcc/call/ret) are emitted inline; every other
//! instruction becomes a call to `helper_exec`, which runs the interpreter on it, so coverage can
//! grow one opcode at a time. See `plans/06-jit.md`.
//!
//! Register roles. The prologue in `arch_a64.rs` sets up the callee-saved ones and enters the
//! block with `br`, so a block is straight-line code, not a function:
//! - `CPU` (x19): `*mut Cpu`.
//! - `TLB_BASE` (x20): `Cpu::tlb_active_base`, latched once on entry. A block therefore never
//!   runs across a CPL change; every instruction that changes CPL ends the block.
//! - `SAVED_ADDRESS` (x22) and `SAVED_NZCV` (x23): hold the access address and the host flags
//!   across a slow-path helper call. x21 and x24 are saved by the prologue but unused.
//! - x0..x6 and x16 are caller-saved scratch: x0/x1 are the ALU operands and the helper
//!   argument/return registers, `ADDRESS` (x2) the guest linear address being accessed,
//!   `SCRATCH_A..C` (x3..x5) temporaries of address arithmetic and the TLB probe, `VALUE` (x6)
//!   the value moved by mov-style instructions, x16 the helper call target.
//!   Any helper call clobbers all of them, which is why operands are loaded in an order that
//!   keeps nothing live across a memory access (see `Gen::alu`).
//!
//! Exit protocol. A block never returns by itself; it ends with a branch to one of two stubs in
//! the prologue (addresses in `Addrs`): `exit_ok` with x0 = `retired << EXIT_COUNT_SHIFT`, or
//! `exit_fail` with x0 = helper status and x1 = index of the instruction that stopped the block.
//! The stub packs `status | retired << EXIT_COUNT_SHIFT` into x0 and returns to
//! `Cpu::step_block`. There is no block chaining: every exit goes back to the run loop, which
//! is also the only place interrupts are taken.
//!
//! `Cpu::rip` is not kept current inside a block. Whatever can fault or leave the block stores
//! it first: helpers get the instruction's rip as an argument, exits store the next rip.
//!
//! Flags are lazy (`LazyFlagsOp`). After every inlined ALU instruction the result and second
//! operand are stored into `Cpu::lazy_*`, so the interpreter and `helper_cond` can compute any flag
//! on demand. Additionally the host NZCV flags still hold that result (`Gen::live`) until
//! something clobbers them, which lets a following `jcc` test them directly.

use std::mem::{offset_of, size_of};

use super::a64::*;
use super::helpers::*;
use crate::cpu::decode::{Insn, NO_REG, NO_SEG};
use crate::cpu::flags::LazyFlagsOp;
use crate::cpu::{Cpu, SEG_FS, SEG_GS, Seg, TLB_ENTRIES_PER_MODE, TlbEntry};

// Byte offsets into the `#[repr(C)]` `Cpu` and `TlbEntry`, burned into the generated code.
// `offset_of!` makes them follow field moves; the generated code only breaks if a field's
// meaning or an adjacency noted below changes. Each must stay below 32 KiB (the scaled
// `ldr`/`str` immediate), or the assembler asserts when the first block is compiled.
const GPR_OFFSET: u32 = offset_of!(Cpu, gpr) as u32;
const GPR_SIZE: u32 = size_of::<u64>() as u32;
const RIP_OFFSET: u32 = offset_of!(Cpu, rip) as u32;
const ICOUNT_OFFSET: u32 = offset_of!(Cpu, icount) as u32;
const BLOCK_EXIT_REQUESTED_OFFSET: u32 = offset_of!(Cpu, block_exit_requested) as u32;
const PREPAID_ICOUNT_OFFSET: u32 = offset_of!(Cpu, prepaid_icount) as u32;
/// `lazy_size` must directly follow `lazy_op`: `store_lazy_op` writes both with one halfword
/// store (op in the low byte, size in the high byte).
const LAZY_OP_OFFSET: u32 = offset_of!(Cpu, lazy_op) as u32;
const LAZY_SIZE_SHIFT: u32 = 8;
const LAZY_DST_OFFSET: u32 = offset_of!(Cpu, lazy_dst) as u32;
const LAZY_SRC_OFFSET: u32 = offset_of!(Cpu, lazy_src) as u32;
const SEG_BASE_OFFSET: u32 = (offset_of!(Cpu, seg) + offset_of!(Seg, base)) as u32;
const SEG_STRIDE: u32 = size_of::<Seg>() as u32;

const TLB_TAG_READ_OFFSET: u32 = offset_of!(TlbEntry, tag_r) as u32;
const TLB_TAG_WRITE_OFFSET: u32 = offset_of!(TlbEntry, tag_w) as u32;
const TLB_ADDEND_OFFSET: u32 = offset_of!(TlbEntry, addend) as u32;
/// log2 of the entry size, so an entry address is `base + (index << TLB_ENTRY_SHIFT)`.
const TLB_ENTRY_SHIFT: u32 = size_of::<TlbEntry>().trailing_zeros();
const _: () = assert!(size_of::<TlbEntry>().is_power_of_two());

const PAGE_SHIFT: u32 = 12;
const PAGE_MASK: u64 = (1 << PAGE_SHIFT) - 1;

pub(super) const CPU: u32 = 19;
pub(super) const TLB_BASE: u32 = 20;
const SAVED_ADDRESS: u32 = 22;
const SAVED_NZCV: u32 = 23;
const ADDRESS: u32 = 2;
const SCRATCH_A: u32 = 3;
const SCRATCH_B: u32 = 4;
const SCRATCH_C: u32 = 5;
const VALUE: u32 = 6;
/// Intra-procedure-call scratch register of AAPCS64; holds the target of `blr`.
const CALL_TARGET: u32 = 16;

/// Index of rsp in `Cpu::gpr`.
const RSP: u8 = 4;
/// Without a REX prefix, byte-register numbers 4..=7 mean ah/ch/dh/bh (bits 15:8 of registers
/// 0..=3); with one they mean spl/bpl/sil/dil.
const FIRST_HIGH_BYTE_REG: u8 = 4;
const LAST_HIGH_BYTE_REG: u8 = 7;
const HIGH_BYTE_SHIFT: u32 = 8;
/// REX.B extends the register number embedded in the opcode (push/pop/mov r, imm).
const REX_B: u8 = 1;
const MODRM_MOD_REGISTER: u8 = 3;
const OPCODE_REG_MASK: u8 = 7;
const REX_B_REG_SHIFT: u8 = 3;

/// The x86 "group 1" ALU operation numbers (opcode bits 5:3, or ModRM.reg of 0x81/0x83).
/// `ALU_TEST` is not an x86 value: it stands for the non-storing `and` used by `test`.
const ALU_ADD: u8 = 0;
const ALU_OR: u8 = 1;
const ALU_ADC: u8 = 2;
const ALU_SBB: u8 = 3;
const ALU_AND: u8 = 4;
const ALU_SUB: u8 = 5;
const ALU_XOR: u8 = 6;
const ALU_CMP: u8 = 7;
const ALU_TEST: u8 = 8;

const PUSH_POP_SIZE: u32 = 8;
/// The largest displacement `add_imm`/`sub_imm` encode directly.
const IMM12_LIMIT_I64: i64 = IMM12_LIMIT as i64;

/// Which x86 flag-producing instruction family the host NZCV flags currently mirror. These are
/// the only `LazyFlagsOp`s the JIT produces; `cond_map` depends on the flag semantics of each.
#[derive(Clone, Copy, PartialEq, Debug)]
enum FlagKind {
    Add,
    Sub,
    Logic,
}

/// Host NZCV holds the flags of the last inlined ALU instruction. Not set at block start and
/// cleared by `helper`, whose call clobbers NZCV. Inline memory accesses preserve it.
#[derive(Clone, Copy)]
struct LiveFlags {
    kind: FlagKind,
}

enum SlowAccess {
    Load { dst: u32, size: u8, signed: bool },
    Store { size: u8 },
}

/// An out-of-line memory access, emitted after the block body by `emit_slows`. Entered when the
/// inline TLB probe misses, and jumps back to `resume` when the access succeeded.
struct SlowPath {
    entry: Label,
    resume: Label,
    access: SlowAccess,
    /// Index in the block of the instruction making the access.
    insn_index: u32,
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

/// What a condition code reduces to right after a flag-setting instruction of a given kind.
enum CondMap {
    /// Test this aarch64 condition with `b.cond`.
    Cond(u32),
    Always,
    Never,
}

/// Maps an x86 condition code (low nibble of the Jcc opcode: O, NO, B, AE, E, NE, BE, A, S, NS,
/// P, NP, L, GE, LE, G) to a test of the host NZCV flags, valid when they were set by the
/// aarch64 instruction that corresponds to `kind`. `None` means the host flags cannot express it
/// and the caller must evaluate it from the lazy flags with `helper_cond`.
///
/// - Sub: aarch64 `subs` carry is the inverse of x86 borrow, so B/AE use CC/CS and BE/A use
///   LS/HI. N, Z and V mean the same as SF, ZF, OF.
/// - Add: carry has the same sense as x86 CF, so B/AE use CS/CC. BE/A (CF | ZF) have no single
///   aarch64 condition.
/// - Logic: x86 clears CF and OF and aarch64 `ands` clears C and V, so O/B are constantly false
///   and NO/AE true, BE reduces to ZF (EQ) and A to NE; L/GE/LE/G compare N against V = 0 as is.
/// - P/NP (parity) are never available: aarch64 has no parity flag.
fn cond_map(kind: FlagKind, condition: u8) -> Option<CondMap> {
    use CondMap::*;
    let c = |cond| Some(Cond(cond));
    let index = condition as usize & 15;
    match kind {
        FlagKind::Sub => [
            c(VS),
            c(VC),
            c(CC),
            c(CS),
            c(EQ),
            c(NE),
            c(LS),
            c(HI),
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][index]
            .take(),
        FlagKind::Add => [
            c(VS),
            c(VC),
            c(CS),
            c(CC),
            c(EQ),
            c(NE),
            None,
            None,
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][index]
            .take(),
        FlagKind::Logic => [
            Some(Never),
            Some(Always),
            Some(Never),
            Some(Always),
            c(EQ),
            c(NE),
            c(EQ),
            c(NE),
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][index]
            .take(),
    }
}

/// Instructions that read guest time, which is derived from `Cpu::icount`: rdtsc, and in/out,
/// which pass the current time to the device.
fn reads_clock(op: u16) -> bool {
    // 0f 31 rdtsc; e4..e7 in/out with imm8 port; ec..ef in/out with dx port
    matches!(op, 0x131 | 0xe4..=0xe7 | 0xec..=0xef)
}

/// The register embedded in the low three opcode bits (push/pop/mov r, imm), extended by REX.B.
fn opcode_register(insn: &Insn) -> u8 {
    (insn.op as u8 & OPCODE_REG_MASK) | ((insn.rex & REX_B) << REX_B_REG_SHIFT)
}

/// ES, CS, SS and DS bases are zero in 64-bit mode (the interpreter's `Cpu::linear_address` ignores them
/// too), so only FS and GS add a base.
fn has_segment_base(insn: &Insn) -> bool {
    insn.seg == SEG_FS as u8 || insn.seg == SEG_GS as u8
}

/// Absolute addresses the generated code needs.
pub(crate) struct Addrs {
    /// Helper functions in `helpers.rs`, reached with `mov_imm` + `blr` because the host binary
    /// is arbitrarily far from the code buffer.
    pub exec: u64,
    pub ld: u64,
    pub st: u64,
    pub cond: u64,
    /// The exit stubs in the prologue at the start of the code buffer, reached with a direct
    /// branch (`Asm::b_abs`).
    pub exit_ok: u64,
    pub exit_fail: u64,
}

pub(crate) struct Gen<'a> {
    pub asm: Asm,
    addrs: &'a Addrs,
    insns: &'a [Insn],
    slow_paths: Vec<SlowPath>,
    /// Out-of-line exits taken after a store set `block_exit_requested`: (label, index of the
    /// storing instruction, rip after it).
    smc_exits: Vec<(Label, u32, u64)>,
    live: Option<LiveFlags>,
    /// Number of instructions lowered inline (not via `helper_exec`), for `Jit::inlined` statistics.
    pub inlined: u32,
}

impl<'a> Gen<'a> {
    pub fn new(base: u64, addrs: &'a Addrs, insns: &'a [Insn]) -> Self {
        Gen {
            asm: Asm::new(base),
            addrs,
            insns,
            slow_paths: Vec::new(),
            smc_exits: Vec::new(),
            live: None,
            inlined: 0,
        }
    }

    fn load_gpr(&mut self, rd: u32, reg: u8) {
        self.asm.ldr(rd, CPU, GPR_OFFSET + GPR_SIZE * reg as u32);
    }

    fn store_gpr(&mut self, reg: u8, rs: u32) {
        self.asm.str(rs, CPU, GPR_OFFSET + GPR_SIZE * reg as u32);
    }

    fn call(&mut self, addr: u64) {
        self.asm.mov_imm(CALL_TARGET, addr);
        self.asm.blr(CALL_TARGET);
    }

    fn set_rip(&mut self, rip: u64) {
        self.asm.mov_imm(1, rip);
        self.asm.str(1, CPU, RIP_OFFSET);
    }

    /// Leaves the block normally with `retired` instructions completed.
    fn exit_ok(&mut self, retired: u32) {
        self.asm.mov_imm(0, (retired as u64) << EXIT_COUNT_SHIFT);
        self.asm.b_abs(self.addrs.exit_ok);
    }

    /// Leaves the block because of a non-zero helper status already in x0; `insn_index` is the
    /// instruction that stopped the block (the stub decides whether it counts as retired).
    fn fail_exit(&mut self, insn_index: u32) {
        self.asm.mov_imm(1, insn_index as u64);
        self.asm.b_abs(self.addrs.exit_fail);
    }

    /// Runs instruction `index` through the interpreter. If `terminal`, the block ends after
    /// it: `exec` already left `Cpu::rip` at the next instruction or the branch target.
    fn helper(&mut self, index: usize, rip: u64, terminal: bool) {
        // Address of the decoded instruction, baked into the code: `Jit::translate` keeps the
        // owning slice alive in the `Block` for as long as this code can run.
        let insn = &self.insns[index] as *const Insn as u64;
        if index > 0 && reads_clock(self.insns[index].op) {
            // `Cpu::icount` is only advanced when the block exits, but these instructions read
            // the clock now. Pay for the `index` instructions already executed (the first one
            // was counted by `run`) and tell `step_block` not to count them again. It is
            // overwritten, not accumulated, so a second rdtsc in the same block would add
            // `index` to `icount` again without a matching deduction.
            self.asm.ldr(SCRATCH_B, CPU, ICOUNT_OFFSET);
            self.asm.add_imm(SCRATCH_B, SCRATCH_B, index as u32, true);
            self.asm.str(SCRATCH_B, CPU, ICOUNT_OFFSET);
            self.asm.mov_imm(SCRATCH_B, index as u64);
            self.asm.str(SCRATCH_B, CPU, PREPAID_ICOUNT_OFFSET);
        }
        // helper_exec(cpu, insn, rip) -> status
        self.asm.mov(0, CPU);
        self.asm.mov_imm(1, insn);
        self.asm.mov_imm(2, rip);
        self.call(self.addrs.exec);
        let ok = self.asm.label();
        self.asm.cbz(0, ok, true);
        self.fail_exit(index as u32);
        self.asm.bind(ok);
        if terminal {
            self.exit_ok(index as u32 + 1);
        }
        // The interpreter wrote its own flag state and the call clobbered host NZCV.
        self.live = None;
    }

    /// Computes the guest linear address of the memory operand into `dest` (the effective
    /// address plus the FS/GS base). Clobbers `SCRATCH_A`. Only valid for 64-bit address size
    /// (see `memory_inlinable`).
    fn effective_address(&mut self, insn: &Insn, next_rip: u64, dest: u32) {
        if insn.rip_rel {
            self.asm
                .mov_imm(dest, next_rip.wrapping_add(insn.disp as u64));
        } else {
            let mut have = false;
            if insn.base != NO_REG {
                self.load_gpr(dest, insn.base);
                have = true;
            }
            if insn.index != NO_REG {
                if have {
                    self.load_gpr(SCRATCH_A, insn.index);
                    self.asm.add_lsl(dest, dest, SCRATCH_A, insn.scale as u32);
                } else {
                    self.load_gpr(dest, insn.index);
                    if insn.scale != 0 {
                        self.asm.lsl_imm(dest, dest, insn.scale as u32);
                    }
                    have = true;
                }
            }
            if !have {
                self.asm.mov_imm(dest, insn.disp as u64);
            } else if insn.disp > 0 && insn.disp < IMM12_LIMIT_I64 {
                self.asm.add_imm(dest, dest, insn.disp as u32, true);
            } else if insn.disp < 0 && insn.disp > -IMM12_LIMIT_I64 {
                self.asm.sub_imm(dest, dest, (-insn.disp) as u32, true);
            } else if insn.disp != 0 {
                self.asm.mov_imm(SCRATCH_A, insn.disp as u64);
                self.asm.add(dest, dest, SCRATCH_A, true);
            }
        }
        if has_segment_base(insn) {
            self.asm.ldr(
                SCRATCH_A,
                CPU,
                SEG_BASE_OFFSET + SEG_STRIDE * insn.seg as u32,
            );
            self.asm.add(dest, dest, SCRATCH_A, true);
        }
    }

    /// Inline soft-TLB lookup for an access of `size` bytes at `ADDRESS`; branches to `slow` on a
    /// miss. On a hit `SCRATCH_A` points at the TLB entry. `tag_offset` selects the read or
    /// write tag.
    ///
    /// This is the same lookup as the first lines of `Cpu::translate`/`mem_read`/`mem_write`:
    /// the slot is `(la >> 12) & (TLB_ENTRIES_PER_MODE - 1)` within the half that `TLB_BASE`
    /// already points at, and the access hits iff the tag equals the page address. The tag is
    /// compared against the page of the access's *last* byte while the slot comes from its
    /// first, so an access that crosses a page never matches and takes the slow path, which
    /// splits it. Tags are only ever set for validated pages, so non-canonical addresses, MMIO,
    /// not-present pages, permission failures and (for stores) pages holding translated code
    /// all miss. `TLB_TAG_INVALID` (all ones) never equals a page-aligned address.
    ///
    /// Nothing here sets the host flags, so a live NZCV survives a fast-path access.
    fn tlb_probe(&mut self, size: u8, tag_offset: u32, slow: Label) {
        let entry = SCRATCH_A;
        let tag = SCRATCH_B;
        let page = SCRATCH_C;
        self.asm.lsr_imm(entry, ADDRESS, PAGE_SHIFT);
        self.asm
            .and_imm(entry, entry, (TLB_ENTRIES_PER_MODE - 1) as u64, true);
        self.asm.add_lsl(entry, TLB_BASE, entry, TLB_ENTRY_SHIFT);
        self.asm.ldr(tag, entry, tag_offset);
        if size == 1 {
            self.asm.and_imm(page, ADDRESS, !PAGE_MASK, true);
        } else {
            self.asm.add_imm(page, ADDRESS, size as u32 - 1, true);
            self.asm.and_imm(page, page, !PAGE_MASK, true);
        }
        self.asm.eor(page, tag, page, true);
        self.asm.cbnz(page, slow, true);
    }

    /// Loads `size` bytes at `ADDRESS` into `dst` (zero- or sign-extended to 64 bits). The slow
    /// path preserves `ADDRESS` and the host flags but, being a call, nothing else.
    fn mem_load(&mut self, dst: u32, size: u8, signed: bool, insn_index: u32, rip: u64) {
        let entry = self.asm.label();
        let resume = self.asm.label();
        self.tlb_probe(size, TLB_TAG_READ_OFFSET, entry);
        // addend = host page - guest page, so the host address is ADDRESS + addend.
        self.asm.ldr(SCRATCH_B, SCRATCH_A, TLB_ADDEND_OFFSET);
        self.asm.ldr_r(dst, ADDRESS, SCRATCH_B, size, signed);
        self.asm.bind(resume);
        self.slow_paths.push(SlowPath {
            entry,
            resume,
            access: SlowAccess::Load { dst, size, signed },
            insn_index,
            rip,
        });
    }

    /// Stores the low `size` bytes of `VALUE` to `ADDRESS`. The slow path reads the value from
    /// `VALUE`. A store to a page holding translated code always misses the write tag; the slow
    /// path then invalidates those blocks (`Cpu::code_written`) and sets `block_exit_requested`,
    /// which `smc_check` must poll after the store.
    fn mem_store(&mut self, size: u8, insn_index: u32, rip: u64) {
        let entry = self.asm.label();
        let resume = self.asm.label();
        self.tlb_probe(size, TLB_TAG_WRITE_OFFSET, entry);
        self.asm.ldr(SCRATCH_B, SCRATCH_A, TLB_ADDEND_OFFSET);
        self.asm.str_r(VALUE, ADDRESS, SCRATCH_B, size);
        self.asm.bind(resume);
        self.slow_paths.push(SlowPath {
            entry,
            resume,
            access: SlowAccess::Store { size },
            insn_index,
            rip,
        });
    }

    /// Self-modifying code: after a store that may have invalidated this very block, stop before
    /// the next instruction. The store itself retired, so the exit counts `insn_index + 1`.
    fn smc_check(&mut self, insn_index: u32, next_rip: u64) {
        let exit = self.asm.label();
        self.asm.ldrb(SCRATCH_B, CPU, BLOCK_EXIT_REQUESTED_OFFSET);
        self.asm.cbnz(SCRATCH_B, exit, false);
        self.smc_exits.push((exit, insn_index, next_rip));
    }

    /// Emits everything out of line: the self-modifying-code exits and the slow memory paths.
    /// Placed after the block body (and after its final exit branch) so the hot path falls
    /// through. Also resolves all labels.
    fn emit_slows(&mut self) {
        let smc_exits = std::mem::take(&mut self.smc_exits);
        for (exit, insn_index, next_rip) in smc_exits {
            self.asm.bind(exit);
            self.set_rip(next_rip);
            self.exit_ok(insn_index + 1);
        }
        let slow_paths = std::mem::take(&mut self.slow_paths);
        for slow in slow_paths {
            self.asm.bind(slow.entry);
            // The helper call clobbers NZCV and the caller-saved registers, but a live flag
            // state and ADDRESS must survive for the instructions after the access.
            self.asm.mrs_nzcv(SAVED_NZCV);
            self.asm.mov(SAVED_ADDRESS, ADDRESS);
            let fail = self.asm.label();
            match slow.access {
                SlowAccess::Load { dst, size, signed } => {
                    // helper_load(cpu, address, size, rip) -> (value in x0, status in x1)
                    self.asm.mov(0, CPU);
                    self.asm.mov(1, ADDRESS);
                    self.asm.mov_imm(2, size as u64);
                    self.asm.mov_imm(3, slow.rip);
                    self.call(self.addrs.ld);
                    self.asm.cbnz(1, fail, true);
                    if signed {
                        match size {
                            1 => self.asm.sxtb(dst, 0, true),
                            2 => self.asm.sxth(dst, 0, true),
                            4 => self.asm.sxtw(dst, 0),
                            _ => self.asm.mov(dst, 0),
                        }
                    } else {
                        self.asm.mov(dst, 0);
                    }
                    self.asm.msr_nzcv(SAVED_NZCV);
                    self.asm.mov(ADDRESS, SAVED_ADDRESS);
                    self.asm.b(slow.resume);
                    self.asm.bind(fail);
                    // Move the status into the position `fail_exit` expects.
                    self.asm.mov(0, 1);
                    self.fail_exit(slow.insn_index);
                }
                SlowAccess::Store { size } => {
                    // helper_store(cpu, address, size, value, rip) -> status. The address is copied
                    // to x1 before x2 is overwritten with the size.
                    self.asm.mov(3, VALUE);
                    self.asm.mov(1, ADDRESS);
                    self.asm.mov_imm(2, size as u64);
                    self.asm.mov_imm(4, slow.rip);
                    self.asm.mov(0, CPU);
                    self.call(self.addrs.st);
                    self.asm.cbnz(0, fail, true);
                    self.asm.msr_nzcv(SAVED_NZCV);
                    self.asm.mov(ADDRESS, SAVED_ADDRESS);
                    self.asm.b(slow.resume);
                    self.asm.bind(fail);
                    self.fail_exit(slow.insn_index);
                }
            }
        }
        self.asm.finish();
    }

    /// Records the flags of the ALU instruction just emitted: the result is in x0, the second
    /// operand in x1.
    fn store_lazy_op(&mut self, kind: FlagKind, operand_size: u8) {
        let lazy_op = match kind {
            FlagKind::Add => LazyFlagsOp::Add,
            FlagKind::Sub => LazyFlagsOp::Sub,
            FlagKind::Logic => LazyFlagsOp::Logic,
        } as u32;
        self.asm.mov_imm(
            SCRATCH_B,
            (lazy_op | (operand_size as u32) << LAZY_SIZE_SHIFT) as u64,
        );
        self.asm.strh(SCRATCH_B, CPU, LAZY_OP_OFFSET);
        self.asm.str(0, CPU, LAZY_DST_OFFSET);
        self.asm.str(1, CPU, LAZY_SRC_OFFSET);
        self.live = Some(LiveFlags { kind });
    }

    /// Whether a memory operand can use the inline access path: 64-bit address size (a 67 prefix
    /// would need 32-bit wrapping), no segment base other than FS/GS, and no LOCK prefix.
    fn memory_inlinable(insn: &Insn) -> bool {
        !insn.addr32 && (insn.seg == NO_SEG || insn.seg <= SEG_GS as u8) && !insn.lock
    }

    /// Emits `dst = dst <alu_op> src` at `operand_size` (4 or 8) and records its flags.
    /// `alu_op` is one of the `ALU_*` values; `ALU_CMP` and `ALU_TEST` discard the result. A
    /// memory destination is only ever used by those two, so nothing is written back to memory.
    ///
    /// Operand order matters: the memory operand is loaded first because its slow path calls a
    /// helper that clobbers x0..x15, and only then is the register/immediate operand loaded.
    #[allow(clippy::too_many_arguments)]
    fn alu(
        &mut self,
        insn: &Insn,
        alu_op: u8,
        dst: Dst,
        src: Src,
        operand_size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) {
        let is64 = operand_size == 8;
        match (dst, src) {
            (Dst::Mem, Src::Reg(reg)) => {
                self.effective_address(insn, next_rip, ADDRESS);
                self.mem_load(0, operand_size, false, insn_index, rip);
                self.load_gpr(1, reg);
            }
            (Dst::Mem, Src::Imm(immediate)) => {
                self.effective_address(insn, next_rip, ADDRESS);
                self.mem_load(0, operand_size, false, insn_index, rip);
                self.asm.mov_imm(1, immediate);
            }
            (Dst::Reg(reg), Src::Mem) => {
                self.effective_address(insn, next_rip, ADDRESS);
                self.mem_load(1, operand_size, false, insn_index, rip);
                self.load_gpr(0, reg);
            }
            (Dst::Reg(dst_reg), Src::Reg(src_reg)) => {
                self.load_gpr(0, dst_reg);
                self.load_gpr(1, src_reg);
            }
            (Dst::Reg(reg), Src::Imm(immediate)) => {
                self.load_gpr(0, reg);
                self.asm.mov_imm(1, immediate);
            }
            (Dst::Mem, Src::Mem) => unreachable!(),
        }
        // `orr` and `eor` do not set flags, so a flag-setting `ands` of the result follows.
        let kind = match alu_op {
            ALU_ADD => {
                self.asm.adds(0, 0, 1, is64);
                FlagKind::Add
            }
            ALU_OR => {
                self.asm.orr(0, 0, 1, is64);
                self.asm.ands(XZR, 0, 0, is64);
                FlagKind::Logic
            }
            ALU_AND | ALU_TEST => {
                self.asm.ands(0, 0, 1, is64);
                FlagKind::Logic
            }
            ALU_SUB | ALU_CMP => {
                self.asm.subs(0, 0, 1, is64);
                FlagKind::Sub
            }
            _ => {
                debug_assert_eq!(alu_op, ALU_XOR);
                self.asm.eor(0, 0, 1, is64);
                self.asm.ands(XZR, 0, 0, is64);
                FlagKind::Logic
            }
        };
        // A 32-bit result written to a register zero-extends, which the W-form op already did.
        if alu_op != ALU_CMP
            && alu_op != ALU_TEST
            && let Dst::Reg(reg) = dst
        {
            self.store_gpr(reg, 0);
        }
        self.store_lazy_op(kind, operand_size);
    }

    /// An immediate as the operand of a 32- or 64-bit operation: sign-extended to 64 bits
    /// only for 64-bit operations.
    fn immediate_for(operand_size: u8, imm: i64) -> u64 {
        if operand_size == 8 {
            imm as u64
        } else {
            imm as u32 as u64
        }
    }

    /// Conditional branch ending the block; both outcomes exit with `retired` instructions done.
    fn jcc(&mut self, condition: u8, target: u64, next_rip: u64, retired: u32) {
        let taken = self.asm.label();
        match self.live.and_then(|live| cond_map(live.kind, condition)) {
            Some(CondMap::Cond(cond)) => self.asm.b_cond(cond, taken),
            Some(CondMap::Always) => self.asm.b(taken),
            Some(CondMap::Never) => {}
            None => {
                // helper_cond(cpu, cc) -> 0 or 1, from the lazy flags in `Cpu`
                self.asm.mov(0, CPU);
                self.asm.mov_imm(1, condition as u64);
                self.call(self.addrs.cond);
                self.asm.cbnz(0, taken, true);
            }
        }
        self.set_rip(next_rip);
        self.exit_ok(retired);
        self.asm.bind(taken);
        self.set_rip(target);
        self.exit_ok(retired);
    }

    /// Emits instruction `index` inline and returns true, or emits nothing and returns false if
    /// it has no inline lowering (or an unsupported form), in which case the caller uses
    /// `helper`. Only 32- and 64-bit operand sizes are lowered; 8- and 16-bit forms need
    /// partial-register merging and go to the interpreter.
    fn try_inline(&mut self, index: usize, rip: u64, next_rip: u64) -> bool {
        let insn: Insn = self.insns[index];
        let insn_index = index as u32;
        let retired = insn_index + 1;
        let operand_size = insn.opsize;
        // No 67/66/LOCK prefix and a 32- or 64-bit operand size.
        let wide_operand =
            !insn.addr32 && !insn.lock && !insn.p66 && (operand_size == 4 || operand_size == 8);
        let is_memory = insn.md != MODRM_MOD_REGISTER;
        let memory_ok = Self::memory_inlinable(&insn);
        let has_rex = insn.rex_present;
        let mut is_store = false;
        match insn.op {
            // add/or/adc/sbb/and/sub/xor/cmp: forms r/m,r (1), r,r/m (3) and eax,imm32 (5);
            // the byte forms (0, 2, 4) fall back.
            0x00..=0x3f if insn.op & 7 < 6 && wide_operand => {
                let alu_op = ((insn.op >> 3) & 7) as u8;
                if alu_op == ALU_ADC || alu_op == ALU_SBB {
                    return false;
                }
                match insn.op & 7 {
                    1 => {
                        if is_memory && (alu_op != ALU_CMP || !memory_ok) {
                            return false;
                        }
                        let dst = if is_memory {
                            Dst::Mem
                        } else {
                            Dst::Reg(insn.rm)
                        };
                        self.alu(
                            &insn,
                            alu_op,
                            dst,
                            Src::Reg(insn.reg),
                            operand_size,
                            next_rip,
                            insn_index,
                            rip,
                        );
                    }
                    3 => {
                        if is_memory && !memory_ok {
                            return false;
                        }
                        let src = if is_memory {
                            Src::Mem
                        } else {
                            Src::Reg(insn.rm)
                        };
                        self.alu(
                            &insn,
                            alu_op,
                            Dst::Reg(insn.reg),
                            src,
                            operand_size,
                            next_rip,
                            insn_index,
                            rip,
                        );
                    }
                    5 => {
                        let immediate = Self::immediate_for(operand_size, insn.imm);
                        self.alu(
                            &insn,
                            alu_op,
                            Dst::Reg(0),
                            Src::Imm(immediate),
                            operand_size,
                            next_rip,
                            insn_index,
                            rip,
                        );
                    }
                    _ => return false,
                }
            }
            // group 1: add/or/.../cmp r/m, imm32 and imm8
            0x81 | 0x83 if wide_operand => {
                let alu_op = insn.reg & 7;
                if alu_op == ALU_ADC
                    || alu_op == ALU_SBB
                    || (is_memory && (alu_op != ALU_CMP || !memory_ok))
                {
                    return false;
                }
                let immediate = Self::immediate_for(operand_size, insn.imm);
                let dst = if is_memory {
                    Dst::Mem
                } else {
                    Dst::Reg(insn.rm)
                };
                self.alu(
                    &insn,
                    alu_op,
                    dst,
                    Src::Imm(immediate),
                    operand_size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // test r/m, r
            0x85 if wide_operand => {
                if is_memory && !memory_ok {
                    return false;
                }
                let dst = if is_memory {
                    Dst::Mem
                } else {
                    Dst::Reg(insn.rm)
                };
                self.alu(
                    &insn,
                    ALU_TEST,
                    dst,
                    Src::Reg(insn.reg),
                    operand_size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // test eax, imm32
            0xa9 if wide_operand => {
                let immediate = Self::immediate_for(operand_size, insn.imm);
                self.alu(
                    &insn,
                    ALU_TEST,
                    Dst::Reg(0),
                    Src::Imm(immediate),
                    operand_size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // group 3 /0: test r/m, imm32
            0xf7 if wide_operand && insn.reg & 7 == 0 => {
                if is_memory && !memory_ok {
                    return false;
                }
                let immediate = Self::immediate_for(operand_size, insn.imm);
                let dst = if is_memory {
                    Dst::Mem
                } else {
                    Dst::Reg(insn.rm)
                };
                self.alu(
                    &insn,
                    ALU_TEST,
                    dst,
                    Src::Imm(immediate),
                    operand_size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // mov r/m, r
            0x89 if wide_operand => {
                if is_memory {
                    if !memory_ok {
                        return false;
                    }
                    self.effective_address(&insn, next_rip, ADDRESS);
                    self.load_gpr(VALUE, insn.reg);
                    self.mem_store(operand_size, insn_index, rip);
                    is_store = true;
                } else {
                    self.load_gpr(VALUE, insn.reg);
                    if operand_size == 4 {
                        self.asm.uxtw(VALUE, VALUE);
                    }
                    self.store_gpr(insn.rm, VALUE);
                }
            }
            // mov r, r/m
            0x8b if wide_operand => {
                if is_memory {
                    if !memory_ok {
                        return false;
                    }
                    self.effective_address(&insn, next_rip, ADDRESS);
                    self.mem_load(VALUE, operand_size, false, insn_index, rip);
                } else {
                    self.load_gpr(VALUE, insn.rm);
                    if operand_size == 4 {
                        self.asm.uxtw(VALUE, VALUE);
                    }
                }
                self.store_gpr(insn.reg, VALUE);
            }
            // mov m8, r8 (register-destination byte moves fall back)
            0x88 if is_memory && memory_ok && !insn.p66 => {
                if !has_rex && (FIRST_HIGH_BYTE_REG..=LAST_HIGH_BYTE_REG).contains(&insn.reg) {
                    self.load_gpr(VALUE, insn.reg - FIRST_HIGH_BYTE_REG);
                    self.asm.lsr_imm(VALUE, VALUE, HIGH_BYTE_SHIFT);
                } else {
                    self.load_gpr(VALUE, insn.reg);
                }
                self.effective_address(&insn, next_rip, ADDRESS);
                self.mem_store(1, insn_index, rip);
                is_store = true;
            }
            // group 11 /0: mov r/m, imm32
            0xc7 if wide_operand && insn.reg & 7 == 0 => {
                let immediate = Self::immediate_for(operand_size, insn.imm);
                if is_memory {
                    if !memory_ok {
                        return false;
                    }
                    self.effective_address(&insn, next_rip, ADDRESS);
                    self.asm.mov_imm(VALUE, immediate);
                    self.mem_store(operand_size, insn_index, rip);
                    is_store = true;
                } else {
                    self.asm.mov_imm(VALUE, immediate);
                    self.store_gpr(insn.rm, VALUE);
                }
            }
            // mov r, imm32/imm64
            0xb8..=0xbf if wide_operand => {
                let reg = opcode_register(&insn);
                self.asm
                    .mov_imm(VALUE, Self::immediate_for(operand_size, insn.imm));
                self.store_gpr(reg, VALUE);
            }
            // lea: the segment base is not part of the result, so FS/GS forms fall back
            0x8d if wide_operand && is_memory => {
                if has_segment_base(&insn) {
                    return false;
                }
                self.effective_address(&insn, next_rip, ADDRESS);
                if operand_size == 4 {
                    self.asm.uxtw(ADDRESS, ADDRESS);
                }
                self.store_gpr(insn.reg, ADDRESS);
            }
            // movsxd r64, r/m32 (REX.W only)
            0x63 if operand_size == 8
                && !insn.addr32
                && !insn.lock
                && !insn.p66
                && (!is_memory || memory_ok) =>
            {
                if is_memory {
                    self.effective_address(&insn, next_rip, ADDRESS);
                    self.mem_load(VALUE, 4, true, insn_index, rip);
                } else {
                    self.load_gpr(VALUE, insn.rm);
                    self.asm.sxtw(VALUE, VALUE);
                }
                self.store_gpr(insn.reg, VALUE);
            }
            // movzx/movsx r, r/m8 and r/m16 (0f b6, 0f b7, 0f be, 0f bf)
            0x1b6 | 0x1b7 | 0x1be | 0x1bf if wide_operand && (!is_memory || memory_ok) => {
                let source_size: u8 = if insn.op & 1 == 0 { 1 } else { 2 };
                let signed = insn.op >= 0x1b8;
                if is_memory {
                    self.effective_address(&insn, next_rip, ADDRESS);
                    self.mem_load(VALUE, source_size, signed, insn_index, rip);
                } else {
                    if source_size == 1
                        && !has_rex
                        && (FIRST_HIGH_BYTE_REG..=LAST_HIGH_BYTE_REG).contains(&insn.rm)
                    {
                        return false;
                    }
                    self.load_gpr(VALUE, insn.rm);
                    match (source_size, signed) {
                        (1, false) => self.asm.uxtb(VALUE, VALUE),
                        (2, false) => self.asm.uxth(VALUE, VALUE),
                        (1, true) => self.asm.sxtb(VALUE, VALUE, true),
                        _ => self.asm.sxth(VALUE, VALUE, true),
                    }
                }
                if operand_size == 4 {
                    self.asm.uxtw(VALUE, VALUE);
                }
                self.store_gpr(insn.reg, VALUE);
            }
            // push r64. rsp is written back only after the store, so a faulting push leaves it
            // unchanged; the value is read from the register file, so `push rsp` pushes the old rsp.
            0x50..=0x57 if !insn.p66 => {
                let reg = opcode_register(&insn);
                self.load_gpr(ADDRESS, RSP);
                self.asm.sub_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.load_gpr(VALUE, reg);
                self.mem_store(8, insn_index, rip);
                self.store_gpr(RSP, ADDRESS);
                is_store = true;
            }
            // pop r64. rsp is stored before the destination so `pop rsp` ends up with the
            // popped value.
            0x58..=0x5f if !insn.p66 => {
                let reg = opcode_register(&insn);
                self.load_gpr(ADDRESS, RSP);
                self.mem_load(VALUE, 8, false, insn_index, rip);
                self.asm.add_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.store_gpr(RSP, ADDRESS);
                self.store_gpr(reg, VALUE);
            }
            // nop, prefetchw, prefetch/hint-nop group (`xchg r8, rax` with REX.B is not a nop)
            0x90 | 0x10d | 0x118..=0x11f => {
                if insn.op == 0x90 && insn.rex & REX_B != 0 {
                    return false;
                }
            }
            // jcc rel8 / rel32
            0x70..=0x7f | 0x180..=0x18f => {
                let target = next_rip.wrapping_add(insn.imm as u64);
                self.jcc((insn.op & 15) as u8, target, next_rip, retired);
                self.inlined += 1;
                return true;
            }
            // jmp rel32 / rel8
            0xe9 | 0xeb => {
                self.set_rip(next_rip.wrapping_add(insn.imm as u64));
                self.exit_ok(retired);
                self.inlined += 1;
                return true;
            }
            // call rel32. The block ends here, so no `smc_check` is needed after the push.
            0xe8 => {
                self.load_gpr(ADDRESS, RSP);
                self.asm.sub_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.asm.mov_imm(VALUE, next_rip);
                self.mem_store(8, insn_index, rip);
                self.store_gpr(RSP, ADDRESS);
                self.set_rip(next_rip.wrapping_add(insn.imm as u64));
                self.exit_ok(retired);
                self.inlined += 1;
                return true;
            }
            // ret
            0xc3 => {
                self.load_gpr(ADDRESS, RSP);
                self.mem_load(VALUE, 8, false, insn_index, rip);
                self.asm.add_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.store_gpr(RSP, ADDRESS);
                self.asm.str(VALUE, CPU, RIP_OFFSET);
                self.exit_ok(retired);
                self.inlined += 1;
                return true;
            }
            _ => return false,
        }
        if is_store {
            self.smc_check(insn_index, next_rip);
        }
        self.inlined += 1;
        true
    }

    /// Compiles the whole block. `rips[k]` is the guest address of instruction `k` and
    /// `end_rip` the address after the last one. A block ends after a terminator (see
    /// `ends_block`), or by falling off the end (instruction limit or page boundary), in which case
    /// it exits to `end_rip`.
    pub fn compile(&mut self, rips: &[u64], end_rip: u64) {
        let count = self.insns.len();
        for index in 0..count {
            let rip = rips[index];
            let next_rip = if index + 1 < count {
                rips[index + 1]
            } else {
                end_rip
            };
            let terminal = ends_block(&self.insns[index]);
            if self.try_inline(index, rip, next_rip) {
                if is_inline_terminator(&self.insns[index]) {
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

/// The instructions `try_inline` lowers that end the block themselves (they emit their own
/// exit): jcc, jmp, call, ret. All of them are also `ends_block`.
fn is_inline_terminator(insn: &Insn) -> bool {
    matches!(
        insn.op,
        0x70..=0x7f | 0x180..=0x18f | 0xe9 | 0xeb | 0xe8 | 0xc3
    )
}
