pub(crate) use super::emit_x64::Addrs;
use super::emit_x64::{FRAME, Gen};
use super::helpers::{self, EXIT_COUNT_SHIFT, STATUS_EXIT_REQUESTED};
use super::x64::*;
use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

/// Entry/exit stubs placed at the start of the code buffer, shared by all blocks.
pub(crate) struct Prologue {
    pub code: Vec<u8>,
    /// Absolute address of the success exit (rax already holds the exit code).
    pub exit_ok: u64,
    /// Absolute address of the failure exit (rax = helper status, rcx = failing instruction index).
    pub exit_fail: u64,
}

/// Pushed in this order and popped in reverse. This is the Win64 callee-saved set (RSI/RDI are
/// callee-saved there and blocks use them as scratch); on System V the extra saves are harmless.
const SAVED_REGS: [u8; 8] = [RBX, RBP, RSI, RDI, R12, R13, R14, R15];

/// Builds `enter(cpu: *mut Cpu, code: u64) -> exit_code`, which `Cpu::step_block` calls through a
/// function pointer: it sets up the frame, pins the `Cpu` pointer and TLB base for the block (see
/// `emit_x64`), then jumps to `code`. Blocks never return themselves; they jump to `exit_ok` or
/// `exit_fail` below.
pub(crate) fn prologue(base: u64, _jump_cache_base: u64) -> Prologue {
    let mut a = Asm::new(base);
    for r in SAVED_REGS {
        a.push(r);
    }
    a.alu_ri(SUB, RSP, FRAME, true);
    a.mov_rr(RBX, ARGS[0], true);
    a.ld(
        RBP,
        mem(RBX, std::mem::offset_of!(Cpu, tlb_active_base) as i32),
    );
    a.jmp_reg(ARGS[1]);
    // rax = helper status, rcx = index of the failing instruction. Fold into the exit code
    // `(retired << EXIT_COUNT_SHIFT) | status`: an exception means rcx instructions retired, but
    // STATUS_EXIT_REQUESTED means the instruction itself retired too. The xor runs before the cmp
    // because it clobbers flags and sete only writes the low byte.
    let exit_fail = a.here();
    a.alu_rr(XOR, RDX, RDX, false);
    a.alu_ri(CMP, RAX, STATUS_EXIT_REQUESTED as i32, true);
    a.sete(RDX);
    a.alu_rr(ADD, RCX, RDX, true);
    a.shift_ri(SHL, RCX, EXIT_COUNT_SHIFT as u8, true);
    a.alu_rr(OR, RAX, RCX, true);
    let exit_ok = a.here();
    a.alu_ri(ADD, RSP, FRAME, true);
    for r in SAVED_REGS.iter().rev() {
        a.pop(*r);
    }
    a.ret();
    a.finish();
    Prologue {
        code: a.code,
        exit_ok,
        exit_fail,
    }
}

pub(crate) fn addrs(prologue: &Prologue, _chaining: bool) -> Addrs {
    Addrs {
        exec: helpers::helper_exec as *const () as usize as u64,
        ld: helpers::helper_load_via_out as *const () as usize as u64,
        st: helpers::helper_store_x64 as *const () as usize as u64,
        ld_stack: helpers::helper_load_stack_via_out as *const () as usize as u64,
        st_stack: helpers::helper_store_stack_x64 as *const () as usize as u64,
        cond: helpers::helper_cond as *const () as usize as u64,
        exit_ok: prologue.exit_ok,
        exit_fail: prologue.exit_fail,
    }
}

/// Generates host code for one block placed at `addr`, which must be its final address
/// (`jmp_abs` encodes displacements relative to it). Returns the code and the number of
/// instructions lowered inline. `insns` must outlive the code, which points into it.
pub(crate) fn compile_block(
    addr: u64,
    addrs: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
) -> (Vec<u8>, u32) {
    let mut generator = Gen::new(addr, addrs, insns);
    generator.compile(rips, end_rip);
    (generator.a.code, generator.inlined)
}
