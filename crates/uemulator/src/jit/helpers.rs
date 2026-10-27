//! `extern "C"` entry points that generated code calls back into, plus the exit-code encoding
//! shared with the emitters (`emit.rs`, `emit_x64.rs`) and the prologues (`arch_a64.rs`,
//! `arch_x64.rs`). The addresses are handed to the emitters through `Addrs` in `arch::addrs`.
//! Argument order and count are hard-coded in the call sequences of `emit.rs` (`helper`,
//! `emit_slows`) and `emit_x64.rs`: changing a signature here means updating those.

use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

/// Helper return value / low byte of the block exit code: nothing special happened.
pub(crate) const STATUS_OK: u64 = 0;
/// The helper stored a guest exception in `Cpu::pending`; the faulting instruction did not
/// retire. `Cpu::step_block` takes the exception out of `pending`.
pub(crate) const STATUS_EXCEPTION: u64 = 1;
/// The instruction retired but wrote to a page that held translated code (`Cpu::code_written`
/// set `block_exit_requested`), so the rest of the block may be stale and must not run.
/// Returned by `helper_exec` only; the `exit_fail` stub counts such an instruction as retired.
pub(crate) const STATUS_EXIT_REQUESTED: u64 = 2;

/// Block exit code layout, produced by the exit stubs in the prologue and decoded by
/// `Cpu::step_block`: `(retired_insns << EXIT_COUNT_SHIFT) | status`. `emit.rs`/`emit_x64.rs`
/// build the success form with `mov x0/rax, n << 8` and jump to `exit_ok`; the `exit_fail` stub
/// takes the helper status in x0/rax and the failing instruction's block index k in x1/rcx and
/// retires k instructions, or k + 1 for `STATUS_EXIT_REQUESTED`. The stubs hard-code 2 and 8.
pub(crate) const EXIT_COUNT_SHIFT: u32 = 8;
pub(crate) const EXIT_STATUS_MASK: u64 = (1 << EXIT_COUNT_SHIFT) - 1;

/// `Insn::op` for bytes that failed to decode, produced by `Jit::translate`. Executes as #UD.
pub(crate) const OP_INVALID: u16 = 0xffff;

/// aarch64 load result. Two `u64`s are returned in x0/x1 (AAPCS64), so `emit.rs` reads the value
/// from x0 and the status from x1.
#[cfg(target_arch = "aarch64")]
#[repr(C)]
pub(crate) struct LoadResult {
    pub val: u64,
    pub status: u64,
}

/// Runs one instruction through the interpreter. `insn` points into the owning `Block`.
/// `rip` is set first because `Cpu::exec` computes the next rip from `cpu.rip`; on success exec
/// leaves `cpu.rip` at the next instruction (or the branch target), so the caller stores nothing.
pub(crate) extern "C" fn helper_exec(cpu: &mut Cpu, insn: &Insn, rip: u64) -> u64 {
    cpu.rip = rip;
    match cpu.exec(insn) {
        Ok(()) => {
            if cpu.block_exit_requested {
                STATUS_EXIT_REQUESTED
            } else {
                STATUS_OK
            }
        }
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

/// Slow-path load, called when the inline TLB probe misses. Generated code does not keep
/// `cpu.rip` current, so the faulting instruction's `rip` is passed in and stored here.
#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_load(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    rip: u64,
) -> LoadResult {
    cpu.rip = rip;
    match cpu.read(linear_addr, size as u8) {
        Ok(value) => LoadResult {
            val: value,
            status: STATUS_OK,
        },
        Err(exception) => {
            cpu.pending = Some(exception);
            LoadResult {
                val: 0,
                status: STATUS_EXCEPTION,
            }
        }
    }
}

/// Slow-path store; returns `STATUS_OK` or `STATUS_EXCEPTION`, never `STATUS_EXIT_REQUESTED`: the
/// emitters poll `block_exit_requested` themselves (`smc_check`) after an inlined instruction
/// that stores, including when the store went through this helper.
#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_store(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    value: u64,
    rip: u64,
) -> u64 {
    cpu.rip = rip;
    match cpu.write(linear_addr, size as u8, value) {
        Ok(()) => STATUS_OK,
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

// x86-64 variants: at most four arguments (Win64 passes more on the stack) and no 16-byte
// struct return (Win64 returns it through a hidden pointer). The JIT stores `cpu.rip` itself.
/// `out` points at the 8-byte result slot in the block's stack frame (`OUT` in `emit_x64.rs`).
#[cfg(target_arch = "x86_64")]
pub(crate) extern "C" fn helper_load_via_out(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    out: *mut u64,
) -> u64 {
    match cpu.read(linear_addr, size as u8) {
        Ok(value) => {
            unsafe { *out = value };
            STATUS_OK
        }
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) extern "C" fn helper_store_x64(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    value: u64,
) -> u64 {
    match cpu.write(linear_addr, size as u8, value) {
        Ok(()) => STATUS_OK,
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

/// Evaluates condition code `cc` (low four bits of the Jcc/SETcc/CMOVcc opcode) against the
/// guest flags. Used when the emitter cannot test host flags directly.
pub(crate) extern "C" fn helper_cond(cpu: &mut Cpu, cc: u64) -> u64 {
    cpu.condition_holds(cc as u8) as u64
}

/// True if the block must end after `insn`. The set is an allow-list of what the block compilers
/// cannot continue past: control transfers; string and I/O instructions (they loop on `rep` or
/// have device side effects); and instructions that change interrupt delivery, segmentation,
/// privilege, paging or other control state that the run loop must observe before the next
/// instruction. Opcodes at or above 0x100 are `0f xx` (`decode` puts the opcode map in bits 8+).
pub(crate) fn ends_block(insn: &Insn) -> bool {
    match insn.op {
        // jcc rel8, loopnz/loopz/loop/jcxz, call rel32, jmp rel32/rel8, ret near/far (with and
        // without imm16), int3, int imm8, iret, hlt
        0x70..=0x7f
        | 0xe0..=0xe3
        | 0xe8
        | 0xe9
        | 0xeb
        | 0xc2
        | 0xc3
        | 0xca
        | 0xcb
        | 0xcc
        | 0xcd
        | 0xcf
        | 0xf4 => true,
        // ins/outs, movs/cmps, stos/lods/scas, in/out imm8, in/out dx, cli, sti, popf,
        // mov sreg, enter
        0x6c..=0x6f
        | 0xa4..=0xa7
        | 0xaa..=0xaf
        | 0xe4..=0xe7
        | 0xec..=0xef
        | 0xfa
        | 0xfb
        | 0x9d
        | 0x8e
        | 0xc8 => true,
        // group 5: call/jmp indirect, near and far
        0xff => matches!(insn.reg & 7, 2..=5),
        // syscall, clts, sysret, ud2, jcc rel32, pop fs, pop gs, mov to/from cr/dr, wrmsr,
        // groups 6 and 7 (lldt/ltr/lgdt/lidt/lmsw/invlpg/...)
        0x105..=0x107
        | 0x10b
        | 0x180..=0x18f
        | 0x1a1
        | 0x1a9
        | 0x120..=0x123
        | 0x130
        | 0x100
        | 0x101
        | OP_INVALID => true,
        _ => false,
    }
}
