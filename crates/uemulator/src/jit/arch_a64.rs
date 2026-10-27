//! aarch64 glue for `native.rs`: the entry/exit trampoline, helper addresses and the block
//! compiler entry point. The x86_64 host has the same three functions in `arch_x64.rs`.

use super::a64::{self, Asm};
use super::emit::{CPU, Gen, TLB_BASE};
use super::helpers::{self, EXIT_COUNT_SHIFT, STATUS_EXIT_REQUESTED};
use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

pub(crate) use super::emit::Addrs;

/// The trampoline placed at the start of the code buffer, and the addresses of its two exit
/// stubs. See the module docs of `emit.rs` for how blocks use them.
pub(crate) struct Prologue {
    pub code: Vec<u8>,
    pub exit_ok: u64,
    pub exit_fail: u64,
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const FRAME_POINTER: u32 = 29;
const LINK_REGISTER: u32 = 30;
/// 16-byte aligned frame: fp/lr, then the callee-saved registers the blocks use. Those are x19 and
/// x20 (`CPU`, `TLB_BASE`) and x22 and x23 (slow-path state, see `emit.rs`); nothing uses x21 or
/// x24 and above, so they are left alone.
const FRAME_SIZE: i32 = 48;
const SAVE_X19_X20: i32 = 16;
const SAVE_X22_X23: i32 = 32;

/// Builds `extern "C" fn(cpu: *mut Cpu, block: u64) -> u64` (the call in `Cpu::step_block`).
/// It saves the callee-saved registers the blocks use, latches `CPU` and `TLB_BASE`, and jumps
/// into the block. Blocks call helpers with `blr`, which overwrites x30, so the return address
/// is saved in the frame and the stubs return through the restored x30.
pub(crate) fn prologue(base: u64) -> Prologue {
    let mut asm = Asm::new(base);
    asm.stp_pre(FRAME_POINTER, LINK_REGISTER, a64::SP, -FRAME_SIZE);
    asm.stp_off(19, 20, a64::SP, SAVE_X19_X20);
    asm.stp_off(22, 23, a64::SP, SAVE_X22_X23);
    asm.mov(CPU, 0);
    asm.ldr(
        TLB_BASE,
        CPU,
        std::mem::offset_of!(Cpu, tlb_active_base) as u32,
    );
    asm.br(1);
    // Block exit with a helper status in x0 and the stopping instruction's index in x1: the
    // exit code is `status | retired << EXIT_COUNT_SHIFT`, where an instruction that
    // completed (`STATUS_EXIT_REQUESTED`) counts as retired and one that faulted does not.
    let exit_fail = asm.here();
    asm.cmp_imm(0, STATUS_EXIT_REQUESTED as u32, true);
    asm.cinc_eq(1, 1);
    asm.lsl_imm(1, 1, EXIT_COUNT_SHIFT);
    asm.orr(0, 0, 1, true);
    // Normal block exit: x0 already holds the finished exit code.
    let exit_ok = asm.here();
    asm.ldp_off(22, 23, a64::SP, SAVE_X22_X23);
    asm.ldp_off(19, 20, a64::SP, SAVE_X19_X20);
    asm.ldp_post(FRAME_POINTER, LINK_REGISTER, a64::SP, FRAME_SIZE);
    asm.ret();
    asm.finish();
    Prologue {
        code: bytes(&asm.code),
        exit_ok,
        exit_fail,
    }
}

pub(crate) fn addrs(exit_ok: u64, exit_fail: u64) -> Addrs {
    Addrs {
        exec: helpers::helper_exec as *const () as usize as u64,
        ld: helpers::helper_load as *const () as usize as u64,
        st: helpers::helper_store as *const () as usize as u64,
        ld_stack: helpers::helper_load_stack as *const () as usize as u64,
        st_stack: helpers::helper_store_stack as *const () as usize as u64,
        cond: helpers::helper_cond as *const () as usize as u64,
        exit_ok,
        exit_fail,
    }
}

/// Compiles one block to machine code meant to run at `addr` (the encoder bakes in pc-relative
/// branches to the exit stubs). Returns the code and the number of inlined instructions.
pub(crate) fn compile_block(
    addr: u64,
    addrs: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
) -> (Vec<u8>, u32) {
    let mut generator = Gen::new(addr, addrs, insns);
    generator.compile(rips, end_rip);
    (bytes(&generator.asm.code), generator.inlined)
}
