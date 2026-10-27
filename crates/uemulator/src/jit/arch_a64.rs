//! aarch64 glue for `native.rs`: the entry/exit trampoline, helper addresses and the block
//! compiler entry point. The x86_64 host has the same three functions in `arch_x64.rs`.

use std::mem::offset_of;

use super::a64::{self, Asm};
use super::emit::{CPU, Gen, TLB_BASE};
use super::helpers::{self, EXIT_COUNT_SHIFT, STATUS_EXIT_REQUESTED};
use super::native::{
    JUMP_ENTRY_CODE_OFFSET, JUMP_ENTRY_PHYS_OFFSET, JUMP_ENTRY_SHIFT, jump_cache_mask,
};
use crate::cpu::decode::Insn;
use crate::cpu::{Cpu, TLB_ENTRIES_PER_MODE, TlbEntry};

pub(crate) use super::emit::Addrs;

/// The trampoline placed at the start of the code buffer, and the addresses of its two exit
/// stubs. See the module docs of `emit.rs` for how blocks use them.
pub(crate) struct Prologue {
    pub code: Vec<u8>,
    pub exit_ok: u64,
    pub exit_fail: u64,
    pub chain: u64,
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const FRAME_POINTER: u32 = 29;
const LINK_REGISTER: u32 = 30;
/// 16-byte aligned frame: fp/lr, then the callee-saved registers the blocks use. Those are x19 and
/// x20 (`CPU`, `TLB_BASE`), x22 and x23 (slow-path state) and x24..x26 (values that must survive
/// a slow-path helper call, see `emit.rs`); nothing uses x21 or x27 and above, so they are left
/// alone.
const FRAME_SIZE: i32 = 80;
const SAVE_X19_X20: i32 = 16;
const SAVE_X22_X23: i32 = 32;
const SAVE_X24_X25: i32 = 48;
const SAVE_X26: i32 = 64;

/// Builds `extern "C" fn(cpu: *mut Cpu, block: u64) -> u64` (the call in `Cpu::step_block`).
/// It saves the callee-saved registers the blocks use, latches `CPU` and `TLB_BASE`, and jumps
/// into the block. Blocks call helpers with `blr`, which overwrites x30, so the return address
/// is saved in the frame and the stubs return through the restored x30.
pub(crate) fn prologue(base: u64, jump_cache_base: u64) -> Prologue {
    let mut asm = Asm::new(base);
    asm.stp_pre(FRAME_POINTER, LINK_REGISTER, a64::SP, -FRAME_SIZE);
    asm.stp_off(19, 20, a64::SP, SAVE_X19_X20);
    asm.stp_off(22, 23, a64::SP, SAVE_X22_X23);
    asm.stp_off(24, 25, a64::SP, SAVE_X24_X25);
    asm.str(26, a64::SP, SAVE_X26 as u32);
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
    asm.ldr(26, a64::SP, SAVE_X26 as u32);
    asm.ldp_off(24, 25, a64::SP, SAVE_X24_X25);
    asm.ldp_off(22, 23, a64::SP, SAVE_X22_X23);
    asm.ldp_off(19, 20, a64::SP, SAVE_X19_X20);
    asm.ldp_post(FRAME_POINTER, LINK_REGISTER, a64::SP, FRAME_SIZE);
    asm.ret();
    let chain = chain_stub(&mut asm, jump_cache_base, exit_ok);
    asm.finish();
    Prologue {
        code: bytes(&asm.code),
        exit_ok,
        exit_fail,
        chain,
    }
}

/// Registers of the chain stub's calling convention (see `Gen::chain_exit`).
const CHAIN_ICOUNT: u32 = 0;
const CHAIN_RIP: u32 = 1;

/// Block-to-block transfer, the equivalent of QEMU's `lookup_and_goto_ptr`. Entered with `x0` =
/// `Cpu::icount` as just stored by the block (already charged for the first instruction of the
/// successor, as `Cpu::run` does before `step_block`) and `x1` = the guest rip to continue at.
///
/// It jumps straight into the successor block when all of these hold, and otherwise leaves for
/// the run loop with `Cpu::rip` set, which then takes the normal path (event servicing,
/// interrupts, page walk, translation):
/// - `icount` has not reached `Cpu::next_event`, so the device tick and interrupt check the loop
///   does at block boundaries are not starved;
/// - `block_exit_requested` is clear: a store of the block just left (for example the push of a
///   `call`) hit a page holding translated code, and the successor may be stale;
/// - the exec TLB (the half `TLB_BASE` selects, which a block never leaves) has the page, so the
///   physical address is known without a walk and permissions were already checked;
/// - the jump cache has a block for exactly this rip and physical address, the same key as
///   `Jit::lookup`. Invalidation clears slots, so a hit is never stale.
///
/// The miss path gives back the successor's pre-charge: the loop adds it again.
fn chain_stub(asm: &mut Asm, jump_cache_base: u64, exit_ok: u64) -> u64 {
    const TLB_SLOT_MASK: u64 = (TLB_ENTRIES_PER_MODE - 1) as u64;
    const PAGE_SHIFT: u32 = 12;
    const PAGE_OFFSET_MASK: u64 = (1 << PAGE_SHIFT) - 1;
    const TLB_ENTRY_SHIFT: u32 = std::mem::size_of::<TlbEntry>().trailing_zeros();
    // `jump_cache_hash` in `native.rs`: rip ^ (rip >> bits), masked.
    const JUMP_HASH_FOLD_SHIFT: u32 = 16;
    let (entry, page, tag, phys, slot, key, miss) = (2, 3, 4, 5, 2, 4, asm.label());
    let start = asm.here();
    asm.ldr(entry, CPU, offset_of!(Cpu, next_event) as u32);
    asm.cmp(CHAIN_ICOUNT, entry, true);
    asm.b_cond(a64::CS, miss);
    asm.ldrb(entry, CPU, offset_of!(Cpu, block_exit_requested) as u32);
    asm.cbnz(entry, miss, false);
    asm.lsr_imm(entry, CHAIN_RIP, PAGE_SHIFT);
    assert!(asm.and_imm(entry, entry, TLB_SLOT_MASK, true));
    asm.add_lsl(entry, TLB_BASE, entry, TLB_ENTRY_SHIFT);
    assert!(asm.and_imm(page, CHAIN_RIP, !PAGE_OFFSET_MASK, true));
    asm.ldr(tag, entry, offset_of!(TlbEntry, tag_x) as u32);
    asm.cmp(page, tag, true);
    asm.b_cond(a64::NE, miss);
    // phys = rip + addend - ram_base
    asm.ldr(phys, entry, offset_of!(TlbEntry, addend) as u32);
    asm.add(phys, phys, CHAIN_RIP, true);
    asm.ldr(tag, CPU, offset_of!(Cpu, ram_base) as u32);
    asm.sub(phys, phys, tag, true);
    asm.lsr_imm(slot, CHAIN_RIP, JUMP_HASH_FOLD_SHIFT);
    asm.eor(slot, slot, CHAIN_RIP, true);
    assert!(asm.and_imm(slot, slot, jump_cache_mask(), true));
    asm.mov_imm(key, jump_cache_base);
    asm.add_lsl(slot, key, slot, JUMP_ENTRY_SHIFT);
    asm.ldr(key, slot, 0);
    asm.cmp(key, CHAIN_RIP, true);
    asm.b_cond(a64::NE, miss);
    asm.ldr(key, slot, JUMP_ENTRY_PHYS_OFFSET);
    asm.cmp(key, phys, true);
    asm.b_cond(a64::NE, miss);
    asm.ldr(key, slot, JUMP_ENTRY_CODE_OFFSET);
    asm.cbz(key, miss, true);
    asm.br(key);
    asm.bind(miss);
    asm.sub_imm(CHAIN_ICOUNT, CHAIN_ICOUNT, 1, true);
    asm.str(CHAIN_ICOUNT, CPU, offset_of!(Cpu, icount) as u32);
    asm.str(CHAIN_RIP, CPU, offset_of!(Cpu, rip) as u32);
    // One instruction retired in the exit code: `step_block` then adds nothing more.
    asm.mov_imm(0, 1 << EXIT_COUNT_SHIFT);
    asm.b_abs(exit_ok);
    start
}

pub(crate) fn addrs(exit_ok: u64, exit_fail: u64, chain: Option<u64>) -> Addrs {
    Addrs {
        exec: helpers::helper_exec as *const () as usize as u64,
        ld: helpers::helper_load as *const () as usize as u64,
        st: helpers::helper_store as *const () as usize as u64,
        ld_stack: helpers::helper_load_stack as *const () as usize as u64,
        st_stack: helpers::helper_store_stack as *const () as usize as u64,
        cond: helpers::helper_cond as *const () as usize as u64,
        exit_ok,
        exit_fail,
        chain: chain.unwrap_or(0),
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
