//! aarch64 glue for `native.rs`: the entry/exit trampoline, helper addresses and the block
//! compiler entry point. The x86_64 host has the same three functions in `arch_x64.rs`.

use std::mem::offset_of;

use super::a64::{self, Asm, NOP_WORD};
use super::emit::{
    CALL_TARGET, CPU, Gen, SAVED_ADDRESS, SAVED_INFO, SAVED_LR, SAVED_NZCV, SAVED_OPERAND,
    SAVED_RESULT, TLB_BASE, VALUE,
};
use super::helpers::{self, EXIT_COUNT_SHIFT, STATUS_EXIT_REQUESTED};
use super::native::{
    CompiledBlock, JUMP_ENTRY_CODE_OFFSET, JUMP_ENTRY_PHYS_OFFSET, JUMP_ENTRY_SHIFT,
    jump_cache_mask,
};
use crate::cpu::decode::Insn;
use crate::cpu::{Cpu, TLB_ENTRIES_PER_MODE, TlbEntry};

pub(crate) use super::emit::Addrs;

/// The trampoline placed at the start of the code buffer, followed by the stubs and thunks
/// every block reaches with a direct branch. See the module docs of `emit.rs`.
pub(crate) struct Prologue {
    pub code: Vec<u8>,
    addrs: Addrs,
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const FRAME_POINTER: u32 = 29;
const LINK_REGISTER: u32 = 30;
/// 16-byte aligned frame: fp/lr, then the callee-saved registers the blocks use. Those are x19 and
/// x20 (`CPU`, `TLB_BASE`), x22 and x23 (slow-path state), x24..x26 (values that must survive a
/// slow-path helper call) and x27/x28 (the memory thunks' info pointer and return address, see
/// `emit.rs`); nothing uses x21, so it is left alone.
const FRAME_SIZE: i32 = 96;
const SAVE_X19_X20: i32 = 16;
const SAVE_X22_X23: i32 = 32;
const SAVE_X24_X25: i32 = 48;
const SAVE_X26_X27: i32 = 64;
const SAVE_X28: i32 = 80;

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
    asm.stp_off(26, 27, a64::SP, SAVE_X26_X27);
    asm.str(28, a64::SP, SAVE_X28 as u32);
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
    asm.ldr(28, a64::SP, SAVE_X28 as u32);
    asm.ldp_off(26, 27, a64::SP, SAVE_X26_X27);
    asm.ldp_off(24, 25, a64::SP, SAVE_X24_X25);
    asm.ldp_off(22, 23, a64::SP, SAVE_X22_X23);
    asm.ldp_off(19, 20, a64::SP, SAVE_X19_X20);
    asm.ldp_post(FRAME_POINTER, LINK_REGISTER, a64::SP, FRAME_SIZE);
    asm.ret();
    let chain = chain_stub(&mut asm, jump_cache_base, exit_ok);
    let addrs = Addrs {
        exec: tail_call_stub(
            &mut asm,
            helpers::helper_exec_pair as *const () as usize as u64,
        ),
        cond: tail_call_stub(&mut asm, helpers::helper_cond as *const () as usize as u64),
        rotate_flags: tail_call_stub(
            &mut asm,
            helpers::helper_set_rotate_flags as *const () as usize as u64,
        ),
        exit_ok,
        exit_fail,
        chain,
        load: SIZES.map(|size| {
            thunk(
                &mut asm,
                Access::Load {
                    size,
                    signed: false,
                },
                exit_fail,
            )
        }),
        load_signed: [1u8, 2, 4]
            .map(|size| thunk(&mut asm, Access::Load { size, signed: true }, exit_fail)),
        load_stack: thunk(&mut asm, Access::LoadStack, exit_fail),
        store: SIZES.map(|size| thunk(&mut asm, Access::Store { size, keep: false }, exit_fail)),
        store_keep: SIZES
            .map(|size| thunk(&mut asm, Access::Store { size, keep: true }, exit_fail)),
        store_stack: thunk(&mut asm, Access::StoreStack, exit_fail),
    };
    asm.finish();
    Prologue {
        code: bytes(&asm.code),
        addrs,
    }
}

/// Operand sizes of the memory thunks, indexed by `Gen::size_index`.
const SIZES: [u8; 4] = [1, 2, 4, 8];

/// A stub that tail-calls a Rust helper, so a block reaches it with a one-instruction `bl`
/// instead of materializing a 64-bit address: the helper returns to the block.
fn tail_call_stub(asm: &mut Asm, helper: u64) -> u64 {
    let start = asm.here();
    asm.mov_imm(CALL_TARGET, helper);
    asm.br(CALL_TARGET);
    start
}

#[derive(Clone, Copy)]
enum Access {
    Load { size: u8, signed: bool },
    LoadStack,
    Store { size: u8, keep: bool },
    StoreStack,
}

/// The out-of-line half of an inline memory access, shared by every block: what a block's slow
/// path does is `adr x7, info; bl thunk; mov dst, x0; b resume`, where `info` is the
/// instruction's `[rip, block index]` pair next to the block. The thunk calls the Rust helper
/// with the address in x2 (and the value in `VALUE` for stores), preserving the host flags,
/// the address and (for read-modify-write stores, `keep`) x0 and x1 across the call, and
/// returns to the block with the loaded value in x0. If the helper reports an exception the
/// thunk leaves the block through `exit_fail` instead, with the status and the instruction's
/// index. All of that used to be emitted per access, which made blocks several times bigger.
fn thunk(asm: &mut Asm, access: Access, exit_fail: u64) -> u64 {
    const ADDRESS: u32 = 2;
    const INFO: u32 = 7;
    const LOAD_STATUS: u32 = 1;
    let start = asm.here();
    asm.mrs_nzcv(SAVED_NZCV);
    asm.mov(SAVED_ADDRESS, ADDRESS);
    asm.mov(SAVED_INFO, INFO);
    asm.mov(SAVED_LR, LINK_REGISTER);
    // Tell the clock where in the block this access is (`Cpu::slow_access_insn`, index + 1);
    // the helper clears it. CALL_TARGET is a scratch until the call below.
    asm.ldr(CALL_TARGET, INFO, 8);
    asm.add_imm(CALL_TARGET, CALL_TARGET, 1, true);
    asm.str(CALL_TARGET, CPU, offset_of!(Cpu, slow_access_insn) as u32);
    let fail = asm.label();
    let (helper, size, is_load, keep) = match access {
        Access::Load { size, .. } => (
            helpers::helper_load as *const () as usize as u64,
            size,
            true,
            false,
        ),
        Access::LoadStack => (
            helpers::helper_load_stack as *const () as usize as u64,
            8,
            true,
            false,
        ),
        Access::Store { size, keep } => (
            helpers::helper_store as *const () as usize as u64,
            size,
            false,
            keep,
        ),
        Access::StoreStack => (
            helpers::helper_store_stack as *const () as usize as u64,
            8,
            false,
            false,
        ),
    };
    if keep {
        asm.mov(SAVED_RESULT, 0);
        asm.mov(SAVED_OPERAND, 1);
    }
    if is_load {
        // helper_load(cpu, address, size, rip) -> (value in x0, status in x1)
        asm.mov(1, ADDRESS);
        asm.mov(0, CPU);
        asm.mov_imm(2, size as u64);
        asm.ldr(3, INFO, 0);
        asm.mov_imm(CALL_TARGET, helper);
        asm.blr(CALL_TARGET);
        asm.cbnz(LOAD_STATUS, fail, true);
        if let Access::Load { size, signed: true } = access {
            match size {
                1 => asm.sxtb(0, 0, true),
                2 => asm.sxth(0, 0, true),
                _ => asm.sxtw(0, 0),
            }
        }
    } else {
        // helper_store(cpu, address, size, value, rip) -> status
        asm.mov(3, VALUE);
        asm.mov(1, ADDRESS);
        asm.mov_imm(2, size as u64);
        asm.ldr(4, INFO, 0);
        asm.mov(0, CPU);
        asm.mov_imm(CALL_TARGET, helper);
        asm.blr(CALL_TARGET);
        asm.cbnz(0, fail, true);
    }
    asm.msr_nzcv(SAVED_NZCV);
    asm.mov(ADDRESS, SAVED_ADDRESS);
    if keep {
        asm.mov(0, SAVED_RESULT);
        asm.mov(1, SAVED_OPERAND);
    }
    asm.ret_reg(SAVED_LR);
    asm.bind(fail);
    if is_load {
        // `exit_fail` wants the status in x0.
        asm.mov(0, LOAD_STATUS);
    }
    asm.ldr(1, SAVED_INFO, 8);
    asm.b_abs(exit_fail);
    start
}

/// Registers of the chain stub's calling convention (see `Gen::chain_exit`): the block
/// has stored `Cpu::icount`, and x1 holds the guest rip to continue at.
const CHAIN_ICOUNT: u32 = 0;
const CHAIN_RIP: u32 = 1;

/// Block-to-block transfer, the equivalent of QEMU's `lookup_and_goto_ptr`. Entered with
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
    asm.ldr(CHAIN_ICOUNT, CPU, offset_of!(Cpu, icount) as u32);
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

/// The addresses blocks are compiled against. Without `chaining` the chain address is 0 and
/// every control transfer returns to the run loop.
pub(crate) fn addrs(prologue: &Prologue, chaining: bool) -> Addrs {
    let mut addrs = prologue.addrs.clone();
    if !chaining {
        addrs.chain = 0;
    }
    addrs
}

/// Compiles one block to machine code meant to run at `addr` (the encoder bakes in pc-relative
/// branches to the exit stubs).
pub(crate) fn compile_block(
    addr: u64,
    addrs: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
    ordered: bool,
) -> CompiledBlock {
    let mut generator = Gen::new(addr, addrs, insns);
    generator.ordered_memory = ordered;
    generator.compile(rips, end_rip);
    CompiledBlock {
        code: bytes(&generator.asm.code),
        inlined: generator.inlined,
        body_offset: generator.body_offset,
        links: generator.links,
        flag_records_skipped: generator.flag_records_skipped,
    }
}

const BRANCH_OPCODE: u32 = 0x1400_0000;
const BRANCH_IMM26_MASK: u32 = 0x03ff_ffff;
const INSN_BYTES: i64 = 4;
/// `b` reaches +-2^25 instructions.
const BRANCH_REACH_INSNS: i64 = 1 << 25;

/// The instruction that makes the slot at `site` jump to `destination`, or `None` when the two
/// are further apart than a `b` reaches (blocks in different code segments).
pub(crate) fn link_bytes(site: u64, destination: u64) -> Option<Vec<u8>> {
    let offset = (destination as i64 - site as i64) / INSN_BYTES;
    if offset.abs() >= BRANCH_REACH_INSNS {
        return None;
    }
    Some(
        (BRANCH_OPCODE | (offset as u32 & BRANCH_IMM26_MASK))
            .to_le_bytes()
            .to_vec(),
    )
}

/// What an unlinked slot holds: a nop, so execution falls into the chain stub sequence after it.
pub(crate) fn unlink_bytes() -> Vec<u8> {
    NOP_WORD.to_le_bytes().to_vec()
}
