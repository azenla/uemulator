//! Block cache, jump cache, invalidation, code-buffer management and `Cpu::step_block`
//! accounting, tested against real translated blocks.

use super::*;
use crate::cpu::exception::Exception;
use crate::jit::serialize_jit_tests;
use crate::testkit::*;

const PAGE_BYTES_FOR_TESTS: u64 = PAGE_SIZE;
/// Page offset of the snippet in the harness (`CODE_BASE` is page-aligned plus this).
const SNIPPET_PAGE_OFFSET: u64 = CODE_BASE & PAGE_MASK;

// add rax, rcx; hlt
const ADD_THEN_HLT: [u8; 4] = [0x48, 0x01, 0xc8, 0xf4];
// add rax, rcx; jmp 1f; 1: hlt
const ADD_THEN_JMP_NEXT: [u8; 6] = [0x48, 0x01, 0xc8, 0xeb, 0x00, 0xf4];
// add rax, rcx; add rax, rcx; ud2
const ADD_THEN_UD2: [u8; 8] = [0x48, 0x01, 0xc8, 0x48, 0x01, 0xc8, 0x0f, 0x0b];
// mov eax, 0x11223344
const MOV_EAX_IMM32: [u8; 5] = [0xb8, 0x44, 0x33, 0x22, 0x11];
// mov rax, [rbx + rcx*8 + 0x12345]
const MOV_RAX_FROM_INDEXED_MEMORY: [u8; 8] = [0x48, 0x8b, 0x84, 0xcb, 0x45, 0x23, 0x01, 0x00];
// nop
const NOP: u8 = 0x90;
// ud2-free invalid opcode (push es is not valid in 64-bit mode)
const INVALID_OPCODE: [u8; 1] = [0x06];

fn cpu_running(code: &[u8]) -> Cpu {
    Harness::new(code).cpu
}

fn new_jit() -> Jit {
    Jit::new(true).expect("the host has a JIT")
}

/// For tests of one block's accounting: the block returns to the caller instead of chaining on.
fn new_unchained_jit() -> Jit {
    Jit::new(false).expect("the host has a JIT")
}

/// Translates the block at `CODE_BASE` (identity-mapped, so physical == linear).
fn translate_at_code_base(jit: &mut Jit, cpu: &mut Cpu) -> u64 {
    jit.translate(cpu, CODE_BASE, CODE_BASE)
        .expect("translation succeeds")
}

fn nops(count: usize) -> Vec<u8> {
    vec![NOP; count]
}

#[test]
fn a_new_jit_places_its_first_block_after_an_aligned_prologue() {
    let _guard = serialize_jit_tests();
    let jit = new_jit();
    assert_eq!(jit.enter_trampoline, jit.exec_mem.base());
    assert_eq!(jit.used_bytes, jit.blocks_start);
    assert_eq!(jit.blocks_start % HEADER_ALIGN, 0);
    assert!(jit.blocks_start > 0);
    assert!(jit.blocks.is_empty());
    assert!(jit.jump_cache.iter().all(|slot| slot.code_addr == 0));
    assert_eq!(jit.stats(), JitStats::default());
}

// aarch64 `b` reaches +-128 MiB from a block back to the stubs at the start of the buffer.
const _: () = assert!(CODE_SIZE <= 128 << 20);
const _: () = assert!(CODE_HEADROOM < CODE_SIZE);

#[test]
fn lookup_misses_before_translation_and_hits_afterwards() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
    let code = translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
    assert_eq!(jit.stats().translated, 1);
}

#[test]
fn translated_blocks_are_aligned_and_do_not_overlap() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(8)[..]].concat());
    let mut jit = new_jit();
    let first = translate_at_code_base(&mut jit, &mut cpu);
    let second = jit
        .translate(&mut cpu, CODE_BASE + 5, CODE_BASE + 5)
        .unwrap();
    assert_eq!(first % BLOCK_ALIGN as u64, 0);
    assert_eq!(second % BLOCK_ALIGN as u64, 0);
    assert!(first >= jit.exec_mem.base() + jit.blocks_start as u64);
    assert!(second > first);
    assert!(second < jit.exec_mem.base() + jit.used_bytes as u64);
}

#[test]
fn lookup_requires_the_same_physical_address() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    assert!(jit.lookup(CODE_BASE, CODE_BASE).is_some());
    assert_eq!(
        jit.lookup(CODE_BASE, CODE_BASE + PAGE_BYTES_FOR_TESTS),
        None,
        "the same virtual address mapped to another page is another block"
    );
}

#[test]
fn lookup_requires_the_same_rip() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(jit.lookup(CODE_BASE + 1, CODE_BASE), None);
    assert_eq!(jit.lookup(CODE_BASE ^ (1 << 40), CODE_BASE), None);
}

#[test]
fn jump_cache_hash_stays_in_range_and_folds_in_the_high_bits() {
    let _guard = serialize_jit_tests();
    for rip in [0, 1, CODE_BASE, u64::MAX, 0xffff_ffff_8000_0000, 1 << 63] {
        assert!(jump_cache_hash(rip) < 1 << JUMP_CACHE_BITS, "{rip:#x}");
    }
    // The bits above the cache index take part, so a rip one 64 KiB away does not alias.
    assert_ne!(
        jump_cache_hash(0x1000),
        jump_cache_hash(0x1000 + (1 << JUMP_CACHE_BITS))
    );
    // A deliberate collision: flipping bit 0 and bit 16 cancels out in `rip ^ (rip >> 16)`.
    assert_eq!(
        jump_cache_hash(CODE_BASE),
        jump_cache_hash(CODE_BASE ^ (1 << 16) ^ 1)
    );
}

#[test]
fn translating_a_colliding_rip_evicts_the_older_block_from_the_jump_cache_only() {
    let _guard = serialize_jit_tests();
    let colliding_rip = CODE_BASE ^ (1 << 16) ^ 1;
    assert_eq!(jump_cache_hash(colliding_rip), jump_cache_hash(CODE_BASE));
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    let first = translate_at_code_base(&mut jit, &mut cpu);
    let second = jit
        .translate(&mut cpu, colliding_rip, colliding_rip)
        .unwrap();
    assert_ne!(first, second);
    assert_eq!(jit.lookup(colliding_rip, colliding_rip), Some(second));
    assert_eq!(
        jit.lookup(CODE_BASE, CODE_BASE),
        None,
        "evicted from the cache"
    );
    assert!(jit.blocks[0].valid, "but the block itself lives on");
    // Asking for it again translates a fresh copy; blocks are never reused or freed.
    let third = translate_at_code_base(&mut jit, &mut cpu);
    assert_ne!(third, first);
    assert_eq!(jit.blocks.len(), 3);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(third));
}

#[test]
fn invalidating_a_page_drops_its_blocks_but_keeps_their_slots() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    let page = CODE_BASE >> PAGE_SHIFT;
    assert!(jit.page_blocks.contains_key(&page));
    jit.invalidate_page(page);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
    assert!(!jit.blocks[0].valid);
    assert_eq!(jit.blocks.len(), 1, "indices held elsewhere stay valid");
    assert!(!jit.page_blocks.contains_key(&page));
}

#[test]
fn invalidating_an_unrelated_page_changes_nothing() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    let code = translate_at_code_base(&mut jit, &mut cpu);
    jit.invalidate_page((CODE_BASE >> PAGE_SHIFT) + 1);
    jit.invalidate_page(0);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
    assert!(jit.blocks[0].valid);
}

#[test]
fn invalidating_a_page_twice_is_harmless() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    let page = CODE_BASE >> PAGE_SHIFT;
    jit.invalidate_page(page);
    jit.invalidate_page(page);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
}

#[test]
fn a_block_translated_after_invalidation_is_valid_again() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    jit.invalidate_page(CODE_BASE >> PAGE_SHIFT);
    let code = translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
    assert_eq!(jit.stats().translated, 2);
}

#[test]
fn translating_marks_the_page_so_stores_to_it_reach_code_written() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    let page = (CODE_BASE >> PAGE_SHIFT) as usize;
    assert_eq!(cpu.page_has_code[page], 0);
    translate_at_code_base(&mut jit, &mut cpu);
    assert_ne!(cpu.page_has_code[page], 0);
}

#[test]
fn a_block_includes_its_terminator_and_stops_there() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(8)[..]].concat());
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    let block = &jit.blocks[0];
    assert_eq!(block._insns.len(), 2, "add and the jmp that ends the block");
    assert_eq!(jit.stats().total, 2);
}

#[test]
fn a_block_holds_at_most_max_insns_instructions() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&nops(MAX_INSNS * 2 + 5));
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(jit.blocks[0]._insns.len(), MAX_INSNS);
    let next_rip = CODE_BASE + MAX_INSNS as u64;
    jit.translate(&mut cpu, next_rip, next_rip).unwrap();
    assert_eq!(jit.blocks[1]._insns.len(), MAX_INSNS);
    jit.translate(
        &mut cpu,
        next_rip + MAX_INSNS as u64,
        next_rip + MAX_INSNS as u64,
    )
    .unwrap();
    // The five remaining nops are followed by zero bytes, which decode as `add [rax], al`.
    let third = &jit.blocks[2]._insns;
    assert!(third[..5].iter().all(|insn| insn.op == NOP as u16));
    assert_eq!(third[5].op, 0x00);
}

#[test]
fn a_block_stops_at_the_end_of_its_page() {
    let _guard = serialize_jit_tests();
    let bytes_to_page_end = (PAGE_SIZE - SNIPPET_PAGE_OFFSET) as usize;
    let mut cpu = cpu_running(&nops(bytes_to_page_end + 10));
    let mut jit = new_jit();
    let page_end_rip = CODE_BASE + bytes_to_page_end as u64 - 3;
    jit.translate(&mut cpu, page_end_rip, page_end_rip).unwrap();
    assert_eq!(
        jit.blocks[0]._insns.len(),
        3,
        "three nops fit before the boundary"
    );
    assert_eq!(
        jit.page_blocks.len(),
        1,
        "and only the first page is involved"
    );
}

#[test]
fn an_instruction_straddling_a_page_is_a_block_of_its_own_listed_under_both_pages() {
    let _guard = serialize_jit_tests();
    let straddle_start = PAGE_SIZE - 2;
    let padding = (straddle_start - SNIPPET_PAGE_OFFSET) as usize;
    let code = [&nops(padding)[..], &MOV_EAX_IMM32[..]].concat();
    let mut cpu = cpu_running(&code);
    let mut jit = new_jit();
    let rip = CODE_BASE + padding as u64;
    // The straddling case decodes through `Cpu::fetch_decode`, which reads `cpu.rip`.
    cpu.rip = rip;
    jit.translate(&mut cpu, rip, rip).unwrap();
    assert_eq!(jit.blocks[0]._insns.len(), 1);
    assert_eq!(jit.blocks[0]._insns[0].len, MOV_EAX_IMM32.len() as u8);
    let first_page = rip >> PAGE_SHIFT;
    assert!(jit.page_blocks.contains_key(&first_page));
    assert!(jit.page_blocks.contains_key(&(first_page + 1)));
    assert!(
        cpu.page_has_code[first_page as usize + 1] != 0,
        "stores to the second page must invalidate it too"
    );
    jit.invalidate_page(first_page + 1);
    assert_eq!(
        jit.lookup(rip, rip),
        None,
        "invalidating either page kills the block"
    );
}

#[test]
fn a_block_before_a_straddling_instruction_ends_ahead_of_it() {
    let _guard = serialize_jit_tests();
    let straddle_start = PAGE_SIZE - 2;
    let padding = (straddle_start - SNIPPET_PAGE_OFFSET) as usize;
    let code = [&nops(padding)[..], &MOV_EAX_IMM32[..]].concat();
    let mut cpu = cpu_running(&code);
    let mut jit = new_jit();
    let rip = CODE_BASE + padding as u64 - 3;
    jit.translate(&mut cpu, rip, rip).unwrap();
    assert_eq!(
        jit.blocks[0]._insns.len(),
        3,
        "three nops, then the next block starts at the mov"
    );
    assert_eq!(jit.page_blocks.len(), 1);
}

#[test]
fn translating_outside_ram_raises_a_page_fault_and_adds_no_block() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    let outside = cpu.ram.len() as u64;
    let error = jit.translate(&mut cpu, outside, outside).unwrap_err();
    assert!(
        matches!(error, Exception::Fault { vector: 14, .. }),
        "{error:?}"
    );
    assert_eq!(cpu.cr2, outside);
    assert!(jit.blocks.is_empty());
    assert_eq!(jit.stats().translated, 0);
}

#[test]
fn undecodable_bytes_become_a_one_byte_marker_block() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&INVALID_OPCODE);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    let block = &jit.blocks[0];
    assert_eq!(block._insns.len(), 1);
    assert_eq!(block._insns[0].op, helpers::OP_INVALID);
    assert_eq!(block._insns[0].len, 1);
}

#[test]
fn an_undecodable_byte_ends_the_block_after_the_instructions_before_it() {
    let _guard = serialize_jit_tests();
    let code = [&ADD_THEN_HLT[..3], &INVALID_OPCODE[..], &nops(4)[..]].concat();
    let mut cpu = cpu_running(&code);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(jit.blocks[0]._insns.len(), 2);
    assert_eq!(jit.blocks[0]._insns[1].op, helpers::OP_INVALID);
}

#[test]
fn the_generated_code_in_the_buffer_is_the_compiler_output_verbatim() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    let code_addr = translate_at_code_base(&mut jit, &mut cpu);
    let block = &jit.blocks[0];
    let rips = [CODE_BASE, CODE_BASE + 3];
    let (expected, _) =
        arch::compile_block(code_addr, &jit.addrs, &block._insns, &rips, CODE_BASE + 4);
    let written = unsafe { std::slice::from_raw_parts(code_addr as *const u8, expected.len()) };
    assert_eq!(written, expected.as_slice());
    assert_eq!(code_addr, jit.exec_mem.base() + jit.blocks_start as u64);
}

#[test]
fn the_exec_memory_write_path_leaves_the_prologue_untouched_by_blocks() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    let prologue_len = jit.blocks_start;
    let before =
        unsafe { std::slice::from_raw_parts(jit.exec_mem.base() as *const u8, prologue_len) }
            .to_vec();
    translate_at_code_base(&mut jit, &mut cpu);
    jit.translate(&mut cpu, CODE_BASE + 3, CODE_BASE + 3)
        .unwrap();
    let after =
        unsafe { std::slice::from_raw_parts(jit.exec_mem.base() as *const u8, prologue_len) };
    assert_eq!(after, before.as_slice());
}

#[test]
fn a_worst_case_block_fits_in_the_headroom_with_a_wide_margin() {
    let _guard = serialize_jit_tests();
    // 48 inline loads, each with a slow path emitted out of line: the largest blocks the
    // compiler produces. `CODE_HEADROOM` is the free space required before translating.
    let code = MOV_RAX_FROM_INDEXED_MEMORY.repeat(MAX_INSNS);
    let mut cpu = cpu_running(&code);
    let mut jit = new_jit();
    let before = jit.used_bytes;
    translate_at_code_base(&mut jit, &mut cpu);
    let block_bytes = jit.used_bytes - before;
    assert_eq!(jit.blocks[0]._insns.len(), MAX_INSNS);
    assert!(
        block_bytes * 16 < CODE_HEADROOM,
        "{block_bytes} bytes for the largest block vs {CODE_HEADROOM} bytes of headroom"
    );
}

#[test]
fn reset_discards_every_block_and_rewinds_to_the_first_block_slot() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_JMP_NEXT);
    let mut jit = new_jit();
    let first = translate_at_code_base(&mut jit, &mut cpu);
    jit.translate(&mut cpu, CODE_BASE + 5, CODE_BASE + 5)
        .unwrap();
    jit.reset();
    assert!(jit.blocks.is_empty());
    assert!(jit.page_blocks.is_empty());
    assert!(jit.jump_cache.iter().all(|slot| slot.code_addr == 0));
    assert_eq!(jit.used_bytes, jit.blocks_start);
    assert_eq!(jit.stats().resets, 1);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
    assert_eq!(
        translate_at_code_base(&mut jit, &mut cpu),
        first,
        "the buffer is reused from the start"
    );
}

#[test]
fn a_nearly_full_code_buffer_is_flushed_before_the_next_translation() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(4)[..]].concat());
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    jit.used_bytes = jit.exec_mem.len() - CODE_HEADROOM + 1;
    let second_rip = CODE_BASE + 5;
    let code = jit.translate(&mut cpu, second_rip, second_rip).unwrap();
    assert_eq!(jit.stats().resets, 1);
    assert_eq!(
        jit.blocks.len(),
        1,
        "only the block translated after the flush remains"
    );
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
    assert_eq!(jit.lookup(second_rip, second_rip), Some(code));
    assert_eq!(code, jit.exec_mem.base() + jit.blocks_start as u64);
}

#[test]
fn a_buffer_with_exactly_the_headroom_free_is_not_flushed() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    let mut jit = new_jit();
    jit.used_bytes = jit.exec_mem.len() - CODE_HEADROOM;
    translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(jit.stats().resets, 0);
}

#[test]
fn step_block_runs_a_block_and_accounts_for_its_instructions() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(1)[..]].concat());
    cpu.jit = Some(Box::new(new_unchained_jit()));
    cpu.gpr[0] = 1;
    cpu.gpr[1] = 2;
    // `Cpu::run` counts the first instruction before calling in.
    cpu.icount += 1;
    cpu.step_block().unwrap();
    assert_eq!(cpu.gpr[0], 3);
    assert_eq!(cpu.last_block_insns, 2);
    assert_eq!(cpu.icount, 2, "the block retired two instructions in total");
    assert_eq!(cpu.rip, CODE_BASE + 5);
}

#[test]
fn step_block_reports_an_exception_with_the_faulting_instruction_counted() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_UD2);
    cpu.jit = Some(Box::new(new_jit()));
    cpu.icount += 1;
    let error = cpu.step_block().unwrap_err();
    assert!(
        matches!(error, Exception::Fault { vector: 6, .. }),
        "{error:?}"
    );
    assert_eq!(cpu.last_block_insns, 2, "two adds retired before the ud2");
    assert_eq!(
        cpu.icount, 3,
        "and the faulting ud2 counts too, as in the interpreter"
    );
    assert!(
        cpu.pending.is_none(),
        "the exception was taken out of `pending`"
    );
}

#[test]
fn step_block_clears_the_exit_request_and_the_prepaid_count() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    cpu.jit = Some(Box::new(new_jit()));
    cpu.block_exit_requested = true;
    cpu.prepaid_icount = 0;
    cpu.icount += 1;
    cpu.step_block().unwrap();
    assert!(!cpu.block_exit_requested);
    assert_eq!(cpu.prepaid_icount, 0);
}

#[test]
fn step_block_keeps_the_jit_when_translation_fails() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    cpu.jit = Some(Box::new(new_jit()));
    cpu.rip = cpu.ram.len() as u64;
    let error = cpu.step_block().unwrap_err();
    assert!(
        matches!(error, Exception::Fault { vector: 14, .. }),
        "{error:?}"
    );
    assert!(cpu.jit.is_some());
}

#[test]
fn step_block_with_an_unmapped_rip_faults_before_touching_the_cache() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&ADD_THEN_HLT);
    cpu.jit = Some(Box::new(new_jit()));
    cpu.rip = 0x1_0000_0000;
    let error = cpu.step_block().unwrap_err();
    assert!(
        matches!(error, Exception::Fault { vector: 14, .. }),
        "{error:?}"
    );
    assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 0);
}

#[test]
fn step_block_reuses_the_cached_block_on_the_second_run() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(1)[..]].concat());
    cpu.jit = Some(Box::new(new_jit()));
    for _ in 0..3 {
        cpu.rip = CODE_BASE;
        cpu.icount += 1;
        cpu.step_block().unwrap();
    }
    assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 1);
}

// mov ecx, 100; 1: sub ecx, 1; jnz 1b; hlt
const COUNTDOWN_LOOP: [u8; 11] = [
    0xb9, 0x64, 0x00, 0x00, 0x00, 0x83, 0xe9, 0x01, 0x75, 0xfb, 0xf4,
];
const RSP_INDEX: usize = 4;
/// mov (1) + 100 * (sub, jnz) + hlt (1).
const COUNTDOWN_LOOP_INSNS: u64 = 202;
// mov eax, 7; call 1f; add eax, 1; hlt; 1: add eax, eax; ret
const CALL_AND_RETURN: [u8; 17] = [
    0xb8, 0x07, 0x00, 0x00, 0x00, 0xe8, 0x04, 0x00, 0x00, 0x00, 0x83, 0xc0, 0x01, 0xf4, 0x01, 0xc0,
    0xc3,
];

/// `Cpu::run`'s per-step bookkeeping around one `step_block` call.
fn step_once(cpu: &mut Cpu) {
    cpu.interrupt_shadow = false;
    cpu.icount += 1;
    cpu.step_block().unwrap();
}

/// Calls `step_block` until the guest halts; returns how many calls it took.
fn steps_to_halt(cpu: &mut Cpu) -> usize {
    for calls in 1..=1000 {
        step_once(cpu);
        if cpu.halted {
            return calls;
        }
    }
    panic!("never halted: {}", cpu.dump_regs());
}

/// Runs the countdown loop a few iterations, so its blocks exist and chain into each other,
/// and stops mid-loop.
fn warm_up_loop(cpu: &mut Cpu) {
    cpu.next_event = 20;
    while cpu.icount < 20 {
        step_once(cpu);
    }
    assert!(!cpu.halted);
    cpu.next_event = u64::MAX;
}

#[test]
fn chained_blocks_run_a_whole_loop_without_returning_to_the_caller() {
    let _guard = serialize_jit_tests();
    let mut chained = cpu_running(&COUNTDOWN_LOOP);
    chained.jit = Some(Box::new(new_jit()));
    chained.next_event = u64::MAX;
    // Only blocks that do not exist yet send the chain back to the caller (to be translated).
    let chained_calls = steps_to_halt(&mut chained);
    assert!(chained_calls <= 3, "{chained_calls}");
    assert_eq!(chained.gpr[1], 0);
    assert_eq!(chained.icount, COUNTDOWN_LOOP_INSNS);

    let mut unchained = cpu_running(&COUNTDOWN_LOOP);
    unchained.jit = Some(Box::new(new_unchained_jit()));
    unchained.next_event = u64::MAX;
    assert!(steps_to_halt(&mut unchained) > 100);
    assert_eq!(unchained.icount, COUNTDOWN_LOOP_INSNS);
}

#[test]
fn a_chain_stops_at_the_event_deadline_and_resumes_without_losing_instructions() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&COUNTDOWN_LOOP);
    cpu.jit = Some(Box::new(new_jit()));
    warm_up_loop(&mut cpu);
    cpu.next_event = cpu.icount + 50;
    let deadline = cpu.next_event;
    step_once(&mut cpu);
    assert!(!cpu.halted);
    assert!(
        cpu.icount >= deadline && cpu.icount < deadline + 4,
        "{} vs {deadline}",
        cpu.icount
    );
    // Every retired instruction is counted once and the next one is not pre-charged: ecx says
    // how many iterations ran.
    let iterations = 100 - cpu.gpr[1];
    assert_eq!(cpu.icount, 1 + 2 * iterations);
    cpu.next_event = u64::MAX;
    steps_to_halt(&mut cpu);
    assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
}

#[test]
fn a_chained_call_and_return_find_their_blocks_through_the_jump_cache() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&CALL_AND_RETURN);
    cpu.jit = Some(Box::new(new_jit()));
    cpu.next_event = u64::MAX;
    steps_to_halt(&mut cpu);
    assert_eq!(cpu.gpr[0], 15);
    // Run it again from the top: now the call, the callee and the return all hit the cache and
    // the whole sequence is a single call.
    cpu.halted = false;
    cpu.rip = CODE_BASE;
    cpu.gpr[RSP_INDEX] = STACK_TOP;
    assert_eq!(steps_to_halt(&mut cpu), 1);
    assert_eq!(cpu.gpr[0], 15);
    assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 3);
}

#[test]
fn a_chain_does_not_enter_a_block_whose_page_was_invalidated() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&COUNTDOWN_LOOP);
    cpu.jit = Some(Box::new(new_jit()));
    warm_up_loop(&mut cpu);
    let translated = cpu.jit.as_ref().unwrap().stats().translated;
    cpu.jit
        .as_mut()
        .unwrap()
        .invalidate_page(CODE_BASE >> PAGE_SHIFT);
    steps_to_halt(&mut cpu);
    assert!(
        cpu.jit.as_ref().unwrap().stats().translated > translated,
        "the stale blocks were retranslated, not chained into"
    );
}

#[test]
fn chain_exit_after_an_rdtsc_does_not_charge_the_prepaid_instructions_twice() {
    let _guard = serialize_jit_tests();
    // mov ecx, 3; 1: sub ecx, 1; rdtsc; jnz... rdtsc clobbers eax/edx only, flags survive.
    // mov ecx, 3; 1: rdtsc; sub ecx, 1; jnz 1b; hlt
    let code = [
        0xb9, 0x03, 0x00, 0x00, 0x00, 0x0f, 0x31, 0x83, 0xe9, 0x01, 0x75, 0xf8, 0xf4,
    ];
    assert_jit_matches_interpreter(&code, |_| {});
}
