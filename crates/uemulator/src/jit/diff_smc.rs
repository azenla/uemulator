//! Self-modifying code. Stores into pages that translated blocks came from must invalidate them,
//! and the block that performed the store must stop before it can run stale bytes. Each case
//! asserts the architectural result (the new bytes execute), on both engines.

use super::diff_tests::*;
use crate::testkit::*;

// mov al, 0x2a; mov [rip + target + 1], al; target: mov ebx, 0x11111111; hlt
const PATCH_NEXT_INSN_INLINE_BYTE_STORE: [u8; 14] = [
    0xb0, 0x2a, 0x88, 0x05, 0x01, 0x00, 0x00, 0x00, 0xbb, 0x11, 0x11, 0x11, 0x11, 0xf4,
];

// mov eax, 0x22222222; mov [rip + target + 1], eax; target: mov ebx, 0x11111111; hlt
const PATCH_NEXT_INSN_INLINE_DWORD_STORE: [u8; 17] = [
    0xb8, 0x22, 0x22, 0x22, 0x22, 0x89, 0x05, 0x01, 0x00, 0x00, 0x00, 0xbb, 0x11, 0x11, 0x11, 0x11,
    0xf4,
];

// mov dword ptr [rip + target + 1], 0x33333333; target: mov ebx, 0x11111111; hlt
const PATCH_NEXT_INSN_INLINE_IMMEDIATE_STORE: [u8; 16] = [
    0xc7, 0x05, 0x01, 0x00, 0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0xbb, 0x11, 0x11, 0x11, 0x11, 0xf4,
];

// mov byte ptr [rip + target + 1], 0x44; target: mov ebx, 0x11111111; hlt
const PATCH_NEXT_INSN_HELPER_BYTE_IMMEDIATE_STORE: [u8; 13] = [
    0xc6, 0x05, 0x01, 0x00, 0x00, 0x00, 0x44, 0xbb, 0x11, 0x11, 0x11, 0x11, 0xf4,
];

// add dword ptr [rip + target + 1], 5; target: mov ebx, 0x11111111; hlt
const PATCH_NEXT_INSN_READ_MODIFY_WRITE: [u8; 13] = [
    0x83, 0x05, 0x01, 0x00, 0x00, 0x00, 0x05, 0xbb, 0x11, 0x11, 0x11, 0x11, 0xf4,
];

// mov eax, 0x55555555; xchg [rip + target + 1], eax; target: mov ebx, 0x11111111; hlt
const PATCH_NEXT_INSN_XCHG: [u8; 17] = [
    0xb8, 0x55, 0x55, 0x55, 0x55, 0x87, 0x05, 0x01, 0x00, 0x00, 0x00, 0xbb, 0x11, 0x11, 0x11, 0x11,
    0xf4,
];

// lea rdi, [rip + target + 1]; mov al, 0x66; stosb; target: mov ebx, 0x11111111; hlt
const PATCH_NEXT_INSN_STOSB: [u8; 16] = [
    0x48, 0x8d, 0x3d, 0x04, 0x00, 0x00, 0x00, 0xb0, 0x66, 0xaa, 0xbb, 0x11, 0x11, 0x11, 0x11, 0xf4,
];

// lea rdi, [rip + target + 1]; lea rsi, [rip + source]; movsb
// target: mov ebx, 0x11111111; hlt; source: .byte 0x77
const PATCH_NEXT_INSN_MOVSB: [u8; 22] = [
    0x48, 0x8d, 0x3d, 0x09, 0x00, 0x00, 0x00, 0x48, 0x8d, 0x35, 0x07, 0x00, 0x00, 0x00, 0xa4, 0xbb,
    0x11, 0x11, 0x11, 0x11, 0xf4, 0x77,
];

// mov al, 0x2a; mov [rip + target + 1], al; nop; target: mov ebx, 0x11111111; hlt
const PATCH_THE_INSTRUCTION_AFTER_NEXT: [u8; 15] = [
    0xb0, 0x2a, 0x88, 0x05, 0x02, 0x00, 0x00, 0x00, 0x90, 0xbb, 0x11, 0x11, 0x11, 0x11, 0xf4,
];

// mov byte ptr [rip + target], 0xcc; target: nop; hlt
const PATCH_LAST_INSTRUCTION_OF_THE_BLOCK_BYTES: [u8; 9] =
    [0xc6, 0x05, 0x00, 0x00, 0x00, 0x00, 0xcc, 0x90, 0xf4];

// mov ecx, 3; xor ebx, ebx; 1: target: add rbx, 1; mov [rip + target + 3], cl; dec ecx
// jnz 1b; hlt
const PATCH_AN_EARLIER_INSTRUCTION_AND_LOOP_BACK: [u8; 22] = [
    0xb9, 0x03, 0x00, 0x00, 0x00, 0x31, 0xdb, 0x48, 0x83, 0xc3, 0x01, 0x88, 0x0d, 0xf9, 0xff, 0xff,
    0xff, 0xff, 0xc9, 0x75, 0xf2, 0xf4,
];

// xor ebx, ebx; call f; add ebx, eax; mov byte ptr [rip + f + 1], 0x22; call f
// add ebx, eax; hlt; f: mov eax, 0x11; ret
const PATCH_FUNCTION_BETWEEN_CALLS: [u8; 30] = [
    0x31, 0xdb, 0xe8, 0x11, 0x00, 0x00, 0x00, 0x01, 0xc3, 0xc6, 0x05, 0x09, 0x00, 0x00, 0x00, 0x22,
    0xe8, 0x03, 0x00, 0x00, 0x00, 0x01, 0xc3, 0xf4, 0xb8, 0x11, 0x00, 0x00, 0x00, 0xc3,
];

// xor ebx, ebx; call f; add ebx, eax; mov dl, 0x22; mov [rip + f + 1], dl; call f
// add ebx, eax; hlt; f: mov eax, 0x11; ret
const PATCH_FUNCTION_BETWEEN_CALLS_INLINE_STORE: [u8; 31] = [
    0x31, 0xdb, 0xe8, 0x12, 0x00, 0x00, 0x00, 0x01, 0xc3, 0xb2, 0x22, 0x88, 0x15, 0x09, 0x00, 0x00,
    0x00, 0xe8, 0x03, 0x00, 0x00, 0x00, 0x01, 0xc3, 0xf4, 0xb8, 0x11, 0x00, 0x00, 0x00, 0xc3,
];

// xor eax, eax; cmp eax, eax; jne 1f; mov byte ptr [rip + 1f + 1], 0x77; jmp 1f
// 1: mov ebx, 0x11; hlt
const PATCH_AFTER_BRANCH_NOT_TAKEN: [u8; 21] = [
    0x31, 0xc0, 0x39, 0xc0, 0x75, 0x09, 0xc6, 0x05, 0x03, 0x00, 0x00, 0x00, 0x77, 0xeb, 0x00, 0xbb,
    0x11, 0x00, 0x00, 0x00, 0xf4,
];

// mov [rip + data], rax; add rcx, 1; mov rdx, [rip + data]; hlt; data: .quad 0
const STORE_NEAR_BUT_NOT_INTO_THE_CODE: [u8; 27] = [
    0x48, 0x89, 0x05, 0x0c, 0x00, 0x00, 0x00, 0x48, 0x83, 0xc1, 0x01, 0x48, 0x8b, 0x15, 0x01, 0x00,
    0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// mov ecx, 5; 1: mov [rip + data], rcx; dec ecx; jnz 1b; hlt; data: .quad 0
const WRITE_CODE_PAGE_ONLY_ONCE_AND_RUN_SAME_BLOCK_AGAIN: [u8; 25] = [
    0xb9, 0x05, 0x00, 0x00, 0x00, 0x48, 0x89, 0x0d, 0x05, 0x00, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xf5,
    0xf4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// mov ebx, 0x11111111; hlt
const SECOND_RUN_TARGET: [u8; 6] = [0xbb, 0x11, 0x11, 0x11, 0x11, 0xf4];

// xor ebx, ebx; lea rdi, [rip + f]; call f; add ebx, eax; mov dl, 0x22
// mov [rdi + 4], dl; call f; add ebx, eax; hlt; .org 3582; f: mov eax, 0x11223311; ret
fn cross_page_function_straddling_the_boundary() -> Vec<u8> {
    parts(&[
        Part::Bytes(&[
            0x31, 0xdb, 0x48, 0x8d, 0x3d, 0xf5, 0x0d, 0x00, 0x00, 0xe8, 0xf0, 0x0d, 0x00, 0x00,
            0x01, 0xc3, 0xb2, 0x22, 0x88, 0x57, 0x04, 0xe8, 0xe4, 0x0d, 0x00, 0x00, 0x01, 0xc3,
            0xf4,
        ]),
        Part::Fill(0x00, 3553),
        Part::Bytes(&[0xb8, 0x11, 0x33, 0x22, 0x11, 0xc3]),
    ])
}

// xor ebx, ebx; lea rdi, [rip + f]; call f; add ebx, eax; mov dl, 0x22
// mov [rdi + 1], dl; call f; add ebx, eax; hlt; .org 3582; f: mov eax, 0x11223311; ret
fn cross_page_function_patched_on_the_first_page() -> Vec<u8> {
    parts(&[
        Part::Bytes(&[
            0x31, 0xdb, 0x48, 0x8d, 0x3d, 0xf5, 0x0d, 0x00, 0x00, 0xe8, 0xf0, 0x0d, 0x00, 0x00,
            0x01, 0xc3, 0xb2, 0x22, 0x88, 0x57, 0x01, 0xe8, 0xe4, 0x0d, 0x00, 0x00, 0x01, 0xc3,
            0xf4,
        ]),
        Part::Fill(0x00, 3553),
        Part::Bytes(&[0xb8, 0x11, 0x33, 0x22, 0x11, 0xc3]),
    ])
}

// xor ebx, ebx; lea rdi, [rip + f]; call f; add ebx, eax; mov byte ptr [rdi + 3], 0x44
// call f; add ebx, eax; hlt; .org 3582; f: mov eax, 0x11223311; ret
fn cross_page_function_patched_by_a_helper_store_on_the_second_page() -> Vec<u8> {
    parts(&[
        Part::Bytes(&[
            0x31, 0xdb, 0x48, 0x8d, 0x3d, 0xf5, 0x0d, 0x00, 0x00, 0xe8, 0xf0, 0x0d, 0x00, 0x00,
            0x01, 0xc3, 0xc6, 0x47, 0x03, 0x44, 0xe8, 0xe5, 0x0d, 0x00, 0x00, 0x01, 0xc3, 0xf4,
        ]),
        Part::Fill(0x00, 3554),
        Part::Bytes(&[0xb8, 0x11, 0x33, 0x22, 0x11, 0xc3]),
    ])
}

// xor ebx, ebx; lea rdi, [rip + g]; call g; add ebx, eax; mov dl, 0x22
// mov [rdi + 1], dl; call g; add ebx, eax; hlt; .org 3584; g: mov eax, 0x11; ret
fn second_page_block_patched_from_the_first_page() -> Vec<u8> {
    parts(&[
        Part::Bytes(&[
            0x31, 0xdb, 0x48, 0x8d, 0x3d, 0xf7, 0x0d, 0x00, 0x00, 0xe8, 0xf2, 0x0d, 0x00, 0x00,
            0x01, 0xc3, 0xb2, 0x22, 0x88, 0x57, 0x01, 0xe8, 0xe6, 0x0d, 0x00, 0x00, 0x01, 0xc3,
            0xf4,
        ]),
        Part::Fill(0x00, 3555),
        Part::Bytes(&[0xb8, 0x11, 0x00, 0x00, 0x00, 0xc3]),
    ])
}

// lea rdi, [rip + last]; mov dl, 0x90; jmp last; .org 3583; last: nop; mov [rdi], dl
// hlt
fn last_byte_of_the_page_patched() -> Vec<u8> {
    parts(&[
        Part::Bytes(&[
            0x48, 0x8d, 0x3d, 0xf8, 0x0d, 0x00, 0x00, 0xb2, 0x90, 0xe9, 0xf1, 0x0d,
        ]),
        Part::Fill(0x00, 3571),
        Part::Bytes(&[0x90, 0x88, 0x17, 0xf4]),
    ])
}

const OLD_IMMEDIATE: u64 = 0x1111_1111;

#[test]
fn inline_byte_store_into_the_next_instruction_executes_the_new_byte() {
    let result = check(&PATCH_NEXT_INSN_INLINE_BYTE_STORE);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1111_112a);
}

#[test]
fn inline_dword_store_into_the_next_instruction_executes_the_new_immediate() {
    let result = check(&PATCH_NEXT_INSN_INLINE_DWORD_STORE);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x2222_2222);
}

#[test]
fn inline_immediate_store_into_the_next_instruction_executes_the_new_immediate() {
    let result = check(&PATCH_NEXT_INSN_INLINE_IMMEDIATE_STORE);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x3333_3333);
}

#[test]
fn helper_executed_byte_store_into_the_next_instruction_executes_the_new_byte() {
    let result = check(&PATCH_NEXT_INSN_HELPER_BYTE_IMMEDIATE_STORE);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1111_1144);
}

#[test]
fn read_modify_write_into_the_next_instruction_executes_the_new_immediate() {
    let result = check(&PATCH_NEXT_INSN_READ_MODIFY_WRITE);
    assert_eq!(result.interpreter.cpu.gpr[RBX], OLD_IMMEDIATE + 5);
}

#[test]
fn xchg_into_the_next_instruction_executes_the_new_immediate_and_returns_the_old_one() {
    let result = check(&PATCH_NEXT_INSN_XCHG);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x5555_5555);
    assert_eq!(result.interpreter.cpu.gpr[RAX], OLD_IMMEDIATE);
}

#[test]
fn stosb_into_the_next_instruction_executes_the_new_byte() {
    let result = check(&PATCH_NEXT_INSN_STOSB);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1111_1166);
}

#[test]
fn movsb_into_the_next_instruction_executes_the_new_byte() {
    let result = check(&PATCH_NEXT_INSN_MOVSB);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1111_1177);
}

#[test]
fn a_store_patching_an_instruction_two_ahead_in_the_same_block_takes_effect() {
    let result = check(&PATCH_THE_INSTRUCTION_AFTER_NEXT);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1111_112a);
}

#[test]
fn patching_a_nop_into_int3_traps_when_reached() {
    check_stop(&PATCH_LAST_INSTRUCTION_OF_THE_BLOCK_BYTES, trap(3));
}

#[test]
fn a_loop_that_patches_its_own_first_instruction_sees_each_new_immediate() {
    // add rbx, 1 then patch the immediate to cl: iterations add 1, 3, 2 (cl = 3, 2, 1 at the
    // patch after each pass).
    let result = check(&PATCH_AN_EARLIER_INSTRUCTION_AND_LOOP_BACK);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 1 + 3 + 2);
}

#[test]
fn patching_a_function_between_two_calls_runs_the_new_body_the_second_time() {
    let result = check(&PATCH_FUNCTION_BETWEEN_CALLS);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x11 + 0x22);
}

#[test]
fn patching_a_function_with_an_inline_store_runs_the_new_body_the_second_time() {
    let result = check(&PATCH_FUNCTION_BETWEEN_CALLS_INLINE_STORE);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x11 + 0x22);
}

#[test]
fn patching_code_after_an_untaken_branch_takes_effect() {
    let result = check(&PATCH_AFTER_BRANCH_NOT_TAKEN);
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x77);
}

#[test]
fn data_stored_in_the_code_page_does_not_disturb_the_running_block() {
    let result = check_with(&STORE_NEAR_BUT_NOT_INTO_THE_CODE, seed_registers_and_memory);
    assert_eq!(result.interpreter.cpu.gpr[RDX], SEED_RAX);
}

#[test]
fn a_loop_storing_into_its_own_page_every_iteration_completes() {
    let result = check(&WRITE_CODE_PAGE_ONLY_ONCE_AND_RUN_SAME_BLOCK_AGAIN);
    assert_eq!(result.interpreter.cpu.gpr[RCX], 0);
}

const SEED_RAX: u64 = 0x8899_aabb_ccdd_eeff;

#[test]
fn a_straddling_instruction_patched_on_its_second_page_runs_the_new_bytes() {
    let result = check(&cross_page_function_straddling_the_boundary());
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1122_3311 + 0x2222_3311);
}

#[test]
fn a_straddling_instruction_patched_on_its_first_page_runs_the_new_bytes() {
    let result = check(&cross_page_function_patched_on_the_first_page());
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1122_3311 + 0x1122_3322);
}

#[test]
fn a_straddling_instruction_patched_by_a_helper_store_runs_the_new_bytes() {
    let result = check(&cross_page_function_patched_by_a_helper_store_on_the_second_page());
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x1122_3311 + 0x1144_3311);
}

#[test]
fn a_block_on_the_second_page_patched_from_the_first_runs_the_new_bytes() {
    let result = check(&second_page_block_patched_from_the_first_page());
    assert_eq!(result.interpreter.cpu.gpr[RBX], 0x11 + 0x22);
}

#[test]
fn patching_the_last_byte_of_a_page_that_holds_a_block_still_halts() {
    check(&last_byte_of_the_page_patched());
}

/// Runs the harness again from the start of its code after the caller changed memory.
fn rerun(harness: &mut Harness) {
    harness.cpu.rip = CODE_BASE;
    harness.cpu.halted = false;
    assert_eq!(harness.run(1_000), Stop::Halted);
}

const CODE_PAGE: usize = (CODE_BASE >> 12) as usize;
const NEW_IMMEDIATE: [u8; 4] = [0x22, 0x22, 0x22, 0x22];

/// What a device DMA write into guest RAM looks like to the CPU: the bytes change without a
/// store instruction and the machine calls `Cpu::code_written` for the page.
fn dma_into_the_code_page(harness: &mut Harness) {
    harness.write(CODE_BASE + 1, &NEW_IMMEDIATE);
    harness.cpu.code_written(CODE_PAGE);
}

fn each_engine(mut body: impl FnMut(&mut Harness, bool)) {
    let _guard = super::serialize_jit_tests();
    for use_jit in [false, true] {
        let mut harness = Harness::new(&SECOND_RUN_TARGET);
        if use_jit && !harness.enable_jit() {
            continue;
        }
        harness.run_to_halt();
        assert_eq!(harness.cpu.gpr[RBX], OLD_IMMEDIATE);
        body(&mut harness, use_jit);
    }
}

#[test]
fn bytes_changed_by_dma_take_effect_after_code_written() {
    each_engine(|harness, _| {
        dma_into_the_code_page(harness);
        rerun(harness);
        assert_eq!(harness.cpu.gpr[RBX], 0x2222_2222);
    });
}

#[test]
fn a_pending_exit_request_from_dma_is_consumed_by_one_block_run() {
    each_engine(|harness, _| {
        dma_into_the_code_page(harness);
        assert!(harness.cpu.block_exit_requested || harness.cpu.jit.is_none());
        rerun(harness);
        assert!(
            !harness.cpu.block_exit_requested,
            "step_block clears the request"
        );
        // A second pass with no further DMA runs the new bytes without any early exit.
        rerun(harness);
        assert_eq!(harness.cpu.gpr[RBX], 0x2222_2222);
    });
}

#[test]
fn code_written_to_an_unrelated_page_does_not_change_behavior() {
    each_engine(|harness, _| {
        harness.cpu.code_written(CODE_PAGE + 100);
        rerun(harness);
        assert_eq!(harness.cpu.gpr[RBX], OLD_IMMEDIATE);
    });
}

#[test]
fn rewriting_identical_bytes_and_invalidating_runs_the_same_code() {
    each_engine(|harness, _| {
        harness.write(CODE_BASE + 1, &OLD_IMMEDIATE.to_le_bytes()[..4]);
        harness.cpu.code_written(CODE_PAGE);
        rerun(harness);
        assert_eq!(harness.cpu.gpr[RBX], OLD_IMMEDIATE);
    });
}

/// Translation counts only exist where there is a JIT.
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod translation_counts {
    use super::*;

    #[test]
    fn rerunning_an_unchanged_block_does_not_translate_it_again() {
        each_engine(|harness, use_jit| {
            let translated =
                |harness: &Harness| harness.cpu.jit.as_ref().map(|jit| jit.stats().translated);
            let before = translated(harness);
            rerun(harness);
            rerun(harness);
            assert_eq!(translated(harness), before);
            if use_jit {
                dma_into_the_code_page(harness);
                rerun(harness);
                assert_eq!(translated(harness), before.map(|count| count + 1));
            }
        });
    }

    #[test]
    fn an_invalidated_block_is_translated_again_with_the_new_bytes() {
        each_engine(|harness, use_jit| {
            dma_into_the_code_page(harness);
            rerun(harness);
            assert_eq!(harness.cpu.gpr[RBX], 0x2222_2222);
            if use_jit {
                assert_eq!(harness.cpu.jit.as_ref().unwrap().stats().translated, 2);
            }
        });
    }
}
