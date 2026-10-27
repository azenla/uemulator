//! Places where a backend that keeps guest flags in host flags (the x86_64 one) can lose them or
//! a parked value: a memory access in the middle of an instruction (the TLB probe and the helper
//! call of a slow path clobber the host flags and every scratch register), read-modify-write
//! stores, partial-register writes, an `inc`/`dec` carry-in, a shift by a count that is zero
//! modulo the width, and indirect control transfers that read their target before pushing. Each
//! case compares the JIT with the interpreter across every condition code. They pass on both
//! backends; the x86_64 lowering is the one they were written against.

use super::diff_tests::*;
use crate::testkit::*;

/// `jcc +5; mov esi, 1; hlt`: the result in esi says which way the branch went.
fn jcc_then_halt(condition: u8) -> Vec<u8> {
    vec![0x70 | condition, 0x05, 0xbe, 0x01, 0x00, 0x00, 0x00, HLT]
}

const CONDITIONS: std::ops::Range<u8> = 0..16;

/// `(rax, rcx, rdx, qword at [rbx])`: zero, equal, carry and signed overflow at the narrow
/// widths, a high-byte pattern, and values whose low byte has odd and even parity.
const INPUTS: [(u64, u64, u64, u64); 8] = [
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (0xff, 1, 0xff, 0xff),
    (0x7f, 1, 0x7f, 0x7f),
    (0x8000, 0x8000, 0x8000_0000, 0x8000_0000),
    (u64::MAX, 2, u64::MAX, u64::MAX),
    (0x1234_5678_9abc_def0, 0xfe, 0x0123, 0x80),
    (0x7fff_ffff_ffff_ffff, 1, 0x7fff_ffff, 0x7fff_ffff_ffff_ffff),
];

fn run_case(code: &[u8], context: &str, rax: u64, rcx: u64, rdx: u64, memory: u64) {
    let result = run_serialized(code, |harness| {
        seed_registers_and_memory(harness);
        harness.cpu.gpr[RAX] = rax;
        harness.cpu.gpr[RCX] = rcx;
        harness.cpu.gpr[RDX] = rdx;
        let base = harness.cpu.gpr[RBX];
        harness.write_u64(base, memory);
    });
    let context = format!("{context} rax={rax:#x} rcx={rcx:#x} rdx={rdx:#x} [rbx]={memory:#x}");
    assert_eq!(
        result.interpreter_stop,
        Stop::Halted,
        "{context}: {}",
        result.interpreter.cpu.dump_regs()
    );
    if let Some(difference) = difference_in(&result, HOT_RAM) {
        panic!("{context}: {difference}");
    }
}

/// Runs `prefix` + `middle` + each condition's `jcc` for every input.
fn run_with_every_condition(prefix: &[u8], middle: &[u8], inputs: &[(u64, u64, u64, u64)]) {
    for condition in CONDITIONS {
        let code = [prefix, middle, &jcc_then_halt(condition)].concat();
        for &(rax, rcx, rdx, memory) in inputs {
            run_case(
                &code,
                &format!("{prefix:02x?} {middle:02x?} jcc {condition:#x}"),
                rax,
                rcx,
                rdx,
                memory,
            );
        }
    }
}

// Read-modify-write instructions: the store's probe and slow path destroy the host flags, so a
// jcc afterwards must see the lazily recorded ones.
const READ_MODIFY_WRITE: [&[u8]; 14] = [
    &[0x01, 0x03],                   // add dword ptr [rbx], eax
    &[0x48, 0x29, 0x03],             // sub qword ptr [rbx], rax
    &[0x20, 0x03],                   // and byte ptr [rbx], al
    &[0x66, 0x09, 0x03],             // or word ptr [rbx], ax
    &[0x31, 0x03],                   // xor dword ptr [rbx], eax
    &[0xff, 0x03],                   // inc dword ptr [rbx]
    &[0xfe, 0x0b],                   // dec byte ptr [rbx]
    &[0x48, 0xf7, 0x1b],             // neg qword ptr [rbx]
    &[0x66, 0xf7, 0x1b],             // neg word ptr [rbx]
    &[0xc1, 0x23, 0x03],             // shl dword ptr [rbx], 3
    &[0x48, 0xd3, 0x2b],             // shr qword ptr [rbx], cl
    &[0xd1, 0x3b],                   // sar dword ptr [rbx], 1
    &[0x00, 0x23],                   // add byte ptr [rbx], ah
    &[0x66, 0xc7, 0x03, 0x01, 0x00], // mov word ptr [rbx], 1 (flags untouched by the store)
];

#[test]
fn a_jcc_after_a_read_modify_write_sees_the_recorded_flags() {
    for producer in READ_MODIFY_WRITE {
        run_with_every_condition(&[], producer, &INPUTS);
    }
}

#[test]
fn a_jcc_after_a_compare_with_a_memory_operand_sees_the_flags() {
    const COMPARES: [&[u8]; 3] = [
        &[0x39, 0x03],       // cmp dword ptr [rbx], eax
        &[0x48, 0x85, 0x03], // test qword ptr [rbx], rax
        &[0x02, 0x23],       // add ah, byte ptr [rbx]
    ];
    for producer in COMPARES {
        run_with_every_condition(&[], producer, &INPUTS);
    }
}

// Writes that must not disturb flags that are still live: byte and word merges into registers
// (done as narrow stores into the register file), a sign extension of rax into rdx, and not.
const FLAG_NEUTRAL_WRITES: [&[u8]; 12] = [
    &[0x88, 0xdc],                   // mov ah, bl
    &[0x66, 0x89, 0xd8],             // mov ax, bx
    &[0xb5, 0x07],                   // mov ch, 7
    &[0x88, 0x03],                   // mov byte ptr [rbx], al
    &[0x99],                         // cdq
    &[0x48, 0x99],                   // cqo
    &[0x98],                         // cwde
    &[0x48, 0x98],                   // cdqe
    &[0x48, 0xf7, 0xd2],             // not rdx
    &[0x0f, 0xae, 0xf0],             // mfence
    &[0x66, 0xc7, 0xc2, 0x34, 0x12], // mov dx, 0x1234
    &[0x8a, 0x03],                   // mov al, byte ptr [rbx]
];

#[test]
fn flags_survive_register_writes_that_do_not_set_flags() {
    const PRODUCERS: [&[u8]; 3] = [
        &[0x48, 0x39, 0xc8], // cmp rax, rcx
        &[0x00, 0xd8],       // add al, bl
        &[0x66, 0x29, 0xd8], // sub ax, bx
    ];
    for producer in PRODUCERS {
        for write in FLAG_NEUTRAL_WRITES {
            run_with_every_condition(producer, write, &INPUTS);
        }
    }
}

#[test]
fn parity_conditions_after_byte_and_word_operations_use_the_narrow_result() {
    const NARROW: [&[u8]; 6] = [
        &[0x00, 0xd8],                   // add al, bl
        &[0x66, 0x29, 0xd8],             // sub ax, bx
        &[0x20, 0xd1],                   // and cl, dl
        &[0x00, 0xdc],                   // add ah, bl
        &[0x66, 0x81, 0xca, 0x34, 0x12], // or dx, 0x1234
        &[0x84, 0xc0],                   // test al, al
    ];
    for producer in NARROW {
        run_with_every_condition(&[], producer, &INPUTS);
    }
}

#[test]
fn inc_and_dec_keep_the_carry_for_conditions_that_read_it() {
    // Four sources of the carry: set and clear by the flag instructions (which the JIT hands to
    // the interpreter, so the host flags are dead), a compare that leaves them live, and a
    // read-modify-write that does not.
    const CARRY_SOURCES: [&[u8]; 5] = [
        &[0xf9],             // stc
        &[0xf8],             // clc
        &[0x48, 0x39, 0xc8], // cmp rax, rcx
        &[0x00, 0x03],       // add byte ptr [rbx], al
        &[0x39, 0xc8],       // cmp eax, ecx
    ];
    const INC_DEC: [&[u8]; 8] = [
        &[0xff, 0xc2],       // inc edx
        &[0xff, 0xca],       // dec edx
        &[0x48, 0xff, 0xc2], // inc rdx
        &[0xfe, 0xc4],       // inc ah
        &[0xfe, 0x03],       // inc byte ptr [rbx]
        &[0x48, 0xff, 0x0b], // dec qword ptr [rbx]
        &[0xff, 0x03],       // inc dword ptr [rbx]
        &[0x66, 0xff, 0x0b], // dec word ptr [rbx]
    ];
    for source in CARRY_SOURCES {
        for operation in INC_DEC {
            run_with_every_condition(source, operation, &INPUTS);
        }
    }
}

#[test]
fn a_shift_by_a_count_that_is_zero_modulo_the_width_leaves_the_earlier_flags() {
    const SHIFTS: [&[u8]; 7] = [
        &[0xd3, 0xe2],       // shl edx, cl
        &[0x48, 0xd3, 0xe2], // shl rdx, cl
        &[0xd3, 0xea],       // shr edx, cl
        &[0xd3, 0xfa],       // sar edx, cl
        &[0xd3, 0x23],       // shl dword ptr [rbx], cl
        &[0x48, 0xd3, 0x2b], // shr qword ptr [rbx], cl
        &[0xd1, 0xe2],       // shl edx, 1
    ];
    const COUNTS: [u64; 8] = [0, 32, 64, 1, 31, 63, 0x100, 0x21];
    let inputs: Vec<(u64, u64, u64, u64)> = COUNTS
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            (
                0x8000_0000 + index as u64,
                count,
                0x8000_0001_8000_0001,
                0xc000_0000_c000_0000,
            )
        })
        .collect();
    for shift in SHIFTS {
        // compare rax with rbx's register-held constant first: its flags are the ones a zero
        // count must leave in place
        run_with_every_condition(&[0x48, 0x39, 0xc2], shift, &inputs);
        run_with_every_condition(&[0xf9], shift, &inputs);
    }
}

#[test]
fn cmov_reads_the_condition_before_its_memory_source_clobbers_the_flags() {
    const PRODUCERS: [&[u8]; 3] = [
        &[0x48, 0x39, 0xc8], // cmp rax, rcx
        &[0xf9],             // stc
        &[0x01, 0x03],       // add dword ptr [rbx], eax
    ];
    for producer in PRODUCERS {
        for condition in CONDITIONS {
            let cmov = 0x40 | condition;
            let consumers: [Vec<u8>; 4] = [
                vec![0x0f, cmov, 0x13],       // cmovcc edx, dword ptr [rbx]
                vec![0x48, 0x0f, cmov, 0x13], // cmovcc rdx, qword ptr [rbx]
                vec![0x0f, cmov, 0xd6],       // cmovcc edx, esi
                vec![0x48, 0x0f, cmov, 0xd6], // cmovcc rdx, rsi
            ];
            for consumer in consumers {
                let code = [producer, &consumer, &[HLT][..]].concat();
                for &(rax, rcx, rdx, memory) in &INPUTS {
                    run_case(
                        &code,
                        &format!("{producer:02x?} {consumer:02x?}"),
                        rax,
                        rcx,
                        rdx | 0xdead_0000_0000_0000,
                        memory,
                    );
                }
            }
        }
    }
}

#[test]
fn a_cmov_whose_memory_source_faults_leaves_state_as_if_it_never_started() {
    // A non-canonical address faults whether or not the condition holds.
    for condition in CONDITIONS {
        let code = [
            &[0x48, 0x39, 0xc8][..],         // cmp rax, rcx
            &[0x48, 0x8b, 0x1b],             // mov rbx, qword ptr [rbx] (the pointer read below)
            &[0x0f, 0x40 | condition, 0x13], // cmovcc edx, dword ptr [rbx]
            &[HLT],
        ]
        .concat();
        for &(rax, rcx, _, _) in &INPUTS[..3] {
            let result = run_serialized(&code, |harness| {
                seed_registers_and_memory(harness);
                harness.cpu.gpr[RAX] = rax;
                harness.cpu.gpr[RCX] = rcx;
                let base = harness.cpu.gpr[RBX];
                harness.write_u64(base, 0xdead_0000_0000_0000);
            });
            if let Some(difference) = difference_in(&result, HOT_RAM) {
                panic!("cmov {condition:#x} rax={rax:#x} rcx={rcx:#x}: {difference}");
            }
            assert_ne!(result.interpreter_stop, Stop::Halted);
        }
    }
}

#[test]
fn setcc_to_memory_reads_the_flags_before_the_store_probe() {
    const PRODUCERS: [&[u8]; 3] = [
        &[0x48, 0x39, 0xc8], // cmp rax, rcx
        &[0xf9],             // stc
        &[0x00, 0x23],       // add byte ptr [rbx], ah
    ];
    for producer in PRODUCERS {
        for condition in CONDITIONS {
            for destination in [
                vec![0x0f, 0x90 | condition, 0x03], // setcc byte ptr [rbx]
                vec![0x0f, 0x90 | condition, 0xc2], // setcc dl
                vec![0x0f, 0x90 | condition, 0xc6], // setcc dh
            ] {
                let code = [producer, &destination, &[HLT][..]].concat();
                for &(rax, rcx, rdx, memory) in &INPUTS {
                    run_case(
                        &code,
                        &format!("{producer:02x?} {destination:02x?}"),
                        rax,
                        rcx,
                        rdx,
                        memory,
                    );
                }
            }
        }
    }
}

#[test]
fn imul_leaves_the_product_flags_and_the_interpreter_agrees_on_every_condition() {
    const IMULS: [&[u8]; 4] = [
        &[0x0f, 0xaf, 0xd1],                   // imul edx, ecx
        &[0x48, 0x0f, 0xaf, 0x13],             // imul rdx, qword ptr [rbx]
        &[0x69, 0xd1, 0x34, 0x12, 0x00, 0x00], // imul edx, ecx, 0x1234
        &[0x48, 0x6b, 0xd0, 0xf0],             // imul rdx, rax, -16
    ];
    for multiply in IMULS {
        run_with_every_condition(&[0x48, 0x39, 0xc8], multiply, &INPUTS);
    }
}

/// Where the target of an indirect transfer lives: a register, or the stack slot above the
/// return address. The hlt at `HALT_OFFSET` is where every variant ends up.
const HALT_OFFSET: u64 = 8;

fn run_indirect(code: &[u8], context: &str, rsp_slot_offset: u64) {
    let halt_at = CODE_BASE + HALT_OFFSET;
    let result = run_serialized(code, |harness| {
        seed_registers_and_memory(harness);
        harness.cpu.gpr[RAX] = halt_at;
        let stack = harness.cpu.gpr[RSP];
        harness.write_u64(stack + rsp_slot_offset, halt_at);
    });
    assert_eq!(result.interpreter_stop, Stop::Halted, "{context}");
    if let Some(difference) = difference_in(&result, HOT_RAM) {
        panic!("{context}: {difference}");
    }
}

#[test]
fn indirect_call_jmp_and_push_read_their_target_before_the_stack_changes() {
    let pad = |bytes: &[u8]| {
        let mut code = bytes.to_vec();
        code.resize(HALT_OFFSET as usize, 0x90);
        code.push(HLT);
        code
    };
    let cases: [(&str, Vec<u8>, u64); 6] = [
        ("call rax", pad(&[0xff, 0xd0]), 0),
        ("jmp rax", pad(&[0xff, 0xe0]), 0),
        ("call [rsp]", pad(&[0xff, 0x14, 0x24]), 0),
        ("call [rsp + 8]", pad(&[0xff, 0x54, 0x24, 0x08]), 8),
        ("jmp [rsp + 8]", pad(&[0xff, 0x64, 0x24, 0x08]), 8),
        // push [rsp] then pop: the pushed value is the old top of stack
        ("push [rsp]; pop rdx", pad(&[0xff, 0x34, 0x24, 0x5a]), 0),
    ];
    for (name, code, slot) in cases {
        run_indirect(&code, name, slot);
    }
}

#[test]
fn fences_are_nops() {
    for fence in [[0x0f, 0xae, 0xe8], [0x0f, 0xae, 0xf0], [0x0f, 0xae, 0xf8]] {
        let code = [&[0x48, 0x39, 0xc8][..], &fence, &jcc_then_halt(2)].concat();
        for &(rax, rcx, rdx, memory) in &INPUTS {
            run_case(&code, &format!("{fence:02x?}"), rax, rcx, rdx, memory);
        }
    }
}
