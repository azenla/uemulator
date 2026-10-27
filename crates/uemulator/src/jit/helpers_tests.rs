//! The block-ending set, the clock-reading set, the exit-code layout and the helper ABI that
//! generated code relies on.

use super::*;
use crate::cpu::decode;
use crate::cpu::exception::Exception;
use crate::cpu::flags::{AF, CF, OF, PF, SF, ZF};
use crate::testkit::*;

fn insn_with_op(op: u16) -> Insn {
    Insn {
        op,
        ..Default::default()
    }
}

fn group_five(reg: u8) -> Insn {
    Insn {
        op: 0xff,
        reg,
        ..Default::default()
    }
}

fn ends(op: u16) -> bool {
    ends_block(&insn_with_op(op))
}

#[test]
fn control_transfers_end_a_block() {
    for op in (0x70..=0x7f).chain([
        0xe0, 0xe1, 0xe2, 0xe3, 0xe8, 0xe9, 0xeb, 0xc2, 0xc3, 0xca, 0xcb, 0xcc, 0xcd, 0xcf, 0xf4,
    ]) {
        assert!(ends(op), "opcode {op:#x}");
    }
}

#[test]
fn conditional_near_jumps_end_a_block() {
    for op in 0x180..=0x18f {
        assert!(ends(op), "opcode {op:#x}");
    }
    assert!(!ends(0x17f));
    assert!(!ends(0x190), "setcc is not a jump");
}

#[test]
fn string_and_port_instructions_end_a_block() {
    for op in (0x6c..=0x6f)
        .chain(0xa4..=0xa7)
        .chain(0xaa..=0xaf)
        .chain(0xe4..=0xe7)
        .chain(0xec..=0xef)
    {
        assert!(ends(op), "opcode {op:#x}");
    }
}

#[test]
fn instructions_that_change_interrupt_or_segment_state_end_a_block() {
    // cli, sti, popf, mov sreg, pop fs, pop gs
    for op in [0xfa, 0xfb, 0x9d, 0x8e, 0x1a1, 0x1a9] {
        assert!(ends(op), "opcode {op:#x}");
    }
}

#[test]
fn system_instructions_end_a_block() {
    // groups 6 and 7, syscall, clts, sysret, ud2, mov to and from cr/dr, wrmsr
    for op in [
        0x100, 0x101, 0x105, 0x106, 0x107, 0x10b, 0x120, 0x121, 0x122, 0x123, 0x130,
    ] {
        assert!(ends(op), "opcode {op:#x}");
    }
}

#[test]
fn the_invalid_marker_ends_a_block() {
    assert!(ends(OP_INVALID));
}

#[test]
fn indirect_calls_and_jumps_end_a_block_but_inc_dec_and_push_do_not() {
    for reg in 2..=5 {
        assert!(ends_block(&group_five(reg)), "ff /{reg}");
    }
    for reg in [0, 1, 6, 7] {
        assert!(!ends_block(&group_five(reg)), "ff /{reg}");
    }
    // The ModRM.reg field is masked to three bits (REX.R is not part of it).
    assert!(ends_block(&group_five(8 + 2)));
}

#[test]
fn ordinary_instructions_do_not_end_a_block() {
    // add/mov/lea/test/imul, nop, leave, enter, cdq, cmpxchg, rdtsc, rdmsr, cpuid, setcc, cmovcc,
    // movzx, bswap, xchg, push/pop register, lahf/sahf, pushf, hlt-adjacent cmc/clc/stc
    for op in [
        0x00, 0x01, 0x05, 0x31, 0x39, 0x3d, 0x50, 0x58, 0x63, 0x69, 0x6b, 0x80, 0x81, 0x83, 0x84,
        0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8d, 0x90, 0x98, 0x99, 0x9c, 0x9e, 0x9f, 0xa8,
        0xa9, 0xb0, 0xb8, 0xc0, 0xc1, 0xc6, 0xc7, 0xc8, 0xc9, 0xd0, 0xd1, 0xd2, 0xd3, 0xf5, 0xf6,
        0xf7, 0xf8, 0xf9, 0xfc, 0xfd, 0xfe, 0x10d, 0x118, 0x131, 0x132, 0x140, 0x1a2, 0x190, 0x1af,
        0x1b6, 0x1be, 0x1c8, 0x1ba,
    ] {
        assert!(!ends(op), "opcode {op:#x}");
    }
}

#[test]
fn opcodes_just_outside_the_block_ending_ranges_do_not_end_one() {
    // below and above jcc rel8, below the string and port ranges, between the call/jmp opcodes
    // and the prefixes at 0xf0..=0xf3
    for op in [
        0x6b, 0x80, 0xa3, 0xa8, 0xa9, 0xb0, 0xdf, 0xea, 0xf0, 0xf3, 0xf5,
    ] {
        assert!(!ends(op), "opcode {op:#x}");
    }
}

#[test]
fn the_clock_reading_instructions_are_exactly_rdtsc_rdmsr_and_port_io() {
    for op in [0x131, 0x132, 0xe4, 0xe5, 0xe6, 0xe7, 0xec, 0xed, 0xee, 0xef] {
        assert!(reads_guest_clock(op), "opcode {op:#x}");
    }
    for op in [
        0x130, 0x133, 0xe3, 0xe8, 0xeb, 0xf0, 0x6c, 0x6d, 0x6e, 0x6f, 0x31, 0x32, 0x90, 0xa2, 0x1a2,
    ] {
        assert!(!reads_guest_clock(op), "opcode {op:#x}");
    }
}

#[test]
fn every_inline_terminator_also_ends_a_block() {
    for op in 0..=0x1ffu16 {
        let insn = insn_with_op(op);
        if is_inline_terminator(&insn) {
            assert!(ends_block(&insn), "opcode {op:#x}");
        }
    }
}

#[test]
fn the_inline_terminators_are_jcc_jmp_call_and_ret() {
    for op in (0x70..=0x7f)
        .chain(0x180..=0x18f)
        .chain([0xe9, 0xeb, 0xe8, 0xc3])
    {
        assert!(is_inline_terminator(&insn_with_op(op)), "opcode {op:#x}");
    }
    for op in [0xe0, 0xe3, 0xc2, 0xcb, 0xf4, 0xff, 0xcc, 0xcd, OP_INVALID] {
        assert!(!is_inline_terminator(&insn_with_op(op)), "opcode {op:#x}");
    }
}

#[test]
fn status_values_are_distinct_and_fit_in_the_status_byte() {
    let statuses = [STATUS_OK, STATUS_EXCEPTION, STATUS_EXIT_REQUESTED];
    assert_eq!(
        statuses[0], 0,
        "success must be zero: generated code tests it with cbz/test"
    );
    for (index, status) in statuses.iter().enumerate() {
        assert!(*status <= EXIT_STATUS_MASK);
        for other in &statuses[index + 1..] {
            assert_ne!(status, other);
        }
    }
    assert_eq!(EXIT_STATUS_MASK, 0xff);
    assert_eq!(
        EXIT_COUNT_SHIFT, 8,
        "the prologue stubs hard-code this shift"
    );
}

#[test]
fn exit_codes_pack_and_unpack_the_retired_count_and_status() {
    for retired in [0u64, 1, 2, 47, 48, 255, 1 << 20] {
        for status in [STATUS_OK, STATUS_EXCEPTION, STATUS_EXIT_REQUESTED] {
            let exit_code = (retired << EXIT_COUNT_SHIFT) | status;
            assert_eq!(exit_code >> EXIT_COUNT_SHIFT, retired);
            assert_eq!(exit_code & EXIT_STATUS_MASK, status);
        }
    }
}

// MAX_INSNS is 48; a block's count is stored as `retired << EXIT_COUNT_SHIFT` through a `mov`
// immediate and must not collide with the status.
const _: () = assert!((48u64 << EXIT_COUNT_SHIFT) & EXIT_STATUS_MASK == 0);

fn decoded(bytes: &[u8]) -> Insn {
    decode::decode(bytes).expect("snippet decodes")
}

// add rax, rcx
const ADD_RAX_RCX: [u8; 3] = [0x48, 0x01, 0xc8];
// ud2
const UD2: [u8; 2] = [0x0f, 0x0b];
// mov [rbx], rax
const MOV_TO_RBX_MEMORY: [u8; 3] = [0x48, 0x89, 0x03];

#[test]
fn helper_exec_runs_the_instruction_and_leaves_rip_at_the_next_one() {
    let mut cpu = Harness::new(&[0xf4]).cpu;
    cpu.gpr[0] = 40;
    cpu.gpr[1] = 2;
    let insn = decoded(&ADD_RAX_RCX);
    let status = helper_exec(&mut cpu, &insn, 0x1234);
    assert_eq!(status, STATUS_OK);
    assert_eq!(cpu.gpr[0], 42);
    assert_eq!(
        cpu.rip,
        0x1234 + ADD_RAX_RCX.len() as u64,
        "exec derives the next rip from the rip passed in"
    );
}

#[test]
fn helper_exec_stores_an_exception_in_pending_and_reports_the_status() {
    let mut cpu = Harness::new(&[0xf4]).cpu;
    let insn = decoded(&UD2);
    let status = helper_exec(&mut cpu, &insn, 0x4000);
    assert_eq!(status, STATUS_EXCEPTION);
    assert!(matches!(
        cpu.pending,
        Some(Exception::Fault { vector: 6, .. })
    ));
    assert_eq!(cpu.rip, 0x4000, "rip names the faulting instruction");
}

#[test]
fn helper_exec_reports_a_requested_block_exit_after_a_successful_instruction() {
    let mut cpu = Harness::new(&[0xf4]).cpu;
    cpu.block_exit_requested = true;
    let insn = decoded(&ADD_RAX_RCX);
    assert_eq!(helper_exec(&mut cpu, &insn, 0x1000), STATUS_EXIT_REQUESTED);
}

#[test]
fn helper_exec_prefers_the_exception_over_a_pending_exit_request() {
    let mut cpu = Harness::new(&[0xf4]).cpu;
    cpu.block_exit_requested = true;
    let insn = decoded(&UD2);
    assert_eq!(helper_exec(&mut cpu, &insn, 0x1000), STATUS_EXCEPTION);
}

#[test]
fn a_store_through_helper_exec_to_a_code_page_requests_a_block_exit() {
    let _guard = crate::jit::serialize_jit_tests();
    let mut harness = Harness::new(&[0xf4]);
    assert!(harness.enable_jit());
    let cpu = &mut harness.cpu;
    // Mark the page holding the snippet as translated code, as `translate` does.
    cpu.mark_code(CODE_BASE);
    cpu.gpr[3] = CODE_BASE + 0x100;
    let insn = decoded(&MOV_TO_RBX_MEMORY);
    assert_eq!(helper_exec(cpu, &insn, CODE_BASE), STATUS_EXIT_REQUESTED);
}

#[test]
fn helper_cond_evaluates_every_condition_against_the_guest_flags() {
    let mut cpu = Harness::new(&[0xf4]).cpu;
    // (flags, conditions that hold): O NO B AE E NE BE A S NS P NP L GE LE G
    let cases: [(u64, [bool; 16]); 4] = [
        (
            0,
            [
                false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false, true,
            ],
        ),
        (
            ZF | PF,
            [
                false, true, false, true, true, false, true, false, false, true, true, false,
                false, true, true, false,
            ],
        ),
        (
            CF | SF | AF,
            [
                false, true, true, false, false, true, true, false, true, false, false, true, true,
                false, true, false,
            ],
        ),
        (
            OF | SF,
            [
                true, false, false, true, false, true, false, true, true, false, false, true,
                false, true, false, true,
            ],
        ),
    ];
    for (flags, expected) in cases {
        cpu.set_eflags_arith(flags);
        for (condition, holds) in expected.iter().enumerate() {
            assert_eq!(
                helper_cond(&mut cpu, condition as u64),
                *holds as u64,
                "flags {flags:#x} condition {condition}"
            );
        }
    }
}

#[test]
fn helper_cond_masks_the_condition_to_the_low_four_bits() {
    let mut cpu = Harness::new(&[0xf4]).cpu;
    cpu.set_eflags_arith(ZF);
    assert_eq!(helper_cond(&mut cpu, 4), 1);
    assert_eq!(
        helper_cond(&mut cpu, 0x74),
        1,
        "the low nibble of the jcc opcode selects the condition"
    );
}

#[cfg(target_arch = "x86_64")]
mod x86_64_memory_helpers {
    use super::*;

    #[test]
    fn helper_load_via_out_writes_the_value_through_the_pointer() {
        let mut harness = Harness::new(&[0xf4]);
        harness.write_u64(SCRATCH, 0x1122_3344_5566_7788);
        let mut slot = 0u64;
        let status = helper_load_via_out(&mut harness.cpu, SCRATCH, 8, &mut slot);
        assert_eq!(status, STATUS_OK);
        assert_eq!(slot, 0x1122_3344_5566_7788);
        let status = helper_load_via_out(&mut harness.cpu, SCRATCH + 2, 2, &mut slot);
        assert_eq!(status, STATUS_OK);
        assert_eq!(slot, 0x5566);
    }

    #[test]
    fn helper_load_via_out_reports_a_fault_and_leaves_the_slot_alone() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        let mut slot = 0xdead;
        let status = helper_load_via_out(&mut cpu, 0x1_0000_0000, 8, &mut slot);
        assert_eq!(status, STATUS_EXCEPTION);
        assert_eq!(slot, 0xdead);
        assert!(matches!(
            cpu.pending,
            Some(Exception::Fault { vector: 14, .. })
        ));
    }

    #[test]
    fn helper_store_x64_writes_memory_and_reports_faults() {
        let mut harness = Harness::new(&[0xf4]);
        assert_eq!(
            helper_store_x64(&mut harness.cpu, SCRATCH, 4, 0xaabb_ccdd),
            STATUS_OK
        );
        assert_eq!(harness.read(SCRATCH, 4), vec![0xdd, 0xcc, 0xbb, 0xaa]);
        assert_eq!(
            helper_store_x64(&mut harness.cpu, 0x1_0000_0000, 8, 1),
            STATUS_EXCEPTION
        );
        assert!(matches!(
            harness.cpu.pending,
            Some(Exception::Fault { vector: 14, .. })
        ));
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_memory_helpers {
    use super::*;

    #[test]
    fn helper_load_returns_the_value_and_ok_status() {
        let mut harness = Harness::new(&[0xf4]);
        harness.write_u64(SCRATCH, 0x1122_3344_5566_7788);
        let result = helper_load(&mut harness.cpu, SCRATCH, 8, 0x5000);
        assert_eq!(result.status, STATUS_OK);
        assert_eq!(result.val, 0x1122_3344_5566_7788);
        let result = helper_load(&mut harness.cpu, SCRATCH + 1, 1, 0x5000);
        assert_eq!(result.val, 0x77);
    }

    #[test]
    fn helper_load_reports_the_fault_with_the_instruction_rip() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        let result = helper_load(&mut cpu, 0x1_0000_0000, 8, 0x5000);
        assert_eq!(result.status, STATUS_EXCEPTION);
        assert_eq!(result.val, 0);
        assert_eq!(
            cpu.rip, 0x5000,
            "generated code does not keep rip current, the helper stores it"
        );
        assert!(matches!(
            cpu.pending,
            Some(Exception::Fault { vector: 14, .. })
        ));
    }

    #[test]
    fn helper_store_writes_memory_and_reports_faults_with_the_rip() {
        let mut harness = Harness::new(&[0xf4]);
        assert_eq!(
            helper_store(&mut harness.cpu, SCRATCH, 2, 0xbeef, 0x5000),
            STATUS_OK
        );
        assert_eq!(harness.read(SCRATCH, 2), vec![0xef, 0xbe]);
        assert_eq!(
            helper_store(&mut harness.cpu, 0x1_0000_0000, 8, 1, 0x6000),
            STATUS_EXCEPTION
        );
        assert_eq!(harness.cpu.rip, 0x6000);
    }

    #[test]
    fn load_result_is_returned_in_two_registers() {
        // x0 = value, x1 = status (AAPCS64 returns a 16-byte struct of two u64 in x0/x1); the
        // emitters read them that way.
        assert_eq!(std::mem::size_of::<LoadResult>(), 16);
        assert_eq!(std::mem::offset_of!(LoadResult, val), 0);
        assert_eq!(std::mem::offset_of!(LoadResult, status), 8);
    }
}
