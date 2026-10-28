//! Several vCPUs running hand-assembled code on one machine (`testkit::run_smp`): start-up by
//! INIT and SIPI, IPIs, wake-ups from halt and MWAIT, and what must stay coherent between vCPUs
//! (translated code, soft TLBs). Every test runs on the interpreter and on the JIT.

use super::smp::Smp;
use super::*;
use crate::testkit::{Harness, SmpRun, run_smp};

/// SIPI vector 8: the start-up code runs in real mode at 0x8000.
const TRAMPOLINE: u64 = 0x8000;
const SIPI_VECTOR: u32 = 8;
/// Where the vCPUs leave notes for each other.
const FLAG: u64 = 0x1000;
const FLAG_DONE: u8 = 0x42;
const COUNTER: u64 = 0x1010;
const HANDLER_NOTE: u64 = 0x1002;
const MWAIT_WOKEN_NOTE: u64 = 0x1008;
const SMC_BEFORE: u64 = 0x1020;
const SMC_AFTER: u64 = 0x1024;
const APIC_ID_NOTES: u64 = 0x1040;
const X2APIC_ID_NOTES: u64 = 0x1050;
const CODE_PAGE: u64 = 0x3000;
const CODE_PAGE_DATA: u64 = 0x3800;

/// ICR low words: assert level, delivery mode INIT (5) or start-up (6); the shorthand sits in
/// bits 19:18 (3 = all excluding self).
const ICR_INIT: u32 = 0x4500;
const ICR_STARTUP: u32 = 0x4600 | SIPI_VECTOR;
const ICR_ALL_EXCLUDING_SELF: u32 = 3 << 18;
const IPI_VECTOR: u32 = 0x40;

// mov ecx, 0x1b; mov eax, 0xfee00d00; xor edx, edx; wrmsr; mov ecx, 0x830
// (IA32_APIC_BASE: x2APIC on; ecx stays the ICR's MSR for the writes after it)
const ENABLE_X2APIC: [u8; 19] = [
    0xb9, 0x1b, 0x00, 0x00, 0x00, 0xb8, 0x00, 0x0d, 0xe0, 0xfe, 0x31, 0xd2, 0x0f, 0x30, 0xb9, 0x30,
    0x08, 0x00, 0x00,
];
// wait: pause; cmp byte [FLAG], 0x42; jne wait
const WAIT_FLAG_PAUSE: [u8; 12] = [
    0xf3, 0x90, 0x80, 0x3c, 0x25, 0x00, 0x10, 0x00, 0x00, 0x42, 0x75, 0xf4,
];
// wait: pause; cmp byte [HANDLER_NOTE], 0x77; jne wait
const WAIT_HANDLER_NOTE: [u8; 12] = [
    0xf3, 0x90, 0x80, 0x3c, 0x25, 0x02, 0x10, 0x00, 0x00, 0x77, 0x75, 0xf4,
];
// wait: pause; cmp word [FLAG], 3; jne wait
const WAIT_THREE_ARRIVALS: [u8; 13] = [
    0xf3, 0x90, 0x66, 0x83, 0x3c, 0x25, 0x00, 0x10, 0x00, 0x00, 0x03, 0x75, 0xf3,
];
// cli; hlt
const CLI_HLT: [u8; 2] = [0xfa, 0xf4];

// mov ecx, 100000; spin: dec ecx; jnz spin
const BSP_BUSY_LOOP: [u8; 9] = [0xb9, 0xa0, 0x86, 0x01, 0x00, 0xff, 0xc9, 0x75, 0xfc];

// inc word [COUNTER]; mov byte [FLAG], 0x42; hlt
const AP_START: [u8; 10] = [0xff, 0x06, 0x10, 0x10, 0xc6, 0x06, 0x00, 0x10, 0x42, 0xf4];
// lock inc word [FLAG]; hlt
const AP_ARRIVE: [u8; 6] = [0xf0, 0xff, 0x06, 0x00, 0x10, 0xf4];
// mov ecx, 100000; spin: dec ecx; jnz spin; mov byte [FLAG], 0x42; hlt
const AP_BUSY: [u8; 16] = [
    0x66, 0xb9, 0xa0, 0x86, 0x01, 0x00, 0x66, 0x49, 0x75, 0xfc, 0xc6, 0x06, 0x00, 0x10, 0x42, 0xf4,
];
// Software-enables the x2APIC (MSR 0x80f), points IVT[0x40] at the handler, notes it is ready
// and halts with interrupts on; the handler leaves a note and returns to a `cli; hlt`.
//   jmp start; handler: mov byte [HANDLER_NOTE], 0x77; iret
//   start: xor ax, ax; mov ds, ax; mov word [0x100], 2; mov word [0x102], 0x800
//   (enable x2APIC; mov ecx, 0x80f; mov eax, 0x1ff; xor edx, edx; wrmsr)
//   mov byte [FLAG], 0x42; sti; hlt; cli; hlt
const AP_WAIT_FOR_IPI: [u8; 67] = [
    0xeb, 0x06, 0xc6, 0x06, 0x02, 0x10, 0x77, 0xcf, 0x31, 0xc0, 0x8e, 0xd8, 0xc7, 0x06, 0x00, 0x01,
    0x02, 0x00, 0xc7, 0x06, 0x02, 0x01, 0x00, 0x08, 0x66, 0xb9, 0x1b, 0x00, 0x00, 0x00, 0x66, 0xb8,
    0x00, 0x0d, 0xe0, 0xfe, 0x66, 0x31, 0xd2, 0x0f, 0x30, 0x66, 0xb9, 0x0f, 0x08, 0x00, 0x00, 0x66,
    0xb8, 0xff, 0x01, 0x00, 0x00, 0x66, 0x31, 0xd2, 0x0f, 0x30, 0xc6, 0x06, 0x00, 0x10, 0x42, 0xfb,
    0xf4, 0xfa, 0xf4,
];
// mov byte [0x1004], 1; lea rax, [0x2000]; xor ecx, ecx; xor edx, edx; monitor; sti; mwait;
// cli; mov byte [MWAIT_WOKEN_NOTE], 0x99; hlt
const BSP_MWAIT: [u8; 37] = [
    0xc6, 0x04, 0x25, 0x04, 0x10, 0x00, 0x00, 0x01, 0x48, 0x8d, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00,
    0x31, 0xc9, 0x31, 0xd2, 0x0f, 0x01, 0xc8, 0xfb, 0x0f, 0x01, 0xc9, 0xfa, 0xc6, 0x04, 0x25, 0x08,
    0x10, 0x00, 0x00, 0x99, 0xf4,
];
// wait: pause; cmp byte [0x1004], 1; jne wait; mov byte [0x2000], 1; hlt
const AP_STORE_TO_MONITORED_LINE: [u8; 15] = [
    0xf3, 0x90, 0x80, 0x3e, 0x04, 0x10, 0x01, 0x75, 0xf7, 0xc6, 0x06, 0x00, 0x20, 0x01, 0xf4,
];
// mov rax, CODE_PAGE; call rax; mov [SMC_BEFORE], eax; mov byte [0x1004], 1;
// wait: pause; cmp byte [0x1008], 1; jne wait;
// mov rax, CODE_PAGE; call rax; mov [SMC_AFTER], eax; cli; hlt
const BSP_CALL_PATCHED_CODE: [u8; 54] = [
    0x48, 0xc7, 0xc0, 0x00, 0x30, 0x00, 0x00, 0xff, 0xd0, 0x89, 0x04, 0x25, 0x20, 0x10, 0x00, 0x00,
    0xc6, 0x04, 0x25, 0x04, 0x10, 0x00, 0x00, 0x01, 0xf3, 0x90, 0x80, 0x3c, 0x25, 0x08, 0x10, 0x00,
    0x00, 0x01, 0x75, 0xf4, 0x48, 0xc7, 0xc0, 0x00, 0x30, 0x00, 0x00, 0xff, 0xd0, 0x89, 0x04, 0x25,
    0x24, 0x10, 0x00, 0x00, 0xfa, 0xf4,
];
// mov eax, 1; ret
const PATCHED_CODE: [u8; 6] = [0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3];
// wait: pause; cmp byte [0x1004], 1; jne wait; mov byte [0x3001], 2 (the immediate of the
// `mov eax, 1` above); mov byte [0x1008], 1; hlt
const AP_PATCH_CODE: [u8; 20] = [
    0xf3, 0x90, 0x80, 0x3e, 0x04, 0x10, 0x01, 0x75, 0xf7, 0xc6, 0x06, 0x01, 0x30, 0x02, 0xc6, 0x06,
    0x08, 0x10, 0x01, 0xf4,
];
// mov byte [CODE_PAGE_DATA], 1 / 2
const STORE_TO_CODE_PAGE_DATA_FIRST: [u8; 8] = [0xc6, 0x04, 0x25, 0x00, 0x38, 0x00, 0x00, 0x01];
const STORE_TO_CODE_PAGE_DATA_AGAIN: [u8; 8] = [0xc6, 0x04, 0x25, 0x00, 0x38, 0x00, 0x00, 0x02];
// ljmp 0:CODE_PAGE
const AP_JUMP_TO_CODE_PAGE: [u8; 5] = [0xea, 0x00, 0x30, 0x00, 0x00];
// mov byte [FLAG], 0x42; hlt (run by the AP from the code page)
const CODE_PAGE_FLAG_AND_HALT: [u8; 6] = [0xc6, 0x06, 0x00, 0x10, 0x42, 0xf4];
// mov eax, 1; cpuid; shr ebx, 24; and ebx, 0xff; mov esi, ebx; mov byte [si+0x1040], 0x7f;
// mov eax, 0xb; mov ecx, 1; cpuid; mov [si+0x1050], dl; hlt
const AP_REPORT_IDS: [u8; 46] = [
    0x66, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x0f, 0xa2, 0x66, 0xc1, 0xeb, 0x18, 0x66, 0x81, 0xe3, 0xff,
    0x00, 0x00, 0x00, 0x66, 0x89, 0xde, 0xc6, 0x84, 0x40, 0x10, 0x7f, 0x66, 0xb8, 0x0b, 0x00, 0x00,
    0x00, 0x66, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x0f, 0xa2, 0x88, 0x94, 0x50, 0x10, 0xf4,
];

/// `wrmsr` of the ICR (MSR 0x830, in ecx since `ENABLE_X2APIC`): destination in the high word.
fn send_icr(destination: u32, low: u32) -> Vec<u8> {
    let mut code = vec![0xba];
    code.extend_from_slice(&destination.to_le_bytes());
    code.push(0xb8);
    code.extend_from_slice(&low.to_le_bytes());
    code.extend_from_slice(&[0x0f, 0x30]);
    code
}

fn program(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

fn place(cpu: &mut Cpu, address: u64, bytes: &[u8]) {
    let start = address as usize;
    cpu.ram[start..start + bytes.len()].copy_from_slice(bytes);
}

fn both_engines(test: impl Fn(bool)) {
    test(false);
    test(true);
}

fn assert_shut_down(run: &SmpRun) {
    assert!(
        matches!(run.exit, ExitReason::Shutdown),
        "{:?}\n{}",
        run.exit,
        run.smp.summary()
    );
}

#[test]
fn a_secondary_vcpu_waits_for_a_startup_ipi_and_owns_no_ram() {
    let smp = Smp::new(
        Harness::with_devices(
            &[0xf4],
            crate::dev::DevConfig {
                cpus: 3,
                ..Default::default()
            },
        )
        .cpu,
    );
    let parked: Vec<_> = smp
        .cpus
        .iter()
        .map(|cpu| (cpu.cpu_index, cpu.halted, cpu.wait_for_sipi))
        .collect();
    assert_eq!(
        parked,
        [(0, false, false), (1, true, true), (2, true, true)]
    );
    assert!(smp.cpus[1].ram.is_empty());
}

#[test]
fn an_ap_starts_in_real_mode_at_the_sipi_vector_and_a_second_sipi_is_ignored() {
    both_engines(|jit| {
        // The AP is halted by the time the second SIPI comes, but it is not waiting for one.
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &send_icr(1, ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_START));
        assert_shut_down(&run);
        assert_eq!(run.read(FLAG, 1), [FLAG_DONE]);
        assert_eq!(run.read_u16(COUNTER), 1, "the AP ran once");
    });
}

#[test]
fn a_halted_ap_wakes_for_a_fixed_ipi_and_takes_it_through_its_ivt() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &send_icr(1, IPI_VECTOR),
            &WAIT_HANDLER_NOTE,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_WAIT_FOR_IPI));
        assert_shut_down(&run);
        assert_eq!(run.read(HANDLER_NOTE, 1), [0x77]);
    });
}

#[test]
fn a_shorthand_ipi_starts_every_other_vcpu() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
            &WAIT_THREE_ARRIVALS,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_ARRIVE));
        assert_shut_down(&run);
        assert_eq!(run.read_u16(FLAG), 3);
    });
}

#[test]
fn each_vcpu_sees_its_own_apic_id_in_cpuid() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_REPORT_IDS));
        assert_shut_down(&run);
        assert_eq!(run.read(APIC_ID_NOTES, 4), [0, 0x7f, 0x7f, 0x7f]);
        assert_eq!(run.read(X2APIC_ID_NOTES, 4), [0, 1, 2, 3]);
    });
}

#[test]
fn the_machine_keeps_running_while_the_bsp_is_halted_and_an_ap_works() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_BUSY));
        assert_shut_down(&run);
        assert_eq!(run.read(FLAG, 1), [FLAG_DONE], "the AP finished its loop");
        assert!(
            run.smp.active().icount > 2 * EVENT_INTERVAL,
            "the loop spans several quanta"
        );
    });
}

#[test]
fn pause_hands_the_cpu_to_the_vcpu_being_waited_for() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_START));
        assert_shut_down(&run);
        assert!(
            run.smp.active().icount < EVENT_INTERVAL / 2,
            "the spin loop burnt its quantum: {} instructions",
            run.smp.active().icount
        );
    });
}

#[test]
fn a_store_to_the_monitored_line_wakes_an_mwait() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_MWAIT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_STORE_TO_MONITORED_LINE)
        });
        assert_shut_down(&run);
        assert_eq!(run.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
    });
}

#[test]
fn code_another_vcpu_overwrites_is_translated_again() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_CALL_PATCHED_CODE,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_PATCH_CODE);
            place(cpu, CODE_PAGE, &PATCHED_CODE);
        });
        assert_shut_down(&run);
        assert_eq!(run.read_u32(SMC_BEFORE), 1);
        assert_eq!(run.read_u32(SMC_AFTER), 2, "a stale block ran");
    });
}

#[test]
fn a_vcpu_cannot_store_past_a_page_another_vcpu_made_code() {
    both_engines(|jit| {
        // The BSP's TLB gets a write tag for the code page's data half, then the AP runs code
        // from that page; the BSP's next store must still notice (`code_written`).
        let bsp = program(&[
            &ENABLE_X2APIC,
            &STORE_TO_CODE_PAGE_DATA_FIRST,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &STORE_TO_CODE_PAGE_DATA_AGAIN,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_JUMP_TO_CODE_PAGE);
            place(cpu, CODE_PAGE, &CODE_PAGE_FLAG_AND_HALT);
        });
        assert_shut_down(&run);
        assert_eq!(run.read(CODE_PAGE_DATA, 1), [2]);
        assert_eq!(
            run.smp.active().page_has_code[(CODE_PAGE >> PAGE_SHIFT) as usize],
            0,
            "the store skipped `code_written`"
        );
    });
}

#[test]
fn guest_time_advances_by_the_instructions_of_one_vcpu_when_two_run_side_by_side() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_BUSY_LOOP,
            &WAIT_FLAG_PAUSE,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_BUSY));
        assert_shut_down(&run);
        let (retired, now) = (run.smp.active().icount, run.smp.active().now_ns());
        // Both loops retire 200k instructions; they overlap almost completely.
        assert!(retired > 350_000, "{retired}");
        assert!(now < retired * 3 / 5, "{now} ns for {retired} instructions");
        assert!(now > retired * 2 / 5, "{now} ns for {retired} instructions");
    });
}

#[test]
fn guest_time_never_runs_backwards_across_switches() {
    // `now_ns` read after every slice of a busy two-vCPU run.
    let bsp = program(&[
        &ENABLE_X2APIC,
        &send_icr(1, ICR_INIT),
        &send_icr(1, ICR_STARTUP),
        &BSP_BUSY_LOOP,
        &WAIT_FLAG_PAUSE,
        &CLI_HLT,
    ]);
    let mut run = run_smp(&bsp, 2, false, |cpu| place(cpu, TRAMPOLINE, &AP_BUSY));
    assert_shut_down(&run);
    // Replay the folding the scheduler does with changing divisors.
    let cpu = run.smp.active_mut();
    let mut last = cpu.now_ns();
    for divisor in [2, 2, 1, 3, 3, 1, 2] {
        cpu.icount += 1000;
        let during = cpu.now_ns();
        assert!(during >= last);
        cpu.start_slice(divisor);
        assert!(cpu.now_ns() >= during);
        last = cpu.now_ns();
    }
}
