//! Several vCPUs running hand-assembled code on one machine (`testkit::run_smp`): start-up by
//! INIT and SIPI, IPIs, wake-ups from halt and MWAIT, and what must stay coherent between vCPUs
//! (translated code, soft TLBs). Every test runs on the interpreter and on the JIT.

use super::smp::Smp;
use super::*;
use crate::testkit::{Harness, SmpRun, ThreadRun, run_smp, run_threads};

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
// The store lands between MONITOR and MWAIT: arms the monitor, tells the AP to go, waits until
// its store is visible, and only then executes MWAIT, which must return at once.
//   mov byte [0x1004], 1; lea rax, [0x2000]; xor ecx, ecx; xor edx, edx; monitor;
//   wait: pause; cmp byte [0x2000], 1; jne wait; sti; mwait; cli; mov byte [0x1008], 0x99; hlt
const BSP_MONITOR_THEN_MWAIT_AFTER_THE_STORE: [u8; 50] = [
    0xc6, 0x04, 0x25, 0x04, 0x10, 0x00, 0x00, 0x01, 0x48, 0x8d, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00,
    0x31, 0xc9, 0x31, 0xd2, 0x0f, 0x01, 0xc8, 0xf3, 0x90, 0x80, 0x3c, 0x25, 0x00, 0x20, 0x00, 0x00,
    0x01, 0x75, 0xf4, 0xfb, 0x0f, 0x01, 0xc9, 0xfa, 0xc6, 0x04, 0x25, 0x08, 0x10, 0x00, 0x00, 0x99,
    0xf4, 0xf4,
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
fn a_secondary_vcpu_waits_for_a_startup_ipi_and_shares_the_ram() {
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
    assert_eq!(smp.cpus[1].ram.as_ptr(), smp.cpus[0].ram.as_ptr());
    assert_eq!(smp.cpus[1].ram.len(), smp.cpus[0].ram.len());
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

// ---- one host thread per vCPU ------------------------------------------------------------

/// Where the threaded tests keep their counters, clear of the single-thread tests' notes.
const ATOMIC_COUNTER: u64 = 0x1100;
const LOCK_WORD: u64 = 0x1104;
const PROTECTED_COUNTER: u64 = 0x1108;
const DONE_COUNTER: u64 = 0x1110;
const MESSAGE_FLAG: u64 = 0x1240;
const MESSAGE_ERRORS: u64 = 0x1128;

const ATOMIC_INCREMENTS_PER_CPU: u32 = 20_000;
const LOCKED_SECTIONS_PER_CPU: u32 = 3_000;
const MESSAGES: u32 = 5_000_000;
const AP_COUNT: u32 = 3;

// Real mode (DS = 0), run by each AP:
//   mov ecx, 20000; l: lock inc dword [ATOMIC_COUNTER]; dec ecx; jnz l; lock inc dword [DONE]; hlt
const AP_LOCKED_INCREMENTS: [u8; 23] = [
    0x66, 0xb9, 0x20, 0x4e, 0x00, 0x00, 0x66, 0xf0, 0xff, 0x06, 0x00, 0x11, 0x66, 0x49, 0x75, 0xf6,
    0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// A test-and-set lock built on `lock cmpxchg`, released with a plain store, guarding a counter
// that is updated with a plain load, add and store; 3000 rounds, then done.
const AP_CMPXCHG_LOCK: [u8; 58] = [
    0x66, 0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x66, 0x31, 0xc0, 0x66, 0xba, 0x01, 0x00, 0x00,
    0x00, 0x66, 0xf0, 0x0f, 0xb1, 0x16, 0x04, 0x11, 0x75, 0xec, 0x66, 0xa1, 0x08, 0x11, 0x66, 0x83,
    0xc0, 0x01, 0x66, 0xa3, 0x08, 0x11, 0x66, 0xc7, 0x06, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x66,
    0x49, 0x75, 0xd3, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// The same with `xchg` (implicitly locked, no prefix) as the lock.
const AP_XCHG_LOCK: [u8; 56] = [
    0x66, 0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x66, 0xba, 0x01, 0x00, 0x00, 0x00, 0x66, 0x87,
    0x16, 0x04, 0x11, 0x66, 0x85, 0xd2, 0x75, 0xee, 0x66, 0xa1, 0x08, 0x11, 0x66, 0x83, 0xc0, 0x01,
    0x66, 0xa3, 0x08, 0x11, 0x66, 0xc7, 0x06, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x66, 0x49, 0x75,
    0xd5, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// Waits for the consumer, then for i in 1..=5000000 { MESSAGE_DATA = i; MESSAGE_FLAG = i } (two
// plain stores to different cache lines), then is done.
const AP_MESSAGE_PRODUCER: [u8; 46] = [
    0xf3, 0x90, 0x66, 0x83, 0x3e, 0x80, 0x12, 0x01, 0x75, 0xf6, 0x66, 0xb9, 0x01, 0x00, 0x00, 0x00,
    0x66, 0x89, 0x0e, 0x00, 0x12, 0x66, 0x89, 0x0e, 0x40, 0x12, 0x66, 0x83, 0xc1, 0x01, 0x66, 0x81,
    0xf9, 0x41, 0x4b, 0x4c, 0x00, 0x72, 0xe9, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// 64-bit, run by the BSP: 20000 `lock inc`s, then waits for three APs to be done; cli; hlt
const BSP_LOCKED_INCREMENTS_AND_WAIT: [u8; 31] = [
    0xb9, 0x20, 0x4e, 0x00, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x00, 0x11, 0x00, 0x00, 0xff, 0xc9, 0x75,
    0xf4, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// The BSP takes the same lock 3000 times, then waits for three APs.
const BSP_CMPXCHG_LOCK_AND_WAIT: [u8; 71] = [
    0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x31, 0xc0, 0xba, 0x01, 0x00, 0x00, 0x00, 0xf0, 0x0f,
    0xb1, 0x14, 0x25, 0x04, 0x11, 0x00, 0x00, 0x75, 0xec, 0x8b, 0x04, 0x25, 0x08, 0x11, 0x00, 0x00,
    0x83, 0xc0, 0x01, 0x89, 0x04, 0x25, 0x08, 0x11, 0x00, 0x00, 0xc7, 0x04, 0x25, 0x04, 0x11, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xcc, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11,
    0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// 64-bit, run by the BSP: says it is ready, then reads MESSAGE_FLAG and then MESSAGE_DATA over
// and over, counting the rounds where the data is older than the flag, until the flag says
// 5000000; stores the count and waits for the producer to be done.
const BSP_MESSAGE_CONSUMER_AND_WAIT: [u8; 65] = [
    0xc7, 0x04, 0x25, 0x80, 0x12, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xbe, 0x00, 0x00, 0x00, 0x00,
    0x8b, 0x04, 0x25, 0x40, 0x12, 0x00, 0x00, 0x8b, 0x1c, 0x25, 0x00, 0x12, 0x00, 0x00, 0x39, 0xc3,
    0x73, 0x03, 0x83, 0xc6, 0x01, 0x3d, 0x40, 0x4b, 0x4c, 0x00, 0x72, 0xe4, 0x89, 0x34, 0x25, 0x28,
    0x11, 0x00, 0x00, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x01, 0x75, 0xf4, 0xfa,
    0xf4,
];

/// Starts every other vCPU at `TRAMPOLINE` and then runs `work`.
fn start_the_others_then(work: &[u8]) -> Vec<u8> {
    program(&[
        &ENABLE_X2APIC,
        &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
        &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
        work,
    ])
}

fn assert_thread_shutdown(run: &ThreadRun) {
    assert!(
        matches!(run.exit, ExitReason::Shutdown),
        "{:?}\n{}",
        run.exit,
        run.summary
    );
}

#[test]
fn every_vcpu_on_its_own_thread_reports_in_and_the_machine_shuts_down() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&program(&[&WAIT_THREE_ARRIVALS, &CLI_HLT]));
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_ARRIVE));
        assert_thread_shutdown(&run);
        assert_eq!(run.read(FLAG, 2), [3, 0]);
    });
}

#[test]
fn locked_increments_from_four_threads_lose_no_update() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_LOCKED_INCREMENTS_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_LOCKED_INCREMENTS)
        });
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(ATOMIC_COUNTER),
            (AP_COUNT + 1) * ATOMIC_INCREMENTS_PER_CPU
        );
    });
}

#[test]
fn a_cmpxchg_lock_released_by_a_plain_store_keeps_a_plain_counter_exact() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_CMPXCHG_LOCK_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_CMPXCHG_LOCK));
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(PROTECTED_COUNTER),
            (AP_COUNT + 1) * LOCKED_SECTIONS_PER_CPU
        );
        assert_eq!(run.read_u32(LOCK_WORD), 0);
    });
}

#[test]
fn xchg_with_memory_is_locked_without_a_prefix() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_CMPXCHG_LOCK_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_XCHG_LOCK));
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(PROTECTED_COUNTER),
            (AP_COUNT + 1) * LOCKED_SECTIONS_PER_CPU
        );
    });
}

#[test]
fn data_written_before_a_flag_is_never_read_stale_after_the_flag() {
    // Message passing: the producer's two plain stores must become visible in order, and the
    // consumer's two plain loads must not be reordered. The consumer is the BSP, whose JIT code
    // is what fails on an aarch64 host with plain `ldr`; Rosetta runs x86 with hardware ordering
    // and cannot catch it.
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_MESSAGE_CONSUMER_AND_WAIT);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_MESSAGE_PRODUCER)
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(MESSAGE_FLAG), MESSAGES);
        assert_eq!(run.read_u32(MESSAGE_ERRORS), 0);
    });
}

// A serializing `cpuid` after the flag: what a guest does between writing code on one vCPU and
// running it on another (the rest of the protocol is the IPI that Linux's `text_poke` sends).
// xor eax, eax; cpuid
const SERIALIZE: [u8; 4] = [0x31, 0xc0, 0x0f, 0xa2];

/// `BSP_CALL_PATCHED_CODE` with `SERIALIZE` between the wait and the second call.
fn bsp_call_patched_code_serialized() -> Vec<u8> {
    let mut code = BSP_CALL_PATCHED_CODE.to_vec();
    const AFTER_WAIT: usize = 36;
    code.splice(AFTER_WAIT..AFTER_WAIT, SERIALIZE);
    code
}

#[test]
fn a_halted_ap_on_its_own_thread_takes_an_ipi_through_its_ivt() {
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
        let run = run_threads(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_WAIT_FOR_IPI));
        assert_thread_shutdown(&run);
        assert_eq!(run.read(HANDLER_NOTE, 1), [0x77]);
    });
}

#[test]
fn each_threaded_vcpu_sees_its_own_apic_id_in_cpuid() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_REPORT_IDS));
        assert_thread_shutdown(&run);
        assert_eq!(run.read(APIC_ID_NOTES, 4), [0, 0x7f, 0x7f, 0x7f]);
        assert_eq!(run.read(X2APIC_ID_NOTES, 4), [0, 1, 2, 3]);
    });
}

#[test]
fn a_store_to_the_monitored_line_wakes_an_mwait_on_another_thread() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_MWAIT,
        ]);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_STORE_TO_MONITORED_LINE)
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
    });
}

#[test]
fn code_another_thread_overwrites_is_translated_again() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &bsp_call_patched_code_serialized(),
        ]);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_PATCH_CODE);
            place(cpu, CODE_PAGE, &PATCHED_CODE);
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(SMC_BEFORE), 1);
        assert_eq!(run.read_u32(SMC_AFTER), 2, "a stale block ran");
    });
}

#[test]
fn a_thread_cannot_store_past_a_page_another_thread_made_code() {
    both_engines(|jit| {
        // The BSP's TLB gets a write tag for the code page's data half, then the AP runs code
        // from that page: flagging the page revokes the BSP's tag from the AP's thread, so the
        // BSP's next store still reaches `code_written`.
        let bsp = program(&[
            &ENABLE_X2APIC,
            &STORE_TO_CODE_PAGE_DATA_FIRST,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &STORE_TO_CODE_PAGE_DATA_AGAIN,
            &CLI_HLT,
        ]);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_JUMP_TO_CODE_PAGE);
            place(cpu, CODE_PAGE, &CODE_PAGE_FLAG_AND_HALT);
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read(CODE_PAGE_DATA, 1), [2]);
        assert_eq!(
            run.bsp.page_has_code[(CODE_PAGE >> PAGE_SHIFT) as usize],
            0,
            "the store skipped `code_written`"
        );
    });
}

/// Real-mode pages that each jump to the next: the first fetch from a page flags it as code,
/// and four vCPUs do that to the same pages at the same time.
const CHAIN_BASE: u64 = 0x20000;
const CHAIN_PAGES: u64 = 64;
const PAGE_BYTES: u64 = 0x1000;
// wait: pause; cmp dword [DONE], 3; jne wait; cli; hlt
const BSP_WAIT_FOR_THREE: [u8; 17] = [
    0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4, 0xf4, 0xf4,
    0xf4,
];
// lock inc dword [DONE]; hlt
const CHAIN_END: [u8; 7] = [0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4];

fn jump_far(segment: u16) -> [u8; 5] {
    let [low, high] = segment.to_le_bytes();
    [0xea, 0x00, 0x00, low, high]
}

#[test]
fn threads_flagging_the_same_pages_at_once_neither_deadlock_nor_corrupt_each_other() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_WAIT_FOR_THREE);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &jump_far((CHAIN_BASE >> 4) as u16));
            for page in 0..CHAIN_PAGES {
                let at = CHAIN_BASE + page * PAGE_BYTES;
                if page + 1 < CHAIN_PAGES {
                    place(cpu, at, &jump_far(((at + PAGE_BYTES) >> 4) as u16));
                } else {
                    place(cpu, at, &CHAIN_END);
                }
            }
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(DONE_COUNTER), AP_COUNT);
        for page in 0..CHAIN_PAGES {
            let flag = run.bsp.page_has_code[((CHAIN_BASE >> 12) + page) as usize];
            assert_eq!(flag, 1, "page {page} was executed but not flagged");
        }
    });
}

#[test]
fn a_store_between_monitor_and_mwait_makes_mwait_return_at_once() {
    // The line is armed by MONITOR, not by MWAIT: with threads the other vCPU's store can land
    // in between, and a vCPU that snapshotted at MWAIT would sleep through it forever.
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_MONITOR_THEN_MWAIT_AFTER_THE_STORE,
        ]);
        let place_ap = |cpu: &mut Cpu| place(cpu, TRAMPOLINE, &AP_STORE_TO_MONITORED_LINE);
        let timeshared = run_smp(&bsp, 2, jit, place_ap);
        assert_shut_down(&timeshared);
        assert_eq!(timeshared.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
        let threaded = run_threads(&bsp, 2, jit, place_ap);
        assert_thread_shutdown(&threaded);
        assert_eq!(threaded.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
    });
}

/// A dword that is misaligned but inside one page, and one that straddles two pages: no host
/// atomic covers either, so the locked instruction runs with the other vCPUs stopped.
const MISALIGNED_COUNTER: u64 = 0x1101;
const STRADDLING_COUNTER: u64 = 0x1ffe;

// lock inc dword [MISALIGNED_COUNTER] 20000 times, then done
const AP_LOCKED_MISALIGNED: [u8; 23] = [
    0x66, 0xb9, 0x20, 0x4e, 0x00, 0x00, 0x66, 0xf0, 0xff, 0x06, 0x01, 0x11, 0x66, 0x49, 0x75, 0xf6,
    0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
const BSP_LOCKED_MISALIGNED_AND_WAIT: [u8; 31] = [
    0xb9, 0x20, 0x4e, 0x00, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x01, 0x11, 0x00, 0x00, 0xff, 0xc9, 0x75,
    0xf4, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// lock add dword [STRADDLING_COUNTER], 1 20000 times, then done
const AP_LOCKED_STRADDLING: [u8; 24] = [
    0x66, 0xb9, 0x20, 0x4e, 0x00, 0x00, 0x66, 0xf0, 0x83, 0x06, 0xfe, 0x1f, 0x01, 0x66, 0x49, 0x75,
    0xf5, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
const BSP_LOCKED_STRADDLING_AND_WAIT: [u8; 32] = [
    0xb9, 0x20, 0x4e, 0x00, 0x00, 0xf0, 0x83, 0x04, 0x25, 0xfe, 0x1f, 0x00, 0x00, 0x01, 0xff, 0xc9,
    0x75, 0xf3, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// A spin lock taken with `lock bts [LOCK_WORD], 0` and released by a plain store, guarding a plain
// counter, 3000 rounds each.
const AP_BTS_LOCK: [u8; 50] = [
    0x66, 0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x66, 0xf0, 0x0f, 0xba, 0x2e, 0x04, 0x11, 0x00,
    0x72, 0xf4, 0x66, 0xa1, 0x08, 0x11, 0x66, 0x83, 0xc0, 0x01, 0x66, 0xa3, 0x08, 0x11, 0x66, 0xc7,
    0x06, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x66, 0x49, 0x75, 0xdb, 0x66, 0xf0, 0xff, 0x06, 0x10,
    0x11, 0xf4,
];
const BSP_BTS_LOCK_AND_WAIT: [u8; 66] = [
    0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0xf0, 0x0f, 0xba, 0x2c, 0x25, 0x04, 0x11, 0x00, 0x00,
    0x00, 0x72, 0xf2, 0x8b, 0x04, 0x25, 0x08, 0x11, 0x00, 0x00, 0x83, 0xc0, 0x01, 0x89, 0x04, 0x25,
    0x08, 0x11, 0x00, 0x00, 0xc7, 0x04, 0x25, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0xc9, 0x75, 0xd2, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa,
    0xf4, 0xf4,
];

#[test]
fn a_locked_increment_of_a_misaligned_operand_loses_no_update() {
    // Not one host atomic: the instruction runs with the other vCPUs stopped. Running that from
    // inside the exclusive section used to wait for itself forever.
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_LOCKED_MISALIGNED_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_LOCKED_MISALIGNED)
        });
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(MISALIGNED_COUNTER),
            (AP_COUNT + 1) * ATOMIC_INCREMENTS_PER_CPU
        );
    });
}

#[test]
fn a_locked_add_straddling_two_pages_loses_no_update() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_LOCKED_STRADDLING_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_LOCKED_STRADDLING)
        });
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(STRADDLING_COUNTER),
            (AP_COUNT + 1) * ATOMIC_INCREMENTS_PER_CPU
        );
    });
}

#[test]
fn a_lock_bts_spin_lock_keeps_a_plain_counter_exact() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_BTS_LOCK_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_BTS_LOCK));
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(PROTECTED_COUNTER),
            (AP_COUNT + 1) * LOCKED_SECTIONS_PER_CPU
        );
    });
}
