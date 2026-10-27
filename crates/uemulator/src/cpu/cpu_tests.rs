use std::sync::atomic::AtomicUsize;

use super::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE};
use super::flags::*;
use super::test_support::*;
use super::*;
use crate::testkit::{CODE_BASE, HLT, Harness, RBX, SCRATCH, STACK_TOP};

const OLD_RIP: u64 = 0x0000_7000_1234_5678;
/// A handler that asks the keyboard controller for a reset and halts.
// mov al, 0xfe; out 0x64, al; hlt
const RESET_HANDLER: [u8; 5] = [0xb0, 0xfe, 0xe6, 0x64, 0xf4];
const HANDLER_ADDRESS: u64 = 0x30_0000;
const KERNEL_STACK: u64 = 0x70_0000;
// l: jmp l
const SPIN: [u8; 2] = [0xeb, 0xfe];
// mov rax, 5; add rax, 7; mov al, 0xfe; out 0x64, al; l: jmp l
const COMPUTE_THEN_RESET: [u8; 17] = [
    0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x83, 0xc0, 0x07, 0xb0, 0xfe, 0xe6, 0x64, 0xeb,
    0xfe,
];
const PIC_MASTER_COMMAND: u16 = 0x20;
const PIC_MASTER_DATA: u16 = 0x21;
const PIC_SLAVE_COMMAND: u16 = 0xa0;
const PIC_SLAVE_DATA: u16 = 0xa1;
const PIT_CHANNEL0_PORT: u16 = 0x40;
const PIT_COMMAND_PORT: u16 = 0x43;
const KEYBOARD_CONTROLLER_COMMAND_PORT: u16 = 0x64;
const KEYBOARD_CONTROLLER_RESET: u32 = 0xfe;
/// Channel 0, low then high byte access, mode 2 (rate generator).
const PIT_CHANNEL0_RATE_GENERATOR: u32 = 0x34;
const IRQ_VECTOR_BASE: u8 = 0x20;
const ICW1_INIT_WITH_ICW4: u32 = 0x11;
const ICW4_8086: u32 = 1;
const ICW3_SLAVE_ON_IRQ2: u32 = 0x04;
const ICW3_SLAVE_ID_2: u32 = 0x02;

fn harness() -> Harness {
    Harness::new(&[HLT])
}

fn with_idt() -> Harness {
    let mut h = harness();
    install_idt(&mut h);
    h.cpu.eflags = RESERVED_ONE | IF;
    h.cpu.rip = OLD_RIP;
    h
}

/// Standard PC wiring: master at `IRQ_VECTOR_BASE`, slave right after, all lines unmasked.
fn init_pics(cpu: &mut Cpu) {
    let devices = &mut cpu.dev;
    devices.io_out(PIC_MASTER_COMMAND, 1, ICW1_INIT_WITH_ICW4, 0);
    devices.io_out(PIC_MASTER_DATA, 1, IRQ_VECTOR_BASE as u32, 0);
    devices.io_out(PIC_MASTER_DATA, 1, ICW3_SLAVE_ON_IRQ2, 0);
    devices.io_out(PIC_MASTER_DATA, 1, ICW4_8086, 0);
    devices.io_out(PIC_SLAVE_COMMAND, 1, ICW1_INIT_WITH_ICW4, 0);
    devices.io_out(PIC_SLAVE_DATA, 1, IRQ_VECTOR_BASE as u32 + 8, 0);
    devices.io_out(PIC_SLAVE_DATA, 1, ICW3_SLAVE_ID_2, 0);
    devices.io_out(PIC_SLAVE_DATA, 1, ICW4_8086, 0);
}

fn program_pit_channel0(cpu: &mut Cpu, count: u16) {
    cpu.dev
        .io_out(PIT_COMMAND_PORT, 1, PIT_CHANNEL0_RATE_GENERATOR, 0);
    cpu.dev
        .io_out(PIT_CHANNEL0_PORT, 1, (count & 0xff) as u32, 0);
    cpu.dev.io_out(PIT_CHANNEL0_PORT, 1, (count >> 8) as u32, 0);
}

fn install_reset_handler(h: &mut Harness, vector: u8) {
    h.write(HANDLER_ADDRESS, &RESET_HANDLER);
    set_gate(h, vector, Gate::interrupt(HANDLER_ADDRESS));
}

mod reset_state {
    use super::*;

    #[test]
    fn control_registers_describe_long_mode_with_paging_on() {
        let h = harness();
        assert_eq!(h.cpu.cr0, 0x8000_0033);
        assert_eq!(h.cpu.cr4, 0x20);
        assert_eq!(h.cpu.efer, 0x500);
        assert_eq!(h.cpu.cr2, 0);
        assert_eq!(h.cpu.cr8, 0);
        assert_eq!(h.cpu.cr3 & PTE_ADDRESS_BITS, 0x9000);
    }

    #[test]
    fn long_mode_is_enabled_without_no_execute_or_write_protect() {
        let h = harness();
        assert_eq!(h.cpu.efer & EFER_NXE, 0);
        assert_eq!(h.cpu.cr0 & CR0_WP, 0);
        assert_ne!(h.cpu.efer & EFER_LMA, 0);
        assert_ne!(h.cpu.efer & EFER_LME, 0);
        assert_ne!(h.cpu.cr4 & CR4_PAE, 0);
    }

    #[test]
    fn segments_are_flat_with_the_boot_selectors() {
        let h = harness();
        let cs = h.cpu.seg[SEG_CS];
        assert_eq!(cs.sel, KERNEL_CS);
        assert_eq!(cs.flags, 0x0020_9a00);
        assert!(cs.is_code() && cs.long() && cs.present());
        assert_eq!(cs.dpl(), 0);
        for index in [SEG_ES, SEG_SS, SEG_DS] {
            let seg = h.cpu.seg[index];
            assert_eq!(seg.sel, KERNEL_DS, "segment {index}");
            assert_eq!(seg.flags, 0x00cf_9200, "segment {index}");
            assert_eq!(seg.base, 0);
            assert_eq!(seg.limit, 0xffff_ffff);
            assert!(!seg.is_code());
        }
        for index in [SEG_FS, SEG_GS] {
            let seg = h.cpu.seg[index];
            assert_eq!((seg.sel, seg.base, seg.limit, seg.flags), (0, 0, 0, 0));
        }
    }

    #[test]
    fn descriptor_tables_point_at_the_boot_gdt_and_an_empty_idt() {
        let h = harness();
        assert_eq!(h.cpu.gdtr.base, GDT_BASE);
        assert_eq!(h.cpu.gdtr.limit, 31);
        assert_eq!((h.cpu.idtr.base, h.cpu.idtr.limit), (0, 0));
        assert_eq!((h.cpu.ldtr.sel, h.cpu.tr.sel), (0, 0));
    }

    #[test]
    fn registers_start_at_the_boot_protocol_values() {
        let h = harness();
        assert_eq!(h.cpu.rip, CODE_BASE);
        assert_eq!(h.cpu.gpr[GPR_RSI], 0x7000, "rsi points at boot_params");
        for (index, value) in h.cpu.gpr.iter().enumerate() {
            if index != GPR_RSI && index != GPR_RSP {
                assert_eq!(*value, 0, "register {index}");
            }
        }
        assert_eq!(h.cpu.gpr[GPR_RSP], STACK_TOP);
        assert_eq!(h.cpu.eflags_full(), RESERVED_ONE);
        assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
        assert_eq!(h.cpu.cpl, 0);
    }

    #[test]
    fn msrs_debug_and_fpu_state_have_their_documented_power_on_values() {
        let h = harness();
        assert_eq!(h.cpu.pat, 0x0007_0406_0007_0406);
        assert_eq!(h.cpu.dr[6], 0xffff_0ff0);
        assert_eq!(h.cpu.dr[7], 0x400);
        assert_eq!(&h.cpu.dr[..6], &[0; 6]);
        assert_eq!(h.cpu.mxcsr, 0x1f80);
        assert_eq!(h.cpu.fcw, 0x37f);
        assert_eq!((h.cpu.fsw, h.cpu.ftw), (0, 0));
        assert_eq!(h.cpu.xmm, [0; 16]);
        assert_eq!(h.cpu.st, [[0; 10]; 8]);
        assert_eq!(
            (h.cpu.star, h.cpu.lstar, h.cpu.cstar, h.cpu.sfmask),
            (0, 0, 0, 0)
        );
        assert_eq!((h.cpu.kgsbase, h.cpu.tsc_aux), (0, 0));
        assert_eq!(h.cpu.sysenter, [0; 3]);
    }

    #[test]
    fn stored_only_msrs_are_seeded() {
        let h = harness();
        let expected: [(u32, u64); 5] = [
            (0x8b, 0),
            (0xc001_0015, 1 << 24),
            (0xc001_001f, 0),
            (0xc001_1020, 0),
            (0xc001_1029, 0),
        ];
        assert_eq!(h.cpu.msr_misc.len(), expected.len());
        for (msr, value) in expected {
            assert_eq!(h.cpu.msr_misc.get(&msr), Some(&value), "msr {msr:#x}");
        }
    }

    #[test]
    fn caches_and_counters_start_empty() {
        let h = harness();
        assert_eq!(h.cpu.tlb.len(), 2 * TLB_ENTRIES_PER_MODE);
        assert!(h.cpu.tlb.iter().all(|entry| entry.tag_r == TLB_TAG_INVALID
            && entry.tag_w == TLB_TAG_INVALID
            && entry.tag_x == TLB_TAG_INVALID));
        assert_eq!(h.cpu.decode_cache.len(), DECODE_CACHE_ENTRIES);
        assert!(h.cpu.decode_cache.iter().all(|entry| entry.generation == 0));
        assert_eq!(h.cpu.decode_cache_generation, 1);
        assert_eq!(h.cpu.page_has_code.len(), h.cpu.ram.len() >> PAGE_SHIFT);
        assert!(h.cpu.page_has_code.iter().all(|&flag| flag == 0));
        assert!(h.cpu.pages_with_code.is_empty());
        assert_eq!(
            (h.cpu.icount, h.cpu.idle_skipped_ns, h.cpu.next_event),
            (0, 0, 0)
        );
        assert_eq!(h.cpu.flushes, 0);
    }

    #[test]
    fn execution_flags_and_hooks_start_clear() {
        let h = harness();
        assert!(!h.cpu.halted);
        assert!(!h.cpu.interrupt_shadow);
        assert!(!h.cpu.block_exit_requested);
        assert!(!h.cpu.realtime);
        assert_eq!(h.cpu.prepaid_icount, 0);
        assert_eq!(h.cpu.last_block_insns, 0);
        assert!(h.cpu.pending.is_none());
        assert!(h.cpu.jit.is_none());
        assert!(h.cpu.trace.is_none());
        assert!(h.cpu.poll.is_none());
        assert!(h.cpu.log.is_none());
        assert!(h.cpu.logged.is_empty());
    }

    #[test]
    fn ram_base_and_the_supervisor_tlb_half_point_into_owned_buffers() {
        let h = harness();
        assert_eq!(h.cpu.ram_base, h.cpu.ram.as_ptr() as u64);
        assert_eq!(h.cpu.tlb_active_base, h.cpu.tlb.as_ptr() as u64);
        assert_eq!(h.cpu.ram.len(), crate::testkit::HARNESS_RAM_BYTES);
    }

    #[test]
    fn a_fresh_cpu_walks_the_boot_identity_map() {
        let mut h = harness();
        assert_eq!(
            h.cpu
                .translate(0x12_3456, mem::AccessKind::Read, false)
                .unwrap(),
            0x12_3456
        );
        assert_eq!(
            h.cpu
                .translate(0x1fff_ffff, mem::AccessKind::Write, false)
                .unwrap(),
            0x1fff_ffff
        );
    }
}

mod segments_and_sizes {
    use super::*;

    #[test]
    fn segment_accessors_read_the_descriptor_flag_bits() {
        let code = Seg {
            sel: 0x10,
            base: 0,
            limit: 0,
            flags: SEG_LONG
                | SEG_PRESENT
                | SEG_NON_SYSTEM
                | SEG_TYPE_EXECUTABLE
                | (2 << SEG_DPL_SHIFT),
        };
        assert!(code.is_code() && code.long() && code.present());
        assert_eq!(code.dpl(), 2);
        let data = Seg {
            flags: SEG_NON_SYSTEM | SEG_PRESENT | SEG_TYPE_READ_WRITE | (3 << SEG_DPL_SHIFT),
            ..code
        };
        assert!(!data.is_code() && !data.long());
        assert_eq!(data.dpl(), 3);
    }

    #[test]
    fn system_descriptor_with_the_executable_bit_pattern_is_not_code() {
        let system = Seg {
            flags: SEG_TYPE_EXECUTABLE | SEG_PRESENT,
            ..Seg::default()
        };
        assert!(!system.is_code());
    }

    #[test]
    fn default_segment_is_null_and_absent() {
        let seg = Seg::default();
        assert!(!seg.present() && !seg.is_code() && !seg.long());
        assert_eq!(seg.dpl(), 0);
    }

    #[test]
    fn tlb_entry_layout_is_what_the_jit_hardcodes() {
        assert_eq!(core::mem::size_of::<TlbEntry>(), 32);
        assert_eq!(core::mem::offset_of!(TlbEntry, tag_r), 0);
        assert_eq!(core::mem::offset_of!(TlbEntry, tag_w), 8);
        assert_eq!(core::mem::offset_of!(TlbEntry, addend), 24);
    }

    #[test]
    fn page_constants_agree() {
        assert_eq!(PAGE_SIZE, 4096);
        assert_eq!(PAGE_OFFSET_MASK, 0xfff);
        assert_eq!(TLB_ENTRIES_PER_MODE, 1024);
        assert_eq!(USER_CPL, 3);
        assert_eq!(GPR_RSP, 4);
        assert_eq!(
            [SEG_ES, SEG_CS, SEG_SS, SEG_DS, SEG_FS, SEG_GS],
            [0, 1, 2, 3, 4, 5]
        );
    }
}

mod time_and_logging {
    use super::*;

    #[test]
    fn now_ns_counts_retired_instructions_plus_idle_time() {
        let mut h = harness();
        assert_eq!(h.cpu.now_ns(), 0);
        h.cpu.icount = 1000;
        assert_eq!(h.cpu.now_ns(), 1000 * NS_PER_INSN);
        h.cpu.idle_skipped_ns = 5_000;
        assert_eq!(h.cpu.now_ns(), 1000 * NS_PER_INSN + 5_000);
    }

    #[test]
    fn tsc_is_the_guest_clock_in_nanoseconds() {
        let mut h = harness();
        h.cpu.icount = 77;
        h.cpu.idle_skipped_ns = 23;
        assert_eq!(h.cpu.tsc(), h.cpu.now_ns());
        assert_eq!(h.cpu.tsc(), 77 * NS_PER_INSN + 23);
    }

    #[test]
    fn tsc_is_monotonic_across_instructions() {
        let mut h = Harness::new(&[0x90, 0x90, 0x90, HLT]);
        let mut previous = h.cpu.tsc();
        for _ in 0..3 {
            h.run(1);
            assert!(h.cpu.tsc() > previous);
            previous = h.cpu.tsc();
        }
    }

    #[test]
    fn log_once_runs_the_message_closure_once_per_key() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        let built = AtomicUsize::new(0);
        for _ in 0..3 {
            h.cpu.log_once(7, || {
                built.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                "seven".to_string()
            });
        }
        assert_eq!(built.load(core::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(*log.lock().unwrap(), ["seven"]);
    }

    #[test]
    fn log_once_keeps_distinct_keys_distinct() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        h.cpu.log_once(1, || "one".into());
        h.cpu.log_once(2, || "two".into());
        h.cpu.log_once(1, || "one again".into());
        assert_eq!(*log.lock().unwrap(), ["one", "two"]);
    }

    #[test]
    fn log_once_without_a_logger_is_silent_and_still_remembers_the_key() {
        let mut h = harness();
        h.cpu.log_once(9, || {
            panic!("the message must not be built without a logger")
        });
        // Installing a logger afterwards does not resurrect the key.
        let log = capture_log(&mut h.cpu);
        h.cpu.log_once(9, || "late".into());
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn log_once_accepts_the_largest_key() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        h.cpu.log_once(u64::MAX, || "max".into());
        h.cpu.log_once(0, || "zero".into());
        assert_eq!(log.lock().unwrap().len(), 2);
    }
}

mod event_servicing {
    use super::*;

    #[test]
    fn stop_flag_is_honoured_by_service_events() {
        let mut h = harness();
        h.cpu.stop.store(true, Ordering::Relaxed);
        assert!(matches!(h.cpu.service_events(), Some(ExitReason::Stopped)));
    }

    #[test]
    fn service_events_schedules_the_next_poll_one_interval_ahead() {
        let mut h = harness();
        h.cpu.icount = 12_345;
        assert!(h.cpu.service_events().is_none());
        assert_eq!(h.cpu.next_event, 12_345 + EVENT_INTERVAL);
    }

    #[test]
    fn a_requested_reset_surfaces_as_a_reset_exit() {
        let mut h = harness();
        h.cpu.dev.io_out(
            KEYBOARD_CONTROLLER_COMMAND_PORT,
            1,
            KEYBOARD_CONTROLLER_RESET,
            0,
        );
        assert!(matches!(h.cpu.service_events(), Some(ExitReason::Reset)));
    }

    #[test]
    fn other_keyboard_controller_commands_do_not_reset() {
        let mut h = harness();
        h.cpu
            .dev
            .io_out(KEYBOARD_CONTROLLER_COMMAND_PORT, 1, 0xaa, 0);
        assert!(h.cpu.service_events().is_none());
    }

    #[test]
    fn tick_devices_calls_the_host_poll_hook_as_not_idle() {
        let mut h = harness();
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = calls.clone();
        h.cpu.poll = Some(std::sync::Arc::new(move |idle| {
            sink.lock().unwrap().push(idle)
        }));
        h.cpu.tick_devices(0);
        h.cpu.tick_devices(10);
        assert_eq!(*calls.lock().unwrap(), [false, false]);
    }

    #[test]
    fn tick_devices_raises_the_pit_interrupt_when_its_deadline_passes() {
        let mut h = harness();
        init_pics(&mut h.cpu);
        program_pit_channel0(&mut h.cpu, 1000);
        let deadline = h.cpu.dev.next_deadline(0).expect("armed timer");
        h.cpu.tick_devices(deadline - 1);
        assert!(!h.cpu.dev.pic.has_pending());
        h.cpu.tick_devices(deadline);
        assert!(h.cpu.dev.pic.has_pending());
    }

    #[test]
    fn pending_interrupt_requires_if_and_no_shadow() {
        let mut h = harness();
        init_pics(&mut h.cpu);
        h.cpu.dev.pic.raise(1);
        h.cpu.eflags = RESERVED_ONE;
        assert_eq!(h.cpu.pending_interrupt(), None);
        h.cpu.eflags = RESERVED_ONE | IF;
        h.cpu.interrupt_shadow = true;
        assert_eq!(h.cpu.pending_interrupt(), None);
        assert!(
            h.cpu.dev.pic.has_pending(),
            "refusing must not acknowledge the interrupt"
        );
        h.cpu.interrupt_shadow = false;
        assert_eq!(h.cpu.pending_interrupt(), Some(IRQ_VECTOR_BASE + 1));
        assert_eq!(h.cpu.pending_interrupt(), None, "now in service");
    }
}

mod before_step {
    use super::*;

    #[test]
    fn executes_when_nothing_is_pending() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
    }

    #[test]
    fn events_are_polled_only_once_the_instruction_count_reaches_next_event() {
        let mut h = with_idt();
        h.cpu.stop.store(true, Ordering::Relaxed);
        h.cpu.next_event = 100;
        h.cpu.icount = 99;
        assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
        h.cpu.icount = 100;
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::Stopped)
        ));
    }

    #[test]
    fn reset_request_exits_at_the_next_poll() {
        let mut h = with_idt();
        h.cpu.dev.io_out(
            KEYBOARD_CONTROLLER_COMMAND_PORT,
            1,
            KEYBOARD_CONTROLLER_RESET,
            0,
        );
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::Reset)
        ));
    }

    #[test]
    fn unmasked_interrupt_with_if_set_is_delivered_through_the_idt() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.dev.pic.raise(1);
        assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
        assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
        assert_eq!(
            h.cpu.eflags_full() & IF,
            0,
            "interrupt gates mask further interrupts"
        );
        let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
        assert_eq!(frame.rip, OLD_RIP);
        assert_eq!(frame.rflags & IF, IF);
        assert!(h.cpu.dev.pic.state().contains("isr 02"));
    }

    #[test]
    fn slave_interrupt_uses_the_slave_vector_base() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.dev.pic.raise(9);
        h.cpu.before_step();
        assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 9));
    }

    #[test]
    fn interrupt_is_held_while_if_is_clear() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.eflags = RESERVED_ONE;
        h.cpu.dev.pic.raise(1);
        assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
        assert_eq!(h.cpu.rip, OLD_RIP);
        assert!(h.cpu.dev.pic.has_pending());
    }

    #[test]
    fn interrupt_is_held_for_one_instruction_after_a_shadowing_instruction() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.interrupt_shadow = true;
        h.cpu.dev.pic.raise(1);
        h.cpu.before_step();
        assert_eq!(h.cpu.rip, OLD_RIP);
        h.cpu.interrupt_shadow = false;
        h.cpu.before_step();
        assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
    }

    #[test]
    fn masked_interrupt_is_not_delivered() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.dev.io_out(PIC_MASTER_DATA, 1, 1 << 1, 0);
        h.cpu.dev.pic.raise(1);
        h.cpu.before_step();
        assert_eq!(h.cpu.rip, OLD_RIP);
    }

    #[test]
    fn delivering_an_interrupt_wakes_a_halted_cpu() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.halted = true;
        h.cpu.dev.pic.raise(1);
        assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
        assert!(!h.cpu.halted);
        assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
    }

    #[test]
    fn halted_cpu_with_interrupts_disabled_is_shut_down_even_with_an_irq_pending() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.eflags = RESERVED_ONE;
        h.cpu.halted = true;
        h.cpu.dev.pic.raise(1);
        // Only NMI, SMI, INIT and RESET wake `cli; hlt`, and none of them exists here.
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::Shutdown)
        ));
        assert!(h.cpu.halted);
        assert_eq!(h.cpu.rip, OLD_RIP);
    }

    #[test]
    fn halted_cpu_with_interrupts_enabled_and_an_irq_pending_idles_without_sleeping() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        // The interrupt shadow keeps the pending IRQ from being delivered this step, but IF is
        // set, so the halt is not terminal.
        h.cpu.interrupt_shadow = true;
        h.cpu.halted = true;
        h.cpu.dev.pic.raise(1);
        assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
        assert!(h.cpu.halted);
        assert_eq!(h.cpu.rip, OLD_RIP);
    }

    #[test]
    fn a_reset_request_beats_the_shutdown_of_a_halt_with_interrupts_disabled() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        h.cpu.eflags = RESERVED_ONE;
        h.cpu.halted = true;
        h.cpu.dev.io_out(
            KEYBOARD_CONTROLLER_COMMAND_PORT,
            1,
            KEYBOARD_CONTROLLER_RESET,
            0,
        );
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::Reset)
        ));
    }

    #[test]
    fn a_stop_request_beats_the_shutdown_of_a_halt_with_interrupts_disabled() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        h.cpu.eflags = RESERVED_ONE;
        h.cpu.halted = true;
        h.cpu.stop.store(true, Ordering::Relaxed);
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::Stopped)
        ));
    }

    #[test]
    fn halted_cpu_exits_on_stop_without_waiting() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        h.cpu.halted = true;
        h.cpu.stop.store(true, Ordering::Relaxed);
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::Stopped)
        ));
    }

    #[test]
    fn halted_cpu_notices_a_reset_request_while_idle() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        h.cpu.halted = true;
        h.cpu.dev.io_out(
            KEYBOARD_CONTROLLER_COMMAND_PORT,
            1,
            KEYBOARD_CONTROLLER_RESET,
            0,
        );
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::Reset)
        ));
    }

    #[test]
    fn halted_cpu_skips_virtual_time_to_the_next_timer_deadline() {
        let mut h = with_idt();
        h.cpu.next_event = u64::MAX;
        init_pics(&mut h.cpu);
        program_pit_channel0(&mut h.cpu, 1000);
        let deadline = h.cpu.dev.next_deadline(0).unwrap();
        h.cpu.halted = true;
        assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
        assert_eq!(h.cpu.idle_skipped_ns, deadline);
        assert_eq!(h.cpu.now_ns(), deadline);
        assert_eq!(h.cpu.icount, 0, "idling retires no instructions");
        assert!(
            h.cpu.dev.pic.has_pending(),
            "the timer fired during the skip"
        );
        // The next step takes the timer interrupt and wakes up.
        assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
        assert!(!h.cpu.halted);
        assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE));
    }

    #[test]
    fn realtime_idle_sleeps_for_a_short_deadline_and_advances_by_it() {
        let mut h = with_idt();
        h.cpu.next_event = u64::MAX;
        h.cpu.realtime = true;
        program_pit_channel0(&mut h.cpu, 100);
        let deadline = h.cpu.dev.next_deadline(0).unwrap();
        h.cpu.halted = true;
        let started = std::time::Instant::now();
        assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
        assert!(started.elapsed() >= std::time::Duration::from_nanos(deadline));
        assert_eq!(h.cpu.idle_skipped_ns, deadline);
    }

    #[test]
    fn realtime_idle_never_sleeps_longer_than_two_milliseconds_per_call() {
        const MAX_SLEEP_NS: u64 = 2_000_000;
        let mut h = with_idt();
        h.cpu.next_event = u64::MAX;
        h.cpu.realtime = true;
        // 65536 ticks of the 1.19 MHz clock is about 55 ms.
        program_pit_channel0(&mut h.cpu, 0);
        h.cpu.halted = true;
        assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
        assert_eq!(h.cpu.idle_skipped_ns, MAX_SLEEP_NS);
    }

    #[test]
    fn delivery_failure_while_taking_an_interrupt_ends_the_run() {
        let mut h = with_idt();
        h.cpu.next_event = 1_000;
        init_pics(&mut h.cpu);
        h.cpu.idtr.limit = 0;
        h.cpu.dev.pic.raise(1);
        assert!(matches!(
            h.cpu.before_step(),
            BeforeStep::Exit(ExitReason::TripleFault)
        ));
    }
}

mod after_step {
    use super::*;

    #[test]
    fn success_continues_the_run() {
        let mut h = with_idt();
        assert!(h.cpu.after_step(Ok(())).is_none());
        assert_eq!(h.cpu.rip, OLD_RIP);
    }

    #[test]
    fn a_fault_is_delivered_through_the_idt() {
        let mut h = with_idt();
        let result = Err(Exception::fault(VECTOR_GENERAL_PROTECTION, Some(0x18)));
        assert!(h.cpu.after_step(result).is_none());
        assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
        let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], true);
        assert_eq!((frame.error_code, frame.rip), (Some(0x18), OLD_RIP));
    }

    #[test]
    fn a_fault_without_an_error_code_pushes_none() {
        let mut h = with_idt();
        assert!(h.cpu.after_step(Err(Exception::invalid_opcode())).is_none());
        assert_eq!(h.cpu.rip, handler_for(VECTOR_INVALID_OPCODE));
        assert_eq!(h.cpu.gpr[GPR_RSP], (STACK_TOP & !0xf) - 40);
    }

    #[test]
    fn software_interrupt_flag_reaches_the_privilege_check() {
        let mut h = with_idt();
        install_tss(&mut h, [KERNEL_STACK, 0, 0], [0; 7]);
        enter_user_mode(&mut h);
        let software = Exception::Fault {
            vector: 0x80,
            error_code: None,
            software: true,
        };
        assert!(h.cpu.after_step(Err(software)).is_none());
        // DPL0 gate from ring 3 raised #GP instead of entering vector 0x80.
        assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
    }

    #[test]
    fn exit_reasons_pass_straight_through() {
        let mut h = with_idt();
        let result = Err(Exception::Exit(ExitReason::Shutdown));
        assert!(matches!(
            h.cpu.after_step(result),
            Some(ExitReason::Shutdown)
        ));
        assert_eq!(h.cpu.rip, OLD_RIP);
    }

    #[test]
    fn a_fault_with_no_working_idt_is_a_triple_fault() {
        let mut h = harness();
        assert!(matches!(
            h.cpu.after_step(Err(Exception::invalid_opcode())),
            Some(ExitReason::TripleFault)
        ));
    }
}

mod run_loop {
    use super::*;

    #[test]
    fn stop_flag_set_before_the_run_exits_immediately() {
        let mut h = Harness::new(&SPIN);
        h.cpu.stop.store(true, Ordering::Relaxed);
        assert!(matches!(h.cpu.run(), ExitReason::Stopped));
        assert_eq!(h.cpu.icount, 0);
    }

    #[test]
    fn run_executes_up_to_the_next_event_poll_then_sees_the_stop_flag() {
        let mut h = Harness::new(&SPIN);
        h.cpu.next_event = 100;
        h.cpu.stop.store(true, Ordering::Relaxed);
        assert!(matches!(h.cpu.run(), ExitReason::Stopped));
        assert_eq!(h.cpu.icount, 100);
        assert_eq!(h.cpu.rip, CODE_BASE);
    }

    #[test]
    fn run_counts_one_instruction_per_step() {
        let mut h = Harness::new(&[0x90; 40]);
        h.cpu.next_event = 25;
        h.cpu.stop.store(true, Ordering::Relaxed);
        h.cpu.run();
        assert_eq!(h.cpu.icount, 25);
        assert_eq!(h.cpu.rip, CODE_BASE + 25);
    }

    #[test]
    fn run_returns_reset_as_soon_as_the_guest_writes_the_reset_port() {
        let mut h = Harness::new(&COMPUTE_THEN_RESET);
        assert!(matches!(h.cpu.run(), ExitReason::Reset));
        assert_eq!(h.cpu.gpr[crate::testkit::RAX], 0xfe);
        // mov, add, mov, out: the port write itself ends the run, no poll interval needed.
        assert_eq!(h.cpu.icount, 4);
    }

    #[test]
    fn run_clears_the_interrupt_shadow_before_each_instruction() {
        let mut h = Harness::new(&SPIN);
        h.cpu.interrupt_shadow = true;
        h.cpu.next_event = 3;
        h.cpu.stop.store(true, Ordering::Relaxed);
        h.cpu.run();
        assert!(!h.cpu.interrupt_shadow);
    }

    #[test]
    fn a_faulting_instruction_is_delivered_and_the_handler_runs() {
        // ud2
        let mut h = Harness::new(&[0x0f, 0x0b]);
        install_idt(&mut h);
        install_reset_handler(&mut h, VECTOR_INVALID_OPCODE);
        assert!(matches!(h.cpu.run(), ExitReason::Reset));
        let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
        assert_eq!(
            frame.rip, CODE_BASE,
            "the saved rip is the faulting instruction"
        );
        // The run ends at the port write, two instructions into the handler.
        assert_eq!(h.cpu.rip, HANDLER_ADDRESS + 4);
    }

    #[test]
    fn a_fault_without_an_idt_ends_in_a_triple_fault() {
        // ud2
        let mut h = Harness::new(&[0x0f, 0x0b]);
        assert!(matches!(h.cpu.run(), ExitReason::TripleFault));
    }

    #[test]
    fn a_timer_interrupt_wakes_a_halted_guest_and_runs_its_handler() {
        let mut h = Harness::new(&[HLT]);
        install_idt(&mut h);
        init_pics(&mut h.cpu);
        program_pit_channel0(&mut h.cpu, 1000);
        install_reset_handler(&mut h, IRQ_VECTOR_BASE);
        h.cpu.eflags = RESERVED_ONE | IF;
        assert!(matches!(h.cpu.run(), ExitReason::Reset));
        assert!(h.cpu.idle_skipped_ns > 0, "the guest slept until the timer");
        let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
        assert_eq!(frame.rip, CODE_BASE + 1, "interrupted after the hlt");
    }

    #[test]
    fn the_jit_run_loop_reaches_the_same_end_state() {
        let mut h = Harness::new(&COMPUTE_THEN_RESET);
        if !h.enable_jit() {
            return;
        }
        assert!(matches!(h.cpu.run(), ExitReason::Reset));
        assert_eq!(h.cpu.gpr[crate::testkit::RAX], 0xfe);
        assert_eq!(h.cpu.icount, 4);
    }

    #[test]
    fn the_jit_run_loop_delivers_faults() {
        // ud2
        let mut h = Harness::new(&[0x0f, 0x0b]);
        if !h.enable_jit() {
            return;
        }
        install_idt(&mut h);
        install_reset_handler(&mut h, VECTOR_INVALID_OPCODE);
        assert!(matches!(h.cpu.run(), ExitReason::Reset));
        assert_eq!(read_frame(&h, h.cpu.gpr[GPR_RSP], false).rip, CODE_BASE);
    }
}

mod single_step {
    use super::*;

    #[test]
    fn step_executes_one_instruction_and_advances_rip() {
        let mut h = Harness::new(&[0x90, 0x90, HLT]);
        h.cpu.step().unwrap();
        assert_eq!(h.cpu.rip, CODE_BASE + 1);
        h.cpu.step().unwrap();
        assert_eq!(h.cpu.rip, CODE_BASE + 2);
    }

    #[test]
    fn step_reports_decode_faults_without_moving_rip() {
        let mut h = Harness::new(&[0x06]);
        assert_eq!(fault_of(h.cpu.step()), (VECTOR_INVALID_OPCODE, None));
        assert_eq!(h.cpu.rip, CODE_BASE);
    }

    #[test]
    fn trace_records_the_address_of_each_executed_instruction() {
        let mut h = Harness::new(&[0x90; 8]);
        h.cpu.trace = Some(Default::default());
        for _ in 0..3 {
            h.cpu.step().unwrap();
        }
        let trace: Vec<u64> = h.cpu.trace.as_ref().unwrap().iter().copied().collect();
        assert_eq!(trace, [CODE_BASE, CODE_BASE + 1, CODE_BASE + 2]);
    }

    #[test]
    fn trace_keeps_only_the_newest_sixty_four_entries() {
        let mut h = Harness::new(&[0x90; 100]);
        h.cpu.trace = Some(Default::default());
        for _ in 0..70 {
            h.cpu.step().unwrap();
        }
        let trace = h.cpu.trace.as_ref().unwrap();
        assert_eq!(trace.len(), TRACE_CAPACITY);
        assert_eq!(*trace.front().unwrap(), CODE_BASE + 6);
        assert_eq!(*trace.back().unwrap(), CODE_BASE + 69);
    }

    #[test]
    fn a_decode_fault_is_not_traced() {
        let mut h = Harness::new(&[0x06]);
        h.cpu.trace = Some(Default::default());
        let _ = h.cpu.step();
        assert!(h.cpu.trace.as_ref().unwrap().is_empty());
    }
}

mod dumps {
    use super::*;

    #[test]
    fn dump_regs_prints_rip_cpl_eflags_and_all_sixteen_registers() {
        let mut h = harness();
        h.cpu.rip = 0x1234_5678_9abc_def0;
        h.cpu.eflags = RESERVED_ONE | IF | ZF;
        for (index, reg) in h.cpu.gpr.iter_mut().enumerate() {
            *reg = 0x1111_1111_1111_1111 * (index as u64 % 16);
        }
        let dump = h.cpu.dump_regs();
        let lines: Vec<&str> = dump.lines().collect();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[0], "rip=123456789abcdef0 cpl=0 eflags=242");
        assert_eq!(
            lines[1],
            "r0 =0000000000000000 r1 =1111111111111111 r2 =2222222222222222 r3 =3333333333333333 "
        );
        assert!(lines[4].starts_with("r12=cccccccccccccccc r13=dddddddddddddddd"));
        assert!(lines[4].contains("r15=ffffffffffffffff"));
    }

    #[test]
    fn dump_regs_resolves_deferred_flags_and_shows_cpl() {
        let mut h = harness();
        h.cpu.set_cpl(3);
        h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
        let dump = h.cpu.dump_regs();
        assert!(dump.starts_with(&format!(
            "rip={:016x} cpl=3 eflags={:x}\n",
            CODE_BASE,
            RESERVED_ONE | CF | ZF | PF | AF
        )));
    }

    #[test]
    fn dump_starts_with_the_state_summary_and_ends_with_device_state() {
        let mut h = harness();
        h.cpu.flushes = 3;
        h.cpu.icount = 42;
        let dump = h.cpu.dump();
        assert!(dump.starts_with("flushes=3 icount=42 rip="), "{dump}");
        assert!(
            dump.contains("cr0=80000033 cr3=9000 cr4=20 efer=500\n"),
            "{dump}"
        );
        assert!(
            dump.contains("halted=false shadow=false dev: m(irr 00"),
            "{dump}"
        );
        assert!(!dump.contains("last rips"));
        assert!(!dump.contains("--- ram match"));
    }

    #[test]
    fn dump_shows_the_halted_and_shadow_flags() {
        let mut h = harness();
        h.cpu.halted = true;
        h.cpu.interrupt_shadow = true;
        assert!(h.cpu.dump().contains("halted=true shadow=true"));
    }

    #[test]
    fn dump_extracts_the_kernel_log_after_the_linux_version_banner() {
        let mut h = harness();
        let mut text = b"Linux version 6.1.0-test (builder) #1 SMP".to_vec();
        text.extend_from_slice(b"\0\0short\0");
        text.extend_from_slice(b"second record with enough characters\0\0");
        h.write(SCRATCH, &text);
        let dump = h.cpu.dump();
        assert!(
            dump.contains(&format!("--- ram match at {SCRATCH:#x}\n")),
            "{dump}"
        );
        assert!(dump.contains("Linux version 6.1.0-test (builder) #1 SMP\n"));
        assert!(dump.contains("second record with enough characters\n"));
        assert!(
            !dump.contains("short"),
            "runs under six characters are record headers"
        );
    }

    #[test]
    fn dump_reports_at_most_six_banner_matches() {
        let mut h = harness();
        for copy in 0..9u64 {
            h.write(SCRATCH + copy * 0x100, b"Linux version 1.2.3 banner\0");
        }
        let dump = h.cpu.dump();
        assert_eq!(dump.matches("--- ram match at").count(), 6);
    }

    #[test]
    fn dump_lists_the_newest_trace_entries_first_and_at_most_sixteen() {
        let mut h = Harness::new(&[0x90; 40]);
        h.cpu.trace = Some(Default::default());
        for _ in 0..30 {
            h.cpu.step().unwrap();
        }
        let dump = h.cpu.dump();
        let line = dump
            .lines()
            .find(|line| line.starts_with("last rips:"))
            .unwrap();
        let rips: Vec<u64> = line
            .trim_start_matches("last rips:")
            .split_whitespace()
            .map(|rip| u64::from_str_radix(rip, 16).unwrap())
            .collect();
        assert_eq!(rips.len(), TRACE_DUMP_ENTRIES);
        assert_eq!(rips[0], CODE_BASE + 29);
        assert_eq!(*rips.last().unwrap(), CODE_BASE + 14);
    }
}

mod verification_helpers {
    use super::*;

    fn pair(code: &[u8]) -> (Harness, Harness) {
        (Harness::new(code), Harness::new(code))
    }

    #[test]
    fn diff_of_identical_cpus_is_none() {
        let (mut a, mut b) = pair(&[HLT]);
        assert_eq!(a.cpu.diff(&mut b.cpu), None);
    }

    #[test]
    fn diff_names_each_differing_register() {
        let (mut a, mut b) = pair(&[HLT]);
        a.cpu.gpr[RBX] = 1;
        a.cpu.gpr[15] = 2;
        let report = a.cpu.diff(&mut b.cpu).unwrap();
        assert!(report.contains("r3: 0x1 vs 0x0"), "{report}");
        assert!(report.contains("r15: 0x2 vs 0x0"), "{report}");
    }

    #[test]
    fn diff_reports_rip_cpl_and_segments() {
        let (mut a, mut b) = pair(&[HLT]);
        a.cpu.rip += 1;
        a.cpu.set_cpl(3);
        a.cpu.seg[SEG_FS].base = 0x1000;
        a.cpu.seg[SEG_SS].sel = 0x2b;
        let report = a.cpu.diff(&mut b.cpu).unwrap();
        assert!(
            report.contains(&format!("rip: {:#x} vs {:#x}", CODE_BASE + 1, CODE_BASE)),
            "{report}"
        );
        assert!(report.contains("cpl"));
        assert!(report.contains("seg4"));
        assert!(report.contains("seg2"));
        assert!(!report.contains("seg1"));
    }

    #[test]
    fn diff_reports_control_registers_efer_and_icount() {
        let (mut a, mut b) = pair(&[HLT]);
        a.cpu.cr0 ^= 1 << 16;
        a.cpu.cr2 = 0x1000;
        a.cpu.cr3 ^= 0x1000;
        a.cpu.cr4 ^= 1 << 7;
        a.cpu.efer ^= EFER_NXE;
        a.cpu.icount = 9;
        let report = a.cpu.diff(&mut b.cpu).unwrap();
        for name in ["cr0", "cr2", "cr3", "cr4", "efer", "icount: 0x9 vs 0x0"] {
            assert!(report.contains(name), "{name} missing from {report}");
        }
    }

    #[test]
    fn diff_reports_xmm_changes() {
        let (mut a, mut b) = pair(&[HLT]);
        a.cpu.xmm[7] = 1;
        assert_eq!(a.cpu.diff(&mut b.cpu).as_deref(), Some("xmm"));
    }

    #[test]
    fn diff_compares_resolved_flags_not_the_deferred_representation() {
        let (mut a, mut b) = pair(&[HLT]);
        // Same architectural flags, one deferred and one materialized.
        a.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
        b.cpu.set_eflags_arith(CF | ZF | PF | AF);
        assert_eq!(a.cpu.diff(&mut b.cpu), None);
        b.cpu.set_eflags_arith(CF);
        let report = a.cpu.diff(&mut b.cpu).unwrap();
        assert!(report.contains("eflags"), "{report}");
    }

    #[test]
    fn run_verify_agrees_on_a_trivial_program() {
        let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
        if !main.enable_jit() {
            return;
        }
        let reason = main.cpu.run_verify(&mut shadow.cpu);
        assert!(matches!(reason, ExitReason::Reset), "{reason:?}");
        assert_eq!(main.cpu.gpr[crate::testkit::RAX] & 0xff, 0xfe);
        assert_eq!(shadow.cpu.gpr[crate::testkit::RAX] & 0xff, 0xfe);
    }

    #[test]
    fn run_verify_reports_a_register_divergence() {
        let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
        if !main.enable_jit() {
            return;
        }
        shadow.cpu.gpr[RBX] = 0xbad;
        let reason = main.cpu.run_verify(&mut shadow.cpu);
        match reason {
            ExitReason::Internal(message) => {
                assert!(
                    message.starts_with("verify: divergence after block at"),
                    "{message}"
                );
                assert!(message.contains("r3: 0x0 vs 0xbad"), "{message}");
            }
            other => panic!("expected a divergence report, got {other:?}"),
        }
    }

    #[test]
    fn run_verify_reports_diverging_code_and_logs_the_page_difference() {
        let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
        if !main.enable_jit() {
            return;
        }
        let log = capture_log(&mut main.cpu);
        // The shadow adds 8 instead of 7: a one-byte difference in the same page.
        shadow.write(CODE_BASE + 10, &[0x08]);
        let reason = main.cpu.run_verify(&mut shadow.cpu);
        assert!(matches!(reason, ExitReason::Internal(_)), "{reason:?}");
        let messages = log.lock().unwrap();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].starts_with("page diffs at "), "{}", messages[0]);
        assert!(messages[0].contains("main bytes"));
    }

    #[test]
    fn run_verify_flags_a_pre_step_divergence() {
        let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
        if !main.enable_jit() {
            return;
        }
        // The main CPU idles (IF set, so the halt is not a shutdown) while the shadow is ready
        // to execute.
        main.cpu.halted = true;
        main.cpu.eflags |= IF;
        main.cpu.next_event = u64::MAX;
        shadow.cpu.next_event = u64::MAX;
        let reason = main.cpu.run_verify(&mut shadow.cpu);
        assert!(
            matches!(&reason, ExitReason::Internal(message) if message.contains("pre-step divergence")),
            "{reason:?}"
        );
    }

    #[test]
    fn run_verify_accepts_matching_faults_and_ends_in_a_triple_fault() {
        // ud2 with no IDT: both engines raise #UD, which cannot be delivered.
        let (mut main, mut shadow) = pair(&[0x0f, 0x0b]);
        if !main.enable_jit() {
            return;
        }
        let reason = main.cpu.run_verify(&mut shadow.cpu);
        assert!(matches!(reason, ExitReason::TripleFault), "{reason:?}");
    }
}
