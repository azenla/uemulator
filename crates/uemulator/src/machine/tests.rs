use super::*;
use crate::dev::Devices;
use crate::disk::Disk;
use crate::platform::{DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE};
use crate::testkit::{CODE_BASE, FakeKernel, Harness, fake_bzimage, run_machine};
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

/// Generous: only a hung guest should ever hit it.
const GENEROUS_LIMIT: Duration = Duration::from_secs(20);
/// Short watchdog for guests that never exit on their own (the watchdog stops them).
const WATCHDOG_LIMIT: Duration = Duration::from_millis(100);
const MIB: usize = 1 << 20;

const SERIAL_PORT_SETUP: [u8; 4] = [0x66, 0xba, 0xf8, 0x03]; // mov dx, 0x3f8
/// mov al, 0xfe; out 0x64, al; hlt: the keyboard controller reset command.
const RESET_AND_HALT: [u8; 5] = [0xb0, 0xfe, 0xe6, 0x64, 0xf4];
/// The serial port must show this before a guest that never exits is stopped.
const ALIVE_MARKER: u8 = b'k';

// mov dx, 0x3f8; mov al, 'h'; out dx, al; mov al, 'i'; out dx, al; mov al, 0xfe; out 0x64, al; hlt
const HELLO_RESET: [u8; 15] = [
    0x66, 0xba, 0xf8, 0x03, 0xb0, 0x68, 0xee, 0xb0, 0x69, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
];
// mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
const PRINT_AND_SPIN: [u8; 13] = [
    0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb, 0xfe, 0x90, 0x90, 0x90, 0x90,
];
/// Offset of the `out 0x64, al` that resets, within `HELLO_RESET`.
const HELLO_RESET_OUT_OFFSET: u64 = 12;
// jmp $
const SPIN: [u8; 2] = [0xeb, 0xfe];
// mov dx, 0x3f8; mov al, 'k'; out dx, al; cli; hlt
const PRINT_THEN_HALT_WITH_INTERRUPTS_OFF: [u8; 9] =
    [0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xfa, 0xf4];
// mov dx, 0x3f8; mov al, 'k'; out dx, al; sti; hlt: with no timer or input nothing ever wakes it,
// but it is waiting, not dead.
const PRINT_THEN_STI_HALT: [u8; 9] = [0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xfb, 0xf4];
// mov eax, 0x8000; lidt [rax]; int3: an empty IDT, so delivering #BP escalates to a triple fault.
const NULL_IDT_THEN_INT3: [u8; 9] = [0xb8, 0x00, 0x80, 0x00, 0x00, 0x0f, 0x01, 0x18, 0xcc];

/// Whether the host has generated-code support, as the machine itself asks.
fn host_has_jit() -> bool {
    crate::jit::probe()
}

fn auto_and_off() -> [JitMode; 2] {
    [JitMode::Off, JitMode::Auto]
}

fn run_guest(code: &[u8], jit: JitMode) -> crate::testkit::MachineRun {
    run_machine(fake_bzimage(code), jit, GENEROUS_LIMIT, |config| config)
}

fn run_spinning(code: &[u8], jit: JitMode) -> crate::testkit::MachineRun {
    run_machine(fake_bzimage(code), jit, WATCHDOG_LIMIT, |config| config)
}

fn new_machine(code: &[u8]) -> EmulatedMachine {
    EmulatedMachine::new(
        MachineConfig::new()
            .reboot(false)
            .kernel(fake_bzimage(code))
            .memory_mib(MIN_MEMORY_MIB),
    )
    .expect("valid configuration")
}

fn expect_config_error(result: Result<EmulatedMachine, Error>) -> String {
    match result {
        Err(Error::BadConfig(message)) => message,
        Err(other) => panic!("expected BadConfig, got {other:?}"),
        Ok(_) => panic!("expected BadConfig, got a machine"),
    }
}

/// Collects the strings the `log` hook receives.
fn log_collector() -> (
    Arc<Mutex<Vec<String>>>,
    impl Fn(&str) + Send + Sync + 'static,
) {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sink = lines.clone();
    (lines, move |line: &str| {
        sink.lock().unwrap().push(line.to_string())
    })
}

mod rtc {
    use super::*;

    /// 2023-11-14 22:13:20 UTC.
    const FIXED_UNIX_TIME: u64 = 1_700_000_000;
    const CMOS_SECONDS: u8 = 0x00;
    const CMOS_MINUTES: u8 = 0x02;
    const CMOS_HOURS: u8 = 0x04;
    const CMOS_DAY_OF_MONTH: u8 = 0x07;
    const CMOS_MONTH: u8 = 0x08;
    const CMOS_YEAR: u8 = 0x09;
    const CMOS_CENTURY: u8 = 0x32;
    const ALL_RTC_REGISTERS: [u8; 6] = [
        CMOS_SECONDS,
        CMOS_MINUTES,
        CMOS_HOURS,
        CMOS_DAY_OF_MONTH,
        CMOS_MONTH,
        CMOS_YEAR,
    ];
    /// BCD: 22:13:20 on 14 November 2023.
    const FIXED_TIME_AS_BCD: [u8; 6] = [0x20, 0x13, 0x22, 0x14, 0x11, 0x23];

    /// Guest that writes the given CMOS registers to the serial port, then resets.
    fn cmos_reader_kernel(registers: &[u8]) -> Vec<u8> {
        let mut code = SERIAL_PORT_SETUP.to_vec();
        for &register in registers {
            // mov al, register; out 0x70, al; in al, 0x71; out dx, al
            code.extend_from_slice(&[0xb0, register, 0xe6, 0x70, 0xe4, 0x71, 0xee]);
        }
        code.extend_from_slice(&RESET_AND_HALT);
        fake_bzimage(&code)
    }

    fn verify_run_reads_fixed_rtc(jit: JitMode) {
        let run = run_machine(
            cmos_reader_kernel(&ALL_RTC_REGISTERS),
            jit,
            GENEROUS_LIMIT,
            |config| config.verify_jit(true).unix_time(FIXED_UNIX_TIME),
        );
        assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
        assert_eq!(run.console_output, FIXED_TIME_AS_BCD, "{jit:?}");
    }

    #[test]
    fn verify_with_fixed_time_and_auto_jit_sees_the_same_rtc_on_both_cpus() {
        verify_run_reads_fixed_rtc(JitMode::Auto);
    }

    #[test]
    fn verify_with_jit_off_runs_the_interpreter_alone() {
        verify_run_reads_fixed_rtc(JitMode::Off);
    }

    #[test]
    fn verify_without_a_fixed_time_also_agrees_across_the_two_cpus() {
        let run = run_machine(
            cmos_reader_kernel(&ALL_RTC_REGISTERS),
            JitMode::Auto,
            GENEROUS_LIMIT,
            |config| config.verify_jit(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        assert_eq!(run.console_output.len(), ALL_RTC_REGISTERS.len());
    }

    #[test]
    fn fixed_time_reaches_the_rtc_without_verify() {
        for jit in auto_and_off() {
            let run = run_machine(
                cmos_reader_kernel(&ALL_RTC_REGISTERS),
                jit,
                GENEROUS_LIMIT,
                |config| config.unix_time(FIXED_UNIX_TIME),
            );
            assert_eq!(run.console_output, FIXED_TIME_AS_BCD, "{jit:?}");
        }
    }

    #[test]
    fn unix_time_zero_is_the_epoch_not_the_wall_clock() {
        let run = run_machine(
            cmos_reader_kernel(&[CMOS_CENTURY, CMOS_YEAR]),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.unix_time(0),
        );
        assert_eq!(run.console_output, [0x19, 0x70]);
    }

    #[test]
    fn without_a_fixed_time_the_rtc_follows_the_host_clock() {
        let run = run_machine(
            cmos_reader_kernel(&[CMOS_CENTURY]),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config,
        );
        // Any real clock is past 2000 and before 2100.
        assert!(matches!(run.console_output[..], [0x20] | [0x21]));
    }

    fn twin_for(boot_unix: Option<u64>) -> Cpu {
        let mut ram = vec![0u8; 64 * MIB];
        let boot = boot::load(&mut ram, &fake_bzimage(&[0xf4]), None, "").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        match boot_unix {
            Some(boot_unix) => interpreter_twin(
                ram,
                boot,
                stop,
                boot_unix,
                DevConfig::default(),
                ConsoleHandle::default(),
            ),
            // What `Vcpu::run` built before the fix.
            None => Cpu::new(
                ram,
                boot,
                ConsoleHandle::default(),
                stop,
                None,
                DevConfig::default(),
            ),
        }
    }

    fn read_cmos(cpu: &mut Cpu, register: u8) -> u8 {
        cpu.dev.io_out(0x70, 1, register as u32, 0);
        cpu.dev.io_in(0x71, 1, 0) as u8
    }

    #[test]
    fn interpreter_twin_takes_the_given_boot_time() {
        let mut twin = twin_for(Some(FIXED_UNIX_TIME));
        assert_eq!(read_cmos(&mut twin, CMOS_SECONDS), 0x20);
        assert_eq!(read_cmos(&mut twin, CMOS_YEAR), 0x23);
        let mut epoch = twin_for(Some(0));
        assert_eq!(read_cmos(&mut epoch, CMOS_CENTURY), 0x19);
    }

    #[test]
    fn a_twin_without_the_boot_time_reads_the_wall_clock_and_would_diverge() {
        let mut fixed = twin_for(Some(0));
        let mut unfixed = twin_for(None);
        assert_eq!(read_cmos(&mut fixed, CMOS_CENTURY), 0x19);
        assert_ne!(
            read_cmos(&mut unfixed, CMOS_CENTURY),
            read_cmos(&mut fixed, CMOS_CENTURY),
            "a twin built like the main CPU but never given its boot time reads another RTC"
        );
    }

    #[test]
    fn interpreter_twin_is_interpreter_only_virtual_time_sharing_the_stop_flag() {
        let mut ram = vec![0u8; 64 * MIB];
        let boot = boot::load(&mut ram, &fake_bzimage(&[0xf4]), None, "").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let twin = interpreter_twin(
            ram,
            boot,
            stop.clone(),
            1,
            DevConfig::default(),
            ConsoleHandle::default(),
        );
        assert!(twin.jit.is_none());
        assert!(!twin.realtime);
        assert!(twin.poll.is_none());
        assert!(twin.log.is_none());
        stop.store(true, Ordering::SeqCst);
        assert!(twin.stop.load(Ordering::SeqCst));
    }
}

mod twin_devices {
    use super::*;

    const MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
    const DISK_BYTES: usize = 2 * MIB;
    const PCI_CONFIG_ADDRESS: u16 = 0xcf8;
    const PCI_CONFIG_DATA: u16 = 0xcfc;
    const PCI_ENABLE: u32 = 1 << 31;
    const SLOT_SHIFT: u32 = 11;
    const VIRTIO_VENDOR: u32 = 0x1af4;
    const VIRTIO_DEVICE_ID_BASE: u32 = 0x1040;
    const DEVICE_TYPE_NET: u32 = 1;
    const DEVICE_TYPE_BLOCK: u32 = 2;
    const DEVICE_TYPE_CONSOLE: u32 = 3;

    fn config_dword(devices: &mut crate::dev::Devices, slot: u32, register: u32) -> u32 {
        devices.io_out(
            PCI_CONFIG_ADDRESS,
            4,
            PCI_ENABLE | (slot << SLOT_SHIFT) | register,
            0,
        );
        devices.io_in(PCI_CONFIG_DATA, 4, 0)
    }

    fn id_of(devices: &mut crate::dev::Devices, slot: u32) -> u32 {
        config_dword(devices, slot, 0)
    }

    fn mirrored(main: &DevConfig) -> DevConfig {
        mirrored_devices(main, &ConsoleHandle::default()).0
    }

    fn full_config(read_only: bool) -> DevConfig {
        DevConfig {
            disks: vec![Disk::from_backend(ReadOnlyFlagDisk(DISK_BYTES, read_only))],
            nets: vec![(MAC, NetHandle::default())],
            virtio_consoles: vec![ConsoleHandle::default()],
            ..Default::default()
        }
    }

    struct ReadOnlyFlagDisk(usize, bool);

    impl crate::disk::BlockBackend for ReadOnlyFlagDisk {
        fn len(&self) -> u64 {
            self.0 as u64
        }
        fn read_at(&mut self, _: u64, buf: &mut [u8]) -> std::io::Result<()> {
            buf.fill(0x5a);
            Ok(())
        }
        fn write_at(&mut self, _: u64, _: &[u8]) -> std::io::Result<()> {
            Ok(())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
        fn read_only(&self) -> bool {
            self.1
        }
    }

    fn device_id(device_type: u32) -> u32 {
        VIRTIO_VENDOR | ((VIRTIO_DEVICE_ID_BASE + device_type) << 16)
    }

    #[test]
    fn mirror_has_the_same_virtio_functions_in_the_same_slots() {
        let main = full_config(false);
        let mut mirror = Devices::new(ConsoleHandle::default(), mirrored(&main));
        let mut original = Devices::new(ConsoleHandle::default(), main);
        for slot in 0..5 {
            assert_eq!(
                id_of(&mut mirror, slot),
                id_of(&mut original, slot),
                "slot {slot}"
            );
        }
        // Console, disk, net: the order `Devices::new` assigns slots.
        assert_eq!(id_of(&mut mirror, 1), device_id(DEVICE_TYPE_CONSOLE));
        assert_eq!(id_of(&mut mirror, 2), device_id(DEVICE_TYPE_BLOCK));
        assert_eq!(id_of(&mut mirror, 3), device_id(DEVICE_TYPE_NET));
    }

    #[test]
    fn mirror_of_an_empty_config_is_empty() {
        let mut mirror = Devices::new(ConsoleHandle::default(), mirrored(&DevConfig::default()));
        assert_eq!(id_of(&mut mirror, 1), u32::MAX);
    }

    #[test]
    fn mirror_gives_each_optional_device_independently() {
        let net_only = DevConfig {
            nets: vec![(MAC, NetHandle::default())],
            ..DevConfig::default()
        };
        let mut mirror = Devices::new(ConsoleHandle::default(), mirrored(&net_only));
        assert_eq!(id_of(&mut mirror, 1), device_id(DEVICE_TYPE_NET));
        assert_eq!(id_of(&mut mirror, 2), u32::MAX);
        let console_only = DevConfig {
            virtio_consoles: vec![ConsoleHandle::default()],
            ..DevConfig::default()
        };
        let mut mirror = Devices::new(ConsoleHandle::default(), mirrored(&console_only));
        assert_eq!(id_of(&mut mirror, 1), device_id(DEVICE_TYPE_CONSOLE));
    }

    #[test]
    fn mirrored_net_device_has_its_own_frame_queue() {
        let host_handle = NetHandle::default();
        let main = DevConfig {
            nets: vec![(MAC, host_handle.clone())],
            ..DevConfig::default()
        };
        let mirror = mirrored(&main);
        let (mac, mirror_handle) = mirror.nets.into_iter().next().unwrap();
        assert_eq!(mac, MAC);
        assert!(host_handle.send_frame(b"frame"));
        assert!(!mirror_handle.has_pending());
    }

    #[test]
    fn mirrored_disk_has_the_capacity_and_read_only_flag_of_the_original() {
        for read_only in [false, true] {
            let mut mirror = mirrored(&full_config(read_only));
            let disk = mirror.disks.remove(0);
            assert_eq!(disk.backend.len(), DISK_BYTES as u64);
            assert_eq!(disk.backend.read_only(), read_only);
        }
    }

    #[test]
    fn mirrored_disk_reads_zeros_and_discards_writes() {
        let mut disk = mirrored(&full_config(false)).disks.remove(0);
        let mut buf = [0xffu8; 16];
        disk.backend.read_at(512, &mut buf).unwrap();
        assert_eq!(buf, [0u8; 16]);
        disk.backend.write_at(0, &[1, 2, 3]).unwrap();
        disk.backend.read_at(0, &mut buf).unwrap();
        assert_eq!(buf, [0u8; 16]);
        disk.backend.flush().unwrap();
    }

    #[test]
    fn mirroring_a_huge_disk_does_not_allocate_it() {
        let huge = DevConfig {
            disks: vec![Disk::from_backend(ReadOnlyFlagDisk(usize::MAX / 2, false))],
            ..DevConfig::default()
        };
        let disk = mirrored(&huge).disks.remove(0);
        assert_eq!(disk.backend.len(), (usize::MAX / 2) as u64);
    }

    // Guest code: read the PCI config dword of slots 1..=3 and print each as 4 bytes.
    fn pci_probe_block(slot: u32) -> Vec<u8> {
        let address = PCI_ENABLE | (slot << SLOT_SHIFT);
        // mov dx, 0xcf8; mov eax, address; out dx, eax; mov dx, 0xcfc; in eax, dx;
        // mov dx, 0x3f8; out dx, al; shr eax, 8; out dx, al; shr eax, 8; out dx, al;
        // shr eax, 8; out dx, al
        let mut block = vec![0x66, 0xba, 0xf8, 0x0c, 0xb8];
        block.extend_from_slice(&address.to_le_bytes());
        block.extend_from_slice(&[
            0xef, 0x66, 0xba, 0xfc, 0x0c, 0xed, 0x66, 0xba, 0xf8, 0x03, 0xee, 0xc1, 0xe8, 0x08,
            0xee, 0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee,
        ]);
        block
    }

    fn pci_probe_kernel() -> Vec<u8> {
        let mut code = Vec::new();
        for slot in 1..=3 {
            code.extend(pci_probe_block(slot));
        }
        code.extend_from_slice(&RESET_AND_HALT);
        fake_bzimage(&code)
    }

    fn id_bytes(device_type: u32) -> [u8; 4] {
        device_id(device_type).to_le_bytes()
    }

    fn probe(configure: impl FnOnce(MachineConfig) -> MachineConfig, jit: JitMode) -> Vec<u8> {
        let run = run_machine(pci_probe_kernel(), jit, GENEROUS_LIMIT, configure);
        assert_eq!(run.exit, ExitReason::Reset);
        run.console_output
    }

    const NOTHING: [u8; 4] = [0xff; 4];

    fn expected_slots(slots: [[u8; 4]; 3]) -> Vec<u8> {
        slots.concat()
    }

    #[test]
    fn a_machine_with_a_serial_console_and_no_other_devices_has_an_empty_pci_bus() {
        for jit in auto_and_off() {
            assert_eq!(
                probe(|config| config.console(ConsoleKind::Serial), jit),
                expected_slots([NOTHING; 3]),
                "{jit:?}"
            );
        }
    }

    #[test]
    fn the_default_console_is_a_virtio_console_function() {
        for jit in auto_and_off() {
            assert_eq!(
                probe(|config| config, jit),
                expected_slots([id_bytes(DEVICE_TYPE_CONSOLE), NOTHING, NOTHING]),
                "{jit:?}"
            );
        }
    }

    #[test]
    fn virtio_console_kind_adds_a_console_function() {
        assert_eq!(
            probe(|config| config.console(ConsoleKind::Virtio), JitMode::Off),
            expected_slots([id_bytes(DEVICE_TYPE_CONSOLE), NOTHING, NOTHING])
        );
    }

    #[test]
    fn serial_console_kind_adds_no_function() {
        assert_eq!(
            probe(|config| config.console(ConsoleKind::Serial), JitMode::Off),
            expected_slots([NOTHING; 3])
        );
    }

    #[test]
    fn a_disk_adds_a_block_function() {
        let disk = Disk::memory(vec![0; MIB]);
        assert_eq!(
            probe(
                |config| config.console(ConsoleKind::Serial).disk(disk),
                JitMode::Off
            ),
            expected_slots([id_bytes(DEVICE_TYPE_BLOCK), NOTHING, NOTHING])
        );
    }

    #[test]
    fn a_net_mac_adds_a_net_function() {
        assert_eq!(
            probe(
                |config| config.console(ConsoleKind::Serial).net(MAC),
                JitMode::Off
            ),
            expected_slots([id_bytes(DEVICE_TYPE_NET), NOTHING, NOTHING])
        );
    }

    #[test]
    fn all_devices_take_slots_in_console_disk_net_order() {
        let disk = Disk::memory(vec![0; MIB]);
        let configure =
            |config: MachineConfig| config.net(MAC).disk(disk).console(ConsoleKind::Virtio);
        assert_eq!(
            probe(configure, JitMode::Off),
            expected_slots([
                id_bytes(DEVICE_TYPE_CONSOLE),
                id_bytes(DEVICE_TYPE_BLOCK),
                id_bytes(DEVICE_TYPE_NET)
            ])
        );
    }

    #[test]
    fn several_disks_and_nets_take_consecutive_slots_in_config_order() {
        let configure = |config: MachineConfig| {
            config
                .console(ConsoleKind::Serial)
                .net(MAC)
                .disk(Disk::memory(vec![0; MIB]))
                .net([2, 0, 0, 0, 0, 9])
                .disk(Disk::memory(vec![0; MIB]))
        };
        // Slots 1..=3 are the first three functions: disks come before nets whatever the call order.
        assert_eq!(
            probe(configure, JitMode::Off),
            expected_slots([
                id_bytes(DEVICE_TYPE_BLOCK),
                id_bytes(DEVICE_TYPE_BLOCK),
                id_bytes(DEVICE_TYPE_NET)
            ])
        );
    }

    #[test]
    fn a_second_virtio_console_takes_the_next_slot() {
        assert_eq!(
            probe(
                |config| config
                    .console(ConsoleKind::Serial)
                    .console(ConsoleKind::Virtio)
                    .console(ConsoleKind::Virtio),
                JitMode::Off
            ),
            expected_slots([
                id_bytes(DEVICE_TYPE_CONSOLE),
                id_bytes(DEVICE_TYPE_CONSOLE),
                NOTHING
            ])
        );
    }

    #[test]
    fn the_verify_twin_sees_the_same_devices_as_the_main_cpu() {
        for jit in auto_and_off() {
            let disk = Disk::memory(vec![0; MIB]);
            let run = run_machine(pci_probe_kernel(), jit, GENEROUS_LIMIT, |config| {
                config
                    .net(MAC)
                    .disk(disk)
                    .console(ConsoleKind::Virtio)
                    .verify_jit(true)
                    .unix_time(1)
            });
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(
                run.console_output,
                expected_slots([
                    id_bytes(DEVICE_TYPE_CONSOLE),
                    id_bytes(DEVICE_TYPE_BLOCK),
                    id_bytes(DEVICE_TYPE_NET)
                ]),
                "{jit:?}"
            );
        }
    }
}

mod multiple_devices {
    use super::*;

    const MAC: [u8; 6] = [2, 0, 0, 0, 0, 1];
    const SECOND_MAC: [u8; 6] = [2, 0, 0, 0, 0, 2];

    fn machine_with(configure: impl FnOnce(MachineConfig) -> MachineConfig) -> EmulatedMachine {
        EmulatedMachine::new(configure(
            MachineConfig::new()
                .reboot(false)
                .jit(JitMode::Off)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB),
        ))
        .expect("valid configuration")
    }

    /// Runs `HELLO_RESET` (writes "hi" to the UART) and returns what each console received.
    fn run_hello(configure: impl FnOnce(MachineConfig) -> MachineConfig) -> Vec<Vec<u8>> {
        let mut machine = machine_with(configure);
        let outputs: Vec<Arc<Mutex<Vec<u8>>>> = machine
            .consoles()
            .iter()
            .map(|console| {
                let sink = Arc::new(Mutex::new(Vec::new()));
                let writer = sink.clone();
                console.on_output(move |bytes| writer.lock().unwrap().extend_from_slice(bytes));
                sink
            })
            .collect();
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        outputs
            .iter()
            .map(|sink| sink.lock().unwrap().clone())
            .collect()
    }

    #[test]
    fn with_no_console_there_is_one_and_the_uart_shares_it() {
        assert_eq!(run_hello(|config| config), [b"hi".to_vec()]);
    }

    #[test]
    fn a_serial_console_next_to_a_virtio_one_owns_the_uart() {
        assert_eq!(
            run_hello(|config| config
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)),
            [b"hi".to_vec(), Vec::new()]
        );
        assert_eq!(
            run_hello(|config| config
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Serial)),
            [Vec::new(), b"hi".to_vec()]
        );
    }

    #[test]
    fn virtio_consoles_have_independent_handles() {
        let machine = machine_with(|config| {
            config
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Virtio)
        });
        let consoles = machine.consoles();
        assert_eq!(consoles.len(), 2);
        consoles[1].resize(100, 40);
        assert_eq!(consoles[0].size(), (80, 25));
        assert_eq!(consoles[1].size(), (100, 40));
        assert_eq!(machine.console().size(), (80, 25));
    }

    #[test]
    fn console_handles_follow_config_order_and_the_first_is_console() {
        let machine = machine_with(|config| {
            config
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)
        });
        let consoles = machine.consoles();
        consoles[0].resize(10, 10);
        assert_eq!(machine.console().size(), (10, 10));
        assert_eq!(consoles[1].size(), (80, 25));
    }

    #[test]
    fn a_second_serial_console_is_a_config_error() {
        let message = expect_config_error(EmulatedMachine::new(
            MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Serial),
        ));
        assert!(message.contains("only once"), "{message}");
    }

    #[test]
    fn nets_are_distinct_handles_wired_to_their_own_device_in_config_order() {
        let machine = machine_with(|config| config.net(MAC).net(SECOND_MAC));
        let handles = machine.nets();
        assert_eq!(handles.len(), 2);
        let devices = &machine.vcpu.as_ref().unwrap().devcfg.nets;
        assert_eq!([devices[0].0, devices[1].0], [MAC, SECOND_MAC]);
        assert!(handles[1].send_frame(b"to the second"));
        assert!(!devices[0].1.has_pending());
        assert_eq!(
            devices[1].1.guest_recv().as_deref(),
            Some(&b"to the second"[..])
        );
        devices[0].1.guest_send(b"from the first".to_vec());
        assert_eq!(
            handles[0].try_recv_frame().as_deref(),
            Some(&b"from the first"[..])
        );
        assert!(handles[1].try_recv_frame().is_none());
        assert!(machine.net().is_some());
    }

    #[test]
    fn disks_keep_their_own_contents_in_config_order() {
        let mut machine = machine_with(|config| {
            config
                .disk(Disk::memory(vec![0xaa; 4096]))
                .disk(Disk::memory(vec![0xbb; 4096]))
        });
        let disks = &mut machine.vcpu.as_mut().unwrap().devcfg.disks;
        let first_bytes: Vec<u8> = disks
            .iter_mut()
            .map(|disk| {
                let mut byte = [0u8; 1];
                disk.backend.read_at(0, &mut byte).unwrap();
                byte[0]
            })
            .collect();
        assert_eq!(first_bytes, [0xaa, 0xbb]);
    }

    #[test]
    fn a_reboot_hands_every_disk_nic_and_console_to_the_new_devices() {
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .jit(JitMode::Off)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .disk(Disk::memory(vec![0; 4096]))
                .disk(Disk::memory(vec![0; 4096]))
                .net(MAC)
                .net(SECOND_MAC)
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Virtio),
        )
        .unwrap();
        let image = machine.vcpu.as_ref().unwrap().reboot.as_ref().unwrap();
        assert_eq!(image.nets.len(), 2);
        assert_eq!(image.virtio_consoles.len(), 2);
        let held = machine.nets();
        assert!(held[1].send_frame(b"frame"));
        assert!(image.nets[1].1.has_pending());
        assert!(!image.nets[0].1.has_pending());
    }

    #[test]
    fn a_machine_that_reboots_keeps_both_disks_in_order() {
        const BOOT_SECTOR_OFFSET: u64 = 512;
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .jit(JitMode::Off)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .disk(Disk::memory(vec![0x11; 4096]))
                .disk(Disk::memory(vec![0x22; 4096])),
        )
        .unwrap();
        let mut vcpu = machine.vcpu.take().unwrap();
        // One power cycle, then the disks are recovered from the finished run.
        let (reason, finished) = vcpu.run_once();
        assert_eq!(reason, ExitReason::Reset);
        let mut disks = finished.disks;
        assert_eq!(disks.len(), 2);
        let mut byte = [0u8; 1];
        disks[0]
            .backend
            .read_at(BOOT_SECTOR_OFFSET, &mut byte)
            .unwrap();
        assert_eq!(byte[0], 0x11);
        disks[1]
            .backend
            .read_at(BOOT_SECTOR_OFFSET, &mut byte)
            .unwrap();
        assert_eq!(byte[0], 0x22);
    }

    #[test]
    fn too_many_virtio_devices_for_the_pci_bus_is_a_config_error() {
        let mut config = MachineConfig::new()
            .kernel(fake_bzimage(&[0xf4]))
            .console(ConsoleKind::Virtio);
        for _ in 0..MAX_VIRTIO_FUNCTIONS {
            config = config.net(MAC);
        }
        let message = expect_config_error(EmulatedMachine::new(config));
        assert!(message.contains("31"), "{message}");
        let mut config = MachineConfig::new()
            .kernel(fake_bzimage(&[0xf4]))
            .console(ConsoleKind::Serial);
        for _ in 0..MAX_VIRTIO_FUNCTIONS {
            config = config.net(MAC);
        }
        assert!(EmulatedMachine::new(config).is_ok());
    }
}

mod config {
    use super::*;

    const MAC: [u8; 6] = [2, 0, 0, 0, 0, 1];

    #[test]
    fn defaults_match_the_documented_ones() {
        let config = MachineConfig::new().reboot(false);
        assert!(config.kernel.is_empty());
        assert!(config.initrd.is_none());
        assert!(config.cmdline.is_none());
        assert!(config.cmdline_extra.is_empty());
        assert_eq!(config.memory_mib, 512);
        assert!(config.disks.is_empty());
        assert!(config.net_macs.is_empty());
        assert!(config.consoles.is_empty());
        assert_eq!(config.jit, JitMode::Auto);
        assert_eq!(config.clock, ClockMode::Virtual);
        assert!(config.log.is_none());
        assert!(config.poll.is_none());
        assert!(!config.debug);
        assert!(!config.verify);
        assert!(config.unix_time.is_none());
    }

    #[test]
    fn default_and_new_agree() {
        let (a, b) = (MachineConfig::new().reboot(false), MachineConfig::default());
        assert_eq!(
            (a.memory_mib, a.jit, a.clock),
            (b.memory_mib, b.jit, b.clock)
        );
    }

    #[test]
    fn every_builder_stores_its_value() {
        let config = MachineConfig::new()
            .reboot(false)
            .kernel(&[1u8, 2, 3][..])
            .initrd(vec![4u8, 5])
            .cmdline("quiet")
            .cmdline_append("a=1")
            .cmdline_append(String::from("b=2"))
            .disk(Disk::memory(vec![0; 512]))
            .net(MAC)
            .console(ConsoleKind::Virtio)
            .memory_mib(128)
            .jit(JitMode::On)
            .clock(ClockMode::Realtime)
            .debug(true)
            .verify_jit(true)
            .unix_time(42)
            .log(|_| {})
            .poll(|_, _| {});
        assert_eq!(config.kernel, [1, 2, 3]);
        assert_eq!(config.initrd.as_deref(), Some(&[4u8, 5][..]));
        assert_eq!(config.cmdline.as_deref(), Some("quiet"));
        assert_eq!(config.cmdline_extra, ["a=1", "b=2"]);
        assert_eq!(config.disks[0].backend.len(), 512);
        assert_eq!(config.net_macs, [MAC]);
        assert_eq!(config.consoles, [ConsoleKind::Virtio]);
        assert_eq!(config.memory_mib, 128);
        assert_eq!(config.jit, JitMode::On);
        assert_eq!(config.clock, ClockMode::Realtime);
        assert!(config.debug);
        assert!(config.verify);
        assert_eq!(config.unix_time, Some(42));
        assert!(config.log.is_some());
        assert!(config.poll.is_some());
    }

    #[test]
    fn a_later_builder_call_overrides_an_earlier_one() {
        let config = MachineConfig::new()
            .reboot(false)
            .memory_mib(100)
            .memory_mib(200)
            .cmdline("a")
            .cmdline("b")
            .debug(true)
            .debug(false)
            .jit(JitMode::Off)
            .jit(JitMode::Auto);
        assert_eq!(config.memory_mib, 200);
        assert_eq!(config.cmdline.as_deref(), Some("b"));
        assert!(!config.debug);
        assert_eq!(config.jit, JitMode::Auto);
    }

    #[test]
    fn enum_defaults() {
        assert_eq!(JitMode::default(), JitMode::Auto);
        assert_eq!(ClockMode::default(), ClockMode::Virtual);
        assert_eq!(ConsoleKind::default(), ConsoleKind::Virtio);
    }

    #[test]
    fn memory_limits_are_the_documented_ones() {
        assert_eq!(MIN_MEMORY_MIB, 64);
        // RAM must end below the 0xc000_0000 MMIO window.
        assert_eq!(MAX_MEMORY_MIB << BYTES_PER_MIB_SHIFT, 0xc000_0000);
    }

    fn with_memory(mib: usize) -> Result<EmulatedMachine, Error> {
        EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(mib),
        )
    }

    #[test]
    fn memory_at_both_limits_is_accepted() {
        assert!(with_memory(MIN_MEMORY_MIB).is_ok());
        assert!(with_memory(MAX_MEMORY_MIB).is_ok());
        assert!(with_memory(MIN_MEMORY_MIB + 1).is_ok());
        assert!(with_memory(MAX_MEMORY_MIB - 1).is_ok());
    }

    #[test]
    fn memory_just_outside_the_limits_is_rejected_with_the_range_in_the_message() {
        for mib in [0, 1, MIN_MEMORY_MIB - 1, MAX_MEMORY_MIB + 1, usize::MAX] {
            let message = expect_config_error(with_memory(mib));
            assert_eq!(
                message,
                format!("memory_mib must be between {MIN_MEMORY_MIB} and {MAX_MEMORY_MIB}"),
                "{mib}"
            );
        }
    }

    #[test]
    fn memory_is_checked_before_the_kernel_is_looked_at() {
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(vec![0u8; 16])
                .memory_mib(1),
        );
        assert!(matches!(result, Err(Error::BadConfig(_))));
    }

    #[test]
    fn a_default_config_without_a_kernel_is_a_bad_kernel() {
        match EmulatedMachine::new(MachineConfig::new().reboot(false)) {
            Err(Error::BadKernel(message)) => assert!(message.contains("HdrS"), "{message}"),
            other => panic!("expected BadKernel, got {:?}", other.err()),
        }
    }

    #[test]
    fn a_kernel_the_loader_rejects_surfaces_its_error() {
        let old = FakeKernel {
            version: 0x0200,
            ..FakeKernel::new(&[0xf4])
        }
        .build();
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(old)
                .memory_mib(64),
        );
        assert!(matches!(result, Err(Error::BadKernel(_))));
    }

    #[test]
    fn an_initrd_too_large_for_ram_is_a_config_error() {
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .initrd(vec![0u8; 65 * MIB])
                .memory_mib(MIN_MEMORY_MIB),
        );
        assert!(expect_config_error(result).contains("initrd"));
    }

    #[test]
    fn a_too_long_command_line_is_a_config_error() {
        let kernel = FakeKernel {
            cmdline_max: 32,
            ..FakeKernel::new(&[0xf4])
        }
        .build();
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(kernel)
                .cmdline("x".repeat(40))
                .memory_mib(64),
        );
        assert!(expect_config_error(result).contains("command line"));
    }

    #[test]
    fn appended_text_counts_towards_the_command_line_limit() {
        let kernel = FakeKernel {
            cmdline_max: 12,
            ..FakeKernel::new(&[0xf4])
        }
        .build();
        let fits = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(kernel.clone())
                .cmdline("abcd")
                .cmdline_append("efgh")
                .memory_mib(64),
        );
        assert!(fits.is_ok());
        let too_long = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(kernel)
                .cmdline("abcd")
                .cmdline_append("efghijk")
                .memory_mib(64),
        );
        expect_config_error(too_long);
    }

    #[test]
    fn jit_off_and_auto_never_fail_for_lack_of_a_jit() {
        for mode in auto_and_off() {
            let config = MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .jit(mode);
            assert!(EmulatedMachine::new(config).is_ok(), "{mode:?}");
        }
    }

    #[test]
    fn jit_on_succeeds_exactly_when_the_host_can_run_generated_code() {
        let config = MachineConfig::new()
            .reboot(false)
            .kernel(fake_bzimage(&[0xf4]))
            .memory_mib(64)
            .jit(JitMode::On);
        match EmulatedMachine::new(config) {
            Ok(_) => assert!(host_has_jit()),
            Err(Error::JitUnavailable) => assert!(!host_has_jit()),
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }

    #[cfg(not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64"))))]
    #[test]
    fn without_a_jit_backend_jit_on_is_jit_unavailable_even_before_the_kernel_is_parsed() {
        let result = EmulatedMachine::new(MachineConfig::new().reboot(false).jit(JitMode::On));
        assert!(matches!(result, Err(Error::JitUnavailable)));
    }

    #[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
    #[test]
    fn with_a_jit_backend_jit_on_is_accepted() {
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .jit(JitMode::On),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn errors_have_readable_messages() {
        assert_eq!(
            Error::BadKernel("x".into()).to_string(),
            "bad kernel image: x"
        );
        assert_eq!(
            Error::BadConfig("y".into()).to_string(),
            "bad configuration: y"
        );
        assert_eq!(
            Error::JitUnavailable.to_string(),
            "JIT requested but unavailable on this host"
        );
        assert_eq!(Error::AlreadyStarted.to_string(), "machine already started");
    }

    #[test]
    fn error_is_a_std_error_without_a_source() {
        let boxed: Box<dyn core::error::Error> = Box::new(Error::AlreadyStarted);
        assert!(boxed.source().is_none());
        assert_eq!(boxed.to_string(), "machine already started");
    }
}

mod guest_view {
    use super::*;

    // mov ebx, [rsi+0x228]; mov dx, 0x3f8; 1: mov al, [rbx]; test al, al; jz 2f; out dx, al;
    // inc rbx; jmp 1b; 2: mov al, 0xfe; out 0x64, al; hlt
    const PRINT_CMDLINE: [u8; 27] = [
        0x8b, 0x9e, 0x28, 0x02, 0x00, 0x00, 0x66, 0xba, 0xf8, 0x03, 0x8a, 0x03, 0x84, 0xc0, 0x74,
        0x06, 0xee, 0x48, 0xff, 0xc3, 0xeb, 0xf4, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    // mov al, [rsi+0x1e8]; mov dx, 0x3f8; out dx, al; mov al, 0xfe; out 0x64, al; hlt
    const PRINT_E820_COUNT: [u8; 16] = [
        0x8a, 0x86, 0xe8, 0x01, 0x00, 0x00, 0x66, 0xba, 0xf8, 0x03, 0xee, 0xb0, 0xfe, 0xe6, 0x64,
        0xf4,
    ];
    // mov dx, 0x3f8; mov eax, [rsi+0x300]; out dx, al; shr eax, 8; out dx, al; shr eax, 8;
    // out dx, al; shr eax, 8; out dx, al; mov al, 0xfe; out 0x64, al; hlt
    const PRINT_E820_LAST_SIZE: [u8; 28] = [
        0x66, 0xba, 0xf8, 0x03, 0x8b, 0x86, 0x00, 0x03, 0x00, 0x00, 0xee, 0xc1, 0xe8, 0x08, 0xee,
        0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    // Prints ramdisk_image and ramdisk_size (4 bytes each, little endian), then the byte sum of
    // the initrd, then resets.
    // mov dx, 0x3f8; mov eax, [rsi+0x218]; out dx, al; shr eax, 8; out dx, al; shr eax, 8;
    // out dx, al; shr eax, 8; out dx, al; mov eax, [rsi+0x21c]; mov ecx, eax; out dx, al;
    // shr eax, 8; out dx, al; shr eax, 8; out dx, al; shr eax, 8; out dx, al;
    // mov ebx, [rsi+0x218]; xor eax, eax; test ecx, ecx; jz 2f; 1: add al, [rbx]; inc rbx;
    // dec ecx; jnz 1b; 2: out dx, al; mov al, 0xfe; out 0x64, al; hlt
    const PRINT_INITRD_AND_CHECKSUM: &[u8] = &[
        0x66, 0xba, 0xf8, 0x03, 0x8b, 0x86, 0x18, 0x02, 0x00, 0x00, 0xee, 0xc1, 0xe8, 0x08, 0xee,
        0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0x8b, 0x86, 0x1c, 0x02, 0x00, 0x00, 0x89,
        0xc1, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0x8b,
        0x9e, 0x18, 0x02, 0x00, 0x00, 0x31, 0xc0, 0x85, 0xc9, 0x74, 0x09, 0x02, 0x03, 0x48, 0xff,
        0xc3, 0xff, 0xc9, 0x75, 0xf7, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    fn printed_cmdline(configure: impl FnOnce(MachineConfig) -> MachineConfig) -> String {
        let run = run_machine(
            fake_bzimage(&PRINT_CMDLINE),
            JitMode::Off,
            GENEROUS_LIMIT,
            configure,
        );
        assert_eq!(run.exit, ExitReason::Reset);
        run.text()
    }

    #[test]
    fn default_command_line_is_the_virtio_one_from_platform() {
        let printed = printed_cmdline(|config| config);
        assert_eq!(printed, DEFAULT_CMDLINE);
        assert!(printed.contains("console=hvc0"));
        assert!(printed.contains("earlyprintk=serial,ttyS0"));
    }

    #[test]
    fn default_command_line_for_a_serial_console_names_ttys0() {
        assert_eq!(
            printed_cmdline(|config| config.console(ConsoleKind::Serial)),
            DEFAULT_SERIAL_CMDLINE
        );
        assert!(DEFAULT_SERIAL_CMDLINE.contains("console=ttyS0"));
    }

    #[test]
    fn the_first_console_picks_the_default_command_line() {
        assert_eq!(
            printed_cmdline(|config| config
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)),
            DEFAULT_SERIAL_CMDLINE
        );
    }

    #[test]
    fn explicit_command_line_replaces_the_default() {
        assert_eq!(
            printed_cmdline(|config| config.cmdline("root=/dev/vda ro")),
            "root=/dev/vda ro"
        );
    }

    #[test]
    fn explicit_command_line_ignores_the_console_kind() {
        for kind in [ConsoleKind::Serial, ConsoleKind::Virtio] {
            assert_eq!(
                printed_cmdline(|config| config.cmdline("single").console(kind)),
                "single"
            );
        }
    }

    #[test]
    fn appended_text_follows_the_default_after_a_space() {
        assert_eq!(
            printed_cmdline(|config| config.cmdline_append("loglevel=7")),
            format!("{DEFAULT_CMDLINE} loglevel=7")
        );
    }

    #[test]
    fn appended_text_follows_an_explicit_command_line_in_call_order() {
        assert_eq!(
            printed_cmdline(|config| config
                .cmdline_append("first")
                .cmdline("base")
                .cmdline_append("second")),
            "base first second"
        );
    }

    #[test]
    fn empty_explicit_command_line_is_an_empty_string_not_the_default() {
        assert_eq!(printed_cmdline(|config| config.cmdline("")), "");
        assert_eq!(
            printed_cmdline(|config| config.cmdline("").cmdline_append("x")),
            " x"
        );
    }

    #[test]
    fn e820_describes_three_ranges() {
        let run = run_guest(&PRINT_E820_COUNT, JitMode::Off);
        assert_eq!(run.console_output, [3]);
    }

    #[test]
    fn e820_extended_range_tracks_memory_mib() {
        for mib in [MIN_MEMORY_MIB, 100, 256] {
            let run = run_machine(
                fake_bzimage(&PRINT_E820_LAST_SIZE),
                JitMode::Off,
                GENEROUS_LIMIT,
                |config| config.memory_mib(mib),
            );
            let size = u32::from_le_bytes(run.console_output[..4].try_into().unwrap());
            assert_eq!(size as usize, (mib - 1) * MIB, "{mib} MiB");
        }
    }

    #[test]
    fn initrd_is_visible_to_the_guest_with_its_contents() {
        let initrd: Vec<u8> = (0..1000u32).map(|index| (index * 31 + 7) as u8).collect();
        let checksum = initrd.iter().fold(0u8, |sum, &byte| sum.wrapping_add(byte));
        for jit in auto_and_off() {
            let initrd = initrd.clone();
            let run = run_machine(
                fake_bzimage(PRINT_INITRD_AND_CHECKSUM),
                jit,
                GENEROUS_LIMIT,
                move |config| config.initrd(initrd),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            let expected_image = ((MIN_MEMORY_MIB * MIB) - 1000) & !0xfff;
            assert_eq!(
                run.console_output[..4],
                (expected_image as u32).to_le_bytes(),
                "{jit:?}"
            );
            assert_eq!(run.console_output[4..8], 1000u32.to_le_bytes(), "{jit:?}");
            assert_eq!(run.console_output[8], checksum, "{jit:?}");
        }
    }

    #[test]
    fn without_an_initrd_the_guest_sees_an_empty_ramdisk() {
        let run = run_guest(PRINT_INITRD_AND_CHECKSUM, JitMode::Off);
        assert_eq!(run.console_output, [0u8; 9]);
    }
}

mod lifecycle {
    use super::*;

    #[test]
    fn wait_without_start_is_stopped() {
        let mut machine = new_machine(&HELLO_RESET);
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn starting_twice_is_an_error() {
        let mut machine = new_machine(&SPIN);
        machine.start().unwrap();
        assert!(matches!(machine.start(), Err(Error::AlreadyStarted)));
        machine.stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn starting_after_the_machine_ran_is_an_error() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        assert!(matches!(machine.start(), Err(Error::AlreadyStarted)));
    }

    #[test]
    fn waiting_twice_reports_stopped_the_second_time() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn stop_before_start_ends_the_guest_before_it_runs() {
        for jit in auto_and_off() {
            let mut machine = EmulatedMachine::new(
                MachineConfig::new()
                    .reboot(false)
                    .kernel(fake_bzimage(&HELLO_RESET))
                    .memory_mib(64)
                    .jit(jit),
            )
            .unwrap();
            let output = Arc::new(Mutex::new(Vec::new()));
            let sink = output.clone();
            machine
                .console()
                .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
            machine.stop();
            machine.start().unwrap();
            assert_eq!(machine.wait(), ExitReason::Stopped, "{jit:?}");
            assert!(output.lock().unwrap().is_empty(), "{jit:?}");
        }
    }

    #[test]
    fn a_stop_handle_from_before_start_also_prevents_the_run() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.stop_handle().stop();
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn stopping_twice_is_harmless() {
        let mut machine = new_machine(&SPIN);
        machine.start().unwrap();
        machine.stop();
        machine.stop();
        machine.stop_handle().stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn stop_handles_are_clones_of_one_flag() {
        let mut machine = new_machine(&SPIN);
        let first = machine.stop_handle();
        let second = first.clone();
        machine.start().unwrap();
        second.stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
        first.stop();
    }

    #[test]
    fn a_stop_handle_stops_a_running_guest_from_another_thread() {
        for jit in auto_and_off() {
            let mut machine = EmulatedMachine::new(
                MachineConfig::new()
                    .reboot(false)
                    .kernel(fake_bzimage(&SPIN))
                    .memory_mib(64)
                    .jit(jit),
            )
            .unwrap();
            let handle = machine.stop_handle();
            machine.start().unwrap();
            let stopper = std::thread::spawn(move || handle.stop());
            assert_eq!(machine.wait(), ExitReason::Stopped, "{jit:?}");
            stopper.join().unwrap();
        }
    }

    #[test]
    fn stop_after_the_guest_exited_does_not_change_the_exit_reason() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.start().unwrap();
        // The reset happens within a few instructions; let it finish, then stop.
        let deadline = Instant::now() + GENEROUS_LIMIT;
        while machine
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        machine.stop();
        assert_eq!(machine.wait(), ExitReason::Reset);
    }

    #[test]
    fn a_new_machine_has_no_thread_until_started() {
        let machine = new_machine(&HELLO_RESET);
        assert!(machine.thread.is_none());
        drop(machine);
    }

    #[test]
    fn console_and_net_handles_are_shared_with_the_devices() {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .net([2, 0, 0, 0, 0, 2]),
        )
        .unwrap();
        machine.console().write(b"in");
        let vcpu = machine.vcpu.as_ref().unwrap();
        assert_eq!(vcpu.uart_console.guest_rx(), Some(b'i'));
        assert_eq!(vcpu.uart_console.guest_rx(), Some(b'n'));
        let net = machine.net().unwrap();
        assert!(net.send_frame(b"frame"));
        let device_side = &vcpu.devcfg.nets[0].1;
        assert_eq!(device_side.guest_recv().as_deref(), Some(&b"frame"[..]));
        // And the other direction.
        device_side.guest_send(b"reply".to_vec());
        assert_eq!(net.try_recv_frame().as_deref(), Some(&b"reply"[..]));
        // The machine never started: nothing runs, nothing to stop.
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn net_handle_exists_only_when_configured() {
        assert!(new_machine(&[0xf4]).net().is_none());
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .net([0; 6]),
        )
        .unwrap();
        assert!(machine.net().is_some());
    }

    #[test]
    fn every_console_handle_from_one_machine_is_the_same_console() {
        let machine = new_machine(&[0xf4]);
        let (a, b) = (machine.console(), machine.console());
        a.resize(100, 40);
        assert_eq!(b.size(), (100, 40));
    }

    #[test]
    fn two_machines_do_not_share_a_console() {
        let (first, second) = (new_machine(&[0xf4]), new_machine(&[0xf4]));
        first.console().resize(10, 10);
        assert_eq!(second.console().size(), (80, 25));
    }
}

mod exits {
    use super::*;

    // mov al, 1; out 0x92, al; hlt
    const FAST_A20_RESET: [u8; 5] = [0xb0, 0x01, 0xe6, 0x92, 0xf4];
    // mov dx, 0xcf9; mov al, 4; out dx, al; hlt
    const RESET_CONTROL_RESET: [u8; 8] = [0x66, 0xba, 0xf9, 0x0c, 0xb0, 0x04, 0xee, 0xf4];
    // mov al, 0xd1; out 0x64, al; mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
    const OTHER_KBC_COMMAND: [u8; 13] = [
        0xb0, 0xd1, 0xe6, 0x64, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb, 0xfe,
    ];
    // mov al, 2; out 0x92, al; mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
    const A20_GATE_ONLY: [u8; 13] = [
        0xb0, 0x02, 0xe6, 0x92, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb, 0xfe,
    ];
    // mov dx, 0xcf9; mov al, 0xb; out dx, al; mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
    const RESET_CONTROL_WITHOUT_CPU_RESET: [u8; 16] = [
        0x66, 0xba, 0xf9, 0x0c, 0xb0, 0x0b, 0xee, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb,
        0xfe,
    ];

    #[test]
    fn keyboard_controller_pulse_reset_is_reset() {
        for jit in auto_and_off() {
            let run = run_guest(&HELLO_RESET, jit);
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.text(), "hi", "{jit:?}");
        }
    }

    #[test]
    fn fast_a20_bit_zero_is_reset() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&FAST_A20_RESET, jit).exit,
                ExitReason::Reset,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn reset_control_bit_two_is_reset() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&RESET_CONTROL_RESET, jit).exit,
                ExitReason::Reset,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn other_keyboard_controller_commands_do_not_reset() {
        for jit in auto_and_off() {
            let run = run_spinning(&OTHER_KBC_COMMAND, jit);
            assert_eq!(run.exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn enabling_a20_alone_does_not_reset() {
        let run = run_spinning(&A20_GATE_ONLY, JitMode::Off);
        assert_eq!(run.exit, ExitReason::Stopped);
        assert_eq!(run.console_output, [ALIVE_MARKER]);
    }

    #[test]
    fn reset_control_without_the_cpu_reset_bit_does_not_reset() {
        let run = run_spinning(&RESET_CONTROL_WITHOUT_CPU_RESET, JitMode::Off);
        assert_eq!(run.exit, ExitReason::Stopped);
        assert_eq!(run.console_output, [ALIVE_MARKER]);
    }

    #[test]
    fn a_looping_guest_runs_until_stopped() {
        for jit in auto_and_off() {
            let run = run_spinning(&PRINT_AND_SPIN, jit);
            assert_eq!(run.exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn a_halted_guest_with_interrupts_off_is_powered_off() {
        for jit in auto_and_off() {
            let run = run_guest(&PRINT_THEN_HALT_WITH_INTERRUPTS_OFF, jit);
            assert_eq!(run.exit, ExitReason::Shutdown, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn sti_then_hlt_waits_for_an_interrupt_instead_of_powering_off() {
        // `sti` leaves a one-instruction interrupt shadow, but IF is set when `hlt` runs.
        for jit in auto_and_off() {
            let run = run_spinning(&PRINT_THEN_STI_HALT, jit);
            assert_eq!(run.exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn a_reset_written_before_a_halt_with_interrupts_off_is_a_reset_not_a_power_off() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&HELLO_RESET, jit).exit,
                ExitReason::Reset,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn an_exception_with_no_idt_is_a_triple_fault() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&NULL_IDT_THEN_INT3, jit).exit,
                ExitReason::TripleFault,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn a_triple_fault_logs_a_state_dump_even_without_debug() {
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&NULL_IDT_THEN_INT3),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log),
        );
        assert_eq!(run.exit, ExitReason::TripleFault);
        let lines = lines.lock().unwrap();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("icount="), "{}", lines[0]);
        assert!(lines[0].contains("rip="), "{}", lines[0]);
        // INT3 is a trap: the state is left after it, at the end of the 9-byte program.
        assert!(
            lines[0].contains(&format!(
                "rip={:016x}",
                CODE_BASE + NULL_IDT_THEN_INT3.len() as u64
            )),
            "{}",
            lines[0]
        );
        assert!(!lines[0].contains("last rips:"), "no trace without debug");
    }

    #[test]
    fn a_plain_reset_logs_nothing() {
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&HELLO_RESET),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        assert!(lines.lock().unwrap().is_empty());
    }

    #[test]
    fn debug_logs_a_dump_with_a_trace_on_every_exit() {
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&HELLO_RESET),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log).debug(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        let lines = lines.lock().unwrap();
        assert_eq!(lines.len(), 1);
        let dump = &lines[0];
        for expected in [
            "flushes=",
            "icount=",
            "cr3=9000",
            "efer=500",
            "halted=",
            "dev:",
            "last rips:",
        ] {
            assert!(dump.contains(expected), "missing {expected:?} in {dump}");
        }
        // The reset instruction is the last one executed; the trace lists newest first.
        let last_rip = format!("last rips: {:x}", CODE_BASE + HELLO_RESET_OUT_OFFSET);
        assert!(dump.contains(&last_rip), "{dump}");
    }

    #[test]
    fn debug_without_a_log_hook_is_silent_and_harmless() {
        let run = run_machine(
            fake_bzimage(&HELLO_RESET),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.debug(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
    }

    #[test]
    fn the_dump_finds_a_kernel_banner_in_guest_ram() {
        let mut code = RESET_AND_HALT.to_vec();
        code.extend_from_slice(b"Linux version 9.9.9-fake (unit test) #1\0");
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&code),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log).debug(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        let lines = lines.lock().unwrap();
        assert!(lines[0].contains("--- ram match at"), "{}", lines[0]);
        assert!(
            lines[0].contains("Linux version 9.9.9-fake (unit test) #1"),
            "{}",
            lines[0]
        );
    }

    #[test]
    fn a_panic_in_a_host_hook_is_reported_as_an_internal_exit_not_propagated() {
        let run = run_machine(
            fake_bzimage(&SPIN),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.poll(|_, _| panic!("host hook failed")),
        );
        assert_eq!(
            run.exit,
            ExitReason::Internal("vcpu thread panicked".into())
        );
    }

    #[test]
    fn a_panic_in_the_console_callback_is_also_an_internal_exit() {
        // Interpreted: the x86_64 JIT calls port I/O helpers through `extern "C"`, so a panic
        // out of the callback there aborts the process instead of unwinding to the join.
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .jit(JitMode::Off),
        )
        .unwrap();
        machine.console().on_output(|_| panic!("embedder bug"));
        machine.start().unwrap();
        assert_eq!(
            machine.wait(),
            ExitReason::Internal("vcpu thread panicked".into())
        );
    }

    /// Exhaustive on purpose: adding or removing a variant must be a conscious API change.
    fn variant_name(reason: &ExitReason) -> &'static str {
        match reason {
            ExitReason::Shutdown => "shutdown",
            ExitReason::Reset => "reset",
            ExitReason::TripleFault => "triple fault",
            ExitReason::Stopped => "stopped",
            ExitReason::Internal(_) => "internal",
        }
    }

    #[test]
    fn exit_reasons_the_machine_produces_are_reset_shutdown_stopped_and_triple_fault() {
        let produced = [
            run_guest(&HELLO_RESET, JitMode::Off).exit,
            run_guest(&FAST_A20_RESET, JitMode::Off).exit,
            run_guest(&RESET_CONTROL_RESET, JitMode::Off).exit,
            run_guest(&NULL_IDT_THEN_INT3, JitMode::Off).exit,
            run_spinning(&SPIN, JitMode::Off).exit,
            run_guest(&PRINT_THEN_HALT_WITH_INTERRUPTS_OFF, JitMode::Off).exit,
            new_machine(&HELLO_RESET).wait(),
        ];
        let names: Vec<_> = produced.iter().map(variant_name).collect();
        assert_eq!(
            names,
            [
                "reset",
                "reset",
                "reset",
                "triple fault",
                "stopped",
                "shutdown",
                "stopped"
            ]
        );
    }

    #[test]
    fn exit_reasons_compare_and_clone() {
        let reason = ExitReason::Internal("x".into());
        assert_eq!(reason.clone(), reason);
        assert_ne!(reason, ExitReason::Internal("y".into()));
        assert_ne!(ExitReason::Reset, ExitReason::Stopped);
    }
}

mod jit_modes {
    use super::*;

    #[test]
    fn every_usable_mode_runs_the_same_guest_to_the_same_result() {
        let mut modes = vec![JitMode::Off, JitMode::Auto];
        if host_has_jit() {
            modes.push(JitMode::On);
        }
        for jit in modes {
            let run = run_guest(&HELLO_RESET, jit);
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.text(), "hi", "{jit:?}");
        }
    }

    #[test]
    fn jit_on_is_refused_before_anything_is_loaded_when_the_host_cannot_jit() {
        let config = MachineConfig::new().reboot(false).jit(JitMode::On);
        match EmulatedMachine::new(config) {
            // Valid host: the empty kernel is what fails.
            Err(Error::BadKernel(_)) => assert!(host_has_jit()),
            Err(Error::JitUnavailable) => assert!(!host_has_jit()),
            other => panic!("unexpected {:?}", other.err()),
        }
    }
}

mod clock {
    use super::*;

    // Sets rsp (interrupt delivery pushes on it; the boot state leaves it 0), installs a gate for vector 0x20 (IRQ 0 after the PIC remap below) in an IDT at 0x8000, then
    // programs PIT channel 0 for a 20 ms one-shot (mode 0, count 0x5d38 at 1.193182 MHz), unmasks
    // IRQ 0 and does `sti; hlt`. The handler prints 't' and resets.
    const WAIT_FOR_PIT_IRQ: [u8; 125] = [
        0xbc, 0x00, 0x00, 0x08, 0x00, 0x48, 0x8d, 0x05, 0x65, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc3,
        0x00, 0x82, 0x00, 0x00, 0x66, 0x89, 0x03, 0x66, 0xc7, 0x43, 0x02, 0x10, 0x00, 0x66, 0xc7,
        0x43, 0x04, 0x00, 0x8e, 0x48, 0xc1, 0xe8, 0x10, 0x66, 0x89, 0x43, 0x06, 0x48, 0xc1, 0xe8,
        0x10, 0x89, 0x43, 0x08, 0xc7, 0x43, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc3, 0x00,
        0x8f, 0x00, 0x00, 0x66, 0xc7, 0x03, 0xff, 0x0f, 0x48, 0xc7, 0x43, 0x02, 0x00, 0x80, 0x00,
        0x00, 0x0f, 0x01, 0x1b, 0xb0, 0x11, 0xe6, 0x20, 0xb0, 0x20, 0xe6, 0x21, 0xb0, 0x04, 0xe6,
        0x21, 0xb0, 0x01, 0xe6, 0x21, 0xb0, 0xfe, 0xe6, 0x21, 0xb0, 0x30, 0xe6, 0x43, 0xb0, 0x38,
        0xe6, 0x40, 0xb0, 0x5d, 0xe6, 0x40, 0xfb, 0xf4, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x74, 0xee,
        0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    const HANDLER_PRINTS: u8 = b't';
    /// PIT input clock and the count programmed above.
    const PIT_HZ: u128 = 1_193_182;
    const PIT_COUNT: u128 = 0x5d38;
    const GUEST_DELAY_NS: u64 = (PIT_COUNT * 1_000_000_000).div_ceil(PIT_HZ) as u64;
    /// Upper bound on the instructions the guest retires before it halts (it is 33); virtual
    /// time counts one nanosecond each.
    const SETUP_INSTRUCTIONS: u64 = 40;

    fn cpu_waiting_for_the_pit(realtime: bool) -> Harness {
        let mut harness = Harness::new(&WAIT_FOR_PIT_IRQ);
        harness.cpu.realtime = realtime;
        harness
    }

    #[test]
    fn virtual_time_skips_the_whole_idle_wait_at_once() {
        let mut harness = cpu_waiting_for_the_pit(false);
        let started = Instant::now();
        assert_eq!(harness.cpu.run(), ExitReason::Reset);
        let skipped = harness.cpu.idle_skipped_ns;
        // Guest time advanced by the PIT delay less the few instructions retired meanwhile.
        assert!(skipped <= GUEST_DELAY_NS, "{skipped}");
        assert!(skipped + SETUP_INSTRUCTIONS >= GUEST_DELAY_NS, "{skipped}");
        assert!(harness.cpu.now_ns() >= GUEST_DELAY_NS);
        assert_eq!(harness.console.read(&mut [0u8; 4], None), 1);
        // Nothing slept on the host for 20 ms of guest time (a loose bound: it is not timing).
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn realtime_accounts_the_same_guest_time_but_really_sleeps_for_it() {
        let mut virtual_cpu = cpu_waiting_for_the_pit(false);
        virtual_cpu.cpu.run();
        let mut realtime_cpu = cpu_waiting_for_the_pit(true);
        let started = Instant::now();
        assert_eq!(realtime_cpu.cpu.run(), ExitReason::Reset);
        let slept = started.elapsed();
        assert_eq!(
            realtime_cpu.cpu.idle_skipped_ns,
            virtual_cpu.cpu.idle_skipped_ns
        );
        // A sleep never ends early, so the wall time is at least the guest time skipped.
        assert!(
            slept >= Duration::from_nanos(realtime_cpu.cpu.idle_skipped_ns),
            "{slept:?}"
        );
    }

    #[test]
    fn realtime_machine_really_waits_for_the_timer_a_virtual_one_does_not() {
        let guest_delay = Duration::from_nanos(GUEST_DELAY_NS - SETUP_INSTRUCTIONS);
        let mut elapsed = Vec::new();
        for clock in [ClockMode::Virtual, ClockMode::Realtime] {
            let started = Instant::now();
            let run = run_machine(
                fake_bzimage(&WAIT_FOR_PIT_IRQ),
                JitMode::Off,
                GENEROUS_LIMIT,
                |config| config.clock(clock),
            );
            elapsed.push(started.elapsed());
            assert_eq!(run.exit, ExitReason::Reset, "{clock:?}");
            assert_eq!(run.console_output, [HANDLER_PRINTS], "{clock:?}");
        }
        assert!(
            elapsed[1] >= guest_delay,
            "realtime run took {:?}",
            elapsed[1]
        );
    }

    #[test]
    fn verify_runs_on_virtual_time_even_when_realtime_was_asked_for() {
        // Both CPUs must idle identically: the lockstep twin skips to the timer at once, so a
        // realtime main CPU (2 ms sleeps) would report a bogus pre-step divergence.
        for jit in [JitMode::Off, JitMode::Auto] {
            let run = run_machine(
                fake_bzimage(&WAIT_FOR_PIT_IRQ),
                jit,
                GENEROUS_LIMIT,
                |config| config.verify_jit(true).clock(ClockMode::Realtime),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, [HANDLER_PRINTS], "{jit:?}");
        }
    }

    #[test]
    fn verify_with_the_default_clock_also_idles_in_step() {
        let run = run_machine(
            fake_bzimage(&WAIT_FOR_PIT_IRQ),
            JitMode::Auto,
            GENEROUS_LIMIT,
            |config| config.verify_jit(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        assert_eq!(run.console_output, [HANDLER_PRINTS]);
    }

    #[test]
    fn a_halt_with_interrupts_on_and_no_timer_idles_in_both_modes_until_stopped() {
        for clock in [ClockMode::Virtual, ClockMode::Realtime] {
            let run = run_machine(
                fake_bzimage(&PRINT_THEN_STI_HALT),
                JitMode::Off,
                WATCHDOG_LIMIT,
                |config| config.clock(clock),
            );
            assert_eq!(run.exit, ExitReason::Stopped, "{clock:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{clock:?}");
        }
    }

    #[test]
    fn virtual_clock_is_the_default_and_guest_time_is_the_instruction_count() {
        let mut harness = Harness::new(&[0x90, 0x90, 0x90, 0xf4]);
        harness.run_to_halt();
        assert!(!harness.cpu.realtime);
        assert_eq!(harness.cpu.idle_skipped_ns, 0);
        assert_eq!(harness.cpu.now_ns(), 4 * crate::cpu::NS_PER_INSN);
    }
}

mod poll_hook {
    use super::*;

    // mov dx, 0x3fd; 1: in al, dx; test al, 1; jz 1b; mov dx, 0x3f8; in al, dx; out dx, al;
    // mov al, 0xfe; out 0x64, al; hlt
    const ECHO_ONE_INPUT_BYTE: [u8; 20] = [
        0x66, 0xba, 0xfd, 0x03, 0xec, 0xa8, 0x01, 0x74, 0xfb, 0x66, 0xba, 0xf8, 0x03, 0xec, 0xee,
        0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    #[test]
    fn the_hook_can_feed_the_guest_console_and_sees_every_tick() {
        for jit in auto_and_off() {
            let calls = Arc::new(AtomicUsize::new(0));
            let counter = calls.clone();
            let run = run_machine(
                fake_bzimage(&ECHO_ONE_INPUT_BYTE),
                jit,
                GENEROUS_LIMIT,
                move |config| {
                    config.poll(move |console, _idle| {
                        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                            console.write(b"Z");
                        }
                    })
                },
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, b"Z", "{jit:?}");
            assert!(calls.load(Ordering::SeqCst) >= 1, "{jit:?}");
        }
    }

    #[test]
    fn the_hook_receives_the_machines_own_console() {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&SPIN))
                .memory_mib(64)
                .jit(JitMode::Off)
                .poll(|console, _| console.resize(132, 50)),
        )
        .unwrap();
        let console = machine.console();
        machine.start().unwrap();
        let deadline = Instant::now() + GENEROUS_LIMIT;
        while console.size() != (132, 50) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        machine.stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
        assert_eq!(console.size(), (132, 50));
    }

    #[test]
    fn host_input_written_before_start_reaches_the_guest() {
        let mut machine = new_machine(&ECHO_ONE_INPUT_BYTE);
        let output = Arc::new(Mutex::new(Vec::new()));
        let sink = output.clone();
        machine
            .console()
            .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
        machine.console().write(b"Q");
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        assert_eq!(output.lock().unwrap().as_slice(), b"Q");
    }

    #[test]
    fn guest_output_without_a_sink_is_readable_with_a_timeout() {
        let mut machine = new_machine(&HELLO_RESET);
        let console = machine.console();
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        let mut buf = [0u8; 8];
        let count = console.read(&mut buf, Some(Duration::from_millis(1)));
        assert_eq!(&buf[..count], b"hi");
    }
}

mod reboot {
    use super::*;

    /// A guest that counts its boots in RAM it cannot keep: the counter lives at 0x400000, which
    /// only a cold boot (zeroed RAM) resets. It prints the counter as a digit, so a machine that
    /// reboots without clearing RAM prints 1, 2, 3 instead of 1, 1, 1.
    // mov ebx, 0x400000; inc byte [rbx]; mov al, [rbx]; add al, '0'; mov dx, 0x3f8; out dx, al;
    // mov al, 0xfe; out 0x64, al; hlt
    const COUNT_BOOTS_IN_RAM: [u8; 21] = [
        0xbb, 0x00, 0x00, 0x40, 0x00, 0xfe, 0x03, 0x8a, 0x03, 0x04, 0x30, 0x66, 0xba, 0xf8, 0x03,
        0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    /// Prints CMOS byte 0x40 as a digit, then stores 7 in it and resets through port 0xcf9. Only a
    /// device reset makes every boot print 0.
    // mov dx, 0x3f8; mov al, 0x40; out 0x70, al; in al, 0x71; add al, '0'; out dx, al;
    // mov al, 0x40; out 0x70, al; mov al, 7; out 0x71, al; mov al, 0x0e; mov dx, 0xcf9; out dx, al;
    // hlt
    const COUNT_BOOTS_IN_CMOS: [u8; 29] = [
        0x66, 0xba, 0xf8, 0x03, 0xb0, 0x40, 0xe6, 0x70, 0xe4, 0x71, 0x04, 0x30, 0xee, 0xb0, 0x40,
        0xe6, 0x70, 0xb0, 0x07, 0xe6, 0x71, 0xb0, 0x0e, 0x66, 0xba, 0xf9, 0x0c, 0xee, 0xf4,
    ];
    const BOOTS_TO_OBSERVE: usize = 4;

    /// Runs `kernel` with rebooting on until it has printed `bytes` bytes, then stops it.
    fn run_until_output(kernel: Vec<u8>, jit: JitMode, bytes: usize) -> (Vec<u8>, ExitReason) {
        let config = MachineConfig::new()
            .kernel(kernel)
            .memory_mib(MIN_MEMORY_MIB)
            .jit(jit)
            .reboot(true);
        let mut machine = EmulatedMachine::new(config).unwrap();
        let output = Arc::new(Mutex::new(Vec::new()));
        let sink = output.clone();
        let stop = machine.stop_handle();
        machine.console().on_output(move |piece| {
            let mut output = sink.lock().unwrap();
            output.extend_from_slice(piece);
            if output.len() >= bytes {
                stop.stop();
            }
        });
        // Backstop for a regression that stops printing.
        let backstop = machine.stop_handle();
        std::thread::spawn(move || {
            std::thread::sleep(GENEROUS_LIMIT);
            backstop.stop();
        });
        machine.start().unwrap();
        let exit = machine.wait();
        let output = output.lock().unwrap().clone();
        (output, exit)
    }

    #[test]
    fn a_reset_reboots_into_zeroed_ram_like_a_cold_boot() {
        for jit in auto_and_off() {
            let _turn = crate::jit::serialize_jit_tests();
            let (output, exit) =
                run_until_output(fake_bzimage(&COUNT_BOOTS_IN_RAM), jit, BOOTS_TO_OBSERVE);
            assert_eq!(exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(output, b"1".repeat(BOOTS_TO_OBSERVE), "{jit:?}");
        }
    }

    #[test]
    fn device_state_does_not_survive_a_reboot() {
        for jit in auto_and_off() {
            let _turn = crate::jit::serialize_jit_tests();
            let (output, exit) =
                run_until_output(fake_bzimage(&COUNT_BOOTS_IN_CMOS), jit, BOOTS_TO_OBSERVE);
            assert_eq!(exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(output, b"0".repeat(BOOTS_TO_OBSERVE), "{jit:?}");
        }
    }

    #[test]
    fn the_console_sink_and_stop_flag_outlive_the_reboots() {
        // The sink is installed once and the stop handle taken before start; seeing every boot's
        // output and a clean `Stopped` shows both stayed attached.
        let (output, exit) = run_until_output(fake_bzimage(&HELLO_RESET), JitMode::Off, 2 * 2);
        assert_eq!(exit, ExitReason::Stopped);
        assert_eq!(output, b"hihi");
    }

    #[test]
    fn reboot_off_ends_the_run_with_reset_after_one_boot() {
        for jit in auto_and_off() {
            let run = run_machine(
                fake_bzimage(&COUNT_BOOTS_IN_RAM),
                jit,
                GENEROUS_LIMIT,
                |config| config.reboot(false),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, b"1", "{jit:?}");
        }
    }

    #[test]
    fn a_triple_fault_is_never_restarted() {
        for reboot in [true, false] {
            let run = run_machine(
                fake_bzimage(&NULL_IDT_THEN_INT3),
                JitMode::Off,
                GENEROUS_LIMIT,
                |config| config.reboot(reboot),
            );
            assert_eq!(run.exit, ExitReason::TripleFault, "reboot {reboot}");
        }
    }

    #[test]
    fn verification_never_restarts_the_machine() {
        for jit in auto_and_off() {
            let _turn = crate::jit::serialize_jit_tests();
            let run = run_machine(fake_bzimage(&HELLO_RESET), jit, GENEROUS_LIMIT, |config| {
                config.verify_jit(true).reboot(true)
            });
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, b"hi", "{jit:?}");
        }
    }

    #[test]
    fn the_reboot_image_shares_the_embedders_net_queues() {
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .net([2, 0, 0, 0, 0, 1]),
        )
        .unwrap();
        let held_by_embedder = machine.net().unwrap();
        let image = machine.vcpu.as_ref().unwrap().reboot.as_ref().unwrap();
        let (mac, rebuilt) = &image.nets[0];
        assert_eq!(*mac, [2, 0, 0, 0, 0, 1]);
        assert!(held_by_embedder.send_frame(b"frame"));
        assert!(
            rebuilt.has_pending(),
            "the rebuilt device sees the same queue"
        );
    }

    #[test]
    fn reboot_is_the_default_and_off_or_verify_keeps_no_image() {
        let image_kept = |config: MachineConfig| {
            let machine = EmulatedMachine::new(config.kernel(fake_bzimage(&SPIN))).unwrap();
            machine.vcpu.as_ref().unwrap().reboot.is_some()
        };
        let base = || MachineConfig::new().memory_mib(MIN_MEMORY_MIB);
        assert!(image_kept(base()));
        assert!(!image_kept(base().reboot(false)));
        assert!(!image_kept(base().verify_jit(true)));
    }
}

mod firmware {
    use super::*;

    /// Bytes from the top of a firmware image to its reset vector (F000:FFF0).
    const RESET_VECTOR_FROM_END: usize = 16;
    // mov dx, 0x3f8; mov al, 'h'; out dx, al; mov al, 'i'; out dx, al; mov al, 0xfe; out 0x64, al;
    // hlt, as 16-bit code: it fits the 16 bytes below the end of the image.
    const REAL_MODE_HELLO_RESET: [u8; 14] = [
        0xba, 0xf8, 0x03, 0xb0, 0x68, 0xee, 0xb0, 0x69, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    fn image_with_reset_code(code: &[u8], length: usize) -> Vec<u8> {
        let mut image = vec![0xf4; length];
        let at = length - RESET_VECTOR_FROM_END;
        image[at..at + code.len()].copy_from_slice(code);
        image
    }

    #[test]
    fn a_firmware_image_starts_in_real_mode_at_the_reset_vector() {
        for jit in auto_and_off() {
            let mut machine = EmulatedMachine::new(
                MachineConfig::new()
                    .reboot(false)
                    .jit(jit)
                    .firmware(image_with_reset_code(&REAL_MODE_HELLO_RESET, 0x1_0000))
                    .memory_mib(MIN_MEMORY_MIB),
            )
            .expect("valid configuration");
            let console = machine.console();
            machine.start().unwrap();
            assert_eq!(machine.wait(), ExitReason::Reset);
            let mut output = [0u8; 8];
            let length = console.read(&mut output, Some(Duration::from_millis(100)));
            assert_eq!(&output[..length], b"hi");
        }
    }

    #[test]
    fn an_empty_or_oversized_firmware_image_is_refused() {
        for length in [0, 0x6_0001] {
            let result = EmulatedMachine::new(
                MachineConfig::new()
                    .firmware(vec![0u8; length])
                    .memory_mib(MIN_MEMORY_MIB),
            );
            expect_config_error(result);
        }
    }

    #[test]
    fn a_guest_reset_with_a_firmware_image_ends_the_run() {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .firmware(image_with_reset_code(&REAL_MODE_HELLO_RESET, 0x2_0000))
                .memory_mib(MIN_MEMORY_MIB),
        )
        .expect("valid configuration");
        machine.start().unwrap();
        assert_eq!(
            machine.wait(),
            ExitReason::Reset,
            "reboot is on, but a firmware image is not rebooted"
        );
    }
}

mod boot_disk {
    use super::*;

    const DISK_BYTES: usize = MIB;
    /// mov ax,0x0e41; int 0x10; cli; hlt
    const PRINT_A_AND_POWER_OFF: [u8; 7] = [0xb8, 0x41, 0x0e, 0xcd, 0x10, 0xfa, 0xf4];
    /// sti; mov ax,[0x46c]; l: cmp ax,[0x46c]; je l; mov dx,0x3f8; mov al,'T'; out dx,al; cli; hlt:
    /// waits for the BIOS tick counter to move, which takes the PIT's IRQ0 through the PIC to the
    /// BIOS's INT 08h.
    const WAIT_FOR_A_TICK: [u8; 18] = [
        0xfb, 0xa1, 0x6c, 0x04, 0x3b, 0x06, 0x6c, 0x04, 0x74, 0xfa, 0xba, 0xf8, 0x03, 0xb0, 0x54,
        0xee, 0xfa, 0xf4,
    ];

    fn disk_with_boot_sector(code: &[u8]) -> Disk {
        let mut image = vec![0u8; DISK_BYTES];
        image[..code.len()].copy_from_slice(code);
        image[510] = 0x55;
        image[511] = 0xaa;
        Disk::memory(image)
    }

    fn boot(disk: Disk) -> (ExitReason, Vec<u8>) {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .boot_disk(true)
                .disk(disk)
                .memory_mib(MIN_MEMORY_MIB),
        )
        .expect("valid configuration");
        let console = machine.console();
        machine.start().unwrap();
        let exit = machine.wait();
        let mut output = [0u8; 64];
        let length = console.read(&mut output, Some(Duration::from_millis(100)));
        (exit, output[..length].to_vec())
    }

    #[test]
    fn the_first_sector_runs_with_bios_services_and_power_off_ends_the_run() {
        let (exit, output) = boot(disk_with_boot_sector(&PRINT_A_AND_POWER_OFF));
        assert_eq!(exit, ExitReason::Shutdown);
        assert_eq!(output, b"A");
    }

    #[test]
    fn the_pit_ticks_the_bios_counter_through_the_pic() {
        let (exit, output) = boot(disk_with_boot_sector(&WAIT_FOR_A_TICK));
        assert_eq!(exit, ExitReason::Shutdown);
        assert_eq!(output, b"T");
    }

    #[test]
    fn a_blank_disk_has_nothing_to_boot() {
        let (exit, output) = boot(Disk::memory(vec![0u8; DISK_BYTES]));
        assert_eq!(exit, ExitReason::Shutdown);
        assert_eq!(output, b"No bootable device.\r\n");
    }

    #[test]
    fn boot_disk_needs_a_disk_and_excludes_a_kernel_a_firmware_and_verify() {
        let base = || {
            MachineConfig::new()
                .boot_disk(true)
                .memory_mib(MIN_MEMORY_MIB)
        };
        expect_config_error(EmulatedMachine::new(base()));
        expect_config_error(EmulatedMachine::new(
            base()
                .disk(disk_with_boot_sector(&[0xf4]))
                .kernel(fake_bzimage(&[0xf4])),
        ));
        expect_config_error(EmulatedMachine::new(
            base()
                .disk(disk_with_boot_sector(&[0xf4]))
                .firmware(vec![0xf4; 16]),
        ));
        expect_config_error(EmulatedMachine::new(
            base().disk(disk_with_boot_sector(&[0xf4])).verify_jit(true),
        ));
    }

    #[test]
    fn a_guest_reset_ends_a_bios_boot_instead_of_restarting_it() {
        // mov al,0xfe; out 0x64,al; hlt
        let (exit, _) = boot(disk_with_boot_sector(&[0xb0, 0xfe, 0xe6, 0x64, 0xf4]));
        assert_eq!(exit, ExitReason::Reset);
    }
}

mod uefi_firmware {
    use super::*;
    use crate::boot::uefi::{FIRMWARE_BASE, HANDOFF_ADDR};
    use crate::kv::{KeyValueError, KeyValueStore, MemoryKeyValueStore};

    /// Prints the first handoff byte ('U'), stores "k" = "v" through the key-value device, reads
    /// it back, prints the value and asks for a reset.
    const STORE_AND_PRINT: [u8; 130] = [
        0x44, 0x0f, 0xb7, 0x47, 0x4a, 0x66, 0xba, 0xf8, 0x03, 0x0f, 0xb6, 0x07, 0xee, 0x44, 0x89,
        0xc2, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x08, 0xb8, 0x01, 0x00, 0x00,
        0x00, 0xef, 0x41, 0x8d, 0x50, 0x04, 0xb8, 0x6b, 0x00, 0x00, 0x00, 0xef, 0x44, 0x89, 0xc2,
        0xb8, 0x02, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x08, 0xb8, 0x01, 0x00, 0x00, 0x00,
        0xef, 0x41, 0x8d, 0x50, 0x04, 0xb8, 0x76, 0x00, 0x00, 0x00, 0xef, 0x44, 0x89, 0xc2, 0xb8,
        0x04, 0x00, 0x00, 0x00, 0xef, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x08,
        0xb8, 0x01, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x04, 0xb8, 0x6b, 0x00, 0x00, 0x00,
        0xef, 0x44, 0x89, 0xc2, 0xb8, 0x03, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x04, 0xed,
        0x66, 0xba, 0xf8, 0x03, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    /// A store the test can look into after the machine owns it.
    #[derive(Clone, Default)]
    struct SharedStore(Arc<Mutex<MemoryKeyValueStore>>);

    impl KeyValueStore for SharedStore {
        fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
            self.0.lock().unwrap().get(key)
        }
        fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), KeyValueError> {
            self.0.lock().unwrap().set(key, value)
        }
        fn delete(&mut self, key: &[u8]) -> Result<(), KeyValueError> {
            self.0.lock().unwrap().delete(key)
        }
        fn list(&mut self, prefix: &[u8]) -> Vec<Vec<u8>> {
            self.0.lock().unwrap().list(prefix)
        }
    }

    /// The image header the library expects, then `code` at `ENTRY_OFFSET`.
    const ENTRY_OFFSET: usize = 0x20;

    fn image(code: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; ENTRY_OFFSET];
        bytes[..8].copy_from_slice(b"UEMUFW64");
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&(ENTRY_OFFSET as u32).to_le_bytes());
        bytes.extend_from_slice(code);
        let footprint = bytes.len() as u64 + 0x1000;
        bytes[16..24].copy_from_slice(&footprint.to_le_bytes());
        bytes
    }

    fn run(
        config: impl FnOnce(MachineConfig) -> MachineConfig,
        limit: Duration,
    ) -> crate::testkit::MachineRun {
        run_machine(Vec::new(), JitMode::Off, limit, config)
    }

    #[test]
    fn the_firmware_runs_with_the_handoff_and_reaches_the_store() {
        let store = SharedStore::default();
        let shared = store.clone();
        let result = run(
            |config| {
                config
                    .uefi_firmware(image(&STORE_AND_PRINT))
                    .key_value_store(shared)
            },
            GENEROUS_LIMIT,
        );
        assert_eq!(result.text(), "Uv");
        assert!(
            matches!(result.exit, ExitReason::Reset),
            "{:?}",
            result.exit
        );
        let mut store = store;
        assert_eq!(store.get(b"k"), Some(b"v".to_vec()));
    }

    #[test]
    fn a_reset_runs_the_firmware_again_and_the_store_survives() {
        let store = SharedStore::default();
        let result = run(
            |config| {
                config
                    .reboot(true)
                    .uefi_firmware(image(&STORE_AND_PRINT))
                    .key_value_store(store)
            },
            WATCHDOG_LIMIT,
        );
        assert!(result.text().starts_with("UvUv"), "{:?}", result.text());
    }

    #[test]
    fn the_handoff_and_image_are_in_guest_memory_at_the_documented_places() {
        assert_eq!(FIRMWARE_BASE, 0x10_0000);
        assert_eq!(HANDOFF_ADDR, 0x7000);
    }

    #[test]
    fn uefi_firmware_excludes_the_other_boot_paths() {
        let firmware = image(&[0xf4]);
        let with_kernel = MachineConfig::new()
            .kernel(fake_bzimage(&SPIN))
            .uefi_firmware(firmware.clone());
        assert!(expect_config_error(EmulatedMachine::new(with_kernel)).contains("uefi_firmware"));
        let with_bios = MachineConfig::new()
            .firmware(vec![0xf4])
            .uefi_firmware(firmware.clone());
        assert!(expect_config_error(EmulatedMachine::new(with_bios)).contains("uefi_firmware"));
        let bad = MachineConfig::new().uefi_firmware(vec![0; 64]);
        assert!(expect_config_error(EmulatedMachine::new(bad)).contains("UEFI firmware image"));
    }
}
