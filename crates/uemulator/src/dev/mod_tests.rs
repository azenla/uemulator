use super::*;
use crate::dev::guestmem::test_mem;
use crate::dev::pci::{CONFIG_ADDRESS_PORT, CONFIG_DATA_PORT, MMIO_BASE};
use crate::netdev::NetHandle;

const NOW_ZERO: u64 = 0;
const ONE_SECOND_NS: u64 = 1_000_000_000;
const RAM_BYTES: usize = 0x2000;
const PCI_ENABLE: u32 = 1 << 31;
const SLOT_SHIFT: u32 = 11;
const COMMAND_REGISTER: u32 = 0x04;
const MEMORY_SPACE_ENABLE: u32 = 1 << 1;
const VIRTIO_VENDOR: u32 = 0x1af4;
const COMMON_DEVICE_STATUS_OFFSET: u64 = 0x14;
const STATUS_DRIVER_OK: u32 = 4;
const UART_SCRATCH_PORT: u16 = 0x3ff;
const UART_DATA_PORT: u16 = 0x3f8;
const UART_IER_PORT: u16 = 0x3f9;
const UART_LSR_PORT: u16 = 0x3fd;
const LSR_THR_EMPTY: u32 = 0x20;
const IER_RX_AVAILABLE: u32 = 1;
const PIT_CONTROL_PORT: u16 = 0x43;
const PIT_CHANNEL0_PORT: u16 = 0x40;
const PIT_CHANNEL2_PORT: u16 = 0x42;
const PIT_CH0_MODE2_LOHI: u32 = 0x34;
const PIT_CH2_MODE0_LOHI: u32 = 0xb0;
const PIC_VECTOR_BASE_MASTER: u32 = 0x20;
const PIC_VECTOR_BASE_SLAVE: u32 = 0x28;

fn bus() -> Devices {
    Devices::new(ConsoleHandle::default(), DevConfig::default())
}

fn bus_with_console(console: &ConsoleHandle) -> Devices {
    Devices::new(
        console.clone(),
        DevConfig {
            virtio_console: true,
            ..Default::default()
        },
    )
}

fn bus_with_pci() -> Devices {
    bus_with_console(&ConsoleHandle::default())
}

fn out(dev: &mut Devices, port: u16, size: u8, value: u32) {
    dev.io_out(port, size, value, NOW_ZERO);
}

fn inp(dev: &mut Devices, port: u16, size: u8) -> u32 {
    dev.io_in(port, size, NOW_ZERO)
}

fn cmos_read(dev: &mut Devices, index: u8, now: u64) -> u8 {
    dev.io_out(CMOS_INDEX_PORT, 1, index as u32, now);
    dev.io_in(CMOS_DATA_PORT, 1, now) as u8
}

fn cmos_write(dev: &mut Devices, index: u8, value: u8) {
    out(dev, CMOS_INDEX_PORT, 1, index as u32);
    out(dev, CMOS_DATA_PORT, 1, value as u32);
}

fn pci_address(slot: u32, reg: u32) -> u32 {
    PCI_ENABLE | slot << SLOT_SHIFT | reg
}

/// Programs both PICs like the kernel does so vectors are 0x20.. and 0x28..
fn init_pics(dev: &mut Devices) {
    for (command, data, base, cascade) in [
        (
            PIC_MASTER_COMMAND,
            PIC_MASTER_DATA,
            PIC_VECTOR_BASE_MASTER,
            0x04,
        ),
        (
            PIC_SLAVE_COMMAND,
            PIC_SLAVE_DATA,
            PIC_VECTOR_BASE_SLAVE,
            0x02,
        ),
    ] {
        out(dev, command, 1, 0x11);
        out(dev, data, 1, base);
        out(dev, data, 1, cascade);
        out(dev, data, 1, 0x01);
    }
}

fn enable_console_driver(dev: &mut Devices) {
    out(
        dev,
        PCI_CONFIG_PORT_FIRST,
        4,
        pci_address(1, COMMAND_REGISTER),
    );
    out(dev, 0xcfc, 2, MEMORY_SPACE_ENABLE);
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert!(dev.mmio_write(
        MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
        1,
        STATUS_DRIVER_OK as u64,
        &mut mem
    ));
}

fn tick(dev: &mut Devices, now: u64) {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    dev.tick(now, &mut mem);
}

fn bcd_of(value: u32) -> u8 {
    (((value / 10) << 4) | (value % 10)) as u8
}

#[test]
fn size_mask_covers_the_low_bytes_and_is_total() {
    assert_eq!(size_mask(0), 0);
    assert_eq!(size_mask(1), 0xff);
    assert_eq!(size_mask(2), 0xffff);
    assert_eq!(size_mask(3), 0x00ff_ffff);
    assert_eq!(size_mask(4), u32::MAX);
    assert_eq!(size_mask(8), u32::MAX);
    assert_eq!(size_mask(u8::MAX), u32::MAX);
}

#[test]
fn bcd_encodes_two_decimal_digits() {
    assert_eq!(bcd(0), 0x00);
    assert_eq!(bcd(9), 0x09);
    assert_eq!(bcd(10), 0x10);
    assert_eq!(bcd(59), 0x59);
    assert_eq!(bcd(99), 0x99);
}

#[test]
fn civil_converts_days_since_the_epoch_to_dates() {
    assert_eq!(civil(0), (1970, 1, 1));
    assert_eq!(civil(-1), (1969, 12, 31));
    assert_eq!(civil(59), (1970, 3, 1));
    assert_eq!(civil(11016), (2000, 2, 29));
    assert_eq!(civil(11017), (2000, 3, 1));
    assert_eq!(civil(47541), (2100, 3, 1), "2100 is not a leap year");
    assert_eq!(civil(47540), (2100, 2, 28));
    assert_eq!(civil(-719468), (0, 3, 1), "the algorithm's own epoch");
}

#[test]
fn unknown_ports_read_all_ones_at_every_size_and_ignore_writes() {
    let mut dev = bus();
    for port in [
        0x0, 0x1f, 0x60, 0x80, 0x1234, 0x3f0, 0x400, 0xcf7, 0xd00, 0xffff,
    ] {
        assert_eq!(inp(&mut dev, port, 1), 0xff, "port {port:#x}");
        assert_eq!(inp(&mut dev, port, 2), 0xffff, "port {port:#x}");
        assert_eq!(inp(&mut dev, port, 4), u32::MAX, "port {port:#x}");
        out(&mut dev, port, 4, u32::MAX);
    }
    assert!(!dev.reset_requested);
}

#[test]
fn io_in_of_size_zero_is_total_and_returns_nothing() {
    let mut dev = bus_with_pci();
    assert_eq!(inp(&mut dev, 0x1234, 0), 0);
    assert_eq!(inp(&mut dev, CMOS_DATA_PORT, 0), 0);
    assert_eq!(inp(&mut dev, 0xcfc, 0), 0);
}

#[test]
fn cmos_index_port_is_write_only() {
    let mut dev = bus();
    out(&mut dev, CMOS_INDEX_PORT, 1, 0x0a);
    assert_eq!(inp(&mut dev, CMOS_INDEX_PORT, 1), 0xff);
}

#[test]
fn cmos_status_registers_have_fixed_values() {
    let mut dev = bus();
    assert_eq!(cmos_read(&mut dev, CMOS_STATUS_A as u8, 0), 0x26);
    assert_eq!(cmos_read(&mut dev, CMOS_STATUS_B as u8, 0), 0x02);
    assert_eq!(cmos_read(&mut dev, CMOS_STATUS_C as u8, 0), 0x00);
    assert_eq!(cmos_read(&mut dev, CMOS_STATUS_D as u8, 0), 0x80);
}

#[test]
fn cmos_status_a_never_reports_update_in_progress() {
    let mut dev = bus();
    for now in [
        0,
        1,
        ONE_SECOND_NS - 1,
        ONE_SECOND_NS,
        123 * ONE_SECOND_NS + 7,
    ] {
        assert_eq!(cmos_read(&mut dev, CMOS_STATUS_A as u8, now) & 0x80, 0);
    }
}

#[test]
fn cmos_status_b_stays_bcd_and_24_hour_whatever_the_guest_writes() {
    let mut dev = bus();
    dev.set_boot_time(1_709_210_096);
    cmos_write(&mut dev, CMOS_STATUS_B as u8, 0x06);
    let status_b = cmos_read(&mut dev, CMOS_STATUS_B as u8, 0);
    assert_eq!(status_b & 0x04, 0, "binary mode bit stays clear");
    assert_ne!(status_b & 0x02, 0, "24-hour mode bit stays set");
    assert_eq!(
        cmos_read(&mut dev, CMOS_HOURS as u8, 0),
        0x12,
        "hours are still BCD"
    );
}

#[test]
fn cmos_status_c_and_d_ignore_writes() {
    let mut dev = bus();
    cmos_write(&mut dev, CMOS_STATUS_C as u8, 0xff);
    cmos_write(&mut dev, CMOS_STATUS_D as u8, 0x00);
    assert_eq!(cmos_read(&mut dev, CMOS_STATUS_C as u8, 0), 0);
    assert_eq!(cmos_read(&mut dev, CMOS_STATUS_D as u8, 0), 0x80);
}

#[test]
fn cmos_nmi_mask_bit_of_the_index_is_not_part_of_the_index() {
    let mut dev = bus();
    assert_eq!(cmos_read(&mut dev, 0x80 | CMOS_STATUS_A as u8, 0), 0x26);
    cmos_write(&mut dev, 0x80 | 0x20, 0xab);
    assert_eq!(cmos_read(&mut dev, 0x20, 0), 0xab);
    assert_eq!(cmos_read(&mut dev, 0xa0, 0), 0xab);
}

#[test]
fn cmos_ram_registers_are_independent_and_start_zero() {
    let mut dev = bus();
    let ram_indices: Vec<u8> = (0x0eu8..0x80)
        .filter(|&index| index as usize != CMOS_CENTURY)
        .collect();
    for &index in &ram_indices {
        assert_eq!(cmos_read(&mut dev, index, 0), 0, "index {index:#x}");
    }
    for &index in &ram_indices {
        cmos_write(&mut dev, index, index ^ 0x5a);
    }
    for &index in &ram_indices {
        assert_eq!(
            cmos_read(&mut dev, index, 0),
            index ^ 0x5a,
            "index {index:#x}"
        );
    }
}

#[test]
fn cmos_alarm_registers_are_plain_ram() {
    let mut dev = bus();
    for (index, value) in [(0x01u8, 0x11u8), (0x03, 0x22), (0x05, 0x33)] {
        cmos_write(&mut dev, index, value);
        assert_eq!(cmos_read(&mut dev, index, 0), value);
    }
}

#[test]
fn cmos_time_registers_ignore_writes() {
    let mut dev = bus();
    dev.set_boot_time(0);
    cmos_write(&mut dev, CMOS_SECONDS as u8, 0x45);
    cmos_write(&mut dev, CMOS_YEAR as u8, 0x99);
    cmos_write(&mut dev, CMOS_CENTURY as u8, 0x30);
    assert_eq!(cmos_read(&mut dev, CMOS_SECONDS as u8, 0), 0x00);
    assert_eq!(cmos_read(&mut dev, CMOS_YEAR as u8, 0), 0x70);
    assert_eq!(cmos_read(&mut dev, CMOS_CENTURY as u8, 0), 0x19);
}

#[test]
fn cmos_data_read_is_masked_to_the_access_size() {
    let mut dev = bus();
    dev.set_boot_time(1_709_210_096);
    out(&mut dev, CMOS_INDEX_PORT, 1, CMOS_MINUTES as u32);
    assert_eq!(inp(&mut dev, CMOS_DATA_PORT, 4), 0x34);
    assert_eq!(inp(&mut dev, CMOS_DATA_PORT, 2), 0x34);
}

struct DateCase {
    unix: u64,
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    /// Sunday is 1, as the RTC counts.
    weekday: u8,
}

/// Expected values come from an independent calendar (Python's datetime), not from this code.
const DATE_CASES: [DateCase; 17] = [
    DateCase {
        unix: 0,
        year: 1970,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 5,
    },
    DateCase {
        unix: 946_684_799,
        year: 1999,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        second: 59,
        weekday: 6,
    },
    DateCase {
        unix: 946_684_800,
        year: 2000,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 7,
    },
    DateCase {
        unix: 951_696_000,
        year: 2000,
        month: 2,
        day: 28,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 2,
    },
    DateCase {
        unix: 951_782_400,
        year: 2000,
        month: 2,
        day: 29,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 3,
    },
    DateCase {
        unix: 951_868_800,
        year: 2000,
        month: 3,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 4,
    },
    DateCase {
        unix: 1_675_123_200,
        year: 2023,
        month: 1,
        day: 31,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 3,
    },
    DateCase {
        unix: 1_675_209_600,
        year: 2023,
        month: 2,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 4,
    },
    DateCase {
        unix: 1_682_812_800,
        year: 2023,
        month: 4,
        day: 30,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 1,
    },
    DateCase {
        unix: 1_704_067_199,
        year: 2023,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        second: 59,
        weekday: 1,
    },
    DateCase {
        unix: 1_709_210_096,
        year: 2024,
        month: 2,
        day: 29,
        hour: 12,
        minute: 34,
        second: 56,
        weekday: 5,
    },
    DateCase {
        unix: 2_147_483_647,
        year: 2038,
        month: 1,
        day: 19,
        hour: 3,
        minute: 14,
        second: 7,
        weekday: 3,
    },
    DateCase {
        unix: 2_147_483_648,
        year: 2038,
        month: 1,
        day: 19,
        hour: 3,
        minute: 14,
        second: 8,
        weekday: 3,
    },
    DateCase {
        unix: 4_107_456_000,
        year: 2100,
        month: 2,
        day: 28,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 1,
    },
    DateCase {
        unix: 4_107_542_400,
        year: 2100,
        month: 3,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 2,
    },
    DateCase {
        unix: 13_574_563_200,
        year: 2400,
        month: 2,
        day: 29,
        hour: 0,
        minute: 0,
        second: 0,
        weekday: 3,
    },
    DateCase {
        unix: 253_402_300_799,
        year: 9999,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        second: 59,
        weekday: 6,
    },
];

#[test]
fn set_boot_time_converts_tricky_dates_to_every_rtc_register() {
    for case in &DATE_CASES {
        let mut dev = bus();
        dev.set_boot_time(case.unix);
        let read = |dev: &mut Devices, index: usize| cmos_read(dev, index as u8, 0);
        let label = format!("unix {}", case.unix);
        assert_eq!(
            read(&mut dev, CMOS_SECONDS),
            bcd_of(case.second),
            "{label} seconds"
        );
        assert_eq!(
            read(&mut dev, CMOS_MINUTES),
            bcd_of(case.minute),
            "{label} minutes"
        );
        assert_eq!(
            read(&mut dev, CMOS_HOURS),
            bcd_of(case.hour),
            "{label} hours"
        );
        assert_eq!(
            read(&mut dev, CMOS_DAY_OF_WEEK),
            case.weekday,
            "{label} weekday"
        );
        assert_eq!(
            read(&mut dev, CMOS_DAY_OF_MONTH),
            bcd_of(case.day),
            "{label} day"
        );
        assert_eq!(
            read(&mut dev, CMOS_MONTH),
            bcd_of(case.month),
            "{label} month"
        );
        assert_eq!(
            read(&mut dev, CMOS_YEAR),
            bcd_of(case.year % 100),
            "{label} year"
        );
        assert_eq!(
            read(&mut dev, CMOS_CENTURY),
            bcd_of(case.year / 100),
            "{label} century"
        );
    }
}

#[test]
fn rtc_follows_virtual_time_not_the_host_clock() {
    let mut dev = bus();
    dev.set_boot_time(1_000_000_000);
    let seconds = |dev: &mut Devices, now| cmos_read(dev, CMOS_SECONDS as u8, now);
    let base = seconds(&mut dev, 0);
    assert_eq!(base, 0x40, "1_000_000_000 is 01:46:40 UTC");
    assert_eq!(
        seconds(&mut dev, ONE_SECOND_NS - 1),
        0x40,
        "sub-second time does not tick"
    );
    assert_eq!(seconds(&mut dev, ONE_SECOND_NS), 0x41);
    assert_eq!(seconds(&mut dev, 20 * ONE_SECOND_NS), 0x00);
    assert_eq!(
        cmos_read(&mut dev, CMOS_MINUTES as u8, 20 * ONE_SECOND_NS),
        0x47
    );
}

#[test]
fn rtc_rolls_over_midnight_month_year_and_leap_day() {
    let mut dev = bus();
    // 2023-12-31 23:59:59 plus one second.
    dev.set_boot_time(1_704_067_199);
    let at = |dev: &mut Devices, index: usize| cmos_read(dev, index as u8, ONE_SECOND_NS);
    assert_eq!(at(&mut dev, CMOS_SECONDS), 0x00);
    assert_eq!(at(&mut dev, CMOS_MINUTES), 0x00);
    assert_eq!(at(&mut dev, CMOS_HOURS), 0x00);
    assert_eq!(at(&mut dev, CMOS_DAY_OF_MONTH), 0x01);
    assert_eq!(at(&mut dev, CMOS_MONTH), 0x01);
    assert_eq!(at(&mut dev, CMOS_YEAR), 0x24);
    assert_eq!(at(&mut dev, CMOS_DAY_OF_WEEK), 2, "Monday");
    // 2000-02-28 23:59:59 plus one second is the leap day.
    dev.set_boot_time(951_782_399);
    assert_eq!(at(&mut dev, CMOS_DAY_OF_MONTH), 0x29);
    assert_eq!(at(&mut dev, CMOS_MONTH), 0x02);
}

#[test]
fn rtc_with_extreme_boot_time_and_clock_never_panics() {
    let mut dev = bus();
    dev.set_boot_time(u64::MAX);
    for index in 0..CMOS_SIZE as u8 {
        let _ = cmos_read(&mut dev, index, u64::MAX);
    }
}

#[test]
fn port_b_returns_only_the_two_writable_bits_plus_synthesized_ones() {
    let mut dev = bus();
    out(&mut dev, SYSTEM_CONTROL_PORT_B, 1, 0xff);
    let value = dev.io_in(SYSTEM_CONTROL_PORT_B, 1, 0);
    assert_eq!(value & 0x03, 0x03);
    assert_eq!(
        value & !0x03 & !(PORT_B_REFRESH_TOGGLE as u32) & !(PORT_B_TIMER2_OUT as u32),
        0
    );
    out(&mut dev, SYSTEM_CONTROL_PORT_B, 1, 0x00);
    assert_eq!(dev.io_in(SYSTEM_CONTROL_PORT_B, 1, 0) & 0x0f, 0);
    out(&mut dev, SYSTEM_CONTROL_PORT_B, 1, 0x02);
    assert_eq!(dev.io_in(SYSTEM_CONTROL_PORT_B, 1, 0) & 0x0f, 0x02);
}

#[test]
fn port_b_refresh_bit_toggles_every_period_of_virtual_time() {
    let mut dev = bus();
    let refresh = |dev: &mut Devices, now: u64| {
        dev.io_in(SYSTEM_CONTROL_PORT_B, 1, now) & PORT_B_REFRESH_TOGGLE as u32 != 0
    };
    let period = REFRESH_TOGGLE_PERIOD_NS;
    assert!(!refresh(&mut dev, 0));
    assert!(!refresh(&mut dev, period - 1));
    assert!(refresh(&mut dev, period));
    assert!(refresh(&mut dev, 2 * period - 1));
    assert!(!refresh(&mut dev, 2 * period));
    assert!(refresh(&mut dev, 3 * period));
    assert!(!refresh(&mut dev, 4 * period));
}

#[test]
fn port_b_timer2_output_reflects_the_pit_channel() {
    let mut dev = bus();
    out(
        &mut dev,
        SYSTEM_CONTROL_PORT_B,
        1,
        PORT_B_TIMER2_GATE as u32,
    );
    out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH2_MODE0_LOHI);
    out(&mut dev, PIT_CHANNEL2_PORT, 1, 0x10);
    out(&mut dev, PIT_CHANNEL2_PORT, 1, 0x00);
    let out2 = |dev: &mut Devices, now: u64| {
        dev.io_in(SYSTEM_CONTROL_PORT_B, 1, now) & PORT_B_TIMER2_OUT as u32 != 0
    };
    assert!(!out2(&mut dev, 1000), "16 PIT ticks have not elapsed");
    assert!(
        out2(&mut dev, ONE_SECOND_NS / 1000),
        "1 ms is far past 16 ticks"
    );
}

#[test]
fn fast_a20_gate_bit_is_stored_and_other_bits_dropped() {
    let mut dev = bus();
    assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0);
    out(&mut dev, FAST_A20_PORT, 1, 0x02);
    assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0x02);
    assert!(!dev.reset_requested);
    out(&mut dev, FAST_A20_PORT, 1, 0xfc);
    assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0);
    assert!(!dev.reset_requested);
}

#[test]
fn fast_a20_bit_zero_requests_a_reset() {
    let mut dev = bus();
    out(&mut dev, FAST_A20_PORT, 1, 0x01);
    assert!(dev.reset_requested);
}

#[test]
fn fast_a20_reset_with_gate_set_keeps_the_gate() {
    let mut dev = bus();
    out(&mut dev, FAST_A20_PORT, 1, 0x03);
    assert!(dev.reset_requested);
    assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0x02);
}

#[test]
fn keyboard_controller_pulse_reset_command_requests_a_reset() {
    let mut dev = bus();
    out(&mut dev, KBC_COMMAND_PORT, 1, 0xfe);
    assert!(dev.reset_requested);
}

#[test]
fn other_keyboard_controller_commands_do_not_reset() {
    let mut dev = bus();
    for command in [0x00, 0x20, 0x60, 0xaa, 0xad, 0xae, 0xd1, 0xfc, 0xfd, 0xff] {
        out(&mut dev, KBC_COMMAND_PORT, 1, command);
    }
    assert!(!dev.reset_requested);
}

#[test]
fn keyboard_data_port_and_status_port_are_not_modelled() {
    let mut dev = bus();
    out(&mut dev, 0x60, 1, 0xfe);
    assert!(!dev.reset_requested);
    assert_eq!(inp(&mut dev, 0x60, 1), 0xff);
    assert_eq!(
        inp(&mut dev, KBC_COMMAND_PORT, 1),
        0xff,
        "status reads as no controller"
    );
}

#[test]
fn keyboard_reset_uses_the_low_byte_of_wider_writes() {
    let mut dev = bus();
    out(&mut dev, KBC_COMMAND_PORT, 4, 0xffff_ff00);
    assert!(!dev.reset_requested);
    out(&mut dev, KBC_COMMAND_PORT, 2, 0x1200 | 0xfe);
    assert!(dev.reset_requested);
}

#[test]
fn reset_control_without_the_cpu_reset_bit_does_not_reset() {
    let mut dev = bus();
    for value in [0x00, 0x01, 0x02, 0x03, 0xfb, 0xf3] {
        out(&mut dev, RESET_CONTROL_PORT, 1, value);
    }
    assert!(!dev.reset_requested);
}

#[test]
fn reset_control_cpu_reset_bit_requests_a_reset_for_hard_and_soft_variants() {
    for value in [0x04, 0x06, 0x0e, 0x05, 0xff] {
        let mut dev = bus();
        out(&mut dev, RESET_CONTROL_PORT, 1, value);
        assert!(dev.reset_requested, "value {value:#x}");
    }
}

#[test]
fn linux_reboot_sequence_resets_on_the_second_write() {
    let mut dev = bus_with_pci();
    out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
    assert!(!dev.reset_requested);
    out(&mut dev, RESET_CONTROL_PORT, 1, 0x06);
    assert!(dev.reset_requested);
}

#[test]
fn reset_control_reads_back_only_the_system_reset_type_bit() {
    let mut dev = bus();
    assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0);
    out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
    assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0x02);
    out(&mut dev, RESET_CONTROL_PORT, 1, 0xfa);
    assert_eq!(
        inp(&mut dev, RESET_CONTROL_PORT, 1),
        0x02,
        "bit 3 and above are not kept"
    );
    out(&mut dev, RESET_CONTROL_PORT, 1, 0x00);
    assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0);
}

#[test]
fn reset_control_works_with_and_without_pci_devices() {
    for with_pci in [false, true] {
        let mut dev = if with_pci { bus_with_pci() } else { bus() };
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
        assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0x02, "pci {with_pci}");
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x04);
        assert!(dev.reset_requested, "pci {with_pci}");
    }
}

#[test]
fn reset_control_write_does_not_touch_the_pci_address_register() {
    let mut dev = bus_with_pci();
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
    out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
    assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), pci_address(1, 0));
    assert_eq!(inp(&mut dev, 0xcfc, 2), VIRTIO_VENDOR);
}

#[test]
fn reset_control_write_is_not_a_config_data_write() {
    let mut dev = bus_with_pci();
    out(
        &mut dev,
        PCI_CONFIG_PORT_FIRST,
        4,
        pci_address(1, COMMAND_REGISTER),
    );
    out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
    assert_eq!(
        inp(&mut dev, 0xcfc, 2),
        0,
        "the command register must not see byte 0x02"
    );
    assert_eq!(inp(&mut dev, 0xcfd, 1), 0);
}

#[test]
fn dword_write_to_the_pci_address_port_does_not_reset_even_if_the_cf9_lane_has_bit_two() {
    let mut dev = bus_with_pci();
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, 0x8000_0400);
    assert!(!dev.reset_requested);
    assert_eq!(
        inp(&mut dev, RESET_CONTROL_PORT, 1),
        0,
        "reset control register unchanged"
    );
    assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), 0x8000_0400);
}

#[test]
fn pci_address_register_accepts_byte_word_and_dword_accesses_through_the_bus() {
    let mut dev = bus_with_pci();
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, 0x8000_0000);
    out(&mut dev, PCI_CONFIG_PORT_FIRST + 2, 1, 0x01);
    out(&mut dev, PCI_CONFIG_PORT_FIRST + 3, 1, 0x80);
    assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), 0x8001_0000);
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 2, 0x0800);
    assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), 0x8001_0800);
    assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST + 2, 2), 0x8001);
    assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST + 3, 1), 0x80);
    assert_eq!(inp(&mut dev, 0xcfa, 1), 0x01);
}

#[test]
fn pci_config_space_is_reachable_through_the_ports() {
    let mut dev = bus_with_pci();
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(0, 0));
    assert_eq!(inp(&mut dev, 0xcfc, 4), 0x1237_8086);
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
    assert_eq!(inp(&mut dev, CONFIG_DATA_PORT, 2), VIRTIO_VENDOR);
    assert_eq!(inp(&mut dev, CONFIG_DATA_PORT + 2, 2), 0x1043);
    assert_eq!(inp(&mut dev, CONFIG_DATA_PORT + 3, 1), 0x10);
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(2, 0));
    assert_eq!(inp(&mut dev, 0xcfc, 4), u32::MAX);
}

#[test]
fn pci_ports_are_dead_when_no_virtio_device_is_attached() {
    let mut dev = bus();
    out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(0, 0));
    assert_eq!(
        inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4),
        u32::MAX,
        "address register does not echo"
    );
    assert_eq!(inp(&mut dev, 0xcfc, 4), u32::MAX);
    assert_eq!(inp(&mut dev, 0xcfe, 2), 0xffff);
}

#[test]
fn pci_windows_do_not_extend_past_cff_or_below_cf8() {
    let mut dev = bus_with_pci();
    assert_eq!(
        inp(&mut dev, PCI_CONFIG_PORT_LAST, 1),
        0xff,
        "cff in the data window, no device selected"
    );
    assert_eq!(inp(&mut dev, 0xd00, 1), 0xff);
    assert_eq!(inp(&mut dev, 0xcf7, 1), 0xff);
    assert_eq!(CONFIG_ADDRESS_PORT, PCI_CONFIG_PORT_FIRST);
}

#[test]
fn devices_attach_in_console_disk_net_order_to_consecutive_slots() {
    let mut dev = Devices::new(
        ConsoleHandle::default(),
        DevConfig {
            disk: Some(Disk::memory(vec![0; 4096])),
            net: Some(([2, 0, 0, 0, 0, 1], NetHandle::default())),
            virtio_console: true,
        },
    );
    let mut device_ids = Vec::new();
    for slot in 1..=4 {
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(slot, 0));
        device_ids.push(inp(&mut dev, 0xcfc, 4) >> 16);
    }
    assert_eq!(device_ids, vec![0x1043, 0x1042, 0x1041, 0xffff]);
}

#[test]
fn a_lone_disk_or_net_device_takes_slot_one() {
    let mut disk_only = Devices::new(
        ConsoleHandle::default(),
        DevConfig {
            disk: Some(Disk::memory(vec![0; 4096])),
            ..Default::default()
        },
    );
    out(&mut disk_only, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
    assert_eq!(inp(&mut disk_only, 0xcfc, 4) >> 16, 0x1042);
    let mut net_only = Devices::new(
        ConsoleHandle::default(),
        DevConfig {
            net: Some(([2, 0, 0, 0, 0, 1], NetHandle::default())),
            ..Default::default()
        },
    );
    out(&mut net_only, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
    assert_eq!(inp(&mut net_only, 0xcfc, 4) >> 16, 0x1041);
}

#[test]
fn mmio_with_no_devices_is_unmapped() {
    let mut dev = bus();
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert_eq!(dev.mmio_read(MMIO_BASE, 4), None);
    assert_eq!(dev.mmio_read(0, 1), None);
    assert!(!dev.mmio_write(MMIO_BASE, 4, 0, &mut mem));
    assert!(!dev.mmio_write(u64::MAX, 1, 0, &mut mem));
}

#[test]
fn mmio_is_unmapped_until_the_guest_enables_memory_space() {
    let mut dev = bus_with_pci();
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert_eq!(dev.mmio_read(MMIO_BASE + 0x12, 2), None);
    assert!(!dev.mmio_write(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1, 4, &mut mem));
    out(
        &mut dev,
        PCI_CONFIG_PORT_FIRST,
        4,
        pci_address(1, COMMAND_REGISTER),
    );
    out(&mut dev, 0xcfc, 2, MEMORY_SPACE_ENABLE);
    assert_eq!(dev.mmio_read(MMIO_BASE + 0x12, 2), Some(2));
    assert!(dev.mmio_write(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1, 4, &mut mem));
    assert_eq!(
        dev.mmio_read(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1),
        Some(4)
    );
}

#[test]
fn mmio_outside_every_bar_stays_unmapped_after_enabling() {
    let mut dev = bus_with_pci();
    enable_console_driver(&mut dev);
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert_eq!(dev.mmio_read(MMIO_BASE - 1, 1), None);
    assert_eq!(dev.mmio_read(MMIO_BASE + 0x4000, 1), None);
    assert_eq!(dev.mmio_read(0xffff_f000, 1), None);
    assert!(!dev.mmio_write(MMIO_BASE + 0x4000, 4, 0, &mut mem));
}

#[test]
fn mmio_write_that_raises_nothing_leaves_the_pic_quiet() {
    let mut dev = bus_with_pci();
    init_pics(&mut dev);
    enable_console_driver(&mut dev);
    assert!(!dev.pic.has_pending());
    assert_eq!(dev.pic.ack(), None);
}

/// Console transmit queue programmed through BAR0, one chain posted, ready to kick.
fn console_with_transmit_chain(dev: &mut Devices, ram: &mut [u8]) {
    const TRANSMIT_QUEUE: u32 = 1;
    const DESC: u64 = 0x100;
    const AVAIL: u64 = 0x200;
    const USED: u64 = 0x300;
    const BUFFER: u64 = 0x400;
    ram[BUFFER as usize] = b'!';
    ram[DESC as usize..DESC as usize + 8].copy_from_slice(&BUFFER.to_le_bytes());
    ram[DESC as usize + 8..DESC as usize + 12].copy_from_slice(&1u32.to_le_bytes());
    ram[AVAIL as usize + 2] = 1;
    out(
        dev,
        PCI_CONFIG_PORT_FIRST,
        4,
        pci_address(1, COMMAND_REGISTER),
    );
    out(dev, 0xcfc, 2, MEMORY_SPACE_ENABLE);
    let mut mem = test_mem(ram);
    let mut write = |offset: u64, size: u8, value: u64| {
        assert!(dev.mmio_write(MMIO_BASE + offset, size, value, &mut mem));
    };
    write(0x16, 2, TRANSMIT_QUEUE as u64);
    write(0x18, 2, 8);
    write(0x20, 4, DESC);
    write(0x28, 4, AVAIL);
    write(0x30, 4, USED);
    write(0x1c, 2, 1);
    write(COMMON_DEVICE_STATUS_OFFSET, 1, STATUS_DRIVER_OK as u64);
    write(0x3000 + 4 * TRANSMIT_QUEUE as u64, 2, TRANSMIT_QUEUE as u64);
}

#[test]
fn mmio_write_that_completes_a_buffer_raises_the_function_irq_on_the_pic() {
    let console = ConsoleHandle::default();
    let mut dev = bus_with_console(&console);
    init_pics(&mut dev);
    let mut ram = vec![0u8; RAM_BYTES];
    console_with_transmit_chain(&mut dev, &mut ram);
    let mut buf = [0u8; 4];
    assert_eq!(console.read(&mut buf, None), 1);
    assert_eq!(buf[0], b'!');
    assert!(dev.pic.has_pending());
    assert_eq!(
        dev.pic.ack(),
        Some(PIC_VECTOR_BASE_SLAVE as u8 + 1),
        "IRQ 9 is slave line 1"
    );
}

#[test]
fn tick_polls_virtio_devices_and_raises_their_irq() {
    let console = ConsoleHandle::default();
    let mut dev = bus_with_console(&console);
    init_pics(&mut dev);
    enable_console_driver(&mut dev);
    tick(&mut dev, 0);
    assert!(!dev.pic.has_pending());
    console.resize(120, 40);
    tick(&mut dev, 1);
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_SLAVE as u8 + 1));
}

#[test]
fn tick_raises_irq0_when_the_pit_deadline_passes() {
    let mut dev = bus();
    init_pics(&mut dev);
    out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0xa9);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x04);
    let deadline = dev.next_deadline(0).expect("channel 0 is armed");
    tick(&mut dev, deadline - 1);
    assert!(!dev.pic.has_pending(), "not due yet");
    tick(&mut dev, deadline);
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8));
}

#[test]
fn next_deadline_is_the_pit_deadline_and_ignores_the_now_argument() {
    let mut dev = bus();
    assert_eq!(dev.next_deadline(0), None);
    assert_eq!(dev.next_deadline(u64::MAX), None);
    out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0xa9);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x04);
    let deadline = dev.next_deadline(0).unwrap();
    assert_eq!(dev.next_deadline(deadline / 2), Some(deadline));
    assert_eq!(
        dev.next_deadline(deadline * 10),
        Some(deadline),
        "a past deadline is still reported"
    );
    assert_eq!(dev.next_deadline(u64::MAX), Some(deadline));
}

#[test]
fn uart_and_virtio_devices_do_not_schedule_deadlines() {
    let console = ConsoleHandle::default();
    let mut dev = bus_with_console(&console);
    console.write(b"pending input");
    console.resize(100, 50);
    out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
    assert_eq!(dev.next_deadline(0), None);
}

#[test]
fn state_dump_combines_pic_and_pit_state_with_the_given_time() {
    let mut dev = bus();
    let first = dev.state(111);
    assert!(first.starts_with("m(irr "), "{first}");
    assert!(first.contains(" | pit0 mode "), "{first}");
    assert!(first.ends_with("now 111"), "{first}");
    assert!(dev.state(222).ends_with("now 222"));
    out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x00);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x10);
    assert!(dev.state(0).contains("reload 4096"), "{}", dev.state(0));
}

#[test]
fn pic_ports_are_routed_to_the_interrupt_controllers() {
    let mut dev = bus();
    for (port, mask) in [(PIC_MASTER_DATA, 0xa5), (PIC_SLAVE_DATA, 0x5a)] {
        out(&mut dev, port, 1, mask);
        assert_eq!(inp(&mut dev, port, 1), mask, "port {port:#x}");
    }
    out(&mut dev, PIC_MASTER_DATA, 1, 0x00);
    assert_eq!(inp(&mut dev, PIC_MASTER_DATA, 1), 0);
    assert_eq!(
        inp(&mut dev, PIC_SLAVE_DATA, 1),
        0x5a,
        "the slave mask is separate"
    );
}

#[test]
fn pic_command_ports_and_elcr_ports_are_claimed_by_the_pics() {
    let mut dev = bus();
    for port in [
        PIC_MASTER_COMMAND,
        PIC_SLAVE_COMMAND,
        PIC_ELCR_MASTER,
        PIC_ELCR_SLAVE,
    ] {
        assert_ne!(
            inp(&mut dev, port, 4),
            u32::MAX,
            "port {port:#x} must not read as unmapped"
        );
    }
    out(&mut dev, PIC_ELCR_MASTER, 1, 0);
    assert_eq!(inp(&mut dev, PIC_ELCR_MASTER, 1), 0);
}

#[test]
fn pic_initialization_through_the_ports_sets_the_vector_bases() {
    let mut dev = bus();
    init_pics(&mut dev);
    dev.pic.raise(IRQ_UART);
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
    dev.pic.raise(12);
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_SLAVE as u8 + 4));
}

#[test]
fn pit_ports_are_routed_to_the_timer() {
    let mut dev = bus();
    out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x34);
    out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x12);
    assert!(dev.state(0).contains("reload 4660"), "{}", dev.state(0));
    assert_eq!(inp(&mut dev, 0x44, 1), 0xff, "0x44 is past the PIT");
}

#[test]
fn uart_ports_are_routed_and_transmit_to_the_console() {
    let console = ConsoleHandle::default();
    let mut dev = Devices::new(console.clone(), DevConfig::default());
    out(&mut dev, UART_SCRATCH_PORT, 1, 0x5a);
    assert_eq!(inp(&mut dev, UART_SCRATCH_PORT, 1), 0x5a);
    assert_ne!(inp(&mut dev, UART_LSR_PORT, 1) & LSR_THR_EMPTY, 0);
    out(&mut dev, UART_DATA_PORT, 1, b'A' as u32);
    out(&mut dev, UART_DATA_PORT, 1, b'B' as u32);
    let mut buf = [0u8; 8];
    assert_eq!(console.read(&mut buf, None), 2);
    assert_eq!(&buf[..2], b"AB");
}

#[test]
fn uart_receive_interrupt_reaches_the_pic_once_per_edge() {
    let console = ConsoleHandle::default();
    let mut dev = Devices::new(console.clone(), DevConfig::default());
    init_pics(&mut dev);
    out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
    assert!(!dev.pic.has_pending());
    console.write(b"x");
    tick(&mut dev, 0);
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
    tick(&mut dev, 1);
    assert_eq!(
        dev.pic.ack(),
        None,
        "the line is still high: no second edge"
    );
    assert_eq!(inp(&mut dev, UART_DATA_PORT, 1), b'x' as u32);
}

#[test]
fn uart_stops_taking_host_input_once_the_virtio_console_driver_is_up() {
    let console = ConsoleHandle::default();
    let mut dev = bus_with_console(&console);
    init_pics(&mut dev);
    out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
    console.write(b"y");
    enable_console_driver(&mut dev);
    tick(&mut dev, 0);
    assert!(console.has_input(), "hvc0 owns the input now");
    assert!(!dev.pic.has_pending());
}

#[test]
fn uart_keeps_host_input_while_the_virtio_console_driver_is_not_up() {
    let console = ConsoleHandle::default();
    let mut dev = bus_with_console(&console);
    init_pics(&mut dev);
    out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
    console.write(b"y");
    tick(&mut dev, 0);
    assert!(!console.has_input());
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
}

#[test]
fn reset_request_is_sticky() {
    let mut dev = bus();
    out(&mut dev, KBC_COMMAND_PORT, 1, 0xfe);
    out(&mut dev, FAST_A20_PORT, 1, 0);
    out(&mut dev, RESET_CONTROL_PORT, 1, 0);
    tick(&mut dev, 0);
    assert!(dev.reset_requested);
}

#[test]
fn boot_time_defaults_to_a_value_the_guest_can_read_without_panicking() {
    let mut dev = bus();
    for index in 0..CMOS_SIZE as u8 {
        let _ = cmos_read(&mut dev, index, 0);
    }
    let seconds = cmos_read(&mut dev, CMOS_SECONDS as u8, 0);
    assert!(
        seconds <= 0x59 && seconds & 0x0f <= 9,
        "valid BCD seconds: {seconds:#x}"
    );
}

#[test]
fn level_triggered_uart_line_is_released_once_the_driver_reads_the_data() {
    const MASTER_ELCR_PORT: u16 = 0x4d0;
    const MASTER_COMMAND_PORT: u16 = 0x20;
    const NON_SPECIFIC_EOI: u32 = 0x20;
    let console = ConsoleHandle::default();
    let mut dev = Devices::new(console.clone(), DevConfig::default());
    init_pics(&mut dev);
    out(&mut dev, MASTER_ELCR_PORT, 1, 1 << IRQ_UART);
    out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
    console.write(b"x");
    tick(&mut dev, 0);
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
    out(&mut dev, MASTER_COMMAND_PORT, 1, NON_SPECIFIC_EOI);
    assert!(
        dev.pic.has_pending(),
        "level line: still asserted while the byte is unread"
    );
    assert_eq!(inp(&mut dev, UART_DATA_PORT, 1), b'x' as u32);
    assert!(
        !dev.pic.has_pending(),
        "reading the data cleared the cause, so the line must drop without waiting for a tick"
    );
    assert_eq!(dev.pic.ack(), None);
}

#[test]
fn edge_triggered_uart_line_is_unaffected_by_lowering() {
    let console = ConsoleHandle::default();
    let mut dev = Devices::new(console.clone(), DevConfig::default());
    init_pics(&mut dev);
    out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
    console.write(b"x");
    tick(&mut dev, 0);
    assert!(dev.pic.has_pending());
    // Servicing the cause does not retract an edge request that is already latched.
    assert_eq!(inp(&mut dev, UART_DATA_PORT, 1), b'x' as u32);
    assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
}

#[test]
fn into_disk_hands_back_the_attached_disk_with_its_contents() {
    const MARKER: &[u8] = b"survives a reboot";
    let dev = Devices::new(
        ConsoleHandle::default(),
        DevConfig {
            disk: Some(Disk::memory(vec![0; 4096])),
            net: Some(([2, 0, 0, 0, 0, 1], NetHandle::default())),
            virtio_console: true,
        },
    );
    let mut disk = dev.into_disk().expect("a disk was attached");
    disk.backend.write_at(512, MARKER).unwrap();
    // The next boot's devices are built around the returned disk and give it back again.
    let rebuilt = Devices::new(
        ConsoleHandle::default(),
        DevConfig {
            disk: Some(disk),
            ..Default::default()
        },
    );
    let mut disk = rebuilt.into_disk().unwrap();
    let mut read_back = [0u8; MARKER.len()];
    disk.backend.read_at(512, &mut read_back).unwrap();
    assert_eq!(read_back, MARKER);
}

#[test]
fn into_disk_without_a_disk_is_none() {
    assert!(bus().into_disk().is_none());
    assert!(bus_with_pci().into_disk().is_none());
}
