//! The clock: the TSC (the emulator counts guest nanoseconds in it, and the handoff says so),
//! the CMOS real-time clock, and `Stall`.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::cpu::{self, inb, outb, rdtsc};
use crate::handoff::Handoff;
use crate::uefi::{Time, UNSPECIFIED_TIMEZONE};

static TSC_HZ: AtomicU64 = AtomicU64::new(1_000_000_000);
const NANOSECONDS: u128 = 1_000_000_000;
/// Waits shorter than this spin; longer ones sleep in `cpu::idle_wait`, whose granularity is
/// the PIT's.
const SPIN_BELOW_NS: u64 = 100_000;

const CMOS_INDEX: u16 = 0x70;
const CMOS_DATA: u16 = 0x71;
const RTC_SECONDS: u8 = 0x00;
const RTC_MINUTES: u8 = 0x02;
const RTC_HOURS: u8 = 0x04;
const RTC_DAY: u8 = 0x07;
const RTC_MONTH: u8 = 0x08;
const RTC_YEAR: u8 = 0x09;
const RTC_STATUS_A: u8 = 0x0a;
const RTC_STATUS_B: u8 = 0x0b;
const RTC_CENTURY: u8 = 0x32;
const STATUS_A_UPDATE_IN_PROGRESS: u8 = 0x80;
const STATUS_B_BINARY: u8 = 1 << 2;
const STATUS_B_24_HOUR: u8 = 1 << 1;
const HOUR_PM: u8 = 0x80;

pub fn init(handoff: &Handoff) {
    if handoff.tsc_hz != 0 {
        TSC_HZ.store(handoff.tsc_hz, Ordering::Relaxed);
    }
}

pub fn now_ns() -> u64 {
    (rdtsc() as u128 * NANOSECONDS / TSC_HZ.load(Ordering::Relaxed) as u128) as u64
}

/// Waits `microseconds`, letting timer events fire meanwhile.
pub fn stall(microseconds: u64) {
    let deadline = now_ns().saturating_add(microseconds.saturating_mul(1000));
    loop {
        crate::events::poll();
        let now = now_ns();
        if now >= deadline {
            return;
        }
        let remaining = deadline - now;
        if remaining < SPIN_BELOW_NS {
            core::hint::spin_loop();
        } else {
            cpu::idle_wait(remaining);
        }
    }
}

fn cmos(register: u8) -> u8 {
    outb(CMOS_INDEX, register);
    inb(CMOS_DATA)
}

fn from_bcd(value: u8) -> u8 {
    (value >> 4) * 10 + (value & 0xf)
}

/// The CMOS RTC as an EFI time. The library derives the RTC from the boot timestamp in the
/// handoff plus guest time, so the two never disagree.
pub fn get_time() -> Time {
    // Registers change under the reader during the update that rolls the seconds over.
    while cmos(RTC_STATUS_A) & STATUS_A_UPDATE_IN_PROGRESS != 0 {
        core::hint::spin_loop();
    }
    let status = cmos(RTC_STATUS_B);
    let convert = |value: u8| if status & STATUS_B_BINARY != 0 { value } else { from_bcd(value) };
    let mut hour = cmos(RTC_HOURS);
    let pm = hour & HOUR_PM != 0 && status & STATUS_B_24_HOUR == 0;
    hour = convert(hour & !HOUR_PM);
    if pm {
        hour = (hour % 12) + 12;
    }
    let century = convert(cmos(RTC_CENTURY)) as u16;
    let year = convert(cmos(RTC_YEAR)) as u16;
    Time {
        year: if century == 0 { 2000 + year } else { century * 100 + year },
        month: convert(cmos(RTC_MONTH)).max(1),
        day: convert(cmos(RTC_DAY)).max(1),
        hour,
        minute: convert(cmos(RTC_MINUTES)),
        second: convert(cmos(RTC_SECONDS)),
        pad1: 0,
        nanosecond: 0,
        time_zone: UNSPECIFIED_TIMEZONE,
        daylight: 0,
        pad2: 0,
    }
}
