use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use core::time::Duration;

use uefi::fs::FileSystem;
use uefi::prelude::*;
use uefi::proto::loaded_image::LoadedImage;
use uefi::runtime::ResetType;
use uemulator::{EmulatedMachine, ExitReason, JitMode, MachineConfig};

use crate::config::{Accel, CONFIG_FILE, Config};
use crate::console;

/// The vCPU calls the poll hook every few thousand instructions; the firmware is only asked
/// for input on every this-many of them.
const INPUT_EVERY: u32 = 32;
/// How long the guest being halted waits in the firmware per idle poll.
const IDLE_STALL: Duration = Duration::from_micros(200);
/// How long the last message stays up before shutdown (clean exit) or return to the firmware
/// (error), so it can be read.
const EXIT_MESSAGE_DELAY: Duration = Duration::from_secs(1);
const ERROR_MESSAGE_DELAY: Duration = Duration::from_secs(10);

// Constants of the days-from-civil algorithm in `unix_time`.
const SECS_PER_MINUTE: i64 = 60;
const SECS_PER_HOUR: i64 = 3600;
const SECS_PER_DAY: i64 = 86400;
const MONTHS_PER_YEAR: i64 = 12;
const FEBRUARY: i64 = 2;
const MARCH: i64 = 3;
const DAYS_PER_YEAR: i64 = 365;
/// A 400-year era of the Gregorian calendar.
const YEARS_PER_ERA: i64 = 400;
const DAYS_PER_ERA: i64 = 146097;
/// Days from 0000-03-01, the start of era 0, to 1970-01-01.
const DAYS_FROM_ERA_START_TO_UNIX_EPOCH: i64 = 719468;
/// Days in any five consecutive months of a year that starts in March (31, 30, 31, 30, 31).
const DAYS_PER_FIVE_MONTHS: i64 = 153;

#[entry]
fn efi_main() -> Status {
    uefi::helpers::init().expect("uefi helpers");
    match run() {
        Ok(reason) => {
            console::print(&format!("\nmachine exited: {reason:?}\n"));
            uefi::boot::stall(EXIT_MESSAGE_DELAY);
            uefi::runtime::reset(ResetType::SHUTDOWN, Status::SUCCESS, None)
        }
        Err(msg) => {
            console::print(&format!("uemulator: {msg}\n"));
            uefi::boot::stall(ERROR_MESSAGE_DELAY);
            Status::ABORTED
        }
    }
}

fn load_options() -> String {
    let Ok(image) = uefi::boot::open_protocol_exclusive::<LoadedImage>(uefi::boot::image_handle())
    else {
        return String::new();
    };
    image
        .load_options_as_cstr16()
        .map(String::from)
        .unwrap_or_default()
}

fn read_file(fs: &mut FileSystem, path: &str) -> Result<Vec<u8>, String> {
    let path = uefi::fs::PathBuf::from(
        uefi::CString16::try_from(path).map_err(|_| format!("bad path {path:?}"))?,
    );
    fs.read(&path)
        .map_err(|e| format!("cannot read {path}: {e}"))
}

fn run() -> Result<ExitReason, String> {
    let mut cfg = Config::default();
    let image = uefi::boot::image_handle();
    let sfs = uefi::boot::get_image_file_system(image)
        .map_err(|e| format!("no filesystem on the boot device: {e}"))?;
    let mut fs = FileSystem::new(sfs);

    if let Ok(bytes) = read_file(&mut fs, CONFIG_FILE) {
        cfg.apply(&String::from_utf8_lossy(&bytes))
            .map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    }
    // Load options come last so they override the config file.
    cfg.apply(&load_options())
        .map_err(|e| format!("load options: {e}"))?;

    console::print(&format!("uemulator: loading {}\n", cfg.kernel));
    let kernel = read_file(&mut fs, &cfg.kernel)?;
    let initrd = match &cfg.initrd {
        Some(p) => {
            console::print(&format!("uemulator: loading {p}\n"));
            Some(read_file(&mut fs, p)?)
        }
        None => None,
    };
    drop(fs);

    #[cfg(target_arch = "x86_64")]
    if cfg.accel != Accel::Emulator {
        match crate::hv::detect() {
            // The hardware path runs the whole guest itself and returns its exit reason.
            Some(kind) => {
                let mut cmdline = String::from(uemulator::DEFAULT_CMDLINE);
                for extra in &cfg.cmdline_extra {
                    cmdline.push(' ');
                    cmdline.push_str(extra);
                }
                return crate::hv::run(
                    kind,
                    &kernel,
                    initrd.as_deref(),
                    cfg.memory_mib,
                    &cmdline,
                    unix_time(),
                );
            }
            None if cfg.accel == Accel::Hardware => {
                return Err(
                    "no usable hardware virtualization (VMX with EPT, or SVM with NPT)".into(),
                );
            }
            None => console::print("uemulator: no hardware virtualization, emulating\n"),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    if cfg.accel == Accel::Hardware {
        return Err("hardware virtualization is only available on x86_64".into());
    }

    let mut mc = MachineConfig::new()
        .kernel(kernel)
        .memory_mib(cfg.memory_mib)
        // The library's JIT needs executable memory, which the UEFI build does not provide.
        .jit(JitMode::Off)
        .log(|s| console::print(&format!("{s}\n")))
        .poll(|con, idle| {
            // Idle polls (guest halted) always run; the others are thinned out to `INPUT_EVERY`.
            static TICKS: AtomicU32 = AtomicU32::new(0);
            if idle {
                uefi::boot::stall(IDLE_STALL);
            } else if !TICKS
                .fetch_add(1, Ordering::Relaxed)
                .is_multiple_of(INPUT_EVERY)
            {
                return;
            }
            console::flush();
            let input = console::read_input();
            if !input.is_empty() {
                con.write(&input);
            }
        });
    if let Some(t) = unix_time() {
        mc = mc.unix_time(t);
    }
    if let Some(i) = initrd {
        mc = mc.initrd(i);
    }
    for extra in &cfg.cmdline_extra {
        mc = mc.cmdline_append(extra.as_str());
    }

    let mut machine = EmulatedMachine::new(mc).map_err(|e| format!("{e}"))?;
    machine.console().on_output(console::queue);
    machine.start().map_err(|e| format!("{e}"))?;
    let reason = machine.wait();
    console::flush();
    Ok(reason)
}

/// The firmware's real-time clock as seconds since the Unix epoch. The time zone is ignored.
fn unix_time() -> Option<u64> {
    let time = uefi::runtime::get_time().ok()?;
    let (year, month, day) = (time.year() as i64, time.month() as i64, time.day() as i64);
    // Days-from-civil (Howard Hinnant): the year starts in March so the leap day is the last day
    // of the year, which makes the month-to-day-of-year step a fixed formula.
    let year = if month <= FEBRUARY { year - 1 } else { year };
    let era = year.div_euclid(YEARS_PER_ERA);
    let year_of_era = year.rem_euclid(YEARS_PER_ERA);
    let months_since_march = if month >= MARCH {
        month - MARCH
    } else {
        month + MONTHS_PER_YEAR - MARCH
    };
    let day_of_year = (DAYS_PER_FIVE_MONTHS * months_since_march + 2) / 5 + day - 1;
    let day_of_era =
        year_of_era * DAYS_PER_YEAR + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * DAYS_PER_ERA + day_of_era - DAYS_FROM_ERA_START_TO_UNIX_EPOCH;
    let secs = days * SECS_PER_DAY
        + time.hour() as i64 * SECS_PER_HOUR
        + time.minute() as i64 * SECS_PER_MINUTE
        + time.second() as i64;
    u64::try_from(secs).ok()
}
