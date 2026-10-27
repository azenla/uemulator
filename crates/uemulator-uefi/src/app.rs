use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use core::time::Duration;

use uefi::fs::FileSystem;
use uefi::prelude::*;
use uefi::proto::loaded_image::LoadedImage;
use uefi::runtime::{Daylight, ResetType};
use uemulator::{EmulatedMachine, ExitReason, JitMode, MachineConfig};

use crate::config::{Accel, CONFIG_FILE, Config};
use crate::console;
use crate::unix_time::FirmwareTime;

/// The vCPU calls the poll hook every few thousand instructions; the firmware is only asked
/// for input on every this-many of them.
const INPUT_EVERY: u32 = 32;
/// How long the guest being halted waits in the firmware per idle poll.
const IDLE_STALL: Duration = Duration::from_micros(200);
/// How long the last message stays up before shutdown (clean exit) or return to the firmware
/// (error), so it can be read.
const EXIT_MESSAGE_DELAY: Duration = Duration::from_secs(1);
const ERROR_MESSAGE_DELAY: Duration = Duration::from_secs(10);

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
                // The serial command line, not `DEFAULT_CMDLINE`: the hardware path only forwards
                // legacy port I/O to `Platform`, which has no virtio devices and no MMIO exits,
                // so an `hvc0` console would never appear.
                let mut cmdline = String::from(uemulator::DEFAULT_SERIAL_CMDLINE);
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

    // The emulator path uses the library's default virtio console (`hvc0`); its device is plain
    // PCI MMIO and DMA, which the no_std build supports like the hosted one.
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

/// The firmware's real-time clock as seconds since the Unix epoch, with its time zone and daylight
/// saving fields applied.
fn unix_time() -> Option<u64> {
    let time = uefi::runtime::get_time().ok()?;
    FirmwareTime {
        year: time.year(),
        month: time.month(),
        day: time.day(),
        hour: time.hour(),
        minute: time.minute(),
        second: time.second(),
        time_zone: time.time_zone(),
        in_daylight: time.daylight().contains(Daylight::IN_DAYLIGHT),
    }
    .unix_seconds()
}
