use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uemulator::{EmulatedMachine, ExitReason, JitMode, MachineConfig};

/// Where the CLI's `quick debian` command caches the kernel and installer initrd. A test that
/// needs an artifact returns early (passes) when it is missing, so a fresh checkout stays green
/// without downloads.
const DEBIAN_ARTIFACTS: &str = "quick/debian/artifacts";
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Only guards against a hang: even the interpreter reaches the banner well within this.
const BANNER_TIMEOUT: Duration = Duration::from_secs(120);
/// Slower than the banner: also decompresses and unpacks the ~39 MB installer initrd.
const USERSPACE_TIMEOUT: Duration = Duration::from_secs(240);
/// The lockstep verifier runs every block twice.
const LOCKSTEP_TIMEOUT: Duration = Duration::from_secs(400);

const KERNEL_BANNER: &str = "Linux version 6.12";
/// Printed by the installer's init scripts once userspace is up and starting services.
const USERSPACE_MARKER: &str = "Starting system log daemon";
const INIT_STARTED: &str = "Run /init as init process";

fn cached(name: &str) -> Option<Vec<u8>> {
    let dir = std::env::var("UEMULATOR_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".uemulator/cache")))
        .ok()?;
    std::fs::read(dir.join(DEBIAN_ARTIFACTS).join(name))
        .ok()
        .filter(|data| !data.is_empty())
}

/// Boots `cfg`, collects console output, and polls for `needle` until `limit`. Returns the output
/// and `None` when the needle appeared, or the exit reason when it did not (the guest may have
/// exited early, or the machine was stopped at the limit).
fn run_until(cfg: MachineConfig, needle: &str, limit: Duration) -> (String, Option<ExitReason>) {
    let mut machine = EmulatedMachine::new(cfg).unwrap();
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let sink = output.clone();
    machine
        .console()
        .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
    let stop = machine.stop_handle();
    machine.start().unwrap();
    let start = Instant::now();
    let mut found = false;
    while start.elapsed() < limit {
        std::thread::sleep(POLL_INTERVAL);
        if String::from_utf8_lossy(&output.lock().unwrap()).contains(needle) {
            found = true;
            break;
        }
    }
    stop.stop();
    let reason = machine.wait();
    let text = String::from_utf8_lossy(&output.lock().unwrap()).into_owned();
    (text, if found { None } else { Some(reason) })
}

/// Guards the loader and early boot: the decompressor runs, long-mode entry works and the kernel
/// reaches `printk` over the serial console.
fn prints_linux_version(jit: JitMode) {
    let Some(kernel) = cached("linux") else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).jit(jit);
    let (text, miss) = run_until(cfg, KERNEL_BANNER, BANNER_TIMEOUT);
    assert!(miss.is_none(), "banner not seen:\n{text}");
}

/// With no initrd and no disk the kernel cannot mount a root filesystem and panics; the default
/// command line has `panic=-1`, so it reboots at once, which must surface as `ExitReason::Reset`
/// (reset ports and exit mapping work) rather than a hang.
fn vfs_panic_then_reset(jit: JitMode) {
    let Some(kernel) = cached("linux") else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).jit(jit);
    let mut m = EmulatedMachine::new(cfg).unwrap();
    m.start().unwrap();
    assert_eq!(m.wait(), ExitReason::Reset);
}

/// Boots the Debian installer initrd to userspace: exercises initrd placement, the timer and
/// interrupt paths and a large amount of ordinary user-mode code.
fn installer_userspace(jit: JitMode) {
    let (Some(kernel), Some(initrd)) = (cached("linux"), cached("initrd.gz")) else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).initrd(initrd).jit(jit);
    let (text, miss) = run_until(cfg, USERSPACE_MARKER, USERSPACE_TIMEOUT);
    assert!(miss.is_none(), "installer userspace not reached:\n{text}");
    assert!(text.contains(INIT_STARTED));
}

#[test]
fn interp_banner() {
    prints_linux_version(JitMode::Off);
}

#[test]
fn interp_vfs_panic_reset() {
    vfs_panic_then_reset(JitMode::Off);
}

#[test]
fn interp_installer() {
    installer_userspace(JitMode::Off);
}

#[test]
fn jit_banner() {
    prints_linux_version(JitMode::Auto);
}

#[test]
fn jit_vfs_panic_reset() {
    vfs_panic_then_reset(JitMode::Auto);
}

#[test]
fn jit_installer() {
    installer_userspace(JitMode::Auto);
}

#[test]
#[ignore = "slow: runs the JIT in lockstep with the interpreter"]
fn jit_lockstep_verify() {
    let (Some(kernel), Some(initrd)) = (cached("linux"), cached("initrd.gz")) else {
        return;
    };
    let cfg = MachineConfig::new()
        .kernel(kernel)
        .initrd(initrd)
        .verify_jit(true);
    let (text, miss) = run_until(cfg, USERSPACE_MARKER, LOCKSTEP_TIMEOUT);
    assert!(miss.is_none(), "lockstep verification failed:\n{text}");
}

/// The kernel prints the e820 map it was given; the usable range starting at 1 MiB must end at
/// the configured size, so `needle` is that line. Guards the e820 table following `memory_mib`.
fn memory_size(mib: usize, needle: &str) {
    let Some(kernel) = cached("linux") else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).memory_mib(mib);
    let (text, miss) = run_until(cfg, needle, BANNER_TIMEOUT);
    assert!(miss.is_none(), "e820 for {mib} MiB not seen:\n{text}");
}

#[test]
fn memory_128_mib() {
    memory_size(128, "[mem 0x0000000000100000-0x0000000007ffffff] usable");
}

#[test]
fn memory_2048_mib() {
    memory_size(2048, "[mem 0x0000000000100000-0x000000007fffffff] usable");
}

#[test]
fn memory_bounds_rejected() {
    // 0 and 16 are below MIN_MEMORY_MIB, 4096 is above MAX_MEMORY_MIB. The kernel image is
    // garbage, so `new` fails for that reason alone too: this does not isolate the size check.
    for mib in [0, 16, 4096] {
        let cfg = MachineConfig::new().kernel(vec![0; 4096]).memory_mib(mib);
        assert!(EmulatedMachine::new(cfg).is_err());
    }
}
