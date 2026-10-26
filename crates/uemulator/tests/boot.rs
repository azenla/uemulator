use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uemulator::{EmulatedMachine, ExitReason, JitMode, MachineConfig};

fn cached(name: &str) -> Option<Vec<u8>> {
    let dir = std::env::var("UEMULATOR_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".uemulator/cache")))
        .ok()?;
    std::fs::read(dir.join("quick/debian/artifacts").join(name))
        .ok()
        .filter(|d| !d.is_empty())
}

fn run_until(cfg: MachineConfig, needle: &str, limit: Duration) -> (String, Option<ExitReason>) {
    let mut m = EmulatedMachine::new(cfg).unwrap();
    let out = Arc::new(Mutex::new(Vec::<u8>::new()));
    let sink = out.clone();
    m.console()
        .on_output(move |b| sink.lock().unwrap().extend_from_slice(b));
    let stop = m.stop_handle();
    m.start().unwrap();
    let start = Instant::now();
    let mut found = false;
    while start.elapsed() < limit {
        std::thread::sleep(Duration::from_millis(100));
        if String::from_utf8_lossy(&out.lock().unwrap()).contains(needle) {
            found = true;
            break;
        }
    }
    stop.stop();
    let reason = m.wait();
    let text = String::from_utf8_lossy(&out.lock().unwrap()).into_owned();
    (text, if found { None } else { Some(reason) })
}

fn prints_linux_version(jit: JitMode) {
    let Some(kernel) = cached("linux") else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).jit(jit);
    let (text, miss) = run_until(cfg, "Linux version 6.12", Duration::from_secs(120));
    assert!(miss.is_none(), "banner not seen:\n{text}");
}

fn vfs_panic_then_reset(jit: JitMode) {
    let Some(kernel) = cached("linux") else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).jit(jit);
    let mut m = EmulatedMachine::new(cfg).unwrap();
    m.start().unwrap();
    assert_eq!(m.wait(), ExitReason::Reset);
}

fn installer_userspace(jit: JitMode) {
    let (Some(kernel), Some(initrd)) = (cached("linux"), cached("initrd.gz")) else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).initrd(initrd).jit(jit);
    let (text, miss) = run_until(cfg, "Starting system log daemon", Duration::from_secs(240));
    assert!(miss.is_none(), "installer userspace not reached:\n{text}");
    assert!(text.contains("Run /init as init process"));
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
    let (text, miss) = run_until(cfg, "Starting system log daemon", Duration::from_secs(400));
    assert!(miss.is_none(), "lockstep verification failed:\n{text}");
}

fn memory_size(mib: usize, needle: &str) {
    let Some(kernel) = cached("linux") else {
        return;
    };
    let cfg = MachineConfig::new().kernel(kernel).memory_mib(mib);
    let (text, miss) = run_until(cfg, needle, Duration::from_secs(120));
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
    for mib in [0, 16, 4096] {
        let cfg = MachineConfig::new().kernel(vec![0; 4096]).memory_mib(mib);
        assert!(EmulatedMachine::new(cfg).is_err());
    }
}
