use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uemulator::{EmulatedMachine, ExitReason, JitMode, MachineConfig};

fn cached(name: &str) -> Option<Vec<u8>> {
    let dir = std::env::var("UEMULATOR_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache/uemulator")))
        .ok()?;
    std::fs::read(dir.join(name)).ok().filter(|d| !d.is_empty())
}

fn run_until(cfg: MachineConfig, needle: &str, limit: Duration) -> (String, Option<ExitReason>) {
    let mut m = EmulatedMachine::new(cfg).unwrap();
    let out = Arc::new(Mutex::new(Vec::<u8>::new()));
    let sink = out.clone();
    m.console().on_output(move |b| sink.lock().unwrap().extend_from_slice(b));
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

#[test]
fn prints_linux_version() {
    let Some(kernel) = cached("linux") else { return };
    let cfg = MachineConfig::new().kernel(kernel).jit(JitMode::Off);
    let (text, miss) = run_until(cfg, "Linux version 6.12", Duration::from_secs(120));
    assert!(miss.is_none(), "banner not seen:\n{text}");
}

#[test]
fn vfs_panic_then_reset() {
    let Some(kernel) = cached("linux") else { return };
    let cfg = MachineConfig::new().kernel(kernel).jit(JitMode::Off);
    let mut m = EmulatedMachine::new(cfg).unwrap();
    m.start().unwrap();
    assert_eq!(m.wait(), ExitReason::Reset);
}

#[test]
fn installer_userspace() {
    let (Some(kernel), Some(initrd)) = (cached("linux"), cached("initrd.gz")) else { return };
    let cfg = MachineConfig::new().kernel(kernel).initrd(initrd).jit(JitMode::Off);
    let (text, miss) = run_until(cfg, "Starting system log daemon", Duration::from_secs(240));
    assert!(miss.is_none(), "installer userspace not reached:\n{text}");
    assert!(text.contains("Run /init as init process"));
}
