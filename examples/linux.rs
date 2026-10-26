use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

use uemulator::{EmulatedMachine, JitMode, MachineConfig};

const BASE: &str = "https://deb.debian.org/debian/dists/stable/main/installer-amd64/current/images/netboot/debian-installer/amd64";

fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("UEMULATOR_CACHE") {
        return d.into();
    }
    let home = std::env::var("HOME").expect("HOME not set");
    Path::new(&home).join(".cache/uemulator")
}

fn fetch(name: &str) -> Vec<u8> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).expect("create cache dir");
    let path = dir.join(name);
    if std::fs::metadata(&path)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
    {
        return std::fs::read(&path).expect("read cache");
    }
    let part = dir.join(format!("{name}.part"));
    eprintln!("downloading {name}");
    let ok = Command::new("curl")
        .args(["-fL", "-o"])
        .arg(&part)
        .arg(format!("{BASE}/{name}"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        let _ = std::fs::remove_file(&part);
        eprintln!("download of {name} failed");
        exit(1);
    }
    std::fs::rename(&part, &path).expect("finalize download");
    std::fs::read(&path).expect("read cache")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = MachineConfig::new().kernel(fetch("linux"));
    if !args.iter().any(|a| a == "--no-initrd") {
        cfg = cfg.initrd(fetch("initrd.gz"));
    }
    if args.iter().any(|a| a == "--no-jit") {
        cfg = cfg.jit(JitMode::Off);
    }
    if let Some(i) = args.iter().position(|a| a == "--cmdline") {
        cfg = cfg.cmdline(args[i + 1].clone());
    }

    let mut machine = match EmulatedMachine::new(cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let console = machine.console();
    console.on_output(|b| {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(b);
        let _ = out.flush();
    });
    let input = console.clone();
    std::thread::spawn(move || {
        let mut buf = [0u8; 256];
        while let Ok(n) = std::io::stdin().read(&mut buf) {
            if n == 0 {
                break;
            }
            input.write(&buf[..n]);
        }
    });

    if let Err(e) = machine.start() {
        eprintln!("error: {e}");
        exit(1);
    }
    eprintln!("machine exited: {:?}", machine.wait());
}
