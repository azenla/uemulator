use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};

use uemulator::{ClockMode, ConsoleKind, Disk, EmulatedMachine, JitMode, MachineConfig, NetHandle};

#[path = "support/netgw.rs"]
mod netgw;

const GUEST_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];

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

fn stty(args: &[&str]) -> Option<String> {
    let out = Command::new("stty")
        .args(args)
        .stdin(Stdio::inherit())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[repr(C)]
#[derive(Default)]
struct WinSize {
    rows: u16,
    cols: u16,
    xpix: u16,
    ypix: u16,
}

unsafe extern "C" {
    fn ioctl(fd: i32, req: u64, ...) -> i32;
}

#[cfg(target_os = "macos")]
const TIOCGWINSZ: u64 = 0x4008_7468;
#[cfg(not(target_os = "macos"))]
const TIOCGWINSZ: u64 = 0x5413;

fn term_size() -> Option<(u16, u16)> {
    let mut ws = WinSize::default();
    let r = unsafe { ioctl(1, TIOCGWINSZ, &mut ws as *mut WinSize) };
    (r == 0 && ws.cols > 0 && ws.rows > 0).then_some((ws.cols, ws.rows))
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].as_str())
}

fn spawn_udp(net: NetHandle, spec: &str) {
    let (local, remote) = spec.split_once(',').expect("--net-udp LOCAL,REMOTE");
    let sock = std::net::UdpSocket::bind(local).expect("bind udp");
    sock.connect(remote).expect("connect udp");
    sock.set_read_timeout(Some(std::time::Duration::from_millis(2)))
        .unwrap();
    std::thread::spawn(move || {
        let mut buf = vec![0u8; 65536];
        loop {
            if let Ok(n) = sock.recv(&mut buf) {
                net.send_frame(&buf[..n]);
            }
            while let Some(f) = net.try_recv_frame() {
                let _ = sock.send(&f);
            }
        }
    });
}

fn spawn_gateway(net: NetHandle) {
    std::thread::spawn(move || {
        loop {
            if let Some(f) = net.recv_frame(std::time::Duration::from_millis(5))
                && let Some(r) = netgw::handle(&f)
            {
                net.send_frame(&r);
            }
        }
    });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = MachineConfig::new()
        .kernel(fetch("linux"))
        .log(|m| eprintln!("[uemu] {m}"))
        .debug(args.iter().any(|a| a == "--debug"))
        .verify_jit(args.iter().any(|a| a == "--verify"));
    if !args.iter().any(|a| a == "--no-initrd") {
        cfg = cfg.initrd(fetch("initrd.gz"));
    }
    if let Some(v) = arg(&args, "--mem") {
        cfg = cfg.memory_mib(v.parse().expect("--mem takes MiB"));
    }
    if let Some(v) = arg(&args, "--disk") {
        let (path, ro) = match v.strip_suffix(",ro") {
            Some(p) => (p, true),
            None => (v, false),
        };
        cfg = cfg.disk(Disk::open(path, ro).expect("open disk image"));
    }
    let want_net = arg(&args, "--net-udp").is_some() || args.iter().any(|a| a == "--net-loop");
    if want_net {
        cfg = cfg.net(GUEST_MAC);
    }
    if arg(&args, "--console") == Some("virtio") {
        cfg = cfg.console(ConsoleKind::Virtio);
    }
    if let Some(i) = args.iter().position(|a| a == "--time") {
        cfg = cfg.unix_time(args[i + 1].parse().expect("unix seconds"));
    }
    let tty = std::io::stdin().is_terminal();
    if tty || args.iter().any(|a| a == "--realtime") {
        cfg = cfg.clock(ClockMode::Realtime);
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
    if let Some(net) = machine.net() {
        if let Some(spec) = arg(&args, "--net-udp") {
            spawn_udp(net.clone(), spec);
        }
        if args.iter().any(|a| a == "--net-loop") {
            spawn_gateway(net);
        }
    }
    if let Some((c, r)) = term_size() {
        console.resize(c, r);
    }
    let sizer = console.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if let Some((c, r)) = term_size() {
                sizer.resize(c, r);
            }
        }
    });
    console.on_output(|b| {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(b);
        let _ = out.flush();
    });
    let saved_tty = if tty {
        let saved = stty(&["-g"]);
        stty(&["raw", "-echo"]);
        eprintln!("[uemu] raw terminal; Ctrl-] stops the machine\r");
        saved
    } else {
        None
    };
    let input = console.clone();
    let stopper = machine.stop_handle();
    std::thread::spawn(move || {
        let mut buf = [0u8; 256];
        while let Ok(n) = std::io::stdin().read(&mut buf) {
            if n == 0 {
                break;
            }
            if buf[..n].contains(&0x1d) {
                stopper.stop();
                break;
            }
            input.write(&buf[..n]);
        }
    });

    if let Some(i) = args.iter().position(|a| a == "--timeout") {
        let secs: u64 = args[i + 1].parse().expect("seconds");
        let stop = machine.stop_handle();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            stop.stop();
        });
    }
    if let Err(e) = machine.start() {
        eprintln!("error: {e}");
        exit(1);
    }
    let reason = machine.wait();
    if let Some(saved) = saved_tty {
        stty(&[&saved]);
    }
    eprintln!("machine exited: {reason:?}");
}
