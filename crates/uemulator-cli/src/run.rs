use std::io::{IsTerminal, Read, Write};
use std::time::Duration;

use uemulator::{ClockMode, ConsoleKind, Disk, EmulatedMachine, JitMode, MachineConfig, NetHandle};

use crate::netgw;
use crate::term::{stty, term_size};
use crate::{ConsoleArg, MachineArgs};

const GUEST_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];

fn spawn_udp(net: NetHandle, spec: &str) -> Result<(), String> {
    let (local, remote) = spec.split_once(',').ok_or("--net-udp takes LOCAL,REMOTE")?;
    let sock = std::net::UdpSocket::bind(local).map_err(|e| format!("bind {local}: {e}"))?;
    sock.connect(remote)
        .map_err(|e| format!("connect {remote}: {e}"))?;
    sock.set_read_timeout(Some(Duration::from_millis(2)))
        .map_err(|e| e.to_string())?;
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
    Ok(())
}

fn spawn_gateway(net: NetHandle) {
    std::thread::spawn(move || {
        loop {
            if let Some(f) = net.recv_frame(Duration::from_millis(5))
                && let Some(r) = netgw::handle(&f)
            {
                net.send_frame(&r);
            }
        }
    });
}

pub fn run(kernel: Vec<u8>, initramfs: Option<Vec<u8>>, a: MachineArgs) -> Result<(), String> {
    let mut cfg = MachineConfig::new()
        .kernel(kernel)
        .log(|m| eprintln!("[uemu] {m}"))
        .debug(a.debug)
        .verify_jit(a.verify);
    if let Some(i) = initramfs {
        cfg = cfg.initrd(i);
    }
    cfg = cfg.memory_mib(a.memory.unwrap_or(2048));
    if let Some(c) = a.cmdline {
        cfg = cfg.cmdline(c);
    }
    if let Some(extra) = &a.append {
        cfg = cfg.cmdline_append(extra.clone());
    }
    if let Some(v) = &a.disk {
        let (path, ro) = match v.strip_suffix(",ro") {
            Some(p) => (p, true),
            None => (v.as_str(), false),
        };
        cfg = cfg.disk(Disk::open(path, ro).map_err(|e| format!("open disk {path}: {e}"))?);
    }
    if a.net_loop || a.net_user || a.net_udp.is_some() {
        cfg = cfg.net(GUEST_MAC);
    }
    if matches!(a.console, ConsoleArg::Virtio) {
        cfg = cfg.console(ConsoleKind::Virtio);
    }
    if let Some(t) = a.time {
        cfg = cfg.unix_time(t);
    }
    let tty = std::io::stdin().is_terminal();
    if tty || a.realtime {
        cfg = cfg.clock(ClockMode::Realtime);
    }
    if a.disable_jit {
        cfg = cfg.jit(JitMode::Off);
    }

    let mut machine = EmulatedMachine::new(cfg).map_err(|e| e.to_string())?;
    let console = machine.console();
    if let Some(net) = machine.net() {
        if let Some(spec) = &a.net_udp {
            spawn_udp(net.clone(), spec)?;
        }
        if a.net_loop {
            spawn_gateway(net.clone());
        }
        if a.net_user {
            crate::usernet::spawn(net);
        }
    }
    if let Some((c, r)) = term_size() {
        console.resize(c, r);
    }
    let sizer = console.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(200));
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
    if let Some(secs) = a.timeout {
        let stop = machine.stop_handle();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(secs));
            stop.stop();
        });
    }

    let started = machine.start().map_err(|e| e.to_string());
    let reason = started.map(|()| machine.wait());
    if let Some(saved) = saved_tty {
        stty(&[&saved]);
    }
    eprintln!("machine exited: {:?}", reason?);
    Ok(())
}
