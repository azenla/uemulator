use std::io::{IsTerminal, Read, Write};
use std::time::Duration;

use uemulator::{ClockMode, ConsoleKind, Disk, EmulatedMachine, JitMode, MachineConfig, NetHandle};

use crate::netgw;
use crate::term::{stty, term_size};
use crate::{ConsoleArg, MachineArgs};

/// QEMU's conventional default NIC MAC (52:54:00 is its locally administered prefix).
pub(crate) const GUEST_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
pub(crate) const DEFAULT_MEMORY_MIB: usize = 2048;
/// Ctrl-] (GS). Raw mode passes Ctrl-C and friends to the guest, so this is the host's escape.
const ESCAPE_BYTE: u8 = 0x1d;
const STDIN_CHUNK_LEN: usize = 256;
/// Polled rather than signal driven (SIGWINCH has no portable path here).
const TERM_SIZE_POLL_INTERVAL: Duration = Duration::from_millis(200);
const UDP_MAX_DATAGRAM: usize = 65536;
/// One thread both receives from the socket and drains guest frames, so a blocking recv must
/// time out quickly to keep guest-to-host frames flowing.
const UDP_RECV_TIMEOUT: Duration = Duration::from_millis(2);
const GATEWAY_RECV_TIMEOUT: Duration = Duration::from_millis(5);

fn spawn_udp(net: NetHandle, spec: &str) -> Result<(), String> {
    let (local, remote) = spec.split_once(',').ok_or("--net-udp takes LOCAL,REMOTE")?;
    let sock = std::net::UdpSocket::bind(local).map_err(|e| format!("bind {local}: {e}"))?;
    sock.connect(remote)
        .map_err(|e| format!("connect {remote}: {e}"))?;
    sock.set_read_timeout(Some(UDP_RECV_TIMEOUT))
        .map_err(|e| e.to_string())?;
    std::thread::spawn(move || {
        let mut buf = vec![0u8; UDP_MAX_DATAGRAM];
        loop {
            if let Ok(n) = sock.recv(&mut buf) {
                net.send_frame(&buf[..n]);
            }
            while let Some(frame) = net.try_recv_frame() {
                let _ = sock.send(&frame);
            }
        }
    });
    Ok(())
}

fn spawn_gateway(net: NetHandle) {
    std::thread::spawn(move || {
        loop {
            if let Some(frame) = net.recv_frame(GATEWAY_RECV_TIMEOUT)
                && let Some(reply) = netgw::handle(&frame)
            {
                net.send_frame(&reply);
            }
        }
    });
}

pub fn run(kernel: Vec<u8>, initramfs: Option<Vec<u8>>, args: MachineArgs) -> Result<(), String> {
    let mut cfg = MachineConfig::new()
        .kernel(kernel)
        .log(|m| eprintln!("[uemu] {m}"))
        .debug(args.debug)
        .verify_jit(args.verify);
    if let Some(initramfs) = initramfs {
        cfg = cfg.initrd(initramfs);
    }
    cfg = cfg.memory_mib(args.memory.unwrap_or(DEFAULT_MEMORY_MIB));
    if let Some(cmdline) = args.cmdline {
        cfg = cfg.cmdline(cmdline);
    }
    if let Some(extra) = &args.append {
        cfg = cfg.cmdline_append(extra.clone());
    }
    if let Some(spec) = &args.disk {
        let (path, read_only) = match spec.strip_suffix(",ro") {
            Some(path) => (path, true),
            None => (spec.as_str(), false),
        };
        cfg = cfg.disk(Disk::open(path, read_only).map_err(|e| format!("open disk {path}: {e}"))?);
    }
    if args.net_loop || args.net_user || args.net_udp.is_some() {
        cfg = cfg.net(GUEST_MAC);
    }
    if matches!(args.console, ConsoleArg::Virtio) {
        cfg = cfg.console(ConsoleKind::Virtio);
    }
    if let Some(t) = args.time {
        cfg = cfg.unix_time(t);
    }
    let tty = std::io::stdin().is_terminal();
    // Virtual time skips ahead while the guest idles, which is right for scripted runs but makes an
    // interactive session race; so pace to the wall clock when someone is typing.
    if tty || args.realtime {
        cfg = cfg.clock(ClockMode::Realtime);
    }
    if args.disable_jit {
        cfg = cfg.jit(JitMode::Off);
    }

    let mut machine = EmulatedMachine::new(cfg).map_err(|e| e.to_string())?;
    let console = machine.console();
    if let Some(net) = machine.net() {
        if let Some(spec) = &args.net_udp {
            spawn_udp(net.clone(), spec)?;
        }
        if args.net_loop {
            spawn_gateway(net.clone());
        }
        if args.net_user {
            crate::usernet::spawn(net);
        }
    }
    if let Some((cols, rows)) = term_size() {
        console.resize(cols, rows);
    }
    let sizer = console.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(TERM_SIZE_POLL_INTERVAL);
            if let Some((cols, rows)) = term_size() {
                sizer.resize(cols, rows);
            }
        }
    });
    console.on_output(|b| {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(b);
        let _ = out.flush();
    });
    // `stty -g` output is the exact setting string to hand back to `stty` on exit.
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
        let mut buf = [0u8; STDIN_CHUNK_LEN];
        while let Ok(n) = std::io::stdin().read(&mut buf) {
            if n == 0 {
                break;
            }
            if buf[..n].contains(&ESCAPE_BYTE) {
                stopper.stop();
                break;
            }
            input.write(&buf[..n]);
        }
    });
    if let Some(secs) = args.timeout {
        let stop = machine.stop_handle();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(secs));
            stop.stop();
        });
    }

    let started = machine.start().map_err(|e| e.to_string());
    let reason = started.map(|()| machine.wait());
    // Restore before `reason?` so a failed start does not leave the terminal raw.
    if let Some(saved) = saved_tty {
        stty(&[&saved]);
    }
    eprintln!("machine exited: {:?}", reason?);
    Ok(())
}
