use std::io::{IsTerminal, Read, Write};
use std::time::Duration;

use uemulator::{
    ClockMode, ConsoleKind, Disk, EmulatedMachine, ExitReason, JitMode, MachineConfig, NetHandle,
};

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

/// Splits `--net-udp LOCAL,REMOTE` at the first comma.
fn parse_udp_spec(spec: &str) -> Result<(&str, &str), String> {
    spec.split_once(',')
        .ok_or_else(|| "--net-udp takes LOCAL,REMOTE".to_string())
}

fn open_udp_socket(spec: &str) -> Result<std::net::UdpSocket, String> {
    let (local, remote) = parse_udp_spec(spec)?;
    let sock = std::net::UdpSocket::bind(local).map_err(|e| format!("bind {local}: {e}"))?;
    sock.connect(remote)
        .map_err(|e| format!("connect {remote}: {e}"))?;
    sock.set_read_timeout(Some(UDP_RECV_TIMEOUT))
        .map_err(|e| e.to_string())?;
    Ok(sock)
}

/// Splits `--disk PATH[,ro]` into the path and whether the disk is read-only.
fn parse_disk_spec(spec: &str) -> (&str, bool) {
    match spec.strip_suffix(",ro") {
        Some(path) => (path, true),
        None => (spec, false),
    }
}

fn network_requested(args: &MachineArgs) -> bool {
    args.net_loop || args.net_user || args.net_udp.is_some()
}

/// Raw mode passes Ctrl-C and friends to the guest, so a chunk holding the escape byte is the
/// host asking to stop. The whole chunk is dropped, not forwarded.
fn contains_escape(chunk: &[u8]) -> bool {
    chunk.contains(&ESCAPE_BYTE)
}

/// Virtual time skips ahead while the guest idles, which is right for scripted runs but makes an
/// interactive session race; so pace to the wall clock when someone is typing or asked for it.
fn wants_realtime_clock(stdin_is_terminal: bool, realtime_flag: bool) -> bool {
    stdin_is_terminal || realtime_flag
}

/// Puts the terminal behind `stty` into raw mode. Returns the settings to hand back to `stty` on
/// exit, or `None` if raw mode was not entered (no `stty`, not a terminal, not unix). Without the
/// saved settings the terminal could not be restored, so raw mode is not attempted then.
fn enter_raw_mode(stty: impl Fn(&[&str]) -> Option<String>) -> Option<String> {
    let saved = stty(&["-g"])?;
    if stty(&["raw", "-echo"]).is_none() {
        stty(&[&saved]);
        return None;
    }
    Some(saved)
}

fn spawn_udp(net: NetHandle, spec: &str) -> Result<(), String> {
    let sock = open_udp_socket(spec)?;
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

fn build_config(
    kernel: Vec<u8>,
    initramfs: Option<Vec<u8>>,
    args: &MachineArgs,
    stdin_is_terminal: bool,
) -> Result<MachineConfig, String> {
    let mut cfg = MachineConfig::new()
        .kernel(kernel)
        .log(|m| eprintln!("[uemu] {m}"))
        .debug(args.debug)
        .verify_jit(args.verify);
    if let Some(initramfs) = initramfs {
        cfg = cfg.initrd(initramfs);
    }
    cfg = cfg.memory_mib(args.memory.unwrap_or(DEFAULT_MEMORY_MIB));
    if let Some(cmdline) = &args.cmdline {
        cfg = cfg.cmdline(cmdline.clone());
    }
    if let Some(extra) = &args.append {
        cfg = cfg.cmdline_append(extra.clone());
    }
    if let Some(spec) = &args.disk {
        let (path, read_only) = parse_disk_spec(spec);
        cfg = cfg.disk(Disk::open(path, read_only).map_err(|e| format!("open disk {path}: {e}"))?);
    }
    if network_requested(args) {
        cfg = cfg.net(GUEST_MAC);
    }
    if matches!(args.console, ConsoleArg::Virtio) {
        cfg = cfg.console(ConsoleKind::Virtio);
    }
    if let Some(t) = args.time {
        cfg = cfg.unix_time(t);
    }
    if wants_realtime_clock(stdin_is_terminal, args.realtime) {
        cfg = cfg.clock(ClockMode::Realtime);
    }
    if args.disable_jit {
        cfg = cfg.jit(JitMode::Off);
    }
    Ok(cfg)
}

/// Runs the machine to its end and reports why it ended. `Err` is a setup failure; the machine's
/// own failures come back as an `ExitReason` for `main` to map to a status.
pub fn run(
    kernel: Vec<u8>,
    initramfs: Option<Vec<u8>>,
    args: MachineArgs,
) -> Result<ExitReason, String> {
    let tty = std::io::stdin().is_terminal();
    let cfg = build_config(kernel, initramfs, &args, tty)?;
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
    let saved_tty = if tty { enter_raw_mode(stty) } else { None };
    if saved_tty.is_some() {
        eprintln!("[uemu] raw terminal; Ctrl-] stops the machine\r");
    }
    let input = console.clone();
    let stopper = machine.stop_handle();
    std::thread::spawn(move || {
        let mut buf = [0u8; STDIN_CHUNK_LEN];
        while let Ok(n) = std::io::stdin().read(&mut buf) {
            if n == 0 {
                break;
            }
            if contains_escape(&buf[..n]) {
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
    let reason = reason?;
    eprintln!("machine exited: {reason:?}");
    Ok(reason)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use clap::Parser;

    use super::*;
    use crate::test_support::TempDir;
    use crate::{Cli, Cmd};

    const SAVED_SETTINGS: &str = "4500:5:bf:8a3b:3:1c:7f:15:4:0:1:0:11:13:1a:0:12:f:17:16:0:0";
    const MIN_MEMORY_MIB: usize = 64;
    const MAX_MEMORY_MIB: usize = 3072;

    fn boot_args(extra: &[&str]) -> MachineArgs {
        let mut argv = vec!["uemulator", "boot", "--kernel", "k"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).unwrap().cmd {
            Cmd::Boot { machine, .. } => machine,
            Cmd::Quick { .. } => unreachable!(),
        }
    }

    /// What the library says about `cfg` before it looks at the kernel. The kernel is empty, so
    /// the only way to get past the configuration checks is to fail on the kernel image.
    fn machine_error(cfg: MachineConfig) -> String {
        match EmulatedMachine::new(cfg) {
            Ok(_) => panic!("an empty kernel must not boot"),
            Err(e) => e.to_string(),
        }
    }

    /// A fake `stty` that records its calls and answers from a script.
    struct FakeStty {
        calls: RefCell<Vec<Vec<String>>>,
        fail_on: Option<&'static str>,
    }

    impl FakeStty {
        fn working() -> Self {
            FakeStty {
                calls: RefCell::new(Vec::new()),
                fail_on: None,
            }
        }

        fn failing_on(first_arg: &'static str) -> Self {
            FakeStty {
                calls: RefCell::new(Vec::new()),
                fail_on: Some(first_arg),
            }
        }

        fn call(&self, args: &[&str]) -> Option<String> {
            self.calls
                .borrow_mut()
                .push(args.iter().map(|a| a.to_string()).collect());
            if self.fail_on == Some(args[0]) {
                return None;
            }
            Some(if args[0] == "-g" {
                SAVED_SETTINGS.to_string()
            } else {
                String::new()
            })
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.borrow().clone()
        }
    }

    #[test]
    fn disk_spec_without_suffix_is_read_write() {
        assert_eq!(parse_disk_spec("disk.img"), ("disk.img", false));
    }

    #[test]
    fn disk_spec_with_ro_suffix_is_read_only() {
        assert_eq!(parse_disk_spec("disk.img,ro"), ("disk.img", true));
    }

    #[test]
    fn disk_spec_only_strips_one_ro_suffix() {
        assert_eq!(parse_disk_spec("a,ro,ro"), ("a,ro", true));
    }

    #[test]
    fn disk_spec_other_suffixes_stay_part_of_the_path() {
        assert_eq!(parse_disk_spec("disk.img,rw"), ("disk.img,rw", false));
        assert_eq!(parse_disk_spec("ro"), ("ro", false));
        assert_eq!(parse_disk_spec("disk.img,RO"), ("disk.img,RO", false));
        assert_eq!(parse_disk_spec("a,roo"), ("a,roo", false));
    }

    #[test]
    fn disk_spec_edge_cases() {
        assert_eq!(parse_disk_spec(",ro"), ("", true));
        assert_eq!(parse_disk_spec(""), ("", false));
        assert_eq!(parse_disk_spec("/a b/c d.img,ro"), ("/a b/c d.img", true));
    }

    #[test]
    fn udp_spec_splits_at_the_first_comma() {
        assert_eq!(
            parse_udp_spec("127.0.0.1:1234,127.0.0.1:5678"),
            Ok(("127.0.0.1:1234", "127.0.0.1:5678"))
        );
        assert_eq!(parse_udp_spec("a,b,c"), Ok(("a", "b,c")));
        assert_eq!(parse_udp_spec(","), Ok(("", "")));
    }

    #[test]
    fn udp_spec_without_comma_is_rejected_with_usage() {
        assert_eq!(
            parse_udp_spec("127.0.0.1:1234"),
            Err("--net-udp takes LOCAL,REMOTE".to_string())
        );
        assert!(parse_udp_spec("").is_err());
    }

    #[test]
    fn udp_socket_is_bound_connected_and_has_a_short_read_timeout() {
        let peer = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let sock = open_udp_socket(&format!("127.0.0.1:0,{}", peer.local_addr().unwrap())).unwrap();
        assert_eq!(sock.read_timeout().unwrap(), Some(UDP_RECV_TIMEOUT));
        assert_eq!(sock.peer_addr().unwrap(), peer.local_addr().unwrap());
        sock.send(b"frame").unwrap();
        let mut buf = [0u8; 16];
        let (n, from) = peer.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"frame");
        assert_eq!(from, sock.local_addr().unwrap());
    }

    #[test]
    fn udp_socket_receive_times_out_instead_of_blocking() {
        let peer = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let sock = open_udp_socket(&format!("127.0.0.1:0,{}", peer.local_addr().unwrap())).unwrap();
        let mut buf = [0u8; 16];
        let started = std::time::Instant::now();
        assert!(sock.recv(&mut buf).is_err());
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn udp_socket_bind_failure_names_the_local_address() {
        let taken = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let local = taken.local_addr().unwrap();
        let err = open_udp_socket(&format!("{local},127.0.0.1:9")).unwrap_err();
        assert!(err.starts_with(&format!("bind {local}: ")), "{err}");
        let err = open_udp_socket("not-an-address,127.0.0.1:9").unwrap_err();
        assert!(err.starts_with("bind not-an-address: "), "{err}");
    }

    #[test]
    fn udp_socket_connect_failure_names_the_remote_address() {
        let err = open_udp_socket("127.0.0.1:0,not-an-address").unwrap_err();
        assert!(err.starts_with("connect not-an-address: "), "{err}");
    }

    #[test]
    fn udp_socket_without_comma_reports_usage() {
        assert_eq!(
            open_udp_socket("127.0.0.1:0").unwrap_err(),
            "--net-udp takes LOCAL,REMOTE"
        );
    }

    #[test]
    fn network_is_requested_by_any_backend_flag() {
        assert!(!network_requested(&boot_args(&[])));
        assert!(network_requested(&boot_args(&["--net-loop"])));
        assert!(network_requested(&boot_args(&["--net-user"])));
        assert!(network_requested(&boot_args(&["--net-udp", "a,b"])));
        assert!(network_requested(&boot_args(&["--net-loop", "--net-user"])));
    }

    #[test]
    fn escape_byte_is_ctrl_right_bracket() {
        assert_eq!(ESCAPE_BYTE, 0x1d);
        assert_eq!(ESCAPE_BYTE, b']' & 0x1f);
    }

    #[test]
    fn chunk_with_escape_byte_anywhere_stops_the_machine() {
        assert!(contains_escape(&[ESCAPE_BYTE]));
        assert!(contains_escape(&[b'a', b'b', ESCAPE_BYTE]));
        assert!(contains_escape(&[ESCAPE_BYTE, b'a']));
        assert!(contains_escape(b"typed ahead \x1d and more"));
    }

    #[test]
    fn chunk_without_escape_byte_is_forwarded() {
        assert!(!contains_escape(&[]));
        assert!(!contains_escape(b"ls -l\r"));
        assert!(!contains_escape(&[0x03, 0x1c, 0x1e, 0x04]));
        assert!(!contains_escape(&[0xff; STDIN_CHUNK_LEN]));
    }

    #[test]
    fn realtime_clock_follows_terminal_or_flag() {
        assert!(!wants_realtime_clock(false, false));
        assert!(wants_realtime_clock(true, false));
        assert!(wants_realtime_clock(false, true));
        assert!(wants_realtime_clock(true, true));
    }

    #[test]
    fn raw_mode_saves_settings_then_goes_raw() {
        let stty = FakeStty::working();
        let saved = enter_raw_mode(|args| stty.call(args));
        assert_eq!(saved.as_deref(), Some(SAVED_SETTINGS));
        assert_eq!(
            stty.calls(),
            vec![vec!["-g".to_string()], vec!["raw".into(), "-echo".into()]]
        );
    }

    #[test]
    fn raw_mode_is_not_entered_when_stty_is_unavailable() {
        // This is the non-unix case, where `stty` always returns `None`: no raw mode, so the
        // caller must not claim there is one.
        assert_eq!(enter_raw_mode(|_| None), None);
    }

    #[test]
    fn raw_mode_is_not_attempted_without_saved_settings() {
        let stty = FakeStty::failing_on("-g");
        assert_eq!(enter_raw_mode(|args| stty.call(args)), None);
        assert_eq!(stty.calls(), vec![vec!["-g".to_string()]]);
    }

    #[test]
    fn failed_raw_mode_restores_the_saved_settings() {
        let stty = FakeStty::failing_on("raw");
        assert_eq!(enter_raw_mode(|args| stty.call(args)), None);
        assert_eq!(
            stty.calls(),
            vec![
                vec!["-g".to_string()],
                vec!["raw".into(), "-echo".into()],
                vec![SAVED_SETTINGS.to_string()],
            ]
        );
    }

    #[cfg(not(unix))]
    #[test]
    fn real_stty_never_enters_raw_mode_off_unix() {
        assert_eq!(enter_raw_mode(stty), None);
    }

    #[test]
    fn guest_mac_is_a_locally_administered_unicast_address() {
        assert_eq!(GUEST_MAC[0] & 0x01, 0, "unicast");
        assert_eq!(GUEST_MAC[0] & 0x02, 0x02, "locally administered");
        assert_eq!(GUEST_MAC[..3], [0x52, 0x54, 0x00]);
    }

    #[test]
    fn default_memory_fits_the_library_limits() {
        assert!((MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&DEFAULT_MEMORY_MIB));
    }

    #[test]
    fn memory_below_the_library_minimum_is_rejected_by_the_library() {
        let memory = (MIN_MEMORY_MIB - 1).to_string();
        let cfg = build_config(
            vec![],
            None,
            &boot_args(&["-m", &memory, "--disable-jit"]),
            false,
        )
        .unwrap();
        assert!(machine_error(cfg).contains("between 64 and 3072"));
    }

    #[test]
    fn memory_above_the_library_maximum_is_rejected_by_the_library() {
        let memory = (MAX_MEMORY_MIB + 1).to_string();
        let cfg = build_config(
            vec![],
            None,
            &boot_args(&["--memory", &memory, "--disable-jit"]),
            false,
        )
        .unwrap();
        assert!(machine_error(cfg).contains("between 64 and 3072"));
    }

    #[test]
    fn the_minimum_memory_is_accepted_by_the_library() {
        // Only the minimum is probed: acceptance at the maximum would allocate 3 GiB of guest RAM.
        let memory = MIN_MEMORY_MIB.to_string();
        let cfg = build_config(
            vec![],
            None,
            &boot_args(&["-m", &memory, "--disable-jit"]),
            false,
        )
        .unwrap();
        assert!(machine_error(cfg).contains("kernel"));
    }

    #[test]
    fn missing_disk_image_is_a_setup_error_naming_the_path() {
        let dir = TempDir::new("run-disk");
        let missing = dir.path().join("missing.img");
        let spec = missing.to_string_lossy().into_owned();
        let err = build_config(vec![], None, &boot_args(&["--disk", &spec]), false)
            .err()
            .unwrap();
        assert!(err.starts_with(&format!("open disk {spec}: ")), "{err}");
    }

    #[test]
    fn read_only_suffix_is_not_part_of_the_opened_path() {
        let dir = TempDir::new("run-disk-ro");
        let missing = dir.path().join("missing.img");
        let path = missing.to_string_lossy().into_owned();
        let spec = format!("{path},ro");
        let err = build_config(vec![], None, &boot_args(&["--disk", &spec]), false)
            .err()
            .unwrap();
        assert!(err.starts_with(&format!("open disk {path}: ")), "{err}");
    }

    #[test]
    fn existing_disk_images_open_read_write_and_read_only() {
        let dir = TempDir::new("run-disk-ok");
        let image = dir.path().join("disk.img");
        std::fs::write(&image, vec![0u8; 4096]).unwrap();
        let path = image.to_string_lossy().into_owned();
        for spec in [path.clone(), format!("{path},ro")] {
            assert!(
                build_config(vec![], None, &boot_args(&["--disk", &spec]), false).is_ok(),
                "{spec}"
            );
        }
    }

    #[test]
    fn every_backend_and_console_combination_builds_a_config() {
        for flags in [
            &["--net-loop"][..],
            &["--net-user"],
            &["--net-udp", "a,b"],
            &["--console", "virtio"],
            &["--console", "serial"],
            &["--time", "1700000000", "--realtime"],
            &["--debug", "--verify", "--disable-jit"],
            &["--cmdline", "console=ttyS0", "--append", "quiet"],
        ] {
            assert!(
                build_config(vec![], Some(vec![1, 2, 3]), &boot_args(flags), true).is_ok(),
                "{flags:?}"
            );
        }
    }
}
