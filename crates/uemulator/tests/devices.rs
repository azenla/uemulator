use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uemulator::{ClockMode, ConsoleHandle, ConsoleKind, Disk, EmulatedMachine, MachineConfig};

#[path = "../../uemulator-cli/src/netgw.rs"]
mod netgw;

fn cache_dir() -> Option<PathBuf> {
    std::env::var("UEMULATOR_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".uemulator/cache")))
        .ok()
}

fn cached(name: &str) -> Option<Vec<u8>> {
    std::fs::read(cache_dir()?.join("quick/debian/artifacts").join(name))
        .ok()
        .filter(|d| !d.is_empty())
}

struct Guest {
    machine: EmulatedMachine,
    out: Arc<Mutex<Vec<u8>>>,
    console: ConsoleHandle,
    seq: u32,
}

impl Guest {
    fn boot(cfg: MachineConfig, kind: ConsoleKind, extra_initrd: Option<Vec<u8>>) -> Option<Guest> {
        let kernel = cached("linux")?;
        let mut initrd = cached("initrd.gz")?;
        if let Some(e) = extra_initrd {
            initrd.extend_from_slice(&e);
        }
        let con = if kind == ConsoleKind::Virtio {
            "hvc0"
        } else {
            "ttyS0"
        };
        let cfg = cfg
            .kernel(kernel)
            .initrd(initrd)
            .console(kind)
            .clock(ClockMode::Realtime)
            .cmdline(format!(
                "console={con} earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1 rdinit=/bin/sh"
            ));
        let mut machine = EmulatedMachine::new(cfg).unwrap();
        let console = machine.console();
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        console.on_output(move |b| sink.lock().unwrap().extend_from_slice(b));
        machine.start().unwrap();
        let g = Guest {
            machine,
            out,
            console,
            seq: 0,
        };
        assert!(g.wait_for("~ # ", 120, 0), "no shell:\n{}", g.text());
        Some(g)
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.out.lock().unwrap()).into_owned()
    }

    fn wait_for(&self, needle: &str, secs: u64, from: usize) -> bool {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(secs) {
            if self.text()[from.min(self.text().len())..].contains(needle) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }

    fn run(&mut self, cmd: &str) -> String {
        self.seq += 1;
        let marker = format!("DONE{}", self.seq);
        let from = self.text().len();
        let line = format!("{cmd}; echo DO''NE{}\n", self.seq);
        self.console.write(line.as_bytes());
        assert!(
            self.wait_for(&marker, 60, from),
            "command timed out: {cmd}\n{}",
            self.text()
        );
        let t = self.text();
        let body = &t[from..];
        let start = body.find('\n').map_or(0, |i| i + 1);
        let end = body.rfind(&marker).unwrap_or(body.len());
        body[start.min(end)..end].trim().to_string()
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        self.machine.stop();
        self.machine.wait();
    }
}

fn spawn_gateway(net: uemulator::NetHandle) {
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

fn cpio_newc(name: &str, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut entry =
        |name: &str, mode: u32, data: &[u8]| {
            let nlen = name.len() + 1;
            write!(
            out,
            "070701{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
            1, mode, 0, 0, 1, 0, data.len(), 0, 0, 0, 0, nlen, 0
        )
        .unwrap();
            out.extend_from_slice(name.as_bytes());
            out.push(0);
            while out.len() % 4 != 0 {
                out.push(0);
            }
            out.extend_from_slice(data);
            while out.len() % 4 != 0 {
                out.push(0);
            }
        };
    entry(name, 0o100644, data);
    entry("TRAILER!!!", 0, &[]);
    out
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut child = Command::new("gzip")
        .arg("-c")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("gzip");
    let mut stdin = child.stdin.take().unwrap();
    let data = data.to_vec();
    let t = std::thread::spawn(move || {
        stdin.write_all(&data).unwrap();
    });
    let out = child.wait_with_output().unwrap();
    t.join().unwrap();
    out.stdout
}

#[test]
fn net_ping_gateway() {
    let cfg = MachineConfig::new().net([2, 0, 0, 0, 0, 1]);
    let Some(mut g) = Guest::boot(cfg, ConsoleKind::Serial, None) else {
        return;
    };
    spawn_gateway(g.machine.net().unwrap());
    g.run("mount -t proc p /proc; mount -t sysfs s /sys; modprobe virtio_net");
    g.run("ip link set eth0 up; ip addr add 10.0.2.15/24 dev eth0");
    let out = g.run("ping -c 3 -W 3 10.0.2.2");
    assert!(out.contains("3 packets received"), "{out}");
}

#[test]
fn console_resize_reaches_guest() {
    let cfg = MachineConfig::new();
    let Some(mut g) = Guest::boot(cfg, ConsoleKind::Virtio, None) else {
        return;
    };
    assert_eq!(g.run("stty size"), "25 80");
    g.console.resize(132, 43);
    let start = Instant::now();
    loop {
        let s = g.run("stty size");
        if s == "43 132" {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "size stuck at {s}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
fn disk_read_write() {
    let Some(ko) = cached("virtio_blk.ko.xz") else {
        return;
    };
    let mut img = vec![0u8; 1 << 20];
    img[..11].copy_from_slice(b"HELLO-UEMU\n");
    let path = std::env::temp_dir().join(format!("uemu-disk-{}.img", std::process::id()));
    std::fs::write(&path, &img).unwrap();
    let cfg = MachineConfig::new().disk(Disk::open(&path, false).unwrap());
    let extra = gzip(&cpio_newc("vblk.ko.xz", &ko));
    let Some(mut g) = Guest::boot(cfg, ConsoleKind::Serial, Some(extra)) else {
        return;
    };
    g.run("mount -t proc p /proc; mount -t sysfs s /sys; mount -t devtmpfs d /dev; insmod /vblk.ko.xz");
    assert_eq!(g.run("cat /sys/block/vda/size"), "2048");
    assert_eq!(
        g.run("dd if=/dev/vda bs=512 count=1 2>/dev/null | head -c 10"),
        "HELLO-UEMU"
    );
    g.run("printf 'GUEST-WROTE' | dd of=/dev/vda bs=512 seek=4 conv=notrunc 2>/dev/null; sync");
    drop(g);
    let after = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(&after[2048..2059], b"GUEST-WROTE");
    assert_eq!(&after[..10], b"HELLO-UEMU");
}
