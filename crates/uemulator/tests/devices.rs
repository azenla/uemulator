use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uemulator::{ClockMode, ConsoleHandle, ConsoleKind, Disk, EmulatedMachine, MachineConfig};

#[path = "../../uemulator-cli/src/netgw.rs"]
mod netgw;

/// Same cache layout as `boot.rs`; a test returns early (passes) when an artifact is missing.
const DEBIAN_ARTIFACTS: &str = "quick/debian/artifacts";
/// Prompt of the initrd's `/bin/sh` when it is started as `rdinit=/bin/sh`.
const SHELL_PROMPT: &str = "~ # ";
const BOOT_TIMEOUT_SECS: u64 = 120;
const COMMAND_TIMEOUT_SECS: u64 = 60;
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long the guest gets to notice a console resize (it arrives as a config-change interrupt
/// handled by the guest asynchronously).
const RESIZE_TIMEOUT: Duration = Duration::from_secs(20);

/// Gateway address of the user-mode network the host-side responder (`netgw`) answers for; the
/// guest takes 10.0.2.15/24 like QEMU's slirp default.
const GATEWAY_IP: &str = "10.0.2.2";
const GUEST_CIDR: &str = "10.0.2.15/24";
/// Locally administered (bit 1 of the first octet), unicast.
const TEST_MAC: [u8; 6] = [2, 0, 0, 0, 0, 1];
const GATEWAY_POLL: Duration = Duration::from_millis(5);

const SECTOR_SIZE: usize = 512;
const DISK_SIZE: usize = 1 << 20;
const DISK_MAGIC_TEXT: &str = "HELLO-UEMU";
const GUEST_WRITE: &str = "GUEST-WROTE";
/// The guest writes this sector (`dd seek=`), so the bytes land at `GUEST_WRITE_SECTOR * 512`.
const GUEST_WRITE_SECTOR: usize = 4;

fn cache_dir() -> Option<PathBuf> {
    std::env::var("UEMULATOR_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".uemulator/cache")))
        .ok()
}

fn cached(name: &str) -> Option<Vec<u8>> {
    std::fs::read(cache_dir()?.join(DEBIAN_ARTIFACTS).join(name))
        .ok()
        .filter(|d| !d.is_empty())
}

/// A booted guest with a root shell on the console and a way to run commands in it.
struct Guest {
    machine: EmulatedMachine,
    /// Everything the guest has written to the console so far.
    out: Arc<Mutex<Vec<u8>>>,
    console: ConsoleHandle,
    /// Number of commands run, to give each completion marker a unique name.
    seq: u32,
}

impl Guest {
    /// Boots the installer initrd straight into a shell. `extra_initrd` is appended to it: the
    /// kernel unpacks concatenated cpio archives in order, which is how test files get in.
    fn boot(cfg: MachineConfig, kind: ConsoleKind, extra_initrd: Option<Vec<u8>>) -> Option<Guest> {
        let kernel = cached("linux")?;
        let mut initrd = cached("initrd.gz")?;
        if let Some(e) = extra_initrd {
            initrd.extend_from_slice(&e);
        }
        let console_device = if kind == ConsoleKind::Virtio {
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
                "console={console_device} earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1 rdinit=/bin/sh"
            ));
        let mut machine = EmulatedMachine::new(cfg).unwrap();
        let console = machine.console();
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        console.on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
        machine.start().unwrap();
        let guest = Guest {
            machine,
            out,
            console,
            seq: 0,
        };
        assert!(
            guest.wait_for(SHELL_PROMPT, BOOT_TIMEOUT_SECS, 0),
            "no shell:\n{}",
            guest.text()
        );
        Some(guest)
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.out.lock().unwrap()).into_owned()
    }

    /// Looks for `needle` in the output from byte offset `from` on.
    fn wait_for(&self, needle: &str, secs: u64, from: usize) -> bool {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(secs) {
            if self.text()[from.min(self.text().len())..].contains(needle) {
                return true;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        false
    }

    /// Runs `cmd` in the guest shell and returns its output. Completion is detected by a marker
    /// echoed after the command. It is typed as `DO''NE<n>` so the shell's echo of the typed line
    /// (the tty echoes input) never contains the literal `DONE<n>`; only the executed `echo` does.
    fn run(&mut self, cmd: &str) -> String {
        self.seq += 1;
        let marker = format!("DONE{}", self.seq);
        let from = self.text().len();
        let line = format!("{cmd}; echo DO''NE{}\n", self.seq);
        self.console.write(line.as_bytes());
        assert!(
            self.wait_for(&marker, COMMAND_TIMEOUT_SECS, from),
            "command timed out: {cmd}\n{}",
            self.text()
        );
        let text = self.text();
        let body = &text[from..];
        // Skip the echoed command line; stop before the marker.
        let start = body.find('\n').map_or(0, |newline| newline + 1);
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

/// Answers the guest's ARP/ICMP for the gateway from a thread that lives until the test process
/// exits.
fn spawn_gateway(net: uemulator::NetHandle) {
    std::thread::spawn(move || {
        loop {
            if let Some(frame) = net.recv_frame(GATEWAY_POLL)
                && let Some(reply) = netgw::handle(&frame)
            {
                net.send_frame(&reply);
            }
        }
    });
}

/// A cpio "newc" archive holding one regular file, in the format the kernel's initramfs unpacker
/// reads: a "070701" magic and thirteen 8-digit hex header fields, then the NUL-terminated name
/// and the data, each padded to a multiple of 4 bytes, ended by a `TRAILER!!!` entry.
fn cpio_newc(name: &str, data: &[u8]) -> Vec<u8> {
    const NEWC_MAGIC: &str = "070701";
    const REGULAR_FILE_0644: u32 = 0o100644;
    const ALIGNMENT: usize = 4;
    let mut out = Vec::new();
    let mut entry = |name: &str, mode: u32, data: &[u8]| {
        let name_len = name.len() + 1;
        // inode, mode, uid, gid, nlink, mtime, filesize, devmajor, devminor, rdevmajor,
        // rdevminor, namesize, check
        write!(
            out,
            "{NEWC_MAGIC}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
            1, mode, 0, 0, 1, 0, data.len(), 0, 0, 0, 0, name_len, 0
        )
        .unwrap();
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        while out.len() % ALIGNMENT != 0 {
            out.push(0);
        }
        out.extend_from_slice(data);
        while out.len() % ALIGNMENT != 0 {
            out.push(0);
        }
    };
    entry(name, REGULAR_FILE_0644, data);
    entry("TRAILER!!!", 0, &[]);
    out
}

/// Compresses through the host `gzip` (the initrd unpacker wants a gzip stream). The input is
/// written from a separate thread so a full stdout pipe cannot deadlock the write.
fn gzip(data: &[u8]) -> Vec<u8> {
    let mut child = Command::new("gzip")
        .arg("-c")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("gzip");
    let mut stdin = child.stdin.take().unwrap();
    let data = data.to_vec();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&data).unwrap();
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    output.stdout
}

/// virtio-net end to end: the guest loads the driver, brings the link up and gets ICMP echo
/// replies from the host-side gateway, so frames flow both ways through `NetHandle`.
#[test]
fn net_ping_gateway() {
    let cfg = MachineConfig::new().net(TEST_MAC);
    let Some(mut guest) = Guest::boot(cfg, ConsoleKind::Serial, None) else {
        return;
    };
    spawn_gateway(guest.machine.net().unwrap());
    // virtio_net is a module in the Debian kernel; modprobe needs /proc and /sys.
    guest.run("mount -t proc p /proc; mount -t sysfs s /sys; modprobe virtio_net");
    guest.run(&format!(
        "ip link set eth0 up; ip addr add {GUEST_CIDR} dev eth0"
    ));
    let output = guest.run(&format!("ping -c 3 -W 3 {GATEWAY_IP}"));
    assert!(output.contains("3 packets received"), "{output}");
}

/// `ConsoleHandle::resize` must reach the guest's tty on the virtio console, and the default
/// size is 80 columns by 25 rows (`stty size` prints rows then columns).
#[test]
fn console_resize_reaches_guest() {
    let cfg = MachineConfig::new();
    let Some(mut guest) = Guest::boot(cfg, ConsoleKind::Virtio, None) else {
        return;
    };
    assert_eq!(guest.run("stty size"), "25 80");
    guest.console.resize(132, 43);
    let start = Instant::now();
    loop {
        let size = guest.run("stty size");
        if size == "43 132" {
            break;
        }
        assert!(start.elapsed() < RESIZE_TIMEOUT, "size stuck at {size}");
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// virtio-blk end to end: capacity, a read of sector 0, and a guest write that must reach the
/// backing file once the machine is dropped. `virtio_blk` is a module, delivered in an extra
/// initrd segment.
#[test]
fn disk_read_write() {
    let Some(module) = cached("virtio_blk.ko.xz") else {
        return;
    };
    let mut image = vec![0u8; DISK_SIZE];
    // The image starts with the text plus a newline.
    let magic = format!("{DISK_MAGIC_TEXT}\n");
    image[..magic.len()].copy_from_slice(magic.as_bytes());
    let path = std::env::temp_dir().join(format!("uemu-disk-{}.img", std::process::id()));
    std::fs::write(&path, &image).unwrap();
    let cfg = MachineConfig::new().disk(Disk::open(&path, false).unwrap());
    let extra = gzip(&cpio_newc("vblk.ko.xz", &module));
    let Some(mut guest) = Guest::boot(cfg, ConsoleKind::Serial, Some(extra)) else {
        return;
    };
    guest.run("mount -t proc p /proc; mount -t sysfs s /sys; mount -t devtmpfs d /dev; insmod /vblk.ko.xz");
    // The block layer reports 512-byte sectors.
    assert_eq!(
        guest.run("cat /sys/block/vda/size"),
        (DISK_SIZE / SECTOR_SIZE).to_string()
    );
    assert_eq!(
        guest.run(&format!(
            "dd if=/dev/vda bs=512 count=1 2>/dev/null | head -c {}",
            DISK_MAGIC_TEXT.len()
        )),
        DISK_MAGIC_TEXT
    );
    guest.run(&format!(
        "printf '{GUEST_WRITE}' | dd of=/dev/vda bs=512 seek={GUEST_WRITE_SECTOR} conv=notrunc 2>/dev/null; sync"
    ));
    drop(guest);
    let after = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let written = GUEST_WRITE_SECTOR * SECTOR_SIZE;
    assert_eq!(
        &after[written..written + GUEST_WRITE.len()],
        GUEST_WRITE.as_bytes()
    );
    assert_eq!(&after[..DISK_MAGIC_TEXT.len()], DISK_MAGIC_TEXT.as_bytes());
}
