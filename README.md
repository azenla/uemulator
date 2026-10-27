# uemulator

Small embeddable x86_64 system emulator in Rust, modeled on QEMU TCG. It boots an unmodified Linux bzImage (no BIOS) on a serial console, using an interpreter everywhere and a host JIT (aarch64 or x86_64; Linux, macOS, Windows) when the `jit` feature is on and the host allows it.

```rust
use uemulator::{EmulatedMachine, MachineConfig};

let mut machine = EmulatedMachine::new(MachineConfig::new().kernel(bzimage).initrd(initrd))?;
let console = machine.console();
console.on_output(|bytes| { /* guest serial output */ });
machine.start()?;
let reason = machine.wait();
```

Try it with the CLI (`crates/uemulator-cli`). `quick` downloads netboot artifacts once into `~/.uemulator/cache/quick/<distro>/artifacts/` (`--update` refetches), attaches a sparse 8 GiB disk at `~/.uemulator/quick/<distro>/disk.img` (`--disk-size GIB`, or `--disk` to use your own) and a user-mode NAT network, and boots the distro's installer (Debian installer, Alpine to its login prompt where you run `setup-alpine`, Fedora's Anaconda in text mode):

```
cargo run --release -- quick debian --memory 1024
cargo run --release -- quick alpine --disable-jit
cargo run --release -- boot --kernel vmlinuz --initramfs initrd --memory 8192
```

In a terminal the CLI puts the tty in raw mode; Ctrl-] stops the machine.

Flags (`boot` and `quick`): `--memory MiB` (64-3072, default 2048; `quick fedora` defaults to 3072 because its installer image is downloaded into RAM), `--append ARGS` (added to the kernel command line), `--net-user` (NAT to the host network: DHCP, DNS, ICMP echo, UDP and TCP; the guest is 10.0.2.15, the gateway 10.0.2.2 reaches the host's loopback), `--cmdline`, `--disk image[,ro]` (raw image as virtio-blk), `--net-loop` (virtio-net with a tiny 10.0.2.2 gateway answering ARP/ping), `--net-udp LOCAL,REMOTE` (frames over UDP), `--console serial|virtio` (hvc0; the host terminal size is forwarded and follows resizes), `--no-reboot` (a guest reset ends the run instead of rebooting the machine in place), `--disable-jit`, `--time`, `--timeout`, `--debug`, `--verify`.

Optional devices (all virtio over PCI, none present unless configured): `MachineConfig::disk(Disk)`, `MachineConfig::net(mac)` + `EmulatedMachine::net()` for a frame-level `NetHandle`, `MachineConfig::console(ConsoleKind::Virtio)` + `ConsoleHandle::resize(cols, rows)`.

Design and status: `plans/` (start at `plans/README.md`). Agent notes: `CLAUDE.md`.
