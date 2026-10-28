# uemulator

Small embeddable x86_64 system emulator in Rust, modeled on QEMU TCG. It boots an unmodified Linux bzImage (no BIOS) on a virtio console (`hvc0`; `--console serial` selects the 16550), using an interpreter everywhere and a host JIT (aarch64 or x86_64; Linux, macOS, Windows) when the `jit` feature is on and the host allows it.

```rust
use uemulator::{EmulatedMachine, MachineConfig};

let mut machine = EmulatedMachine::new(MachineConfig::new().kernel(bzimage).initrd(initrd))?;
let console = machine.console();
console.on_output(|bytes| { /* guest console output */ });
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

Flags (`boot` and `quick`): `--memory MiB` (64-3072, default 2048; `quick fedora` defaults to 3072 because its installer image is downloaded into RAM), `--append ARGS` (added to the kernel command line), `--net-user` (NAT to the host network: DHCP, DNS, ICMP echo, UDP and TCP; the guest is 10.0.2.15, the gateway 10.0.2.2 reaches the host's loopback), `--cmdline`, `--disk image[,ro]` (raw image as virtio-blk), `--net-loop` (virtio-net with a tiny 10.0.2.2 gateway answering ARP/ping), `--net-udp LOCAL,REMOTE` (frames over UDP), `--console virtio|serial` (default virtio: hvc0, the host terminal size is forwarded and follows resizes; serial is ttyS0 for kernels without a built-in `virtio_console`). `--disk`, `--console`, `--net-user`, `--net-loop` and `--net-udp` can each be given several times: disks appear as `vda`, `vdb`, ... in command-line order, each NIC flag adds one virtio-net (in command-line order across the three kinds, each with its own MAC), `--console serial --console virtio` gives both a `ttyS0` and an `hvc0` (serial at most once; the first console is the primary and the one the terminal is attached to), `--no-reboot` (a guest reset ends the run instead of rebooting the machine in place), `--disable-jit`, `--time`, `--timeout`, `--debug`, `--verify`.

Optional devices (all virtio over PCI, none present unless configured; each builder call adds one more device, up to 31 in all): `MachineConfig::disk(Disk)`, `MachineConfig::net(mac)` + `EmulatedMachine::nets()` (or `net()` for the first) for frame-level `NetHandle`s, `MachineConfig::console(ConsoleKind::Serial)` to opt out of the default virtio console (`EmulatedMachine::consoles()` returns one `ConsoleHandle` per console), `ConsoleHandle::resize(cols, rows)`. A guest reset reboots the machine in place (`MachineConfig::reboot(false)` / `--no-reboot` end the run instead) and `hlt` with interrupts off ends it as a power-off (`ExitReason::Shutdown`).

Design and status: `plans/` (start at `plans/README.md`). Agent notes: `CLAUDE.md`.
