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

Try it with the CLI (`crates/uemulator-cli`). `quick` keeps named machines in `~/.uemulator/machines/<name>/`: a `machine.toml` with the distro to install and the machine's hardware, the distro's netboot kernel and initramfs in `artifacts/`, and the sparse disk images `--create-disk` makes. A machine boots the distro's installer (Debian installer, Alpine to its login prompt where you run `setup-alpine`, Fedora's Anaconda in text mode); `UEMULATOR_HOME` moves `~/.uemulator`.

```
# create a machine (downloads its kernel and initramfs; --boot starts it right away)
cargo run --release -- quick create test --install alpine --memory 2048 --create-disk 16384 \
    --console virtio --network virtio --network-link-user
# change its saved configuration
cargo run --release -- quick edit test --memory 1024
# boot it, optionally with overrides that last for this run only
cargo run --release -- quick boot test
cargo run --release -- quick boot test --memory 512 --disable-jit
# throw away the kernel and initramfs (disks and settings stay), optionally for another distro
cargo run --release -- quick reinstall test --install debian
# fetch the distro's current kernel and initramfs
cargo run --release -- quick update-artifacts test
# no named machine: spell everything out
cargo run --release -- advanced boot --kernel vmlinuz --initramfs initrd --memory 2048
cargo run --release -- advanced boot-firmware --firmware bios.bin
```

In a terminal the CLI puts the tty in raw mode; Ctrl-] stops the machine.

Options shared by `quick create|edit|boot` and `advanced boot|boot-firmware`: `--memory MiB` (64-3072, default 2048; a fedora machine defaults to 3072 because its installer image is downloaded into RAM), `--disk image[,ro]` (raw image as virtio-blk), `--network virtio` (a NIC; the link option after it picks the host side: `--network-link-user` is NAT to the host network, with DHCP, DNS, ICMP echo, UDP and TCP, the guest being 10.0.2.15 and the gateway 10.0.2.2 reaching the host's loopback; `--network-link-loop` is a tiny 10.0.2.2 gateway answering ARP/ping; `--network-link-udp LOCAL,REMOTE` sends frames over UDP; a NIC with no link has no host side), `--console virtio|serial` (default virtio: hvc0, the host terminal size is forwarded and follows resizes; serial is ttyS0 for kernels without a built-in `virtio_console`), `--disable-jit`. Everything but `--memory` and `--disable-jit` can be given several times: disks appear as `vda`, `vdb`, ... in command-line order, `--network` adds one virtio-net each with its own MAC, `--console serial --console virtio` gives both a `ttyS0` and an `hvc0` (serial at most once; the first console is the primary and the one the terminal is attached to). Kernel boots (not `boot-firmware`) also take `--cmdline`, `--append` and `--no-reboot` (a guest reset ends the run instead of rebooting the machine in place); `--disable-jit` and `--no-reboot` take `=false` to turn a saved setting off, and an empty `--cmdline ""` or `--append ""` clears a saved one. `quick boot` and both `advanced` commands also take the options of a single run: `--time`, `--timeout`, `--debug`, `--verify`, `--realtime`.

On a saved machine the device options add to what it has: `quick create` falls back to an 8 GiB disk, one NAT NIC and a virtio console when it is given none; `quick edit` and `quick boot` append. `--create-disk MIB` (create and edit) makes a sparse image in the machine's directory and attaches it after the existing disks, in command-line order with any `--disk`. `--remove-disk INDEX`, `--remove-network INDEX` and `--remove-console INDEX` (edit and boot) take devices away first, counting from 0 in the order `create` and `edit` print them and as the list was before the command; a removed disk is only detached, its image file stays where it is. So replacing a disk is `quick edit test --remove-disk 0 --create-disk 32768`. Old `~/.uemulator/quick/<distro>/disk.img` disks from the previous `quick <distro>` command can be attached to a machine with `--disk`.

Optional devices (all virtio over PCI, none present unless configured; each builder call adds one more device, up to 31 in all): `MachineConfig::disk(Disk)`, `MachineConfig::net(mac)` + `EmulatedMachine::nets()` (or `net()` for the first) for frame-level `NetHandle`s, `MachineConfig::console(ConsoleKind::Serial)` to opt out of the default virtio console (`EmulatedMachine::consoles()` returns one `ConsoleHandle` per console), `ConsoleHandle::resize(cols, rows)`. A guest reset reboots the machine in place (`MachineConfig::reboot(false)` / `--no-reboot` end the run instead) and `hlt` with interrupts off ends it as a power-off (`ExitReason::Shutdown`).

Design and status: `plans/` (start at `plans/README.md`). Agent notes: `CLAUDE.md`.
