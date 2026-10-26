# uemulator

Small embeddable x86_64 system emulator in Rust, modeled on QEMU TCG. It boots an unmodified Linux bzImage (no BIOS) on a serial console, using an interpreter everywhere and an aarch64 JIT when the host allows it.

```rust
use uemulator::{EmulatedMachine, MachineConfig};

let mut machine = EmulatedMachine::new(MachineConfig::new().kernel(bzimage).initrd(initrd))?;
let console = machine.console();
console.on_output(|bytes| { /* guest serial output */ });
machine.start()?;
let reason = machine.wait();
```

Try it with the Debian netboot kernel (downloaded once into `~/.cache/uemulator`):

```
cargo run --release --example boot-linux            # kernel + installer initrd, JIT on
cargo run --release --example boot-linux -- --no-initrd --no-jit
```

In a terminal the example puts the tty in raw mode; Ctrl-] stops the machine.

Example flags: `--mem MiB` (64-3072, default 512), `--disk image[,ro]` (raw image as virtio-blk), `--net-loop` (virtio-net with a tiny 10.0.2.2 gateway answering ARP/ping), `--net-udp LOCAL,REMOTE` (frames over UDP), `--console virtio` (hvc0 console; the host terminal size is forwarded to the guest and follows resizes), `--no-jit`, `--no-initrd`, `--cmdline`, `--time`, `--timeout`, `--debug`, `--verify`.

Optional devices (all virtio over PCI, none present unless configured): `MachineConfig::disk(Disk)`, `MachineConfig::net(mac)` + `EmulatedMachine::net()` for a frame-level `NetHandle`, `MachineConfig::console(ConsoleKind::Virtio)` + `ConsoleHandle::resize(cols, rows)`.

Design and status: `plans/` (start at `plans/README.md`). Agent notes: `CLAUDE.md`.
