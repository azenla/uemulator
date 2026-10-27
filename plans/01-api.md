# Public API

Small surface; everything else is `pub(crate)`.

```rust
pub struct MachineConfig { /* Default-able builder */ }
impl MachineConfig {
    pub fn new() -> Self;
    pub fn kernel(self, bzimage: impl Into<Vec<u8>>) -> Self;
    pub fn initrd(self, data: impl Into<Vec<u8>>) -> Self;
    pub fn cmdline(self, s: impl Into<String>) -> Self;
    pub fn memory_mib(self, n: usize) -> Self;      // default 512
    pub fn jit(self, mode: JitMode) -> Self;        // Auto (default) | Off | On
    pub fn clock(self, mode: ClockMode) -> Self;    // Virtual (default) | Realtime
    pub fn disk(self, d: Disk) -> Self;                 // raw image, virtio-blk (optional)
    pub fn net(self, mac: [u8; 6]) -> Self;             // virtio-net; frames via EmulatedMachine::net() (optional)
    pub fn console(self, k: ConsoleKind) -> Self;       // Virtio (default, resizable hvc0) | Serial (ttyS0)
    pub fn reboot(self, on: bool) -> Self;              // true (default): guest reset restarts the machine in place; false: ExitReason::Reset
    pub fn log(self, f: impl Fn(&str) + Send + 'static) -> Self;
}

pub struct EmulatedMachine { .. }
impl EmulatedMachine {
    pub fn new(cfg: MachineConfig) -> Result<Self, Error>;   // validates + loads kernel, no execution
    pub fn console(&self) -> ConsoleHandle;                  // cloneable, Send + Sync
    pub fn start(&mut self) -> Result<(), Error>;            // spawns the vCPU thread, returns immediately
    pub fn wait(&mut self) -> ExitReason;                    // blocks until the guest exits
    pub fn stop(&self);                                      // callable from any thread via StopHandle
    pub fn stop_handle(&self) -> StopHandle;
}

pub enum ExitReason { Shutdown, Reset, TripleFault, Stopped, Internal(String) }

pub struct ConsoleHandle { .. }
impl ConsoleHandle {
    pub fn write(&self, bytes: &[u8]);                       // host → guest (serial RX)
    pub fn read(&self, buf: &mut [u8], timeout: Option<Duration>) -> usize; // guest → host (serial TX)
    pub fn on_output(&self, f: impl Fn(&[u8]) + Send + 'static); // push alternative to read
    pub fn resize(&self, cols: u16, rows: u16);              // delivered to the guest with ConsoleKind::Virtio
    pub fn size(&self) -> (u16, u16);
}
```

## Rules

- `new` does all fallible setup (parse bzImage, allocate RAM, probe JIT) so `start` failing is rare.
- Guest shutdown is a value (`ExitReason`), never a process exit. `hlt` with IF=0 (nothing can wake it; with `acpi=off` this is how Linux poweroff/halt ends, `cli; hlt` in `stop_this_cpu`) is `Shutdown`; a triple fault is `TripleFault`, so automated runs terminate. A reset request (0x64←0xFE, 0x92 bit 0, 0xCF9 RST_CPU) wins over a `hlt` right after it.
- Reset restarts the machine in the library by default (`reboot(true)`): the vCPU thread rebuilds the CPU and every device (PIC, PIT, CMOS, UART, PCI, virtio) from power-on state, reloads kernel, initrd and command line into freshly zeroed RAM, and runs again, so a Linux `panic=-1` loops forever until `stop()`. Kept across the restart: the `ConsoleHandle`/`NetHandle` (devices are rebuilt around the same queues), the stop flag, the disk backend (handed back from the old virtio-blk), the poll hook and the log. The JIT starts with an empty code cache (a fresh `Jit` per boot, so translations of the old RAM can never run). The RTC is re-read from the host clock (or `unix_time`), virtual time restarts at 0. `reboot(false)` returns `ExitReason::Reset` instead (CLI `--no-reboot`). A triple fault always ends the run (`TripleFault`), and under `verify_jit` a reset always ends the run with `Reset` (the twin's devices cannot be rebuilt in step). The UEFI hardware-virtualization path has no in-process reboot and returns `Reset`.
- Console is a plain byte stream backed by bounded queues; output with no reader drops oldest beyond a cap rather than blocking the guest.
- Stdio bridging (raw tty, Ctrl-C) lives in `crates/uemulator-cli`, not the library.
- No networking in the library. The CLI downloads with ureq.

## CLI (`crates/uemulator-cli`, binary `uemulator`)

`uemulator boot --kernel K [--initramfs I] [flags]` and `uemulator quick <debian|alpine|fedora> [--update] [flags]` (flags in README).

1. Cache dir: `$UEMULATOR_CACHE`, else `~/.uemulator/cache`; quick artifacts at `quick/<distro>/artifacts/<file>`; `$UEMULATOR_HOME` (default `~/.uemulator`) holds everything, with the cache in its `cache/` subdirectory and the per-distro disk at `quick/<distro>/disk.img`.
2. Sources: Debian trixie netboot (deb.debian.org), Alpine latest-stable `virt` netboot (dl-cdn.alpinelinux.org; boot.alpinelinux.org only hosts iPXE), Fedora 44 Server pxeboot. Downloads go to `<file>.part`, then `rename`. Skipped when cached and non-empty unless `--update`.
3. Build config, start machine, pump stdin → console and console → stdout until exit; print `ExitReason`.

Default cmdline (`platform::DEFAULT_CMDLINE`, console `virtio`): `console=hvc0 earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1`; `DEFAULT_SERIAL_CMDLINE` swaps in `console=ttyS0` (`--console serial`, and the UEFI hardware-virtualization path whose `Platform` has no virtio). The Alpine virt kernel has no `earlyprintk`, so `quick alpine` adds `earlycon=uart8250,io,0x3f8` to keep the early log.

Debug-profile interpreter is too slow for the decompressor; set `[profile.dev] opt-level = 3` in Cargo.toml and document `--release`.
