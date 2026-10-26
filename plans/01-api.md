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
}
```

## Rules

- `new` does all fallible setup (parse bzImage, allocate RAM, probe JIT) so `start` failing is rare.
- Guest shutdown is a value (`ExitReason`), never a process exit. Triple fault and 0x64←0xFE / port 0xCF9 reset map to `Reset`/`TripleFault` so automated runs terminate.
- Console is a plain byte stream backed by bounded queues; output with no reader drops oldest beyond a cap rather than blocking the guest.
- Stdio bridging (raw tty, Ctrl-C) lives in `examples/linux.rs`, not the library.
- No networking in the library. The example shells out to `curl -fL` for downloads.

## Example (`examples/linux.rs`, target `boot-linux`)

`cargo run --release --example boot-linux [-- --no-initrd --no-jit --cmdline ...]`

1. Cache dir: `$UEMULATOR_CACHE`, else `~/.cache/uemulator`.
2. Fetch `https://deb.debian.org/debian/dists/stable/main/installer-amd64/current/images/netboot/debian-installer/amd64/{linux,initrd.gz}` with `curl -fL -o <name>.part`, then `rename` so partial downloads never get cached. Skip when the file exists and is non-empty.
3. Build config, start machine, pump stdin → console and console → stdout until exit; print `ExitReason`.

Default cmdline: `console=ttyS0 earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1`.

Debug-profile interpreter is too slow for the decompressor; set `[profile.dev] opt-level = 3` in Cargo.toml and document `--release`.
