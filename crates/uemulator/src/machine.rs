#[cfg(freestanding)]
use crate::prelude::*;
use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(freestanding))]
use std::thread::JoinHandle;

use crate::boot::{self, BootState};
use crate::console::ConsoleHandle;
use crate::cpu::Cpu;
use crate::dev::{DevConfig, MAX_VIRTIO_FUNCTIONS};
#[cfg(not(freestanding))]
use crate::disk::Disk;
use crate::netdev::NetHandle;
use crate::platform::{DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE};
use crate::sync::Arc;

#[derive(Debug)]
pub enum Error {
    /// The bzImage is not a Linux x86 boot-protocol image this loader can start.
    BadKernel(String),
    /// A setting is out of range or the guest does not fit in the configured RAM.
    BadConfig(String),
    /// `JitMode::On` was requested but the host cannot run generated code.
    JitUnavailable,
    AlreadyStarted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadKernel(m) => write!(f, "bad kernel image: {m}"),
            Error::BadConfig(m) => write!(f, "bad configuration: {m}"),
            Error::JitUnavailable => write!(f, "JIT requested but unavailable on this host"),
            Error::AlreadyStarted => write!(f, "machine already started"),
        }
    }
}

impl core::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JitMode {
    /// Use the JIT when the host allows it, otherwise run the interpreter.
    #[default]
    Auto,
    Off,
    /// Fail `EmulatedMachine::new` with `JitUnavailable` instead of falling back.
    On,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClockMode {
    /// Guest time is the instruction count (one nanosecond each) plus the time skipped while
    /// halted: with a timer pending the clock jumps straight to its deadline, with none it stands
    /// still while the host naps 1 ms per idle poll. Runs are deterministic.
    #[default]
    Virtual,
    /// While halted with a timer pending the host really sleeps (up to 2 ms per wait) and guest
    /// time advances by exactly what was slept, so a 20 ms guest timeout takes at least 20 ms of
    /// wall time. Running code is not paced: guest time still counts instructions, so this is not
    /// the `max(virtual, wall)` clock `plans/07-devices.md` describes and only approximates wall
    /// time. Ignored on UEFI, where the clock stays virtual, and under `verify_jit`.
    Realtime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    /// The guest executed `hlt` with interrupts disabled (IF=0), which nothing can wake: with
    /// `acpi=off` that is how Linux's poweroff and halt end (`cli; hlt` in a loop). There is no
    /// ACPI or power-off device, so this is the only power-off signal. A reset request made just
    /// before the `hlt` wins and is reported as `Reset`.
    Shutdown,
    /// The guest asked for a reset through the keyboard controller (port 0x64), fast A20 (0x92)
    /// or reset control (0xcf9) port; see `dev/mod.rs`. Only produced with
    /// `MachineConfig::reboot(false)` or under `verify_jit`: by default the machine restarts
    /// itself instead.
    Reset,
    TripleFault,
    /// `stop()` / `StopHandle::stop()`, or `wait()` with nothing running.
    Stopped,
    /// An emulator-side failure, not a guest behavior.
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsoleKind {
    /// 16550 UART at COM1 (`ttyS0`). There is one UART, so this may be configured once.
    Serial,
    /// virtio-console (`hvc0`); the only kind that delivers `ConsoleHandle::resize` to the guest.
    /// The default, and it may be configured several times (`hvc0`, `hvc1`, ...). The UART is
    /// present either way and carries early boot text (`earlyprintk`), but this needs a kernel
    /// whose `virtio_console` driver is built in (an initrd that only loads it as a module has
    /// no console until then); pick `Serial` for such kernels.
    #[default]
    Virtio,
}

pub const MIN_MEMORY_MIB: usize = 64;
/// RAM must stay below the 0xc000_0000 MMIO window (3 GiB) where the virtio devices' BARs live.
pub const MAX_MEMORY_MIB: usize = 3072;
const DEFAULT_MEMORY_MIB: usize = 512;
const BYTES_PER_MIB_SHIFT: usize = 20;

pub(crate) type LogFn = Arc<dyn Fn(&str) + Send + Sync>;
/// The poll hook as the CPU sees it: the console is already bound, the flag means "guest is
/// halted with nothing scheduled, the host may wait".
pub(crate) type PollFn = Arc<dyn Fn(bool) + Send + Sync>;
/// The embedder-facing poll hook of `MachineConfig::poll`, which receives the console itself.
type HostPollFn = Arc<dyn Fn(&ConsoleHandle, bool) + Send + Sync>;

pub struct MachineConfig {
    kernel: Vec<u8>,
    firmware: Option<Vec<u8>>,
    initrd: Option<Vec<u8>>,
    cmdline: Option<String>,
    cmdline_extra: Vec<String>,
    memory_mib: usize,
    #[cfg(not(freestanding))]
    disks: Vec<Disk>,
    net_macs: Vec<[u8; 6]>,
    consoles: Vec<ConsoleKind>,
    jit: JitMode,
    clock: ClockMode,
    log: Option<LogFn>,
    poll: Option<HostPollFn>,
    debug: bool,
    verify: bool,
    unix_time: Option<u64>,
    random_seed: Option<u64>,
    reboot: bool,
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            kernel: Vec::new(),
            firmware: None,
            initrd: None,
            cmdline: None,
            cmdline_extra: Vec::new(),
            memory_mib: DEFAULT_MEMORY_MIB,
            #[cfg(not(freestanding))]
            disks: Vec::new(),
            net_macs: Vec::new(),
            consoles: Vec::new(),
            jit: JitMode::Auto,
            clock: ClockMode::Virtual,
            log: None,
            poll: None,
            debug: false,
            verify: false,
            unix_time: None,
            random_seed: None,
            reboot: true,
        }
    }
}

impl MachineConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, bzimage: impl Into<Vec<u8>>) -> Self {
        self.kernel = bzimage.into();
        self
    }

    /// Boots a firmware image (a BIOS, or any real-mode program) instead of a Linux kernel: the
    /// image is placed so that it ends at 1 MiB and the CPU starts in real mode at F000:FFF0.
    /// No BIOS services exist beyond what the image brings, and a guest reset ends the run.
    pub fn firmware(mut self, image: impl Into<Vec<u8>>) -> Self {
        self.firmware = Some(image.into());
        self
    }

    pub fn initrd(mut self, data: impl Into<Vec<u8>>) -> Self {
        self.initrd = Some(data.into());
        self
    }

    pub fn cmdline(mut self, s: impl Into<String>) -> Self {
        self.cmdline = Some(s.into());
        self
    }

    /// Appended to the command line, whether it is the default or set by `cmdline`.
    pub fn cmdline_append(mut self, s: impl Into<String>) -> Self {
        self.cmdline_extra.push(s.into());
        self
    }

    /// Adds a virtio-blk disk. Call it again for more: the guest sees them in call order
    /// (`vda`, `vdb`, ...).
    #[cfg(not(freestanding))]
    pub fn disk(mut self, disk: Disk) -> Self {
        self.disks.push(disk);
        self
    }

    /// Adds a virtio-net device with this MAC; frames flow through `EmulatedMachine::nets()`
    /// (`net()` for the first). Call it again for more NICs, each with its own MAC: the n-th
    /// call is the n-th handle.
    pub fn net(mut self, mac: [u8; 6]) -> Self {
        self.net_macs.push(mac);
        self
    }

    /// Adds a guest console. The first one configured is the guest's primary console (the
    /// default command line's `console=`). `Serial` may be added once, `Virtio` as often as
    /// wanted; with none added the machine gets one virtio console. Each console has its own
    /// `ConsoleHandle`, in call order (`EmulatedMachine::consoles()`).
    pub fn console(mut self, kind: ConsoleKind) -> Self {
        self.consoles.push(kind);
        self
    }

    pub fn memory_mib(mut self, n: usize) -> Self {
        self.memory_mib = n;
        self
    }

    pub fn jit(mut self, mode: JitMode) -> Self {
        self.jit = mode;
        self
    }

    pub fn clock(mut self, mode: ClockMode) -> Self {
        self.clock = mode;
        self
    }

    pub fn debug(mut self, on: bool) -> Self {
        self.debug = on;
        self
    }

    /// Runs the JIT in lockstep with a second, interpreter-only CPU and stops at the first block
    /// where their state diverges. Very slow; for JIT development. A guest reset ends the run
    /// with `ExitReason::Reset` whatever `reboot` says. The second CPU has the same RTC
    /// boot time and the same virtio functions in the same slots, but not their data: disk
    /// contents and host console or network input are seen by the main CPU only, so a guest that
    /// reads them will report a divergence. Verification always runs on virtual time, whatever
    /// `clock` says. Without a JIT to check (`JitMode::Off`, or a host that has none) the machine
    /// just interprets.
    #[doc(hidden)]
    pub fn verify_jit(mut self, on: bool) -> Self {
        self.verify = on;
        self
    }

    /// Whether a guest reset request (keyboard controller, port 0x92, port 0xCF9) restarts the
    /// machine in place (the default) or ends the run with `ExitReason::Reset`.
    ///
    /// A restart looks like a cold boot to the guest: the kernel, initrd and command line are
    /// loaded again into freshly zeroed RAM, the CPU and every device start from power-on state,
    /// virtual time starts at 0 and the JIT starts with an empty code cache. What survives is
    /// what the embedder holds or what a real machine keeps: the `ConsoleHandle` and `NetHandle`
    /// (with whatever is queued in them), the stop flag, the disk image and its contents, and the
    /// RTC (re-read from the host clock, or re-applied from `unix_time`). A guest that panics
    /// with `panic=-1` therefore reboots forever; end such a run with `stop()`.
    ///
    /// A triple fault is never restarted: it ends the run with `ExitReason::TripleFault` so
    /// diagnostics and exit codes keep working. Under `verify_jit` a reset always ends the run
    /// with `Reset`, because the twin interpreter's devices cannot be rebuilt in step.
    pub fn reboot(mut self, on: bool) -> Self {
        self.reboot = on;
        self
    }

    /// Fixes the RTC's boot time. Guest behavior must not depend on it, so a failure that appears
    /// only at some wall-clock time is an emulator bug.
    pub fn unix_time(mut self, secs: u64) -> Self {
        self.unix_time = Some(secs);
        self
    }

    /// Seeds the generator behind the guest's `rdrand` and `rdseed`. The library has no entropy
    /// source of its own (it is `no_std` on some targets), so without this the sequence is fixed
    /// and every boot of the same image reads the same "random" numbers: fine for tests and
    /// reproducible runs, wrong for a guest that generates keys. An embedder that cares passes
    /// bytes from the host's CSPRNG. The generator is splitmix64, not a cryptographic one, and
    /// the guest kernel mixes it with its other entropy sources instead of trusting it alone.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    pub fn log(mut self, f: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.log = Some(Arc::new(f));
        self
    }
}

impl MachineConfig {
    /// Called from the vCPU loop. The flag is true while the guest is halted with nothing to
    /// do, when the host may wait. Single-threaded hosts use it to pump their console.
    pub fn poll(mut self, f: impl Fn(&ConsoleHandle, bool) + Send + Sync + 'static) -> Self {
        self.poll = Some(Arc::new(f));
        self
    }
}

/// Cloneable, thread-safe way to end a running machine. The vCPU polls the flag at its event
/// ticks and while idle, so `wait()` returns `Stopped` shortly after, not instantly.
#[derive(Clone)]
pub struct StopHandle(Arc<AtomicBool>);

impl StopHandle {
    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Everything the vCPU thread owns once `start` moves it off the caller's thread.
struct Vcpu {
    ram: Vec<u8>,
    boot: BootState,
    /// What the host poll hook receives: the first configured console.
    primary_console: ConsoleHandle,
    /// The console behind the 16550: the `Serial` console if one is configured, else (so early
    /// boot text reaches the host) the first console, shared with its virtio device.
    uart_console: ConsoleHandle,
    stop: Arc<AtomicBool>,
    log: Option<LogFn>,
    poll: Option<HostPollFn>,
    debug: bool,
    jit: JitMode,
    clock: ClockMode,
    verify: bool,
    unix_time: Option<u64>,
    random_seed: Option<u64>,
    /// Pristine copy of RAM for the lockstep interpreter CPU; only set when `verify` is on.
    boot_ram: Option<Vec<u8>>,
    devcfg: DevConfig,
    /// What a restart needs; `None` when a reset should end the run.
    reboot: Option<RebootImage>,
}

/// Everything `boot::load` needs to lay the guest out again, plus the parts of the device set
/// that cannot be rebuilt from the old `Devices` (the disk is recovered from them instead).
struct RebootImage {
    kernel: Vec<u8>,
    initrd: Option<Vec<u8>>,
    /// The final command line, defaults and `cmdline_append` already applied.
    cmdline: String,
    /// The embedder's handle shares its queues with the device, so the new device must be built
    /// around this very handle.
    nets: Vec<([u8; 6], NetHandle)>,
    virtio_consoles: Vec<ConsoleHandle>,
}

/// Wall-clock seconds, or 0 where there is no clock (UEFI keeps its own, see `unix_time`).
fn host_unix_time() -> u64 {
    #[cfg(not(freestanding))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0)
    }
    #[cfg(freestanding)]
    {
        0
    }
}

/// Stands in for a disk on the verify twin: same capacity and read-only flag as the real one so
/// the guest sees the same device, but no data behind it. Reads return zeros, writes vanish.
#[cfg(not(freestanding))]
struct ShadowDisk {
    len: u64,
    read_only: bool,
}

#[cfg(not(freestanding))]
impl crate::disk::BlockBackend for ShadowDisk {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, _offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        buf.fill(0);
        Ok(())
    }
    fn write_at(&mut self, _offset: u64, _data: &[u8]) -> std::io::Result<()> {
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
}

/// Devices for the verify twin that look the same to the guest as `main`: same virtio functions
/// in the same PCI slots. The twin cannot share the real backends (a disk or a host frame
/// queue has one owner), so their data is not mirrored: the first guest read of disk contents
/// is a legitimate divergence of this check, as is host console input.
///
/// Returns the twin's UART console too: consoles that share a handle with the UART on `main`
/// share one on the twin as well, because that sharing decides whether the UART takes host input.
fn mirrored_devices(main: &DevConfig, main_uart: &ConsoleHandle) -> (DevConfig, ConsoleHandle) {
    let twin_uart = ConsoleHandle::default();
    let devices = DevConfig {
        #[cfg(not(freestanding))]
        disks: main
            .disks
            .iter()
            .map(|disk| {
                Disk::from_backend(ShadowDisk {
                    len: disk.backend.len(),
                    read_only: disk.backend.read_only(),
                })
            })
            .collect(),
        nets: main
            .nets
            .iter()
            .map(|(mac, _)| (*mac, NetHandle::default()))
            .collect(),
        virtio_consoles: main
            .virtio_consoles
            .iter()
            .map(|console| {
                if console.is_same_console(main_uart) {
                    twin_uart.clone()
                } else {
                    ConsoleHandle::default()
                }
            })
            .collect(),
    };
    (devices, twin_uart)
}

/// The interpreter-only oracle for `Cpu::run_verify`. Anything a guest can observe must be
/// identical to the JIT-enabled CPU, or the first guest read of it shows up as a bogus JIT
/// divergence: the RTC boot time (the twin would otherwise read the wall clock) and the
/// device set. Time is always virtual here (`realtime` stays off) because only instruction
/// counts, which both CPUs share, keep the two clocks equal.
fn interpreter_twin(
    ram: Vec<u8>,
    boot: BootState,
    stop: Arc<AtomicBool>,
    boot_unix: u64,
    devices: DevConfig,
    uart_console: ConsoleHandle,
) -> Cpu {
    let mut twin = Cpu::new(ram, boot, uart_console, stop, None, devices);
    twin.dev.set_boot_time(boot_unix);
    twin.realtime = false;
    twin
}

/// What a finished run hands to the next one.
struct Finished {
    ram_bytes: usize,
    #[cfg(not(freestanding))]
    disks: Vec<Disk>,
}

impl Vcpu {
    /// Boots, and boots again after every guest reset while `reboot` is set.
    fn run(mut self) -> ExitReason {
        loop {
            let (reason, finished) = self.run_once();
            let (ExitReason::Reset, Some(image)) = (&reason, &self.reboot) else {
                return reason;
            };
            if let Some(log) = &self.log {
                log("guest reset: rebooting");
            }
            // Fresh zeroed RAM, not a `fill(0)` of the old one: `boot::load` relies on zeroes,
            // and filling would commit every page the host had lazily left untouched.
            let mut ram = vec![0u8; finished.ram_bytes];
            match boot::load(
                &mut ram,
                &image.kernel,
                image.initrd.as_deref(),
                &image.cmdline,
            ) {
                Ok(boot) => self.boot = boot,
                Err(e) => return ExitReason::Internal(format!("reboot: {e}")),
            }
            self.ram = ram;
            self.devcfg = DevConfig {
                #[cfg(not(freestanding))]
                disks: finished.disks,
                nets: image.nets.clone(),
                virtio_consoles: image.virtio_consoles.clone(),
            };
        }
    }

    /// One power-on to the next exit. Everything the CPU owns is built here, so a reboot starts
    /// from a CPU with no leftover state and from a JIT with an empty code cache: a reused cache
    /// would never see stores to the new CPU's empty `page_has_code` map and could run
    /// translations of the previous run's RAM.
    fn run_once(&mut self) -> (ExitReason, Finished) {
        let console = self.primary_console.clone();
        // Under verify both CPUs must start from one boot time even when the caller gave none;
        // two separate wall-clock reads could straddle a second boundary.
        let verify_twin = self.verify.then(|| {
            let (devices, uart_console) = mirrored_devices(&self.devcfg, &self.uart_console);
            (
                devices,
                uart_console,
                self.unix_time.unwrap_or_else(host_unix_time),
            )
        });
        let ram = core::mem::take(&mut self.ram);
        let ram_bytes = ram.len();
        let mut cpu = Cpu::new(
            ram,
            self.boot,
            self.uart_console.clone(),
            self.stop.clone(),
            self.log.clone(),
            core::mem::take(&mut self.devcfg),
        );
        let boot_unix = verify_twin
            .as_ref()
            .map(|(_, _, boot_unix)| *boot_unix)
            .or(self.unix_time);
        if let Some(boot_unix) = boot_unix {
            cpu.dev.set_boot_time(boot_unix);
        }
        if let Some(seed) = self.random_seed {
            cpu.rng_state = seed;
        }
        // Cloned, not taken: a reboot builds a new CPU that needs the hook again.
        cpu.poll = self.poll.clone().map(|host_poll| {
            let poll: PollFn = Arc::new(move |idle| host_poll(&console, idle));
            poll
        });
        if self.jit != JitMode::Off {
            // Lockstep compares state after every block, so blocks must not chain into each other.
            cpu.jit = crate::jit::Jit::new(verify_twin.is_none()).map(Box::new);
        }
        if self.debug {
            cpu.trace = Some(Default::default());
        }
        // Lockstep needs both CPUs to idle identically, and the twin skips to the next timer at
        // once, so verification always runs on virtual time (a realtime main CPU sleeps in 2 ms
        // slices and reports a bogus pre-step divergence).
        let lockstep = verify_twin.is_some() && cpu.jit.is_some();
        cpu.realtime = self.clock == ClockMode::Realtime && !lockstep;
        // Lockstep needs generated code to compare; without any (`JitMode::Off`, a host without
        // a backend) `step_block` has nothing to run, so just interpret.
        let reason = match verify_twin {
            Some((devices, uart_console, boot_unix)) if lockstep => {
                let mut twin = interpreter_twin(
                    self.boot_ram.take().unwrap(),
                    self.boot,
                    cpu.stop.clone(),
                    boot_unix,
                    devices,
                    uart_console,
                );
                twin.rng_state = cpu.rng_state;
                cpu.run_verify(&mut twin)
            }
            _ => cpu.run(),
        };
        if let (Some(l), true) = (
            &self.log,
            self.debug || matches!(reason, ExitReason::TripleFault | ExitReason::Internal(_)),
        ) {
            l(&cpu.dump());
        }
        let finished = Finished {
            ram_bytes,
            #[cfg(not(freestanding))]
            disks: cpu.dev.into_disks(),
        };
        (reason, finished)
    }
}

pub struct EmulatedMachine {
    /// Taken by `start` (hosted) or `wait` (UEFI); `None` afterwards.
    vcpu: Option<Vcpu>,
    #[cfg(not(freestanding))]
    thread: Option<JoinHandle<ExitReason>>,
    #[cfg(freestanding)]
    started: bool,
    /// One per configured console, in configuration order.
    consoles: Vec<ConsoleHandle>,
    /// Same queues as the virtio-net devices in `Vcpu::devcfg`, in configuration order.
    nets: Vec<NetHandle>,
    stop: Arc<AtomicBool>,
}

impl EmulatedMachine {
    /// Validates the configuration and loads the kernel into guest RAM. Nothing executes until
    /// `start`.
    pub fn new(cfg: MachineConfig) -> Result<Self, Error> {
        if !(MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&cfg.memory_mib) {
            return Err(Error::BadConfig(format!(
                "memory_mib must be between {MIN_MEMORY_MIB} and {MAX_MEMORY_MIB}"
            )));
        }
        if cfg.jit == JitMode::On && !crate::jit::probe() {
            return Err(Error::JitUnavailable);
        }
        let mut ram = vec![0u8; cfg.memory_mib << BYTES_PER_MIB_SHIFT];
        let kinds = if cfg.consoles.is_empty() {
            vec![ConsoleKind::Virtio]
        } else {
            cfg.consoles.clone()
        };
        if kinds
            .iter()
            .filter(|kind| **kind == ConsoleKind::Serial)
            .count()
            > 1
        {
            return Err(Error::BadConfig(
                "there is one UART: ConsoleKind::Serial can be configured only once".into(),
            ));
        }
        #[cfg(not(freestanding))]
        let disk_count = cfg.disks.len();
        #[cfg(freestanding)]
        let disk_count = 0;
        let virtio_console_count = kinds
            .iter()
            .filter(|kind| **kind == ConsoleKind::Virtio)
            .count();
        let functions = virtio_console_count + disk_count + cfg.net_macs.len();
        if functions > MAX_VIRTIO_FUNCTIONS {
            return Err(Error::BadConfig(format!(
                "{functions} virtio devices configured (consoles, disks and NICs); \
                 the PCI bus has room for {MAX_VIRTIO_FUNCTIONS}"
            )));
        }
        let mut cmdline = cfg.cmdline.clone().unwrap_or_else(|| {
            // The first console is the guest's primary one.
            let default = if kinds[0] == ConsoleKind::Virtio {
                DEFAULT_CMDLINE
            } else {
                DEFAULT_SERIAL_CMDLINE
            };
            default.to_string()
        });
        for extra in &cfg.cmdline_extra {
            cmdline.push(' ');
            cmdline.push_str(extra);
        }
        let boot = match &cfg.firmware {
            Some(image) => boot::load_firmware(&mut ram, image)?,
            None => boot::load(&mut ram, &cfg.kernel, cfg.initrd.as_deref(), &cmdline)?,
        };
        let nets: Vec<([u8; 6], NetHandle)> = cfg
            .net_macs
            .iter()
            .map(|mac| (*mac, NetHandle::default()))
            .collect();
        let net_handles = nets.iter().map(|(_, handle)| handle.clone()).collect();
        let consoles: Vec<ConsoleHandle> = kinds.iter().map(|_| ConsoleHandle::default()).collect();
        let uart_console = kinds
            .iter()
            .position(|kind| *kind == ConsoleKind::Serial)
            .map_or_else(|| consoles[0].clone(), |index| consoles[index].clone());
        let virtio_consoles = kinds
            .iter()
            .zip(&consoles)
            .filter(|(kind, _)| **kind == ConsoleKind::Virtio)
            .map(|(_, handle)| handle.clone())
            .collect();
        let devcfg = DevConfig {
            #[cfg(not(freestanding))]
            disks: cfg.disks,
            nets,
            virtio_consoles,
        };
        let boot_ram = cfg.verify.then(|| ram.clone());
        // Verification cannot restart (the twin's devices cannot be rebuilt in step), so it keeps
        // no image and a reset ends the run.
        let reboot = (cfg.reboot && !cfg.verify && cfg.firmware.is_none()).then(|| RebootImage {
            kernel: cfg.kernel,
            initrd: cfg.initrd,
            cmdline,
            nets: devcfg.nets.clone(),
            virtio_consoles: devcfg.virtio_consoles.clone(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        Ok(Self {
            vcpu: Some(Vcpu {
                ram,
                boot,
                primary_console: consoles[0].clone(),
                uart_console,
                stop: stop.clone(),
                log: cfg.log,
                poll: cfg.poll,
                debug: cfg.debug,
                jit: cfg.jit,
                clock: cfg.clock,
                verify: cfg.verify,
                unix_time: cfg.unix_time,
                random_seed: cfg.random_seed,
                boot_ram,
                devcfg,
                reboot,
            }),
            #[cfg(not(freestanding))]
            thread: None,
            #[cfg(freestanding)]
            started: false,
            consoles,
            nets: net_handles,
            stop,
        })
    }

    /// The first configured console (the only one unless `MachineConfig::console` was called
    /// more than once).
    pub fn console(&self) -> ConsoleHandle {
        self.consoles[0].clone()
    }

    /// Every console, in `MachineConfig::console` order (one virtio console if none was added).
    /// Each has its own queues and size.
    pub fn consoles(&self) -> Vec<ConsoleHandle> {
        self.consoles.clone()
    }

    /// The first NIC's frame channel, `None` without one.
    pub fn net(&self) -> Option<NetHandle> {
        self.nets.first().cloned()
    }

    /// Every NIC's frame channel, in `MachineConfig::net` order.
    pub fn nets(&self) -> Vec<NetHandle> {
        self.nets.clone()
    }

    pub fn stop_handle(&self) -> StopHandle {
        StopHandle(self.stop.clone())
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Hosted: spawns the vCPU thread and returns. UEFI has no threads, so this only marks the
    /// machine started and the vCPU runs inside `wait`.
    pub fn start(&mut self) -> Result<(), Error> {
        #[cfg(not(freestanding))]
        {
            let vcpu = self.vcpu.take().ok_or(Error::AlreadyStarted)?;
            self.thread = Some(std::thread::spawn(move || vcpu.run()));
        }
        #[cfg(freestanding)]
        {
            if self.started || self.vcpu.is_none() {
                return Err(Error::AlreadyStarted);
            }
            self.started = true;
        }
        Ok(())
    }

    /// Blocks until the guest exits. `Stopped` if the machine was never started or was already
    /// waited on.
    #[cfg(freestanding)]
    pub fn wait(&mut self) -> ExitReason {
        match self.vcpu.take() {
            Some(v) if self.started => v.run(),
            _ => ExitReason::Stopped,
        }
    }

    /// Blocks until the guest exits; `Stopped` if the machine was never started.
    #[cfg(not(freestanding))]
    pub fn wait(&mut self) -> ExitReason {
        match self.thread.take() {
            Some(t) => t
                .join()
                .unwrap_or_else(|_| ExitReason::Internal("vcpu thread panicked".into())),
            None => ExitReason::Stopped,
        }
    }
}

#[cfg(test)]
mod tests;
