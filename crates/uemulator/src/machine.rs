#[cfg(target_os = "uefi")]
use crate::prelude::*;
use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_os = "uefi"))]
use std::thread::JoinHandle;

use crate::boot::{self, BootState};
use crate::console::ConsoleHandle;
use crate::cpu::Cpu;
use crate::dev::DevConfig;
#[cfg(not(target_os = "uefi"))]
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
    /// 16550 UART at COM1 (`ttyS0`).
    Serial,
    /// virtio-console (`hvc0`); the only kind that delivers `ConsoleHandle::resize` to the guest.
    /// The default. The UART is present either way and carries early boot text (`earlyprintk`),
    /// but this needs a kernel whose `virtio_console` driver is built in (an initrd that only
    /// loads it as a module has no console until then); pick `Serial` for such kernels.
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
    initrd: Option<Vec<u8>>,
    cmdline: Option<String>,
    cmdline_extra: Vec<String>,
    memory_mib: usize,
    #[cfg(not(target_os = "uefi"))]
    disk: Option<Disk>,
    net_mac: Option<[u8; 6]>,
    console_kind: ConsoleKind,
    jit: JitMode,
    clock: ClockMode,
    log: Option<LogFn>,
    poll: Option<HostPollFn>,
    debug: bool,
    verify: bool,
    unix_time: Option<u64>,
    reboot: bool,
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            kernel: Vec::new(),
            initrd: None,
            cmdline: None,
            cmdline_extra: Vec::new(),
            memory_mib: DEFAULT_MEMORY_MIB,
            #[cfg(not(target_os = "uefi"))]
            disk: None,
            net_mac: None,
            console_kind: ConsoleKind::Virtio,
            jit: JitMode::Auto,
            clock: ClockMode::Virtual,
            log: None,
            poll: None,
            debug: false,
            verify: false,
            unix_time: None,
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

    #[cfg(not(target_os = "uefi"))]
    pub fn disk(mut self, disk: Disk) -> Self {
        self.disk = Some(disk);
        self
    }

    /// Adds a virtio-net device with this MAC; frames flow through `EmulatedMachine::net()`.
    pub fn net(mut self, mac: [u8; 6]) -> Self {
        self.net_mac = Some(mac);
        self
    }

    pub fn console(mut self, kind: ConsoleKind) -> Self {
        self.console_kind = kind;
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
    console: ConsoleHandle,
    stop: Arc<AtomicBool>,
    log: Option<LogFn>,
    poll: Option<HostPollFn>,
    debug: bool,
    jit: JitMode,
    clock: ClockMode,
    verify: bool,
    unix_time: Option<u64>,
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
    net: Option<([u8; 6], NetHandle)>,
    virtio_console: bool,
}

/// Wall-clock seconds, or 0 where there is no clock (UEFI keeps its own, see `unix_time`).
fn host_unix_time() -> u64 {
    #[cfg(not(target_os = "uefi"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0)
    }
    #[cfg(target_os = "uefi")]
    {
        0
    }
}

/// Stands in for a disk on the verify twin: same capacity and read-only flag as the real one so
/// the guest sees the same device, but no data behind it. Reads return zeros, writes vanish.
#[cfg(not(target_os = "uefi"))]
struct ShadowDisk {
    len: u64,
    read_only: bool,
}

#[cfg(not(target_os = "uefi"))]
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
fn mirrored_devices(main: &DevConfig) -> DevConfig {
    DevConfig {
        #[cfg(not(target_os = "uefi"))]
        disk: main.disk.as_ref().map(|disk| {
            Disk::from_backend(ShadowDisk {
                len: disk.backend.len(),
                read_only: disk.backend.read_only(),
            })
        }),
        net: main
            .net
            .as_ref()
            .map(|(mac, _)| (*mac, NetHandle::default())),
        virtio_console: main.virtio_console,
    }
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
) -> Cpu {
    let mut twin = Cpu::new(ram, boot, ConsoleHandle::default(), stop, None, devices);
    twin.dev.set_boot_time(boot_unix);
    twin.realtime = false;
    twin
}

/// What a finished run hands to the next one.
struct Finished {
    ram_bytes: usize,
    #[cfg(not(target_os = "uefi"))]
    disk: Option<Disk>,
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
                #[cfg(not(target_os = "uefi"))]
                disk: finished.disk,
                net: image.net.clone(),
                virtio_console: image.virtio_console,
            };
        }
    }

    /// One power-on to the next exit. Everything the CPU owns is built here, so a reboot starts
    /// from a CPU with no leftover state and from a JIT with an empty code cache: a reused cache
    /// would never see stores to the new CPU's empty `page_has_code` map and could run
    /// translations of the previous run's RAM.
    fn run_once(&mut self) -> (ExitReason, Finished) {
        let console = self.console.clone();
        // Under verify both CPUs must start from one boot time even when the caller gave none;
        // two separate wall-clock reads could straddle a second boundary.
        let verify_twin = self.verify.then(|| {
            (
                mirrored_devices(&self.devcfg),
                self.unix_time.unwrap_or_else(host_unix_time),
            )
        });
        let ram = core::mem::take(&mut self.ram);
        let ram_bytes = ram.len();
        let mut cpu = Cpu::new(
            ram,
            self.boot,
            self.console.clone(),
            self.stop.clone(),
            self.log.clone(),
            core::mem::take(&mut self.devcfg),
        );
        let boot_unix = verify_twin
            .as_ref()
            .map(|(_, boot_unix)| *boot_unix)
            .or(self.unix_time);
        if let Some(boot_unix) = boot_unix {
            cpu.dev.set_boot_time(boot_unix);
        }
        // Cloned, not taken: a reboot builds a new CPU that needs the hook again.
        cpu.poll = self.poll.clone().map(|host_poll| {
            let poll: PollFn = Arc::new(move |idle| host_poll(&console, idle));
            poll
        });
        if self.jit != JitMode::Off {
            cpu.jit = crate::jit::Jit::new().map(Box::new);
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
            Some((devices, boot_unix)) if lockstep => {
                let mut twin = interpreter_twin(
                    self.boot_ram.take().unwrap(),
                    self.boot,
                    cpu.stop.clone(),
                    boot_unix,
                    devices,
                );
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
            #[cfg(not(target_os = "uefi"))]
            disk: cpu.dev.into_disk(),
        };
        (reason, finished)
    }
}

pub struct EmulatedMachine {
    /// Taken by `start` (hosted) or `wait` (UEFI); `None` afterwards.
    vcpu: Option<Vcpu>,
    #[cfg(not(target_os = "uefi"))]
    thread: Option<JoinHandle<ExitReason>>,
    #[cfg(target_os = "uefi")]
    started: bool,
    console: ConsoleHandle,
    /// Same queues as the virtio-net device in `Vcpu::devcfg`.
    net: Option<NetHandle>,
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
        let mut cmdline = cfg.cmdline.clone().unwrap_or_else(|| {
            let default = if cfg.console_kind == ConsoleKind::Virtio {
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
        let boot = boot::load(&mut ram, &cfg.kernel, cfg.initrd.as_deref(), &cmdline)?;
        let net = cfg.net_mac.map(|mac| (mac, NetHandle::default()));
        let net_handle = net.as_ref().map(|(_, handle)| handle.clone());
        let virtio_console = cfg.console_kind == ConsoleKind::Virtio;
        let devcfg = DevConfig {
            #[cfg(not(target_os = "uefi"))]
            disk: cfg.disk,
            net,
            virtio_console,
        };
        let boot_ram = cfg.verify.then(|| ram.clone());
        // Verification cannot restart (the twin's devices cannot be rebuilt in step), so it keeps
        // no image and a reset ends the run.
        let reboot = (cfg.reboot && !cfg.verify).then(|| RebootImage {
            kernel: cfg.kernel,
            initrd: cfg.initrd,
            cmdline,
            net: devcfg.net.clone(),
            virtio_console,
        });
        let console = ConsoleHandle::default();
        let stop = Arc::new(AtomicBool::new(false));
        Ok(Self {
            vcpu: Some(Vcpu {
                ram,
                boot,
                console: console.clone(),
                stop: stop.clone(),
                log: cfg.log,
                poll: cfg.poll,
                debug: cfg.debug,
                jit: cfg.jit,
                clock: cfg.clock,
                verify: cfg.verify,
                unix_time: cfg.unix_time,
                boot_ram,
                devcfg,
                reboot,
            }),
            #[cfg(not(target_os = "uefi"))]
            thread: None,
            #[cfg(target_os = "uefi")]
            started: false,
            console,
            net: net_handle,
            stop,
        })
    }

    pub fn console(&self) -> ConsoleHandle {
        self.console.clone()
    }

    pub fn net(&self) -> Option<NetHandle> {
        self.net.clone()
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
        #[cfg(not(target_os = "uefi"))]
        {
            let vcpu = self.vcpu.take().ok_or(Error::AlreadyStarted)?;
            self.thread = Some(std::thread::spawn(move || vcpu.run()));
        }
        #[cfg(target_os = "uefi")]
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
    #[cfg(target_os = "uefi")]
    pub fn wait(&mut self) -> ExitReason {
        match self.vcpu.take() {
            Some(v) if self.started => v.run(),
            _ => ExitReason::Stopped,
        }
    }

    /// Blocks until the guest exits; `Stopped` if the machine was never started.
    #[cfg(not(target_os = "uefi"))]
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
