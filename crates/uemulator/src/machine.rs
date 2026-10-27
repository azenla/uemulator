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
    /// Guest time is derived from the instruction count and jumps ahead to the next device
    /// deadline while the guest is halted, so runs are deterministic.
    #[default]
    Virtual,
    /// While halted, sleeps on the host (up to 2 ms per wait) before advancing guest time by what
    /// was slept, instead of skipping ahead at once. Guest time still counts instructions, so it
    /// only approximates wall time. Ignored on UEFI, where the clock stays virtual.
    Realtime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    /// Never produced at present; the only guest-initiated exit besides a triple fault is `Reset`.
    Shutdown,
    /// The guest asked for a reset through the keyboard controller (port 0x64), fast A20 (0x92)
    /// or reset control (0xcf9) port; see `dev/mod.rs`. Kernels reboot on panic with `panic=-1`,
    /// so a panic ends a run this way.
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
    #[default]
    Serial,
    /// virtio-console (`hvc0`); the only kind that delivers `ConsoleHandle::resize` to the guest.
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
            console_kind: ConsoleKind::Serial,
            jit: JitMode::Auto,
            clock: ClockMode::Virtual,
            log: None,
            poll: None,
            debug: false,
            verify: false,
            unix_time: None,
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
    /// where their state diverges. Very slow; for JIT development.
    #[doc(hidden)]
    pub fn verify_jit(mut self, on: bool) -> Self {
        self.verify = on;
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
}

impl Vcpu {
    fn run(mut self) -> ExitReason {
        let console = self.console.clone();
        let mut cpu = Cpu::new(
            self.ram,
            self.boot,
            self.console,
            self.stop,
            self.log.clone(),
            core::mem::take(&mut self.devcfg),
        );
        if let Some(unix_time) = self.unix_time {
            cpu.dev.set_boot_time(unix_time);
        }
        cpu.realtime = self.clock == ClockMode::Realtime;
        cpu.poll = self.poll.take().map(|host_poll| {
            let poll: PollFn = Arc::new(move |idle| host_poll(&console, idle));
            poll
        });
        if self.jit != JitMode::Off {
            cpu.jit = crate::jit::Jit::new().map(Box::new);
        }
        if self.debug {
            cpu.trace = Some(Default::default());
        }
        let reason = if self.verify {
            // The interpreter-only twin is the oracle for the JIT-enabled `cpu`. It shares the
            // stop flag but has no console or devices wired to the host.
            let mut interpreter_cpu = Cpu::new(
                self.boot_ram.take().unwrap(),
                self.boot,
                ConsoleHandle::default(),
                cpu.stop.clone(),
                None,
                DevConfig::default(),
            );
            cpu.run_verify(&mut interpreter_cpu)
        } else {
            cpu.run()
        };
        if let (Some(l), true) = (
            &self.log,
            self.debug || matches!(reason, ExitReason::TripleFault | ExitReason::Internal(_)),
        ) {
            l(&cpu.dump());
        }
        reason
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
            // Keep in sync with `platform::DEFAULT_CMDLINE`.
            let console_device = if cfg.console_kind == ConsoleKind::Virtio {
                "hvc0"
            } else {
                "ttyS0"
            };
            format!(
                "console={console_device} earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1"
            )
        });
        for extra in &cfg.cmdline_extra {
            cmdline.push(' ');
            cmdline.push_str(extra);
        }
        let boot = boot::load(&mut ram, &cfg.kernel, cfg.initrd.as_deref(), &cmdline)?;
        let net = cfg.net_mac.map(|mac| (mac, NetHandle::default()));
        let net_handle = net.as_ref().map(|(_, handle)| handle.clone());
        let devcfg = DevConfig {
            #[cfg(not(target_os = "uefi"))]
            disk: cfg.disk,
            net,
            virtio_console: cfg.console_kind == ConsoleKind::Virtio,
        };
        let boot_ram = cfg.verify.then(|| ram.clone());
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
