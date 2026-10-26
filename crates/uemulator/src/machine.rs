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
    BadKernel(String),
    BadConfig(String),
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
    #[default]
    Auto,
    Off,
    On,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClockMode {
    #[default]
    Virtual,
    Realtime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    Shutdown,
    Reset,
    TripleFault,
    Stopped,
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsoleKind {
    #[default]
    Serial,
    Virtio,
}

pub const MIN_MEMORY_MIB: usize = 64;
pub const MAX_MEMORY_MIB: usize = 3072;

pub(crate) type LogFn = Arc<dyn Fn(&str) + Send + Sync>;
pub(crate) type PollFn = Arc<dyn Fn(bool) + Send + Sync>;
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
            memory_mib: 512,
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

    #[doc(hidden)]
    pub fn verify_jit(mut self, on: bool) -> Self {
        self.verify = on;
        self
    }

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

#[derive(Clone)]
pub struct StopHandle(Arc<AtomicBool>);

impl StopHandle {
    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

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
        if let Some(t) = self.unix_time {
            cpu.dev.set_boot_time(t);
        }
        cpu.realtime = self.clock == ClockMode::Realtime;
        cpu.poll = self.poll.take().map(|p| {
            let f: PollFn = Arc::new(move |idle| p(&console, idle));
            f
        });
        if self.jit != JitMode::Off {
            cpu.jit = crate::jit::Jit::new().map(Box::new);
        }
        if self.debug {
            cpu.trace = Some(Default::default());
        }
        let r = if self.verify {
            let mut sh = Cpu::new(
                self.boot_ram.take().unwrap(),
                self.boot,
                ConsoleHandle::default(),
                cpu.stop.clone(),
                None,
                DevConfig::default(),
            );
            cpu.run_verify(&mut sh)
        } else {
            cpu.run()
        };
        if let (Some(l), true) = (
            &self.log,
            self.debug || matches!(r, ExitReason::TripleFault | ExitReason::Internal(_)),
        ) {
            l(&cpu.dump());
        }
        r
    }
}

pub struct EmulatedMachine {
    vcpu: Option<Vcpu>,
    #[cfg(not(target_os = "uefi"))]
    thread: Option<JoinHandle<ExitReason>>,
    #[cfg(target_os = "uefi")]
    started: bool,
    console: ConsoleHandle,
    net: Option<NetHandle>,
    stop: Arc<AtomicBool>,
}

impl EmulatedMachine {
    pub fn new(cfg: MachineConfig) -> Result<Self, Error> {
        if !(MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&cfg.memory_mib) {
            return Err(Error::BadConfig(format!(
                "memory_mib must be between {MIN_MEMORY_MIB} and {MAX_MEMORY_MIB}"
            )));
        }
        if cfg.jit == JitMode::On && !crate::jit::probe() {
            return Err(Error::JitUnavailable);
        }
        let mut ram = vec![0u8; cfg.memory_mib << 20];
        let mut cmdline = cfg.cmdline.clone().unwrap_or_else(|| {
            let con = if cfg.console_kind == ConsoleKind::Virtio {
                "hvc0"
            } else {
                "ttyS0"
            };
            format!(
                "console={con} earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1"
            )
        });
        for extra in &cfg.cmdline_extra {
            cmdline.push(' ');
            cmdline.push_str(extra);
        }
        let boot = boot::load(&mut ram, &cfg.kernel, cfg.initrd.as_deref(), &cmdline)?;
        let net = cfg.net_mac.map(|m| (m, NetHandle::default()));
        let net_handle = net.as_ref().map(|(_, h)| h.clone());
        let devcfg = DevConfig {
            #[cfg(not(target_os = "uefi"))]
            disk: cfg.disk,
            net,
            virtio_console: cfg.console_kind == ConsoleKind::Virtio,
        };
        let ram_copy = if cfg.verify { ram.clone() } else { Vec::new() };
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
                boot_ram: if cfg.verify { Some(ram_copy) } else { None },
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

    pub fn start(&mut self) -> Result<(), Error> {
        #[cfg(not(target_os = "uefi"))]
        {
            let vcpu = self.vcpu.take().ok_or(Error::AlreadyStarted)?;
            self.thread = Some(std::thread::spawn(move || vcpu.run()));
        }
        // There are no threads: the vCPU runs inside `wait`.
        #[cfg(target_os = "uefi")]
        {
            if self.started || self.vcpu.is_none() {
                return Err(Error::AlreadyStarted);
            }
            self.started = true;
        }
        Ok(())
    }

    #[cfg(target_os = "uefi")]
    pub fn wait(&mut self) -> ExitReason {
        match self.vcpu.take() {
            Some(v) if self.started => v.run(),
            _ => ExitReason::Stopped,
        }
    }

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
