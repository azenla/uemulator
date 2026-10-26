use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crate::boot::{self, BootState};
use crate::console::ConsoleHandle;

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

impl std::error::Error for Error {}

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

type LogFn = Arc<dyn Fn(&str) + Send + Sync>;

pub struct MachineConfig {
    kernel: Vec<u8>,
    initrd: Option<Vec<u8>>,
    cmdline: String,
    memory_mib: usize,
    jit: JitMode,
    clock: ClockMode,
    log: Option<LogFn>,
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            kernel: Vec::new(),
            initrd: None,
            cmdline:
                "console=ttyS0 earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1"
                    .into(),
            memory_mib: 512,
            jit: JitMode::Auto,
            clock: ClockMode::Virtual,
            log: None,
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
        self.cmdline = s.into();
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

    pub fn log(mut self, f: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.log = Some(Arc::new(f));
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

#[allow(dead_code)]
struct Vcpu {
    ram: Vec<u8>,
    boot: BootState,
    console: ConsoleHandle,
    stop: Arc<AtomicBool>,
    jit: JitMode,
    clock: ClockMode,
    log: Option<LogFn>,
}

impl Vcpu {
    fn run(self) -> ExitReason {
        if self.stop.load(Ordering::SeqCst) {
            return ExitReason::Stopped;
        }
        ExitReason::Internal("cpu core not implemented yet (see plans/10-milestones.md M1)".into())
    }
}

pub struct EmulatedMachine {
    vcpu: Option<Vcpu>,
    thread: Option<JoinHandle<ExitReason>>,
    console: ConsoleHandle,
    stop: Arc<AtomicBool>,
}

impl EmulatedMachine {
    pub fn new(cfg: MachineConfig) -> Result<Self, Error> {
        if cfg.memory_mib < 64 {
            return Err(Error::BadConfig("memory_mib must be at least 64".into()));
        }
        if cfg.jit == JitMode::On {
            return Err(Error::JitUnavailable);
        }
        let mut ram = vec![0u8; cfg.memory_mib << 20];
        let boot = boot::load(&mut ram, &cfg.kernel, cfg.initrd.as_deref(), &cfg.cmdline)?;
        let console = ConsoleHandle::default();
        let stop = Arc::new(AtomicBool::new(false));
        Ok(Self {
            vcpu: Some(Vcpu {
                ram,
                boot,
                console: console.clone(),
                stop: stop.clone(),
                jit: cfg.jit,
                clock: cfg.clock,
                log: cfg.log,
            }),
            thread: None,
            console,
            stop,
        })
    }

    pub fn console(&self) -> ConsoleHandle {
        self.console.clone()
    }

    pub fn stop_handle(&self) -> StopHandle {
        StopHandle(self.stop.clone())
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn start(&mut self) -> Result<(), Error> {
        let vcpu = self.vcpu.take().ok_or(Error::AlreadyStarted)?;
        self.thread = Some(std::thread::spawn(move || vcpu.run()));
        Ok(())
    }

    pub fn wait(&mut self) -> ExitReason {
        match self.thread.take() {
            Some(t) => t
                .join()
                .unwrap_or_else(|_| ExitReason::Internal("vcpu thread panicked".into())),
            None => ExitReason::Stopped,
        }
    }
}
