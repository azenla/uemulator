#![cfg_attr(target_os = "uefi", no_std)]

extern crate alloc;

mod boot;
mod console;
mod cpu;
mod dev;
#[cfg(not(target_os = "uefi"))]
mod disk;
mod jit;
mod machine;
#[cfg(target_os = "uefi")]
mod math;
mod netdev;
mod platform;
#[cfg(target_os = "uefi")]
mod prelude;
mod sync;
#[cfg(feature = "jit")]
mod sys;

pub use boot::BootState;
pub use console::ConsoleHandle;
#[cfg(not(target_os = "uefi"))]
pub use disk::{BlockBackend, Disk};
pub use machine::{
    ClockMode, ConsoleKind, EmulatedMachine, Error, ExitReason, JitMode, MachineConfig, StopHandle,
};
pub use netdev::NetHandle;
pub use platform::{DEFAULT_CMDLINE, Platform, load_linux};
