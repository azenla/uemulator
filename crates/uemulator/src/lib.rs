//! Embeddable x86_64 PC emulator that boots an unmodified Linux bzImage. Start from
//! `MachineConfig` and `EmulatedMachine`.
//!
//! The cfg gates: `target_os = "uefi"` selects the `no_std` build, which has no threads and no
//! `std::fs` (so no `disk`) and needs `prelude` and `math` to replace what `std` provides. The
//! `jit` feature enables `sys` (executable memory); without it the crate has no dependencies and
//! only interprets.

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
