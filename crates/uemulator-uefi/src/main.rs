#![cfg_attr(target_os = "uefi", no_std)]
#![cfg_attr(target_os = "uefi", no_main)]
// On a hosted target only `cargo test` builds anything: the hardware-independent modules, whose
// items are partly used by UEFI-only code that is not compiled there.
#![cfg_attr(not(target_os = "uefi"), allow(dead_code))]

#[cfg(not(target_os = "uefi"))]
fn main() {
    eprintln!("uemulator-uefi only builds for x86_64-unknown-uefi and aarch64-unknown-uefi");
}

#[cfg(any(target_os = "uefi", test))]
extern crate alloc;

#[cfg(target_os = "uefi")]
mod app;
#[cfg(any(target_os = "uefi", test))]
mod config;
#[cfg(target_os = "uefi")]
mod console;
#[cfg(any(all(target_os = "uefi", target_arch = "x86_64"), test))]
mod hv;
#[cfg(any(target_os = "uefi", test))]
mod text;
#[cfg(all(target_os = "uefi", target_arch = "x86_64"))]
mod timer;
#[cfg(any(target_os = "uefi", test))]
mod unix_time;
