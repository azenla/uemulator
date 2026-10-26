#![cfg_attr(target_os = "uefi", no_std)]
#![cfg_attr(target_os = "uefi", no_main)]

#[cfg(not(target_os = "uefi"))]
fn main() {
    eprintln!("uemulator-uefi only builds for x86_64-unknown-uefi and aarch64-unknown-uefi");
}

#[cfg(target_os = "uefi")]
extern crate alloc;

#[cfg(target_os = "uefi")]
mod app;
#[cfg(target_os = "uefi")]
mod config;
#[cfg(target_os = "uefi")]
mod console;
#[cfg(all(target_os = "uefi", target_arch = "x86_64"))]
mod hv;
#[cfg(all(target_os = "uefi", target_arch = "x86_64"))]
mod timer;
