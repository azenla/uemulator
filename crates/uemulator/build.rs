//! Defines `cfg(freestanding)` for the targets that have no threads, clock or filesystem behind
//! `std`: the library then builds `no_std`, runs the vCPU on the caller's thread and is driven
//! through `MachineConfig::poll`. UEFI and WebAssembly are such targets.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(freestanding)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os == "uefi" || arch == "wasm32" {
        println!("cargo:rustc-cfg=freestanding");
    }
}
