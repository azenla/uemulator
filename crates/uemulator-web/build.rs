//! The decoder and interpreter recurse enough that the 1 MiB default wasm stack is tight. The JIT
//! needs the function table: generated blocks are added to it by the page (`www/jit_host.js`) and
//! called through it, and the helpers they call are in it.
//!
//! These are wasm-ld flags only: on any other target the crate is empty and a host linker rejects
//! them, which would break `cargo build --workspace`.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("wasm32") {
        return;
    }
    println!(
        "cargo:rustc-link-arg-cdylib=-zstack-size={}",
        16 * 1024 * 1024
    );
    println!("cargo:rustc-link-arg-cdylib=--export-table");
    println!("cargo:rustc-link-arg-cdylib=--growable-table");
}
