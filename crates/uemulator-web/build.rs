//! The decoder and interpreter recurse enough that the 1 MiB default wasm stack is tight. The JIT
//! needs the function table: generated blocks are added to it by the page (`www/jit_host.js`) and
//! called through it, and the helpers they call are in it.

fn main() {
    println!(
        "cargo:rustc-link-arg-cdylib=-zstack-size={}",
        16 * 1024 * 1024
    );
    println!("cargo:rustc-link-arg-cdylib=--export-table");
    println!("cargo:rustc-link-arg-cdylib=--growable-table");
}
