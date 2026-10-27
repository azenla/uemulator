//! The decoder and interpreter recurse enough that the 1 MiB default wasm stack is tight.

fn main() {
    println!(
        "cargo:rustc-link-arg-cdylib=-zstack-size={}",
        16 * 1024 * 1024
    );
}
