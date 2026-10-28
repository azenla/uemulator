//! Links the firmware as a flat image at its load address (`link.ld`). Host builds, which only
//! run the unit tests, link normally.

fn main() {
    println!("cargo:rerun-if-changed=link.ld");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("none") {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!("cargo:rustc-link-arg-bins=-T{dir}/link.ld");
        // An ELF with symbols, for objdump, when debugging the firmware itself.
        println!("cargo:rerun-if-env-changed=UEMULATOR_FIRMWARE_ELF");
        if std::env::var_os("UEMULATOR_FIRMWARE_ELF").is_none() {
            println!("cargo:rustc-link-arg-bins=--oformat=binary");
        }
        println!("cargo:rustc-link-arg-bins=--no-pie");
    }
}
