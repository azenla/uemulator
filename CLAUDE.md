# uemulator

Minimal embeddable x86_64 system emulator in Rust. Boots an unmodified Linux bzImage with a serial console, no firmware. Design is modeled on QEMU TCG: decode to micro-ops, TB cache, interpreter plus aarch64 and x86_64 host JITs with runtime fallback.

Read `plans/README.md` first; the plans are the source of truth for design and milestones (`plans/10-milestones.md` for current status).

## Commands

- Run: `cargo run --release -- quick debian` (downloads and caches netboot artifacts in `~/.uemulator/cache/quick/<distro>/artifacts/` and creates a sparse disk at `~/.uemulator/quick/<distro>/disk.img`; `UEMULATOR_HOME` moves `~/.uemulator`, `UEMULATOR_CACHE` moves just the cache; `--update` refetches; `quick` uses user-mode networking so the installers can reach their mirrors) or `-- boot --kernel K --initramfs I`
- UEFI: `./hack/uefi/boot.sh [x86_64|aarch64]` builds `crates/uemulator-uefi` and boots it in QEMU (accelerated when the host matches; `UEFI_ACCEL=hv|emu` picks hardware virtualization or the emulator, and `-cpu max,svm=on` under TCG exercises the SVM path); `cargo clippy --release --target x86_64-unknown-uefi -p uemulator-uefi` (and `aarch64-unknown-uefi`) must print zero warnings too
- Test: `cargo test --release --workspace` (also `-p uemulator --no-default-features`)
- Before EVERY commit: run `./hack/autofix.sh` (clippy --fix + fmt), then fix by hand every warning it leaves; `cargo clippy --release --all-targets` must print zero warnings. Never commit with warnings.

## Rules

- Cargo workspace: `crates/uemulator-uefi` (`no_std` UEFI app, empty on hosted targets), `crates/uemulator` (library; `no_std` + `alloc` on `target_os = "uefi"`; the default `jit` feature pulls in `libc` on unix and `windows-sys` on Windows, `--no-default-features` has zero dependencies and is interpreter-only) and `crates/uemulator-cli` (clap, ureq; may use crates freely). Tests live in the library crate.
- `lib.rs` is the entire public API: `EmulatedMachine::new(cfg)` then `start()`. Everything else is `pub(crate)`.
- The library never touches stdio, never calls `process::exit`, holds no global mutable state, never panics on guest behavior. Guest faults are guest exceptions.
- Comments minimal; only for non-obvious why.
- `../ext/qemu-uplink` is read-only reference (see `plans/09-qemu-map.md`). Study behavior; write Rust from scratch, don't transliterate.
- The interpreter is the oracle for the JIT. Any new uop gets an interpreter implementation and a test first.
- Advertise a CPUID bit only together with the tests for the instructions behind it.
- Debug-profile is optimized (`opt-level = 3`) because the kernel decompressor is ~1e9 guest instructions.

## Debugging

- `--debug` on the CLI dumps CPU state and scans guest RAM for the kernel log ring buffer on exit.
- `--verify` runs the JIT in lockstep with the interpreter and stops at the first diverging block; use it after any change under `crates/uemulator/src/jit/`.
- `--time <unix>` fixes the RTC; guest behavior must not depend on it, so a time-dependent failure is an emulator bug.
- Slow lockstep test: `cargo test --release -- --ignored`.
- On an Apple Silicon Mac the x86_64 JIT runs under Rosetta: `cargo test --release -p uemulator --target x86_64-apple-darwin` (`rustup target add x86_64-apple-darwin`).
