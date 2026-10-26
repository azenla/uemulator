# uemulator

Minimal embeddable x86_64 system emulator in Rust. Boots an unmodified Linux bzImage with a serial console, no firmware. Design is modeled on QEMU TCG: decode to micro-ops, TB cache, interpreter plus an aarch64 JIT with runtime fallback.

Read `plans/README.md` first; the plans are the source of truth for design and milestones (`plans/10-milestones.md` for current status).

## Commands

- Run: `cargo run --release -- quick debian` (downloads and caches netboot artifacts in `~/.cache/uemulator/quick/<distro>/artifacts/`, override the base with `UEMULATOR_CACHE`; `--update` refetches) or `-- boot --kernel K --initramfs I`
- Test: `cargo test --release`
- Before EVERY commit: run `./hack/autofix.sh` (clippy --fix + fmt), then fix by hand every warning it leaves; `cargo clippy --release --all-targets` must print zero warnings. Never commit with warnings.

## Rules

- Cargo workspace: `crates/uemulator` (library, no dependencies) and `crates/uemulator-cli` (clap, ureq; may use crates freely). Tests live in the library crate.
- `lib.rs` is the entire public API: `EmulatedMachine::new(cfg)` then `start()`. Everything else is `pub(crate)`.
- The library never touches stdio, never calls `process::exit`, holds no global mutable state, never panics on guest behavior. Guest faults are guest exceptions.
- Comments minimal; only for non-obvious why.
- `../ext/qemu-uplink` is read-only reference (see `plans/09-qemu-map.md`). Study behavior; write Rust from scratch, don't transliterate.
- The interpreter is the oracle for the JIT. Any new uop gets an interpreter implementation and a test first.
- Advertise a CPUID bit only together with the tests for the instructions behind it.
- Debug-profile is optimized (`opt-level = 3`) because the kernel decompressor is ~1e9 guest instructions.

## Debugging

- `--debug` on the example dumps CPU state and scans guest RAM for the kernel log ring buffer on exit.
- `--verify` runs the JIT in lockstep with the interpreter and stops at the first diverging block; use it after any change under `crates/uemulator/src/jit/`.
- `--time <unix>` fixes the RTC; guest behavior must not depend on it, so a time-dependent failure is an emulator bug.
- Slow lockstep test: `cargo test --release -- --ignored`.
