# uemulator

Minimal embeddable x86_64 system emulator in Rust. Boots an unmodified Linux bzImage with a serial console, no firmware. Design is modeled on QEMU TCG: decode to micro-ops, TB cache, interpreter plus an aarch64 JIT with runtime fallback.

Read `plans/README.md` first; the plans are the source of truth for design and milestones (`plans/10-milestones.md` for current status).

## Commands

- Run: `cargo run --release --example boot-linux` (downloads and caches the Debian netboot kernel/initrd in `~/.cache/uemulator`, override with `UEMULATOR_CACHE`)
- Test: `cargo test --release`
- Before finishing: `hack/autofix.sh` (clippy fix + fmt), or `hack/format.sh` for fmt only

## Rules

- Zero crate dependencies. libc functions are declared by hand in `src/sys.rs` (`unsafe extern "C"`, edition 2024); take signatures from the SDK headers.
- `lib.rs` is the entire public API: `EmulatedMachine::new(cfg)` then `start()`. Everything else is `pub(crate)`.
- The library never touches stdio, never calls `process::exit`, holds no global mutable state, never panics on guest behavior. Guest faults are guest exceptions.
- Comments minimal; only for non-obvious why.
- `../ext/qemu-uplink` is read-only reference (see `plans/09-qemu-map.md`). Study behavior; write Rust from scratch, don't transliterate.
- The interpreter is the oracle for the JIT. Any new uop gets an interpreter implementation and a test first.
- Advertise a CPUID bit only together with the tests for the instructions behind it.
- Debug-profile is optimized (`opt-level = 3`) because the kernel decompressor is ~1e9 guest instructions.
