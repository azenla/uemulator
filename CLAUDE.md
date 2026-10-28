# uemulator

Minimal embeddable x86_64 system emulator in Rust. Boots an unmodified Linux bzImage with a virtio console (`hvc0`; the 16550 serial port is always present and carries early boot text), no firmware. Design is modeled on QEMU TCG: decode to micro-ops, TB cache, interpreter plus aarch64 and x86_64 host JITs with runtime fallback.

Read `plans/README.md` first; the plans are the source of truth for design and milestones (`plans/10-milestones.md` for current status).

## Commands

- Run: `cargo run --release -- quick debian` (downloads and caches netboot artifacts in `~/.uemulator/cache/quick/<distro>/artifacts/` and creates a sparse disk at `~/.uemulator/quick/<distro>/disk.img`; `UEMULATOR_HOME` moves `~/.uemulator`, `UEMULATOR_CACHE` moves just the cache; `--update` refetches; `quick` uses user-mode networking so the installers can reach their mirrors) or `-- boot --kernel K --initramfs I`
- Web: `./hack/web.sh serve` builds `crates/uemulator-web` for `wasm32-unknown-unknown` (`rustup target add` it first) into `target/web` and serves it on :8080 (needs the COOP/COEP headers `hack/web/serve.py` sends); `cargo clippy --release --target wasm32-unknown-unknown -p uemulator-web` must print zero warnings too
- UEFI: `./hack/uefi/boot.sh [x86_64|aarch64]` builds `crates/uemulator-uefi` and boots it in QEMU (accelerated when the host matches; `UEFI_ACCEL=hv|emu` picks hardware virtualization or the emulator, and `-cpu max,svm=on` under TCG exercises the SVM path); `cargo clippy --release --target x86_64-unknown-uefi -p uemulator-uefi` (and `aarch64-unknown-uefi`) must print zero warnings too
- Firmware / real mode: `uemulator firmware IMAGE` (image ends at 1 MiB, CPU starts in real mode at F000:FFF0); a bzImage without a 64-bit entry boots in 32-bit protected mode (`plans/15-legacy-modes.md`)
- Linux KVM check: `hack/kvm/run.sh` loads `kvm_amd` in the emulated guest and runs vCPUs in every CPU mode on the emulator's SVM (needs `unsquashfs`, a musl cross gcc, the Alpine artifacts and network for the modloop); run it after touching `cpu/svm.rs`, paging or the 16/32-bit modes
- Test: `cargo test --release --workspace` (also `-p uemulator --no-default-features`)
- Before EVERY commit: run `./hack/autofix.sh` (clippy --fix + fmt), then fix by hand every warning it leaves; `cargo clippy --release --all-targets` must print zero warnings. Never commit with warnings.

## Rules

- Cargo workspace: `crates/uemulator-uefi` (`no_std` UEFI app; on hosted targets only its hardware-independent modules build, which is what `cargo test -p uemulator-uefi` runs), `crates/uemulator` (library; `no_std` + `alloc` when `freestanding`; the default `jit` feature pulls in `libc` on unix and `windows-sys` on Windows, `--no-default-features` has zero dependencies and is interpreter-only), `crates/uemulator-cli` (clap, ureq; may use crates freely) and `crates/uemulator-web` (wasm32 only, empty elsewhere; C ABI plus `www/`). `cfg(freestanding)` (from `crates/uemulator/build.rs`: UEFI and wasm32) replaces `target_os = "uefi"` in the library. Tests live next to the code in each crate; the library's shared helpers are in `crates/uemulator/src/testkit.rs`.
- `lib.rs` is the entire public API: `EmulatedMachine::new(cfg)` then `start()`. Everything else is `pub(crate)`.
- The library never touches stdio, never calls `process::exit`, holds no global mutable state, never panics on guest behavior. Guest faults are guest exceptions.
- Comment what is surprising, looks obscure, or is named something non-obvious: hardware quirks, ordering constraints, invariants the JIT relies on, why a value is what it is. Cite the SDM section or the QEMU behavior when that is the real source. Don't restate what the code says.
- Write comments for the next Claude who has to change this without having seen the history. Say what breaks if the code is changed carelessly. Plain, terse prose like a human maintainer would write: no "This function...", no filler, no banner or section-divider comments.
- Prefer descriptive names. Don't shorten to 3-5 letters when a real word reads better; rename when you touch something badly named. Architectural names from the Intel/AMD manuals (`cr0`, `efer`, `rip`, `gdtr`, `cpl`, `mxcsr`) stay as-is.
- No magic constants. Name register bits, MSR numbers, descriptor flags, vector numbers, sizes and timeouts as `const`s at the narrowest scope that makes sense, and let comments refer to the name.
- `../ext/qemu-uplink` is read-only reference (see `plans/09-qemu-map.md`). Study behavior; write Rust from scratch, don't transliterate.
- The interpreter is the oracle for the JIT. Any new uop gets an interpreter implementation and a test first.
- Advertise a CPUID bit only together with the tests for the instructions behind it.
- Debug-profile is optimized (`opt-level = 3`) because the kernel decompressor is ~1e9 guest instructions.

## Tests

- Unit tests only: no network, no cached kernels or initrds, nothing under `~/.uemulator`. A test that needs a guest builds a tiny one: `testkit::FakeKernel` makes a valid bzImage whose 64-bit entry point is hand-assembled code (`echo 'mov rax, 5; hlt' | hack/asm.sh` prints the bytes; `BITS=16` or `BITS=32` assembles for real mode or 32-bit protected mode).
- `testkit::Harness` runs a snippet on one CPU, interpreter or JIT, with undelivered faults reported as `Stop::Fault`; `testkit::run_machine` boots a fake kernel on a full `EmulatedMachine`.
- Anything the JIT compiles gets a case that runs `testkit::assert_jit_matches_interpreter`. The interpreter is the oracle.
- Fixing a bug: write the failing test first.
- Under Rosetta (`--target x86_64-apple-darwin`) tests that run generated code take turns through `jit::serialize_jit_tests`; Rosetta's translation cache goes stale when one thread frees a code buffer and another maps one at the same address. `testkit::Harness::enable_jit` holds it for you; take it yourself when you use `Jit::new` directly.
- Real-kernel lockstep check (not a test, needs a kernel you supply): `--verify` on the CLI.

## Debugging

- `--debug` on the CLI dumps CPU state and scans guest RAM for the kernel log ring buffer on exit.
- `--verify` runs the JIT in lockstep with the interpreter and stops at the first diverging block; use it after any change under `crates/uemulator/src/jit/`. Do not type into it: host input reaches only the JIT side's console, so the first keystroke shows up as a bogus divergence at an interrupt entry (boot to the shell prompt and stop instead).
- `--time <unix>` fixes the RTC; guest behavior must not depend on it, so a time-dependent failure is an emulator bug.
- On an Apple Silicon Mac the x86_64 JIT runs under Rosetta: `cargo test --release -p uemulator --target x86_64-apple-darwin` (`rustup target add x86_64-apple-darwin`).
