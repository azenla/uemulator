# Milestones

Each has an observable serial-output exit criterion. Interpreter first.

| # | deliverable | exit criterion |
|---|-------------|----------------|
| M0 | API skeleton, config, bzImage parse + loader, example downloads/caches kernel, clock + console + UART | `cargo run --release --example boot-linux` downloads (once), loads, and reports the parsed header; `cargo test` passes |
| M1 | 64-bit decoder + interpreter for the integer subset, exceptions, paging, soft TLB, PIC/PIT/CMOS | console prints `Linux version 6.12.107...` (decompressor finished, kernel `start_kernel` running) |
| M2 | Rest of integer ISA, SSE2 basics, x87 basics, TSC calibration, SMC invalidation, reset/triple-fault exit | no initrd: `Kernel panic - not syncing: VFS: Unable to mount root fs` then clean `ExitReason::Reset`; ≥50 MIPS release |
| M3 | Full initrd boot | `Run /init as init process`, then installer userspace output (SSE2 libc paths, syscalls, signals, fork/exec) |
| M4 | aarch64 JIT: probe, buffer, encoder, lowering, chaining, differential suite | M2 and M3 criteria reached with `JitMode::On`; differential suite green; ≥300 MIPS |
| M5 | Polish: realtime clock mode, stdio raw-mode in example, perf work from the fork list, optional LAPIC | interactive console usable; documented perf numbers |

## M0 task list

1. `Cargo.toml`: `[profile.dev] opt-level = 3`, `[profile.test]` same.
2. `lib.rs` re-exports; `machine.rs`, `console.rs`, `boot.rs`, `clock.rs`, `sys.rs` stubs with the API from 01.
3. `boot.rs`: header parse + validation errors (`Error::BadKernel(..)`), loader into a RAM buffer, boot_params and e820, initial GDT/page tables.
4. Example: cache dir, `curl -fL` to `.part` then rename, args, stdio bridge.
5. CLAUDE.md, README usage.

## M1 task list (first CPU work)

1. `Env`, register file, flags, hflags.
2. Decoder tables for the integer subset; `translate.rs` emitting uops.
3. Interpreter + helpers; `tlb`/`mmu`; exceptions/IDT delivery.
4. Devices from 07; run loop with clock and HLT warp.
5. Iterate on the boot: each unimplemented opcode/MSR/port is logged once; fix in order of appearance.
