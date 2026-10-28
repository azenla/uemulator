# Milestones

## Status

- M0-M3 done: kernel banner, VFS panic then `ExitReason::Reset`, and the Debian installer userspace (dialogs drawn) with both interpreter and JIT.
- M4 done: aarch64 JIT (block translator, inline TLB fast paths, ALU/mov/lea/push/pop/jcc/call/ret inlined, helper fallback for everything else, lazy-flag-aware jcc, per-page invalidation, self-modifying-code exit). ~5x faster than the interpreter (no-initrd boot: 6.4 s vs 32 s; roughly 300 MIPS). A lockstep verifier (`MachineConfig::verify_jit`, `--verify` in the example) runs the JIT against the interpreter per block; the whole installer boot is divergence-free.
- M6 done (see 11-storage-net-console.md): configurable RAM (`--mem`), virtio-pci disk, network and console (with resize), guest-verified: ping through the gateway responder, `stty size` follows `resize()`, disk read/write through `virtio_blk`.
- M5 partly done: realtime idle (`ClockMode::Realtime`), raw tty in the example, fixed RTC time option, aarch64 block chaining and wide inline lowering (see 06: roughly 3-4x over the M4 JIT; no-initrd Alpine boot 4.8 s -> 1.3 s CPU, initrd boot to the installer 13.9 s -> 4.7 s, interpreter ~60 s, lockstep-verified over the whole no-initrd boot). Not done: guest register caching, dead-flag elimination, LAPIC/IOAPIC, SMP.

- Machine lifecycle done: a guest reset reboots the machine in place by default (`MachineConfig::reboot`, CLI `--no-reboot` to end with `ExitReason::Reset`; checked on the real Alpine kernel with `busybox reboot -f`, interpreter and JIT), `hlt` with IF=0 is `ExitReason::Shutdown` (checked with `busybox poweroff -f`; the UEFI hardware-virtualization loop applies the same rule, but has no in-process reboot). `ConsoleKind::Virtio` (hvc0) is the default everywhere except the UEFI hardware-virtualization path (no virtio in `Platform`). The M2 criterion "VFS panic then clean Reset" now needs `--no-reboot`; by default `panic=-1` reboots forever.

- Workspace split done: `crates/uemulator` (library) + `crates/uemulator-cli` (`uemulator boot|quick`, clap). 
- `jit` feature (default on) gates `libc`/`windows-sys`; `--no-default-features` is dependency-free and interpreter-only.
- x86_64 host JIT done (see 06): same inlined subset as aarch64, verified in lockstep. Exercised on macOS x86_64 under Rosetta; Linux and Windows x86_64 hosts compile but are untested.

- Instruction extensions done (see 03, "Instruction extensions"): POPCNT/LZCNT, CMPXCHG16B, SSE3, SSSE3 (XMM forms), SSE4.1, SSE4.2 (incl. CRC32 and pcmp[ei]str[im]), PCLMULQDQ, AES-NI, MOVBE and FSGSBASE are implemented and advertised in CPUID (leaf 1 ECX, 0x80000001 ECX, leaf 7). The Alpine boot now executes cmpxchg16b, popcnt, pshufb and palignr in the kernel. Not advertised: VMX.

- SVM done for 64-bit guests (see 03): vmrun/vmload/vmsave/stgi/clgi/invlpga, intercepts, event injection, nested paging, TSC offset; tested with hand-built VMCBs (interpreter and JIT) and unproven against a real nested hypervisor.

- ERMS/FSRM done: byte-sized rep slices make `rep movsb` copy-heavy guest workloads ~12x faster (03).

- AVX, AVX2, FMA and F16C done (see 03): VEX.128/256 through the SSE kernels, YMM upper state in XSAVE, gathers, masked moves, a fused multiply-add on the soft-float engine, half-precision conversion. Found while booting Debian: a register-collision bug in the VEX staging that broke glibc's AVX2 `strcmp`, now covered by tests.

- Real mode, 16/32-bit protected mode, compatibility mode and virtual-8086 mode done (see 15): mode derived from CS, a mode-aware decoder, segment bases and limits, 16/32-bit interrupt and call-gate delivery, hardware task switching, sysenter/sysexit, and `MachineConfig::firmware` / `uemulator firmware` to boot a real-mode image. 49 tests in `cpu/legacy_tests.rs`. The JIT stays 64-bit-only.

- Non-long-mode paging done (see 04 and 03, "Control-register rules outside long mode"): paging off, 32-bit two-level paging (4 MiB pages, PSE-36 advertised), PAE with latched PDPTEs, and the CR0/CR3/CR4/EFER transition rules; 82 tests in `cpu/paging_tests.rs`.

- AVX-512 done (see 03, "AVX-512"): EVEX decoding, 32 ZMM registers, opmasks and XSAVE components 5-7, AVX512F/VL/BW/DQ/CD/IFMA/VBMI/VBMI2/VNNI/BITALG/VPOPCNTDQ, GFNI, VAES and VPCLMULQDQ, with masking, broadcast, embedded rounding and fault suppression; advertised in CPUID. Alpine and the Debian installer boot with it on.

- UMIP, SMEP/SMAP, PCID/INVPCID, MONITOR/MWAIT, RDTSCP done (see 03).

- VEX decoding, BMI1 and BMI2 done (see 03, "VEX and BMI").

- Speculation control, XSAVE and RDRAND done (see 03, "Speculation control" and "XSAVE and RDRAND"): guests see an in-order CPU immune to every speculative-execution bug (no retpolines, return thunks, KPTI or VERW), use `xsaveopt` for context switches, and seed the crng from `rdrand`/`rdseed`.

- WebAssembly port done (see 13): `crates/uemulator-web` + `hack/web.sh serve`; interpreter in a Web Worker with an xterm.js console, boots Alpine to a busybox shell in Chrome. No JIT, disk or network.

- UEFI application done (see 12): `crates/uemulator-uefi`, library `no_std` on `target_os = "uefi"`, `hack/uefi/boot.sh`; emulator (interpreter) everywhere, plus a hardware-virtualization path on x86_64 (AMD SVM verified under TCG, Intel VMX untested), no disk/net.

Each has an observable serial-output exit criterion. Interpreter first.

| # | deliverable | exit criterion |
|---|-------------|----------------|
| M0 | API skeleton, config, bzImage parse + loader, example downloads/caches kernel, clock + console + UART | `uemulator quick debian` downloads (once), loads, and reports the parsed header; `cargo test` passes |
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
