# JIT

## Scope

v1: **aarch64 host only** (macOS Apple Silicon, Linux aarch64). Other hosts run the interpreter. A second backend (x86_64 host) is a stretch goal behind the same `Backend` trait; not scheduled.

## Runtime probe (`jit/probe.rs`)

At `EmulatedMachine::new` when `JitMode != Off` and `cfg!(target_arch = "aarch64")`:
1. `mmap` a page (macOS: `MAP_JIT|MAP_PRIVATE|MAP_ANON`, prot RWX; Linux: RW then `mprotect` RX or RWX).
2. Write `ret` (`0xD65F03C0`), flush icache, call it, expect return.
3. Success → JIT available. Failure (any step) → `Auto` falls back to interpreter; `On` returns `Error::JitUnavailable`.

## Memory protection (macOS)

- Per-thread write protection: `pthread_jit_write_protect_np(0)` before writing code, `(1)` after, then `sys_icache_invalidate(ptr,len)`. Linux: `__builtin___clear_cache` equivalent via `libc` symbol `__clear_cache`.
- RAII `JitWrite` guard around translate-and-emit and chain-patch sites. The vCPU thread is the only writer.
- Take exact FFI signatures from the SDK headers (`pthread.h`, `libkern/OSCacheControl.h`), not from memory. Edition 2024: `unsafe extern "C" { ... }`.
- Hardened-runtime binaries need the `com.apple.security.cs.allow-jit` entitlement; document for embedders. Unsigned dev builds (cargo run) work with MAP_JIT.

## Code buffer

One region (default 64 MiB), bump allocator; TB structs and metadata live outside it (fork's `x-tblayout`). Full → flush all TBs and reset. Alignment 16 for TB entry points.

## Code generation (`jit/a64.rs`)

- Pinned registers: `x19 = &Env`, `x20 = tlb base`, `x21 = RAM host base` (only for physical-index fast path), `x22 = icount budget`; temps map to `x0–x15` by linear allocation per TB; spill to a small frame slot area when exhausted.
- Guest GPRs live in `Env`; v1 loads/stores them per uop. Optimization phase: cache hot guest regs in `x23–x28` within a TB and write back at exits.
- Prologue/epilogue trampoline (`enter_tb(env, code) -> exit_code`) saves callee-saved regs once; TBs are plain code that `ret`s to the trampoline or jumps to the next chained TB.
- Memory op fast path: compute `idx`, load tag, compare, `add` addend, `ldr/str`; slow path out-of-line calls `helper_ld/st(env, va, size, mmu_idx)` which may raise a guest exception. Helpers return a status; non-zero → `b epilogue_exception`. Faulting insns preceded by `str rip` (see 03).
- Flags: condition evaluation inlined for the common `cc_op` known at translate time (translator tracks static `cc_op` within the TB like QEMU, so `cmp; jcc` becomes `cmp; b.cond` or `subs`+`b.cond`-style code).
- Helpers: `extern "C" fn(env: *mut Env, ...) -> u64`; address loaded into a register and `blr`. Large immediate/address loads via `movz/movk` sequences.
- Branch ranges: ±128 MiB for `b`; the 64 MiB buffer keeps direct branches in range. Conditional branches ±1 MiB: use `b.cond` over a `b` when targets may exceed.
- Encodings: a small hand-written assembler module with unit tests that check bytes against known-good encodings. Reference: `tcg/aarch64/tcg-target.c.inc` for instruction forms (note the fork's "Correct register MIN/MAX instructions" fix); read-only.

## Chaining

`GotoTb(slot)`: emit `b <epilogue_stub>` initially; stub returns to the run loop with the slot; once the target TB exists the loop patches the `b` to jump directly (under `JitWrite`). Invalidation un-patches incoming links via per-TB incoming-link lists. Chained execution still observes the icount budget and `exit_request` (05).

## Exceptions and state sync

- Guest RIP is stored by `InsnStart` before any faulting uop; icount for the partial TB is recovered from a per-TB table `(host_pc_offset → insns_retired)` or, in v1, from a store of the retired-count alongside RIP.
- Exceptions inside helpers longjmp-free: helpers return an error code, generated code branches to the shared exception epilogue, which returns to the run loop; the loop delivers the exception using Env state.

## Future perf (post-milestone-4, from the fork)

jump-cache lookup inline in generated code (`x-tblookup`), cold-path separation (`x-tblayout`), inline MMIO fast paths (`x-iofast`), exit-request checks on cycle edges only (`x-edgesys`), slow-path thunks (`x-slowthunk`), SSE FP on the host FPU instead of softfloat (`x-ssefp`; aarch64 host FPU semantics differ in NaN/denormal handling, verify), guest-register caching, NEON for SSE2 integer ops.

## Fallback behavior

Any translation the JIT cannot handle (an uop without a lowering) → emit a call to an interpreter-equivalent helper for that uop so coverage grows incrementally; the differential harness (08) tracks lowered vs helper-backed uops.
