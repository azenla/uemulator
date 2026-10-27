# JIT

## Scope

Backends: **aarch64** (macOS Apple Silicon, Linux) and **x86_64** (Linux, macOS, Windows). Other hosts, and builds without the `jit` feature, run the interpreter. The feature gate is `jit` (default on): it enables the `libc` / `windows-sys` dependencies and every file under `jit/` and `sys.rs`.

## Runtime probe (`sys::jit_probe`)

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

## x86_64 backend (`jit/x64.rs`, `jit/emit_x64.rs`, `jit/arch_x64.rs`)

Same block shape and helper-fallback structure as the aarch64 backend (`jit/native.rs` is shared; `arch_*.rs` supplies the prologue, helper addresses and block compiler).

- Registers: `rbx` = Cpu, `rbp` = TLB base, `rax`/`rcx` = ALU operands, `rdx` = guest address, `rsi`/`rdi`/`r8`/`r9` scratch, `r12`/`r13` survive slow-path helper calls. The prologue saves `rbx rbp rsi rdi r12-r15` (Win64 callee-saved set) and reserves 56 bytes: 32 of Win64 shadow space plus an 8-byte load-result slot, keeping calls 16-byte aligned.
- ABI: SysV and Win64 differ only in the argument registers (`x64::ARGS`). Helpers take at most four arguments and never return a 16-byte struct (Win64 returns it by hidden pointer), so the x86 backend uses `h_ld_x`/`h_st_x`, which write the load result through a pointer; the block stores `cpu.rip` itself before calling them.
- Flags: guest and host flags agree for add/sub/logic, so a `jcc` right after an inlined ALU op tests the host flags directly (`Gen::live`). TLB probes and helper calls clobber them and end the window; effective addresses use `lea` only so they do not. Otherwise `jcc` calls `h_cond`.
- Memory: `mmap` RWX (`MAP_JIT` on macOS) or `VirtualAlloc(PAGE_EXECUTE_READWRITE)`; the x86 icache is coherent so no flush is needed.
- Tests: encoder tests in `x64.rs` execute generated code; the boot/devices tests and `--verify` exercise the whole backend. On an Apple Silicon host run them under Rosetta: `cargo test --release -p uemulator --target x86_64-apple-darwin`.

## Chaining

Implemented as a lookup stub rather than patched jumps (aarch64 now; see the x86_64 section for that backend). A block exit to a guest address (jcc, jmp, call, ret, indirect call/jmp, fallthrough) charges `Cpu::icount` (its unpaid instructions plus the successor's first, which `Cpu::run` would have charged) and branches to the `chain` stub in the prologue. The stub continues straight into the successor when `icount < Cpu::next_event`, `block_exit_requested` is clear, the exec TLB half the block runs in has the page (so the physical address is known without a walk) and the jump cache has a block for exactly that `(rip, phys)`. Otherwise it gives back the successor's pre-charge, stores `rip` and returns to the run loop, which does the usual event servicing, interrupt delivery, translation and re-entry. Invalidation clears jump cache slots, so a hit is never stale; nothing needs un-patching. Instructions that must be seen by the run loop (hlt, sti/cli, port I/O, mov to cr, syscall/iret...) always exit unchained. `--verify` and `JitMode`-less test harnesses that need one block per `step_block` build the `Jit` unchained.

## Code size and buffer (aarch64)

Translation cost scales with emitted host instructions and dominates a cold boot, so blocks are kept small: the slow path of every inline memory access is `adr x7, info; bl thunk; mov dst, x0; b resume`, with the thunks (loads by size and signedness, stores, stores that keep x0/x1, stack variants) shared in the prologue and the `[rip, block index]` pair next to the block; helpers are reached with one `bl` through tail-call stubs; `helper_exec_pair` takes its instruction pointer and rip from a literal pair addressed with `adr` (`Asm::literal_pair`, a pool flushed after the block). The code buffer is `SEGMENT_COUNT` segments of 96 MiB, each with its own prologue copy (aarch64 `b`/`bl` reach 128 MiB), so the ~500k blocks of a kernel plus installer boot are not flushed halfway. Block lookup is the jump cache backed by a `(rip, phys)` hash map: the cache is direct-mapped (64k slots), and when it was the only lookup every collision evicted a block and the next visit translated it again (2.8M translations for 535k distinct blocks in an Alpine boot).

## Lowered instructions (aarch64)

ALU add/or/and/sub/xor/cmp/test at 8/16/32/64 bits including read-modify-write memory forms (flags are recorded only after the store succeeded; x0/x1 are parked in callee-saved registers across the store's slow path), mov at every size including ah/ch/dh/bh, movzx/movsx, lea, inc/dec (CF carried in through `lazy_src`), neg/not, shl/shr/sar and rol/ror by imm/1/cl at 32/64 bits (shifts record `LazyFlagsOp::SzpCfOf`, rotates merge CF/OF with a helper because they keep SF/ZF/PF/AF), imul r,r/m and imm forms, setcc, cmovcc, cdq/cqo/cdqe/cwde, push/pop, push/call/jmp r/m64, near jcc/jmp/call/ret, lfence/mfence/sfence. adc/sbb, bt*, xchg/cmpxchg/xadd, division, lock-prefixed forms, SSE/x87 and string instructions go through `helper_exec` (forward `rep movs`/`rep stos` are bulk inside the interpreter, `Cpu::rep_bulk`). Byte and word ALU ops run in W registers on operands shifted to the top of the word, so the host NZCV flags are exact for the narrow operation and jcc can use them.

## Exceptions and state sync

- Guest RIP is stored by `InsnStart` before any faulting uop; icount for the partial TB is recovered from a per-TB table `(host_pc_offset → insns_retired)` or, in v1, from a store of the retired-count alongside RIP.
- Exceptions inside helpers longjmp-free: helpers return an error code, generated code branches to the shared exception epilogue, which returns to the run loop; the loop delivers the exception using Env state.

## Future perf (post-milestone-4, from the fork)

jump-cache lookup inline in generated code (`x-tblookup`), cold-path separation (`x-tblayout`), inline MMIO fast paths (`x-iofast`), exit-request checks on cycle edges only (`x-edgesys`), slow-path thunks (`x-slowthunk`), SSE FP on the host FPU instead of softfloat (`x-ssefp`; aarch64 host FPU semantics differ in NaN/denormal handling, verify), guest-register caching, NEON for SSE2 integer ops.

## Fallback behavior

Any translation the JIT cannot handle (an uop without a lowering) → emit a call to an interpreter-equivalent helper for that uop so coverage grows incrementally; the differential harness (08) tracks lowered vs helper-backed uops.
