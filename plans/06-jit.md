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

- Registers: `rbx` = Cpu, `rbp` = TLB base, `rax`/`rcx` = ALU operands (rcx doubles as shift count and carries `lazy_src`), `rdx` = guest address, `rsi`/`rdi`/`r8`/`r10` scratch, `r9` = value moved by mov-style instructions, `r11` = call target. `r12`/`r13` survive slow-path helper calls (address, store value); `r14`/`r15` park the result and second operand of a read-modify-write across its store's slow path, and an inc/dec carry-in or cmov condition across the operand's memory access. The prologue saves `rbx rbp rsi rdi r12-r15` (Win64 callee-saved set) and reserves 56 bytes: 32 of Win64 shadow space plus an 8-byte load-result slot, keeping calls 16-byte aligned.
- ABI: SysV and Win64 differ only in the argument registers (`x64::ARGS`). Helpers take at most four arguments and never return a 16-byte struct (Win64 returns it by hidden pointer), so the x86 backend uses `h_ld_x`/`h_st_x`, which write the load result through a pointer; the block stores `cpu.rip` itself before calling them.
- Lowering: the same set of forms as the aarch64 backend (`emit.rs`, which the translation-statistics tests mirror): ALU/test at 8/16/32/64 bits including read-modify-write, mov at every size, inc/dec/neg/not, shl/shr/sar at 32/64 bits, imul r,r/m(,imm), setcc/cmovcc, cdq family, push/call/jmp r/m64, fences. Guest registers live in `Cpu::gpr`; byte/word writes (al, ax, ah) are narrow stores into the little-endian slot, so they do not disturb host flags.
- Flags: guest and host flags agree for add/sub/logic/inc/dec/neg at every width (the host op runs at the operand width), so jcc/setcc/cmovcc right after one tests the host flags directly (`Gen::live`), all 16 conditions. `inc`/`dec` re-establish host CF from the carry-in (`bt`) because the operand's memory access loses it. TLB probes, helper calls, the SMC check and `chain_exit` clobber host flags (effective addresses use `lea` only), so any instruction that stores to memory after computing its flags ends the window and the lazy flags (recorded only after the store succeeded) are the truth. Shifts and `imul` record `LazyFlagsOp::SzpCfOf`: host OF/SF/ZF are undefined or only defined for count 1, so CF and OF are worked out explicitly (as `Cpu::shift_rotate`/`imul_truncated` do) and the host flags are dead afterwards. Otherwise a condition calls `helper_cond`.
- Chaining: `arch_x64::chain_stub` (in the prologue, one copy per segment) is the `lookup_and_goto_ptr` equivalent, entered with rax = icount (already charged for the successor's first instruction) and rcx = rip. It checks `next_event`, `block_exit_requested`, the exec TLB (`tag_x`, `addend`, minus `ram_base`) and the jump cache (`JumpEntry`), and jumps into the successor or gives the pre-charge back and exits with count 1 and `Cpu::rip` set. `Gen::chain_exit` mirrors the aarch64 accounting (`retired - prepaid_insns`, `prepaid_icount` cleared). `Addrs::chain == 0` (verify mode) means every exit returns to the run loop.
- Memory: `mmap` RWX (`MAP_JIT` on macOS) or `VirtualAlloc(PAGE_EXECUTE_READWRITE)`; the x86 icache is coherent so no flush is needed.
- Tests: encoder tests in `x64.rs` execute generated code; the boot/devices tests and `--verify` exercise the whole backend. On an Apple Silicon host run them under Rosetta: `cargo test --release -p uemulator --target x86_64-apple-darwin`.

## Chaining

A block exit to a guest address (jcc, jmp, call, ret, indirect call/jmp, fallthrough) charges `Cpu::icount` (its unpaid instructions plus the successor's first, which `Cpu::run` would have charged) and branches to the `chain` stub in the prologue. The stub continues straight into the successor when `icount < Cpu::next_event`, `block_exit_requested` is clear, the exec TLB half the block runs in has the page (so the physical address is known without a walk) and the jump cache has a block for exactly that `(rip, phys)`. Otherwise it gives back the successor's pre-charge, stores `rip` and returns to the run loop, which does the usual event servicing, interrupt delivery, translation and re-entry. Invalidation clears jump cache slots, so a hit is never stale. Instructions that must be seen by the run loop (hlt, sti/cli, port I/O, mov to cr, syscall/iret...) always exit unchained. `--verify` and `JitMode`-less test harnesses that need one block per `step_block` build the `Jit` unchained.

**Direct links** (`plans/20-jit-perf.md`): an exit to a fixed address in the block's own page also has a patchable slot (`Gen::link_slot`: a nop of `arch::link_bytes` size right before the `mov rip; b chain` sequence). `Jit::translate` turns it into a direct `b`/`jmp` to the successor's *link entry* once the successor exists (`Jit::pending` holds slots waiting for a block, `Jit::incoming` the patched ones per destination). Every block starts with its link entry, which repeats the stub's `next_event` and `block_exit_requested` checks and falls into the body; the run loop, the stub and the jump cache enter at the body (`Block::code_addr`) because they have already checked. Same page is enough to skip the TLB probe and the jump-cache key check: the source was entered through a validated mapping of the page and the destination shares it. `invalidate_page` writes the nop back into every slot that jumps into an invalidated block and queues it for the block's next translation; `reset` drops both tables. Indirect exits, cross-page targets and page-straddling blocks keep the stub.

## Code size and buffer (aarch64)

Translation cost scales with emitted host instructions and dominates a cold boot, so blocks are kept small: the slow path of every inline memory access is `adr x7, info; bl thunk; mov dst, x0; b resume`, with the thunks (loads by size and signedness, stores, stores that keep x0/x1, stack variants) shared in the prologue and the `[rip, block index]` pair next to the block; helpers are reached with one `bl` through tail-call stubs; `helper_exec_pair` takes its instruction pointer and rip from a literal pair addressed with `adr` (`Asm::literal_pair`, a pool flushed after the block). The code buffer is `SEGMENT_COUNT` segments of 96 MiB, each with its own prologue copy (aarch64 `b`/`bl` reach 128 MiB), so the ~500k blocks of a kernel plus installer boot are not flushed halfway. Block lookup is the jump cache backed by a `(rip, phys)` hash map: the cache is direct-mapped (64k slots), and when it was the only lookup every collision evicted a block and the next visit translated it again (2.8M translations for 535k distinct blocks in an Alpine boot).

## Lowered instructions (aarch64)

ALU add/or/and/sub/xor/cmp/test at 8/16/32/64 bits including read-modify-write memory forms (flags are recorded only after the store succeeded; x0/x1 are parked in callee-saved registers across the store's slow path), mov at every size including ah/ch/dh/bh, movzx/movsx, lea, inc/dec (CF carried in through `lazy_src`), neg/not, shl/shr/sar by imm/1/cl at 8/16/32/64 bits (an immediate count of zero or at least the width of a narrow operand stays in the interpreter) and rol/ror at 32/64 bits (shifts record `LazyFlagsOp::SzpCfOf`, rotates merge CF/OF with a helper because they keep SF/ZF/PF/AF), imul r,r/m and imm forms, setcc, cmovcc, cdq/cqo/cdqe/cwde, push/pop, push/call/jmp r/m64, near jcc/jmp/call/ret, lfence/mfence/sfence, adc/sbb at 32/64 bits (host C is set from the guest CF and `adcs`/`sbcs` run; the lazy record carries the carry-in in `lazy_src2`; x86_64 lowers register destinations only), and the VEX BMI shlx/shrx/sarx, bzhi, andn and blsr/blsmsk/blsi. Lazy-flag records that a later register-only writer overwrites before anything can look (a reader, a possible fault, a helper, the block's end) are not stored (`helpers::flag_record_needed`). bt*, bextr/pdep/pext, tzcnt/lzcnt/bsf/bsr, mul, xchg/cmpxchg/xadd, division, lock-prefixed forms, SSE/x87 and string instructions go through `helper_exec` (forward `rep movs`/`rep stos` are bulk inside the interpreter, `Cpu::rep_bulk`). Byte and word ALU ops run in W registers on operands shifted to the top of the word, so the host NZCV flags are exact for the narrow operation and jcc can use them.

## Several vCPUs

With one host thread per vCPU (`plans/22-smp-threads.md`) each vCPU has its own `Jit`: its own code buffer, block cache and chain links, so no thread patches code another is running, and a vCPU that has not run yet owns none. Stale translations are dropped through the shared invalidation log, and on an aarch64 host the generator emits `ldar`/`stlr` for guest loads and stores (`Jit::set_ordered_memory`, `Gen::ordered_memory`; a misaligned access takes the slow path since those faults). When time-sharing, the JIT, its block cache and the code buffer are machine-wide: `Cpu::swap_shared` moves the `Box<Jit>` to the vCPU that runs next, and generated code only addresses state through the `Cpu` pointer in its pinned register, so nothing is duplicated or retagged. Blocks are keyed by (rip, physical address) as before. Two things keep the sharing correct: a store by any vCPU to a code page runs `code_written` on the shared cache (and a vCPU's private TLB loses stale write tags when it is switched in), and a quantum ends through the same `next_event` / `block_exit_requested` checks chained blocks already make (`pause` sets both, so the running block stops at the instruction after it). Generated code needs no change for SMP; only 64-bit code is compiled, so real-mode start-up code of the other vCPUs runs on the interpreter.

## Exceptions and state sync

- Guest RIP is stored by `InsnStart` before any faulting uop; icount for the partial TB is recovered from a per-TB table `(host_pc_offset → insns_retired)` or, in v1, from a store of the retired-count alongside RIP.
- Exceptions inside helpers longjmp-free: helpers return an error code, generated code branches to the shared exception epilogue, which returns to the run loop; the loop delivers the exception using Env state.

## Future perf (post-milestone-4, from the fork)

jump-cache lookup inline in generated code (`x-tblookup`), cold-path separation (`x-tblayout`), inline MMIO fast paths (`x-iofast`), exit-request checks on cycle edges only (`x-edgesys`), slow-path thunks (`x-slowthunk`), SSE FP on the host FPU instead of softfloat (`x-ssefp`; aarch64 host FPU semantics differ in NaN/denormal handling, verify), guest-register caching, NEON for SSE2 integer ops.

## Fallback behavior

Any translation the JIT cannot handle (an uop without a lowering) → emit a call to an interpreter-equivalent helper for that uop so coverage grows incrementally; the differential harness (08) tracks lowered vs helper-backed uops.
