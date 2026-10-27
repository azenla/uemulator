# CPU

## State (`Env`, `#[repr(C)]`, pointed to by a pinned host register in the JIT)

- 16 GPRs (u64), `rip`, `rflags` (without lazy bits), segment regs (selector, base, limit, attrs) incl. hidden parts, GDTR/IDTR/LDTR/TR
- CR0, CR2, CR3, CR4, CR8 (stored only), EFER, STAR/LSTAR/CSTAR/SFMASK, KERNEL_GS_BASE, FS/GS base, TSC_AUX, PAT, APIC base (stub), MTRR msrs (stored)
- lazy flags: `lazy_op`, `lazy_size`, `lazy_dst`, `lazy_src`, `lazy_src2` (QEMU calls these `cc_*`)
- x87: 8×80-bit regs (stored as raw 10 bytes + software softfloat ops), FCW/FSW/FTW, MXCSR, XMM0–15 (u128)
- pending interrupt state, `halted`, `exit_request`, icount, soft TLB arrays (see 04)
- hflags cache: CPL, code size (16/32/64), addseg, SS32, protected/real, used in TB key

## Decoder (`decode.rs`, `tables.rs`)

Table-driven, one-byte, 0F, 0F38, 0F3A maps; prefix handling (66, 67, F2, F3, REX, segment overrides, LOCK); ModRM/SIB/disp/imm; RIP-relative; operand size and address size per mode. Table entry: mnemonic class, operand specs, flags-effect class, whether it can fault. Output: `Insn { len, op, operands, ... }` consumed by `translate.rs` which emits uops.

Coverage priority (what Linux + Debian userspace uses):
1. Integer: mov/movzx/movsx(d), lea, add/or/adc/sbb/and/sub/xor/cmp/test, inc/dec/neg/not, mul/imul/div/idiv, shifts/rotates/shld/shrd, bt*/bsf/bsr/popcnt/lzcnt/tzcnt (only if advertised), setcc/cmovcc/jcc, push/pop, call/ret/leave/enter, jmp, loop*, string ops with rep (movs/stos/lods/scas/cmps), xchg/cmpxchg/cmpxchg16b/xadd (lock is a no-op), cdq/cqo/cbw family, bswap, cpuid, rdtsc, rdtscp?, rdmsr/wrmsr, syscall/sysret, sysenter(no), swapgs, int/int3/iret(q), hlt, cli/sti, in/out/ins/outs, lgdt/lidt/ltr/lldt/sgdt/sidt/str, mov cr/dr, invlpg, wbinvd (nop), lfence/mfence/sfence, clflush, pause, ud2, xgetbv (not advertised; #UD).
2. SSE/SSE2 integer and scalar/packed FP used by libc memcpy/strlen/etc: movdqa/u, movq/d, movaps/ups, pxor/por/pand/pandn, padd/psub, pcmpeq/gt, pmovmskb, pshufd, punpck*, packss/usb, psrl/psll/psra, shufps, paddq, pmuludq, cvt*, add/sub/mul/div/sqrt/min/max ss/sd/ps/pd, comisd/ucomisd, movmsk. SSSE3/SSE4 only if CPUID advertises them (start without).
3. x87: fninit, fld/fst/fstp (m32/m64/m80, st(i)), fild/fistp, fadd/fsub/fmul/fdiv family, fxch, fchs/fabs, fcom/fucom(i)(p), fnstcw/fldcw/fnstsw, fxsave/fxrstor, fwait; transcendentals later via host libm with documented precision loss.
4. Unimplemented opcode → #UD, with a one-shot log through the log hook (opcode bytes + rip).

0F 18–1F: NOP hint space (prefetch, ENDBR64 `F3 0F 1E FA`, multi-byte nops).

## Lazy flags

CC_OP-style: ALU ops store `(op, dst, src[, src2])`; flags are materialized when consumed (jcc/setcc/cmov/pushf/adc/sbb/rcl/...) or when the op class changes. Reuse the QEMU scheme conceptually (`cc_helper*`): ops per width (B/W/L/Q) for ADD, ADC, SUB, SBB, LOGIC, INC, DEC, SHL, SHR, SAR, MUL, ADCX/OX (later), plus EFLAGS (materialized) and CLR. Fork note: INC/DEC keep carry via `lazy_src`, a cheap win.

Interpreter and JIT both call the same `flags::compute_all(cc_op, dst, src, src2)`; the JIT inlines the hot cases (ZF/SF/CF from SUB/CMP/TEST/AND followed by jcc).

## Exactness rules

- An instruction commits GPR/flag writes only after all its faulting loads/stores; pushes and pops compute RSP in a temp and write it last; string ops commit per iteration (rep is interruptible, RIP stays on the instruction until RCX hits 0).
- Every faulting uop sequence is preceded by an `insn_start(rip)` that stores guest RIP to `Env` (JIT: one `str`; coalesced when there are no faulting ops in between). No unwind tables in v1.
- Exception delivery (`exception.rs`): long mode IDT gate lookup, IST, TSS rsp0, error code push, privilege change stack switch, #DF escalation, triple fault → exit. Also real-address and protected-mode 32-bit delivery only as far as the decompressor needs.
- `iretq`, `sysretq`, far calls/jumps through call gates (not needed) — implement iret and far jmp/ret with descriptor loads in long mode.
- Segment loads in 64-bit mode: validate and cache base/limit/attrs for CS/SS/DS/ES; FS/GS use MSR bases (`wrfsbase` not advertised).
- Page faults: CR2 + error code (P/W/U/RSVD/I), fixups via kernel `ex_table` work because RIP is exact.
- `rdmsr`/`wrmsr` on unknown MSR → #GP (kernel uses `rdmsr_safe`).

## CPUID/MSR contract

Start from the fork's identity (read `target/i386/cpu.c` "XTCG CPU" model for the exact leaves) and then **subtract** to what we implement. Initial advertised set:

- Leaf 0/1: vendor AuthenticAMD (fork) family 0x15 model 0, FPU, VME no, PSE, TSC, MSR, PAE, CX8, SEP no (syscall only), MTRR, PGE, CMOV, PAT, PSE36 no, CLFLUSH, MMX, FXSR, SSE, SSE2; **APIC clear**, SSE3 later after implementing, **hypervisor bit clear**
- Leaf 7: nothing (no FSGSBASE, SMEP, SMAP, AVX2, ERMS, PCID, LA57)
- Leaf 0x80000001: SYSCALL, NX, LM, (RDTSCP only if implemented)
- Leaf 0x80000007: invariant TSC set so the kernel trusts the TSC
- No XSAVE/OSXSAVE/AVX/RDRAND (kernel then uses FXSAVE/FXRSTOR and falls back for random)

Advertising a bit is a promise; add bits only together with the test that proves the instructions.
