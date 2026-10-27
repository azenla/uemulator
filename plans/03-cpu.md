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
2. SSE/SSE2 integer and scalar/packed FP used by libc memcpy/strlen/etc: movdqa/u, movq/d, movaps/ups, pxor/por/pand/pandn, padd/psub, pcmpeq/gt, pmovmskb, pshufd, punpck*, packss/usb, psrl/psll/psra, shufps, paddq, pmuludq, cvt*, add/sub/mul/div/sqrt/min/max ss/sd/ps/pd, comisd/ucomisd, movmsk. SSE3, SSSE3 (XMM forms only), SSE4.1, SSE4.2, PCLMULQDQ and AES-NI are implemented in `cpu/sse_ext.rs` (see "Instruction extensions" below).
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
- Segment loads in 64-bit mode: validate and cache base/limit/attrs for CS/SS/DS/ES; FS/GS use MSR bases (or `rdfsbase`/`wrfsbase` and the GS pair, which need CR4.FSGSBASE).
- Page faults: CR2 + error code (P/W/U/RSVD/I), fixups via kernel `ex_table` work because RIP is exact.
- `rdmsr`/`wrmsr` on unknown MSR → #GP (kernel uses `rdmsr_safe`).

## CPUID/MSR contract

Start from the fork's identity (read `target/i386/cpu.c` "XTCG CPU" model for the exact leaves) and then **subtract** to what we implement. Initial advertised set:

- Leaf 0/1: vendor AuthenticAMD family 0x14 (the fork says 0x15, but Linux blacklists 0x15 for RETBLEED and installs return thunks; see "Speculation control") model 0, FPU, VME no, PSE, TSC, MSR, PAE, CX8, SEP no (syscall only), MTRR, PGE, CMOV, PAT, PSE36 no, CLFLUSH, MMX, FXSR, SSE, SSE2; **APIC clear**, SSE3 later after implementing, **hypervisor bit clear**
- Leaf 7 subleaf 0: only FSGSBASE (EBX bit 0); EAX = 0. No SMEP, SMAP, AVX2, BMI, ERMS/FSRM, PCID, LA57
- Leaf 1 ECX: SSE3, PCLMULQDQ, SSSE3, CX16, SSE4.1, SSE4.2, MOVBE, POPCNT, AES; 0x80000001 ECX adds ABM (LZCNT). MONITOR/MWAIT, VMX/SVM, XSAVE/OSXSAVE, AVX, RDRAND and the hypervisor bit stay clear
- Leaf 0x80000001: SYSCALL, NX, LM, (RDTSCP)
- Leaf 0x80000007: invariant TSC set so the kernel trusts the TSC
- XSAVE/OSXSAVE (x87, SSE and AVX components), RDRAND/RDSEED, AVX, AVX2, FMA, F16C are advertised, see "XSAVE and RDRAND" and "AVX, AVX2, FMA, F16C"

Advertising a bit is a promise; add bits only together with the test that proves the instructions.

## Instruction extensions

Dispatch: `Cpu::sse` hands everything outside the `0F` map (0F 38, 0F 3A) and every `0F` opcode it does not know to `sse::ext` (`sse_ext.rs`, a child module that reuses the SSE2 soft-float engine and lane helpers). `Op` is the decoded operation, `Source` how the second operand is read (aligned/unaligned m128, a low part, none), and the same prologue applies the SSE enable rules (#UD for CR0.EM or CR4.OSFXSR clear, then #NM for CR0.TS), the 16-byte alignment #GP and the mandatory-prefix check; an invalid ModRM form is #UD even under CR0.TS. Pure helpers: `string_compare.rs` (pcmp[ei]str[im]: SDM invalid-element table, all aggregations and polarities) and `crypto.rs` (CRC-32C table, AES S-box generated from its GF(2^8) definition, rounds, carry-less multiply).

- Integer, in `exec.rs`: POPCNT (F3 0F B8), LZCNT (F3 0F BD; plain BSR without the prefix, so the ABM bit and the arm go together), CMPXCHG16B (REX.W 0F C7 /1, #GP(0) unless 16-byte aligned, always writes), and in `sse_ext.rs` CRC32 (F2 0F 38 F0/F1) and MOVBE (0F 38 F0/F1). TZCNT (F3 0F BC) is BSF without the prefix, so it and the BMI1 bit go together (see "VEX and BMI").
- SSE3: movddup, movsldup, movshdup, lddqu, addsub/hadd/hsub (ps, pd), x87 fisttp (m16/m32/m64).
- SSSE3: all the XMM forms. The MMX-register forms (no 66 prefix) are deliberately #UD (logged once as unimplemented): glibc and the kernel only use the XMM forms, and each MMX form needs its own lane layout. The CPUID bit describes the XMM forms.
- SSE4.1: blends (immediate and xmm0-implicit), ptest, pmovsx/zx, pmuldq/pmulld, pmin/pmax, pextr/pinsr/extractps/insertps, round{ps,pd,ss,sd} (immediate mode, MXCSR.RC, precision suppression, SNaN quieting, DAZ), dpps/dppd (sum order `(p0+p1)+(p2+p3)`), mpsadbw, phminposuw, packusdw, pcmpeqq, movntdqa.
- SSE4.2: pcmpgtq, CRC32, pcmpestri/pcmpestrm/pcmpistri/pcmpistrm (all imm8 modes, REX.W lengths).
- PCLMULQDQ and AES-NI (aesenc/last, aesdec/last, aesimc, aeskeygenassist), tested with FIPS-197 appendix B and C.1 end to end.
- FSGSBASE: F3 0F AE /0../3 register forms, gated by CR4.FSGSBASE (bit 16, accepted by mov cr4), canonical check on writes.

Left out on purpose: ERMS/FSRM (the kernel would switch memcpy and clear_page to `rep movsb`/`stosb`, which `string_op` runs one byte per iteration instead of eight: advertise only together with a bulk fast path there), MONITOR/MWAIT, AVX/AVX2/FMA/F16C (VEX vector forms). The JIT runs all of the above through `helper_exec` (none is a block ender); `testkit::assert_jit_matches_interpreter` cases cover representative forms. Several encodings are prefix-dependent (F3 0F B8 POPCNT vs JMPE, F3 0F BD LZCNT vs BSR, F3 0F BC still BSF, 0F 38 F0/F1 MOVBE vs CRC32 by F2, F3 0F AE FSGSBASE vs the fences): any inline JIT lowering of the neighbouring opcodes must check `insn.rep` and `insn.p66`, or it silently diverges from the interpreter.


## Speculation control (mitigations)

The CPU is in-order and never speculates, so it says so, to keep the guest kernel from running mitigation code the emulator would have to interpret:

- Leaf 7 EDX: SPEC_CTRL, STIBP, L1D_FLUSH, ARCH_CAPABILITIES, SSBD; leaf 0x80000008 EBX: IBPB, IBRS, STIBP, SSBD. MSRs: `IA32_SPEC_CTRL` (0x48, bits 0-2 stored, others #GP), `IA32_PRED_CMD` (0x49) and `IA32_FLUSH_CMD` (0x10b) write-only with bit 0 only, `IA32_ARCH_CAPABILITIES` (0x10a, read-only) with every `*_NO` bit Linux 6.18 checks plus IBRS_ALL (so Spectre v2 resolves to one-time "Enhanced IBRS", no retpolines) and BHI_NO (no BHB clearing loop on syscall entry). RSBA stays clear.
- Family 0x14 rather than 0x15: `VULNBL_AMD` matches families 0x15-0x17,0x19,0x1a by number for RETBLEED/SRSO regardless of ARCH_CAPABILITIES, and the answer is an untrained-return thunk on every `ret`.
- Check (Alpine 6.18 guest, `rdinit=/bin/sh`, cmdline without `mitigations=off`): every file in `/sys/devices/system/cpu/vulnerabilities` reads "Not affected" except `spectre_v2` ("Enhanced / Automatic IBRS", a single MSR write at boot) and `spectre_v1` (usercopy barriers). Before this change retbleed selected "untrained return thunk" and spectre_v2 was "Vulnerable" only because the quick cmdline passes `mitigations=off`.

## XSAVE and RDRAND

`cpu/xsave.rs`. The only state components are x87 (XCR0 bit 0) and SSE (bit 1), so the XSAVE area is the 512-byte fxsave image plus the 64-byte header (576 bytes, standard format; CPUID leaf 0xD reports it).

- `xsetbv`/`xgetbv` (0F 01 D1/D0): `Cpu::xcr0`, #UD without CR4.OSXSAVE (bit 18, accepted by `mov cr4`, mirrored into CPUID.1:ECX bit 27 when read), `xsetbv` CPL 0 only and #GP for ECX != 0, x87 clear, or components beyond `XCR0_SUPPORTED`. Extending that mask means extending leaf 0xD and the area layout too.
- `xsave`/`xsaveopt` (0F AE /4, /6) and `xrstor` (/5): #UD, then #NM (CR0.TS), then #GP(0) for a non-64-byte-aligned area. The request bitmap is EDX:EAX masked by XCR0. Saved components are always marked in use in XSTATE_BV, so `xsaveopt` is `xsave` (software can only see the difference through XSTATE_BV). `xrstor` validates XSTATE_BV, XCOMP_BV (compaction is not supported, so must be 0) and the reserved header bytes, then reads everything before committing; a component in the bitmap with its XSTATE_BV bit clear is reset to its initial state, MXCSR is read whenever SSE is in the bitmap. XSAVEC/XSAVES/XGETBV1 are not advertised, so Linux uses the standard format with xsaveopt (checked: "context size is 576 bytes, using 'standard' format"). fxsave/fxrstor share `save_legacy_region`/`restore_legacy_region` in `fpu.rs`.
- `rdrand`/`rdseed` (0F C7 /6, /7, register forms): always CF=1, other arithmetic flags clear. The generator is splitmix64 in `Cpu::rng_state`, seeded by `MachineConfig::random_seed` (the CLI passes host entropy; the library default is a fixed constant, which is right for tests and `--verify` because the twin interpreter copies the state, but wrong for a guest that generates keys). Not cryptographic; the kernel mixes it with other sources. Side effect: the guest's crng is initialised at boot (`random: crng init done` at 0.05 s) instead of waiting for interrupt entropy.

## VEX and BMI

`decode` understands `c4`/`c5` (always VEX in 64-bit mode; `62` EVEX is still invalid). The result is an ordinary `Insn` with `vex`, `vex_l` and `vex_v` (un-inverted vvvv) set, R/X/B/W folded into `rex` (with `rex_present` false), `pp` in `p66`/`rep`, `opsize` from VEX.W only, and `op = VEX_OP_FLAG (0x1000) | (map << 8) | opcode`. The flag keeps VEX opcodes from matching any legacy `op` arm in the interpreter or the JIT (vex `0F38 F6` is mulx, legacy `66 0F 38 F6` is adcx; vex `0F 10` is vmovups, legacy is movups); `exec` hands `vex` instructions to `exec_vex` before the legacy dispatch, and the JIT, which only matches exact legacy ops, runs them through its interpreter helper. A legacy 66/F2/F3/F0/REX prefix in front of VEX, or a map outside 1-3, is #UD.

`cpu/bmi.rs` (CPUID.7.0:EBX bits 3 and 8): andn, bextr, blsi, blsmsk, blsr, bzhi, mulx, pdep, pext, rorx, sarx, shlx, shrx plus tzcnt in `exec.rs`. They are LZ (VEX.L set is #UD); rorx also needs vvvv = 1111. Flag behavior: andn/blsr/blsmsk/blsi/bzhi set ZF, SF, PF from the result and clear OF; CF is "source is zero" (blsr, blsmsk), "source is non-zero" (blsi) and "index > width-1" (bzhi); bextr clears CF/OF and sets ZF only; mulx, pdep, pext, rorx and the shifts leave all flags alone. Shift counts are masked to width-1, and mulx writes the low half first so the high half wins when both destinations are the same register. AF/PF are architecturally undefined for the ones that set flags; ours are 0/computed as noted. Tests (`bmi.rs`) compare against independent reference models over a spread of edge values, and `jit/diff_alu.rs::bmi_instructions` pins JIT = interpreter.

The vector VEX forms are in "AVX, AVX2, FMA, F16C" below.

## RDTSCP

`0F 01 F9`, advertised in 0x80000001 EDX bit 27: EDX:EAX = TSC (same CR4.TSD rule as rdtsc), ECX = IA32_TSC_AUX. Linux uses it for `rdpid`-less CPU/node lookup in the vDSO.

## UMIP, SMEP/SMAP, PCID/INVPCID, MONITOR/MWAIT

`cpu/protection.rs` plus the page walk in `mem.rs`.

- UMIP (CR4 bit 11, CPUID.7.0:ECX bit 2): sgdt, sidt, sldt, str, smsw are #GP(0) at CPL > 0.
- SMEP/SMAP (CR4 bits 20/21, CPUID.7.0:EBX bits 7/20): in `walk`, a supervisor-mode access (`!user`) to a page that is user-accessible at every level is #PF (P=1; the fetch bit is reported when NX *or* SMEP is on) for fetches under SMEP and for data accesses under SMAP while EFLAGS.AC is clear. `stac`/`clac` (0F 01 CB/CA) set/clear AC and are #UD outside CPL 0. **Such a translation is never put in the soft TLB** while SMEP or SMAP is on: AC changes (stac, clac, popf) without flushing, so a cached entry would let a later access skip the check. The kernel's user-copy paths are the only accesses affected, and they were slow-path walks anyway. Implicit supervisor accesses (descriptor tables) cannot be told from explicit ones inside `walk`, so they get the AC waiver too; no kernel relies on the difference.
- PCID/INVPCID (CR4.PCIDE bit 17, CPUID.1:ECX bit 17, CPUID.7.0:EBX bit 10): CR3[11:0] holds the PCID and bit 63 is accepted and ignored. The TLB is untagged, so every CR3 write and every INVPCID type flushes everything (always a legal superset); operand validation follows the SDM. PCIDE can only be set while CR3[11:0] is zero.
- MONITOR/MWAIT (CPUID.1:ECX bit 3, leaf 5 = 64-byte lines, extensions + interrupt-break): CPL 0, ECX/EDX checked. `mwait` after a `monitor` behaves like `hlt` with interrupts enabled (only interrupts wake it) and returns immediately with IF=0, which hardware may also do; the armed address is not tracked because nothing but DMA can write guest memory while the single CPU waits.
- Linux 6.18 guest: "x86/cpu: User Mode Instruction Prevention (UMIP) activated"; `smep smap umip pcid invpcid monitor` appear in /proc/cpuinfo; lockstep boot is divergence-free.
- Found by the lockstep verifier: `rdtscp` (group 7) reads guest time, so `jit::helpers::reads_guest_clock` now includes the whole group, otherwise the JIT read a stale `icount` mid-block.

## AVX, AVX2, FMA, F16C

Files: `avx.rs` (the wrapper and its table), `avx_special.rs` (everything else), `sse_fma.rs` (FMA, F16C and the 32 compare predicates, a child of `sse` because it needs the soft-float internals), tests in `avx_tests.rs`, `avx_special_tests.rs`, `avx_cases.rs` / `avx_special_encodings.rs` (encodings generated from the assembler; regenerate with the scripts' recipe: assemble each mnemonic with `hack/asm.sh`).

State: `Cpu::ymm_hi: [u128; 16]` holds bits 255:128; legacy SSE leaves it alone, VEX.128 zeroes it. XCR0 bit 2 enables it (`XCR0_SUPPORTED` is now 7, AVX needs SSE); XSAVE component 2 is 256 bytes at offset 576 (area 832 bytes, CPUID.0xD.2 = {256, 576, 0, 0}), saved by xsave/xsaveopt and loaded or reset by xrstor. `fxsave` does not touch it, as on hardware. `Cpu::diff` compares it, so `--verify` covers it.

Gate (`vex_gate`): #UD unless CR4.OSXSAVE and XCR0[2:1] = 11, then #NM under CR0.TS. CR4.OSFXSR does not matter for VEX.

Wrapper (`exec_vex_vector`): most VEX instructions are an SSE instruction plus a non-destructive first source, so they run through the existing SSE kernels. `vex_form` classifies each (map, opcode, pp) as: `Merge` (is the destination seeded from vvvv: always, never, or only for the register form as in vmovss), `Dest` (reg, rm, or no vector destination) and `Width` (`Pair`: 128 bits, or two lane passes with VEX.L; `Scalar`: VEX.L ignored; `Xmm`: VEX.L is #UD). A pass seeds `xmm[dest]` from `xmm[vvvv]`, calls `sse()` with `vex_active` set (which drops the OSFXSR gate and the 16-byte alignment rule, except for the explicit vmovaps/vmovapd/vmovdqa/vmovntdqa loads), and for the upper lane swaps `xmm` with `ymm_hi` and adds 16 to the displacement. Both lanes run on a snapshot, so a fault in the second leaves nothing written. **Collisions**: seeding overwrites the destination, so if the kernel still has to read it as its other operand (`vpcmpeqb ymm1, ymm0, ymm1`, the destination is also the r/m source), that operand is redirected to a scratch register first. This was found by booting Debian: glibc's AVX2 string functions use that form constantly, and a wrong remap broke `strcmp` ("applet not found" in busybox). The plumbing tests now cover it. Explicitly aligned 256-bit moves need 32-byte alignment (checked once before the passes).

`avx_special.rs`: movmskps/pd, pmovmskb (256-bit), cvtps2pd/cvtdq2pd (widening) and cvtpd2ps/cvt(t)pd2dq (narrowing, each half through the legacy kernel), shifts by xmm count (count is the low quadword of an xmm/m128, shared by both lanes) and by immediate (destination is vvvv), vptest/vtestps/vtestpd, vpermilps/pd (variable and immediate), vpermq/vpermpd/vpermps/vpermd, vperm2f128/i128, vinsert/vextract f128/i128, all the broadcasts, pmovsx/zx (the 256-bit forms widen twice as many elements, memory operands read exactly the bytes they use), vblendv* (mask register in imm8[7:4]), vpblendd, vpsllv/vpsrlv/vpsravd, vmaskmovps/pd and vpmaskmovd/q (masked-out elements are never read, written or faulted on), vldmxcsr/vstmxcsr, vzeroupper/vzeroall, vpcmpistrm/vpcmpestrm (also zero the upper half of xmm0), and the gathers.

Gathers (VSIB): the decoder keeps SIB index 4 for the 0F38 90-93 opcodes (it is xmm4, not "no index"). The mask is normalised to all-ones/all-zeros per element, each completed element clears its mask element, and a fault writes the partial registers back before reporting (restartable, as on hardware); on completion the mask register is zero and everything above the gathered elements is zeroed. #UD when destination, mask and index registers are not all different.

`sse_fma.rs`: FMA3 (132/213/231 of madd/msub/nmadd/nmsub, packed and scalar, and maddsub/msubadd) rounds once: the exact product of the two significands (at most 106 bits) is one term of `add_terms`, the same routine the legacy adder uses, so DAZ/FTZ/rounding mode/flags behave like the other soft-float instructions. NaN selection takes the first NaN in operand order (destination, vvvv, r/m), quieted; infinity times zero is invalid. Tested bit-for-bit against the host's `mul_add` for every form, with subnormals and forced cancellations. F16C: `vcvtph2ps` is exact (all 65536 inputs checked), `vcvtps2ph` rounds under imm8[1:0] or MXCSR.RC (imm8[2]) with overflow/underflow/precision flags (checked against an independent table-search reference in all four modes). The compare predicates beyond the legacy eight (imm8 8..31: the 32 SDM names, with the `_S` forms signalling on quiet NaNs) are all tested.

Not implemented: AVX-512 (EVEX), VAES/VPCLMULQDQ (the 256-bit forms of AES/PCLMUL are #UD), VPOPCNT, GFNI, SHA, AVX-VNNI. The JIT runs every VEX instruction through its interpreter helper.
