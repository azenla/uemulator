# AVX and nested virtualization (not implemented)

Status: designs only. VEX decoding and the GPR-only BMI1/BMI2 forms exist (03, "VEX and BMI"); everything below builds on that. Nothing here is advertised in CPUID.

## AVX, AVX2, FMA, F16C

Why not yet: advertising `avx` (leaf 1 ECX bit 28) or `avx2` (leaf 7 EBX bit 5) is a promise that glibc's ifunc-selected `memcpy`/`strlen`/`memchr`, the kernel's crypto and `copy_user` paths and every compiler's `-march=x86-64-v3` output will run. A partial set turns into #UD in userspace, so the bits come last, after the whole set below passes. `x86-64-v3` also needs BMI1/2, LZCNT, MOVBE, FMA and F16C (BMI/LZCNT/MOVBE are done).

State: `Cpu::xmm: [u128; 16]` becomes the low halves; add `ymm_hi: [u128; 16]`. XCR0 bit 2 (AVX) joins `XCR0_SUPPORTED` in `cpu/xsave.rs`; xsave gets the component at offset 576, size 256 (CPUID.0xD.2 = {256, 576, 0, 0}, leaf 0xD.0 EBX/ECX become 832); `xrstor` treats XSTATE_BV bit 2 like the other components; `enabled_area_size` already exists for this. VEX needs no CR4 bit beyond OSXSAVE, but `#UD` if XCR0 lacks SSE|AVX for any ymm form, and `#NM` under CR0.TS.

Execution plan, keeping the interpreter the oracle:
1. VEX.128 forms of every SSE/SSE2/SSE3/SSSE3/SSE4/AES/PCLMUL op are the legacy op with a separate first source (`vex_v`) and the destination's upper half (and ymm upper) zeroed; legacy SSE leaves the upper half alone. Implement as a wrapper in `sse.rs`/`sse_ext.rs` that routes `i.vex` through the same lane kernels (`zip_lanes`, `integer_op`, the soft-float engine), so the arithmetic is shared, not copied. Rules that differ: no alignment #GP for VEX loads/stores except the explicitly aligned `vmovaps`/`vmovdqa`; scalar ops merge the upper bits from `vex_v`, not the destination.
2. VEX.256 lane-wise ops run the 128-bit kernel twice; cross-lane ones (vperm2f128, vpermq, vpermps, vinsertf128, vextractf128, vbroadcast*, vpmovmskb ymm, vpshufb per-lane, vpalignr per-lane, pack/unpack per-lane) get explicit tests per the SDM pseudo-code.
3. New instructions: vzeroupper/vzeroall, vmaskmov*, vpermilps/pd, vtestps/pd, vbroadcast*, vpgather* (AVX2; each element's fault leaves the mask intact), vpsllv/vpsrlv/vpsrav, FMA3 (vfmadd/vfmsub/vfnmadd/vfnmsub in 132/213/231 forms: needs a fused multiply-add on the soft-float engine, i.e. one rounding of a 2x-wide product plus addend), F16C vcvtph2ps/vcvtps2ph.
4. JIT: all of it stays on the interpreter helper (`VEX_OP_FLAG` keeps VEX ops out of the inline matchers). Add `assert_jit_matches_interpreter` cases per group; the lockstep verifier compares `xmm` today and must learn `ymm_hi` (`Cpu::diff`).
5. Only then: CPUID bits AVX, AVX2, FMA, F16C (+ `xsave` size), `xcr0` AVX, `avx` in the `xsaveopt` area, and drop the "No AVX" lines in 03.
Do not add AVX-512 (EVEX, opmask state, 3 more XCR0 components); `x86-64-v3` is the target.

## SVM (AMD-V) for nested guests

The CPU identity is AMD, so SVM is the natural one. Cost estimate: large; every item needs the exception-delivery and MMU code to know about a second "mode".

- Enable: CPUID 0x8000_0001 ECX bit 2 (SVM), leaf 0x8000_000A (revision, ASID count, features), `EFER.SVME` (bit 12, currently rejected by `EFER_WRMSR_ALLOWED`), `VM_CR` and `VM_HSAVE_PA` MSRs (0xC001_0114/0xC001_0117).
- Instructions (0F 01 D8-DF): `vmrun`, `vmmcall` (already #UD-ish), `vmload`, `vmsave`, `stgi`, `clgi`, `skinit`, `invlpga`. `vmrun` saves host state to the HSAVE area, loads the VMCB guest state, runs until an intercept, and writes exit code/info back (`#VMEXIT`).
- Intercepts: the VMCB intercept bitmaps (CR/DR read/write, exceptions, INTR/NMI, CPUID, HLT, IOIO and MSR permission maps, VMRUN...). Needs a check at the top of each intercepted handler (`exec`), at exception delivery and at interrupt injection; `event_inj` injection on entry; TSC offset; interrupt-shadow and V_INTR virtual interrupts.
- Memory: nested paging (NPT) is a second page-walk dimension in `cpu/mem.rs` (guest-physical to host-physical through `nCR3`, with its own fault codes in `exit_info`). Without NPT the L1 hypervisor needs shadow paging, which Linux KVM no longer supports on AMD, so nested KVM means NPT. The soft TLB tags would have to include the ASID/nested mode.
- Order of work: (1) EFER.SVME + `vmrun`/`#VMEXIT` for a trivial hand-built guest with intercept-everything and no NPT, tested with `testkit` snippets; (2) intercepts one class at a time; (3) NPT; (4) boot KVM or Xen inside the emulator as the exit test. The UEFI crate already has the *host* side of SVM (it runs the emulated guest on hardware SVM), which is useful as a reference for VMCB layout, not as emulation code.

## VMX (Intel VT-x)

Not a fit for the AMD identity (`GenuineIntel` would also change Linux's model-specific quirks and mitigation tables, see 03 "Speculation control"). VMCS fields via `vmread`/`vmwrite`, EPT, VPID, the capability MSRs 0x480-0x491: a larger surface than SVM with no benefit for this project unless an Intel-only guest hypervisor must run. Defer until SVM works and then only if needed.

## Other items not done

- ERMS/FSRM: measured slower, see the comment on `STRUCTURED_FEATURE_FSGSBASE` in `cpu/sys.rs`; needs a bulk `rep movsb`/`stosb` path first.
- MONITOR/MWAIT, PCID/INVPCID, SMEP/SMAP/UMIP (MMU and CR4 changes), PKU, CET (shadow stacks; the kernel only needs `endbr64`, which is a NOP here), LA57, XSAVEC/XSAVES, x2APIC/APIC.
