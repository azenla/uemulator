# AVX and nested virtualization (not implemented)

Status: AVX, AVX2, FMA and F16C are implemented (03). SVM and VMX below are designs only and nothing about them is advertised in CPUID.

## AVX, AVX2, FMA, F16C

Done: see 03, "AVX, AVX2, FMA, F16C". Not done: AVX-512 (EVEX), VAES, VPCLMULQDQ.

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
- PKU, CET (shadow stacks; the kernel only needs `endbr64`, which is a NOP here), LA57, XSAVEC/XSAVES, x2APIC/APIC.
