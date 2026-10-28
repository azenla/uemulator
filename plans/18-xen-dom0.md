# Xen dom0 on a modern AMD machine

Goal: boot a stock Xen with a Fedora dom0 (PVH, x86_64) through UEFI and GRUB, on a machine that looks like a current AMD server part. Success line: Fedora dom0 at a login prompt under Xen; stretch: a PVH domU running inside it. Alpine's `xen` packages are the cheap stand-in for Fedora while developing (same hypervisor, much smaller userspace).

Written by the first session on this work; update the status list as pieces land.

## What Xen needs from the machine

| need | piece |
|------|-------|
| ACPI tables: RSDP, XSDT, FADT, DSDT, MADT, MCFG, HPET, IVRS | `acpi/` in the library (step 1) |
| platform timer (HPET, ACPI PM timer), ACPI power-off and reset | `dev/hpet.rs`, `dev/acpi_pm.rs` (step 1) |
| IOMMU found through IVRS and interrupts remapped | IOAPIC, MSI and x2APIC call `Iommu::remap_interrupt` (step 2) |
| a CPU Xen recognises as AMD Zen: family 19h CPUID, SVM + NPT, x2APIC, AVX2/AVX-512 working | step 3 |
| more than one vCPU | step 4 |
| UEFI: memory map, config tables, runtime services, variables, block I/O, a FAT ESP, loaded-image, GRUB | `crates/uemulator-firmware` (step 5) |
| persistent variables | key-value store provided by the embedder (step 5) |

## Order

1. ACPI and platform devices. Verified by booting Linux with `acpi_rsdp` (boot_params has the field), no firmware needed: `dmesg` shows ACPI, HPET, `AMD-Vi`.
2. IOMMU wiring. A guest Linux finds `AMD-Vi`, enables DMA remapping, virtio devices keep working behind it, x2APIC interrupt remapping is on.
3. CPU model and AVX. Family 19h model, cache/topology leaves, MSRs; AVX checked in a real guest, not only in unit tests.
4. SMP, cooperative round robin on the one host thread.
5. Firmware, key-value store, `quick` support, GRUB.
6. Xen: boot `xen.efi` or GRUB multiboot2 + Linux dom0, then a PVH domU.

## Firmware contract (library <-> `crates/uemulator-firmware`)

- Built for `x86_64-unknown-none` (`rustup target add x86_64-unknown-none`), `no_std`, embedded by the CLI with `include_bytes!` from a `build.rs` that runs cargo for the firmware crate into its own target dir. The library only sees an opaque image: `MachineConfig::uefi_firmware(image)`.
- The CPU starts at the image's entry in 64-bit mode, identity-mapped low 4 GiB, interrupts off, with RDI pointing at a `FirmwareHandoff` page (RAM size, RSDP address, ACPI table region, the PCI ECAM base, the key-value device ports). The library builds the ACPI tables host-side because it knows the topology (and owns `Iommu::ivrs_table`); the firmware publishes them as EFI configuration tables.
- Key-value store: `trait KeyValueStore` (get, set, delete, list by prefix) in the library; a file-backed implementation (hosted only); a port-I/O device the firmware's `GetVariable`/`SetVariable`/`GetNextVariableName` use. `quick` machines get `nvram.kv` in their directory.
- Disks are the existing virtio-blk functions; the firmware carries a small polling driver, a GPT parser and a read-only FAT reader for the ESP.
- Console is the 16550.

## Status

- Step 3 (CPU model and AVX), done except what needs steps 1 and 2 to show: `cpu/cpuid.rs` is a Zen 4 (family 19h model 11h stepping 1, signature 00A10F11, "AMD EPYC-Genoa Processor") with SVM + NPT + NRIPS + VMCB clean + flush-by-ASID (decode assists, the pause filter, AVIC and the TSC ratio MSR are not implemented and not advertised), x2APIC, XSAVE with correct sizes in both formats (XSAVEC/XSAVES/XGETBV1), invariant TSC, cache/TLB/topology leaves (0xB, 0x8000001D, 0x8000001E, 0x80000008 ECX) consistent with `Cpu::topology` (cpu count, threads per core, x2APIC id; defaults to one processor, to be driven by the SMP work), the AMD workaround MSRs as stored registers (HWCR, LS_CFG, DE_CFG, BP_CFG, the legacy perf counters, PATCH_LEVEL). A guest Linux 6.18 prints the Zen 4 identity, `xsaves`/compacted format and no PMU or microcode complaints. AVX was investigated in a real guest (see 03, "Real-guest check"): VEX instruction semantics match QEMU TCG bit for bit; the failures were missing instructions (`vpsubsw` EVEX, `vdbpsadbw`, SHA, BF16, ADX) and `clwb` executing as xsaveopt, all fixed. Not done: Xen itself has not been booted (it needs steps 1, 2, 5), so "no warnings from Xen" is not verified; leaf 0x1F, PKU, LA57, SSE4A, MWAITX, RDPRU, INVLPGB, perfctr_core and IBS are absent on purpose.
- Steps 1, 2, 4, 5, 6: not started.
