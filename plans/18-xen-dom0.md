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
4. SMP, cooperative round robin on the one host thread. Done.
5. Firmware, key-value store, `quick` support, GRUB.
6. Xen: boot `xen.efi` or GRUB multiboot2 + Linux dom0, then a PVH domU.

## Firmware contract (library <-> `crates/uemulator-firmware`)

- Built for `x86_64-unknown-none` (`rustup target add x86_64-unknown-none`), `no_std`, embedded by the CLI with `include_bytes!` from a `build.rs` that runs cargo for the firmware crate into its own target dir. The library only sees an opaque image: `MachineConfig::uefi_firmware(image)`.
- The CPU starts at the image's entry in 64-bit mode, identity-mapped low 4 GiB, interrupts off, with RDI pointing at a `FirmwareHandoff` page (RAM size, RSDP address, ACPI table region, the PCI ECAM base, the key-value device ports). The library builds the ACPI tables host-side because it knows the topology (and owns `Iommu::ivrs_table`); the firmware publishes them as EFI configuration tables.
- Key-value store: `trait KeyValueStore` (get, set, delete, list by prefix) in the library; a file-backed implementation (hosted only); a port-I/O device the firmware's `GetVariable`/`SetVariable`/`GetNextVariableName` use. `quick` machines get `nvram.kv` in their directory.
- Disks are the existing virtio-blk functions; the firmware carries a small polling driver, a GPT parser and a read-only FAT reader for the ESP.
- Console is the 16550.

## Status

Steps 1, 2 and 4 are done and checked in real guests (the interpreter and the JIT run the same code; nothing under `jit/` changed).

- `MachineConfig::acpi` (default on) publishes RSDP rev 2, RSDT/XSDT, FACS, FADT rev 6, DSDT, MADT, MCFG, HPET and, with `MachineConfig::iommu`, IVRS at 0xE0000, marked ACPI data in e820. The default command line drops `acpi=off`; `acpi(false)` restores the table-less machine, and `Platform` hosts keep `acpi=off`.
- Devices: ACPI PM1 block and 24-bit PM timer (S5 write = `ExitReason::Shutdown`, FADT reset register = reset), a three-timer HPET (I/O APIC pin or FSB delivery), PCIe ECAM for bus 0, MSI-X on every virtio function, a Linux-compatible `_PRT` and the interrupt line plumbing (SCI on IRQ 12).
- IOMMU: all interrupt sources meet in `Devices::deliver`, which remaps through the IOMMU once the guest enables it; the IOMMU's event/completion interrupts reach the LAPIC (MSI, or IntCapXT with XTSup); IRTEs honour the 32-bit destination under XTEn; virtio functions offer `ACCESS_PLATFORM` behind the IOMMU; the CLI has `--iommu`.
- Guests: Alpine 6.18 virt kernel (no AMD IOMMU driver): ACPI parsed without errors, `hpet` and `acpi_pm` clocksource candidates (the kernel settles on `tsc`), I/O APIC interrupts, virtio-pci with MSI-X, virtio disk read/write and virtio-net through the NAT, `poweroff` through ACPI S5. Debian trixie 6.12 installer kernel with `--iommu`: `AMD-Vi: Interrupt remapping enabled`, `X2APIC enabled`, `IR-IO-APIC`/`IR-PCI-MSIX`/`IOMMU-MSI` in `/proc/interrupts`, virtio-net pings through the IOMMU, `reboot: Power down` through S5, no unhandled IRQs.

Known gaps: no SMP; virtio-blk behind the IOMMU is only unit-tested (no cached guest with both `AMD_IOMMU` and `virtio_blk` built in); interrupts that need the IOMMU's interrupt table and are raised on the CPU's polling path wait for the next device tick (at most 4096 instructions); no power button, GPE blocks, S3/S4 or hotplug; the 20th virtio function (PCI slot 0x14) shares its device-table entry with the I/O APIC's requester id; the BIOS boot path (`boot_disk`) does not publish ACPI tables.

- Step 5 (firmware): done except GRUB multiboot2, Xen and shim, which are untested. `crates/uemulator-firmware`, the key-value store, `quick` machines with `firmware = "uefi"` and `nvram.kv`; the Linux EFI stub and Debian GRUB 2.12 boot Alpine to userspace and efivars persist across resets. See `19-uefi-firmware.md`.
- Steps 1-4 and 6: not done.

- Step 3 (CPU model and AVX), done except what needs steps 1 and 2 to show: `cpu/cpuid.rs` is a Zen 4 (family 19h model 11h stepping 1, signature 00A10F11, "AMD EPYC-Genoa Processor") with SVM + NPT + NRIPS + VMCB clean + flush-by-ASID (decode assists, the pause filter, AVIC and the TSC ratio MSR are not implemented and not advertised), x2APIC, XSAVE with correct sizes in both formats (XSAVEC/XSAVES/XGETBV1), invariant TSC, cache/TLB/topology leaves (0xB, 0x8000001D, 0x8000001E, 0x80000008 ECX) consistent with `Cpu::topology` (cpu count, threads per core, x2APIC id; defaults to one processor, to be driven by the SMP work), the AMD workaround MSRs as stored registers (HWCR, LS_CFG, DE_CFG, BP_CFG, the legacy perf counters, PATCH_LEVEL). A guest Linux 6.18 prints the Zen 4 identity, `xsaves`/compacted format and no PMU or microcode complaints. AVX was investigated in a real guest (see 03, "Real-guest check"): VEX instruction semantics match QEMU TCG bit for bit; the failures were missing instructions (`vpsubsw` EVEX, `vdbpsadbw`, SHA, BF16, ADX) and `clwb` executing as xsaveopt, all fixed. Not done: Xen itself has not been booted (it needs steps 1, 2, 5), so "no warnings from Xen" is not verified; leaf 0x1F, PKU, LA57, SSE4A, MWAITX, RDPRU, INVLPGB, perfctr_core and IBS are absent on purpose.
- Steps 1, 2, 4, 5, 6: not started.

- SMP (step 4): `MachineConfig::cpus(1..=64)`, `--cpus`, `cpus` in `machine.toml`; design and semantics in `plans/03-cpu.md` ("SMP") and `plans/07-devices.md` ("LAPIC and IOAPIC"). Linux sees N vCPUs from the MADT, starts them with INIT/SIPI (x2APIC and xAPIC), runs IPIs, TLB shootdowns and CPU hotplug; Alpine 6.18 on the JIT (4 and 64 vCPUs) and the interpreter, Debian 6.12 with `--iommu` (cluster x2APIC with interrupt remapping). Default stays 1: the vCPUs share one host thread, so more of them cost host time without adding speed. Xen should set the count explicitly. CPUID's topology leaves are minimal (`Cpu::smp_cpuid`) and will need the Zen 4 model's real layout (leaf 0x8000001E) merged in.

Known gaps: virtio-blk behind the IOMMU is only unit-tested (no cached guest with both `AMD_IOMMU` and `virtio_blk` built in); interrupts that need the IOMMU's interrupt table and are raised on the CPU's polling path wait for the next device tick (at most 4096 instructions); no power button, GPE blocks, S3/S4 or hotplug; the 20th virtio function (PCI slot 0x14) shares its device-table entry with the I/O APIC's requester id; the BIOS boot path (`boot_disk`) does not publish ACPI tables.
