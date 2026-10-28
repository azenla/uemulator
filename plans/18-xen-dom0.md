# Fedora Xen dom0 through UEFI

Goal: boot a Fedora Xen dom0 (Xen hypervisor, then its Linux dom0, started by a UEFI boot loader) on the emulated machine. Xen is the first guest that insists on a full platform description: it parses ACPI itself (MADT for CPUs and the I/O APIC, MCFG, HPET, IVRS for the IOMMU), runs on the I/O APIC with x2APIC when it can, uses the HPET and PM timer for calibration, and gives dom0 the IOMMU's remapping and DMA protection. Everything below is what the library provides for it; the UEFI firmware side that hands these tables to the boot loader is a later step.

Read `07-devices.md` ("ACPI, PM block, HPET, ECAM") and `16-iommu.md` first.

## Steps

1. ACPI platform: tables, PM block and timer, S5 power-off, HPET, ECAM, `boot_params.acpi_rsdp_addr`.
2. IOMMU wiring: I/O APIC, MSI/MSI-X and HPET interrupts through `Iommu::remap_interrupt`, the IOMMU's own interrupts, XTSup, `ACCESS_PLATFORM` on virtio.
3. Later: SMP (the MADT already lists `AcpiConfig::cpus` processors, nothing starts them), the same tables from a UEFI firmware (`crates/uemulator-uefi` and a boot-loader path, the BIOS boot of a disk publishes none), enabling the IOMMU on `quick` machines, Xen-specific platform bits.

## Status

Steps 1 and 2 are done and checked in real guests (the interpreter and the JIT run the same code; nothing under `jit/` changed).

- `MachineConfig::acpi` (default on) publishes RSDP rev 2, RSDT/XSDT, FACS, FADT rev 6, DSDT, MADT, MCFG, HPET and, with `MachineConfig::iommu`, IVRS at 0xE0000, marked ACPI data in e820. The default command line drops `acpi=off`; `acpi(false)` restores the table-less machine, and `Platform` hosts keep `acpi=off`.
- Devices: ACPI PM1 block and 24-bit PM timer (S5 write = `ExitReason::Shutdown`, FADT reset register = reset), a three-timer HPET (I/O APIC pin or FSB delivery), PCIe ECAM for bus 0, MSI-X on every virtio function, a Linux-compatible `_PRT` and the interrupt line plumbing (SCI on IRQ 12).
- IOMMU: all interrupt sources meet in `Devices::deliver`, which remaps through the IOMMU once the guest enables it; the IOMMU's event/completion interrupts reach the LAPIC (MSI, or IntCapXT with XTSup); IRTEs honour the 32-bit destination under XTEn; virtio functions offer `ACCESS_PLATFORM` behind the IOMMU; the CLI has `--iommu`.
- Guests: Alpine 6.18 virt kernel (no AMD IOMMU driver): ACPI parsed without errors, `hpet` and `acpi_pm` clocksource candidates (the kernel settles on `tsc`), I/O APIC interrupts, virtio-pci with MSI-X, virtio disk read/write and virtio-net through the NAT, `poweroff` through ACPI S5. Debian trixie 6.12 installer kernel with `--iommu`: `AMD-Vi: Interrupt remapping enabled`, `X2APIC enabled`, `IR-IO-APIC`/`IR-PCI-MSIX`/`IOMMU-MSI` in `/proc/interrupts`, virtio-net pings through the IOMMU, `reboot: Power down` through S5, no unhandled IRQs.

Known gaps: no SMP; virtio-blk behind the IOMMU is only unit-tested (no cached guest with both `AMD_IOMMU` and `virtio_blk` built in); interrupts that need the IOMMU's interrupt table and are raised on the CPU's polling path wait for the next device tick (at most 4096 instructions); no power button, GPE blocks, S3/S4 or hotplug; the 20th virtio function (PCI slot 0x14) shares its device-table entry with the I/O APIC's requester id; the BIOS boot path (`boot_disk`) does not publish ACPI tables.
