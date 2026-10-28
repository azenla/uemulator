# AMD IOMMU (AMD-Vi)

`MachineConfig::iommu(true)` adds an AMD IOMMU as PCI function 00:00.2 and puts every PCI function's DMA behind it. Code is in `crates/uemulator/src/dev/iommu/`. It exists for Windows 11 and Linux, which use it for DMA protection and (Windows with VBS, Linux with x2APIC and large guests) interrupt remapping; neither finds it without an ACPI IVRS table, so until there is firmware that publishes one it is exercised by unit tests only.

Spec: AMD I/O Virtualization Technology (IOMMU) Specification, rev 3.x (publication 48882). QEMU reference: `hw/i386/amd_iommu.{c,h}` (register layout, command set, event encoding, interrupt remapping decision tree), read for behaviour and not transliterated. Consumer used as the checklist of what must work: Linux `drivers/iommu/amd/amd_iommu_types.h` (bit positions, the IOMMU page table encoding `PAGE_SIZE_PTE`, the event flags).

## Layout

| file | contents |
|------|----------|
| `iommu/mod.rs` | `Iommu` state, the PCI function (config space, capability block, MSI), the MMIO register file, `translate` / `probe` |
| `iommu/commands.rs` | command buffer execution |
| `iommu/events.rs` | event encoding, the log, overflow, the event interrupt |
| `iommu/tables.rs` | device table entries, the 1 to 6 level page walk |
| `iommu/irq.rs` | `remap_interrupt`, `ioapic_request`, IRTE decoding |
| `iommu/ivrs.rs` | `Iommu::ivrs_table` |
| `iommu/tests.rs` | the fake driver and 76 tests |

## PCI function and registers

- 00:00.2, vendor 0x1022 device 0x1419, class 08h/06h. The host bridge at 00:00.0 turns on the multifunction bit when the IOMMU is present; Linux and Windows only scan functions 1-7 of a slot when function 0 says so.
- Capability 0x40 (id 0Fh, type 3, EFRSup and CapExt set): base address low/high with an enable bit (preset to 0xFED80000 enabled, because nothing programs it; the guest may move or disable it), range register (RngValid clear: the IVHD says which devices), misc info (48-bit PA, 64-bit VA). MSI capability at 0x58 (64-bit, one message) delivers the event and completion-wait interrupts: `Iommu::take_msi()` returns the message once, or `None` if the guest has MSI off. Delivering it is the LAPIC's job.
- MMIO window 16 KiB at the capability's base: device table base (size field = pages-1), command buffer and event log bases (length 2^8..2^15 entries, writing a base resets its head and tail), control, exclusion range, extended feature register, head/tail pointers, status (overflow, event interrupt and completion-wait interrupt are write-1-to-clear; the running bits mirror CmdBufEn and EventLogEn). 32-bit halves are accessible. PPR, GA log, SMI filters, hardware error and performance counter registers read zero and ignore writes.
- Extended features advertised: PreFSup, IASup, GASup (128-bit IRTEs), HATS 6-level. Not advertised: PPR, guest translation (GT), x2APIC (XT), NX, HE, performance counters, ATS. XTSup is the one to add when the LAPIC grows x2APIC: it needs the 32-bit IRTE destination (`destination_hi`) and `CONTROL_XT_EN`.

## Commands, events

- Executed synchronously when software writes the tail or sets CmdBufEn: COMPLETION_WAIT (store and interrupt), INVALIDATE_DEVTAB_ENTRY, INVALIDATE_IOMMU_PAGES (drops the whole domain: over-invalidating is legal), INVALIDATE_IOTLB_PAGES, INVALIDATE_INTERRUPT_TABLE (no-op, IRTEs are read on every use), PREFETCH, COMPLETE_PPR_REQUEST, INVALIDATE_IOMMU_ALL. An unknown opcode logs ILLEGAL_COMMAND_ERROR with the command's address and stalls the buffer (CmdBufRun clear, head left on it) until software toggles CmdBufEn; a completion-wait store to unmapped memory does the same with COMMAND_HW_ERROR.
- Events: ILLEGAL_DEV_TABLE_ENTRY, IO_PAGE_FAULT (flags PR/RW/I as Linux reads them), DEV_TAB_HW_ERROR, PAGE_TAB_HW_ERROR, ILLEGAL_COMMAND_ERROR, COMMAND_HW_ERROR. Faults found during DMA are queued (`GuestMem::read` is `&self`) and written to the log by `service`, which every MMIO write and the PCI poll call. A full log sets EventOverflow and stops (EventLogRun clear) until software toggles EventLogEn, as `amd_iommu_restart_event_logging` does. Events while the log is off are dropped.

## Translation

- `Iommu::translate(mem, requester, iova, access) -> Translation { phys, bytes_left }`: IOMMU off or no device table programmed = identity. Order: IOTLB hit, DTE fetch, V=0 passes through (QEMU; Linux blocks with V=1, TV=1, mode 0, no IR/IW), reserved DTE bits are an illegal entry, the exclusion range (all devices with ExclusionAllow, else those with DTE.EX), TV=0 passes through, mode 0 passes through if the DTE's IR/IW allow the access, else the walk.
- Walk: modes 1..6, permissions are the AND of the DTE's IR/IW and every entry's, NextLevel may skip levels, 0 is a leaf of the level's size (4 KiB, 2 MiB, 1 GiB...), 7 is a leaf whose size is 2^(p+1) for the lowest clear address bit p >= 12. Such an entry must be repeated in every slot the page covers; the walk does not search for the first. An address above what the mode covers is not present.
- IOTLB: 4 KiB granular, keyed by (requester, page), carries domain and permissions, dropped by the invalidation commands, by changing the device table base or the IOMMU enable, and wholesale at 1024 entries. A translation stays in use until the guest invalidates it, as on hardware.
- Interrupt window 0xFEE00000..0xFEEFFFFF: DMA there is refused without an event (interrupt messages go through `remap_interrupt`, not memory).

## DMA hook

`GuestMem::with_dma(&mut iommu, requester, |mem| ..)` makes every `read`/`write`/`contains` inside the closure a device access from `requester`, split at translation boundaries; `read_raw`/`write_raw` and the `PhysRead`/`PhysWrite` traits are the untranslated physical view (the IOMMU's own table reads, the event log). The IOMMU is moved into the `GuestMem` for the closure and put back after, so no lifetime ties them. `contains` probes without permissions (a write-only buffer still passes) and logs the not-present fault when it refuses, since the caller is about to fail the device's request. Hooks in shared code, all marked by `iommu`: `PciHost::with_iommu`, `PciHost::{iommu_mmio_read, iommu_mmio_write}` called first in `Devices::mmio_read/write`, `PciHost::mmio_write` and `poll` wrap each virtio function in `with_dma` with requester = its devfn, `PciHost::host_bridge` sets the multifunction bit, `DevConfig::iommu`, `MachineConfig::iommu`.

## Interrupt remapping

`Iommu::remap_interrupt(mem, InterruptRequest { requester, address, data }) -> Result<RemappedInterrupt, InterruptFault>` is what the LAPIC/IOAPIC code calls for every device interrupt; an IOAPIC redirection entry becomes a request with `ioapic_request` (requester = the IOAPIC's IVRS special-device id; with remapping on its vector field carries the IRTE index). IOMMU off, V=0 or IV=0: the message is returned as sent. IV=1: IntCtl abort / forward / remap for fixed and lowest-priority types (SMI is unsupported), NMI/INIT/ExtInt only with their DTE pass bit (LINT1 pass also admits NMI and LINT0 pass ExtInt, my reading of the spec's pin semantics). Remap reads the IRTE at `table + index * 4` (or `* 16` with control.GAEn), index = data[10:0], bounded by 2^IntTabLen (1..11); RemapEn clear aborts (SuppIOPF silences the event), guest mode and non-fixed/arbitrated types are unsupported. Aborts log an IO_PAGE_FAULT with the interrupt flag and the message address.

## IVRS

`Iommu::ivrs_table(&IvrsTopology { pci_devices, ioapics, hpets })` returns the complete table, checksum included: IVinfo (EFRSup, 48-bit PA, 64-bit VA), then IVHD blocks of types 10h, 11h and 40h with the IOMMU's device id and capability offset, its MMIO base, the EFR image (11h/40h), SELECT device entries, and special entries for IOAPICs and HPETs. ACPI generation puts it in the XSDT; the requester ids of the IOAPIC and HPET must match what the interrupt controllers pass to `remap_interrupt`.

## Integration still to do

- The LAPIC delivers `take_msi()` messages (poll it after `Devices::tick` and after an MMIO write) and calls `remap_interrupt` for IOAPIC and MSI sources; `PciHost::iommu()` is the handle.
- Firmware/ACPI includes the IVRS table and the requester ids it lists.
- Under SVM/VMX nested guests the translation of the *host's* IOMMU is not involved; this is a model of an IOMMU the guest programs, nothing more.
- Not modeled: PPR, guest page tables (GT), ATS and IOTLB invalidation timeouts, the second event log, performance counters, x2APIC (XT), per-device IRTE caching.
