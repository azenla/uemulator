# Devices and time

Port I/O only (no MMIO in v1). Unknown ports read all ones, writes ignored. Devices implement a small trait: `io_read(port, size) -> u32`, `io_write(port, size, val)`, optional `next_deadline() -> Option<u64>` and `tick(now)`.

## Clock (`clock.rs`)

- `now_ns()` = single virtual time. `ClockMode::Virtual`: `ns = icount * NS_PER_INSN (default 2)`, plus warps. `Realtime`: `max(virtual, wall)`; HLT sleeps on a condvar until the next deadline or console input.
- TSC = `now_ns * tsc_hz / 1e9` (advertised 1 GHz-ish; kernel calibrates via PIT, so PIT and TSC must share this clock).
- Timer queue: sorted deadlines from PIT, RTC periodic, (LAPIC timer later). The run loop checks `now >= next_deadline` at TB boundaries and on the icount budget (05).
- HLT with interrupts enabled in Virtual mode: advance the clock to `next_deadline` immediately (skip idle). With no deadline and no input in Realtime mode: block on the condvar. With no deadline at all in Virtual mode and IF=1 the host naps and polls (host input can still wake the guest).
- HLT with IF=0 is `ExitReason::Shutdown`: only NMI/SMI/INIT/RESET wake it (SDM vol. 3 6.8) and none exists, so the guest is dead. This is how Linux ends poweroff/halt with `acpi=off` (`stop_this_cpu`: `cli; hlt` loop). Decided in `Cpu::idle` (after stop and reset checks), so interpreter and JIT agree; only IF counts, not the STI/MOV SS shadow (`sti; hlt` waits). A PIC line pending behind IF=0 does not save it. The UEFI hardware-virtualization loop applies the same rule (`hlt_outcome`).

## 16550 UART at 0x3F8, IRQ4

- Registers: RBR/THR, IER, IIR/FCR, LCR (DLAB swaps in DLL/DLM), MCR, LSR, MSR, SCR. Loopback bit in MCR must work (8250 autoconfig probes it).
- LSR bit5 THRE and bit6 TEMT always set after TX (output is instantaneous); bit0 DR when RX queue non-empty.
- IER must round-trip; IIR reflects pending cause (THRE interrupt on IER enable edge, RX-available) and the FIFO-enabled bits (0xC0 when FCR bit0 set) so the kernel detects a 16550A rather than "no port".
- TX bytes go to the console's output queue; RX from `ConsoleHandle::write` raises IRQ4 through the PIC.
- Also stub 0x2F8/0x3E8/0x2E8 as absent (read 0xFF) so autoprobe doesn't wedge.

## 8259 PIC pair (0x20/0x21, 0xA0/0xA1)

ICW1–4, OCW1–3, IRR/ISR/IMR, cascade on IR2, specific/non-specific EOI, edge triggered, ELCR ports 0x4D0/0x4D1 optional. Interrupt acknowledge returns vector = base + irq. CPU polls `pic.has_pending()` when IF=1 and not in an interrupt shadow (STI/MOV SS one-instruction shadow).

## 8254 PIT (0x40–0x43) and port 0x61

Channels 0 (system timer → IRQ0), 2 (speaker gate, used for TSC calibration). Modes 0, 2, 3 at least; latch and read-back commands; counter derived from the virtual clock at 1.193182 MHz. **Port 0x61**: bit0 gate for ch2, bit1 speaker, bit4 refresh toggle, bit5 = OUT2 status. The kernel's `pit_calibrate_tsc`/`quick_pit_calibrate` poll these; get them right or TSC khz is nonsense (and `tsc=reliable lpj=` cmdline is the fallback).

## CMOS RTC (0x70/0x71)

Time registers (BCD, 24h) from host wall clock at boot (optional `fixed_time` config for determinism), status A/B/C/D, NMI mask bit in 0x70, periodic IRQ8 optional, memory-size bytes (0x15–0x18, 0x30/0x31, 0x34/0x35) filled for old probes.

## Misc ports

- 0x92 fast A20 / reset bit; 0x64/0x60 i8042 stub: status 0, command 0xFE → reset request (`ExitReason::Reset`, or an in-library reboot, see 01-api); 0xCF9 reset control; 0x80 POST delay (ignore); 0xCF8/0xCFC PCI config: reads return 0xFFFFFFFF (no devices); 0x3F6.. IDE absent → 0xFF.
- DMA controller 0x00–0x0F, 0x81–0x8F: accept writes, reads 0 (kernel may touch for `dma` reservation).
- Kernel `x86_platform`/`legacy_pic`: with `acpi=off` and no APIC it expects PIC+PIT+RTC; no HPET.

## LAPIC and IOAPIC (`dev/apic.rs`, `dev/ioapic.rs`)

One local APIC at 0xFEE00000 (xAPIC MMIO, or x2APIC through MSRs 0x800..=0x8ff once IA32_APIC_BASE.EXTD is set) and one 24-pin I/O APIC at 0xFEC00000. Advertised in CPUID leaf 1 (APIC, x2APIC, TSC-deadline) together with leaf 0xB (one logical processor).

- **Reset state is the firmware-left virtual wire** (LINT0 = ExtINT unmasked, LINT1 = NMI masked, SVR disabled), so a kernel that finds no ACPI/MP tables still gets PIC interrupts through the APIC (`extint_connected`, QEMU's `apic_accept_pic_intr`: only LINT0's mask bit counts). Linux boots in `APIC_VIRTUAL_WIRE_NO_CONFIG` mode with the TSC-deadline timer.
- **Interrupt path.** Devices still call `Pics::raise/lower`; the PIC logs each line change (`line_events`) and `Devices::route_line_events` feeds it to the I/O APIC, so the same lines drive both controllers as on a PC. IRQ0 maps to pin 2 (the PIT's wiring), the other lines map straight through. `Devices::interrupt_pending`/`interrupt_ack` are what the CPU polls; LAPIC vectors win over ExtINT, and delivery honours TPR/PPR. CR8 is TPR[7:4] and lives in the LAPIC.
- **Level triggering**: an I/O APIC pin sets remote IRR when it sends and sends again after the LAPIC's EOI (broadcast to the I/O APIC through `WriteEffect::eoi_vector`, plus the version-0x20 EOI register) only if the line is still high. Device lines must be routed before an EOI is processed, hence `route_line_events` at the start of every APIC/IOAPIC register access.
- **Timer**: 100 MHz bus clock, divide configuration, one-shot, periodic and TSC-deadline (the TSC is virtual ns). Deadlines join `Devices::next_deadline`, so a halted guest sleeps to the APIC timer.
- **NMI** (LINT1, ICR, I/O APIC entries): `Cpu::nmi_blocked` is set on delivery and cleared by a completed `iret`; NMIs ignore IF and wake `hlt` with IF=0 (which otherwise ends the machine).
- Not modeled: other CPUs (IPIs to them go nowhere, INIT/SIPI/SMI are dropped), MSI delivery from devices (no device has MSI yet), the AMD extended APIC space (reads as zero), interrupt remapping of I/O APIC entries.

## MTRR, PAT and machine checks (`cpu/mtrr.rs`, `cpu/mca.rs`)

Validated storage only, because nothing here has caches: MTRR capability/default/fixed/variable registers (reserved memory types and bits are #GP), PAT validation, AMD SYSCFG/IORR/TOP_MEM/TOP_MEM2 (TOP_MEM = RAM size), and an inert machine-check architecture (MCG_CAP/STATUS/CTL and six banks, status always zero) behind CPUID MCE/MCA and CR4.MCE.
