# Milestones

## Status

- M0-M3 done: kernel banner, VFS panic then `ExitReason::Reset`, and the Debian installer userspace (dialogs drawn) with both interpreter and JIT.
- M4 done: aarch64 JIT (block translator, inline TLB fast paths, ALU/mov/lea/push/pop/jcc/call/ret inlined, helper fallback for everything else, lazy-flag-aware jcc, per-page invalidation, self-modifying-code exit). ~5x faster than the interpreter (no-initrd boot: 6.4 s vs 32 s; roughly 300 MIPS). A lockstep verifier (`MachineConfig::verify_jit`, `--verify` in the example) runs the JIT against the interpreter per block; the whole installer boot is divergence-free.
- M6 done (see 11-storage-net-console.md): configurable RAM (`--mem`), virtio-pci disk, network and console (with resize), guest-verified: ping through the gateway responder, `stty size` follows `resize()`, disk read/write through `virtio_blk`.
- M5 partly done: realtime idle (`ClockMode::Realtime`), raw tty in the example, fixed RTC time option, aarch64 block chaining and wide inline lowering (see 06: roughly 3-4x over the M4 JIT; no-initrd Alpine boot 4.8 s -> 1.3 s CPU, initrd boot to the installer 13.9 s -> 4.7 s, interpreter ~60 s, lockstep-verified over the whole no-initrd boot). Not done: guest register caching, dead-flag elimination, LAPIC/IOAPIC, SMP.

- Machine lifecycle done: a guest reset reboots the machine in place by default (`MachineConfig::reboot`, CLI `--no-reboot` to end with `ExitReason::Reset`; checked on the real Alpine kernel with `busybox reboot -f`, interpreter and JIT), `hlt` with IF=0 is `ExitReason::Shutdown` (checked with `busybox poweroff -f`; the UEFI hardware-virtualization loop applies the same rule, but has no in-process reboot). `ConsoleKind::Virtio` (hvc0) is the default everywhere except the UEFI hardware-virtualization path (no virtio in `Platform`). The M2 criterion "VFS panic then clean Reset" now needs `--no-reboot`; by default `panic=-1` reboots forever.

- Workspace split done: `crates/uemulator` (library) + `crates/uemulator-cli` (`uemulator quick create|edit|boot|reinstall|update-artifacts|list|show|delete` for named machines and `uemulator advanced boot|boot-firmware`, clap). 
- `jit` feature (default on) gates `libc`/`windows-sys`; `--no-default-features` is dependency-free and interpreter-only.
- x86_64 host JIT done (see 06): same inlined subset as aarch64, verified in lockstep. Exercised on macOS x86_64 under Rosetta; Linux and Windows x86_64 hosts compile but are untested.

- Instruction extensions done (see 03, "Instruction extensions"): POPCNT/LZCNT, CMPXCHG16B, SSE3, SSSE3 (XMM forms), SSE4.1, SSE4.2 (incl. CRC32 and pcmp[ei]str[im]), PCLMULQDQ, AES-NI, MOVBE and FSGSBASE are implemented and advertised in CPUID (leaf 1 ECX, 0x80000001 ECX, leaf 7). The Alpine boot now executes cmpxchg16b, popcnt, pshufb and palignr in the kernel. Not advertised: VMX.

- SVM done for 64-bit guests (see 03): vmrun/vmload/vmsave/stgi/clgi/invlpga, intercepts, event injection, nested paging, TSC offset; tested with hand-built VMCBs (interpreter and JIT); since validated against KVM and Xen (see 18, step 6).

- ERMS/FSRM done: byte-sized rep slices make `rep movsb` copy-heavy guest workloads ~12x faster (03).

- AVX, AVX2, FMA and F16C done (see 03): VEX.128/256 through the SSE kernels, YMM upper state in XSAVE, gathers, masked moves, a fused multiply-add on the soft-float engine, half-precision conversion. Found while booting Debian: a register-collision bug in the VEX staging that broke glibc's AVX2 `strcmp`, now covered by tests.

- BIOS boot of a disk done (see 17): `MachineConfig::boot_disk` runs the first sector in real mode with the BIOS services as host code, enough for Alpine's syslinux 6.04 to load the installed kernel and boot it to a login prompt; `quick boot` uses it once the disk is not blank.

- Real mode, 16/32-bit protected mode, compatibility mode and virtual-8086 mode done (see 15): mode derived from CS, a mode-aware decoder, segment bases and limits, 16/32-bit interrupt and call-gate delivery, hardware task switching, sysenter/sysexit, and `MachineConfig::firmware` / `uemulator advanced boot-firmware` to boot a real-mode image. 49 tests in `cpu/legacy_tests.rs`. The JIT stays 64-bit-only.

- Non-long-mode paging done (see 04 and 03, "Control-register rules outside long mode"): paging off, 32-bit two-level paging (4 MiB pages, PSE-36 advertised), PAE with latched PDPTEs, and the CR0/CR3/CR4/EFER transition rules; 82 tests in `cpu/paging_tests.rs`.

- AVX-512 done (see 03, "AVX-512"): EVEX decoding, 32 ZMM registers, opmasks and XSAVE components 5-7, AVX512F/VL/BW/DQ/CD/IFMA/VBMI/VBMI2/VNNI/BITALG/VPOPCNTDQ, GFNI, VAES and VPCLMULQDQ, with masking, broadcast, embedded rounding and fault suppression; advertised in CPUID. Alpine and the Debian installer boot with it on. **Correction:** this entry (and the AVX one above) claimed more than was true. Unit tests covered the instructions that existed, but `vpsubsw` (EVEX) and `vdbpsadbw` were #UD while AVX512BW was advertised, which only a real guest running one of every intrinsic showed; both are fixed, as are AVX512_BF16 and SHA-NI (added), and 03 ("Real-guest check") records how AVX was verified in a guest against QEMU TCG, under signals and context switches, and JIT versus interpreter.

- SVM validated against Linux KVM (see 03, "SVM", and `hack/kvm/run.sh`): the emulated Alpine guest loads `kvm_amd` and runs vCPUs in real, protected, long, PAE and 32-bit-paging mode on the emulator's SVM and nested paging. A 32-bit Alpine i686 kernel also boots to a shell (see 15).

- Zen 4 CPU model done (see 03, "CPUID/MSR contract"; `cpu/cpuid.rs`): family 19h model 11h (EPYC 9004 Genoa), cache, TLB and topology leaves (0xB, 0x8000001D/1E) consistent with a configurable processor count and x2APIC id, SHA-NI, ADX, CLWB/CLFLUSHOPT, RDPID, CLZERO, WBNOINVD, AVX512_BF16, XSAVEC/XSAVES/XGETBV1 with the compacted format, 1 GiB pages, and the AMD configuration MSRs Linux and Xen touch. Linux reports "AMD EPYC-Genoa Processor", `sha_ni avx512_bf16 xsavec xsaves clzero rdpid ...` and the compacted XSAVE format.

- UMIP, SMEP/SMAP, PCID/INVPCID, MONITOR/MWAIT, RDTSCP done (see 03).

- VEX decoding, BMI1 and BMI2 done (see 03, "VEX and BMI").

- Speculation control, XSAVE and RDRAND done (see 03, "Speculation control" and "XSAVE and RDRAND"): guests see an in-order CPU immune to every speculative-execution bug (no retpolines, return thunks, KPTI or VERW), use `xsaveopt` for context switches, and seed the crng from `rdrand`/`rdseed`.

- AMD IOMMU done (see 16): `MachineConfig::iommu(true)` adds PCI function 00:00.2 with the IOMMU and MSI capabilities, the MMIO registers, command buffer, event log, device table, 1-6 level I/O page tables (large and arbitrary-size pages), an IOTLB, DMA translation for every virtio function through `GuestMem::with_dma`, interrupt remapping (32- and 128-bit IRTEs) behind `Iommu::remap_interrupt`, and an IVRS table builder. XTSup (x2APIC IRTEs, IntCapXT) came with the ACPI work; a Debian 6.12 guest enables interrupt remapping and x2APIC on it and runs virtio-net through it (16).

- WebAssembly port done (see 13): `crates/uemulator-web` + `hack/web.sh serve`; interpreter in a Web Worker with an xterm.js console, boots Alpine to a busybox shell in Chrome. No JIT, disk or network.

- UEFI application done (see 12): `crates/uemulator-uefi`, library `no_std` on `target_os = "uefi"`, `hack/uefi/boot.sh`; emulator (interpreter) everywhere, plus a hardware-virtualization path on x86_64 (AMD SVM verified under TCG, Intel VMX untested), no disk/net.

- Interrupt controllers and CPU platform state for Windows done (see 07, "LAPIC and IOAPIC"): xAPIC and x2APIC local APIC with timer and TSC-deadline, 24-pin I/O APIC fed by the same device lines as the PIC, CR8/TPR priority, NMI delivery with `iret` unblocking, MTRRs, PAT validation, inert machine-check banks, AMD memory-typing MSRs, CPUID leaf 0xB. The Alpine kernel boots with the APIC advertised (virtual wire mode, TSC-deadline timer). SMP is done (below). The HPET, the ACPI PM timer and ACPI tables are done (below).

Each has an observable serial-output exit criterion. Interpreter first.

| # | deliverable | exit criterion |
|---|-------------|----------------|
| M0 | API skeleton, config, bzImage parse + loader, example downloads/caches kernel, clock + console + UART | `uemulator quick create` downloads (once), loads, and reports the parsed header; `cargo test` passes |
| M1 | 64-bit decoder + interpreter for the integer subset, exceptions, paging, soft TLB, PIC/PIT/CMOS | console prints `Linux version 6.12.107...` (decompressor finished, kernel `start_kernel` running) |
| M2 | Rest of integer ISA, SSE2 basics, x87 basics, TSC calibration, SMC invalidation, reset/triple-fault exit | no initrd: `Kernel panic - not syncing: VFS: Unable to mount root fs` then clean `ExitReason::Reset`; ≥50 MIPS release |
| M3 | Full initrd boot | `Run /init as init process`, then installer userspace output (SSE2 libc paths, syscalls, signals, fork/exec) |
| M4 | aarch64 JIT: probe, buffer, encoder, lowering, chaining, differential suite | M2 and M3 criteria reached with `JitMode::On`; differential suite green; ≥300 MIPS |
| M5 | Polish: realtime clock mode, stdio raw-mode in example, perf work from the fork list, optional LAPIC | interactive console usable; documented perf numbers |

## M0 task list

1. `Cargo.toml`: `[profile.dev] opt-level = 3`, `[profile.test]` same.
2. `lib.rs` re-exports; `machine.rs`, `console.rs`, `boot.rs`, `clock.rs`, `sys.rs` stubs with the API from 01.
3. `boot.rs`: header parse + validation errors (`Error::BadKernel(..)`), loader into a RAM buffer, boot_params and e820, initial GDT/page tables.
4. Example: cache dir, `curl -fL` to `.part` then rename, args, stdio bridge.
5. CLAUDE.md, README usage.

## M1 task list (first CPU work)

1. `Env`, register file, flags, hflags.
2. Decoder tables for the integer subset; `translate.rs` emitting uops.
3. Interpreter + helpers; `tlb`/`mmu`; exceptions/IDT delivery.
4. Devices from 07; run loop with clock and HLT warp.
5. Iterate on the boot: each unimplemented opcode/MSR/port is logged once; fix in order of appearance.

- ACPI platform done (see 07, "ACPI, PM block, HPET, ECAM"): `acpi/` builds RSDP/XSDT/FADT/DSDT/MADT/MCFG/HPET/IVRS into the BIOS area for direct Linux boots (`boot_params.acpi_rsdp_addr`, e820 ACPI data), with the PM1 block and PM timer, ACPI S5 power-off as `ExitReason::Shutdown`, a three-timer HPET, PCIe ECAM, virtio MSI-X and `MachineConfig::acpi` (default on; the default command line loses `acpi=off`). Alpine 6.18 and Debian 6.12 boot on the I/O APIC with `hpet` and `acpi_pm` clocksource candidates, virtio disk and network work, `poweroff` ends the run through the sleep register. Steps 1 and 2 of 18-xen-dom0.md.
## UEFI firmware (plans/19)

Done: `MachineConfig::uefi_firmware` and `key_value_store`, `crates/uemulator-firmware`, `advanced boot-uefi`, `quick` machines with `firmware = "uefi"`. Exit criteria met in a real guest: the Linux EFI stub and Debian GRUB 2.12 boot Alpine from a GPT ESP, and a variable written by Linux survives a guest reset through the nvram file. Open: Xen (`xen.efi`, multiboot2), Fedora shim, ACPI tables in the handoff.

- SMP done (see 03, "SMP"; step 4 of `plans/18-xen-dom0.md`): `MachineConfig::cpus(n)`, `--cpus N` on the CLI and `cpus` in a `quick` machine's `machine.toml`, 1 to 64, default 1. One `Cpu` per vCPU over one host thread, cooperative round robin with a 4096-instruction quantum, a switch at `hlt`/`mwait`/`pause`, the machine-wide state (RAM, devices, JIT) moved between vCPUs rather than duplicated, a clock that divides instruction counts by the number of running vCPUs, per-vCPU local APICs with IPIs of every addressing mode, INIT/SIPI start-up into real mode, MSI/I/O APIC/IOMMU destinations routed to the right APIC, per-vCPU CPUID ids and topology. Checked in real guests: Alpine 6.18 with 4 and 64 vCPUs on the JIT and 4 on the interpreter (`nproc`, distinct apicids, `Total of N processors activated`, no RCU stalls or watchdog reports, a pthread mmap/mprotect/munmap + spin lock + fork stress and four parallel `md5sum`s correct, CPU offline/online through sysfs), and the Debian trixie 6.12 kernel with `--iommu` (x2APIC cluster mode with interrupt remapping). A parallel workload takes about a third of the guest time of the serial one on 4 vCPUs. Not done: `--verify` with more than one vCPU (refused), firmware and BIOS-disk boots with more than one vCPU (refused), real parallelism on host threads.

- Xen 4.21 boots on the emulated Zen 4 machine (see 18, step 6): GRUB multiboot2 or `xen.efi` from the UEFI firmware, a PVH Alpine dom0 to its login prompt in 26 s of host time, `xl info`, `xl list` and a PVH domU running its initramfs. Seven emulator bugs fixed on the way (compatibility-mode returns, EFER.LMA at #VMEXIT, IOMMU/FCH overlap, stale JIT blocks after SVM exits, MMIO clock skew in the JIT, MADT/`_PRT` polarity, 16550 input pacing). Fedora 44 cloud image boots to a root login through shim and GRUB on 4 vCPUs and installs Xen 4.21.2 with dnf over the NAT; its Xen entry is not yet booted. Open: `xen.efi` through GRUB's chainloader, Fedora Xen dom0.

- JIT performance (see 20): direct block chaining for same-page static exits, inline shl/shr/sar at 8/16 bits, shlx/shrx/sarx/bzhi/andn/blsr group, adc/sbb, dead lazy-flag records skipped, hardware carry-less multiply for pclmulqdq, unmasked EVEX commits by lane, `--debug` histogram of what still reaches the interpreter. The 252 MB Fedora netboot initramfs unpacks in 14.7 s of host time instead of 22.1 s on an Apple Silicon host (`hack/perf/unpack.sh`).
