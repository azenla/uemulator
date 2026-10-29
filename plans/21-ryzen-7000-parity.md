# 1:1 with a Ryzen 5 7000-class part

Goal: a guest cannot tell the emulated machine from a desktop or mobile Ryzen 5 of the 7000 series (Zen 4: Raphael 7600/7600X, Phoenix 7640U/7640HS, Hawk Point 8640U; family 19h, models 61h and 74h). Today the identity is an EPYC 9004 Genoa (family 19h model 11h, `plans/03-cpu.md`, CPUID contract), which shares the core but not the SoC.

Nothing below is implemented. Every concrete value for the target (CPUID words, MSR contents, table bytes) is **unverified until Phase 0 captures it from silicon**: do not copy numbers from this file into code, and do not copy them from memory either. QEMU has no Raphael model (nearest: `EPYC-Genoa`), so there is no reference to crib from.

## What "1:1" means, and what it cannot

Four levels, each with its own test, checked in order:

1. **Identity.** CPUID (every leaf and subleaf, per logical processor) and the readable MSR set are bit-identical to the reference, except where level 4 says otherwise.
2. **Architecture.** Every instruction the part executes behaves as it does on silicon: results, flags (including the "undefined" ones), exceptions and their error codes, faulting order, alignment and canonicality rules, FP rounding and denormal handling, MXCSR and x87 status, AVX-512 masking and exception suppression.
3. **Platform.** What firmware and the OS enumerate: ACPI tables, PCI topology and config space, the MSR/SMN-visible fabric, the IOMMU, timers and their frequencies, power and thermal interfaces that exist.
4. **Declared departures**, kept in one list in this file and mirrored in `cpuid.rs`: things an in-order, non-speculating emulator cannot be. These are not bugs, but each must be a decision with a reason.

Cannot match, by construction: instruction latency and throughput, speculation and the side channels around it (so mitigation bits keep claiming "not affected" only where that is true of this implementation), exact cache and TLB replacement, performance-counter and IBS values (counts need a microarchitecture model), thermal and power telemetry, microcode behaviour beyond a plausible patch level, SMM, SEV.

## Gap list, from the code as it stands

Identity (`cpu/cpuid.rs`, `cpu/sys.rs`):
- Signature `0x00A10F11`, brand `AMD EPYC-Genoa Processor`. Target is model 61h stepping 2 (Raphael) or 74h (Phoenix) with its marketing string, per part. Linux and Xen key errata, mitigations and perf drivers on family and model, so the model change alone moves guest behaviour; it needs the Phase 3 re-check below.
- Not advertised although Zen 4 has them (each needs instructions and tests first): SSE4A, MMXEXT, FFXSR, MWAITX, LA57 (5-level paging), PKU/OSPKE, CET shadow stacks, OSVW, EXTAPIC, CR8LEGACY, RDPRU, TSC_ADJUST if the part reports it, the AMD performance-counter extensions (PERFCTR_CORE/NB/L2), IBS.
- Advertised although the part lacks them: TSC-deadline timer, the Intel spelling of the speculation-control leaf and `IA32_ARCH_CAPABILITIES` (`plans/03-cpu.md`, "Deliberate departures"). Real parts expose AMD's spellings (leaf 0x80000008 EBX, `PRED_CMD`/`SPEC_CTRL` MSRs, `VIRT_SPEC_CTRL` under SVM).
- Address sizes 40/48 where the part reports 48/57 (physical limit of the RAM model, not a hardware choice).
- Topology leaves model one die; the part's CCD/CCX layout (Raphael: one CCD of 6 enabled cores of 8, one 32 MiB L3; Phoenix: monolithic, 16 MiB L3) needs `0x8000001D/E` and `0xB` driven by the chosen SKU instead of constants.
- MSR gaps: P-state and CPPC MSRs (`0xC0010061..`, `0xC00102B0..`), RAPL energy (`0xC001029A..`), HWCR/SYSCFG bit behaviour beyond storage, MCA bank layout (bank count and the IPID/SYND/DESTAT registers), LBR/BRS, `0xC0011020..` family of configuration MSRs that must read what Linux expects.

Architecture (`cpu/*.rs`):
- The AVX-512 set is implemented; confirm the *undefined-flag* and *denormal* behaviour of every scalar and vector instruction against silicon, not the SDM/APM text, which leaves the AMD-specific results open (`bsf/bsr` with zero source, `shld/shrd` counts above the operand size, `div/idiv` flags, `bt` with register offsets, `aam/aad`, `sahf`/`lahf` in long mode, `rcl/rcr` OF).
- Instruction timing-dependent behaviour that software can observe: `rdtsc` ordering, `lfence` serialization, MWAIT/MONITOR wake rules, and self-modifying code visibility (AMD does not require a serializing instruction for SMC on the same core; the TB cache already tracks it).

Platform (`acpi/`, `dev/pci.rs`, `dev/iommu`):
- ACPI is a generic virtual PC (FADT rev 6, MADT, MCFG, HPET, DSDT with virtio, IVRS). A Ryzen exposes more: FACP with the AMD flag set, SRAT/SLIT (one node: absent or trivial), CRAT, CDIT, a PCI root complex with the real device scope, `_CPC`/CPPC objects, `_OSC`, `_PSS`/`_PCT` legacy P-states if the BIOS offers them, EC and GPIO devices on mobile parts, ASPM and `_DSM` objects Linux probes.
- PCI: the host bridge (AMD vendor id, the SoC's device ids), the IOMMU function (already 00:00.2), root ports, and the fixed functions every Ryzen has at 00:01/00:02 and 00:08 (GPP bridges, PSP, SMBus/LPC on a chipset-less view). Their config-space bytes matter to Linux quirks; their *absence* is also observable.
- Timers: the part's TSC runs at the P0 frequency (not 1 GHz as here), `0xC0010064` reports it, and Linux calibrates against the HPET (14.318 MHz) or PM timer (3.579545 MHz). The virtual clock model (`plans/07-devices.md`) must expose a TSC frequency and a bus-clock ratio consistent with the chosen SKU, and the LAPIC timer must tick at the bus clock the part reports.
- Fabric registers reached through SMN and index/data pairs (`0x60/0x64` on the host bridge): Linux reads thermal and `amd_nb` data from them; unknown reads must return what the part returns, not 0xffffffff.

## Method

The repo's rules apply: the interpreter is the oracle for the JIT, a bit is advertised only with its tests, tests are unit tests with no network and no files under `~/.uemulator`. Real-hardware comparison data is **checked in as small fixtures**, not fetched by tests.

### Phase 0: capture ground truth (needs the real part; blocks everything else)

A collector in `hack/ryzen/` that runs on the reference machine under Linux and writes one directory of plain text:
- `cpuid.txt`: every leaf in `0..=0x20`, `0x40000000..`, `0x80000000..=0x80000026`, each subleaf until it repeats, **on every logical CPU** (leaf 1 EBX, 0xB, 0x8000001E differ per CPU), via `/dev/cpu/N/cpuid`.
- `msr.txt`: `rdmsr` of every MSR in a candidate list (the union of what Linux, Xen and this emulator read, plus ranges `0xC0010000-0xC00102FF`, `0xC0011000-0xC00110FF`), recording both the value and a #GP result.
- `acpi/` (all tables, `acpidump`, plus the decompiled DSDT/SSDTs), `lspci -xxxx -vvv` for every function, `dmesg` of a cold boot, `/proc/cpuinfo`, `/proc/iomem`, `/proc/ioports`, `lscpu -e`.
- `smn.txt`: reads of the SMN registers Linux's `k10temp` and `amd_nb` touch.
Fixtures from the collector go under `crates/uemulator/tests-data/ryzen-<sku>/` (check the policy on binary size first). A second SKU's directory is how the method generalises.

### Phase 1: differential harness for the ISA

- A generator that emits small guest programs (one instruction form, operand patterns that hit boundaries: zero, all ones, sign boundaries, denormals, NaN payloads, shift counts around the width, misaligned and page-straddling memory), run on the **host CPU** when it is the reference part, or from recorded results otherwise.
- Output is the full architectural state after each snippet; the same snippet runs in the emulator through `testkit::Harness` (interpreter and JIT) and the states must match. Divergences are triaged into: emulator bug (fix, test first), documented departure (list it), or reference noise (a flag or register the part leaves undefined and varies run to run: record it as such and mask it).
- Build on `hack/asm.sh` and `testkit::assert_jit_matches_interpreter`; extend with Sandsifter-style exhaustive opcode scanning for the decode side (which byte sequences #UD, which are NOPs, which length). Target: decode agreement on 100% of scanned encodings below an agreed prefix depth.
- Exit: zero unexplained divergences over integer, x87, SSE through AVX-512 and the system instructions (`rdmsr/wrmsr` faulting, `cpuid` leaves, `xsave` formats for the part's XCR0 set).

### Phase 2: identity

- `CpuProfile` in `cpu/profile.rs`: one struct holding signature, brand string, per-leaf words, MSR defaults, topology and cache geometry, selected by `MachineConfig::cpu_model` (default stays Genoa until the Ryzen profile passes its tests; `ryzen-7600` etc. added as data, not branches). `cpuid.rs` reads the profile instead of constants.
- A test per profile compares `cpuid` over the Phase 0 fixture, leaf by leaf, per logical CPU for the SKU's core and thread count. Bits for features the emulator lacks are listed in the departures table with the guard that clears them; the day the feature lands and the guard is removed, the test turns the bit on.
- Features are added one at a time in this order, each with instructions, tests and CPUID in one commit: SSE4A and MMXEXT/FFXSR (small), MWAITX, LA57, PKU, RDPRU, CET shadow stacks, then PMU basics (`PERFCTR_CORE` counting retired instructions and branches deterministically).
- MSR defaults: every MSR in `msr.txt` either reads the recorded value (stored, inert, or computed), or is listed as unimplemented with #GP matching the part's recorded #GP.

### Phase 3: guest software sanity on the new identity

Changing family/model changes what Linux and Xen do, so these checks run before the default flips:
- Boot the Alpine and Debian installers and a Fedora Xen dom0 (`plans/18-xen-dom0.md`) and compare `dmesg` against the Phase 0 `dmesg` after normalising: same CPU feature line, same mitigations and "Vulnerable"/"Not affected" lines in `/sys/devices/system/cpu/vulnerabilities/`, same `lscpu` flags (minus declared departures), same `amd_pstate`/`acpi-cpufreq` choice, same `k10temp` probe outcome.
- Run `kvm-unit-tests` (x86 subset) against the emulated SVM through `hack/kvm/run.sh` and Xen's XTF for what the platform claims.

### Phase 4: platform

- ACPI: generate the tables from a per-SKU description checked against the captured `acpi/` (field-by-field diff test, with a short list of fields that must differ: addresses of our own devices, OEM ids, checksums).
- PCI: add the fixed functions with captured config-space contents; unimplemented BAR-backed fabric devices answer with captured defaults and ignore writes, logging each distinct access once (`log_once`) so missing behaviour is found by boots, not guessed.
- Timers: SKU-specific TSC frequency and P-state MSRs; keep guest time realtime (`plans/07-devices.md`) so the frequency is honest in wall time. Re-derive the PIT, HPET, PM timer and LAPIC rates from the same constants and test the ratios.
- IOMMU: confirm the AMD-Vi feature bits and extended feature register match the part (`plans/16-iommu.md`).

### Phase 5: keep it true

- CI runs the fixture comparisons (cpuid, msr, acpi, pci) on every change: they are fast unit tests.
- A "reference refresh" doc in `hack/ryzen/README.md`: how to add a SKU, what to re-capture when a BIOS or microcode update changes an answer.
- The departures list is generated into the docs from the code (`DEPARTURES` table in `profile.rs` with a one-line reason each) so it cannot drift.

## Order of work and sizing

| step | content | blocked by |
|------|---------|------------|
| 0 | collector script, captured fixtures for one 7000-series part | access to the machine |
| 1 | differential ISA harness, first divergence triage | 0 for host-run mode; independent for recorded mode |
| 2a | `CpuProfile`, Genoa expressed as a profile (no behaviour change) | none |
| 2b | Ryzen profile data, CPUID/MSR fixture tests, missing features one by one | 0, 2a |
| 3 | guest boots on the new identity, dmesg/vulnerabilities diff | 2b |
| 4 | ACPI/PCI/timer parity | 0, 3 |
| 5 | CI, refresh docs, departures table | 2-4 |

Step 2a and the harness skeleton of step 1 can start without the hardware. Everything else waits for Phase 0 data.

## Open questions

- Which exact SKU is the reference (7600X Raphael vs 7640U Phoenix differ in model, cache, topology, platform devices and ACPI)? The method is per-SKU data, but Phase 0 needs one machine to read.
- Is TSC-deadline kept as a declared departure (fast calibration, no AMD guest relies on its absence) or removed for fidelity? Removing it changes guest timer paths to the LAPIC one-shot timer; measure the boot cost first.
- Do we model CPPC/`amd_pstate` (needs the P-state and CPPC MSRs plus `_CPC`) or hide them like a part with the feature disabled in BIOS? Both exist in the field.
- Fixture size and licensing: ACPI tables and PCI config dumps from OEM boards carry vendor strings. Capture from a part we can publish, or scrub OEM ids.
