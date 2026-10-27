# uemulator plans

Index. Read in order; each file ends with open questions where the answer needs checking against the QEMU tree or the kernel docs.

| file | topic |
|------|-------|
| 00-overview.md | goals, non-goals, architecture, crate layout, key decisions |
| 01-api.md | public embedding API, console, exit reasons, threading |
| 02-boot.md | the target kernel, direct 64-bit boot, boot_params, e820 |
| 03-cpu.md | x86_64 state, decoder, lazy flags, exceptions, CPUID/MSR contract |
| 04-memory.md | physical memory, bus, soft TLB, page walk, SMC tracking |
| 05-ir-interp.md | micro-op IR, TB cache, interpreter |
| 06-jit.md | W^X handling, aarch64 backend, probe/fallback, chaining |
| 07-devices.md | serial, PIC, PIT, CMOS, clock, reset, gotchas |
| 08-testing.md | unit, differential, boot tests |
| 09-qemu-map.md | which QEMU files to read for which subsystem; fork notes |
| 10-milestones.md | ordered milestones with observable exit criteria |
| 11-storage-net-console.md | configurable RAM, virtio-pci disk/net/console (resize) |
| 12-uefi.md | UEFI application, `no_std` port, hardware-MMU analysis |
| 13-web.md | WebAssembly port |
| 14-avx-and-virtualization.md | designs for AVX/AVX2/FMA, SVM and VMX (not implemented) |

Rules for implementers: `../ext/qemu-uplink` is read-only reference. Behaviour and algorithms are studied there; code is written fresh in Rust, not transliterated. Zero crate dependencies.
