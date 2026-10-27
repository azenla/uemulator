# Configurable RAM, disk, network, console resize

## Findings that drive the design (Debian 6.12.107 kernel + d-i initrd)

- Built in: `virtio-pci` (PCI bus driver), `virtio_console`, `virtiofs`. Modules: `virtio_net` (in the d-i initrd), `virtio_blk` (not in the d-i initrd, present in a normal installed system), `virtio_mmio` (module).
- Therefore the transport is **virtio over PCI**: the stock kernel probes it with no cmdline help and auto-loads `virtio_net`/`virtio_blk` by modalias.
- With `acpi=off` PCI needs config mechanism #1 (ports 0xCF8/0xCFC) and a host bridge at 00:00.0 (class 0x0600) to pass `pci_sanity_check`. Interrupts: Interrupt Line register preloaded with a legacy PIC IRQ; x86 non-IOAPIC accepts a preset `dev->irq`.

## Memory

`MachineConfig::memory_mib(n)` (already present), validated: 64 ..= 3072 MiB (RAM must stay below the 0xC000_0000 PCI/MMIO window). CLI: `--mem <MiB>`. e820 and the boot page tables follow the configured size; the loader rejects sizes too small for kernel+initrd.

## MMIO + PCI

- Physical accesses at or above RAM size dispatch to `Devices::mmio_read/write` (TLB never caches MMIO pages, so JIT fast paths miss and fall to the helper).
- Device DMA gets a `GuestMem` view (RAM + code-page flags); DMA writes that touch a page holding translated code invalidate it (same path as CPU stores).
- PCI host: bus 0, devfn 0 = i440FX-like host bridge, devfn 8/16/24 = virtio devices (console, blk, net as configured). Each virtio device: vendor 0x1af4, device 0x1040+type, one 64-bit-free 32-bit memory BAR0 of 16 KiB at 0xC000_0000 + n*0x4000 (sizing via all-ones write works), capability list with virtio vendor caps: common cfg (BAR0+0x0000), ISR (0x1000), device cfg (0x2000), notify (0x3000, multiplier 4). Interrupt: one legacy IRQ per device (PIC 9/10/11), asserted while ISR != 0, edge delivered on 0 -> nonzero.
- Feature selectors: Linux 6.18+ negotiates 128 feature bits and reads/writes selectors 2 and 3; those read as zero and writes are ignored (aliasing them onto 0/1 cleared VERSION_1 and the device refused FEATURES_OK).
- Modern (VERSION_1) split virtqueues only; no indirect descriptors, no event idx, no MSI-X.

## Devices (optional, off by default except the console)

- **Disk** (`MachineConfig::disk(Disk)`): raw image. `Disk::open(path, read_only)` (file backed, `read_at`/`write_at`), `Disk::memory(Vec<u8>)`, or any `impl BlockBackend`. virtio-blk: features VERSION_1, FLUSH, SEG_MAX, BLK_SIZE, optional RO; requests IN/OUT/FLUSH/GET_ID, completed synchronously on notify.
- **Network** (`MachineConfig::net(NetConfig)`): virtio-net, features VERSION_1, MAC, STATUS (no offloads), 12-byte header. Backend is a frame channel: `NetHandle` (`send_frame` host->guest, `recv_frame` guest->host, both nonblocking, bounded queues) obtained from `machine.net()`. RX frames are polled on the 4096-instruction event tick and when idle. The example offers `--net-udp local:remote` (frames in UDP datagrams, QEMU `-netdev socket` style) and `--net-loop` (host echoes ARP/ICMP for 10.0.2.2, used by tests). No built-in NAT in the library; the CLI's `--net-user` (`crates/uemulator-cli/src/usernet.rs`) is a SLIRP-style stack over `NetHandle`: ARP for 10.0.2.x, DHCP (guest 10.0.2.15, gateway 10.0.2.2 which maps to host loopback, DNS 10.0.2.3 forwarded to the host resolver), ICMP echo, UDP NAT, and a TCP proxy that terminates guest connections and relays to host sockets (no window scaling, no retransmission: the virtual link is lossless, and frames the guest cannot take yet are retried in order).
- **Console resize**: `ConsoleHandle::resize(cols, rows)` / `size()`. Delivered to the guest through a virtio-console device with `VIRTIO_CONSOLE_F_SIZE` (config space cols/rows + config-change interrupt -> `hvc_resize`). `MachineConfig::console(ConsoleKind::Virtio | ConsoleKind::Serial)`: Serial cannot carry the size. Virtio, the default, makes `/dev/hvc0` the console (`console=hvc0`; the 16550 stays silent during boot) and host input goes to hvc0 once the driver is up. The CLI defaults to `--console virtio` (`--console serial` for kernels where `virtio_console` is a module and the initrd lacks it) and polls `TIOCGWINSZ` every 200 ms. Alpine's virt kernel has virtio_console built in (checked by booting it); Debian is built in per the findings above, Fedora is unchecked. The UEFI emulator path uses the library default (checked in QEMU, aarch64 host); the UEFI hardware-virtualization path stays on the serial console because `Platform` has no virtio.

## Tests

- virtqueue/blk/net/console logic: host-side fake driver writing descriptors into a plain RAM vec (no guest).
- PCI config: sizing, capability walk.
- Guest: `rdinit=/bin/sh` boots that `modprobe virtio_net`, configure 10.0.2.15, ping 10.0.2.2 against the test responder; resize check by reading `stty size` on hvc0 after `resize()`; disk check with `virtio_blk` extracted from the Debian virtio-modules udeb when available (ignored by default).
