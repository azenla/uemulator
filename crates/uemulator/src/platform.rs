#[cfg(target_os = "uefi")]
use crate::prelude::*;

use crate::boot::{self, BootState};
use crate::console::ConsoleHandle;
use crate::dev::guestmem::GuestMem;
use crate::dev::{DevConfig, Devices};
use crate::machine::Error;

/// The default `EmulatedMachine` builds for a serial console. `machine.rs` formats its own copy
/// of this string (with `hvc0` for a virtio console); keep the two in sync.
pub const DEFAULT_CMDLINE: &str =
    "console=ttyS0 earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1";

/// Lays out a bzImage, an optional initramfs, the command line, boot parameters, a GDT and
/// identity-mapped page tables in `ram` (guest physical address 0 is `ram[0]`), and returns the
/// 64-bit entry state. For hosts that start the guest themselves; `EmulatedMachine` calls the
/// same loader.
pub fn load_linux(
    ram: &mut [u8],
    kernel: &[u8],
    initrd: Option<&[u8]>,
    cmdline: &str,
) -> Result<BootState, Error> {
    boot::load(ram, kernel, initrd, cmdline)
}

/// The legacy PC devices (UART, PIC, PIT, CMOS, reset ports) without a CPU, for hosts that run the
/// guest on hardware virtualization and forward port I/O and interrupts to it. Times are
/// nanoseconds on any monotonic clock, as long as it is the same clock for every call.
pub struct Platform {
    dev: Devices,
    /// Empty `page_has_code` map for `GuestMem`: there is no translated code here to invalidate
    /// when a device writes guest RAM by DMA.
    no_code_pages: Vec<u8>,
}

impl Platform {
    pub fn new(console: ConsoleHandle) -> Self {
        Self {
            dev: Devices::new(console, DevConfig::default()),
            no_code_pages: Vec::new(),
        }
    }

    pub fn set_boot_time(&mut self, unix: u64) {
        self.dev.set_boot_time(unix);
    }

    pub fn io_in(&mut self, port: u16, size: u8, now_ns: u64) -> u32 {
        self.dev.io_in(port, size, now_ns)
    }

    pub fn io_out(&mut self, port: u16, size: u8, value: u32, now_ns: u64) {
        self.dev.io_out(port, size, value, now_ns);
    }

    /// Runs due device work. `Devices::tick` takes guest memory for DMA, though nothing in a
    /// `Platform` (no disk, net or virtio console) writes it.
    pub fn tick(&mut self, now_ns: u64, ram: &mut [u8]) {
        let mut mem = GuestMem::new(ram, &self.no_code_pages);
        self.dev.tick(now_ns, &mut mem);
    }

    pub fn irq_pending(&self) -> bool {
        self.dev.pic.has_pending()
    }

    /// Takes the highest-priority pending interrupt and returns its vector.
    pub fn irq_ack(&mut self) -> Option<u8> {
        self.dev.pic.ack()
    }

    /// Earliest time a device needs `tick`, e.g. the next PIT edge; `None` if nothing is scheduled.
    pub fn next_deadline(&self, now_ns: u64) -> Option<u64> {
        self.dev.next_deadline(now_ns)
    }

    /// Set once the guest wrote a reset command (ports 0x64, 0x92 or 0xcf9); stays set.
    pub fn reset_requested(&self) -> bool {
        self.dev.reset_requested
    }
}
