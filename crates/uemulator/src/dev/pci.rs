use super::guestmem::GuestMem;
use super::size_mask;
use super::virtio::{
    COMMON_CFG_OFFSET, DEVICE_CFG_OFFSET, DEVICE_TYPE_BLOCK, DEVICE_TYPE_CONSOLE, DEVICE_TYPE_NET,
    ISR_OFFSET, NOTIFY_OFF_MULTIPLIER, NOTIFY_OFFSET, REGION_SIZE, Transport,
};
#[cfg(target_os = "uefi")]
use crate::prelude::*;

/// Start of the MMIO window; everything at or above guest RAM size dispatches to the devices.
/// RAM is capped at 0xC000_0000 so the two never overlap (plans/11).
pub(crate) const MMIO_BASE: u64 = 0xc000_0000;
/// Each virtio function gets one BAR0 of this size, laid out back to back from `MMIO_BASE`.
/// Four `REGION_SIZE` regions: common cfg, ISR, device cfg, notify.
const BAR_SIZE: u64 = 4 * REGION_SIZE;
/// Legacy PIC lines handed out to functions in order. There is no IOAPIC or ACPI, so the
/// kernel takes `int_line` as the IRQ. Never 0, `mmio_write` uses 0 for "no interrupt".
const LEGACY_IRQS: [u8; 3] = [9, 10, 11];

/// Config mechanism #1 ports: the address dword at 0xCF8..=0xCFB and the data window at
/// 0xCFC..=0xCFF. Either may be accessed as a byte, word or dword; the low two bits of the
/// port are the byte lane. Port 0xCF9 is also the reset control register, which `Devices`
/// claims before it gets here.
pub(crate) const CONFIG_ADDRESS_PORT: u16 = 0xcf8;
pub(crate) const CONFIG_DATA_PORT: u16 = 0xcfc;
const BYTES_PER_DWORD: usize = 4;
const BYTE_LANE_MASK: u16 = 3;
const CONFIG_ADDRESS_ENABLE: u32 = 1 << 31;
const CONFIG_ADDRESS_BUS_SHIFT: u32 = 16;
const CONFIG_ADDRESS_DEVFN_SHIFT: u32 = 8;
/// Dword-aligned register number in the low byte; the low two bits are the byte offset
/// within the dword and come from the data port instead.
const CONFIG_ADDRESS_REGISTER_MASK: u32 = 0xfc;
/// devfn = slot << 3 | function.
const DEVFN_SLOT_SHIFT: u32 = 3;
/// Slot 0 is the host bridge, virtio functions start at slot 1.
const FIRST_VIRTIO_SLOT: u8 = 1;
const CONFIG_SPACE_SIZE: usize = 256;

/// Standard config header offsets.
const CFG_VENDOR_ID: usize = 0x00;
const CFG_DEVICE_ID: usize = 0x02;
const CFG_COMMAND: usize = 0x04;
const CFG_COMMAND_HIGH_BYTE: usize = CFG_COMMAND + 1;
const CFG_STATUS: usize = 0x06;
const CFG_REVISION: usize = 0x08;
const CFG_SUBCLASS: usize = 0x0a;
const CFG_CLASS: usize = 0x0b;
const CFG_BAR0: usize = 0x10;
const CFG_BAR0_LAST_BYTE: usize = CFG_BAR0 + 3;
const CFG_SUBSYSTEM_VENDOR_ID: usize = 0x2c;
const CFG_SUBSYSTEM_ID: usize = 0x2e;
const CFG_CAPABILITIES_POINTER: usize = 0x34;
const CFG_INTERRUPT_LINE: usize = 0x3c;
const CFG_INTERRUPT_PIN: usize = 0x3d;

const COMMAND_MEMORY_SPACE: u16 = 1 << 1;
const COMMAND_INTX_DISABLE: u16 = 1 << 10;
const STATUS_CAPABILITIES_LIST: u16 = 1 << 4;
/// Interrupt pin register value for INTA#.
const INTERRUPT_PIN_INTA: u8 = 1;

const VENDOR_INTEL: u16 = 0x8086;
/// 440FX host bridge (QEMU's default), only needs to exist and have class 0x0600 so the
/// kernel's `pci_sanity_check` accepts config mechanism #1.
const DEVICE_I440FX_HOST: u16 = 0x1237;
const CLASS_BRIDGE: u8 = 0x06;
const SUBCLASS_HOST_BRIDGE: u8 = 0x00;

const VENDOR_VIRTIO: u16 = 0x1af4;
/// Modern (non-transitional) PCI device id is this plus the virtio device type.
const VIRTIO_PCI_DEVICE_ID_BASE: u16 = 0x1040;
/// Non-transitional virtio devices report revision 1 or higher.
const VIRTIO_PCI_REVISION: u8 = 1;
/// Class codes follow QEMU: net is ethernet, block is SCSI storage, everything else (console)
/// is "communication, other".
const CLASS_NETWORK: u8 = 0x02;
const SUBCLASS_ETHERNET: u8 = 0x00;
const CLASS_STORAGE: u8 = 0x01;
const SUBCLASS_SCSI: u8 = 0x00;
const CLASS_COMMUNICATION: u8 = 0x07;
const SUBCLASS_COMMUNICATION_OTHER: u8 = 0x80;

/// Vendor-specific capability id used by every virtio PCI capability.
const CAP_ID_VENDOR_SPECIFIC: u8 = 0x09;
/// `cfg_type` of the virtio capabilities we expose (`VIRTIO_PCI_CAP_*`). The PCI-config-access
/// capability (type 5) is not provided.
const VIRTIO_PCI_CAP_COMMON_CFG: u8 = 1;
const VIRTIO_PCI_CAP_NOTIFY_CFG: u8 = 2;
const VIRTIO_PCI_CAP_ISR_CFG: u8 = 3;
const VIRTIO_PCI_CAP_DEVICE_CFG: u8 = 4;
/// Capability structure: id, next, len, cfg_type, bar (+4), padding, offset u32 (+8), length
/// u32 (+12). The notify capability adds a u32 multiplier at +16, so it is 20 bytes, others 16.
const VIRTIO_CAP_LEN: u8 = 16;
const VIRTIO_NOTIFY_CAP_LEN: u8 = 20;
const VIRTIO_CAP_OFFSET_FIELD: usize = 8;
const VIRTIO_CAP_LENGTH_FIELD: usize = 12;
const VIRTIO_CAP_NOTIFY_MULTIPLIER_FIELD: usize = 16;
/// The capability chain is fixed: common -> notify -> ISR -> device, each placed right after the previous.
const CAP_COMMON_AT: usize = 0x40;
const CAP_NOTIFY_AT: usize = 0x50;
const CAP_ISR_AT: usize = 0x64;
const CAP_DEVICE_AT: usize = 0x74;
/// `struct virtio_pci_common_cfg` is 56 bytes.
const COMMON_CFG_LENGTH: u32 = 0x38;
const ISR_LENGTH: u32 = 4;
const DEVICE_CFG_LENGTH: u32 = 0x100;

/// One virtio device on bus 0. Config space is synthesized on every read from this state.
struct PciFunction {
    transport: Transport,
    devfn: u8,
    /// Preassigned at construction because there is no firmware to program it; the guest may relocate it.
    bar0: u32,
    command: u16,
    int_line: u8,
}

pub(crate) struct PciHost {
    /// Value of the address dword at 0xCF8.
    config_address: u32,
    funcs: Vec<PciFunction>,
}

/// Fills in one virtio vendor capability at `at`; `next` is the config-space offset of the
/// following capability (0 ends the list). All regions live in BAR0 (the `bar` byte stays 0).
fn write_virtio_cap(
    config: &mut [u8; CONFIG_SPACE_SIZE],
    at: usize,
    next: usize,
    len: u8,
    cfg_type: u8,
    region_offset: u32,
    region_length: u32,
) {
    config[at] = CAP_ID_VENDOR_SPECIFIC;
    config[at + 1] = next as u8;
    config[at + 2] = len;
    config[at + 3] = cfg_type;
    config[at + 4] = 0;
    config[at + VIRTIO_CAP_OFFSET_FIELD..at + VIRTIO_CAP_OFFSET_FIELD + 4]
        .copy_from_slice(&region_offset.to_le_bytes());
    config[at + VIRTIO_CAP_LENGTH_FIELD..at + VIRTIO_CAP_LENGTH_FIELD + 4]
        .copy_from_slice(&region_length.to_le_bytes());
}

impl PciFunction {
    fn config(&self) -> [u8; CONFIG_SPACE_SIZE] {
        let mut b = [0u8; CONFIG_SPACE_SIZE];
        let device_type = self.transport.dev.device_type();
        let device_id = VIRTIO_PCI_DEVICE_ID_BASE + device_type;
        b[CFG_VENDOR_ID..CFG_VENDOR_ID + 2].copy_from_slice(&VENDOR_VIRTIO.to_le_bytes());
        b[CFG_DEVICE_ID..CFG_DEVICE_ID + 2].copy_from_slice(&device_id.to_le_bytes());
        b[CFG_COMMAND..CFG_COMMAND + 2].copy_from_slice(&self.command.to_le_bytes());
        b[CFG_STATUS..CFG_STATUS + 2].copy_from_slice(&STATUS_CAPABILITIES_LIST.to_le_bytes());
        b[CFG_REVISION] = VIRTIO_PCI_REVISION;
        let (subclass, class) = match device_type {
            DEVICE_TYPE_NET => (SUBCLASS_ETHERNET, CLASS_NETWORK),
            DEVICE_TYPE_BLOCK => (SUBCLASS_SCSI, CLASS_STORAGE),
            _ => (SUBCLASS_COMMUNICATION_OTHER, CLASS_COMMUNICATION),
        };
        b[CFG_SUBCLASS] = subclass;
        b[CFG_CLASS] = class;
        b[CFG_BAR0..CFG_BAR0 + 4].copy_from_slice(&self.bar0.to_le_bytes());
        b[CFG_SUBSYSTEM_VENDOR_ID..CFG_SUBSYSTEM_VENDOR_ID + 2]
            .copy_from_slice(&VENDOR_VIRTIO.to_le_bytes());
        b[CFG_SUBSYSTEM_ID..CFG_SUBSYSTEM_ID + 2].copy_from_slice(&device_type.to_le_bytes());
        b[CFG_CAPABILITIES_POINTER] = CAP_COMMON_AT as u8;
        b[CFG_INTERRUPT_LINE] = self.int_line;
        b[CFG_INTERRUPT_PIN] = INTERRUPT_PIN_INTA;
        write_virtio_cap(
            &mut b,
            CAP_COMMON_AT,
            CAP_NOTIFY_AT,
            VIRTIO_CAP_LEN,
            VIRTIO_PCI_CAP_COMMON_CFG,
            COMMON_CFG_OFFSET as u32,
            COMMON_CFG_LENGTH,
        );
        write_virtio_cap(
            &mut b,
            CAP_NOTIFY_AT,
            CAP_ISR_AT,
            VIRTIO_NOTIFY_CAP_LEN,
            VIRTIO_PCI_CAP_NOTIFY_CFG,
            NOTIFY_OFFSET as u32,
            REGION_SIZE as u32,
        );
        let multiplier_at = CAP_NOTIFY_AT + VIRTIO_CAP_NOTIFY_MULTIPLIER_FIELD;
        b[multiplier_at..multiplier_at + 4].copy_from_slice(&NOTIFY_OFF_MULTIPLIER.to_le_bytes());
        write_virtio_cap(
            &mut b,
            CAP_ISR_AT,
            CAP_DEVICE_AT,
            VIRTIO_CAP_LEN,
            VIRTIO_PCI_CAP_ISR_CFG,
            ISR_OFFSET as u32,
            ISR_LENGTH,
        );
        write_virtio_cap(
            &mut b,
            CAP_DEVICE_AT,
            0,
            VIRTIO_CAP_LEN,
            VIRTIO_PCI_CAP_DEVICE_CFG,
            DEVICE_CFG_OFFSET as u32,
            DEVICE_CFG_LENGTH,
        );
        b
    }

    /// Byte-wise so sub-dword writes to the data port work. Only the command register, BAR0
    /// and the interrupt line are writable; everything else is read-only.
    fn write_config(&mut self, reg: usize, size: usize, val: u32) {
        for k in 0..size {
            let byte_reg = reg + k;
            let byte = (val >> (8 * k)) as u8;
            match byte_reg {
                CFG_COMMAND => self.command = (self.command & 0xff00) | byte as u16,
                CFG_COMMAND_HIGH_BYTE => {
                    self.command = (self.command & 0x00ff) | ((byte as u16) << 8)
                }
                CFG_BAR0..=CFG_BAR0_LAST_BYTE => {
                    let shift = 8 * (byte_reg - CFG_BAR0);
                    // Low bits are hardwired to zero: this is what makes the all-ones sizing
                    // write read back as !(BAR_SIZE - 1).
                    let size_mask = !((BAR_SIZE - 1) as u32);
                    self.bar0 = (self.bar0 & !(0xff << shift)) | ((byte as u32) << shift);
                    self.bar0 &= size_mask;
                }
                CFG_INTERRUPT_LINE => self.int_line = byte,
                _ => {}
            }
        }
    }

    fn memory_space_enabled(&self) -> bool {
        self.command & COMMAND_MEMORY_SPACE != 0
    }

    /// Offset into BAR0 if `phys_addr` is decoded by this function, honoring the command
    /// register's memory-space enable and an unassigned (zero) BAR.
    fn bar0_offset(&self, phys_addr: u64) -> Option<u64> {
        let base = self.bar0 as u64 & !(BAR_SIZE - 1);
        (self.memory_space_enabled()
            && base != 0
            && phys_addr >= base
            && phys_addr < base + BAR_SIZE)
            .then(|| phys_addr - base)
    }

    /// False when the guest set the command register's INTx-disable bit.
    fn intx_enabled(&self) -> bool {
        self.command & COMMAND_INTX_DISABLE == 0
    }
}

impl PciHost {
    pub fn new(devs: Vec<Transport>) -> Self {
        let funcs = devs
            .into_iter()
            .enumerate()
            .map(|(i, transport)| PciFunction {
                transport,
                devfn: ((i as u8 + FIRST_VIRTIO_SLOT) << DEVFN_SLOT_SHIFT),
                bar0: (MMIO_BASE + i as u64 * BAR_SIZE) as u32,
                command: 0,
                int_line: LEGACY_IRQS[i % LEGACY_IRQS.len()],
            })
            .collect();
        PciHost {
            config_address: 0,
            funcs,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.funcs.is_empty()
    }

    /// True once a virtio-console driver is up; the UART stops taking host input then so
    /// keystrokes go to hvc0 only.
    pub fn console_active(&self) -> bool {
        self.funcs.iter().any(|f| {
            f.transport.dev.device_type() == DEVICE_TYPE_CONSOLE && f.transport.driver_ok()
        })
    }

    fn host_bridge() -> [u8; CONFIG_SPACE_SIZE] {
        let mut b = [0u8; CONFIG_SPACE_SIZE];
        b[CFG_VENDOR_ID..CFG_VENDOR_ID + 2].copy_from_slice(&VENDOR_INTEL.to_le_bytes());
        b[CFG_DEVICE_ID..CFG_DEVICE_ID + 2].copy_from_slice(&DEVICE_I440FX_HOST.to_le_bytes());
        b[CFG_SUBCLASS] = SUBCLASS_HOST_BRIDGE;
        b[CFG_CLASS] = CLASS_BRIDGE;
        b
    }

    /// `byte` is the byte lane within the dword (low two bits of the data port). Absent
    /// devices read all ones.
    fn config_read(&self, addr: u32, size: usize, byte: usize) -> u32 {
        let (bus, devfn, reg) = (
            (addr >> CONFIG_ADDRESS_BUS_SHIFT) & 0xff,
            ((addr >> CONFIG_ADDRESS_DEVFN_SHIFT) & 0xff) as u8,
            (addr & CONFIG_ADDRESS_REGISTER_MASK) as usize,
        );
        let ones = size_mask(size as u8);
        if bus != 0 {
            return ones;
        }
        let config = if devfn == 0 {
            Self::host_bridge()
        } else if let Some(f) = self.funcs.iter().find(|f| f.devfn == devfn) {
            f.config()
        } else {
            return ones;
        };
        let dword = u32::from_le_bytes(config[reg..reg + 4].try_into().unwrap());
        (dword >> (8 * byte)) & ones
    }

    pub fn io_in(&self, port: u16, size: u8) -> u32 {
        let lane = (port.wrapping_sub(CONFIG_ADDRESS_PORT) & BYTE_LANE_MASK) as usize;
        if port < CONFIG_DATA_PORT {
            return (self.config_address >> (8 * lane)) & size_mask(size);
        }
        if self.config_address & CONFIG_ADDRESS_ENABLE == 0 {
            return size_mask(size);
        }
        self.config_read(self.config_address, size as usize, lane)
    }

    pub fn io_out(&mut self, port: u16, size: u8, val: u32) {
        let lane = (port.wrapping_sub(CONFIG_ADDRESS_PORT) & BYTE_LANE_MASK) as usize;
        // An access reaching past the dword is clamped; the rest would belong to the next port.
        let bytes = (size as usize).min(BYTES_PER_DWORD - lane);
        if port < CONFIG_DATA_PORT {
            // A narrow write replaces only its own bytes of the address register.
            let mask = (size_mask(bytes as u8) as u64) << (8 * lane);
            let merged = (self.config_address as u64 & !mask) | ((val as u64) << (8 * lane) & mask);
            self.config_address = merged as u32;
            return;
        }
        let bus = (self.config_address >> CONFIG_ADDRESS_BUS_SHIFT) & 0xff;
        if self.config_address & CONFIG_ADDRESS_ENABLE == 0 || bus != 0 {
            return;
        }
        let devfn = ((self.config_address >> CONFIG_ADDRESS_DEVFN_SHIFT) & 0xff) as u8;
        let reg = (self.config_address & CONFIG_ADDRESS_REGISTER_MASK) as usize + lane;
        if let Some(f) = self.funcs.iter_mut().find(|f| f.devfn == devfn) {
            f.write_config(reg, bytes, val);
        }
    }

    fn find_by_address(&mut self, phys_addr: u64) -> Option<(&mut PciFunction, u64)> {
        self.funcs
            .iter_mut()
            .find_map(|f| f.bar0_offset(phys_addr).map(|offset| (f, offset)))
    }

    /// IRQ numbers the functions can be wired to.
    pub fn interrupt_lines() -> [u8; 3] {
        LEGACY_IRQS
    }

    /// Whether any function wired to `irq` is still asserting its interrupt: the ISR has a bit
    /// set (it clears on read) and the guest has not disabled INTx. The PIC needs this to drop a
    /// level-triggered line once the driver has acknowledged the device.
    pub fn line_asserted(&self, irq: u8) -> bool {
        self.funcs
            .iter()
            .any(|f| f.int_line == irq && f.intx_enabled() && f.transport.isr != 0)
    }

    /// None if no BAR decodes `phys_addr`.
    pub fn mmio_read(&mut self, phys_addr: u64, size: u8) -> Option<u32> {
        self.find_by_address(phys_addr)
            .map(|(f, offset)| f.transport.mmio_read(offset, size))
    }

    /// None if no BAR decodes `phys_addr`; otherwise the PIC IRQ to raise, or 0 for none.
    pub fn mmio_write(
        &mut self,
        phys_addr: u64,
        size: u8,
        val: u32,
        mem: &mut GuestMem,
    ) -> Option<u8> {
        let (f, offset) = self.find_by_address(phys_addr)?;
        let raise = f.transport.mmio_write(offset, size, val, mem);
        Some(if raise && f.intx_enabled() {
            f.int_line
        } else {
            0
        })
    }

    /// Appends the IRQ of every function that has a new interrupt to `raise`.
    pub fn poll(&mut self, mem: &mut GuestMem, raise: &mut Vec<u8>) {
        for f in &mut self.funcs {
            if f.transport.poll(mem) && f.intx_enabled() {
                raise.push(f.int_line);
            }
        }
    }
}

#[cfg(test)]
#[path = "pci_tests.rs"]
mod tests;
