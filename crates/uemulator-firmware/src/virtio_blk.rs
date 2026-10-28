//! A polling driver for modern virtio-blk over PCI, enough for the boot path: one queue, one
//! request in flight, no interrupts. The device model is `crates/uemulator/src/dev/virtio.rs`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

use crate::memory;
use crate::pci::{self, Function};
use crate::uefi::memory_type::BOOT_SERVICES_DATA;

const VENDOR_VIRTIO: u16 = 0x1af4;
/// Modern block device: `0x1040` plus device type 2.
const DEVICE_ID_BLOCK: u16 = 0x1042;
pub const SECTOR_BYTES: usize = 512;
/// Largest transfer in one request; the device allows 16 MiB, this keeps buffers modest.
const MAX_REQUEST_BYTES: usize = 1 << 20;

const CAP_ID_VENDOR: u8 = 0x09;
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_DEVICE: u8 = 4;
const CAP_TYPE: u8 = 3;
const CAP_BAR: u8 = 4;
const CAP_OFFSET: u8 = 8;
const CAP_NOTIFY_MULTIPLIER: u8 = 16;

const COMMON_DEVICE_FEATURE_SELECT: usize = 0x00;
const COMMON_DEVICE_FEATURE: usize = 0x04;
const COMMON_DRIVER_FEATURE_SELECT: usize = 0x08;
const COMMON_DRIVER_FEATURE: usize = 0x0c;
const COMMON_DEVICE_STATUS: usize = 0x14;
const COMMON_QUEUE_SELECT: usize = 0x16;
const COMMON_QUEUE_SIZE: usize = 0x18;
const COMMON_QUEUE_ENABLE: usize = 0x1c;
const COMMON_QUEUE_NOTIFY_OFFSET: usize = 0x1e;
const COMMON_QUEUE_DESC: usize = 0x20;
const COMMON_QUEUE_DRIVER: usize = 0x28;
const COMMON_QUEUE_DEVICE: usize = 0x30;

const STATUS_ACKNOWLEDGE: u8 = 1;
const STATUS_DRIVER: u8 = 2;
const STATUS_DRIVER_OK: u8 = 4;
const STATUS_FEATURES_OK: u8 = 8;
/// `VIRTIO_F_VERSION_1` is bit 32: bit 0 of feature word 1.
const FEATURES_WORD_VERSION_1: u32 = 1;
const FEATURE_RO: u32 = 1 << 5;

const QUEUE_SIZE: u16 = 16;
const DESC_BYTES: usize = 16;
const DESC_F_NEXT: u16 = 1;
const DESC_F_WRITE: u16 = 2;
const REQUEST_IN: u32 = 0;
const REQUEST_OUT: u32 = 1;
const REQUEST_FLUSH: u32 = 4;
const HEADER_BYTES: usize = 16;
const STATUS_OK: u8 = 0;
/// The device answers during the notify write in this emulator; this bound is for a device that
/// never does.
const POLL_LIMIT: u32 = 50_000_000;
const DEVICE_CONFIG_CAPACITY: usize = 0;

/// One page for the request header and status byte, then the rings.
const RING_PAGES: u64 = 3;
const SCRATCH_OFFSET: usize = 0;
const AVAIL_OFFSET: usize = DESC_BYTES * QUEUE_SIZE as usize;
const USED_OFFSET: usize = 4096;
const STATUS_BYTE_OFFSET: usize = 256;
const SCRATCH_PAGE: usize = 8192;

pub struct VirtioBlk {
    pub function: Function,
    common: *mut u8,
    notify: *mut u8,
    ring: *mut u8,
    pub sector_count: u64,
    pub read_only: bool,
    last_used: u16,
    available: u16,
}

unsafe fn read16(base: *mut u8, offset: usize) -> u16 {
    unsafe { read_volatile(base.add(offset).cast::<u16>()) }
}

unsafe fn write16(base: *mut u8, offset: usize, value: u16) {
    unsafe { write_volatile(base.add(offset).cast::<u16>(), value) }
}

unsafe fn write32(base: *mut u8, offset: usize, value: u32) {
    unsafe { write_volatile(base.add(offset).cast::<u32>(), value) }
}

/// The queue address registers are 64 bits wide but the device decodes them as two 32-bit
/// halves, low first.
unsafe fn write64(base: *mut u8, offset: usize, value: u64) {
    unsafe {
        write32(base, offset, value as u32);
        write32(base, offset + 4, (value >> 32) as u32);
    }
}

/// The virtio block devices on the bus, initialised and ready.
pub fn probe_all() -> Vec<Box<VirtioBlk>> {
    pci::scan()
        .into_iter()
        .filter(|function| {
            function.vendor == VENDOR_VIRTIO && function.device_id == DEVICE_ID_BLOCK
        })
        .filter_map(VirtioBlk::new)
        .map(Box::new)
        .collect()
}

impl VirtioBlk {
    fn new(function: Function) -> Option<Self> {
        let (device, number) = (function.device, function.function);
        let command = pci::read32(device, number, pci::REGISTER_COMMAND);
        pci::write32(
            device,
            number,
            pci::REGISTER_COMMAND,
            command | pci::COMMAND_MEMORY_SPACE | pci::COMMAND_BUS_MASTER,
        );
        // The machine has no firmware to assign BARs, so the library preassigned BAR0 and every
        // region the capabilities describe lives in it.
        let bar0 = (pci::read32(device, number, pci::REGISTER_BAR0) & !0xf) as usize;
        if bar0 == 0 {
            return None;
        }
        let (mut common, mut notify, mut device_config, mut multiplier) =
            (0usize, 0usize, 0usize, 0u32);
        let mut capability = pci::read8(device, number, pci::REGISTER_CAPABILITIES) as usize;
        for _ in 0..16 {
            if capability == 0 {
                break;
            }
            let register = |offset: u8| pci::read32(device, number, capability as u8 + offset);
            if pci::read8(device, number, capability as u8) == CAP_ID_VENDOR {
                let region = bar0 + register(CAP_OFFSET) as usize;
                let kind = pci::read8(device, number, capability as u8 + CAP_TYPE);
                debug_assert_eq!(pci::read8(device, number, capability as u8 + CAP_BAR), 0);
                match kind {
                    CFG_COMMON => common = region,
                    CFG_NOTIFY => {
                        notify = region;
                        multiplier = register(CAP_NOTIFY_MULTIPLIER);
                    }
                    CFG_DEVICE => device_config = region,
                    _ => {}
                }
            }
            capability = pci::read8(device, number, capability as u8 + 1) as usize;
        }
        if common == 0 || notify == 0 || device_config == 0 {
            return None;
        }
        let (common, notify, device_config) = (
            common as *mut u8,
            notify as *mut u8,
            device_config as *mut u8,
        );
        let ring = memory::allocate_zeroed_pages(BOOT_SERVICES_DATA, RING_PAGES)?;
        unsafe {
            write_volatile(common.add(COMMON_DEVICE_STATUS), 0);
            write_volatile(
                common.add(COMMON_DEVICE_STATUS),
                STATUS_ACKNOWLEDGE | STATUS_DRIVER,
            );
            write32(common, COMMON_DEVICE_FEATURE_SELECT, 0);
            let low = read_volatile(common.add(COMMON_DEVICE_FEATURE).cast::<u32>());
            write32(common, COMMON_DEVICE_FEATURE_SELECT, 1);
            let high = read_volatile(common.add(COMMON_DEVICE_FEATURE).cast::<u32>());
            if high & FEATURES_WORD_VERSION_1 == 0 {
                return None;
            }
            write32(common, COMMON_DRIVER_FEATURE_SELECT, 0);
            write32(common, COMMON_DRIVER_FEATURE, 0);
            write32(common, COMMON_DRIVER_FEATURE_SELECT, 1);
            write32(common, COMMON_DRIVER_FEATURE, FEATURES_WORD_VERSION_1);
            write_volatile(
                common.add(COMMON_DEVICE_STATUS),
                STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK,
            );
            if read_volatile(common.add(COMMON_DEVICE_STATUS)) & STATUS_FEATURES_OK == 0 {
                return None;
            }
            write16(common, COMMON_QUEUE_SELECT, 0);
            if read16(common, COMMON_QUEUE_SIZE) < QUEUE_SIZE {
                return None;
            }
            write16(common, COMMON_QUEUE_SIZE, QUEUE_SIZE);
            let base = ring as u64;
            write64(common, COMMON_QUEUE_DESC, base);
            write64(common, COMMON_QUEUE_DRIVER, base + AVAIL_OFFSET as u64);
            write64(common, COMMON_QUEUE_DEVICE, base + USED_OFFSET as u64);
            let notify_offset = read16(common, COMMON_QUEUE_NOTIFY_OFFSET) as usize;
            write16(common, COMMON_QUEUE_ENABLE, 1);
            write_volatile(
                common.add(COMMON_DEVICE_STATUS),
                STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK,
            );
            let sector_count =
                read_volatile(device_config.add(DEVICE_CONFIG_CAPACITY).cast::<u64>());
            Some(Self {
                function,
                common,
                notify: notify.add(notify_offset * multiplier as usize),
                ring,
                sector_count,
                read_only: low & FEATURE_RO != 0,
                last_used: 0,
                available: 0,
            })
        }
    }

    /// Stops the device, so the OS finds it as it would after power-on.
    pub fn reset(&mut self) {
        unsafe { write_volatile(self.common.add(COMMON_DEVICE_STATUS), 0) }
    }

    /// One request: header, optional data buffer, status byte. `device_writes` says whether the
    /// device fills the buffer (a read) or only reads it (a write).
    fn submit(
        &mut self,
        kind: u32,
        sector: u64,
        data: *mut u8,
        length: usize,
        device_writes: bool,
    ) -> bool {
        unsafe {
            let scratch = self.ring.add(SCRATCH_PAGE + SCRATCH_OFFSET);
            scratch.cast::<u32>().write(kind);
            scratch.add(4).cast::<u32>().write(0);
            scratch.add(8).cast::<u64>().write(sector);
            let status = scratch.add(STATUS_BYTE_OFFSET);
            status.write(0xff);
            let descriptor = |index: usize, address: u64, length: u32, flags: u16, next: u16| {
                let at = self.ring.add(index * DESC_BYTES);
                at.cast::<u64>().write(address);
                at.add(8).cast::<u32>().write(length);
                at.add(12).cast::<u16>().write(flags);
                at.add(14).cast::<u16>().write(next);
            };
            let header_flags = DESC_F_NEXT;
            descriptor(
                0,
                scratch as u64,
                HEADER_BYTES as u32,
                header_flags,
                if length > 0 { 1 } else { 2 },
            );
            if length > 0 {
                let flags = DESC_F_NEXT | if device_writes { DESC_F_WRITE } else { 0 };
                descriptor(1, data as u64, length as u32, flags, 2);
            }
            descriptor(2, status as u64, 1, DESC_F_WRITE, 0);
            let avail = self.ring.add(AVAIL_OFFSET);
            let slot = (self.available % QUEUE_SIZE) as usize;
            avail.add(4 + 2 * slot).cast::<u16>().write(0);
            self.available = self.available.wrapping_add(1);
            write_volatile(avail.add(2).cast::<u16>(), self.available);
            write16(self.notify, 0, 0);
            let used = self.ring.add(USED_OFFSET);
            for _ in 0..POLL_LIMIT {
                if read_volatile(used.add(2).cast::<u16>()) != self.last_used {
                    self.last_used = self.last_used.wrapping_add(1);
                    return read_volatile(status) == STATUS_OK;
                }
                core::hint::spin_loop();
            }
            false
        }
    }

    pub fn read(&mut self, first_sector: u64, buffer: &mut [u8]) -> bool {
        if buffer.len() % SECTOR_BYTES != 0
            || first_sector + (buffer.len() / SECTOR_BYTES) as u64 > self.sector_count
        {
            return false;
        }
        let mut sector = first_sector;
        for chunk in buffer.chunks_mut(MAX_REQUEST_BYTES) {
            if !self.submit(REQUEST_IN, sector, chunk.as_mut_ptr(), chunk.len(), true) {
                return false;
            }
            sector += (chunk.len() / SECTOR_BYTES) as u64;
        }
        true
    }

    pub fn write(&mut self, first_sector: u64, buffer: &[u8]) -> bool {
        if self.read_only
            || buffer.len() % SECTOR_BYTES != 0
            || first_sector + (buffer.len() / SECTOR_BYTES) as u64 > self.sector_count
        {
            return false;
        }
        let mut sector = first_sector;
        for chunk in buffer.chunks(MAX_REQUEST_BYTES) {
            if !self.submit(
                REQUEST_OUT,
                sector,
                chunk.as_ptr().cast_mut(),
                chunk.len(),
                false,
            ) {
                return false;
            }
            sector += (chunk.len() / SECTOR_BYTES) as u64;
        }
        true
    }

    pub fn flush(&mut self) -> bool {
        self.submit(REQUEST_FLUSH, 0, core::ptr::null_mut(), 0, false)
    }
}
