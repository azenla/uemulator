//! PCI configuration access through the legacy 0xCF8/0xCFC ports, and enumeration of bus 0.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, Ordering};

use crate::cpu::{inl, outl};
use crate::handoff::Handoff;

static CONFIG_ADDRESS_PORT: AtomicU16 = AtomicU16::new(0xcf8);
const CONFIG_DATA_OFFSET: u16 = 4;
const ENABLE: u32 = 1 << 31;
const DEVICES_PER_BUS: u8 = 32;
const MISSING_VENDOR: u16 = 0xffff;

pub const REGISTER_VENDOR_DEVICE: u8 = 0x00;
pub const REGISTER_COMMAND: u8 = 0x04;
pub const REGISTER_BAR0: u8 = 0x10;
pub const REGISTER_CAPABILITIES: u8 = 0x34;
pub const COMMAND_MEMORY_SPACE: u32 = 1 << 1;
pub const COMMAND_BUS_MASTER: u32 = 1 << 2;

#[derive(Clone, Copy, Debug)]
pub struct Function {
    pub device: u8,
    pub function: u8,
    pub vendor: u16,
    pub device_id: u16,
}

pub fn init(handoff: &Handoff) {
    CONFIG_ADDRESS_PORT.store(handoff.pci_config_port, Ordering::Relaxed);
}

fn address(device: u8, function: u8, register: u8) -> u32 {
    ENABLE | (device as u32) << 11 | (function as u32) << 8 | (register & 0xfc) as u32
}

pub fn read32(device: u8, function: u8, register: u8) -> u32 {
    let port = CONFIG_ADDRESS_PORT.load(Ordering::Relaxed);
    outl(port, address(device, function, register));
    inl(port + CONFIG_DATA_OFFSET)
}

pub fn write32(device: u8, function: u8, register: u8, value: u32) {
    let port = CONFIG_ADDRESS_PORT.load(Ordering::Relaxed);
    outl(port, address(device, function, register));
    outl(port + CONFIG_DATA_OFFSET, value);
}

pub fn read8(device: u8, function: u8, register: u8) -> u8 {
    (read32(device, function, register) >> (8 * (register & 3))) as u8
}

/// The functions on bus 0. The machine has one bus, and its virtio devices are single-function.
pub fn scan() -> Vec<Function> {
    (0..DEVICES_PER_BUS)
        .filter_map(|device| {
            let id = read32(device, 0, REGISTER_VENDOR_DEVICE);
            let vendor = id as u16;
            (vendor != MISSING_VENDOR).then_some(Function {
                device,
                function: 0,
                vendor,
                device_id: (id >> 16) as u16,
            })
        })
        .collect()
}
