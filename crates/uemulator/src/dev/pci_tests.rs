use super::*;
use crate::console::ConsoleHandle;
use crate::dev::guestmem::test_mem;
use crate::dev::vblk::VirtioBlk;
use crate::dev::vcons::VirtioConsole;
use crate::dev::vnet::VirtioNet;
use crate::disk::Disk;
use crate::netdev::NetHandle;

const PCI_SLOTS: u32 = 32;
const PCI_FUNCTIONS: u32 = 8;
const SLOT_SHIFT: u32 = 11;
const FUNCTION_SHIFT: u32 = 8;
const CONSOLE_SLOT: u32 = 1;
const BLOCK_SLOT: u32 = 2;
const NET_SLOT: u32 = 3;
const NO_DEVICE_DWORD: u32 = u32::MAX;
const BAR_SIZING_VALUE: u32 = !(BAR_SIZE as u32 - 1);
/// Common config `num_queues`, a register whose value tells which device a BAR belongs to.
const COMMON_NUM_QUEUES_OFFSET: u64 = 0x12;
const COMMON_DEVICE_STATUS_OFFSET: u64 = 0x14;
const STATUS_DRIVER_OK: u32 = 4;
const CAPABILITY_ID_VENDOR: u8 = 9;
const HEADER_TYPE_OFFSET: usize = 0x0e;
const HEADER_TYPE_ENDPOINT: u32 = 0;

fn console_transport() -> Transport {
    Transport::new(Box::new(VirtioConsole::new(ConsoleHandle::default())))
}

fn block_transport() -> Transport {
    Transport::new(Box::new(VirtioBlk::new(Disk::memory(vec![0; 4096]))))
}

fn net_transport() -> Transport {
    Transport::new(Box::new(VirtioNet::new(
        [2, 0, 0, 0, 0, 1],
        NetHandle::default(),
    )))
}

fn host() -> PciHost {
    PciHost::new(vec![console_transport()])
}

/// Console in slot 1, block in slot 2, net in slot 3, the order `Devices::new` uses.
fn full_host() -> PciHost {
    PciHost::new(vec![
        console_transport(),
        block_transport(),
        net_transport(),
    ])
}

fn address(slot: u32, function: u32, reg: u32) -> u32 {
    CONFIG_ADDRESS_ENABLE | slot << SLOT_SHIFT | function << FUNCTION_SHIFT | reg
}

fn cfg_read(h: &mut PciHost, slot: u32, reg: u32, size: u8) -> u32 {
    h.io_out(CONFIG_ADDRESS_PORT, 4, address(slot, 0, reg & !3));
    h.io_in(CONFIG_DATA_PORT + (reg & 3) as u16, size)
}

fn cfg_read32(h: &mut PciHost, slot: u32, reg: u32) -> u32 {
    cfg_read(h, slot, reg, 4)
}

fn cfg_write(h: &mut PciHost, slot: u32, reg: u32, size: u8, value: u32) {
    h.io_out(CONFIG_ADDRESS_PORT, 4, address(slot, 0, reg & !3));
    h.io_out(CONFIG_DATA_PORT + (reg & 3) as u16, size, value);
}

fn enable_memory_space(h: &mut PciHost, slot: u32) {
    cfg_write(h, slot, CFG_COMMAND as u32, 2, COMMAND_MEMORY_SPACE as u32);
}

fn bar_base(h: &mut PciHost, slot: u32) -> u64 {
    cfg_read32(h, slot, CFG_BAR0 as u32) as u64
}

#[test]
fn every_slot_and_function_is_enumerated_and_only_the_present_ones_answer() {
    let mut h = full_host();
    let mut present = Vec::new();
    for slot in 0..PCI_SLOTS {
        for function in 0..PCI_FUNCTIONS {
            h.io_out(CONFIG_ADDRESS_PORT, 4, address(slot, function, 0));
            let id = h.io_in(CONFIG_DATA_PORT, 4);
            if id != NO_DEVICE_DWORD {
                present.push((slot, function, id));
            }
        }
    }
    assert_eq!(
        present,
        vec![
            (
                0,
                0,
                VENDOR_INTEL as u32 | (DEVICE_I440FX_HOST as u32) << 16
            ),
            (1, 0, VENDOR_VIRTIO as u32 | 0x1043 << 16),
            (2, 0, VENDOR_VIRTIO as u32 | 0x1042 << 16),
            (3, 0, VENDOR_VIRTIO as u32 | 0x1041 << 16),
        ]
    );
}

#[test]
fn absent_function_reads_all_ones_at_every_access_size() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, address(CONSOLE_SLOT, 1, 0));
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), u32::MAX);
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), 0xffff);
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 1), 0xff);
    assert_eq!(h.io_in(CONFIG_DATA_PORT + 3, 1), 0xff);
}

#[test]
fn nonzero_bus_reads_all_ones_even_for_a_present_devfn() {
    let mut h = host();
    h.io_out(
        CONFIG_ADDRESS_PORT,
        4,
        address(CONSOLE_SLOT, 0, 0) | 1 << CONFIG_ADDRESS_BUS_SHIFT,
    );
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), u32::MAX);
}

#[test]
fn config_data_reads_all_ones_while_enable_bit_is_clear() {
    let mut h = host();
    h.io_out(
        CONFIG_ADDRESS_PORT,
        4,
        address(CONSOLE_SLOT, 0, 0) & !CONFIG_ADDRESS_ENABLE,
    );
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), u32::MAX);
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), 0xffff);
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 1), 0xff);
}

#[test]
fn config_data_write_is_ignored_while_enable_bit_is_clear() {
    let mut h = host();
    let disabled = address(CONSOLE_SLOT, 0, CFG_COMMAND as u32) & !CONFIG_ADDRESS_ENABLE;
    h.io_out(CONFIG_ADDRESS_PORT, 4, disabled);
    h.io_out(CONFIG_DATA_PORT, 2, COMMAND_MEMORY_SPACE as u32);
    assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
}

#[test]
fn config_write_to_nonzero_bus_is_ignored() {
    let mut h = host();
    h.io_out(
        CONFIG_ADDRESS_PORT,
        4,
        address(CONSOLE_SLOT, 0, CFG_COMMAND as u32) | 1 << CONFIG_ADDRESS_BUS_SHIFT,
    );
    h.io_out(CONFIG_DATA_PORT, 2, COMMAND_MEMORY_SPACE as u32);
    assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
}

#[test]
fn config_write_to_absent_slot_is_ignored() {
    let mut h = host();
    cfg_write(&mut h, NET_SLOT, CFG_COMMAND as u32, 2, 0xffff);
    assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
}

#[test]
fn host_bridge_identifies_as_intel_440fx_host_bridge() {
    let mut h = host();
    assert_eq!(
        cfg_read(&mut h, 0, CFG_VENDOR_ID as u32, 2),
        VENDOR_INTEL as u32
    );
    assert_eq!(
        cfg_read(&mut h, 0, CFG_DEVICE_ID as u32, 2),
        DEVICE_I440FX_HOST as u32
    );
    assert_eq!(
        cfg_read(&mut h, 0, CFG_CLASS as u32, 1),
        CLASS_BRIDGE as u32
    );
    assert_eq!(
        cfg_read(&mut h, 0, CFG_SUBCLASS as u32, 1),
        SUBCLASS_HOST_BRIDGE as u32
    );
    assert_eq!(cfg_read32(&mut h, 0, CFG_REVISION as u32) >> 16, 0x0600);
}

#[test]
fn host_bridge_has_no_capabilities_bars_or_interrupt_pin() {
    let mut h = host();
    assert_eq!(cfg_read32(&mut h, 0, CFG_CAPABILITIES_POINTER as u32), 0);
    assert_eq!(cfg_read32(&mut h, 0, CFG_BAR0 as u32), 0);
    assert_eq!(cfg_read(&mut h, 0, CFG_INTERRUPT_PIN as u32, 1), 0);
}

#[test]
fn host_bridge_config_is_read_only() {
    let mut h = host();
    cfg_write(&mut h, 0, CFG_COMMAND as u32, 2, 0xffff);
    cfg_write(&mut h, 0, CFG_VENDOR_ID as u32, 4, 0);
    assert_eq!(cfg_read(&mut h, 0, CFG_COMMAND as u32, 2), 0);
    assert_eq!(
        cfg_read(&mut h, 0, CFG_VENDOR_ID as u32, 2),
        VENDOR_INTEL as u32
    );
}

#[test]
fn host_bridge_is_present_even_with_no_virtio_devices() {
    let mut h = PciHost::new(Vec::new());
    assert!(h.is_empty());
    assert_eq!(
        cfg_read(&mut h, 0, CFG_VENDOR_ID as u32, 2),
        VENDOR_INTEL as u32
    );
    assert_eq!(cfg_read32(&mut h, 1, CFG_VENDOR_ID as u32), u32::MAX);
}

#[test]
fn host_is_not_empty_with_a_device() {
    assert!(!host().is_empty());
}

#[test]
fn virtio_functions_report_vendor_and_modern_device_ids() {
    let mut h = full_host();
    for (slot, device_type) in [
        (CONSOLE_SLOT, DEVICE_TYPE_CONSOLE),
        (BLOCK_SLOT, DEVICE_TYPE_BLOCK),
        (NET_SLOT, DEVICE_TYPE_NET),
    ] {
        assert_eq!(cfg_read(&mut h, slot, CFG_VENDOR_ID as u32, 2), 0x1af4);
        assert_eq!(
            cfg_read(&mut h, slot, CFG_DEVICE_ID as u32, 2),
            0x1040 + device_type as u32
        );
        assert_eq!(
            cfg_read(&mut h, slot, CFG_SUBSYSTEM_VENDOR_ID as u32, 2),
            0x1af4
        );
        assert_eq!(
            cfg_read(&mut h, slot, CFG_SUBSYSTEM_ID as u32, 2),
            device_type as u32
        );
    }
}

#[test]
fn virtio_functions_are_revision_one_endpoints() {
    let mut h = full_host();
    for slot in CONSOLE_SLOT..=NET_SLOT {
        assert_eq!(cfg_read(&mut h, slot, CFG_REVISION as u32, 1), 1);
        assert_eq!(
            cfg_read(&mut h, slot, HEADER_TYPE_OFFSET as u32, 1),
            HEADER_TYPE_ENDPOINT
        );
        assert_eq!(cfg_read(&mut h, slot, 0x09, 1), 0, "programming interface");
    }
}

#[test]
fn class_codes_follow_qemu() {
    let mut h = full_host();
    let class_of = |h: &mut PciHost, slot| cfg_read32(h, slot, CFG_REVISION as u32) >> 16;
    assert_eq!(class_of(&mut h, CONSOLE_SLOT), 0x0780);
    assert_eq!(class_of(&mut h, BLOCK_SLOT), 0x0100);
    assert_eq!(class_of(&mut h, NET_SLOT), 0x0200);
}

#[test]
fn bar0_starts_at_the_mmio_window_one_bar_apart_and_is_memory_type() {
    let mut h = full_host();
    for (index, slot) in [CONSOLE_SLOT, BLOCK_SLOT, NET_SLOT].into_iter().enumerate() {
        let bar = bar_base(&mut h, slot);
        assert_eq!(bar, MMIO_BASE + index as u64 * BAR_SIZE);
        assert_eq!(bar & 0xf, 0, "32-bit non-prefetchable memory BAR");
    }
}

#[test]
fn other_bars_read_zero() {
    let mut h = host();
    for reg in (CFG_BAR0 as u32 + 4..0x28).step_by(4) {
        assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, reg), 0, "BAR at {reg:#x}");
    }
}

#[test]
fn bar_sizing_with_dword_all_ones_reports_the_size_mask() {
    let mut h = host();
    let original = bar_base(&mut h, CONSOLE_SLOT) as u32;
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, u32::MAX);
    let sized = cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32);
    assert_eq!(sized, BAR_SIZING_VALUE);
    assert_eq!(!sized + 1, BAR_SIZE as u32, "size is the lowest set bit");
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, original);
    assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32), original);
}

#[test]
fn bar_sizing_by_bytes_also_works() {
    let mut h = host();
    for lane in 0..4 {
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + lane, 1, 0xff);
    }
    assert_eq!(
        cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
        BAR_SIZING_VALUE
    );
}

#[test]
fn bar_sizing_by_words_also_works() {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 2, 0xffff);
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + 2, 2, 0xffff);
    assert_eq!(
        cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
        BAR_SIZING_VALUE
    );
}

#[test]
fn bar_low_bits_below_the_size_are_hardwired_to_zero() {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, 0xd000_3fff);
    assert_eq!(
        cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
        0xd000_0000
    );
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, 0xd000_4001);
    assert_eq!(
        cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
        0xd000_4000
    );
}

#[test]
fn bar_sizing_does_not_change_other_devices_bars() {
    let mut h = full_host();
    let before = bar_base(&mut h, BLOCK_SLOT);
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, u32::MAX);
    assert_eq!(bar_base(&mut h, BLOCK_SLOT), before);
}

#[test]
fn memory_space_disabled_means_bar_is_not_decoded() {
    let mut h = host();
    assert!(
        h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
            .is_none()
    );
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    assert_eq!(h.mmio_write(MMIO_BASE, 4, 0, &mut mem), None);
}

#[test]
fn enabling_memory_space_decodes_bar0() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    assert_eq!(
        h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2),
        Some(2)
    );
}

#[test]
fn disabling_memory_space_again_stops_decoding() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, 0);
    assert!(
        h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
            .is_none()
    );
}

#[test]
fn io_space_bit_alone_does_not_enable_mmio_decoding() {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, 1);
    assert!(
        h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
            .is_none()
    );
}

#[test]
fn bar_decode_covers_exactly_bar_size_bytes() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    assert!(h.mmio_read(MMIO_BASE + BAR_SIZE - 1, 1).is_some());
    assert!(h.mmio_read(MMIO_BASE + BAR_SIZE, 1).is_none());
    assert!(h.mmio_read(MMIO_BASE - 1, 1).is_none());
}

#[test]
fn each_device_decodes_only_its_own_bar() {
    let mut h = full_host();
    for slot in [CONSOLE_SLOT, BLOCK_SLOT, NET_SLOT] {
        enable_memory_space(&mut h, slot);
    }
    // num_queues: console 2, block 1, net 2.
    assert_eq!(
        h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2),
        Some(2)
    );
    assert_eq!(
        h.mmio_read(MMIO_BASE + BAR_SIZE + COMMON_NUM_QUEUES_OFFSET, 2),
        Some(1)
    );
    assert_eq!(
        h.mmio_read(MMIO_BASE + 2 * BAR_SIZE + COMMON_NUM_QUEUES_OFFSET, 2),
        Some(2)
    );
    assert!(h.mmio_read(MMIO_BASE + 3 * BAR_SIZE, 1).is_none());
}

#[test]
fn relocating_bar0_moves_the_decode_window() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    let new_base = 0xd000_0000u32;
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, new_base);
    assert!(
        h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
            .is_none()
    );
    assert_eq!(
        h.mmio_read(new_base as u64 + COMMON_NUM_QUEUES_OFFSET, 2),
        Some(2)
    );
}

#[test]
fn bar_programmed_to_zero_is_not_decoded_even_with_memory_space_on() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, 0);
    assert!(h.mmio_read(COMMON_NUM_QUEUES_OFFSET, 2).is_none());
}

#[test]
fn sizing_a_bar_with_decode_disabled_leaves_the_decode_off() {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, u32::MAX);
    assert!(
        h.mmio_read(BAR_SIZING_VALUE as u64 + COMMON_NUM_QUEUES_OFFSET, 2)
            .is_none()
    );
}

#[test]
fn command_register_stores_written_bits_and_status_is_constant() {
    let mut h = host();
    assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
    cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, 0x0406);
    assert_eq!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2),
        0x0406
    );
    cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 1, 0x02);
    assert_eq!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2),
        0x0402
    );
    cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32 + 1, 1, 0x00);
    assert_eq!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2),
        0x0002
    );
    assert_eq!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2),
        STATUS_CAPABILITIES_LIST as u32
    );
    cfg_write(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2, 0xffff);
    assert_eq!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2),
        STATUS_CAPABILITIES_LIST as u32
    );
}

#[test]
fn dword_write_at_command_register_updates_command_and_ignores_status() {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 4, 0xffff_0002);
    assert_eq!(
        cfg_read32(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32),
        0x0010_0002
    );
}

#[test]
fn id_class_and_subsystem_registers_are_read_only() {
    let mut h = host();
    let before: Vec<u32> = (0..0x40)
        .step_by(4)
        .map(|reg| cfg_read32(&mut h, CONSOLE_SLOT, reg))
        .collect();
    for reg in [0x00, 0x08, 0x0c, 0x2c, 0x34] {
        cfg_write(&mut h, CONSOLE_SLOT, reg, 4, 0xa5a5_a5a5);
    }
    let after: Vec<u32> = (0..0x40)
        .step_by(4)
        .map(|reg| cfg_read32(&mut h, CONSOLE_SLOT, reg))
        .collect();
    assert_eq!(before, after);
}

#[test]
fn capability_list_is_flagged_in_status_and_pointer_is_dword_aligned() {
    let mut h = host();
    assert_ne!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2) & STATUS_CAPABILITIES_LIST as u32,
        0
    );
    let pointer = cfg_read(&mut h, CONSOLE_SLOT, CFG_CAPABILITIES_POINTER as u32, 1);
    assert_eq!(pointer, CAP_COMMON_AT as u32);
    assert_eq!(pointer % 4, 0);
}

#[derive(Debug, PartialEq)]
struct Capability {
    at: u32,
    id: u8,
    next: u8,
    len: u8,
    cfg_type: u8,
    bar: u8,
    region_offset: u32,
    region_length: u32,
}

fn walk_capabilities(h: &mut PciHost, slot: u32) -> Vec<Capability> {
    let mut caps = Vec::new();
    let mut at = cfg_read(h, slot, CFG_CAPABILITIES_POINTER as u32, 1);
    // A well-formed list in 256 bytes of config space cannot have more than 48 entries.
    for _ in 0..48 {
        if at == 0 {
            return caps;
        }
        let header = cfg_read32(h, slot, at);
        caps.push(Capability {
            at,
            id: header as u8,
            next: (header >> 8) as u8,
            len: (header >> 16) as u8,
            cfg_type: (header >> 24) as u8,
            bar: cfg_read(h, slot, at + 4, 1) as u8,
            region_offset: cfg_read32(h, slot, at + VIRTIO_CAP_OFFSET_FIELD as u32),
            region_length: cfg_read32(h, slot, at + VIRTIO_CAP_LENGTH_FIELD as u32),
        });
        at = (header >> 8) & 0xff;
    }
    panic!("capability list does not terminate");
}

#[test]
fn capability_chain_has_four_vendor_capabilities_and_terminates() {
    let mut h = host();
    let caps = walk_capabilities(&mut h, CONSOLE_SLOT);
    assert_eq!(caps.len(), 4);
    assert_eq!(
        caps.last().unwrap().next,
        0,
        "last capability has next == 0"
    );
    assert!(caps.iter().all(|cap| cap.id == CAPABILITY_ID_VENDOR));
    assert!(caps.iter().all(|cap| cap.at % 4 == 0 && cap.at >= 0x40));
}

#[test]
fn capability_types_regions_and_lengths_match_the_bar_layout() {
    let mut h = host();
    let caps = walk_capabilities(&mut h, CONSOLE_SLOT);
    let summary: Vec<(u8, u8, u32, u32)> = caps
        .iter()
        .map(|c| (c.cfg_type, c.bar, c.region_offset, c.region_length))
        .collect();
    assert_eq!(
        summary,
        vec![
            (1, 0, 0x0000, 0x38),
            (2, 0, 0x3000, 0x1000),
            (3, 0, 0x1000, 4),
            (4, 0, 0x2000, 0x100),
        ]
    );
}

#[test]
fn capability_lengths_are_sixteen_bytes_and_twenty_for_notify() {
    let mut h = host();
    let lengths: Vec<(u8, u8)> = walk_capabilities(&mut h, CONSOLE_SLOT)
        .iter()
        .map(|c| (c.cfg_type, c.len))
        .collect();
    assert_eq!(lengths, vec![(1, 16), (2, 20), (3, 16), (4, 16)]);
}

#[test]
fn capabilities_do_not_overlap_each_other() {
    let mut h = host();
    let caps = walk_capabilities(&mut h, CONSOLE_SLOT);
    for pair in caps.windows(2) {
        assert!(pair[0].at + pair[0].len as u32 <= pair[1].at, "{pair:?}");
    }
    let last = caps.last().unwrap();
    assert!(last.at + last.len as u32 <= 256);
}

#[test]
fn notify_capability_carries_the_queue_notify_multiplier() {
    let mut h = host();
    let notify = walk_capabilities(&mut h, CONSOLE_SLOT)
        .into_iter()
        .find(|cap| cap.cfg_type == 2)
        .unwrap();
    let multiplier = cfg_read32(
        &mut h,
        CONSOLE_SLOT,
        notify.at + VIRTIO_CAP_NOTIFY_MULTIPLIER_FIELD as u32,
    );
    assert_eq!(multiplier, NOTIFY_OFF_MULTIPLIER);
    // Every queue's notify address must stay inside the advertised notify region.
    let queue_count = 2;
    assert!(
        notify.region_offset + (queue_count - 1) * multiplier + 2
            <= notify.region_offset + notify.region_length
    );
}

#[test]
fn capability_regions_fit_in_the_bar_and_do_not_overlap() {
    let mut h = host();
    let mut regions: Vec<(u32, u32)> = walk_capabilities(&mut h, CONSOLE_SLOT)
        .iter()
        .map(|cap| (cap.region_offset, cap.region_length))
        .collect();
    regions.sort();
    for pair in regions.windows(2) {
        assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{pair:?}");
    }
    assert!(
        regions
            .iter()
            .all(|&(start, len)| (start + len) as u64 <= BAR_SIZE)
    );
}

#[test]
fn capability_chain_is_identical_for_every_device_type() {
    let mut h = full_host();
    let console = walk_capabilities(&mut h, CONSOLE_SLOT);
    assert_eq!(walk_capabilities(&mut h, BLOCK_SLOT), console);
    assert_eq!(walk_capabilities(&mut h, NET_SLOT), console);
}

#[test]
fn bytes_past_the_last_capability_read_zero() {
    let mut h = host();
    for reg in (CAP_DEVICE_AT as u32 + VIRTIO_CAP_LEN as u32..CONFIG_SPACE_SIZE as u32).step_by(4) {
        assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, reg), 0, "offset {reg:#x}");
    }
}

#[test]
fn unimplemented_registers_of_a_present_device_read_zero_not_all_ones() {
    let mut h = host();
    for (reg, what) in [
        (0x28, "cardbus cis pointer"),
        (0x30, "expansion rom base"),
        (0x38, "reserved"),
    ] {
        assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, reg), 0, "{what}");
    }
}

#[test]
fn interrupt_line_and_pin_follow_the_legacy_irq_assignment() {
    let mut h = full_host();
    for (slot, irq) in [(CONSOLE_SLOT, 9), (BLOCK_SLOT, 10), (NET_SLOT, 11)] {
        assert_eq!(cfg_read(&mut h, slot, CFG_INTERRUPT_LINE as u32, 1), irq);
        assert_eq!(
            cfg_read(&mut h, slot, CFG_INTERRUPT_PIN as u32, 1),
            INTERRUPT_PIN_INTA as u32
        );
    }
    let dword = cfg_read32(&mut h, NET_SLOT, CFG_INTERRUPT_LINE as u32);
    assert_eq!(
        dword & 0xffff,
        0x0100 | 11,
        "min_gnt/max_lat bytes above are zero"
    );
}

#[test]
fn interrupt_line_is_writable_and_pin_is_read_only() {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_LINE as u32, 1, 5);
    assert_eq!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_LINE as u32, 1),
        5
    );
    cfg_write(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_PIN as u32, 1, 3);
    assert_eq!(
        cfg_read(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_PIN as u32, 1),
        1
    );
}

#[test]
fn irq_assignment_wraps_after_three_devices_and_never_uses_zero() {
    let h = PciHost::new(vec![
        console_transport(),
        console_transport(),
        console_transport(),
        console_transport(),
    ]);
    let lines: Vec<u8> = h.funcs.iter().map(|f| f.int_line).collect();
    assert_eq!(lines, vec![9, 10, 11, 9]);
    assert!(LEGACY_IRQS.iter().all(|&irq| irq != 0));
}

#[test]
fn devfn_is_slot_times_eight_starting_at_slot_one() {
    let h = full_host();
    let devfns: Vec<u8> = h.funcs.iter().map(|f| f.devfn).collect();
    assert_eq!(devfns, vec![8, 16, 24]);
}

#[test]
fn address_register_reads_back_what_was_written() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, 0x8000_1234);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8000_1234);
}

#[test]
fn address_register_byte_and_word_reads_use_the_lane_of_the_port() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, 0x8112_3456);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 1), 0x56);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 1, 1), 0x34);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 2, 1), 0x12);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 3, 1), 0x81);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 2), 0x3456);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 2, 2), 0x8112);
}

#[test]
fn address_register_byte_writes_merge_into_their_lane() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, 0x1111_1111);
    h.io_out(CONFIG_ADDRESS_PORT + 3, 1, 0x80);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8011_1111);
    h.io_out(CONFIG_ADDRESS_PORT + 1, 1, 0x22);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8011_2211);
    h.io_out(CONFIG_ADDRESS_PORT + 2, 1, 0x33);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8033_2211);
    h.io_out(CONFIG_ADDRESS_PORT, 1, 0x44);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8033_2244);
}

#[test]
fn address_register_word_writes_merge_into_their_half() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, 0xaaaa_bbbb);
    h.io_out(CONFIG_ADDRESS_PORT, 2, 0x1234);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0xaaaa_1234);
    h.io_out(CONFIG_ADDRESS_PORT + 2, 2, 0x8000);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8000_1234);
}

#[test]
fn word_write_ignores_the_value_bits_above_the_access_size() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, 0);
    h.io_out(CONFIG_ADDRESS_PORT, 2, 0xdead_beef);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0xbeef);
    h.io_out(CONFIG_ADDRESS_PORT, 1, 0xffff_ff01);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0xbe01);
}

#[test]
fn byte_write_to_address_register_keeps_the_other_bytes() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, 0x8000_0810);
    h.io_out(CONFIG_ADDRESS_PORT, 1, 0x14);
    assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8000_0814);
}

#[test]
fn config_register_number_ignores_the_low_address_bits() {
    let mut h = host();
    h.io_out(CONFIG_ADDRESS_PORT, 4, address(CONSOLE_SLOT, 0, 0x03));
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), VENDOR_VIRTIO as u32);
}

#[test]
fn data_window_byte_lanes_select_bytes_of_the_dword() {
    let mut h = host();
    let dword = cfg_read32(&mut h, CONSOLE_SLOT, CFG_VENDOR_ID as u32);
    for lane in 0..4u16 {
        assert_eq!(
            h.io_in(CONFIG_DATA_PORT + lane, 1),
            (dword >> (8 * lane)) & 0xff,
            "lane {lane}"
        );
    }
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), dword & 0xffff);
    assert_eq!(h.io_in(CONFIG_DATA_PORT + 2, 2), dword >> 16);
    assert_eq!(h.io_in(CONFIG_DATA_PORT + 1, 2), (dword >> 8) & 0xffff);
}

#[test]
fn data_window_dword_read_at_lane_one_is_zero_extended_not_wrapped() {
    let mut h = host();
    let dword = cfg_read32(&mut h, CONSOLE_SLOT, CFG_VENDOR_ID as u32);
    assert_eq!(h.io_in(CONFIG_DATA_PORT + 1, 4), dword >> 8);
}

#[test]
fn data_window_byte_write_changes_only_that_byte() {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + 3, 1, 0xd0);
    assert_eq!(
        cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
        0xd000_0000
    );
    cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + 2, 1, 0x40);
    assert_eq!(
        cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
        0xd040_0000
    );
}

#[test]
fn data_window_access_reaching_past_the_dword_does_not_spill_into_the_next_register() {
    let mut h = host();
    let before = cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32);
    h.io_out(CONFIG_ADDRESS_PORT, 4, address(CONSOLE_SLOT, 0, 0x0c));
    // Lane 3 of register 0x0c is byte 0x0f; bytes 1..3 of the value would land on BAR0.
    h.io_out(CONFIG_DATA_PORT + 3, 4, 0x12c0_0000);
    assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32), before);
}

#[test]
fn memory_and_config_accesses_do_not_interfere_with_each_other() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    h.io_out(
        CONFIG_ADDRESS_PORT,
        4,
        address(CONSOLE_SLOT, 0, CFG_COMMAND as u32),
    );
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    h.mmio_write(
        MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
        1,
        STATUS_DRIVER_OK,
        &mut mem,
    );
    assert_eq!(
        h.io_in(CONFIG_ADDRESS_PORT, 4),
        address(CONSOLE_SLOT, 0, CFG_COMMAND as u32)
    );
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), COMMAND_MEMORY_SPACE as u32);
}

#[test]
fn console_is_active_only_after_the_driver_sets_driver_ok() {
    let mut h = host();
    assert!(!h.console_active());
    enable_memory_space(&mut h, CONSOLE_SLOT);
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    h.mmio_write(
        MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
        1,
        STATUS_DRIVER_OK,
        &mut mem,
    );
    assert!(h.console_active());
    h.mmio_write(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1, 0, &mut mem);
    assert!(!h.console_active(), "a reset deactivates it");
}

#[test]
fn only_a_console_function_counts_for_console_active() {
    let mut h = PciHost::new(vec![block_transport()]);
    enable_memory_space(&mut h, CONSOLE_SLOT);
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    h.mmio_write(
        MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
        1,
        STATUS_DRIVER_OK,
        &mut mem,
    );
    assert!(!h.console_active());
}

fn host_with_pending_config_event() -> (PciHost, ConsoleHandle) {
    let console = ConsoleHandle::default();
    let mut h = PciHost::new(vec![Transport::new(Box::new(VirtioConsole::new(
        console.clone(),
    )))]);
    enable_memory_space(&mut h, CONSOLE_SLOT);
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    h.mmio_write(
        MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
        1,
        STATUS_DRIVER_OK,
        &mut mem,
    );
    console.resize(100, 40);
    (h, console)
}

#[test]
fn poll_reports_the_function_interrupt_line() {
    let (mut h, _console) = host_with_pending_config_event();
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    let mut raise = Vec::new();
    h.poll(&mut mem, &mut raise);
    assert_eq!(raise, vec![9]);
}

#[test]
fn poll_reports_nothing_when_intx_is_disabled() {
    let (mut h, _console) = host_with_pending_config_event();
    cfg_write(
        &mut h,
        CONSOLE_SLOT,
        CFG_COMMAND as u32,
        2,
        (COMMAND_MEMORY_SPACE | COMMAND_INTX_DISABLE) as u32,
    );
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    let mut raise = Vec::new();
    h.poll(&mut mem, &mut raise);
    assert!(raise.is_empty());
}

#[test]
fn poll_with_nothing_to_report_adds_no_irqs() {
    let mut h = full_host();
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    let mut raise = Vec::new();
    h.poll(&mut mem, &mut raise);
    assert!(raise.is_empty());
}

#[test]
fn mmio_write_that_raises_nothing_returns_zero_not_none() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    let result = h.mmio_write(
        MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
        1,
        STATUS_DRIVER_OK,
        &mut mem,
    );
    assert_eq!(result, Some(0));
}

const COMMON_QUEUE_SELECT_OFFSET: u64 = 0x16;
const COMMON_QUEUE_SIZE_OFFSET: u64 = 0x18;
const COMMON_QUEUE_ENABLE_OFFSET: u64 = 0x1c;
const COMMON_QUEUE_DESC_OFFSET: u64 = 0x20;
const COMMON_QUEUE_DRIVER_OFFSET: u64 = 0x28;
const COMMON_QUEUE_DEVICE_OFFSET: u64 = 0x30;
const NOTIFY_REGION_OFFSET: u64 = 0x3000;
const TRANSMIT_QUEUE: u64 = 1;
const TX_DESC: u64 = 0x100;
const TX_AVAIL: u64 = 0x200;
const TX_USED: u64 = 0x300;
const TX_BUFFER: u64 = 0x400;

/// Console with its transmit queue programmed through BAR0 and one buffer ready; the next
/// kick completes it.
fn console_with_transmit_ready(command: u16) -> (PciHost, Vec<u8>) {
    let mut h = host();
    cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, command as u32);
    let mut ram = vec![0u8; 0x1000];
    ram[TX_BUFFER as usize] = b'x';
    ram[TX_DESC as usize..TX_DESC as usize + 8].copy_from_slice(&TX_BUFFER.to_le_bytes());
    ram[TX_DESC as usize + 8..TX_DESC as usize + 12].copy_from_slice(&1u32.to_le_bytes());
    ram[TX_AVAIL as usize + 2] = 1;
    let mut mem = test_mem(&mut ram);
    let mut write = |offset: u64, size: u8, value: u32| {
        h.mmio_write(MMIO_BASE + offset, size, value, &mut mem);
    };
    write(COMMON_QUEUE_SELECT_OFFSET, 2, TRANSMIT_QUEUE as u32);
    write(COMMON_QUEUE_SIZE_OFFSET, 2, 8);
    write(COMMON_QUEUE_DESC_OFFSET, 4, TX_DESC as u32);
    write(COMMON_QUEUE_DRIVER_OFFSET, 4, TX_AVAIL as u32);
    write(COMMON_QUEUE_DEVICE_OFFSET, 4, TX_USED as u32);
    write(COMMON_QUEUE_ENABLE_OFFSET, 2, 1);
    write(COMMON_DEVICE_STATUS_OFFSET, 1, STATUS_DRIVER_OK);
    (h, ram)
}

fn kick_transmit(h: &mut PciHost, ram: &mut [u8]) -> Option<u8> {
    let mut mem = test_mem(ram);
    h.mmio_write(
        MMIO_BASE + NOTIFY_REGION_OFFSET + TRANSMIT_QUEUE * NOTIFY_OFF_MULTIPLIER as u64,
        2,
        TRANSMIT_QUEUE as u32,
        &mut mem,
    )
}

#[test]
fn mmio_write_that_completes_a_buffer_returns_the_function_irq() {
    let (mut h, mut ram) = console_with_transmit_ready(COMMAND_MEMORY_SPACE);
    assert_eq!(kick_transmit(&mut h, &mut ram), Some(9));
}

#[test]
fn mmio_write_that_completes_a_buffer_returns_zero_when_intx_is_disabled() {
    let (mut h, mut ram) = console_with_transmit_ready(COMMAND_MEMORY_SPACE | COMMAND_INTX_DISABLE);
    assert_eq!(kick_transmit(&mut h, &mut ram), Some(0));
}

#[test]
fn mmio_write_to_unmapped_address_returns_none() {
    let mut h = host();
    enable_memory_space(&mut h, CONSOLE_SLOT);
    let mut ram = vec![0u8; 0x1000];
    let mut mem = test_mem(&mut ram);
    assert_eq!(h.mmio_write(MMIO_BASE + BAR_SIZE, 4, 0, &mut mem), None);
}

fn config_address(slot: u32, reg: u32) -> u32 {
    CONFIG_ADDRESS_ENABLE | slot << (CONFIG_ADDRESS_DEVFN_SHIFT + DEVFN_SLOT_SHIFT) | reg
}

fn rd(h: &mut PciHost, slot: u32, reg: u32) -> u32 {
    h.io_out(CONFIG_ADDRESS_PORT, 4, config_address(slot, reg));
    h.io_in(CONFIG_DATA_PORT, 4)
}

#[test]
fn enumerate_and_size_bar() {
    let mut h = host();
    assert_eq!(
        rd(&mut h, 0, CFG_VENDOR_ID as u32) & 0xffff,
        VENDOR_INTEL as u32
    );
    // Dword at 8 is revision, prog_if, subclass, class: the upper half is the class code.
    assert_eq!(rd(&mut h, 0, CFG_REVISION as u32) >> 16, 0x0600);
    assert_eq!(
        rd(&mut h, 1, CFG_VENDOR_ID as u32),
        VENDOR_VIRTIO as u32 | ((VIRTIO_PCI_DEVICE_ID_BASE + DEVICE_TYPE_CONSOLE) as u32) << 16
    );
    assert_eq!(rd(&mut h, 2, CFG_VENDOR_ID as u32), u32::MAX);
    let bar = rd(&mut h, 1, CFG_BAR0 as u32);
    h.io_out(CONFIG_ADDRESS_PORT, 4, config_address(1, CFG_BAR0 as u32));
    h.io_out(CONFIG_DATA_PORT, 4, u32::MAX);
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), !(BAR_SIZE as u32 - 1));
    h.io_out(CONFIG_DATA_PORT, 4, bar);
    assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), bar);
    let mut caps = Vec::new();
    let mut p = rd(&mut h, 1, CFG_CAPABILITIES_POINTER as u32) & 0xff;
    while p != 0 {
        let w = rd(&mut h, 1, p);
        caps.push((w >> 24) as u8);
        p = (w >> 8) & 0xff;
    }
    assert_eq!(
        caps,
        vec![
            VIRTIO_PCI_CAP_COMMON_CFG,
            VIRTIO_PCI_CAP_NOTIFY_CFG,
            VIRTIO_PCI_CAP_ISR_CFG,
            VIRTIO_PCI_CAP_DEVICE_CFG
        ]
    );
}

#[test]
fn bar_decode_follows_command() {
    let mut h = host();
    assert!(h.mmio_read(MMIO_BASE + 0x14, 1).is_none());
    h.io_out(
        CONFIG_ADDRESS_PORT,
        4,
        config_address(1, CFG_COMMAND as u32),
    );
    h.io_out(CONFIG_DATA_PORT, 2, COMMAND_MEMORY_SPACE as u32);
    // BAR0 offset 0x12 is the common config num_queues field; the console has 2 queues.
    assert_eq!(h.mmio_read(MMIO_BASE + 0x12, 2), Some(2));
}

#[test]
fn a_line_is_asserted_only_while_a_function_wired_to_it_has_isr_bits_set() {
    let mut host = host();
    let line = host.funcs[0].int_line;
    assert!(!host.line_asserted(line), "a fresh device asserts nothing");
    host.funcs[0].transport.isr = 1;
    assert!(host.line_asserted(line));
    for other in PciHost::interrupt_lines() {
        if other != line {
            assert!(
                !host.line_asserted(other),
                "only the wired line is asserted"
            );
        }
    }
    host.funcs[0].transport.isr = 0;
    assert!(
        !host.line_asserted(line),
        "reading the ISR register releases the line"
    );
}

#[test]
fn intx_disable_releases_the_line_even_with_isr_bits_set() {
    let mut host = host();
    let line = host.funcs[0].int_line;
    host.funcs[0].transport.isr = 1;
    host.funcs[0].command |= COMMAND_INTX_DISABLE;
    assert!(!host.line_asserted(line));
}
