use super::testutil::*;
use super::*;
use crate::dev::guestmem::test_mem;
use crate::dev::vblk::VirtioBlk;
use crate::disk::Disk;
use std::sync::{Arc, Mutex};

const STATUS_ACKNOWLEDGE: u8 = 1;
const STATUS_DRIVER: u8 = 2;
const STATUS_DEVICE_NEEDS_RESET: u8 = 0x40;
const RING_F_NO_INTERRUPT: u16 = 1;

const RAM_BYTES: usize = 0x2_0000;
const QUEUE_SIZE: u16 = 8;
const DESC0: u64 = 0x1000;
const AVAIL0: u64 = 0x2000;
const USED0: u64 = 0x3000;
const DESC1: u64 = 0x4000;
const AVAIL1: u64 = 0x5000;
const USED1: u64 = 0x6000;
const BUF_A: u64 = 0x8000;
const BUF_B: u64 = 0x9000;

/// A popped chain: head index and its (address, length, device-writable) segments.
type PoppedChain = (u16, Vec<(u64, u32, bool)>);

/// What the probe device saw and how it should answer; shared with the test through an `Arc`.
#[derive(Default)]
struct Probe {
    features: u64,
    queue_count: usize,
    config: [u8; 16],
    config_writes: Vec<(usize, u8)>,
    kicks: Vec<usize>,
    resets: u32,
    /// Heads and segment lists of every chain the device popped.
    popped: Vec<PoppedChain>,
    poll_pops_queue: Option<usize>,
    config_event: bool,
}

struct ProbeDev(Arc<Mutex<Probe>>);

impl ProbeDev {
    fn drain(&self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        let mut any = false;
        while let Some(chain) = queues[queue].pop(mem) {
            let mut probe = self.0.lock().unwrap();
            probe.popped.push((
                chain.head,
                chain
                    .segs
                    .iter()
                    .map(|seg| (seg.addr, seg.len, seg.device_writable))
                    .collect(),
            ));
            drop(probe);
            queues[queue].push_used(mem, chain.head, chain.writable() as u32);
            any = true;
        }
        any
    }
}

impl VirtioDev for ProbeDev {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_BLOCK
    }
    fn features(&self) -> u64 {
        self.0.lock().unwrap().features
    }
    fn num_queues(&self) -> usize {
        self.0.lock().unwrap().queue_count
    }
    fn config_read(&self, off: usize) -> u8 {
        self.0.lock().unwrap().config.get(off).copied().unwrap_or(0)
    }
    fn config_write(&mut self, off: usize, value: u8) {
        self.0.lock().unwrap().config_writes.push((off, value));
    }
    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        self.0.lock().unwrap().kicks.push(queue);
        self.drain(queue, queues, mem)
    }
    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        let queue = self.0.lock().unwrap().poll_pops_queue;
        queue.is_some_and(|queue| self.drain(queue, queues, mem))
    }
    fn config_event(&mut self) -> bool {
        std::mem::take(&mut self.0.lock().unwrap().config_event)
    }
    fn reset(&mut self) {
        self.0.lock().unwrap().resets += 1;
    }
}

struct Rig {
    transport: Transport,
    probe: Arc<Mutex<Probe>>,
    ram: Vec<u8>,
}

impl Rig {
    fn new(queue_count: usize, features: u64) -> Rig {
        let probe = Arc::new(Mutex::new(Probe {
            queue_count,
            features,
            ..Default::default()
        }));
        Rig {
            transport: Transport::new(Box::new(ProbeDev(probe.clone()))),
            probe,
            ram: vec![0; RAM_BYTES],
        }
    }

    fn read(&mut self, off: u64, size: u8) -> u32 {
        self.transport.mmio_read(off, size)
    }

    fn write(&mut self, off: u64, size: u8, value: u32) -> bool {
        let mut mem = test_mem(&mut self.ram);
        self.transport.mmio_write(off, size, value, &mut mem)
    }

    fn poll(&mut self) -> bool {
        let mut mem = test_mem(&mut self.ram);
        self.transport.poll(&mut mem)
    }

    fn status(&mut self) -> u8 {
        self.read(COMMON_DEVICE_STATUS, 1) as u8
    }

    /// Programs queue `index` through the common config registers like a driver would.
    fn program_queue(&mut self, index: u16, size: u16, desc: u64, avail: u64, used: u64) {
        self.write(COMMON_QUEUE_SELECT, 2, index as u32);
        self.write(COMMON_QUEUE_SIZE, 2, size as u32);
        for (base, addr) in [
            (COMMON_QUEUE_DESC, desc),
            (COMMON_QUEUE_DRIVER, avail),
            (COMMON_QUEUE_DEVICE, used),
        ] {
            self.write(base, 4, addr as u32);
            self.write(base + QUEUE_ADDR_HIGH_HALF, 4, (addr >> 32) as u32);
        }
        self.write(COMMON_QUEUE_ENABLE, 2, 1);
    }

    /// Full driver bring-up: negotiate everything offered, program queue 0 (and 1 if present),
    /// set DRIVER_OK.
    fn bring_up(&mut self) {
        self.write(COMMON_DEVICE_STATUS, 1, STATUS_ACKNOWLEDGE as u32);
        self.write(
            COMMON_DEVICE_STATUS,
            1,
            (STATUS_ACKNOWLEDGE | STATUS_DRIVER) as u32,
        );
        let features = self.transport.dev_features();
        for select in 0..2 {
            self.write(COMMON_DRIVER_FEATURE_SELECT, 4, select);
            self.write(COMMON_DRIVER_FEATURE, 4, (features >> (32 * select)) as u32);
        }
        let base = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK;
        self.write(COMMON_DEVICE_STATUS, 1, base as u32);
        self.program_queue(0, QUEUE_SIZE, DESC0, AVAIL0, USED0);
        if self.transport.queues.len() > 1 {
            self.program_queue(1, QUEUE_SIZE, DESC1, AVAIL1, USED1);
        }
        self.write(COMMON_DEVICE_STATUS, 1, (base | STATUS_DRIVER_OK) as u32);
    }

    fn kick(&mut self, queue: u64) -> bool {
        self.write(
            NOTIFY_OFFSET + queue * NOTIFY_OFF_MULTIPLIER as u64,
            2,
            queue as u32,
        )
    }

    fn popped(&self) -> usize {
        self.probe.lock().unwrap().popped.len()
    }
}

#[test]
fn extended_feature_selectors_do_not_alias() {
    let mut t = Transport::new(Box::new(VirtioBlk::new(Disk::memory(vec![0; 4096]))));
    let mut ram = vec![0u8; 4096];
    let mut mem = test_mem(&mut ram);
    let mut w = |t: &mut Transport, off: u64, size, v| t.mmio_write(off, size, v, &mut mem);
    // Linux 6.18 negotiates 128 feature bits: it reads and writes selectors 0..=3.
    let mut offered = [0u32; 4];
    for sel in 0..4 {
        w(&mut t, COMMON_DEVICE_FEATURE_SELECT, 4, sel);
        offered[sel as usize] = t.mmio_read(COMMON_DEVICE_FEATURE, 4);
    }
    assert_eq!(offered[2..], [0, 0]);
    assert_ne!(offered[1] & 1, 0, "VERSION_1 offered");
    for sel in 0..4 {
        w(&mut t, COMMON_DRIVER_FEATURE_SELECT, 4, sel);
        w(&mut t, COMMON_DRIVER_FEATURE, 4, offered[sel as usize]);
    }
    w(
        &mut t,
        COMMON_DEVICE_STATUS,
        1,
        (STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK) as u32,
    );
    assert_ne!(
        t.mmio_read(COMMON_DEVICE_STATUS, 1) & STATUS_FEATURES_OK as u32,
        0,
        "FEATURES_OK must stick"
    );
}

#[test]
fn avail_idx_ahead_of_ring_size_marks_device_needs_reset() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    set_avail_idx(&mut rig.ram, AVAIL0, QUEUE_SIZE + 1);
    rig.kick(0);
    assert_eq!(rig.popped(), 0);
    assert_ne!(rig.status() & STATUS_DEVICE_NEEDS_RESET, 0);
}

#[test]
fn avail_flag_no_interrupt_suppresses_interrupt_and_isr() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    set_avail_flags(&mut rig.ram, AVAIL0, RING_F_NO_INTERRUPT);
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 16, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(!rig.kick(0));
    assert_eq!(rig.popped(), 1);
    assert_eq!(used_idx(&rig.ram, USED0), 1);
    assert_eq!(rig.read(ISR_OFFSET, 1), 0);
}

#[test]
fn descriptor_loop_marks_device_needs_reset() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 16, DESC_F_NEXT, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    rig.kick(0);
    assert_eq!(rig.popped(), 0);
    assert_ne!(rig.status() & STATUS_DEVICE_NEEDS_RESET, 0);
}

#[test]
fn head_index_beyond_queue_size_marks_device_needs_reset() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, QUEUE_SIZE);
    rig.kick(0);
    assert_eq!(rig.popped(), 0);
    assert_ne!(rig.status() & STATUS_DEVICE_NEEDS_RESET, 0);
}

#[test]
fn descriptor_outside_ram_marks_device_needs_reset() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    put_desc(&mut rig.ram, DESC0, 0, RAM_BYTES as u64 - 4, 16, 0, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    rig.kick(0);
    assert_eq!(rig.popped(), 0);
    assert_ne!(rig.status() & STATUS_DEVICE_NEEDS_RESET, 0);
}

/// Offsets inside the common config region, as the spec's `struct virtio_pci_common_cfg` lays
/// them out; spelled out here so a renumbering in the implementation is caught.
const SPEC_DEVICE_FEATURE_SELECT: u64 = 0x00;
const SPEC_DEVICE_FEATURE: u64 = 0x04;
const SPEC_DRIVER_FEATURE_SELECT: u64 = 0x08;
const SPEC_DRIVER_FEATURE: u64 = 0x0c;
const SPEC_MSIX_CONFIG: u64 = 0x10;
const SPEC_NUM_QUEUES: u64 = 0x12;
const SPEC_DEVICE_STATUS: u64 = 0x14;
const SPEC_CONFIG_GENERATION: u64 = 0x15;
const SPEC_QUEUE_SELECT: u64 = 0x16;
const SPEC_QUEUE_SIZE: u64 = 0x18;
const SPEC_QUEUE_MSIX_VECTOR: u64 = 0x1a;
const SPEC_QUEUE_ENABLE: u64 = 0x1c;
const SPEC_QUEUE_NOTIFY_OFF: u64 = 0x1e;
const SPEC_QUEUE_DESC: u64 = 0x20;
const SPEC_QUEUE_DRIVER: u64 = 0x28;
const SPEC_QUEUE_DEVICE: u64 = 0x30;
const VERSION_1_HIGH_WORD_BIT: u32 = 1;
const NO_VECTOR: u32 = 0xffff;
const OFFERED_FEATURES: u64 = (1 << 40) | 0x24;

#[test]
fn spec_offsets_match_the_implementation() {
    assert_eq!(COMMON_DEVICE_FEATURE_SELECT, SPEC_DEVICE_FEATURE_SELECT);
    assert_eq!(COMMON_DEVICE_FEATURE, SPEC_DEVICE_FEATURE);
    assert_eq!(COMMON_DRIVER_FEATURE_SELECT, SPEC_DRIVER_FEATURE_SELECT);
    assert_eq!(COMMON_DRIVER_FEATURE, SPEC_DRIVER_FEATURE);
    assert_eq!(COMMON_MSIX_CONFIG, SPEC_MSIX_CONFIG);
    assert_eq!(COMMON_NUM_QUEUES, SPEC_NUM_QUEUES);
    assert_eq!(COMMON_DEVICE_STATUS, SPEC_DEVICE_STATUS);
    assert_eq!(COMMON_CONFIG_GENERATION, SPEC_CONFIG_GENERATION);
    assert_eq!(COMMON_QUEUE_SELECT, SPEC_QUEUE_SELECT);
    assert_eq!(COMMON_QUEUE_SIZE, SPEC_QUEUE_SIZE);
    assert_eq!(COMMON_QUEUE_MSIX_VECTOR, SPEC_QUEUE_MSIX_VECTOR);
    assert_eq!(COMMON_QUEUE_ENABLE, SPEC_QUEUE_ENABLE);
    assert_eq!(COMMON_QUEUE_NOTIFY_OFF, SPEC_QUEUE_NOTIFY_OFF);
    assert_eq!(COMMON_QUEUE_DESC, SPEC_QUEUE_DESC);
    assert_eq!(COMMON_QUEUE_DRIVER, SPEC_QUEUE_DRIVER);
    assert_eq!(COMMON_QUEUE_DEVICE, SPEC_QUEUE_DEVICE);
}

#[test]
fn queue_max_is_a_power_of_two_the_16_bit_indices_wrap_on() {
    assert!(QUEUE_MAX.is_power_of_two());
    assert_eq!(0x1_0000u32 % QUEUE_MAX as u32, 0);
}

#[test]
fn bar_regions_are_disjoint_and_page_sized() {
    let mut regions = [
        COMMON_CFG_OFFSET,
        ISR_OFFSET,
        DEVICE_CFG_OFFSET,
        NOTIFY_OFFSET,
    ];
    regions.sort();
    for pair in regions.windows(2) {
        assert!(pair[1] - pair[0] >= REGION_SIZE);
    }
    assert!(REGION_SIZE.is_power_of_two());
}

#[test]
fn device_feature_words_are_selected_by_device_feature_select() {
    let mut rig = Rig::new(1, OFFERED_FEATURES);
    rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, 0);
    assert_eq!(rig.read(COMMON_DEVICE_FEATURE, 4), 0x24);
    rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, 1);
    assert_eq!(
        rig.read(COMMON_DEVICE_FEATURE, 4),
        (1 << 8) | VERSION_1_HIGH_WORD_BIT
    );
}

#[test]
fn version_1_is_offered_even_when_the_device_offers_nothing() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, 0);
    assert_eq!(rig.read(COMMON_DEVICE_FEATURE, 4), 0);
    rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, 1);
    assert_eq!(rig.read(COMMON_DEVICE_FEATURE, 4), VERSION_1_HIGH_WORD_BIT);
}

#[test]
fn feature_selectors_read_back_what_was_written_including_large_values() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, 3);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 0xdead_beef);
    assert_eq!(rig.read(COMMON_DEVICE_FEATURE_SELECT, 4), 3);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE_SELECT, 4), 0xdead_beef);
}

#[test]
fn device_feature_for_selectors_two_and_up_reads_zero() {
    let mut rig = Rig::new(1, u64::MAX >> 1);
    for select in [2, 3, 0x8000_0000, u32::MAX] {
        rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, select);
        assert_eq!(
            rig.read(COMMON_DEVICE_FEATURE, 4),
            0,
            "selector {select:#x}"
        );
    }
}

#[test]
fn driver_features_are_stored_per_32_bit_word_and_read_back() {
    let mut rig = Rig::new(1, OFFERED_FEATURES);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 0);
    rig.write(COMMON_DRIVER_FEATURE, 4, 0x24);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 1);
    rig.write(COMMON_DRIVER_FEATURE, 4, 0x101);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE, 4), 0x101);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 0);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE, 4), 0x24);
    rig.write(COMMON_DRIVER_FEATURE, 4, 0x20);
    assert_eq!(
        rig.read(COMMON_DRIVER_FEATURE, 4),
        0x20,
        "rewriting a word replaces it"
    );
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 1);
    assert_eq!(
        rig.read(COMMON_DRIVER_FEATURE, 4),
        0x101,
        "other word untouched"
    );
}

#[test]
fn driver_feature_writes_for_selectors_two_and_up_are_ignored() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 2);
    rig.write(COMMON_DRIVER_FEATURE, 4, 0xffff_ffff);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE, 4), 0);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 0);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE, 4), 0);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 1);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE, 4), 0);
}

fn negotiate(rig: &mut Rig, low: u32, high: u32) -> u8 {
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 0);
    rig.write(COMMON_DRIVER_FEATURE, 4, low);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 1);
    rig.write(COMMON_DRIVER_FEATURE, 4, high);
    rig.write(
        COMMON_DEVICE_STATUS,
        1,
        (STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK) as u32,
    );
    rig.status()
}

#[test]
fn features_ok_sticks_for_version_1_plus_a_subset_of_the_offered_features() {
    let mut rig = Rig::new(1, OFFERED_FEATURES);
    let status = negotiate(&mut rig, 0x04, VERSION_1_HIGH_WORD_BIT);
    assert_eq!(
        status,
        STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK
    );
}

#[test]
fn features_ok_sticks_for_exactly_everything_offered() {
    let mut rig = Rig::new(1, OFFERED_FEATURES);
    let status = negotiate(&mut rig, 0x24, (1 << 8) | VERSION_1_HIGH_WORD_BIT);
    assert_ne!(status & STATUS_FEATURES_OK, 0);
}

#[test]
fn features_ok_is_refused_without_version_1() {
    let mut rig = Rig::new(1, OFFERED_FEATURES);
    let status = negotiate(&mut rig, 0x24, 0);
    assert_eq!(status, STATUS_ACKNOWLEDGE | STATUS_DRIVER);
}

#[test]
fn features_ok_is_refused_for_a_feature_the_device_did_not_offer() {
    let mut rig = Rig::new(1, OFFERED_FEATURES);
    let status = negotiate(&mut rig, 0x24 | 0x100, VERSION_1_HIGH_WORD_BIT);
    assert_eq!(status & STATUS_FEATURES_OK, 0);
    let status = negotiate(&mut rig, 0x24, VERSION_1_HIGH_WORD_BIT | 0x4);
    assert_eq!(status & STATUS_FEATURES_OK, 0);
}

#[test]
fn features_ok_can_be_retried_after_fixing_the_feature_set() {
    let mut rig = Rig::new(1, OFFERED_FEATURES);
    assert_eq!(negotiate(&mut rig, 0x100, 0) & STATUS_FEATURES_OK, 0);
    assert_ne!(
        negotiate(&mut rig, 0x4, VERSION_1_HIGH_WORD_BIT) & STATUS_FEATURES_OK,
        0
    );
}

#[test]
fn refused_features_ok_keeps_the_other_status_bits() {
    let mut rig = Rig::new(1, 0);
    rig.write(
        COMMON_DEVICE_STATUS,
        1,
        (STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK) as u32,
    );
    assert_eq!(
        rig.status(),
        STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_DRIVER_OK
    );
}

#[test]
fn status_bits_are_stored_and_read_back() {
    let mut rig = Rig::new(1, 0);
    assert_eq!(rig.status(), 0);
    rig.write(COMMON_DEVICE_STATUS, 1, STATUS_ACKNOWLEDGE as u32);
    assert_eq!(rig.status(), STATUS_ACKNOWLEDGE);
    rig.write(
        COMMON_DEVICE_STATUS,
        1,
        (STATUS_ACKNOWLEDGE | STATUS_DRIVER) as u32,
    );
    assert_eq!(rig.status(), STATUS_ACKNOWLEDGE | STATUS_DRIVER);
    rig.write(COMMON_DEVICE_STATUS, 1, STATUS_FAILED as u32 | 3);
    assert_eq!(rig.status(), STATUS_FAILED | 3);
}

#[test]
fn status_writes_of_the_wrong_width_are_ignored() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_DEVICE_STATUS, 2, 3);
    rig.write(COMMON_DEVICE_STATUS, 4, 3);
    assert_eq!(rig.status(), 0);
}

#[test]
fn writing_zero_status_resets_the_device() {
    let mut rig = Rig::new(2, 0);
    rig.bring_up();
    rig.write(COMMON_QUEUE_SELECT, 2, 1);
    rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, 1);
    rig.write(COMMON_DRIVER_FEATURE_SELECT, 4, 1);
    rig.write(COMMON_DRIVER_FEATURE, 4, 1);
    rig.transport.isr = 3;
    rig.transport.config_gen = 5;
    rig.write(COMMON_DEVICE_STATUS, 1, 0);
    assert_eq!(rig.status(), 0);
    assert_eq!(rig.transport.isr, 0);
    assert_eq!(rig.read(COMMON_QUEUE_SELECT, 2), 0);
    assert_eq!(rig.read(COMMON_DEVICE_FEATURE_SELECT, 4), 0);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE_SELECT, 4), 0);
    assert_eq!(rig.read(COMMON_DRIVER_FEATURE, 4), 0);
    assert_eq!(
        rig.read(COMMON_CONFIG_GENERATION, 1),
        5,
        "config_gen survives a reset"
    );
    assert_eq!(rig.probe.lock().unwrap().resets, 1);
    for queue in &rig.transport.queues {
        assert!(!queue.enabled);
        assert_eq!(
            (queue.desc, queue.avail, queue.used, queue.last_avail),
            (0, 0, 0, 0)
        );
        assert_eq!(queue.size, QUEUE_MAX);
    }
}

#[test]
fn reset_lets_the_driver_bring_the_device_up_again() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    rig.write(COMMON_DEVICE_STATUS, 1, 0);
    rig.bring_up();
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(rig.kick(0));
    assert_eq!(rig.popped(), 1);
}

#[test]
fn queue_count_is_reported() {
    for count in [1, 2, 5] {
        let mut rig = Rig::new(count, 0);
        assert_eq!(rig.read(COMMON_NUM_QUEUES, 2), count as u32);
    }
}

#[test]
fn config_generation_is_readable_as_a_byte() {
    let mut rig = Rig::new(1, 0);
    rig.transport.config_gen = 0x42;
    assert_eq!(rig.read(COMMON_CONFIG_GENERATION, 1), 0x42);
}

#[test]
fn msix_vectors_read_no_vector_and_cannot_be_set() {
    let mut rig = Rig::new(1, 0);
    assert_eq!(rig.read(COMMON_MSIX_CONFIG, 2), NO_VECTOR);
    assert_eq!(rig.read(COMMON_QUEUE_MSIX_VECTOR, 2), NO_VECTOR);
    rig.write(COMMON_MSIX_CONFIG, 2, 0);
    rig.write(COMMON_QUEUE_MSIX_VECTOR, 2, 3);
    assert_eq!(rig.read(COMMON_MSIX_CONFIG, 2), NO_VECTOR);
    assert_eq!(rig.read(COMMON_QUEUE_MSIX_VECTOR, 2), NO_VECTOR);
}

#[test]
fn queue_defaults_to_max_size_disabled_with_zero_addresses() {
    let mut rig = Rig::new(2, 0);
    assert_eq!(rig.read(COMMON_QUEUE_SIZE, 2), QUEUE_MAX as u32);
    assert_eq!(rig.read(COMMON_QUEUE_ENABLE, 2), 0);
    for base in [COMMON_QUEUE_DESC, COMMON_QUEUE_DRIVER, COMMON_QUEUE_DEVICE] {
        assert_eq!(rig.read(base, 4), 0);
        assert_eq!(rig.read(base + QUEUE_ADDR_HIGH_HALF, 4), 0);
    }
}

#[test]
fn queue_select_picks_which_queue_the_registers_address() {
    let mut rig = Rig::new(2, 0);
    rig.write(COMMON_QUEUE_SELECT, 2, 1);
    rig.write(COMMON_QUEUE_SIZE, 2, 64);
    rig.write(COMMON_QUEUE_SELECT, 2, 0);
    assert_eq!(rig.read(COMMON_QUEUE_SIZE, 2), QUEUE_MAX as u32);
    rig.write(COMMON_QUEUE_SELECT, 2, 1);
    assert_eq!(rig.read(COMMON_QUEUE_SIZE, 2), 64);
    assert_eq!(rig.read(COMMON_QUEUE_SELECT, 2), 1);
}

#[test]
fn queue_notify_off_is_the_queue_index() {
    let mut rig = Rig::new(3, 0);
    for index in 0..3 {
        rig.write(COMMON_QUEUE_SELECT, 2, index);
        assert_eq!(rig.read(COMMON_QUEUE_NOTIFY_OFF, 2), index);
    }
}

#[test]
fn nonexistent_queue_reads_size_zero_and_ignores_writes() {
    let mut rig = Rig::new(2, 0);
    rig.write(COMMON_QUEUE_SELECT, 2, 2);
    assert_eq!(rig.read(COMMON_QUEUE_SIZE, 2), 0);
    rig.write(COMMON_QUEUE_SIZE, 2, 16);
    rig.write(COMMON_QUEUE_ENABLE, 2, 1);
    rig.write(COMMON_QUEUE_DESC, 4, 0x1000);
    assert_eq!(rig.read(COMMON_QUEUE_SIZE, 2), 0);
    assert_eq!(rig.read(COMMON_QUEUE_ENABLE, 2), 0);
    assert_eq!(rig.read(COMMON_QUEUE_DESC, 4), 0);
    rig.write(COMMON_QUEUE_SELECT, 2, 0xffff);
    assert_eq!(rig.read(COMMON_QUEUE_SIZE, 2), 0);
    for queue in &rig.transport.queues {
        assert!(!queue.enabled && queue.desc == 0);
    }
}

#[test]
fn queue_size_accepts_powers_of_two_up_to_the_maximum() {
    let mut rig = Rig::new(1, 0);
    for size in [1u32, 2, 4, 8, 16, 32, 64, 128, 256] {
        rig.write(COMMON_QUEUE_SIZE, 2, size);
        assert_eq!(rig.read(COMMON_QUEUE_SIZE, 2), size);
    }
}

#[test]
fn queue_size_rejects_zero_non_powers_and_too_large_values() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_QUEUE_SIZE, 2, 16);
    for bad in [0u32, 3, 100, 255, 512, 0x8000, 0x1_0010] {
        rig.write(COMMON_QUEUE_SIZE, 2, bad);
        assert_eq!(
            rig.read(COMMON_QUEUE_SIZE, 2),
            16,
            "size {bad:#x} must be refused"
        );
    }
}

#[test]
fn queue_addresses_roundtrip_as_two_32_bit_halves() {
    let mut rig = Rig::new(1, 0);
    for base in [COMMON_QUEUE_DESC, COMMON_QUEUE_DRIVER, COMMON_QUEUE_DEVICE] {
        rig.write(base, 4, 0x1234_5000);
        rig.write(base + QUEUE_ADDR_HIGH_HALF, 4, 0x0000_00ab);
        assert_eq!(rig.read(base, 4), 0x1234_5000);
        assert_eq!(rig.read(base + QUEUE_ADDR_HIGH_HALF, 4), 0xab);
    }
    let queue = &rig.transport.queues[0];
    assert_eq!(queue.desc, 0xab_1234_5000);
    assert_eq!(queue.avail, 0xab_1234_5000);
    assert_eq!(queue.used, 0xab_1234_5000);
}

#[test]
fn writing_one_half_of_a_queue_address_keeps_the_other() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_QUEUE_DESC, 4, 0xffff_ffff);
    rig.write(COMMON_QUEUE_DESC + QUEUE_ADDR_HIGH_HALF, 4, 0x7);
    rig.write(COMMON_QUEUE_DESC, 4, 0x1000);
    assert_eq!(rig.transport.queues[0].desc, 0x7_0000_1000);
    rig.write(COMMON_QUEUE_DESC + QUEUE_ADDR_HIGH_HALF, 4, 0x1);
    assert_eq!(rig.transport.queues[0].desc, 0x1_0000_1000);
}

#[test]
fn queue_address_reads_work_at_byte_and_word_granularity() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_QUEUE_DESC, 4, 0x4433_2211);
    rig.write(COMMON_QUEUE_DESC + QUEUE_ADDR_HIGH_HALF, 4, 0x8877_6655);
    assert_eq!(rig.read(COMMON_QUEUE_DESC, 1), 0x11);
    assert_eq!(rig.read(COMMON_QUEUE_DESC + 1, 1), 0x22);
    assert_eq!(rig.read(COMMON_QUEUE_DESC + 2, 2), 0x4433);
    assert_eq!(rig.read(COMMON_QUEUE_DESC + 6, 2), 0x8877);
    assert_eq!(rig.read(COMMON_QUEUE_DESC + 7, 1), 0x88);
}

#[test]
fn queue_address_writes_that_are_not_dwords_are_ignored() {
    let mut rig = Rig::new(1, 0);
    rig.write(COMMON_QUEUE_DESC, 2, 0x1234);
    rig.write(COMMON_QUEUE_DESC, 1, 0x12);
    rig.write(COMMON_QUEUE_DESC + 2, 4, 0x1234);
    assert_eq!(rig.transport.queues[0].desc, 0);
}

#[test]
fn queue_enable_sets_the_flag_and_rewinds_last_avail() {
    let mut rig = Rig::new(1, 0);
    rig.transport.queues[0].last_avail = 77;
    rig.write(COMMON_QUEUE_ENABLE, 2, 1);
    assert_eq!(rig.read(COMMON_QUEUE_ENABLE, 2), 1);
    assert_eq!(rig.transport.queues[0].last_avail, 0);
    rig.write(COMMON_QUEUE_ENABLE, 2, 0);
    assert_eq!(rig.read(COMMON_QUEUE_ENABLE, 2), 0);
}

#[test]
fn common_registers_read_at_odd_offsets_see_the_right_bytes() {
    let mut rig = Rig::new(3, 0);
    assert_eq!(rig.read(COMMON_NUM_QUEUES, 1), 3);
    assert_eq!(rig.read(COMMON_NUM_QUEUES + 1, 1), 0);
    rig.write(COMMON_DEVICE_FEATURE_SELECT, 4, 0x1122_3344);
    assert_eq!(rig.read(COMMON_DEVICE_FEATURE_SELECT + 1, 1), 0x33);
    assert_eq!(rig.read(COMMON_DEVICE_FEATURE_SELECT + 2, 2), 0x1122);
}

#[test]
fn common_config_bytes_past_the_last_register_read_zero() {
    let mut rig = Rig::new(1, 0);
    for off in [0x38, 0x3c, 0x40, 0x100, 0xffc, 0xfff] {
        assert_eq!(rig.read(COMMON_CFG_OFFSET + off, 4), 0, "offset {off:#x}");
    }
}

#[test]
fn region_offsets_wrap_within_their_4k_window() {
    let mut rig = Rig::new(2, 0);
    assert_eq!(rig.read(COMMON_CFG_OFFSET + 0x12, 2), 2);
    assert_eq!(
        rig.read(NOTIFY_OFFSET, 4),
        0,
        "the notify region is write-only"
    );
    assert_eq!(rig.read(REGION_SIZE * 4, 4), 0);
    assert_eq!(rig.read(REGION_SIZE * 7, 4), 0);
}

#[test]
fn device_config_reads_compose_bytes_little_endian() {
    let mut rig = Rig::new(1, 0);
    rig.probe.lock().unwrap().config[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(rig.read(DEVICE_CFG_OFFSET, 1), 1);
    assert_eq!(rig.read(DEVICE_CFG_OFFSET + 1, 2), 0x0302);
    assert_eq!(rig.read(DEVICE_CFG_OFFSET, 4), 0x0403_0201);
    assert_eq!(rig.read(DEVICE_CFG_OFFSET + 4, 4), 0x0807_0605);
    assert_eq!(
        rig.read(DEVICE_CFG_OFFSET + 15, 4),
        0,
        "past the config bytes the device returns 0"
    );
}

#[test]
fn device_config_writes_are_forwarded_byte_by_byte() {
    let mut rig = Rig::new(1, 0);
    rig.write(DEVICE_CFG_OFFSET + 4, 4, 0x4433_2211);
    rig.write(DEVICE_CFG_OFFSET + 9, 1, 0x7f);
    rig.write(DEVICE_CFG_OFFSET + 2, 2, 0xbeef);
    assert_eq!(
        rig.probe.lock().unwrap().config_writes,
        vec![
            (4, 0x11),
            (5, 0x22),
            (6, 0x33),
            (7, 0x44),
            (9, 0x7f),
            (2, 0xef),
            (3, 0xbe)
        ]
    );
}

#[test]
fn notify_before_driver_ok_does_nothing() {
    let mut rig = Rig::new(1, 0);
    rig.program_queue(0, QUEUE_SIZE, DESC0, AVAIL0, USED0);
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(!rig.kick(0));
    assert!(rig.probe.lock().unwrap().kicks.is_empty());
    assert_eq!(used_idx(&rig.ram, USED0), 0);
}

#[test]
fn notify_after_failed_does_nothing() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    rig.write(
        COMMON_DEVICE_STATUS,
        1,
        (STATUS_DRIVER_OK | STATUS_FAILED) as u32,
    );
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(!rig.kick(0));
    assert!(rig.probe.lock().unwrap().kicks.is_empty());
    assert!(!rig.transport.driver_ok());
}

#[test]
fn notify_address_selects_the_queue_by_multiplier() {
    let mut rig = Rig::new(2, 0);
    rig.bring_up();
    rig.write(NOTIFY_OFFSET, 2, 7);
    rig.write(NOTIFY_OFFSET + NOTIFY_OFF_MULTIPLIER as u64, 2, 7);
    rig.write(NOTIFY_OFFSET + 3, 1, 7);
    rig.write(NOTIFY_OFFSET + 2 * NOTIFY_OFF_MULTIPLIER as u64, 2, 0);
    rig.write(NOTIFY_OFFSET + 0xff0, 2, 0);
    assert_eq!(rig.probe.lock().unwrap().kicks, vec![0, 1, 0]);
}

#[test]
fn notify_value_written_is_irrelevant() {
    let mut rig = Rig::new(2, 0);
    rig.bring_up();
    rig.write(NOTIFY_OFFSET + NOTIFY_OFF_MULTIPLIER as u64, 2, 0);
    rig.write(NOTIFY_OFFSET, 4, 1);
    assert_eq!(rig.probe.lock().unwrap().kicks, vec![1, 0]);
}

#[test]
fn completed_kick_raises_once_until_the_isr_is_read() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    for head in 0..2u16 {
        put_desc(&mut rig.ram, DESC0, head, BUF_A, 4, DESC_F_WRITE, 0);
        post(&mut rig.ram, AVAIL0, QUEUE_SIZE, head);
    }
    // Both buffers are available before the first kick: one interrupt covers them.
    assert!(rig.kick(0));
    assert_eq!(rig.popped(), 2);
    put_desc(&mut rig.ram, DESC0, 2, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 2);
    assert!(
        !rig.kick(0),
        "ISR still set: edge-triggered line must not re-raise"
    );
    assert_eq!(rig.read(ISR_OFFSET, 1), 1);
    assert_eq!(rig.read(ISR_OFFSET, 1), 0, "reading the ISR clears it");
    put_desc(&mut rig.ram, DESC0, 3, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 3);
    assert!(
        rig.kick(0),
        "after the read the next completion raises again"
    );
}

#[test]
fn kick_with_nothing_available_raises_nothing() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    assert!(!rig.kick(0));
    assert_eq!(rig.read(ISR_OFFSET, 1), 0);
}

#[test]
fn isr_reads_clear_at_any_offset_and_size() {
    let mut rig = Rig::new(1, 0);
    rig.transport.isr = 3;
    assert_eq!(rig.read(ISR_OFFSET + 0x20, 4), 3);
    assert_eq!(rig.transport.isr, 0);
}

#[test]
fn isr_writes_are_ignored() {
    let mut rig = Rig::new(1, 0);
    rig.transport.isr = 1;
    rig.write(ISR_OFFSET, 1, 0);
    assert_eq!(rig.transport.isr, 1);
}

#[test]
fn poll_is_inert_before_driver_ok() {
    let mut rig = Rig::new(1, 0);
    rig.probe.lock().unwrap().config_event = true;
    assert!(!rig.poll());
    assert_eq!(rig.read(ISR_OFFSET, 1), 0);
    assert_eq!(rig.transport.config_gen, 0);
}

#[test]
fn poll_completion_sets_the_queue_interrupt_bit_once() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    rig.probe.lock().unwrap().poll_pops_queue = Some(0);
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(rig.poll());
    put_desc(&mut rig.ram, DESC0, 1, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 1);
    assert!(!rig.poll(), "ISR not read yet");
    assert_eq!(rig.read(ISR_OFFSET, 1), 1);
}

#[test]
fn config_event_bumps_generation_and_sets_the_config_isr_bit() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    rig.probe.lock().unwrap().config_event = true;
    assert!(rig.poll());
    assert_eq!(rig.read(COMMON_CONFIG_GENERATION, 1), 1);
    assert_eq!(rig.read(ISR_OFFSET, 1), 2);
    assert!(!rig.poll(), "the event is consumed");
    assert_eq!(rig.read(COMMON_CONFIG_GENERATION, 1), 1);
}

#[test]
fn config_generation_wraps_at_256() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    rig.transport.config_gen = 255;
    rig.probe.lock().unwrap().config_event = true;
    rig.poll();
    assert_eq!(rig.read(COMMON_CONFIG_GENERATION, 1), 0);
}

#[test]
fn completion_and_config_event_in_one_poll_set_both_isr_bits() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    {
        let mut probe = rig.probe.lock().unwrap();
        probe.poll_pops_queue = Some(0);
        probe.config_event = true;
    }
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(rig.poll());
    assert_eq!(rig.read(ISR_OFFSET, 1), 3);
}

#[test]
fn driver_ok_requires_the_bit_and_no_failed_or_needs_reset() {
    let mut rig = Rig::new(1, 0);
    assert!(!rig.transport.driver_ok());
    for (status, expected) in [
        (STATUS_DRIVER_OK, true),
        (STATUS_DRIVER_OK | STATUS_FAILED, false),
        (STATUS_ACKNOWLEDGE | STATUS_DRIVER, false),
    ] {
        rig.transport.status = status;
        assert_eq!(rig.transport.driver_ok(), expected, "status {status:#x}");
    }
    rig.transport.status = STATUS_DRIVER_OK | STATUS_DEVICE_NEEDS_RESET;
    assert!(!rig.transport.driver_ok());
}

#[test]
fn avail_overrun_notifies_with_a_config_change_and_stops_the_device() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    set_avail_idx(&mut rig.ram, AVAIL0, QUEUE_SIZE + 1);
    assert!(rig.kick(0), "the driver is told through an interrupt");
    assert_eq!(rig.read(ISR_OFFSET, 1), 2, "configuration change bit");
    assert_eq!(
        rig.status(),
        STATUS_ACKNOWLEDGE
            | STATUS_DRIVER
            | STATUS_FEATURES_OK
            | STATUS_DRIVER_OK
            | STATUS_DEVICE_NEEDS_RESET
    );
    assert!(rig.transport.queues[0].broken);
    // Fixing the ring does not revive the device: only a reset does.
    set_avail_idx(&mut rig.ram, AVAIL0, 0);
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(!rig.kick(0));
    assert_eq!(rig.popped(), 0);
    assert_eq!(
        rig.probe.lock().unwrap().kicks,
        vec![0],
        "no kick reaches the device again"
    );
}

#[test]
fn reset_clears_needs_reset_and_the_broken_queue() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    set_avail_idx(&mut rig.ram, AVAIL0, QUEUE_SIZE + 1);
    rig.kick(0);
    rig.write(COMMON_DEVICE_STATUS, 1, 0);
    assert_eq!(rig.status(), 0);
    assert!(!rig.transport.queues[0].broken);
    set_avail_idx(&mut rig.ram, AVAIL0, 0);
    rig.bring_up();
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(rig.kick(0));
}

#[test]
fn driver_cannot_set_or_clear_needs_reset() {
    let mut rig = Rig::new(1, 0);
    rig.write(
        COMMON_DEVICE_STATUS,
        1,
        (STATUS_DRIVER | STATUS_DEVICE_NEEDS_RESET) as u32,
    );
    assert_eq!(rig.status(), STATUS_DRIVER);
    rig.bring_up();
    set_avail_idx(&mut rig.ram, AVAIL0, QUEUE_SIZE + 1);
    rig.kick(0);
    rig.write(
        COMMON_DEVICE_STATUS,
        1,
        (STATUS_ACKNOWLEDGE | STATUS_DRIVER) as u32,
    );
    assert_ne!(rig.status() & STATUS_DEVICE_NEEDS_RESET, 0);
}

#[test]
fn broken_queue_found_while_polling_sets_needs_reset_and_raises_config_interrupt() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    rig.probe.lock().unwrap().poll_pops_queue = Some(0);
    set_avail_idx(&mut rig.ram, AVAIL0, QUEUE_SIZE + 1);
    assert!(rig.poll());
    assert_ne!(rig.status() & STATUS_DEVICE_NEEDS_RESET, 0);
    assert_eq!(rig.read(ISR_OFFSET, 1), 2);
    assert!(!rig.poll(), "a device that needs reset is no longer polled");
}

#[test]
fn no_interrupt_flag_applies_per_completion() {
    let mut rig = Rig::new(1, 0);
    rig.bring_up();
    set_avail_flags(&mut rig.ram, AVAIL0, RING_F_NO_INTERRUPT);
    put_desc(&mut rig.ram, DESC0, 0, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 0);
    assert!(!rig.kick(0));
    set_avail_flags(&mut rig.ram, AVAIL0, 0);
    put_desc(&mut rig.ram, DESC0, 1, BUF_A, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL0, QUEUE_SIZE, 1);
    assert!(rig.kick(0));
}

#[test]
fn queues_of_one_device_have_independent_state() {
    let mut rig = Rig::new(2, 0);
    rig.bring_up();
    put_desc(&mut rig.ram, DESC1, 0, BUF_B, 4, DESC_F_WRITE, 0);
    post(&mut rig.ram, AVAIL1, QUEUE_SIZE, 0);
    assert!(rig.kick(1));
    assert_eq!(used_idx(&rig.ram, USED1), 1);
    assert_eq!(used_idx(&rig.ram, USED0), 0);
}

struct QueueRig {
    ram: Vec<u8>,
    queue: Queue,
}

impl QueueRig {
    fn new(size: u16) -> QueueRig {
        let mut ram = vec![0u8; RAM_BYTES];
        let queue = setup_queue(&mut ram, DESC0, AVAIL0, USED0, size);
        QueueRig { ram, queue }
    }

    fn pop(&mut self) -> Option<Chain> {
        let mem = test_mem(&mut self.ram);
        self.queue.pop(&mem)
    }

    fn push_used(&mut self, head: u16, len: u32) {
        let mut mem = test_mem(&mut self.ram);
        self.queue.push_used(&mut mem, head, len);
    }
}

#[test]
fn ready_needs_enable_size_and_all_three_addresses() {
    let full = Queue {
        size: 8,
        enabled: true,
        desc: 1,
        avail: 2,
        used: 3,
        ..Default::default()
    };
    assert!(full.ready());
    assert!(
        !Queue {
            enabled: false,
            ..full.clone()
        }
        .ready()
    );
    assert!(
        !Queue {
            size: 0,
            ..full.clone()
        }
        .ready()
    );
    assert!(
        !Queue {
            desc: 0,
            ..full.clone()
        }
        .ready()
    );
    assert!(
        !Queue {
            avail: 0,
            ..full.clone()
        }
        .ready()
    );
    assert!(
        !Queue {
            used: 0,
            ..full.clone()
        }
        .ready()
    );
}

#[test]
fn pop_from_a_queue_that_is_not_ready_returns_none() {
    let mut q = QueueRig::new(8);
    post(&mut q.ram, AVAIL0, 8, 0);
    q.queue.enabled = false;
    assert!(q.pop().is_none());
    assert!(!q.queue.broken);
}

#[test]
fn pop_on_an_empty_ring_returns_none() {
    let mut q = QueueRig::new(8);
    assert!(q.pop().is_none());
    assert!(!q.queue.broken);
}

#[test]
fn pop_returns_a_single_descriptor_chain() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 3, BUF_A, 100, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 3);
    let chain = q.pop().unwrap();
    assert_eq!(chain.head, 3);
    assert_eq!(chain.segs.len(), 1);
    assert_eq!(
        (
            chain.segs[0].addr,
            chain.segs[0].len,
            chain.segs[0].device_writable
        ),
        (BUF_A, 100, false)
    );
    assert!(q.pop().is_none());
    assert_eq!(q.queue.last_avail, 1);
}

#[test]
fn pop_follows_next_links_and_keeps_segment_order_and_direction() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 16, DESC_F_NEXT, 5);
    put_desc(
        &mut q.ram,
        DESC0,
        5,
        BUF_B,
        32,
        DESC_F_NEXT | DESC_F_WRITE,
        2,
    );
    put_desc(&mut q.ram, DESC0, 2, BUF_B + 0x100, 1, DESC_F_WRITE, 7);
    post(&mut q.ram, AVAIL0, 8, 0);
    let chain = q.pop().unwrap();
    assert_eq!(chain.head, 0);
    let segs: Vec<_> = chain
        .segs
        .iter()
        .map(|s| (s.addr, s.len, s.device_writable))
        .collect();
    assert_eq!(
        segs,
        vec![
            (BUF_A, 16, false),
            (BUF_B, 32, true),
            (BUF_B + 0x100, 1, true)
        ]
    );
}

#[test]
fn next_field_is_ignored_when_the_next_flag_is_clear() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 4, 0, 1);
    put_desc(&mut q.ram, DESC0, 1, BUF_B, 4, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert_eq!(q.pop().unwrap().segs.len(), 1);
}

#[test]
fn pop_returns_chains_in_posting_order() {
    let mut q = QueueRig::new(8);
    for head in [4u16, 1, 6] {
        put_desc(&mut q.ram, DESC0, head, BUF_A, 4, 0, 0);
        post(&mut q.ram, AVAIL0, 8, head);
    }
    let heads: Vec<u16> = std::iter::from_fn(|| q.pop())
        .map(|chain| chain.head)
        .collect();
    assert_eq!(heads, vec![4, 1, 6]);
}

#[test]
fn chain_using_every_descriptor_of_the_queue_is_valid() {
    let mut q = QueueRig::new(4);
    for index in 0..4u16 {
        let (flags, next) = if index == 3 {
            (0, 0)
        } else {
            (DESC_F_NEXT, index + 1)
        };
        put_desc(
            &mut q.ram,
            DESC0,
            index,
            BUF_A + 0x10 * index as u64,
            4,
            flags,
            next,
        );
    }
    post(&mut q.ram, AVAIL0, 4, 0);
    assert_eq!(q.pop().unwrap().segs.len(), 4);
    assert!(!q.queue.broken);
}

#[test]
fn full_ring_of_outstanding_entries_is_not_an_overrun() {
    let mut q = QueueRig::new(4);
    for head in 0..4u16 {
        put_desc(&mut q.ram, DESC0, head, BUF_A, 4, 0, 0);
        post(&mut q.ram, AVAIL0, 4, head);
    }
    assert_eq!(std::iter::from_fn(|| q.pop()).count(), 4);
    assert!(!q.queue.broken);
}

#[test]
fn ring_slot_is_the_free_running_index_modulo_the_size() {
    let mut q = QueueRig::new(4);
    for round in 0..3u16 {
        for slot in 0..4u16 {
            let head = (slot + round) % 4;
            put_desc(&mut q.ram, DESC0, head, BUF_A, 4, 0, 0);
            post(&mut q.ram, AVAIL0, 4, head);
            assert_eq!(q.pop().unwrap().head, head);
        }
    }
    assert_eq!(q.queue.last_avail, 12);
}

#[test]
fn available_index_wrap_around_at_65536_keeps_working() {
    let mut q = QueueRig::new(8);
    q.queue.last_avail = u16::MAX;
    set_avail_idx(&mut q.ram, AVAIL0, u16::MAX);
    put_desc(&mut q.ram, DESC0, 2, BUF_A, 4, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 2);
    assert_eq!(
        q.ram[AVAIL0 as usize + 2..AVAIL0 as usize + 4],
        [0, 0],
        "idx wrapped to 0"
    );
    // 65535 % 8 == 7: the head was posted into slot 7.
    assert_eq!(q.pop().unwrap().head, 2);
    assert_eq!(q.queue.last_avail, 0);
    assert!(q.pop().is_none());
    assert!(!q.queue.broken);
}

#[test]
fn used_index_wrap_around_at_65536_keeps_working() {
    let mut q = QueueRig::new(8);
    let used = USED0 as usize;
    q.ram[used + 2..used + 4].copy_from_slice(&u16::MAX.to_le_bytes());
    q.push_used(6, 0x1234);
    assert_eq!(used_idx(&q.ram, USED0), 0);
    assert_eq!(used_entry(&q.ram, USED0, 7), (6, 0x1234));
    q.push_used(1, 9);
    assert_eq!(used_idx(&q.ram, USED0), 1);
    assert_eq!(used_entry(&q.ram, USED0, 0), (1, 9));
}

#[test]
fn push_used_records_head_and_length_and_increments_idx() {
    let mut q = QueueRig::new(8);
    q.push_used(5, 100);
    q.push_used(2, 0);
    q.push_used(7, u32::MAX);
    assert_eq!(used_idx(&q.ram, USED0), 3);
    assert_eq!(used_entry(&q.ram, USED0, 0), (5, 100));
    assert_eq!(used_entry(&q.ram, USED0, 1), (2, 0));
    assert_eq!(used_entry(&q.ram, USED0, 2), (7, u32::MAX));
}

#[test]
fn push_used_wraps_the_slot_by_queue_size() {
    let mut q = QueueRig::new(2);
    for head in 0..3u16 {
        q.push_used(head, head as u32 * 10);
    }
    assert_eq!(used_entry(&q.ram, USED0, 0), (2, 20));
    assert_eq!(used_entry(&q.ram, USED0, 1), (1, 10));
}

#[test]
fn push_used_flags_an_interrupt_unless_the_driver_suppressed_it() {
    let mut q = QueueRig::new(8);
    q.push_used(0, 0);
    assert!(q.queue.interrupt_due);
    q.queue.interrupt_due = false;
    set_avail_flags(&mut q.ram, AVAIL0, RING_F_NO_INTERRUPT);
    q.push_used(0, 0);
    assert!(!q.queue.interrupt_due);
    set_avail_flags(&mut q.ram, AVAIL0, 0x2);
    q.push_used(0, 0);
    assert!(q.queue.interrupt_due, "only bit 0 suppresses");
}

#[test]
fn suppressed_completion_does_not_clear_a_pending_interrupt() {
    let mut q = QueueRig::new(8);
    q.push_used(0, 0);
    set_avail_flags(&mut q.ram, AVAIL0, RING_F_NO_INTERRUPT);
    q.push_used(1, 0);
    assert!(q.queue.interrupt_due);
}

#[test]
fn unpop_returns_the_chain_to_the_ring() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 3, BUF_A, 4, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 3);
    assert_eq!(q.pop().unwrap().head, 3);
    q.queue.unpop();
    assert_eq!(q.pop().unwrap().head, 3);
    assert!(q.pop().is_none());
}

#[test]
fn unpop_wraps_below_zero() {
    let mut q = QueueRig::new(8);
    q.queue.unpop();
    assert_eq!(q.queue.last_avail, u16::MAX);
}

#[test]
fn available_index_ahead_by_more_than_the_size_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    set_avail_idx(&mut q.ram, AVAIL0, 9);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn available_index_behind_last_avail_is_an_overrun_too() {
    let mut q = QueueRig::new(8);
    q.queue.last_avail = 10;
    set_avail_idx(&mut q.ram, AVAIL0, 5);
    assert!(q.pop().is_none());
    assert!(
        q.queue.broken,
        "idx moving backwards wraps to a huge pending count"
    );
}

#[test]
fn broken_queue_never_yields_again_even_if_the_ring_is_repaired() {
    let mut q = QueueRig::new(8);
    set_avail_idx(&mut q.ram, AVAIL0, 200);
    assert!(q.pop().is_none());
    set_avail_idx(&mut q.ram, AVAIL0, 1);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 4, 0, 0);
    assert!(q.pop().is_none());
}

#[test]
fn indirect_descriptors_break_the_queue() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 16, DESC_F_INDIRECT, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn indirect_descriptor_in_the_middle_of_a_chain_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 16, DESC_F_NEXT, 1);
    put_desc(&mut q.ram, DESC0, 1, BUF_B, 16, DESC_F_INDIRECT, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn self_referencing_descriptor_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 2, BUF_A, 4, DESC_F_NEXT, 2);
    post(&mut q.ram, AVAIL0, 8, 2);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn two_descriptor_cycle_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 4, DESC_F_NEXT, 1);
    put_desc(&mut q.ram, DESC0, 1, BUF_A, 4, DESC_F_NEXT, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn head_index_equal_to_the_queue_size_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    post(&mut q.ram, AVAIL0, 8, 8);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn next_index_outside_the_queue_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 4, DESC_F_NEXT, 8);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn descriptor_table_partly_outside_ram_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    q.queue.desc = RAM_BYTES as u64 - 16 * 8 + 1;
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn rings_ending_exactly_at_the_end_of_ram_are_accepted() {
    let mut q = QueueRig::new(8);
    q.queue.desc = RAM_BYTES as u64 - 16 * 8;
    q.queue.used = RAM_BYTES as u64 - (6 + 8 * 8);
    q.queue.avail = RAM_BYTES as u64 - (6 + 2 * 8);
    assert!(q.pop().is_none());
    assert!(!q.queue.broken);
}

#[test]
fn available_and_used_rings_outside_ram_break_the_queue() {
    for which in 0..2 {
        let mut q = QueueRig::new(8);
        if which == 0 {
            q.queue.avail = RAM_BYTES as u64 - 4;
        } else {
            q.queue.used = RAM_BYTES as u64 - 4;
        }
        assert!(q.pop().is_none());
        assert!(q.queue.broken, "ring {which}");
    }
}

#[test]
fn queue_address_overflowing_u64_breaks_the_queue_without_panicking() {
    let mut q = QueueRig::new(8);
    q.queue.desc = u64::MAX - 4;
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn segment_ending_exactly_at_the_end_of_ram_is_valid() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, RAM_BYTES as u64 - 16, 16, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_some());
}

#[test]
fn segment_crossing_the_end_of_ram_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, RAM_BYTES as u64 - 16, 17, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn segment_with_overflowing_address_breaks_the_queue() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, u64::MAX - 1, 16, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert!(q.pop().is_none());
    assert!(q.queue.broken);
}

#[test]
fn zero_length_segment_is_allowed() {
    let mut q = QueueRig::new(8);
    put_desc(&mut q.ram, DESC0, 0, BUF_A, 0, 0, 0);
    post(&mut q.ram, AVAIL0, 8, 0);
    assert_eq!(q.pop().unwrap().segs[0].len, 0);
}

fn chain_of(segs: &[(u64, u32, bool)]) -> Chain {
    Chain {
        head: 0,
        segs: segs
            .iter()
            .map(|&(addr, len, device_writable)| Segment {
                addr,
                len,
                device_writable,
            })
            .collect(),
    }
}

#[test]
fn read_all_concatenates_readable_segments_and_skips_writable_ones() {
    let mut ram = vec![0u8; RAM_BYTES];
    ram[0x100..0x104].copy_from_slice(b"abcd");
    ram[0x200..0x202].copy_from_slice(b"ef");
    ram[0x300..0x303].copy_from_slice(b"zzz");
    let chain = chain_of(&[(0x100, 4, false), (0x300, 3, true), (0x200, 2, false)]);
    let mem = test_mem(&mut ram);
    assert_eq!(chain.read_all(&mem, usize::MAX), b"abcdef");
    assert_eq!(chain.readable(), 6);
    assert_eq!(chain.writable(), 3);
}

#[test]
fn read_all_truncates_at_the_limit_even_inside_a_segment() {
    let mut ram = vec![0u8; RAM_BYTES];
    ram[0x100..0x104].copy_from_slice(b"abcd");
    ram[0x200..0x202].copy_from_slice(b"ef");
    let chain = chain_of(&[(0x100, 4, false), (0x200, 2, false)]);
    let mem = test_mem(&mut ram);
    assert_eq!(chain.read_all(&mem, 5), b"abcde");
    assert_eq!(chain.read_all(&mem, 4), b"abcd");
    assert_eq!(chain.read_all(&mem, 1), b"a");
    assert_eq!(chain.read_all(&mem, 0), b"");
}

#[test]
fn read_all_of_a_chain_without_readable_segments_is_empty() {
    let mut ram = vec![0u8; RAM_BYTES];
    let chain = chain_of(&[(0x100, 4, true)]);
    let mem = test_mem(&mut ram);
    assert!(chain.read_all(&mem, 100).is_empty());
    assert_eq!(chain.readable(), 0);
}

#[test]
fn read_chunks_delivers_every_readable_byte_in_bounded_pieces() {
    let mut ram = vec![0u8; RAM_BYTES];
    for (i, byte) in ram[0x100..0x10a].iter_mut().enumerate() {
        *byte = i as u8;
    }
    ram[0x200..0x203].copy_from_slice(&[0xa0, 0xa1, 0xa2]);
    let chain = chain_of(&[(0x100, 10, false), (0x180, 5, true), (0x200, 3, false)]);
    let mem = test_mem(&mut ram);
    let mut pieces: Vec<Vec<u8>> = Vec::new();
    chain.read_chunks(&mem, 4, |piece| pieces.push(piece.to_vec()));
    assert_eq!(
        pieces,
        vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9],
            vec![0xa0, 0xa1, 0xa2]
        ]
    );
}

#[test]
fn read_chunks_on_empty_chain_never_calls_the_sink() {
    let mut ram = vec![0u8; RAM_BYTES];
    let chain = chain_of(&[(0x100, 0, false)]);
    let mem = test_mem(&mut ram);
    chain.read_chunks(&mem, 16, |_| panic!("no data"));
}

#[test]
fn write_at_fills_writable_segments_in_order_and_reports_what_fit() {
    let mut ram = vec![0u8; RAM_BYTES];
    let chain = chain_of(&[(0x100, 3, true), (0x180, 4, false), (0x200, 4, true)]);
    let mut mem = test_mem(&mut ram);
    assert_eq!(chain.write_at(&mut mem, 0, b"0123456789"), 7);
    assert_eq!(&ram[0x100..0x103], b"012");
    assert_eq!(
        &ram[0x180..0x184],
        &[0; 4],
        "readable segments are never written"
    );
    assert_eq!(&ram[0x200..0x204], b"3456");
    assert_eq!(&ram[0x204..0x208], &[0; 4]);
}

#[test]
fn write_at_offset_skips_into_the_right_segment() {
    let mut ram = vec![0u8; RAM_BYTES];
    let chain = chain_of(&[(0x100, 4, true), (0x200, 4, true)]);
    let mut mem = test_mem(&mut ram);
    assert_eq!(chain.write_at(&mut mem, 2, b"abcd"), 4);
    assert_eq!(&ram[0x100..0x104], &[0, 0, b'a', b'b']);
    assert_eq!(&ram[0x200..0x204], &[b'c', b'd', 0, 0]);
    let mut mem = test_mem(&mut ram);
    assert_eq!(chain.write_at(&mut mem, 4, b"X"), 1);
    assert_eq!(ram[0x200], b'X');
}

#[test]
fn write_at_offset_beyond_the_writable_space_writes_nothing() {
    let mut ram = vec![0u8; RAM_BYTES];
    let chain = chain_of(&[(0x100, 4, true)]);
    let mut mem = test_mem(&mut ram);
    assert_eq!(chain.write_at(&mut mem, 4, b"abc"), 0);
    assert_eq!(chain.write_at(&mut mem, 100, b"abc"), 0);
    assert_eq!(chain.write_at(&mut mem, 0, b""), 0);
    assert_eq!(&ram[0x100..0x108], &[0; 8]);
}

#[test]
fn write_at_the_last_byte_of_the_last_segment() {
    let mut ram = vec![0u8; RAM_BYTES];
    let chain = chain_of(&[(0x100, 2, true), (0x200, 3, true)]);
    let mut mem = test_mem(&mut ram);
    assert_eq!(chain.write_at(&mut mem, 4, &[0x55]), 1);
    assert_eq!(ram[0x202], 0x55);
}

#[test]
fn write_at_skips_zero_length_writable_segments() {
    let mut ram = vec![0u8; RAM_BYTES];
    let chain = chain_of(&[(0x100, 0, true), (0x200, 2, true)]);
    let mut mem = test_mem(&mut ram);
    assert_eq!(chain.write_at(&mut mem, 0, b"ab"), 2);
    assert_eq!(&ram[0x200..0x202], b"ab");
}

#[test]
fn write_at_records_code_pages_it_dirtied() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut code = vec![0u8; RAM_BYTES >> 12];
    code[0x1f] = 1;
    let chain = chain_of(&[(0x1f000, 8, true)]);
    let mut mem = GuestMem::new(&mut ram, &code);
    chain.write_at(&mut mem, 0, b"12345678");
    assert_eq!(mem.code_pages_written, vec![0x1f]);
}

#[test]
fn push_used_records_code_pages_of_the_used_ring() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut code = vec![0u8; RAM_BYTES >> 12];
    code[(USED0 >> 12) as usize] = 1;
    let mut queue = setup_queue(&mut ram, DESC0, AVAIL0, USED0, 8);
    let mut mem = GuestMem::new(&mut ram, &code);
    queue.push_used(&mut mem, 0, 1);
    assert_eq!(mem.code_pages_written, vec![(USED0 >> 12) as usize]);
}
