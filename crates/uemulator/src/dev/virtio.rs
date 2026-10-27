use super::guestmem::GuestMem;
#[cfg(target_os = "uefi")]
use crate::prelude::*;

/// Largest queue size offered. Must be a power of two: `push_used` and `pop` index the
/// rings with `index % size` while the 16-bit indices wrap at 65536.
pub(crate) const QUEUE_MAX: u16 = 256;

/// Virtio device types (`VIRTIO_ID_*`); the PCI device id is `0x1040 + type`.
pub(crate) const DEVICE_TYPE_NET: u16 = 1;
pub(crate) const DEVICE_TYPE_BLOCK: u16 = 2;
pub(crate) const DEVICE_TYPE_CONSOLE: u16 = 3;

/// Feature bit 32. We only implement modern (virtio 1.x) split queues, so the driver must accept it.
const VIRTIO_F_VERSION_1: u64 = 1 << 32;

/// Device status bits (virtio "device status field"). ACKNOWLEDGE (1) and DRIVER (2) are not
/// checked, the driver sets them and we just store them.
const STATUS_DRIVER_OK: u8 = 4;
const STATUS_FEATURES_OK: u8 = 8;
const STATUS_FAILED: u8 = 0x80;

/// ISR status bits; reading the ISR register returns and clears them.
const ISR_QUEUE_INTERRUPT: u8 = 1;
const ISR_CONFIG_CHANGE: u8 = 2;

/// BAR0 layout, one 4 KiB region per capability. `pci.rs` advertises these offsets in the
/// virtio vendor capabilities, so the two files must agree.
pub(crate) const COMMON_CFG_OFFSET: u64 = 0x0000;
pub(crate) const ISR_OFFSET: u64 = 0x1000;
pub(crate) const DEVICE_CFG_OFFSET: u64 = 0x2000;
pub(crate) const NOTIFY_OFFSET: u64 = 0x3000;
pub(crate) const REGION_SIZE: u64 = 0x1000;
/// Notify address of queue N is `NOTIFY_OFFSET + N * NOTIFY_OFF_MULTIPLIER`; the multiplier is
/// advertised in the notify capability and `queue_notify_off` reads back as the queue index.
pub(crate) const NOTIFY_OFF_MULTIPLIER: u32 = 4;

/// Common configuration structure field offsets (`struct virtio_pci_common_cfg`).
const COMMON_DEVICE_FEATURE_SELECT: u64 = 0x00;
const COMMON_DEVICE_FEATURE: u64 = 0x04;
const COMMON_DRIVER_FEATURE_SELECT: u64 = 0x08;
const COMMON_DRIVER_FEATURE: u64 = 0x0c;
const COMMON_MSIX_CONFIG: u64 = 0x10;
const COMMON_NUM_QUEUES: u64 = 0x12;
const COMMON_DEVICE_STATUS: u64 = 0x14;
const COMMON_CONFIG_GENERATION: u64 = 0x15;
const COMMON_QUEUE_SELECT: u64 = 0x16;
const COMMON_QUEUE_SIZE: u64 = 0x18;
const COMMON_QUEUE_MSIX_VECTOR: u64 = 0x1a;
const COMMON_QUEUE_ENABLE: u64 = 0x1c;
const COMMON_QUEUE_NOTIFY_OFF: u64 = 0x1e;
const COMMON_QUEUE_DESC: u64 = 0x20;
const COMMON_QUEUE_DRIVER: u64 = 0x28;
const COMMON_QUEUE_DEVICE: u64 = 0x30;
/// The 64-bit queue address registers are accessed as two 32-bit halves; the high one is at +4.
const QUEUE_ADDR_HIGH_HALF: u64 = 4;

/// (offset, width in bytes) of every readable common config field, used to find which field a
/// read of any size/alignment falls into.
const COMMON_FIELDS: [(u64, u64); 16] = [
    (COMMON_DEVICE_FEATURE_SELECT, 4),
    (COMMON_DEVICE_FEATURE, 4),
    (COMMON_DRIVER_FEATURE_SELECT, 4),
    (COMMON_DRIVER_FEATURE, 4),
    (COMMON_MSIX_CONFIG, 2),
    (COMMON_NUM_QUEUES, 2),
    (COMMON_DEVICE_STATUS, 1),
    (COMMON_CONFIG_GENERATION, 1),
    (COMMON_QUEUE_SELECT, 2),
    (COMMON_QUEUE_SIZE, 2),
    (COMMON_QUEUE_MSIX_VECTOR, 2),
    (COMMON_QUEUE_ENABLE, 2),
    (COMMON_QUEUE_NOTIFY_OFF, 2),
    (COMMON_QUEUE_DESC, 8),
    (COMMON_QUEUE_DRIVER, 8),
    (COMMON_QUEUE_DEVICE, 8),
];

/// Read back for msix_config and queue_msix_vector (`VIRTIO_MSI_NO_VECTOR`): MSI-X is not supported.
const MSI_NO_VECTOR: u64 = 0xffff;

/// Split virtqueue layout (descriptor table, available ring, used ring). Descriptor is 16 bytes:
/// addr u64, len u32, flags u16, next u16.
const DESC_SIZE: u64 = 16;
const DESC_ADDR: u64 = 0;
const DESC_LEN: u64 = 8;
const DESC_FLAGS: u64 = 12;
const DESC_NEXT: u64 = 14;
pub(crate) const DESC_F_NEXT: u16 = 1;
/// Buffer is device-writable (otherwise device-readable).
pub(crate) const DESC_F_WRITE: u16 = 2;
const DESC_F_INDIRECT: u16 = 4;
/// Available ring: flags u16, idx u16, ring[size] of u16 head indices. Used ring: flags u16,
/// idx u16, ring[size] of {id u32, len u32}. The ring flags are ignored (no interrupt
/// suppression), and event idx is not negotiated.
const AVAIL_IDX: u64 = 2;
const AVAIL_RING: u64 = 4;
const AVAIL_ENTRY_SIZE: u64 = 2;
const USED_IDX: u64 = 2;
const USED_RING: u64 = 4;
const USED_ENTRY_SIZE: u64 = 8;
const USED_ENTRY_LEN: u64 = 4;

/// Split-virtqueue state the driver programs through the common config registers.
#[derive(Clone, Default)]
pub(crate) struct Queue {
    /// Starts at `QUEUE_MAX`; the driver may only shrink it to another power of two.
    pub size: u16,
    pub enabled: bool,
    /// Guest physical addresses of the descriptor table, available ring and used ring.
    pub desc: u64,
    pub avail: u64,
    pub used: u64,
    /// Free-running count of available entries consumed; compared against the ring's `idx`.
    pub last_avail: u16,
}

pub(crate) struct Segment {
    pub addr: u64,
    pub len: u32,
    pub device_writable: bool,
}

/// One descriptor chain popped from the available ring.
pub(crate) struct Chain {
    /// Index of the first descriptor; this is the id reported back in the used ring.
    pub head: u16,
    pub segs: Vec<Segment>,
}

impl Chain {
    /// Concatenates the device-readable segments, truncated to `limit` bytes.
    pub fn read_all(&self, mem: &GuestMem, limit: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for seg in self.segs.iter().filter(|seg| !seg.device_writable) {
            let n = (seg.len as usize).min(limit.saturating_sub(out.len()));
            let at = out.len();
            out.resize(at + n, 0);
            mem.read(seg.addr, &mut out[at..]);
        }
        out
    }

    /// Total bytes in device-writable segments.
    pub fn writable(&self) -> usize {
        self.segs
            .iter()
            .filter(|seg| seg.device_writable)
            .map(|seg| seg.len as usize)
            .sum()
    }

    /// Writes `data` into the device-writable segments starting `offset` bytes in, spilling
    /// across segments. Returns the number of bytes that fit.
    pub fn write_at(&self, mem: &mut GuestMem, mut offset: usize, mut data: &[u8]) -> usize {
        let mut done = 0;
        for seg in self.segs.iter().filter(|seg| seg.device_writable) {
            let len = seg.len as usize;
            if offset >= len {
                offset -= len;
                continue;
            }
            let n = (len - offset).min(data.len());
            mem.write(seg.addr + offset as u64, &data[..n]);
            data = &data[n..];
            done += n;
            offset = 0;
            if data.is_empty() {
                break;
            }
        }
        done
    }
}

impl Queue {
    /// False until the driver has enabled the queue and programmed all three ring addresses,
    /// so a stray kick on a half-configured queue does nothing.
    pub fn ready(&self) -> bool {
        self.enabled && self.size > 0 && self.desc != 0 && self.avail != 0 && self.used != 0
    }

    /// Takes the next available chain. A malformed chain (index out of range, indirect
    /// descriptor, loop) is cut short rather than rejected: the device sees the segments
    /// gathered so far.
    pub fn pop(&mut self, mem: &GuestMem) -> Option<Chain> {
        if !self.ready() {
            return None;
        }
        let avail_idx = mem.read_u16(self.avail + AVAIL_IDX);
        if avail_idx == self.last_avail {
            return None;
        }
        let slot = (self.last_avail % self.size) as u64;
        let head = mem.read_u16(self.avail + AVAIL_RING + AVAIL_ENTRY_SIZE * slot);
        self.last_avail = self.last_avail.wrapping_add(1);
        let mut segs = Vec::new();
        let mut current = head;
        // At most `size` descriptors in a valid chain; also stops descriptor loops.
        for _ in 0..self.size {
            if current >= self.size {
                break;
            }
            let desc = self.desc + DESC_SIZE * current as u64;
            let flags = mem.read_u16(desc + DESC_FLAGS);
            // VIRTIO_RING_F_INDIRECT_DESC is not offered.
            if flags & DESC_F_INDIRECT != 0 {
                break;
            }
            segs.push(Segment {
                addr: mem.read_u64(desc + DESC_ADDR),
                len: mem.read_u32(desc + DESC_LEN),
                device_writable: flags & DESC_F_WRITE != 0,
            });
            if flags & DESC_F_NEXT == 0 {
                break;
            }
            current = mem.read_u16(desc + DESC_NEXT);
        }
        Some(Chain { head, segs })
    }

    /// Completes the chain headed at `head`; `len` is the number of bytes the device wrote
    /// into its writable segments. The used `idx` is published last, after the entry.
    pub fn push_used(&mut self, mem: &mut GuestMem, head: u16, len: u32) {
        let used_idx = mem.read_u16(self.used + USED_IDX);
        let entry = self.used + USED_RING + USED_ENTRY_SIZE * (used_idx % self.size) as u64;
        mem.write_u32(entry, head as u32);
        mem.write_u32(entry + USED_ENTRY_LEN, len);
        mem.write_u16(self.used + USED_IDX, used_idx.wrapping_add(1));
    }
}

/// A virtio device model behind a PCI `Transport`. The callbacks return true when buffers
/// were completed (or config changed) and the guest should get an interrupt.
pub(crate) trait VirtioDev: Send {
    /// Virtio device type (1 net, 2 block, 3 console). PCI device id is `0x1040 + id`.
    fn device_type(&self) -> u16;
    /// Device-specific feature bits; the transport adds `VIRTIO_F_VERSION_1`.
    fn features(&self) -> u64;
    fn num_queues(&self) -> usize;
    /// Byte `off` of the device-specific config space; out-of-range reads should return 0.
    fn config_read(&self, off: usize) -> u8;
    fn config_write(&mut self, _off: usize, _v: u8) {}
    /// Driver wrote to the notify register of queue `q`.
    fn kick(&mut self, q: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool;
    /// Called every device tick (not only on kicks) for host-initiated work such as receive data.
    fn poll(&mut self, _queues: &mut [Queue], _mem: &mut GuestMem) -> bool {
        false
    }
    /// True once per device config change; the transport bumps `config_gen` and raises the ISR bit.
    fn config_event(&mut self) -> bool {
        false
    }
    fn reset(&mut self) {}
}

/// virtio-pci modern transport for one device: decodes accesses to BAR0 and owns the common
/// config state (feature negotiation, status, queue setup).
pub(crate) struct Transport {
    pub dev: Box<dyn VirtioDev>,
    pub queues: Vec<Queue>,
    pub status: u8,
    pub isr: u8,
    /// Bumped on every config change so a driver reading config in pieces can detect tearing.
    pub config_gen: u8,
    driver_feature_select: u32,
    device_feature_select: u32,
    driver_features: u64,
    queue_select: u16,
}

impl Transport {
    pub fn new(dev: Box<dyn VirtioDev>) -> Self {
        let queue_count = dev.num_queues();
        Transport {
            dev,
            queues: vec![
                Queue {
                    size: QUEUE_MAX,
                    ..Default::default()
                };
                queue_count
            ],
            status: 0,
            isr: 0,
            config_gen: 0,
            driver_feature_select: 0,
            device_feature_select: 0,
            driver_features: 0,
            queue_select: 0,
        }
    }

    /// Devices do no work (no kicks, no polling, no interrupts) until the driver reaches DRIVER_OK.
    pub fn driver_ok(&self) -> bool {
        self.status & STATUS_DRIVER_OK != 0 && self.status & STATUS_FAILED == 0
    }

    /// Device reset, triggered by writing 0 to the status register. `config_gen` survives.
    fn reset(&mut self) {
        for queue in &mut self.queues {
            *queue = Queue {
                size: QUEUE_MAX,
                ..Default::default()
            };
        }
        self.status = 0;
        self.isr = 0;
        self.driver_feature_select = 0;
        self.device_feature_select = 0;
        self.driver_features = 0;
        self.queue_select = 0;
        self.dev.reset();
    }

    fn dev_features(&self) -> u64 {
        self.dev.features() | VIRTIO_F_VERSION_1
    }

    /// Sets ISR bits and returns true only on the 0 -> nonzero transition. The PIC line is
    /// edge triggered, so the caller must raise the IRQ only on that edge; a read of the ISR
    /// register clears it and re-arms the next edge.
    fn set_isr(&mut self, bit: u8) -> bool {
        let edge = self.isr == 0;
        self.isr |= bit;
        edge
    }

    pub fn poll(&mut self, mem: &mut GuestMem) -> bool {
        if !self.driver_ok() {
            return false;
        }
        let mut raise = false;
        if self.dev.poll(&mut self.queues, mem) {
            raise |= self.set_isr(ISR_QUEUE_INTERRUPT);
        }
        if self.dev.config_event() {
            self.config_gen = self.config_gen.wrapping_add(1);
            raise |= self.set_isr(ISR_CONFIG_CHANGE);
        }
        raise
    }

    /// `off` is relative to BAR0.
    pub fn mmio_read(&mut self, off: u64, size: u8) -> u32 {
        let in_region = off & (REGION_SIZE - 1);
        match off & !(REGION_SIZE - 1) {
            COMMON_CFG_OFFSET => self.common_read(in_region, size),
            ISR_OFFSET => {
                let v = self.isr;
                self.isr = 0;
                v as u32
            }
            DEVICE_CFG_OFFSET => (0..size as usize).fold(0u32, |v, k| {
                v | (self.dev.config_read(in_region as usize + k) as u32) << (8 * k)
            }),
            _ => 0,
        }
    }

    /// Returns true if the guest should get an interrupt as a result of the write.
    pub fn mmio_write(&mut self, off: u64, size: u8, val: u32, mem: &mut GuestMem) -> bool {
        let in_region = off & (REGION_SIZE - 1);
        match off & !(REGION_SIZE - 1) {
            COMMON_CFG_OFFSET => self.common_write(in_region, size, val),
            DEVICE_CFG_OFFSET => {
                for k in 0..size as usize {
                    self.dev
                        .config_write(in_region as usize + k, (val >> (8 * k)) as u8);
                }
            }
            NOTIFY_OFFSET => {
                // The queue is identified by the address, the written value is ignored.
                let queue = (in_region / NOTIFY_OFF_MULTIPLIER as u64) as usize;
                if queue < self.queues.len()
                    && self.driver_ok()
                    && self.dev.kick(queue, &mut self.queues, mem)
                {
                    return self.set_isr(ISR_QUEUE_INTERRUPT);
                }
            }
            _ => {}
        }
        false
    }

    /// Queue picked by `queue_select`, if it exists.
    fn selected_queue(&mut self) -> Option<&mut Queue> {
        self.queues.get_mut(self.queue_select as usize)
    }

    fn common_read(&mut self, off: u64, size: u8) -> u32 {
        let Some((base, _)) = COMMON_FIELDS
            .iter()
            .copied()
            .find(|&(start, width)| off >= start && off < start + width)
        else {
            return 0;
        };
        let value: u64 = match base {
            COMMON_DEVICE_FEATURE_SELECT => self.device_feature_select as u64,
            // Selectors 2 and 3 are the extended (bits 64-127) words newer kernels probe; they
            // must read as zero, not alias onto 0/1 (aliasing cleared VERSION_1 in the driver).
            COMMON_DEVICE_FEATURE => match self.device_feature_select {
                0 | 1 => (self.dev_features() >> (32 * self.device_feature_select)) as u32 as u64,
                _ => 0,
            },
            COMMON_DRIVER_FEATURE_SELECT => self.driver_feature_select as u64,
            COMMON_DRIVER_FEATURE => match self.driver_feature_select {
                0 | 1 => (self.driver_features >> (32 * self.driver_feature_select)) as u32 as u64,
                _ => 0,
            },
            COMMON_MSIX_CONFIG | COMMON_QUEUE_MSIX_VECTOR => MSI_NO_VECTOR,
            COMMON_NUM_QUEUES => self.queues.len() as u64,
            COMMON_DEVICE_STATUS => self.status as u64,
            COMMON_CONFIG_GENERATION => self.config_gen as u64,
            COMMON_QUEUE_SELECT => self.queue_select as u64,
            // Size 0 for a nonexistent queue; Linux treats that as "no such queue".
            COMMON_QUEUE_SIZE => self.selected_queue().map_or(0, |q| q.size) as u64,
            COMMON_QUEUE_ENABLE => self.selected_queue().map_or(0, |q| q.enabled as u64),
            COMMON_QUEUE_NOTIFY_OFF => self.queue_select as u64,
            COMMON_QUEUE_DESC => self.selected_queue().map_or(0, |q| q.desc),
            COMMON_QUEUE_DRIVER => self.selected_queue().map_or(0, |q| q.avail),
            COMMON_QUEUE_DEVICE => self.selected_queue().map_or(0, |q| q.used),
            _ => return 0,
        };
        let byte_shift = 8 * (off - base);
        let mask = if size >= 4 {
            u32::MAX
        } else {
            (1u32 << (8 * size)) - 1
        };
        ((value >> byte_shift) as u32) & mask
    }

    /// Writes must match a register's exact offset and width; anything else is ignored.
    fn common_write(&mut self, off: u64, size: u8, val: u32) {
        match (off, size) {
            (COMMON_DEVICE_FEATURE_SELECT, 4) => self.device_feature_select = val,
            (COMMON_DRIVER_FEATURE_SELECT, 4) => self.driver_feature_select = val,
            (COMMON_DRIVER_FEATURE, 4) if self.driver_feature_select < 2 => {
                let shift = 32 * self.driver_feature_select;
                self.driver_features =
                    (self.driver_features & !(0xffff_ffffu64 << shift)) | (val as u64) << shift;
            }
            (COMMON_DEVICE_STATUS, 1) => {
                let status = val as u8;
                if status == 0 {
                    self.reset();
                } else {
                    let mut accepted = status;
                    if accepted & STATUS_FEATURES_OK != 0 {
                        // Refuse by not latching FEATURES_OK; the driver re-reads the status to find out.
                        let wanted = self.driver_features;
                        if wanted & VIRTIO_F_VERSION_1 == 0 || wanted & !self.dev_features() != 0 {
                            accepted &= !STATUS_FEATURES_OK;
                        }
                    }
                    self.status = accepted;
                }
            }
            (COMMON_QUEUE_SELECT, 2) => self.queue_select = val as u16,
            (COMMON_QUEUE_SIZE, 2) => {
                let size = val as u16;
                if let Some(queue) = self.selected_queue()
                    && size.is_power_of_two()
                    && size <= QUEUE_MAX
                {
                    queue.size = size;
                }
            }
            (COMMON_QUEUE_ENABLE, 2) => {
                if let Some(queue) = self.selected_queue() {
                    queue.enabled = val & 1 != 0;
                    queue.last_avail = 0;
                }
            }
            (_, 4)
                if matches!(
                    off & !QUEUE_ADDR_HIGH_HALF,
                    COMMON_QUEUE_DESC | COMMON_QUEUE_DRIVER | COMMON_QUEUE_DEVICE
                ) =>
            {
                let high_half = off & QUEUE_ADDR_HIGH_HALF != 0;
                let which = off & !QUEUE_ADDR_HIGH_HALF;
                if let Some(queue) = self.selected_queue() {
                    let addr = match which {
                        COMMON_QUEUE_DESC => &mut queue.desc,
                        COMMON_QUEUE_DRIVER => &mut queue.avail,
                        _ => &mut queue.used,
                    };
                    *addr = if high_half {
                        (*addr & 0xffff_ffff) | (val as u64) << 32
                    } else {
                        (*addr & !0xffff_ffff) | val as u64
                    };
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;

    pub fn setup_queue(ram: &mut [u8], desc: u64, avail: u64, used: u64, size: u16) -> Queue {
        let _ = ram;
        Queue {
            size,
            enabled: true,
            desc,
            avail,
            used,
            last_avail: 0,
        }
    }

    pub fn put_desc(ram: &mut [u8], desc: u64, i: u16, addr: u64, len: u32, flags: u16, next: u16) {
        let d = (desc + DESC_SIZE * i as u64) as usize;
        ram[d..d + 8].copy_from_slice(&addr.to_le_bytes());
        ram[d + 8..d + 12].copy_from_slice(&len.to_le_bytes());
        ram[d + 12..d + 14].copy_from_slice(&flags.to_le_bytes());
        ram[d + 14..d + 16].copy_from_slice(&next.to_le_bytes());
    }

    pub fn post(ram: &mut [u8], avail: u64, size: u16, head: u16) {
        let a = avail as usize;
        let idx = u16::from_le_bytes([ram[a + 2], ram[a + 3]]);
        let slot = (idx % size) as usize;
        ram[a + 4 + 2 * slot..a + 6 + 2 * slot].copy_from_slice(&head.to_le_bytes());
        ram[a + 2..a + 4].copy_from_slice(&idx.wrapping_add(1).to_le_bytes());
    }

    pub fn used_idx(ram: &[u8], used: u64) -> u16 {
        let u = used as usize;
        u16::from_le_bytes([ram[u + 2], ram[u + 3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::guestmem::test_mem;
    use crate::dev::vblk::VirtioBlk;
    use crate::disk::Disk;

    const STATUS_ACKNOWLEDGE: u32 = 1;
    const STATUS_DRIVER: u32 = 2;

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
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK as u32,
        );
        assert_ne!(
            t.mmio_read(COMMON_DEVICE_STATUS, 1) & STATUS_FEATURES_OK as u32,
            0,
            "FEATURES_OK must stick"
        );
    }
}
