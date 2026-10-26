use super::guestmem::GuestMem;
#[cfg(target_os = "uefi")]
use crate::prelude::*;

pub(crate) const QUEUE_MAX: u16 = 256;
pub(crate) const F_VERSION_1: u64 = 1 << 32;
const ST_DRIVER_OK: u8 = 4;
const ST_FEATURES_OK: u8 = 8;
const ST_FAILED: u8 = 0x80;

#[derive(Clone, Default)]
pub(crate) struct Queue {
    pub size: u16,
    pub enabled: bool,
    pub desc: u64,
    pub avail: u64,
    pub used: u64,
    pub last_avail: u16,
}

pub(crate) struct Seg {
    pub addr: u64,
    pub len: u32,
    pub write: bool,
}

pub(crate) struct Chain {
    pub head: u16,
    pub segs: Vec<Seg>,
}

impl Chain {
    pub fn read_all(&self, mem: &GuestMem, limit: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for s in self.segs.iter().filter(|s| !s.write) {
            let n = (s.len as usize).min(limit.saturating_sub(out.len()));
            let at = out.len();
            out.resize(at + n, 0);
            mem.read(s.addr, &mut out[at..]);
        }
        out
    }

    pub fn writable(&self) -> usize {
        self.segs
            .iter()
            .filter(|s| s.write)
            .map(|s| s.len as usize)
            .sum()
    }

    pub fn write_at(&self, mem: &mut GuestMem, mut off: usize, mut data: &[u8]) -> usize {
        let mut done = 0;
        for s in self.segs.iter().filter(|s| s.write) {
            let len = s.len as usize;
            if off >= len {
                off -= len;
                continue;
            }
            let n = (len - off).min(data.len());
            mem.write(s.addr + off as u64, &data[..n]);
            data = &data[n..];
            done += n;
            off = 0;
            if data.is_empty() {
                break;
            }
        }
        done
    }
}

impl Queue {
    pub fn ready(&self) -> bool {
        self.enabled && self.size > 0 && self.desc != 0 && self.avail != 0 && self.used != 0
    }

    pub fn pop(&mut self, mem: &GuestMem) -> Option<Chain> {
        if !self.ready() {
            return None;
        }
        let idx = mem.u16(self.avail + 2);
        if idx == self.last_avail {
            return None;
        }
        let slot = (self.last_avail % self.size) as u64;
        let head = mem.u16(self.avail + 4 + 2 * slot);
        self.last_avail = self.last_avail.wrapping_add(1);
        let mut segs = Vec::new();
        let mut cur = head;
        for _ in 0..self.size {
            if cur >= self.size {
                break;
            }
            let d = self.desc + 16 * cur as u64;
            let flags = mem.u16(d + 12);
            if flags & 4 != 0 {
                break;
            }
            segs.push(Seg {
                addr: mem.u64(d),
                len: mem.u32(d + 8),
                write: flags & 2 != 0,
            });
            if flags & 1 == 0 {
                break;
            }
            cur = mem.u16(d + 14);
        }
        Some(Chain { head, segs })
    }

    pub fn push_used(&mut self, mem: &mut GuestMem, head: u16, len: u32) {
        let idx = mem.u16(self.used + 2);
        let e = self.used + 4 + 8 * (idx % self.size) as u64;
        mem.put_u32(e, head as u32);
        mem.put_u32(e + 4, len);
        mem.put_u16(self.used + 2, idx.wrapping_add(1));
    }
}

pub(crate) trait VirtioDev: Send {
    fn id(&self) -> u16;
    fn features(&self) -> u64;
    fn num_queues(&self) -> usize;
    fn config_read(&self, off: usize) -> u8;
    fn config_write(&mut self, _off: usize, _v: u8) {}
    fn kick(&mut self, q: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool;
    fn poll(&mut self, _queues: &mut [Queue], _mem: &mut GuestMem) -> bool {
        false
    }
    fn config_event(&mut self) -> bool {
        false
    }
    fn reset(&mut self) {}
}

pub(crate) struct Transport {
    pub dev: Box<dyn VirtioDev>,
    pub queues: Vec<Queue>,
    pub status: u8,
    pub isr: u8,
    pub config_gen: u8,
    fsel: u32,
    dfsel: u32,
    driver_features: u64,
    qsel: u16,
}

impl Transport {
    pub fn new(dev: Box<dyn VirtioDev>) -> Self {
        let n = dev.num_queues();
        Transport {
            dev,
            queues: vec![
                Queue {
                    size: QUEUE_MAX,
                    ..Default::default()
                };
                n
            ],
            status: 0,
            isr: 0,
            config_gen: 0,
            fsel: 0,
            dfsel: 0,
            driver_features: 0,
            qsel: 0,
        }
    }

    pub fn driver_ok(&self) -> bool {
        self.status & ST_DRIVER_OK != 0 && self.status & ST_FAILED == 0
    }

    fn reset(&mut self) {
        for q in &mut self.queues {
            *q = Queue {
                size: QUEUE_MAX,
                ..Default::default()
            };
        }
        self.status = 0;
        self.isr = 0;
        self.fsel = 0;
        self.dfsel = 0;
        self.driver_features = 0;
        self.qsel = 0;
        self.dev.reset();
    }

    fn dev_features(&self) -> u64 {
        self.dev.features() | F_VERSION_1
    }

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
            raise |= self.set_isr(1);
        }
        if self.dev.config_event() {
            self.config_gen = self.config_gen.wrapping_add(1);
            raise |= self.set_isr(2);
        }
        raise
    }

    pub fn mmio_read(&mut self, off: u64, size: u8) -> u32 {
        match off >> 12 {
            0 => self.common_read(off & 0xfff, size),
            1 => {
                let v = self.isr;
                self.isr = 0;
                v as u32
            }
            2 => (0..size as usize).fold(0u32, |v, k| {
                v | (self.dev.config_read((off & 0xfff) as usize + k) as u32) << (8 * k)
            }),
            _ => 0,
        }
    }

    pub fn mmio_write(&mut self, off: u64, size: u8, val: u32, mem: &mut GuestMem) -> bool {
        match off >> 12 {
            0 => self.common_write(off & 0xfff, size, val),
            2 => {
                for k in 0..size as usize {
                    self.dev
                        .config_write((off & 0xfff) as usize + k, (val >> (8 * k)) as u8);
                }
            }
            3 => {
                let q = ((off & 0xfff) / 4) as usize;
                if q < self.queues.len()
                    && self.driver_ok()
                    && self.dev.kick(q, &mut self.queues, mem)
                {
                    return self.set_isr(1);
                }
            }
            _ => {}
        }
        false
    }

    fn cur(&mut self) -> Option<&mut Queue> {
        self.queues.get_mut(self.qsel as usize)
    }

    fn common_read(&mut self, off: u64, size: u8) -> u32 {
        let (base, v): (u64, u64) = match off {
            0x00..=0x03 => (0x00, self.dfsel as u64),
            // selectors 2 and 3 are the extended (bits 64-127) words newer kernels probe
            0x04..=0x07 => (
                0x04,
                match self.dfsel {
                    0 | 1 => (self.dev_features() >> (32 * self.dfsel)) as u32 as u64,
                    _ => 0,
                },
            ),
            0x08..=0x0b => (0x08, self.fsel as u64),
            0x0c..=0x0f => (
                0x0c,
                match self.fsel {
                    0 | 1 => (self.driver_features >> (32 * self.fsel)) as u32 as u64,
                    _ => 0,
                },
            ),
            0x10..=0x11 => (0x10, 0xffff),
            0x12..=0x13 => (0x12, self.queues.len() as u64),
            0x14 => (0x14, self.status as u64),
            0x15 => (0x15, self.config_gen as u64),
            0x16..=0x17 => (0x16, self.qsel as u64),
            0x18..=0x19 => (0x18, self.cur().map_or(0, |q| q.size) as u64),
            0x1a..=0x1b => (0x1a, 0xffff),
            0x1c..=0x1d => (0x1c, self.cur().map_or(0, |q| q.enabled as u64)),
            0x1e..=0x1f => (0x1e, self.qsel as u64),
            0x20..=0x27 => (0x20, self.cur().map_or(0, |q| q.desc)),
            0x28..=0x2f => (0x28, self.cur().map_or(0, |q| q.avail)),
            0x30..=0x37 => (0x30, self.cur().map_or(0, |q| q.used)),
            _ => return 0,
        };
        let sh = 8 * (off - base);
        let mask = if size >= 4 {
            u32::MAX
        } else {
            (1u32 << (8 * size)) - 1
        };
        ((v >> sh) as u32) & mask
    }

    fn common_write(&mut self, off: u64, size: u8, val: u32) {
        match (off, size) {
            (0x00, 4) => self.dfsel = val,
            (0x08, 4) => self.fsel = val,
            (0x0c, 4) if self.fsel < 2 => {
                let sh = 32 * self.fsel;
                self.driver_features =
                    (self.driver_features & !(0xffff_ffffu64 << sh)) | (val as u64) << sh;
            }
            (0x14, 1) => {
                let v = val as u8;
                if v == 0 {
                    self.reset();
                } else {
                    let mut st = v;
                    if st & ST_FEATURES_OK != 0 {
                        let want = self.driver_features;
                        if want & F_VERSION_1 == 0 || want & !self.dev_features() != 0 {
                            st &= !ST_FEATURES_OK;
                        }
                    }
                    self.status = st;
                }
            }
            (0x16, 2) => self.qsel = val as u16,
            (0x18, 2) => {
                let v = val as u16;
                if let Some(q) = self.cur()
                    && v.is_power_of_two()
                    && v <= QUEUE_MAX
                {
                    q.size = v;
                }
            }
            (0x1c, 2) => {
                if let Some(q) = self.cur() {
                    q.enabled = val & 1 != 0;
                    q.last_avail = 0;
                }
            }
            (0x20 | 0x28 | 0x30, 4) | (0x24 | 0x2c | 0x34, 4) => {
                let hi = off & 4 != 0;
                let which = off & !4;
                if let Some(q) = self.cur() {
                    let f = match which {
                        0x20 => &mut q.desc,
                        0x28 => &mut q.avail,
                        _ => &mut q.used,
                    };
                    *f = if hi {
                        (*f & 0xffff_ffff) | (val as u64) << 32
                    } else {
                        (*f & !0xffff_ffff) | val as u64
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
        let d = (desc + 16 * i as u64) as usize;
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

    #[test]
    fn extended_feature_selectors_do_not_alias() {
        let mut t = Transport::new(Box::new(VirtioBlk::new(Disk::memory(vec![0; 4096]))));
        let mut ram = vec![0u8; 4096];
        let mut mem = test_mem(&mut ram);
        let mut w = |t: &mut Transport, off, size, v| t.mmio_write(off, size, v, &mut mem);
        // Linux 6.18 negotiates 128 feature bits: it reads and writes selectors 0..=3.
        let mut offered = [0u32; 4];
        for sel in 0..4 {
            w(&mut t, 0x00, 4, sel);
            offered[sel as usize] = t.mmio_read(0x04, 4);
        }
        assert_eq!(offered[2..], [0, 0]);
        assert_ne!(offered[1] & 1, 0, "VERSION_1 offered");
        for sel in 0..4 {
            w(&mut t, 0x08, 4, sel);
            w(&mut t, 0x0c, 4, offered[sel as usize]);
        }
        w(&mut t, 0x14, 1, 1 | 2 | 8);
        assert_ne!(t.mmio_read(0x14, 1) & 8, 0, "FEATURES_OK must stick");
    }
}
