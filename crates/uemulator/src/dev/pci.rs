use super::guestmem::GuestMem;
use super::virtio::Transport;
#[cfg(target_os = "uefi")]
use crate::prelude::*;

pub(crate) const MMIO_BASE: u64 = 0xc000_0000;
const BAR_SIZE: u64 = 0x4000;
const IRQS: [u8; 3] = [9, 10, 11];

struct Function {
    t: Transport,
    devfn: u8,
    bar0: u32,
    command: u16,
    int_line: u8,
}

pub(crate) struct PciHost {
    cfg_addr: u32,
    funcs: Vec<Function>,
}

fn cap(buf: &mut [u8; 256], at: usize, next: usize, len: u8, ty: u8, off: u32, size: u32) {
    buf[at] = 0x09;
    buf[at + 1] = next as u8;
    buf[at + 2] = len;
    buf[at + 3] = ty;
    buf[at + 4] = 0;
    buf[at + 8..at + 12].copy_from_slice(&off.to_le_bytes());
    buf[at + 12..at + 16].copy_from_slice(&size.to_le_bytes());
}

impl Function {
    fn config(&self) -> [u8; 256] {
        let mut b = [0u8; 256];
        let id = self.t.dev.id();
        b[0..2].copy_from_slice(&0x1af4u16.to_le_bytes());
        b[2..4].copy_from_slice(&(0x1040 + id).to_le_bytes());
        b[4..6].copy_from_slice(&self.command.to_le_bytes());
        b[6..8].copy_from_slice(&0x0010u16.to_le_bytes());
        b[8] = 1;
        let (sub, class) = match id {
            1 => (0x00, 0x02),
            2 => (0x00, 0x01),
            _ => (0x80, 0x07),
        };
        b[10] = sub;
        b[11] = class;
        b[0x10..0x14].copy_from_slice(&self.bar0.to_le_bytes());
        b[0x2c..0x2e].copy_from_slice(&0x1af4u16.to_le_bytes());
        b[0x2e..0x30].copy_from_slice(&id.to_le_bytes());
        b[0x34] = 0x40;
        b[0x3c] = self.int_line;
        b[0x3d] = 1;
        cap(&mut b, 0x40, 0x50, 16, 1, 0, 0x38);
        cap(&mut b, 0x50, 0x64, 20, 2, 0x3000, 0x1000);
        b[0x50 + 16..0x50 + 20].copy_from_slice(&4u32.to_le_bytes());
        cap(&mut b, 0x64, 0x74, 16, 3, 0x1000, 4);
        cap(&mut b, 0x74, 0, 16, 4, 0x2000, 0x100);
        b
    }

    fn write_config(&mut self, reg: usize, size: usize, val: u32) {
        for k in 0..size {
            let r = reg + k;
            let byte = (val >> (8 * k)) as u8;
            match r {
                0x04 => self.command = (self.command & 0xff00) | byte as u16,
                0x05 => self.command = (self.command & 0x00ff) | ((byte as u16) << 8),
                0x10..=0x13 => {
                    let sh = 8 * (r - 0x10);
                    let mask = !((BAR_SIZE - 1) as u32);
                    self.bar0 = (self.bar0 & !(0xff << sh)) | ((byte as u32) << sh);
                    self.bar0 &= mask;
                }
                0x3c => self.int_line = byte,
                _ => {}
            }
        }
    }

    fn mem_enabled(&self) -> bool {
        self.command & 2 != 0
    }

    fn hit(&self, pa: u64) -> Option<u64> {
        let base = self.bar0 as u64 & !(BAR_SIZE - 1);
        (self.mem_enabled() && base != 0 && pa >= base && pa < base + BAR_SIZE).then(|| pa - base)
    }

    fn irq_ok(&self) -> bool {
        self.command & 0x400 == 0
    }
}

impl PciHost {
    pub fn new(devs: Vec<Transport>) -> Self {
        let funcs = devs
            .into_iter()
            .enumerate()
            .map(|(i, t)| Function {
                t,
                devfn: ((i as u8 + 1) << 3),
                bar0: (MMIO_BASE + i as u64 * BAR_SIZE) as u32,
                command: 0,
                int_line: IRQS[i % IRQS.len()],
            })
            .collect();
        PciHost { cfg_addr: 0, funcs }
    }

    pub fn is_empty(&self) -> bool {
        self.funcs.is_empty()
    }

    pub fn console_active(&self) -> bool {
        self.funcs
            .iter()
            .any(|f| f.t.dev.id() == 3 && f.t.driver_ok())
    }

    fn host_bridge() -> [u8; 256] {
        let mut b = [0u8; 256];
        b[0..2].copy_from_slice(&0x8086u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x1237u16.to_le_bytes());
        b[10] = 0;
        b[11] = 6;
        b
    }

    fn config_read(&self, addr: u32, size: usize, byte: usize) -> u32 {
        let (bus, devfn, reg) = (
            (addr >> 16) & 0xff,
            ((addr >> 8) & 0xff) as u8,
            (addr & 0xfc) as usize,
        );
        let ones = u32::MAX >> (32 - 8 * size as u32);
        if bus != 0 {
            return ones;
        }
        let cfg = if devfn == 0 {
            Self::host_bridge()
        } else if let Some(f) = self.funcs.iter().find(|f| f.devfn == devfn) {
            f.config()
        } else {
            return ones;
        };
        let d = u32::from_le_bytes(cfg[reg..reg + 4].try_into().unwrap());
        (d >> (8 * byte)) & ones
    }

    pub fn io_in(&self, port: u16, size: u8) -> u32 {
        if port == 0xcf8 {
            return self.cfg_addr;
        }
        if self.cfg_addr & 0x8000_0000 == 0 {
            return u32::MAX >> (32 - 8 * size as u32);
        }
        self.config_read(self.cfg_addr, size as usize, (port & 3) as usize)
    }

    pub fn io_out(&mut self, port: u16, size: u8, val: u32) {
        if port == 0xcf8 {
            self.cfg_addr = val;
            return;
        }
        if self.cfg_addr & 0x8000_0000 == 0 || (self.cfg_addr >> 16) & 0xff != 0 {
            return;
        }
        let devfn = ((self.cfg_addr >> 8) & 0xff) as u8;
        let reg = (self.cfg_addr & 0xfc) as usize + (port & 3) as usize;
        if let Some(f) = self.funcs.iter_mut().find(|f| f.devfn == devfn) {
            f.write_config(reg, size as usize, val);
        }
    }

    fn find(&mut self, pa: u64) -> Option<(&mut Function, u64)> {
        self.funcs
            .iter_mut()
            .find_map(|f| f.hit(pa).map(|o| (f, o)))
    }

    pub fn mmio_read(&mut self, pa: u64, size: u8) -> Option<u32> {
        self.find(pa).map(|(f, off)| f.t.mmio_read(off, size))
    }

    pub fn mmio_write(&mut self, pa: u64, size: u8, val: u32, mem: &mut GuestMem) -> Option<u8> {
        let (f, off) = self.find(pa)?;
        let raise = f.t.mmio_write(off, size, val, mem);
        Some(if raise && f.irq_ok() { f.int_line } else { 0 })
    }

    pub fn poll(&mut self, mem: &mut GuestMem, raise: &mut Vec<u8>) {
        for f in &mut self.funcs {
            if f.t.poll(mem) && f.irq_ok() {
                raise.push(f.int_line);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::ConsoleHandle;
    use crate::dev::vcons::VirtioConsole;

    fn host() -> PciHost {
        PciHost::new(vec![Transport::new(Box::new(VirtioConsole::new(
            ConsoleHandle::default(),
        )))])
    }

    fn rd(h: &mut PciHost, dev: u32, reg: u32) -> u32 {
        h.io_out(0xcf8, 4, 0x8000_0000 | dev << 11 | reg);
        h.io_in(0xcfc, 4)
    }

    #[test]
    fn enumerate_and_size_bar() {
        let mut h = host();
        assert_eq!(rd(&mut h, 0, 0) & 0xffff, 0x8086);
        assert_eq!(rd(&mut h, 0, 8) >> 16, 0x0600);
        assert_eq!(rd(&mut h, 1, 0), 0x1af4 | 0x1043 << 16);
        assert_eq!(rd(&mut h, 2, 0), u32::MAX);
        let bar = rd(&mut h, 1, 0x10);
        h.io_out(0xcf8, 4, 0x8000_0000 | 1 << 11 | 0x10);
        h.io_out(0xcfc, 4, u32::MAX);
        assert_eq!(h.io_in(0xcfc, 4), !(BAR_SIZE as u32 - 1));
        h.io_out(0xcfc, 4, bar);
        assert_eq!(h.io_in(0xcfc, 4), bar);
        let mut caps = Vec::new();
        let mut p = rd(&mut h, 1, 0x34) & 0xff;
        while p != 0 {
            let w = rd(&mut h, 1, p);
            caps.push((w >> 24) as u8);
            p = (w >> 8) & 0xff;
        }
        assert_eq!(caps, vec![1, 2, 3, 4]);
    }

    #[test]
    fn bar_decode_follows_command() {
        let mut h = host();
        assert!(h.mmio_read(MMIO_BASE + 0x14, 1).is_none());
        h.io_out(0xcf8, 4, 0x8000_0000 | 1 << 11 | 4);
        h.io_out(0xcfc, 2, 2);
        assert_eq!(h.mmio_read(MMIO_BASE + 0x12, 2), Some(2));
    }
}
