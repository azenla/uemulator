#[cfg(target_os = "uefi")]
use crate::prelude::*;
const HZ: u128 = 1_193_182;

#[derive(Default, Clone, Copy)]
struct Chan {
    reload: u32,
    mode: u8,
    access: u8,
    wr_hi: bool,
    rd_hi: bool,
    lo_byte: u8,
    latched: Option<u16>,
    latch_hi: bool,
    gate: bool,
    running: bool,
    start_ns: u64,
    start_count: u32,
    loaded: bool,
}

#[derive(Default)]
pub(crate) struct Pit {
    ch: [Chan; 3],
    next_irq: Option<u64>,
}

fn ticks(ns: u64) -> u64 {
    (ns as u128 * HZ / 1_000_000_000) as u64
}

fn ns_of_ticks(t: u64) -> u64 {
    ((t as u128 * 1_000_000_000).div_ceil(HZ)) as u64
}

impl Chan {
    fn count(&self, now: u64) -> u32 {
        if !self.running {
            return self.start_count;
        }
        let t = ticks(now.saturating_sub(self.start_ns));
        match self.mode & 7 {
            2 | 3 | 6 | 7 => {
                let r = self.reload as u64;
                (r - (t % r)) as u32
            }
            _ => (self.start_count as u64).wrapping_sub(t) as u32 & 0xffff,
        }
    }

    fn out(&self, now: u64) -> bool {
        if !self.loaded {
            return true;
        }
        let t = ticks(now.saturating_sub(self.start_ns));
        match self.mode & 7 {
            0 => self.running && t >= self.start_count as u64,
            2 | 6 => {
                let r = self.reload as u64;
                !(self.running && t % r == r - 1)
            }
            3 | 7 => {
                let r = self.reload as u64;
                !self.running || (t % r) < r.div_ceil(2)
            }
            _ => true,
        }
    }
}

impl Pit {
    pub fn new() -> Self {
        Pit::default()
    }

    pub fn set_gate2(&mut self, on: bool, now: u64) {
        let c = &mut self.ch[2];
        if c.gate == on {
            return;
        }
        c.gate = on;
        if !c.loaded {
            return;
        }
        if on {
            if matches!(c.mode & 7, 2 | 3 | 6 | 7) {
                c.start_count = c.reload;
            }
            c.start_ns = now;
            c.running = true;
        } else {
            c.start_count = c.count(now);
            c.running = false;
        }
    }

    pub fn out2(&self, now: u64) -> bool {
        self.ch[2].out(now)
    }

    fn load(&mut self, n: usize, now: u64) {
        let c = &mut self.ch[n];
        let v = if c.reload == 0 { 0x10000 } else { c.reload };
        c.reload = v;
        c.start_count = v;
        c.start_ns = now;
        c.loaded = true;
        c.running = n != 2 || c.gate;
        if n == 0 {
            self.next_irq = match c.mode & 7 {
                0 | 2 | 3 | 4 | 6 | 7 => Some(now + ns_of_ticks(v as u64)),
                _ => None,
            };
        }
    }

    pub fn write(&mut self, port: u16, v: u8, now: u64) {
        if port == 0x43 {
            let n = (v >> 6) as usize;
            if n == 3 {
                for k in 0..3 {
                    if v & (2 << k) != 0 {
                        let c = &mut self.ch[k];
                        if c.latched.is_none() {
                            c.latched = Some(c.count(now) as u16);
                            c.latch_hi = false;
                        }
                    }
                }
                return;
            }
            let acc = (v >> 4) & 3;
            let c = &mut self.ch[n];
            if acc == 0 {
                c.latched = Some(c.count(now) as u16);
                c.latch_hi = false;
                return;
            }
            c.access = acc;
            c.mode = (v >> 1) & 7;
            c.wr_hi = false;
            c.rd_hi = false;
            c.loaded = false;
            c.running = false;
            if n == 0 {
                self.next_irq = None;
            }
            return;
        }
        let n = (port - 0x40) as usize;
        let c = &mut self.ch[n];
        match c.access {
            1 => {
                c.reload = v as u32;
                self.load(n, now);
            }
            2 => {
                c.reload = (v as u32) << 8;
                self.load(n, now);
            }
            _ => {
                if !c.wr_hi {
                    c.lo_byte = v;
                    c.wr_hi = true;
                } else {
                    c.reload = c.lo_byte as u32 | (v as u32) << 8;
                    c.wr_hi = false;
                    self.load(n, now);
                }
            }
        }
    }

    pub fn read(&mut self, port: u16, now: u64) -> u8 {
        let n = (port - 0x40) as usize;
        let c = &mut self.ch[n];
        if let Some(l) = c.latched {
            let hi = c.access == 2 || (c.access == 3 && c.latch_hi);
            if c.access == 3 && !c.latch_hi {
                c.latch_hi = true;
                return l as u8;
            }
            c.latched = None;
            c.latch_hi = false;
            return if hi { (l >> 8) as u8 } else { l as u8 };
        }
        let cnt = c.count(now) as u16;
        match c.access {
            1 => cnt as u8,
            2 => (cnt >> 8) as u8,
            _ => {
                c.rd_hi = !c.rd_hi;
                if c.rd_hi { cnt as u8 } else { (cnt >> 8) as u8 }
            }
        }
    }

    pub fn state(&self, now: u64) -> String {
        let c = &self.ch[0];
        format!(
            "pit0 mode {} reload {} next_irq {:?} now {now}",
            c.mode, c.reload, self.next_irq
        )
    }

    pub fn next_deadline(&self) -> Option<u64> {
        self.next_irq
    }

    pub fn tick(&mut self, now: u64) -> bool {
        let Some(t) = self.next_irq else {
            return false;
        };
        if now < t {
            return false;
        }
        let c = &self.ch[0];
        let period = ns_of_ticks(c.reload as u64).max(1);
        self.next_irq = match c.mode & 7 {
            2 | 3 | 6 | 7 => {
                let k = (now - t) / period + 1;
                Some(t + k * period)
            }
            _ => None,
        };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ch2_mode0() {
        let mut p = Pit::new();
        p.set_gate2(true, 0);
        p.write(0x43, 0xb0, 0);
        p.write(0x42, 0x10, 0);
        p.write(0x42, 0x00, 0);
        assert!(!p.out2(1000));
        assert!(p.out2(ns_of_ticks(16) + 10));
    }

    #[test]
    fn ch0_periodic() {
        let mut p = Pit::new();
        p.write(0x43, 0x34, 0);
        p.write(0x40, 0xa9, 0);
        p.write(0x40, 0x04, 0);
        let d = p.next_deadline().unwrap();
        assert!(d > 0);
        assert!(p.tick(d));
        assert!(p.next_deadline().unwrap() > d);
    }

    #[test]
    fn ch0_mode4_oneshot_fires() {
        let mut p = Pit::new();
        p.write(0x43, 0x38, 0);
        p.write(0x40, 0x10, 0);
        p.write(0x40, 0x00, 0);
        let d = p.next_deadline().expect("mode 4 must schedule an irq");
        assert!(p.tick(d));
        assert!(p.next_deadline().is_none());
    }

    #[test]
    fn lohi_read_toggles() {
        let mut p = Pit::new();
        p.set_gate2(true, 0);
        p.write(0x43, 0xb0, 0);
        p.write(0x42, 0xff, 0);
        p.write(0x42, 0xff, 0);
        assert_eq!(p.read(0x42, 10), 0xff);
        assert_eq!(p.read(0x42, 10), 0xff);
    }
}
