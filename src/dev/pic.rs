#[derive(Default)]
struct Pic {
    irr: u8,
    isr: u8,
    imr: u8,
    base: u8,
    step: u8,
    icw4: bool,
    read_isr: bool,
}

#[derive(Default)]
pub(crate) struct Pics {
    m: Pic,
    s: Pic,
    elcr: [u8; 2],
}

fn candidate(p: &Pic, extra: u8) -> Option<u8> {
    let irr = (p.irr | extra) & !p.imr;
    for n in 0..8 {
        if p.isr & (1 << n) != 0 {
            return None;
        }
        if irr & (1 << n) != 0 {
            return Some(n);
        }
    }
    None
}

impl Pic {
    fn command(&mut self, v: u8) {
        if v & 0x10 != 0 {
            self.irr = 0;
            self.isr = 0;
            self.imr = 0;
            self.icw4 = v & 1 != 0;
            self.step = 1;
            self.read_isr = false;
        } else if v & 0x08 != 0 {
            if v & 2 != 0 {
                self.read_isr = v & 1 != 0;
            }
        } else {
            match v >> 5 {
                1 | 5 => {
                    if self.isr != 0 {
                        self.isr &= !(1 << self.isr.trailing_zeros());
                    }
                }
                3 | 7 => self.isr &= !(1 << (v & 7)),
                _ => {}
            }
        }
    }

    fn data(&mut self, v: u8) {
        match self.step {
            1 => {
                self.base = v & 0xf8;
                self.step = 2;
            }
            2 => self.step = if self.icw4 { 3 } else { 0 },
            3 => self.step = 0,
            _ => self.imr = v,
        }
    }
}

impl Pics {
    pub fn state(&self) -> String {
        format!(
            "m(irr {:02x} isr {:02x} imr {:02x} base {:02x}) s(irr {:02x} isr {:02x} imr {:02x} base {:02x})",
            self.m.irr,
            self.m.isr,
            self.m.imr,
            self.m.base,
            self.s.irr,
            self.s.isr,
            self.s.imr,
            self.s.base
        )
    }

    pub fn raise(&mut self, irq: u8) {
        if irq < 8 {
            self.m.irr |= 1 << irq;
        } else {
            self.s.irr |= 1 << (irq - 8);
        }
    }

    fn pick(&self) -> Option<(bool, u8)> {
        let sc = candidate(&self.s, 0);
        let extra = if sc.is_some() { 4 } else { 0 };
        match candidate(&self.m, extra) {
            Some(2) => sc.map(|n| (true, n)),
            Some(n) => Some((false, n)),
            None => None,
        }
    }

    pub fn has_pending(&self) -> bool {
        (self.m.irr | self.s.irr) != 0 && self.pick().is_some()
    }

    pub fn ack(&mut self) -> Option<u8> {
        match self.pick()? {
            (true, n) => {
                self.s.irr &= !(1 << n);
                self.s.isr |= 1 << n;
                self.m.isr |= 4;
                Some(self.s.base + n)
            }
            (false, n) => {
                self.m.irr &= !(1 << n);
                self.m.isr |= 1 << n;
                Some(self.m.base + n)
            }
        }
    }

    pub fn io_read(&mut self, port: u16) -> u8 {
        match port {
            0x20 | 0xa0 => {
                let p = if port == 0x20 { &self.m } else { &self.s };
                if p.read_isr { p.isr } else { p.irr }
            }
            0x21 => self.m.imr,
            0xa1 => self.s.imr,
            0x4d0 => self.elcr[0],
            _ => self.elcr[1],
        }
    }

    pub fn io_write(&mut self, port: u16, v: u8) {
        match port {
            0x20 => self.m.command(v),
            0xa0 => self.s.command(v),
            0x21 => self.m.data(v),
            0xa1 => self.s.data(v),
            0x4d0 => self.elcr[0] = v,
            _ => self.elcr[1] = v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init(p: &mut Pics) {
        for (c, d, base, cas) in [(0x20, 0x21, 0x20, 4), (0xa0, 0xa1, 0x28, 2)] {
            p.io_write(c, 0x11);
            p.io_write(d, base);
            p.io_write(d, cas);
            p.io_write(d, 1);
            p.io_write(d, 0);
        }
    }

    #[test]
    fn irq_flow() {
        let mut p = Pics::default();
        init(&mut p);
        p.raise(4);
        assert_eq!(p.ack(), Some(0x24));
        p.raise(3);
        assert_eq!(p.ack(), Some(0x23));
        p.io_write(0x20, 0x20);
        p.io_write(0x20, 0x20);
        p.raise(9);
        assert_eq!(p.ack(), Some(0x29));
        assert_eq!(p.ack(), None);
        p.io_write(0xa0, 0x20);
        p.io_write(0x20, 0x20);
        p.io_write(0x21, 0x10);
        p.raise(4);
        assert_eq!(p.ack(), None);
    }
}
