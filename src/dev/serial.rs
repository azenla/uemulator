use std::collections::VecDeque;

use crate::console::ConsoleHandle;

pub(crate) struct Uart {
    console: ConsoleHandle,
    rx: VecDeque<u8>,
    dll: u8,
    dlm: u8,
    ier: u8,
    fcr: u8,
    lcr: u8,
    mcr: u8,
    scr: u8,
    thre_pending: bool,
    last_level: bool,
}

impl Uart {
    pub fn new(console: ConsoleHandle) -> Self {
        Uart {
            console,
            rx: VecDeque::new(),
            dll: 0,
            dlm: 0,
            ier: 0,
            fcr: 0,
            lcr: 0,
            mcr: 0,
            scr: 0,
            thre_pending: false,
            last_level: false,
        }
    }

    fn level(&self) -> bool {
        (self.ier & 1 != 0 && !self.rx.is_empty()) || (self.ier & 2 != 0 && self.thre_pending)
    }

    pub fn irq_edge(&mut self) -> bool {
        let l = self.level();
        let edge = l && !self.last_level;
        self.last_level = l;
        edge
    }

    pub fn poll(&mut self) -> bool {
        let mut any = false;
        while self.rx.len() < 16 {
            match self.console.guest_rx() {
                Some(b) => {
                    self.rx.push_back(b);
                    any = true;
                }
                None => break,
            }
        }
        any
    }

    fn lsr(&self) -> u8 {
        0x60 | (!self.rx.is_empty()) as u8
    }

    fn msr(&self) -> u8 {
        if self.mcr & 0x10 != 0 {
            ((self.mcr & 2) << 3)
                | ((self.mcr & 1) << 5)
                | ((self.mcr & 4) << 4)
                | ((self.mcr & 8) << 4)
        } else {
            0xb0
        }
    }

    pub fn read(&mut self, reg: u16) -> u8 {
        let dlab = self.lcr & 0x80 != 0;
        match reg {
            0 if dlab => self.dll,
            0 => {
                let v = self.rx.pop_front().unwrap_or(0);
                self.poll();
                v
            }
            1 if dlab => self.dlm,
            1 => self.ier,
            2 => {
                let fifo = if self.fcr & 1 != 0 { 0xc0 } else { 0 };
                if self.ier & 1 != 0 && !self.rx.is_empty() {
                    0x04 | fifo
                } else if self.ier & 2 != 0 && self.thre_pending {
                    self.thre_pending = false;
                    0x02 | fifo
                } else {
                    0x01 | fifo
                }
            }
            3 => self.lcr,
            4 => self.mcr,
            5 => self.lsr(),
            6 => self.msr(),
            _ => self.scr,
        }
    }

    pub fn write(&mut self, reg: u16, v: u8) {
        let dlab = self.lcr & 0x80 != 0;
        match reg {
            0 if dlab => self.dll = v,
            0 => {
                self.thre_pending = true;
                if self.mcr & 0x10 != 0 {
                    if self.rx.len() < 16 {
                        self.rx.push_back(v);
                    }
                } else {
                    self.console.guest_tx(v);
                }
            }
            1 if dlab => self.dlm = v,
            1 => {
                let was = self.ier;
                self.ier = v & 0x0f;
                if self.ier & 2 != 0 && was & 2 == 0 {
                    self.thre_pending = true;
                }
            }
            2 => {
                self.fcr = v & 0xc9;
                if v & 2 != 0 {
                    self.rx.clear();
                }
            }
            3 => self.lcr = v,
            4 => self.mcr = v & 0x1f,
            7 => self.scr = v,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autoconfig_probe() {
        let mut u = Uart::new(ConsoleHandle::default());
        u.write(1, 0);
        assert_eq!(u.read(1), 0);
        u.write(4, 0x1a);
        assert_eq!(u.read(6) & 0xf0, 0x90);
        u.write(2, 0x01);
        assert_eq!(u.read(2) & 0xc0, 0xc0);
        u.write(7, 0x5a);
        assert_eq!(u.read(7), 0x5a);
    }

    #[test]
    fn tx_and_rx() {
        let c = ConsoleHandle::default();
        let mut u = Uart::new(c.clone());
        u.write(0, b'x');
        let mut b = [0u8; 4];
        assert_eq!(c.read(&mut b, None), 1);
        assert_eq!(b[0], b'x');
        c.write(b"hi");
        assert!(u.poll());
        u.write(1, 1);
        assert!(u.irq_edge());
        assert_eq!(u.read(2) & 0xf, 4);
        assert_eq!(u.read(0), b'h');
        assert_eq!(u.read(0), b'i');
        assert_eq!(u.read(5) & 1, 0);
    }
}
