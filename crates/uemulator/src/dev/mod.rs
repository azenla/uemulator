#[cfg(target_os = "uefi")]
use crate::prelude::*;
pub(crate) mod guestmem;
mod pci;
mod pic;
mod pit;
mod serial;
#[cfg(not(target_os = "uefi"))]
mod vblk;
mod vcons;
mod virtio;
mod vnet;

use crate::console::ConsoleHandle;
#[cfg(not(target_os = "uefi"))]
use crate::disk::Disk;
use crate::netdev::NetHandle;
use guestmem::GuestMem;
use pci::PciHost;
use pic::Pics;
use pit::Pit;
use serial::Uart;
#[cfg(not(target_os = "uefi"))]
use vblk::VirtioBlk;
use vcons::VirtioConsole;
use virtio::Transport;
use vnet::VirtioNet;

#[derive(Default)]
pub(crate) struct DevConfig {
    #[cfg(not(target_os = "uefi"))]
    pub disk: Option<Disk>,
    pub net: Option<([u8; 6], NetHandle)>,
    pub virtio_console: bool,
}

pub(crate) struct Devices {
    pub pic: Pics,
    pit: Pit,
    uart: Uart,
    cmos_idx: u8,
    cmos: [u8; 128],
    boot_unix: u64,
    port61: u8,
    port92: u8,
    pci: PciHost,
    pub reset_requested: bool,
}

fn bcd(v: u64) -> u8 {
    (((v / 10) << 4) | (v % 10)) as u8
}

fn civil(days: i64) -> (i64, u64, u64) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u64;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u64;
    let y = yoe + era * 400 + (m <= 2) as i64;
    (y, m, d)
}

impl Devices {
    pub fn new(console: ConsoleHandle, cfg: DevConfig) -> Self {
        let mut devs = Vec::new();
        if cfg.virtio_console {
            devs.push(Transport::new(Box::new(VirtioConsole::new(
                console.clone(),
            ))));
        }
        #[cfg(not(target_os = "uefi"))]
        if let Some(d) = cfg.disk {
            devs.push(Transport::new(Box::new(VirtioBlk::new(d))));
        }
        if let Some((mac, h)) = cfg.net {
            devs.push(Transport::new(Box::new(VirtioNet::new(mac, h))));
        }
        #[cfg(not(target_os = "uefi"))]
        let boot_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        #[cfg(target_os = "uefi")]
        let boot_unix = 0;
        Devices {
            pic: Pics::default(),
            pit: Pit::new(),
            uart: Uart::new(console),
            cmos_idx: 0,
            cmos: [0; 128],
            boot_unix,
            port61: 0,
            port92: 0,
            pci: PciHost::new(devs),
            reset_requested: false,
        }
    }

    pub fn state(&self, now: u64) -> String {
        format!("{} | {}", self.pic.state(), self.pit.state(now))
    }

    pub fn set_boot_time(&mut self, unix: u64) {
        self.boot_unix = unix;
    }

    pub fn next_deadline(&self, _now: u64) -> Option<u64> {
        self.pit.next_deadline()
    }

    fn update_uart_irq(&mut self) {
        if self.uart.irq_edge() {
            self.pic.raise(4);
        }
    }

    pub fn tick(&mut self, now: u64, mem: &mut GuestMem) {
        if self.pit.tick(now) {
            self.pic.raise(0);
        }
        let mut raise = Vec::new();
        self.pci.poll(mem, &mut raise);
        for irq in raise {
            self.pic.raise(irq);
        }
        self.uart.set_rx_enabled(!self.pci.console_active());
        self.uart.poll();
        self.update_uart_irq();
    }

    pub fn mmio_read(&mut self, pa: u64, size: u8) -> Option<u64> {
        self.pci.mmio_read(pa, size).map(|v| v as u64)
    }

    pub fn mmio_write(&mut self, pa: u64, size: u8, val: u64, mem: &mut GuestMem) -> bool {
        match self.pci.mmio_write(pa, size, val as u32, mem) {
            Some(irq) => {
                if irq != 0 {
                    self.pic.raise(irq);
                }
                true
            }
            None => false,
        }
    }

    fn cmos_read(&self, now: u64) -> u8 {
        let idx = (self.cmos_idx & 0x7f) as usize;
        let t = self.boot_unix + now / 1_000_000_000;
        let (y, mo, d) = civil((t / 86400) as i64);
        let s = t % 86400;
        let dow = (t / 86400 + 4) % 7 + 1;
        match idx {
            0 => bcd(s % 60),
            2 => bcd((s / 60) % 60),
            4 => bcd(s / 3600),
            6 => dow as u8,
            7 => bcd(d),
            8 => bcd(mo),
            9 => bcd(y as u64 % 100),
            0x32 => bcd(y as u64 / 100),
            0x0a => 0x26,
            0x0b => 0x02,
            0x0c => 0,
            0x0d => 0x80,
            _ => self.cmos[idx],
        }
    }

    pub fn io_in(&mut self, port: u16, size: u8, now: u64) -> u32 {
        let ones = u32::MAX >> (32 - 8 * size as u32);
        let v: u8 = match port {
            0x20 | 0x21 | 0xa0 | 0xa1 | 0x4d0 | 0x4d1 => self.pic.io_read(port),
            0x40..=0x42 => self.pit.read(port, now),
            0x61 => {
                let mut v = self.port61 & 0x03;
                if (now / 15085) & 1 != 0 {
                    v |= 0x10;
                }
                if self.pit.out2(now) {
                    v |= 0x20;
                }
                v
            }
            0x71 => self.cmos_read(now),
            0x92 => self.port92,
            0x3f8..=0x3ff => self.uart.read(port - 0x3f8),
            0xcf8..=0xcff if !self.pci.is_empty() => return self.pci.io_in(port, size),
            _ => return ones,
        };
        if matches!(port, 0x3f8..=0x3ff) {
            self.update_uart_irq();
        }
        v as u32 & ones
    }

    pub fn io_out(&mut self, port: u16, size: u8, v: u32, now: u64) {
        let b = v as u8;
        match port {
            0x20 | 0x21 | 0xa0 | 0xa1 | 0x4d0 | 0x4d1 => self.pic.io_write(port, b),
            0x40..=0x43 => self.pit.write(port, b, now),
            0x61 => {
                self.port61 = b & 0x03;
                self.pit.set_gate2(b & 1 != 0, now);
            }
            0x70 => self.cmos_idx = b,
            0x71 => self.cmos[(self.cmos_idx & 0x7f) as usize] = b,
            0x64 if b == 0xfe => self.reset_requested = true,
            0x92 => {
                self.port92 = b & 2;
                if b & 1 != 0 {
                    self.reset_requested = true;
                }
            }
            0xcf9 if b & 4 != 0 => self.reset_requested = true,
            0xcf8..=0xcff if !self.pci.is_empty() => self.pci.io_out(port, size, v),
            0x3f8..=0x3ff => {
                self.uart.write(port - 0x3f8, b);
                self.update_uart_irq();
            }
            _ => {}
        }
    }
}
