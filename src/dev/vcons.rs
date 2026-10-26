use super::guestmem::GuestMem;
use super::virtio::{Queue, VirtioDev};
use crate::console::ConsoleHandle;

const F_SIZE: u64 = 1;

pub(crate) struct VirtioConsole {
    console: ConsoleHandle,
    cols: u16,
    rows: u16,
    changed: bool,
}

impl VirtioConsole {
    pub fn new(console: ConsoleHandle) -> Self {
        let (cols, rows) = console.size();
        VirtioConsole {
            console,
            cols,
            rows,
            changed: false,
        }
    }
}

impl VirtioDev for VirtioConsole {
    fn id(&self) -> u16 {
        3
    }

    fn features(&self) -> u64 {
        F_SIZE
    }

    fn num_queues(&self) -> usize {
        2
    }

    fn config_read(&self, off: usize) -> u8 {
        let mut cfg = [0u8; 12];
        cfg[0..2].copy_from_slice(&self.cols.to_le_bytes());
        cfg[2..4].copy_from_slice(&self.rows.to_le_bytes());
        cfg[4..8].copy_from_slice(&1u32.to_le_bytes());
        cfg.get(off).copied().unwrap_or(0)
    }

    fn kick(&mut self, q: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if q != 1 {
            return false;
        }
        let mut any = false;
        while let Some(chain) = queues[1].pop(mem) {
            let data = chain.read_all(mem, 1 << 16);
            self.console.guest_tx_bytes(&data);
            queues[1].push_used(mem, chain.head, 0);
            any = true;
        }
        any
    }

    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if let Some((c, r)) = self.console.take_size_change() {
            (self.cols, self.rows) = (c, r);
            self.changed = true;
        }
        let mut any = false;
        while self.console.has_input() {
            let Some(chain) = queues[0].pop(mem) else {
                break;
            };
            let data = self.console.guest_rx_bulk(chain.writable().max(1));
            let n = chain.write_at(mem, 0, &data);
            queues[0].push_used(mem, chain.head, n as u32);
            any = true;
        }
        any
    }

    fn config_event(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::virtio::testutil::*;
    use super::*;
    use crate::dev::guestmem::test_mem;

    #[test]
    fn io_and_resize() {
        let c = ConsoleHandle::default();
        let mut dev = VirtioConsole::new(c.clone());
        assert_eq!(dev.config_read(0), 80);
        let mut ram = vec![0u8; 0x20000];
        let mut qs = vec![
            setup_queue(&mut ram, 0x1000, 0x2000, 0x3000, 8),
            setup_queue(&mut ram, 0x8000, 0x9000, 0xa000, 8),
        ];
        ram[0xc000..0xc003].copy_from_slice(b"hey");
        put_desc(&mut ram, 0x8000, 0, 0xc000, 3, 0, 0);
        post(&mut ram, 0x9000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(1, &mut qs, &mut mem));
        let mut b = [0u8; 8];
        assert_eq!(c.read(&mut b, None), 3);
        c.write(b"ok");
        put_desc(&mut ram, 0x1000, 0, 0xd000, 64, 2, 0);
        post(&mut ram, 0x2000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.poll(&mut qs, &mut mem));
        assert_eq!(&ram[0xd000..0xd002], b"ok");
        c.resize(132, 43);
        let mut mem = test_mem(&mut ram);
        dev.poll(&mut qs, &mut mem);
        assert!(dev.config_event());
        assert_eq!((dev.config_read(0), dev.config_read(2)), (132, 43));
    }
}
