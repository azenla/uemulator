use super::guestmem::GuestMem;
use super::virtio::{Queue, VirtioDev};
use crate::console::ConsoleHandle;

/// Virtio device id for a console.
const VIRTIO_ID_CONSOLE: u16 = 3;
/// VIRTIO_CONSOLE_F_SIZE: config space carries cols/rows and changes raise a config interrupt.
/// The multiport feature is not offered, so there are no control queues: just port 0's pair.
const F_SIZE: u64 = 1;

/// Host-to-guest queue (guest posts empty buffers).
const RECEIVEQ: usize = 0;
/// Guest-to-host queue.
const TRANSMITQ: usize = 1;
const QUEUE_COUNT: usize = 2;

/// Device config layout: le16 cols, le16 rows, le32 max_nr_ports, le32 emerg_wr.
const CONFIG_LEN: usize = 12;
const CONFIG_COLS_OFFSET: usize = 0;
const CONFIG_ROWS_OFFSET: usize = 2;
const CONFIG_MAX_NR_PORTS_OFFSET: usize = 4;
const MAX_NR_PORTS: u32 = 1;

/// Bytes read from one transmit chain; anything beyond is dropped.
const MAX_TX_CHAIN_BYTES: usize = 1 << 16;

pub(crate) struct VirtioConsole {
    console: ConsoleHandle,
    cols: u16,
    rows: u16,
    /// Size changed since the guest was last told; consumed by `config_event`.
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
    fn device_type(&self) -> u16 {
        VIRTIO_ID_CONSOLE
    }

    fn features(&self) -> u64 {
        F_SIZE
    }

    fn num_queues(&self) -> usize {
        QUEUE_COUNT
    }

    fn config_read(&self, off: usize) -> u8 {
        let mut cfg = [0u8; CONFIG_LEN];
        cfg[CONFIG_COLS_OFFSET..CONFIG_COLS_OFFSET + 2].copy_from_slice(&self.cols.to_le_bytes());
        cfg[CONFIG_ROWS_OFFSET..CONFIG_ROWS_OFFSET + 2].copy_from_slice(&self.rows.to_le_bytes());
        cfg[CONFIG_MAX_NR_PORTS_OFFSET..CONFIG_MAX_NR_PORTS_OFFSET + 4]
            .copy_from_slice(&MAX_NR_PORTS.to_le_bytes());
        cfg.get(off).copied().unwrap_or(0)
    }

    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if queue != TRANSMITQ {
            return false;
        }
        let mut any = false;
        while let Some(chain) = queues[TRANSMITQ].pop(mem) {
            let data = chain.read_all(mem, MAX_TX_CHAIN_BYTES);
            self.console.guest_tx_bytes(&data);
            queues[TRANSMITQ].push_used(mem, chain.head, 0);
            any = true;
        }
        any
    }

    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if let Some((cols, rows)) = self.console.take_size_change() {
            (self.cols, self.rows) = (cols, rows);
            self.changed = true;
        }
        let mut any = false;
        while self.console.has_input() {
            let Some(chain) = queues[RECEIVEQ].pop(mem) else {
                break;
            };
            let data = self.console.guest_rx_bulk(chain.writable().max(1));
            let written = chain.write_at(mem, 0, &data);
            queues[RECEIVEQ].push_used(mem, chain.head, written as u32);
            any = true;
        }
        any
    }

    fn config_event(&mut self) -> bool {
        core::mem::take(&mut self.changed)
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
        assert_eq!(dev.config_read(CONFIG_COLS_OFFSET), 80);
        let mut ram = vec![0u8; 0x20000];
        let mut qs = vec![
            setup_queue(&mut ram, 0x1000, 0x2000, 0x3000, 8),
            setup_queue(&mut ram, 0x8000, 0x9000, 0xa000, 8),
        ];
        ram[0xc000..0xc003].copy_from_slice(b"hey");
        put_desc(&mut ram, 0x8000, 0, 0xc000, 3, 0, 0);
        post(&mut ram, 0x9000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(TRANSMITQ, &mut qs, &mut mem));
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
        assert_eq!(
            (
                dev.config_read(CONFIG_COLS_OFFSET),
                dev.config_read(CONFIG_ROWS_OFFSET)
            ),
            (132, 43)
        );
    }
}
