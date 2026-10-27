use super::guestmem::GuestMem;
use super::virtio::{DEVICE_TYPE_CONSOLE, Queue, VirtioDev};
use crate::console::ConsoleHandle;

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

/// Transmit data is handed to the console in pieces of this size. A chain longer than one piece
/// is forwarded whole, never truncated: dropping console output silently is worse than briefly
/// holding one more piece of a buffer the guest already proved is in RAM.
const TX_CHUNK_BYTES: usize = 1 << 16;

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
        DEVICE_TYPE_CONSOLE
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
            chain.read_chunks(mem, TX_CHUNK_BYTES, |piece| {
                self.console.guest_tx_bytes(piece)
            });
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
            // Take only what the buffer can hold: bytes pulled from the console and not stored
            // would be lost. A buffer with no writable space is returned empty.
            let data = self.console.guest_rx_bulk(chain.writable());
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
#[path = "vcons_tests.rs"]
mod tests;
