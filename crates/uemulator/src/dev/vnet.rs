use super::guestmem::GuestMem;
use super::virtio::{DEVICE_TYPE_NET, Queue, VirtioDev};
use crate::netdev::NetHandle;
#[cfg(freestanding)]
use crate::prelude::*;

const F_MAC: u64 = 1 << 5;
const F_STATUS: u64 = 1 << 16;

const RECEIVEQ: usize = 0;
const TRANSMITQ: usize = 1;
const QUEUE_COUNT: usize = 2;

/// virtio_net_hdr with num_buffers. VERSION_1 is always negotiated, which makes the header this
/// long even without MRG_RXBUF. No offloads are advertised, so on transmit it carries nothing
/// we need and is skipped; on receive it is zero except num_buffers.
const HEADER_LEN: usize = 12;
const HEADER_NUM_BUFFERS_OFFSET: usize = 10;

/// Device config: the MAC at 0..6, then le16 status.
const MAC_LEN: usize = 6;
const CONFIG_STATUS_OFFSET: usize = 6;
/// VIRTIO_NET_S_LINK_UP; the link is always up.
const STATUS_LINK_UP: u8 = 1;

/// Largest frame forwarded from the guest transmit queue (excluding the header); a longer one is
/// dropped whole rather than sent truncated.
const MAX_FRAME_LEN: usize = 65536;

pub(crate) struct VirtioNet {
    mac: [u8; 6],
    net: NetHandle,
}

impl VirtioNet {
    pub fn new(mac: [u8; 6], net: NetHandle) -> Self {
        VirtioNet { mac, net }
    }
}

impl VirtioDev for VirtioNet {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_NET
    }

    fn features(&self) -> u64 {
        F_MAC | F_STATUS
    }

    fn num_queues(&self) -> usize {
        QUEUE_COUNT
    }

    fn config_read(&self, off: usize) -> u8 {
        match off {
            0..MAC_LEN => self.mac[off],
            CONFIG_STATUS_OFFSET => STATUS_LINK_UP,
            _ => 0,
        }
    }

    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if queue != TRANSMITQ {
            return false;
        }
        let mut any = false;
        while let Some(chain) = queues[TRANSMITQ].pop(mem) {
            let readable = chain.readable();
            if readable > HEADER_LEN && readable - HEADER_LEN <= MAX_FRAME_LEN {
                let data = chain.read_all(mem, readable);
                self.net.guest_send(data[HEADER_LEN..].to_vec());
            }
            queues[TRANSMITQ].push_used(mem, chain.head, 0);
            any = true;
        }
        any
    }

    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        let mut any = false;
        while self.net.has_pending() {
            let Some(chain) = queues[RECEIVEQ].pop(mem) else {
                break;
            };
            let Some(frame) = self.net.guest_recv() else {
                // Defensive: `has_pending` said a frame was queued. Give the popped buffer back.
                queues[RECEIVEQ].unpop();
                break;
            };
            // MRG_RXBUF is not offered, so a frame must fit one buffer. Like QEMU's
            // virtio_net_receive_rcu, drop one that does not and leave the buffer posted for
            // the next frame; truncating would hand the guest a corrupt packet.
            if chain.writable() < HEADER_LEN + frame.len() {
                queues[RECEIVEQ].unpop();
                continue;
            }
            let mut buf = vec![0u8; HEADER_LEN];
            buf[HEADER_NUM_BUFFERS_OFFSET..HEADER_NUM_BUFFERS_OFFSET + 2]
                .copy_from_slice(&1u16.to_le_bytes());
            buf.extend_from_slice(&frame);
            let written = chain.write_at(mem, 0, &buf);
            queues[RECEIVEQ].push_used(mem, chain.head, written as u32);
            any = true;
        }
        any
    }
}

#[cfg(test)]
#[path = "vnet_tests.rs"]
mod tests;
