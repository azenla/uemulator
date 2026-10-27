use super::guestmem::GuestMem;
use super::virtio::{Queue, VirtioDev};
use crate::netdev::NetHandle;
#[cfg(target_os = "uefi")]
use crate::prelude::*;

/// Virtio device id for a network card.
const VIRTIO_ID_NET: u16 = 1;
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

/// Largest frame accepted from the guest transmit queue (excluding the header).
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
        VIRTIO_ID_NET
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
            let data = chain.read_all(mem, MAX_FRAME_LEN + HEADER_LEN);
            if data.len() > HEADER_LEN {
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
                queues[RECEIVEQ].last_avail = queues[RECEIVEQ].last_avail.wrapping_sub(1);
                break;
            };
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
mod tests {
    use super::super::virtio::testutil::*;
    use super::*;
    use crate::dev::guestmem::test_mem;

    #[test]
    fn tx_and_rx() {
        let net = NetHandle::default();
        let mut dev = VirtioNet::new([2, 0, 0, 0, 0, 1], net.clone());
        let mut ram = vec![0u8; 0x20000];
        let mut qs = vec![
            setup_queue(&mut ram, 0x1000, 0x2000, 0x3000, 8),
            setup_queue(&mut ram, 0x8000, 0x9000, 0xa000, 8),
        ];
        let frame = [0xabu8; 60];
        ram[0xc000 + HEADER_LEN..0xc000 + HEADER_LEN + 60].copy_from_slice(&frame);
        put_desc(&mut ram, 0x8000, 0, 0xc000, (HEADER_LEN + 60) as u32, 0, 0);
        post(&mut ram, 0x9000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(TRANSMITQ, &mut qs, &mut mem));
        assert_eq!(net.try_recv_frame().unwrap(), frame);
        assert!(!dev.poll(&mut qs, &mut mem));
        assert!(net.send_frame(&[1, 2, 3, 4]));
        put_desc(&mut ram, 0x1000, 0, 0xd000, 2048, 2, 0);
        post(&mut ram, 0x2000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.poll(&mut qs, &mut mem));
        assert_eq!(
            &ram[0xd000 + HEADER_LEN..0xd000 + HEADER_LEN + 4],
            &[1, 2, 3, 4]
        );
        assert_eq!(used_idx(&ram, 0x3000), 1);
    }
}
