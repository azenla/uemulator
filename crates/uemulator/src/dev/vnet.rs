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
mod tests {
    use super::super::virtio::testutil::*;
    use super::super::virtio::{DESC_F_NEXT, DESC_F_WRITE};
    use super::*;
    use crate::dev::guestmem::test_mem;

    const QUEUE_SIZE: u16 = 8;
    const RX_DESC: u64 = 0x1000;
    const RX_AVAIL: u64 = 0x2000;
    const RX_USED: u64 = 0x3000;
    const TX_DESC: u64 = 0x8000;
    const TX_AVAIL: u64 = 0x9000;
    const TX_USED: u64 = 0xa000;
    const TX_BUF: u64 = 0xc000;
    const RX_BUF: u64 = 0xd000;
    const RAM_BYTES: usize = 0x2_0000;
    const MAC: [u8; MAC_LEN] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];

    struct Rig {
        net: NetHandle,
        dev: VirtioNet,
        queues: Vec<Queue>,
        ram: Vec<u8>,
    }

    impl Rig {
        fn new() -> Rig {
            let net = NetHandle::default();
            let mut ram = vec![0u8; RAM_BYTES];
            let queues = vec![
                setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
                setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
            ];
            Rig {
                dev: VirtioNet::new(MAC, net.clone()),
                net,
                queues,
                ram,
            }
        }

        fn kick(&mut self, queue: usize) -> bool {
            let mut mem = test_mem(&mut self.ram);
            self.dev.kick(queue, &mut self.queues, &mut mem)
        }

        fn poll(&mut self) -> bool {
            let mut mem = test_mem(&mut self.ram);
            self.dev.poll(&mut self.queues, &mut mem)
        }

        fn post_receive(&mut self, head: u16, addr: u64, len: u32) {
            put_desc(&mut self.ram, RX_DESC, head, addr, len, DESC_F_WRITE, 0);
            post(&mut self.ram, RX_AVAIL, QUEUE_SIZE, head);
        }
    }

    #[test]
    fn frame_larger_than_receive_buffer_is_dropped_not_truncated() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 20);
        assert!(rig.net.send_frame(&[0xab; 60]));
        assert!(!rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
        assert!(!rig.net.has_pending(), "the oversize frame is consumed");
    }

    fn put_tx_frame(rig: &mut Rig, head: u16, header: &[u8], frame: &[u8]) {
        let addr = TX_BUF + 0x1000 * head as u64;
        let at = addr as usize;
        rig.ram[at..at + header.len()].copy_from_slice(header);
        rig.ram[at + header.len()..at + header.len() + frame.len()].copy_from_slice(frame);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            head,
            addr,
            (header.len() + frame.len()) as u32,
            0,
            0,
        );
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, head);
    }

    fn frame_of(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(3).wrapping_add(seed))
            .collect()
    }

    fn received(rig: &Rig, buffer: u64, len: usize) -> Vec<u8> {
        rig.ram[buffer as usize..buffer as usize + len].to_vec()
    }

    #[test]
    fn device_identity_features_and_queues() {
        let rig = Rig::new();
        assert_eq!(rig.dev.device_type(), 1);
        assert_eq!(
            rig.dev.features(),
            (1 << 5) | (1 << 16),
            "MAC and STATUS only"
        );
        assert_eq!(rig.dev.num_queues(), 2);
        assert_eq!(HEADER_LEN, 12);
    }

    #[test]
    fn config_space_holds_the_mac_then_link_up_status() {
        let rig = Rig::new();
        let config: Vec<u8> = (0..12).map(|off| rig.dev.config_read(off)).collect();
        assert_eq!(&config[..6], &MAC);
        assert_eq!(
            u16::from_le_bytes([config[6], config[7]]),
            1,
            "link up, high status byte clear"
        );
        assert_eq!(&config[8..], &[0; 4]);
        for off in [12, 64, usize::MAX] {
            assert_eq!(rig.dev.config_read(off), 0);
        }
    }

    #[test]
    fn config_writes_do_not_change_the_mac_or_status() {
        let mut rig = Rig::new();
        for off in 0..8 {
            rig.dev.config_write(off, 0xee);
        }
        let config: Vec<u8> = (0..7).map(|off| rig.dev.config_read(off)).collect();
        assert_eq!(&config[..6], &MAC);
        assert_eq!(config[6], 1);
    }

    #[test]
    fn transmitted_frame_loses_its_virtio_header() {
        let mut rig = Rig::new();
        let frame = frame_of(60, 0x10);
        put_tx_frame(&mut rig, 0, &[0xee; HEADER_LEN], &frame);
        assert!(rig.kick(TRANSMITQ));
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
        assert!(rig.net.try_recv_frame().is_none());
        assert_eq!(used_entry(&rig.ram, TX_USED, 0), (0, 0));
    }

    #[test]
    fn the_header_contents_do_not_matter_on_transmit() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0u8; HEADER_LEN], &[1, 2, 3, 4]);
        put_tx_frame(&mut rig, 1, &[0xff; HEADER_LEN], &[5, 6, 7, 8]);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![5, 6, 7, 8]);
    }

    #[test]
    fn frames_go_out_in_posting_order() {
        let mut rig = Rig::new();
        for head in 0..4u16 {
            put_tx_frame(
                &mut rig,
                head,
                &[0; HEADER_LEN],
                &frame_of(20 + head as usize, head as u8),
            );
        }
        assert!(rig.kick(TRANSMITQ));
        for head in 0..4u16 {
            assert_eq!(
                rig.net.try_recv_frame().unwrap(),
                frame_of(20 + head as usize, head as u8)
            );
        }
        assert_eq!(used_idx(&rig.ram, TX_USED), 4);
    }

    #[test]
    fn header_and_frame_in_separate_descriptors_are_handled() {
        let mut rig = Rig::new();
        let frame = frame_of(100, 0x40);
        rig.ram[TX_BUF as usize..TX_BUF as usize + HEADER_LEN].fill(0);
        rig.ram[TX_BUF as usize + 0x100..TX_BUF as usize + 0x100 + 100].copy_from_slice(&frame);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            0,
            TX_BUF,
            HEADER_LEN as u32,
            DESC_F_NEXT,
            1,
        );
        put_desc(&mut rig.ram, TX_DESC, 1, TX_BUF + 0x100, 100, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn header_split_across_descriptors_is_still_skipped_correctly() {
        let mut rig = Rig::new();
        let frame = frame_of(30, 0x70);
        rig.ram[TX_BUF as usize..TX_BUF as usize + 5].fill(0xaa);
        rig.ram[TX_BUF as usize + 0x100..TX_BUF as usize + 0x100 + 7].fill(0xbb);
        rig.ram[TX_BUF as usize + 0x200..TX_BUF as usize + 0x200 + 30].copy_from_slice(&frame);
        put_desc(&mut rig.ram, TX_DESC, 0, TX_BUF, 5, DESC_F_NEXT, 1);
        put_desc(&mut rig.ram, TX_DESC, 1, TX_BUF + 0x100, 7, DESC_F_NEXT, 2);
        put_desc(&mut rig.ram, TX_DESC, 2, TX_BUF + 0x200, 30, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn chain_with_only_a_header_or_less_sends_nothing_but_is_completed() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[]);
        put_tx_frame(&mut rig, 1, &[0; 5], &[]);
        put_desc(&mut rig.ram, TX_DESC, 2, TX_BUF, 0, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 2);
        assert!(rig.kick(TRANSMITQ));
        assert!(rig.net.try_recv_frame().is_none());
        assert_eq!(used_idx(&rig.ram, TX_USED), 3);
    }

    #[test]
    fn smallest_frame_is_one_byte_after_the_header() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[0x42]);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![0x42]);
    }

    #[test]
    fn frame_of_exactly_the_maximum_length_is_sent() {
        let mut ram = vec![0u8; 1 << 20];
        let net = NetHandle::default();
        let mut dev = VirtioNet::new(MAC, net.clone());
        let mut queues = vec![
            setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
            setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
        ];
        let frame = frame_of(MAX_FRAME_LEN, 1);
        let at = 0x1_0000usize;
        ram[at + HEADER_LEN..at + HEADER_LEN + MAX_FRAME_LEN].copy_from_slice(&frame);
        put_desc(
            &mut ram,
            TX_DESC,
            0,
            at as u64,
            (HEADER_LEN + MAX_FRAME_LEN) as u32,
            0,
            0,
        );
        post(&mut ram, TX_AVAIL, QUEUE_SIZE, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(TRANSMITQ, &mut queues, &mut mem));
        assert_eq!(net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn frame_longer_than_the_maximum_is_dropped_not_truncated() {
        let mut ram = vec![0u8; 1 << 20];
        let net = NetHandle::default();
        let mut dev = VirtioNet::new(MAC, net.clone());
        let mut queues = vec![
            setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
            setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
        ];
        let at = 0x1_0000u64;
        put_desc(
            &mut ram,
            TX_DESC,
            0,
            at,
            (HEADER_LEN + MAX_FRAME_LEN + 1) as u32,
            0,
            0,
        );
        post(&mut ram, TX_AVAIL, QUEUE_SIZE, 0);
        let mut mem = test_mem(&mut ram);
        assert!(
            dev.kick(TRANSMITQ, &mut queues, &mut mem),
            "the buffer is still returned"
        );
        assert!(net.try_recv_frame().is_none());
    }

    #[test]
    fn writable_segments_of_a_transmit_chain_are_not_part_of_the_frame() {
        let mut rig = Rig::new();
        let frame = frame_of(10, 5);
        rig.ram[TX_BUF as usize + HEADER_LEN..TX_BUF as usize + HEADER_LEN + 10]
            .copy_from_slice(&frame);
        rig.ram[TX_BUF as usize + 0x200..TX_BUF as usize + 0x210].fill(0xcc);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            0,
            TX_BUF,
            (HEADER_LEN + 10) as u32,
            DESC_F_NEXT,
            1,
        );
        put_desc(
            &mut rig.ram,
            TX_DESC,
            1,
            TX_BUF + 0x200,
            16,
            DESC_F_WRITE,
            0,
        );
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn kick_on_the_receive_queue_sends_nothing() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[1, 2, 3]);
        assert!(!rig.kick(RECEIVEQ));
        assert!(rig.net.try_recv_frame().is_none());
        assert_eq!(used_idx(&rig.ram, TX_USED), 0);
    }

    #[test]
    fn transmit_kick_with_nothing_posted_reports_no_work() {
        let mut rig = Rig::new();
        assert!(!rig.kick(TRANSMITQ));
    }

    #[test]
    fn received_frame_gets_a_zero_header_with_one_buffer_after_it() {
        let mut rig = Rig::new();
        rig.ram[RX_BUF as usize..RX_BUF as usize + 128].fill(0xff);
        rig.post_receive(0, RX_BUF, 128);
        let frame = frame_of(60, 0x21);
        assert!(rig.net.send_frame(&frame));
        assert!(rig.poll());
        let header = received(&rig, RX_BUF, HEADER_LEN);
        let mut expected_header = [0u8; HEADER_LEN];
        expected_header[HEADER_NUM_BUFFERS_OFFSET] = 1;
        assert_eq!(
            header, expected_header,
            "flags, gso, csum fields zero; num_buffers == 1 little endian"
        );
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 60), frame);
        assert_eq!(
            rig.ram[RX_BUF as usize + HEADER_LEN + 60],
            0xff,
            "nothing past the frame is touched"
        );
        assert_eq!(
            used_entry(&rig.ram, RX_USED, 0),
            (0, (HEADER_LEN + 60) as u32)
        );
    }

    #[test]
    fn frame_that_exactly_fills_the_buffer_is_delivered() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, (HEADER_LEN + 60) as u32);
        let frame = frame_of(60, 1);
        rig.net.send_frame(&frame);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 60), frame);
    }

    #[test]
    fn frame_one_byte_too_big_for_the_buffer_is_dropped() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, (HEADER_LEN + 59) as u32);
        rig.net.send_frame(&frame_of(60, 1));
        assert!(!rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
        assert!(!rig.net.has_pending());
        assert!(
            received(&rig, RX_BUF, HEADER_LEN + 59)
                .iter()
                .all(|&b| b == 0),
            "no partial data"
        );
    }

    #[test]
    fn dropped_frame_leaves_its_buffer_posted_for_the_next_frame() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 40);
        rig.net.send_frame(&frame_of(100, 1));
        assert!(!rig.poll());
        assert_eq!(
            rig.queues[RECEIVEQ].last_avail, 0,
            "the buffer is available again"
        );
        let small = frame_of(20, 9);
        rig.net.send_frame(&small);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 20), small);
        assert_eq!(
            used_entry(&rig.ram, RX_USED, 0),
            (0, (HEADER_LEN + 20) as u32)
        );
    }

    #[test]
    fn oversize_frame_followed_by_a_fitting_one_in_the_same_poll() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 64);
        let big = frame_of(200, 1);
        let small = frame_of(30, 2);
        rig.net.send_frame(&big);
        rig.net.send_frame(&small);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 30), small);
        assert_eq!(used_idx(&rig.ram, RX_USED), 1);
        assert!(!rig.net.has_pending());
    }

    #[test]
    fn frame_waits_in_the_host_queue_until_a_buffer_is_posted() {
        let mut rig = Rig::new();
        rig.net.send_frame(&[7; 30]);
        assert!(!rig.poll());
        assert!(
            rig.net.has_pending(),
            "no buffer: the frame must not be lost"
        );
        rig.post_receive(0, RX_BUF, 128);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 30), vec![7; 30]);
    }

    #[test]
    fn several_frames_fill_several_buffers_in_order() {
        let mut rig = Rig::new();
        for head in 0..3u16 {
            rig.post_receive(head, RX_BUF + 0x400 * head as u64, 256);
            rig.net.send_frame(&frame_of(40, head as u8));
        }
        assert!(rig.poll());
        for head in 0..3u16 {
            assert_eq!(
                received(&rig, RX_BUF + 0x400 * head as u64 + HEADER_LEN as u64, 40),
                frame_of(40, head as u8)
            );
            assert_eq!(
                used_entry(&rig.ram, RX_USED, head as usize),
                (head as u32, (HEADER_LEN + 40) as u32)
            );
        }
    }

    #[test]
    fn more_frames_than_buffers_leave_the_extra_ones_queued() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 256);
        rig.net.send_frame(&frame_of(40, 1));
        rig.net.send_frame(&frame_of(40, 2));
        assert!(rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 1);
        assert!(rig.net.has_pending());
    }

    #[test]
    fn empty_frame_is_delivered_as_just_a_header() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 64);
        rig.net.send_frame(&[]);
        assert!(rig.poll());
        assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, HEADER_LEN as u32));
    }

    #[test]
    fn receive_buffer_with_no_room_for_the_header_drops_the_frame() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, (HEADER_LEN - 1) as u32);
        rig.net.send_frame(&[]);
        assert!(!rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
    }

    #[test]
    fn frame_may_span_several_writable_segments_of_one_buffer() {
        let mut rig = Rig::new();
        put_desc(
            &mut rig.ram,
            RX_DESC,
            0,
            RX_BUF,
            20,
            DESC_F_WRITE | DESC_F_NEXT,
            1,
        );
        put_desc(
            &mut rig.ram,
            RX_DESC,
            1,
            RX_BUF + 0x100,
            20,
            DESC_F_WRITE | DESC_F_NEXT,
            2,
        );
        put_desc(
            &mut rig.ram,
            RX_DESC,
            2,
            RX_BUF + 0x200,
            40,
            DESC_F_WRITE,
            0,
        );
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        let frame = frame_of(50, 3);
        rig.net.send_frame(&frame);
        assert!(rig.poll());
        // 12 header bytes + 8 frame bytes, then 20, then 22 more.
        let mut gathered = received(&rig, RX_BUF + HEADER_LEN as u64, 8);
        gathered.extend(received(&rig, RX_BUF + 0x100, 20));
        gathered.extend(received(&rig, RX_BUF + 0x200, 22));
        assert_eq!(gathered, frame);
        assert_eq!(
            used_entry(&rig.ram, RX_USED, 0),
            (0, (HEADER_LEN + 50) as u32)
        );
    }

    #[test]
    fn the_buffer_size_check_counts_writable_segments_only() {
        let mut rig = Rig::new();
        put_desc(&mut rig.ram, RX_DESC, 0, RX_BUF, 1000, DESC_F_NEXT, 1);
        put_desc(
            &mut rig.ram,
            RX_DESC,
            1,
            RX_BUF + 0x400,
            20,
            DESC_F_WRITE,
            0,
        );
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        rig.net.send_frame(&frame_of(30, 1));
        assert!(
            !rig.poll(),
            "a 20-byte writable area cannot take 12 + 30 bytes"
        );
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
    }

    #[test]
    fn receive_buffer_outside_ram_breaks_the_queue_and_keeps_the_frame() {
        let mut rig = Rig::new();
        rig.post_receive(0, RAM_BYTES as u64 - 4, 64);
        rig.net.send_frame(&[1, 2, 3]);
        assert!(!rig.poll());
        assert!(rig.queues[RECEIVEQ].broken);
        assert!(rig.net.has_pending());
    }

    #[test]
    fn guest_to_host_and_host_to_guest_traffic_do_not_interfere() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 128);
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[9, 9, 9]);
        rig.net.send_frame(&[4, 4, 4, 4]);
        assert!(rig.kick(TRANSMITQ));
        assert!(rig.poll());
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![9, 9, 9]);
        assert_eq!(
            received(&rig, RX_BUF + HEADER_LEN as u64, 4),
            vec![4, 4, 4, 4]
        );
    }

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
