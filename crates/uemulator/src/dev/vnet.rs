use super::guestmem::GuestMem;
use super::virtio::{Queue, VirtioDev};
use crate::netdev::NetHandle;
#[cfg(target_os = "uefi")]
use crate::prelude::*;

const F_MAC: u64 = 1 << 5;
const F_STATUS: u64 = 1 << 16;
const HDR: usize = 12;
const MAX_FRAME: usize = 65536;

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
    fn id(&self) -> u16 {
        1
    }

    fn features(&self) -> u64 {
        F_MAC | F_STATUS
    }

    fn num_queues(&self) -> usize {
        2
    }

    fn config_read(&self, off: usize) -> u8 {
        match off {
            0..=5 => self.mac[off],
            6 => 1,
            _ => 0,
        }
    }

    fn kick(&mut self, q: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if q != 1 {
            return false;
        }
        let mut any = false;
        while let Some(chain) = queues[1].pop(mem) {
            let data = chain.read_all(mem, MAX_FRAME + HDR);
            if data.len() > HDR {
                self.net.guest_send(data[HDR..].to_vec());
            }
            queues[1].push_used(mem, chain.head, 0);
            any = true;
        }
        any
    }

    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        let mut any = false;
        while self.net.has_pending() {
            let Some(chain) = queues[0].pop(mem) else {
                break;
            };
            let Some(frame) = self.net.guest_recv() else {
                queues[0].last_avail = queues[0].last_avail.wrapping_sub(1);
                break;
            };
            let mut buf = vec![0u8; HDR];
            buf[10..12].copy_from_slice(&1u16.to_le_bytes());
            buf.extend_from_slice(&frame);
            let n = chain.write_at(mem, 0, &buf);
            queues[0].push_used(mem, chain.head, n as u32);
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
        ram[0xc000 + HDR..0xc000 + HDR + 60].copy_from_slice(&frame);
        put_desc(&mut ram, 0x8000, 0, 0xc000, (HDR + 60) as u32, 0, 0);
        post(&mut ram, 0x9000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(1, &mut qs, &mut mem));
        assert_eq!(net.try_recv_frame().unwrap(), frame);
        assert!(!dev.poll(&mut qs, &mut mem));
        assert!(net.send_frame(&[1, 2, 3, 4]));
        put_desc(&mut ram, 0x1000, 0, 0xd000, 2048, 2, 0);
        post(&mut ram, 0x2000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.poll(&mut qs, &mut mem));
        assert_eq!(&ram[0xd000 + HDR..0xd000 + HDR + 4], &[1, 2, 3, 4]);
        assert_eq!(used_idx(&ram, 0x3000), 1);
    }
}
