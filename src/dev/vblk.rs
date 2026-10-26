use super::guestmem::GuestMem;
use super::virtio::{Chain, Queue, VirtioDev};
use crate::disk::Disk;

const SECTOR: u64 = 512;
const F_SEG_MAX: u64 = 1 << 2;
const F_RO: u64 = 1 << 5;
const F_BLK_SIZE: u64 = 1 << 6;
const F_FLUSH: u64 = 1 << 9;
const MAX_REQ: usize = 16 << 20;

pub(crate) struct VirtioBlk {
    disk: Disk,
    id: [u8; 20],
}

impl VirtioBlk {
    pub fn new(disk: Disk) -> Self {
        let mut id = [0u8; 20];
        id[..9].copy_from_slice(b"uemulator");
        VirtioBlk { disk, id }
    }

    fn capacity(&self) -> u64 {
        self.disk.backend.len() / SECTOR
    }

    fn request(&mut self, mem: &mut GuestMem, chain: &Chain) -> u32 {
        let stream = chain.read_all(mem, MAX_REQ + 16);
        let total_w = chain.writable();
        if stream.len() < 16 || total_w == 0 {
            return 0;
        }
        let ty = u32::from_le_bytes(stream[0..4].try_into().unwrap());
        let sector = u64::from_le_bytes(stream[8..16].try_into().unwrap());
        let data_w = total_w - 1;
        let mut written = 0usize;
        let status: u8 = match ty {
            0 => {
                let off = sector.saturating_mul(SECTOR);
                let n = data_w.min(MAX_REQ);
                if off
                    .checked_add(n as u64)
                    .is_none_or(|e| e > self.disk.backend.len())
                {
                    1
                } else {
                    let mut buf = vec![0u8; n];
                    match self.disk.backend.read_at(off, &mut buf) {
                        Ok(()) => {
                            written = chain.write_at(mem, 0, &buf);
                            0
                        }
                        Err(_) => 1,
                    }
                }
            }
            1 => {
                let data = &stream[16..];
                let off = sector.saturating_mul(SECTOR);
                if self.disk.backend.read_only()
                    || off
                        .checked_add(data.len() as u64)
                        .is_none_or(|e| e > self.disk.backend.len())
                {
                    1
                } else if self.disk.backend.write_at(off, data).is_ok() {
                    0
                } else {
                    1
                }
            }
            4 => {
                if self.disk.backend.flush().is_ok() {
                    0
                } else {
                    1
                }
            }
            8 => {
                written = chain.write_at(mem, 0, &self.id[..data_w.min(20)]);
                0
            }
            _ => 2,
        };
        chain.write_at(mem, data_w, &[status]);
        (written + 1) as u32
    }
}

impl VirtioDev for VirtioBlk {
    fn id(&self) -> u16 {
        2
    }

    fn features(&self) -> u64 {
        let mut f = F_SEG_MAX | F_BLK_SIZE | F_FLUSH;
        if self.disk.backend.read_only() {
            f |= F_RO;
        }
        f
    }

    fn num_queues(&self) -> usize {
        1
    }

    fn config_read(&self, off: usize) -> u8 {
        let mut cfg = [0u8; 24];
        cfg[0..8].copy_from_slice(&self.capacity().to_le_bytes());
        cfg[12..16].copy_from_slice(&126u32.to_le_bytes());
        cfg[20..24].copy_from_slice(&(SECTOR as u32).to_le_bytes());
        cfg.get(off).copied().unwrap_or(0)
    }

    fn kick(&mut self, q: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        let mut any = false;
        while let Some(chain) = queues[q].pop(mem) {
            let len = self.request(mem, &chain);
            queues[q].push_used(mem, chain.head, len);
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

    const DESC: u64 = 0x1000;
    const AVAIL: u64 = 0x2000;
    const USED: u64 = 0x3000;

    fn submit(
        dev: &mut VirtioBlk,
        q: &mut Queue,
        ram: &mut [u8],
        ty: u32,
        sector: u64,
        data: Option<&[u8]>,
        dlen: u32,
    ) -> u8 {
        let hdr = 0x4000usize;
        ram[hdr..hdr + 4].copy_from_slice(&ty.to_le_bytes());
        ram[hdr + 8..hdr + 16].copy_from_slice(&sector.to_le_bytes());
        put_desc(ram, DESC, 0, hdr as u64, 16, 1, 1);
        let mut next = 1u16;
        if let Some(d) = data {
            ram[0x5000..0x5000 + d.len()].copy_from_slice(d);
            put_desc(ram, DESC, 1, 0x5000, d.len() as u32, 1, 2);
            next = 2;
        } else if dlen > 0 {
            put_desc(ram, DESC, 1, 0x5000, dlen, 1 | 2, 2);
            next = 2;
        }
        put_desc(ram, DESC, next, 0x6000, 1, 2, 0);
        ram[0x6000] = 0xff;
        post(ram, AVAIL, q.size, 0);
        let mut mem = test_mem(ram);
        assert!(dev.kick(0, std::slice::from_mut(q), &mut mem));
        ram[0x6000]
    }

    #[test]
    fn read_write_flush_roundtrip() {
        let mut ram = vec![0u8; 0x10000];
        let mut dev = VirtioBlk::new(Disk::memory(vec![0u8; 8192]));
        let mut q = setup_queue(&mut ram, DESC, AVAIL, USED, 16);
        let payload: Vec<u8> = (0..1024u32).map(|i| i as u8).collect();
        assert_eq!(
            submit(&mut dev, &mut q, &mut ram, 1, 2, Some(&payload), 0),
            0
        );
        assert_eq!(submit(&mut dev, &mut q, &mut ram, 4, 0, None, 0), 0);
        ram[0x5000..0x5400].fill(0);
        assert_eq!(submit(&mut dev, &mut q, &mut ram, 0, 2, None, 1024), 0);
        assert_eq!(&ram[0x5000..0x5400], &payload[..]);
        assert_eq!(used_idx(&ram, USED), 3);
        assert_eq!(submit(&mut dev, &mut q, &mut ram, 0, 100, None, 512), 1);
        assert_eq!(submit(&mut dev, &mut q, &mut ram, 99, 0, None, 0), 2);
        assert_eq!(dev.capacity(), 16);
    }

    #[test]
    fn read_only_rejects_writes() {
        struct Ro;
        impl crate::disk::BlockBackend for Ro {
            fn len(&self) -> u64 {
                4096
            }
            fn read_at(&mut self, _: u64, b: &mut [u8]) -> std::io::Result<()> {
                b.fill(7);
                Ok(())
            }
            fn write_at(&mut self, _: u64, _: &[u8]) -> std::io::Result<()> {
                Ok(())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
            fn read_only(&self) -> bool {
                true
            }
        }
        let mut ram = vec![0u8; 0x10000];
        let mut dev = VirtioBlk::new(Disk::from_backend(Ro));
        assert_ne!(dev.features() & F_RO, 0);
        let mut q = setup_queue(&mut ram, DESC, AVAIL, USED, 16);
        assert_eq!(
            submit(&mut dev, &mut q, &mut ram, 1, 0, Some(&[1u8; 512]), 0),
            1
        );
    }
}
