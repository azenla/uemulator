use super::guestmem::GuestMem;
use super::virtio::{Chain, DEVICE_TYPE_BLOCK, Queue, VirtioDev};
use crate::disk::Disk;

/// virtio-blk addresses the disk in 512-byte units regardless of `BLK_SIZE`.
const SECTOR_SIZE: u64 = 512;

/// Feature bits (`VIRTIO_BLK_F_*`).
const VIRTIO_BLK_F_SEG_MAX: u64 = 1 << 2;
const VIRTIO_BLK_F_RO: u64 = 1 << 5;
const VIRTIO_BLK_F_BLK_SIZE: u64 = 1 << 6;
const VIRTIO_BLK_F_FLUSH: u64 = 1 << 9;

/// Request header, first 16 device-readable bytes: type u32, reserved u32, sector u64. The
/// status byte is the last byte of the last device-writable segment.
const REQUEST_HEADER_SIZE: usize = 16;
const HEADER_SECTOR_OFFSET: usize = 8;

/// Request types (`VIRTIO_BLK_T_*`).
const REQUEST_IN: u32 = 0;
const REQUEST_OUT: u32 = 1;
const REQUEST_FLUSH: u32 = 4;
const REQUEST_GET_ID: u32 = 8;

/// Status byte values (`VIRTIO_BLK_S_*`).
const STATUS_OK: u8 = 0;
const STATUS_IOERR: u8 = 1;
const STATUS_UNSUPP: u8 = 2;

/// Length of the string returned by GET_ID (`VIRTIO_BLK_ID_BYTES`).
const ID_BYTES: usize = 20;
const ID_STRING: &[u8] = b"uemulator";

/// Config space (`struct virtio_blk_config`): capacity u64 at 0, seg_max u32 at 12, blk_size
/// u32 at 20; the rest (size_max, geometry, ...) reads as zero.
const CONFIG_CAPACITY_OFFSET: usize = 0;
const CONFIG_SEG_MAX_OFFSET: usize = 12;
const CONFIG_BLK_SIZE_OFFSET: usize = 20;
const CONFIG_SIZE: usize = 24;
/// Same value QEMU reports by default (a 128-entry queue minus the header and status descriptors).
const SEG_MAX: u32 = 126;

/// Cap on the data bytes moved by one request, bounding host allocation for a hostile descriptor.
const MAX_REQUEST_BYTES: usize = 16 << 20;

pub(crate) struct VirtioBlk {
    disk: Disk,
    id: [u8; ID_BYTES],
}

impl VirtioBlk {
    pub fn new(disk: Disk) -> Self {
        let mut id = [0u8; ID_BYTES];
        id[..ID_STRING.len()].copy_from_slice(ID_STRING);
        VirtioBlk { disk, id }
    }

    fn capacity(&self) -> u64 {
        self.disk.backend.len() / SECTOR_SIZE
    }

    /// Executes one request synchronously and returns the number of bytes written into the
    /// chain including the status byte (the used-ring `len`). A chain too short to hold a
    /// header and a status byte is completed with length 0 and no status.
    fn request(&mut self, mem: &mut GuestMem, chain: &Chain) -> u32 {
        let readable = chain.read_all(mem, MAX_REQUEST_BYTES + REQUEST_HEADER_SIZE);
        let writable_bytes = chain.writable();
        if readable.len() < REQUEST_HEADER_SIZE || writable_bytes == 0 {
            return 0;
        }
        let request_type = u32::from_le_bytes(readable[0..4].try_into().unwrap());
        let sector = u64::from_le_bytes(
            readable[HEADER_SECTOR_OFFSET..HEADER_SECTOR_OFFSET + 8]
                .try_into()
                .unwrap(),
        );
        // Everything writable except the trailing status byte is data for the driver.
        let data_capacity = writable_bytes - 1;
        let mut written = 0usize;
        let status: u8 = match request_type {
            REQUEST_IN => {
                let offset = sector.saturating_mul(SECTOR_SIZE);
                let len = data_capacity.min(MAX_REQUEST_BYTES);
                if offset
                    .checked_add(len as u64)
                    .is_none_or(|end| end > self.disk.backend.len())
                {
                    STATUS_IOERR
                } else {
                    let mut buf = vec![0u8; len];
                    match self.disk.backend.read_at(offset, &mut buf) {
                        Ok(()) => {
                            written = chain.write_at(mem, 0, &buf);
                            STATUS_OK
                        }
                        Err(_) => STATUS_IOERR,
                    }
                }
            }
            REQUEST_OUT => {
                let data = &readable[REQUEST_HEADER_SIZE..];
                let offset = sector.saturating_mul(SECTOR_SIZE);
                if self.disk.backend.read_only()
                    || offset
                        .checked_add(data.len() as u64)
                        .is_none_or(|end| end > self.disk.backend.len())
                {
                    STATUS_IOERR
                } else if self.disk.backend.write_at(offset, data).is_ok() {
                    STATUS_OK
                } else {
                    STATUS_IOERR
                }
            }
            REQUEST_FLUSH => {
                if self.disk.backend.flush().is_ok() {
                    STATUS_OK
                } else {
                    STATUS_IOERR
                }
            }
            REQUEST_GET_ID => {
                written = chain.write_at(mem, 0, &self.id[..data_capacity.min(ID_BYTES)]);
                STATUS_OK
            }
            _ => STATUS_UNSUPP,
        };
        chain.write_at(mem, data_capacity, &[status]);
        (written + 1) as u32
    }
}

impl VirtioDev for VirtioBlk {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_BLOCK
    }

    fn features(&self) -> u64 {
        let mut features = VIRTIO_BLK_F_SEG_MAX | VIRTIO_BLK_F_BLK_SIZE | VIRTIO_BLK_F_FLUSH;
        if self.disk.backend.read_only() {
            features |= VIRTIO_BLK_F_RO;
        }
        features
    }

    fn num_queues(&self) -> usize {
        1
    }

    fn config_read(&self, off: usize) -> u8 {
        let mut config = [0u8; CONFIG_SIZE];
        config[CONFIG_CAPACITY_OFFSET..CONFIG_CAPACITY_OFFSET + 8]
            .copy_from_slice(&self.capacity().to_le_bytes());
        config[CONFIG_SEG_MAX_OFFSET..CONFIG_SEG_MAX_OFFSET + 4]
            .copy_from_slice(&SEG_MAX.to_le_bytes());
        config[CONFIG_BLK_SIZE_OFFSET..CONFIG_BLK_SIZE_OFFSET + 4]
            .copy_from_slice(&(SECTOR_SIZE as u32).to_le_bytes());
        config.get(off).copied().unwrap_or(0)
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
    use super::super::virtio::{DESC_F_NEXT, DESC_F_WRITE};
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
        ram[hdr + HEADER_SECTOR_OFFSET..hdr + REQUEST_HEADER_SIZE]
            .copy_from_slice(&sector.to_le_bytes());
        put_desc(
            ram,
            DESC,
            0,
            hdr as u64,
            REQUEST_HEADER_SIZE as u32,
            DESC_F_NEXT,
            1,
        );
        let mut next = 1u16;
        if let Some(d) = data {
            ram[0x5000..0x5000 + d.len()].copy_from_slice(d);
            put_desc(ram, DESC, 1, 0x5000, d.len() as u32, DESC_F_NEXT, 2);
            next = 2;
        } else if dlen > 0 {
            put_desc(ram, DESC, 1, 0x5000, dlen, DESC_F_NEXT | DESC_F_WRITE, 2);
            next = 2;
        }
        put_desc(ram, DESC, next, 0x6000, 1, DESC_F_WRITE, 0);
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
            submit(
                &mut dev,
                &mut q,
                &mut ram,
                REQUEST_OUT,
                2,
                Some(&payload),
                0
            ),
            STATUS_OK
        );
        assert_eq!(
            submit(&mut dev, &mut q, &mut ram, REQUEST_FLUSH, 0, None, 0),
            STATUS_OK
        );
        ram[0x5000..0x5400].fill(0);
        assert_eq!(
            submit(&mut dev, &mut q, &mut ram, REQUEST_IN, 2, None, 1024),
            STATUS_OK
        );
        assert_eq!(&ram[0x5000..0x5400], &payload[..]);
        assert_eq!(used_idx(&ram, USED), 3);
        assert_eq!(
            submit(&mut dev, &mut q, &mut ram, REQUEST_IN, 100, None, 512),
            STATUS_IOERR
        );
        assert_eq!(
            submit(&mut dev, &mut q, &mut ram, 99, 0, None, 0),
            STATUS_UNSUPP
        );
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
        assert_ne!(dev.features() & VIRTIO_BLK_F_RO, 0);
        let mut q = setup_queue(&mut ram, DESC, AVAIL, USED, 16);
        assert_eq!(
            submit(
                &mut dev,
                &mut q,
                &mut ram,
                REQUEST_OUT,
                0,
                Some(&[1u8; 512]),
                0
            ),
            STATUS_IOERR
        );
    }
}
