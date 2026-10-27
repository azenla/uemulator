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
/// A larger request is answered with IOERR; QEMU likewise fails what it cannot map.
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

    /// Offset of `sector` and whether `len` bytes from there stay inside the disk.
    fn range_in_disk(&self, sector: u64, len: usize) -> Option<u64> {
        let offset = sector.checked_mul(SECTOR_SIZE)?;
        let end = offset.checked_add(len as u64)?;
        (end <= self.disk.backend.len()).then_some(offset)
    }

    /// A data length the device will transfer: whole sectors, within `MAX_REQUEST_BYTES`.
    fn valid_transfer_len(len: usize) -> bool {
        len.is_multiple_of(SECTOR_SIZE as usize) && len <= MAX_REQUEST_BYTES
    }

    /// Executes one request synchronously and returns the number of bytes written into the
    /// chain including the status byte (the used-ring `len`). A chain too short to hold a
    /// header and a status byte is completed with length 0 and no status.
    fn request(&mut self, mem: &mut GuestMem, chain: &Chain) -> u32 {
        let readable_bytes = chain.readable();
        let writable_bytes = chain.writable();
        if readable_bytes < REQUEST_HEADER_SIZE || writable_bytes == 0 {
            return 0;
        }
        let header = chain.read_all(mem, REQUEST_HEADER_SIZE);
        let request_type = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let sector = u64::from_le_bytes(
            header[HEADER_SECTOR_OFFSET..HEADER_SECTOR_OFFSET + 8]
                .try_into()
                .unwrap(),
        );
        // Everything writable except the trailing status byte is data for the driver.
        let in_len = writable_bytes - 1;
        let out_len = readable_bytes - REQUEST_HEADER_SIZE;
        let mut written = 0usize;
        // Lengths are checked before anything is allocated or read, so a request beyond the
        // limit is refused as a whole instead of being cut short and reported as done.
        let status: u8 = match request_type {
            REQUEST_IN => match self.range_in_disk(sector, in_len) {
                Some(offset) if Self::valid_transfer_len(in_len) => {
                    let mut buf = vec![0u8; in_len];
                    match self.disk.backend.read_at(offset, &mut buf) {
                        Ok(()) => {
                            written = chain.write_at(mem, 0, &buf);
                            STATUS_OK
                        }
                        Err(_) => STATUS_IOERR,
                    }
                }
                _ => STATUS_IOERR,
            },
            REQUEST_OUT => match self.range_in_disk(sector, out_len) {
                Some(offset)
                    if Self::valid_transfer_len(out_len) && !self.disk.backend.read_only() =>
                {
                    let data = chain.read_all(mem, REQUEST_HEADER_SIZE + out_len);
                    if self
                        .disk
                        .backend
                        .write_at(offset, &data[REQUEST_HEADER_SIZE..])
                        .is_ok()
                    {
                        STATUS_OK
                    } else {
                        STATUS_IOERR
                    }
                }
                _ => STATUS_IOERR,
            },
            REQUEST_FLUSH => {
                if self.disk.backend.flush().is_ok() {
                    STATUS_OK
                } else {
                    STATUS_IOERR
                }
            }
            REQUEST_GET_ID => {
                written = chain.write_at(mem, 0, &self.id[..in_len.min(ID_BYTES)]);
                STATUS_OK
            }
            _ => STATUS_UNSUPP,
        };
        chain.write_at(mem, in_len, &[status]);
        (written + 1) as u32
    }
}

impl VirtioDev for VirtioBlk {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_BLOCK
    }

    fn into_disk(self: Box<Self>) -> Option<Disk> {
        Some(self.disk)
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
#[path = "vblk_tests.rs"]
mod tests;
