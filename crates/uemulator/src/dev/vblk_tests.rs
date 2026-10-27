use super::super::virtio::testutil::*;
use super::super::virtio::{DESC_F_NEXT, DESC_F_WRITE};
use super::*;
use crate::dev::guestmem::test_mem;
use std::io;
use std::sync::{Arc, Mutex};

const DESC: u64 = 0x1000;
const AVAIL: u64 = 0x2000;
const USED: u64 = 0x3000;
const HEADER_ADDR: u64 = 0x4000;
const STATUS_ADDR: u64 = 0x6000;
const DATA_OUT: u64 = 0x1_0000;
const QUEUE_SIZE: u16 = 16;
const STATUS_SENTINEL: u8 = 0xff;
const SMALL_RAM: usize = 0x4_0000;
const SMALL_DATA_IN: u64 = 0x2_0000;

struct Outcome {
    status: u8,
    used_len: u32,
}

/// One device, one queue and a RAM image; each `run` builds the usual header / data / status chain.
struct Rig {
    ram: Vec<u8>,
    dev: VirtioBlk,
    queue: Queue,
    data_in: u64,
}

impl Rig {
    fn new(disk: Disk) -> Rig {
        Rig::with_ram(disk, SMALL_RAM, SMALL_DATA_IN)
    }

    fn with_ram(disk: Disk, ram_bytes: usize, data_in: u64) -> Rig {
        let mut ram = vec![0u8; ram_bytes];
        let queue = setup_queue(&mut ram, DESC, AVAIL, USED, QUEUE_SIZE);
        Rig {
            ram,
            dev: VirtioBlk::new(disk),
            queue,
            data_in,
        }
    }

    fn memory(bytes: usize) -> Rig {
        Rig::new(Disk::memory(vec![0u8; bytes]))
    }

    /// `out_len` bytes of device-readable data are taken from RAM at `DATA_OUT`; `in_len` bytes of
    /// device-writable space are posted at `data_in`.
    fn run(&mut self, ty: u32, sector: u64, out_len: usize, in_len: usize) -> Outcome {
        self.ram[HEADER_ADDR as usize..HEADER_ADDR as usize + 4].copy_from_slice(&ty.to_le_bytes());
        self.ram[HEADER_ADDR as usize + HEADER_SECTOR_OFFSET
            ..HEADER_ADDR as usize + REQUEST_HEADER_SIZE]
            .copy_from_slice(&sector.to_le_bytes());
        let mut index = 0u16;
        let mut link = |ram: &mut [u8], addr: u64, len: u32, flags: u16, last: bool| {
            let flags = if last { flags } else { flags | DESC_F_NEXT };
            put_desc(ram, DESC, index, addr, len, flags, index + 1);
            index += 1;
        };
        link(
            &mut self.ram,
            HEADER_ADDR,
            REQUEST_HEADER_SIZE as u32,
            0,
            false,
        );
        if out_len > 0 {
            link(&mut self.ram, DATA_OUT, out_len as u32, 0, false);
        }
        if in_len > 0 {
            link(
                &mut self.ram,
                self.data_in,
                in_len as u32,
                DESC_F_WRITE,
                false,
            );
        }
        link(&mut self.ram, STATUS_ADDR, 1, DESC_F_WRITE, true);
        self.ram[STATUS_ADDR as usize] = STATUS_SENTINEL;
        post(&mut self.ram, AVAIL, QUEUE_SIZE, 0);
        let before = used_idx(&self.ram, USED);
        let mut mem = test_mem(&mut self.ram);
        assert!(
            self.dev
                .kick(0, std::slice::from_mut(&mut self.queue), &mut mem)
        );
        let after = used_idx(&self.ram, USED);
        assert_eq!(after, before.wrapping_add(1), "exactly one completion");
        let (id, used_len) = used_entry(&self.ram, USED, before as usize % QUEUE_SIZE as usize);
        assert_eq!(id, 0, "used id is the head descriptor");
        Outcome {
            status: self.ram[STATUS_ADDR as usize],
            used_len,
        }
    }

    fn put_out(&mut self, data: &[u8]) {
        self.ram[DATA_OUT as usize..DATA_OUT as usize + data.len()].copy_from_slice(data);
    }

    fn data_in(&self, len: usize) -> &[u8] {
        &self.ram[self.data_in as usize..self.data_in as usize + len]
    }
}

/// Backend that records calls, can be read-only, and can fail on demand.
#[derive(Default)]
struct Log {
    reads: Vec<(u64, usize)>,
    writes: Vec<(u64, Vec<u8>)>,
    flushes: u32,
}

struct Scripted {
    len: u64,
    read_only: bool,
    fail_io: bool,
    log: Arc<Mutex<Log>>,
}

impl Scripted {
    fn new(len: u64) -> (Scripted, Arc<Mutex<Log>>) {
        let log = Arc::new(Mutex::new(Log::default()));
        (
            Scripted {
                len,
                read_only: false,
                fail_io: false,
                log: log.clone(),
            },
            log,
        )
    }
}

impl crate::disk::BlockBackend for Scripted {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.log.lock().unwrap().reads.push((offset, buf.len()));
        if self.fail_io {
            return Err(io::Error::other("injected"));
        }
        buf.fill(0x5a);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        self.log
            .lock()
            .unwrap()
            .writes
            .push((offset, data.to_vec()));
        if self.fail_io {
            return Err(io::Error::other("injected"));
        }
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.log.lock().unwrap().flushes += 1;
        if self.fail_io {
            return Err(io::Error::other("injected"));
        }
        Ok(())
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
}

#[test]
fn out_request_over_the_size_limit_fails_instead_of_truncating() {
    let disk_bytes = 20 << 20;
    let mut rig = Rig::with_ram(Disk::memory(vec![0u8; disk_bytes]), 40 << 20, 30 << 20);
    let too_big = MAX_REQUEST_BYTES + SECTOR_SIZE as usize;
    let outcome = rig.run(REQUEST_OUT, 0, too_big, 0);
    assert_eq!(outcome.status, STATUS_IOERR);
}

#[test]
fn in_request_over_the_size_limit_fails_instead_of_truncating() {
    let disk_bytes = 20 << 20;
    let mut rig = Rig::with_ram(Disk::memory(vec![0u8; disk_bytes]), 48 << 20, 20 << 20);
    let too_big = MAX_REQUEST_BYTES + SECTOR_SIZE as usize;
    let outcome = rig.run(REQUEST_IN, 0, 0, too_big);
    assert_eq!(outcome.status, STATUS_IOERR);
    assert_eq!(outcome.used_len, 1, "only the status byte was written");
}

#[test]
fn out_request_with_partial_sector_fails() {
    let mut rig = Rig::memory(8192);
    let outcome = rig.run(REQUEST_OUT, 0, 100, 0);
    assert_eq!(outcome.status, STATUS_IOERR);
}

#[test]
fn in_request_with_partial_sector_buffer_fails() {
    let mut rig = Rig::memory(8192);
    // 100 data bytes plus the status byte: the data length is not a sector multiple.
    let outcome = rig.run(REQUEST_IN, 0, 0, 100);
    assert_eq!(outcome.status, STATUS_IOERR);
}

const SECTOR: usize = SECTOR_SIZE as usize;
const REQUEST_DISCARD: u32 = 11;
const REQUEST_WRITE_ZEROES: u32 = 13;
const REQUEST_SECURE_ERASE: u32 = 14;

impl Rig {
    /// Posts one chain made of the given (address, length, device-writable) segments, kicks, and
    /// returns the used length that was reported.
    fn run_segments(&mut self, segments: &[(u64, u32, bool)]) -> u32 {
        for (index, &(addr, len, writable)) in segments.iter().enumerate() {
            let last = index + 1 == segments.len();
            let flags =
                (if writable { DESC_F_WRITE } else { 0 }) | if last { 0 } else { DESC_F_NEXT };
            put_desc(
                &mut self.ram,
                DESC,
                index as u16,
                addr,
                len,
                flags,
                index as u16 + 1,
            );
        }
        post(&mut self.ram, AVAIL, QUEUE_SIZE, 0);
        let before = used_idx(&self.ram, USED);
        let mut mem = test_mem(&mut self.ram);
        self.dev
            .kick(0, std::slice::from_mut(&mut self.queue), &mut mem);
        assert_eq!(used_idx(&self.ram, USED), before.wrapping_add(1));
        used_entry(&self.ram, USED, before as usize % QUEUE_SIZE as usize).1
    }

    fn set_header(&mut self, ty: u32, sector: u64) {
        let at = HEADER_ADDR as usize;
        self.ram[at..at + 4].copy_from_slice(&ty.to_le_bytes());
        self.ram[at + 4..at + 8].copy_from_slice(&0u32.to_le_bytes());
        self.ram[at + HEADER_SECTOR_OFFSET..at + REQUEST_HEADER_SIZE]
            .copy_from_slice(&sector.to_le_bytes());
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

#[test]
fn write_then_read_roundtrip_and_flush() {
    let mut rig = Rig::memory(8192);
    let payload = pattern(1024, 1);
    rig.put_out(&payload);
    let write = rig.run(REQUEST_OUT, 2, 1024, 0);
    assert_eq!((write.status, write.used_len), (STATUS_OK, 1));
    let flush = rig.run(REQUEST_FLUSH, 0, 0, 0);
    assert_eq!((flush.status, flush.used_len), (STATUS_OK, 1));
    let read = rig.run(REQUEST_IN, 2, 0, 1024);
    assert_eq!((read.status, read.used_len), (STATUS_OK, 1025));
    assert_eq!(rig.data_in(1024), &payload[..]);
    assert_eq!(used_idx(&rig.ram, USED), 3);
}

#[test]
fn writes_land_at_the_sector_offset_and_nowhere_else() {
    let (backend, log) = Scripted::new(1 << 20);
    let mut rig = Rig::new(Disk::from_backend(backend));
    rig.put_out(&pattern(SECTOR, 9));
    rig.run(REQUEST_OUT, 7, SECTOR, 0);
    let log = log.lock().unwrap();
    assert_eq!(log.writes.len(), 1);
    assert_eq!(log.writes[0].0, 7 * SECTOR_SIZE);
    assert_eq!(log.writes[0].1, pattern(SECTOR, 9));
    assert!(log.reads.is_empty());
}

#[test]
fn reads_use_the_sector_offset_and_buffer_length() {
    let (backend, log) = Scripted::new(1 << 20);
    let mut rig = Rig::new(Disk::from_backend(backend));
    let outcome = rig.run(REQUEST_IN, 3, 0, 2 * SECTOR);
    assert_eq!(outcome.status, STATUS_OK);
    assert_eq!(
        log.lock().unwrap().reads,
        vec![(3 * SECTOR_SIZE, 2 * SECTOR)]
    );
    assert!(rig.data_in(2 * SECTOR).iter().all(|&b| b == 0x5a));
}

#[test]
fn zero_length_read_and_write_succeed_without_touching_data() {
    let mut rig = Rig::memory(4096);
    assert_eq!(rig.run(REQUEST_IN, 0, 0, 0).status, STATUS_OK);
    assert_eq!(rig.run(REQUEST_OUT, 0, 0, 0).status, STATUS_OK);
}

#[test]
fn access_to_the_last_sector_succeeds_and_one_past_it_fails() {
    let mut rig = Rig::memory(8192);
    rig.put_out(&pattern(SECTOR, 3));
    assert_eq!(rig.run(REQUEST_OUT, 15, SECTOR, 0).status, STATUS_OK);
    assert_eq!(rig.run(REQUEST_IN, 15, 0, SECTOR).status, STATUS_OK);
    assert_eq!(rig.data_in(SECTOR), &pattern(SECTOR, 3)[..]);
    assert_eq!(rig.run(REQUEST_OUT, 16, SECTOR, 0).status, STATUS_IOERR);
    assert_eq!(rig.run(REQUEST_IN, 16, 0, SECTOR).status, STATUS_IOERR);
}

#[test]
fn request_straddling_the_end_of_the_disk_fails_whole() {
    let (backend, log) = Scripted::new(8192);
    let mut rig = Rig::new(Disk::from_backend(backend));
    assert_eq!(rig.run(REQUEST_IN, 15, 0, 2 * SECTOR).status, STATUS_IOERR);
    assert_eq!(rig.run(REQUEST_OUT, 15, 2 * SECTOR, 0).status, STATUS_IOERR);
    let log = log.lock().unwrap();
    assert!(
        log.reads.is_empty() && log.writes.is_empty(),
        "the backend must not be touched"
    );
}

#[test]
fn sector_numbers_that_overflow_the_byte_offset_fail_without_panicking() {
    let mut rig = Rig::memory(8192);
    for sector in [
        1u64 << 55,
        1 << 63,
        u64::MAX,
        u64::MAX / SECTOR_SIZE,
        u64::MAX / SECTOR_SIZE + 1,
    ] {
        assert_eq!(
            rig.run(REQUEST_IN, sector, 0, SECTOR).status,
            STATUS_IOERR,
            "sector {sector:#x}"
        );
        assert_eq!(
            rig.run(REQUEST_OUT, sector, SECTOR, 0).status,
            STATUS_IOERR,
            "sector {sector:#x}"
        );
    }
}

#[test]
fn failed_request_reports_only_the_status_byte_as_written() {
    let mut rig = Rig::memory(4096);
    let outcome = rig.run(REQUEST_IN, 100, 0, SECTOR);
    assert_eq!(outcome.used_len, 1);
    assert!(
        rig.data_in(SECTOR).iter().all(|&b| b == 0),
        "no data is written on failure"
    );
}

#[test]
fn flush_reaches_the_backend_and_reports_ok() {
    let (backend, log) = Scripted::new(4096);
    let mut rig = Rig::new(Disk::from_backend(backend));
    assert_eq!(rig.run(REQUEST_FLUSH, 0, 0, 0).status, STATUS_OK);
    assert_eq!(log.lock().unwrap().flushes, 1);
}

#[test]
fn flush_failure_is_reported_as_ioerr() {
    let (mut backend, _log) = Scripted::new(4096);
    backend.fail_io = true;
    let mut rig = Rig::new(Disk::from_backend(backend));
    assert_eq!(rig.run(REQUEST_FLUSH, 0, 0, 0).status, STATUS_IOERR);
}

#[test]
fn backend_io_errors_become_ioerr() {
    let (mut backend, _log) = Scripted::new(4096);
    backend.fail_io = true;
    let mut rig = Rig::new(Disk::from_backend(backend));
    assert_eq!(rig.run(REQUEST_IN, 0, 0, SECTOR).status, STATUS_IOERR);
    assert_eq!(rig.run(REQUEST_OUT, 0, SECTOR, 0).status, STATUS_IOERR);
}

#[test]
fn unsupported_request_types_get_unsupp_and_touch_nothing() {
    let (backend, log) = Scripted::new(1 << 20);
    let mut rig = Rig::new(Disk::from_backend(backend));
    for ty in [
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        REQUEST_DISCARD,
        REQUEST_WRITE_ZEROES,
        REQUEST_SECURE_ERASE,
        99,
        0x8000_0000,
        u32::MAX,
    ] {
        let outcome = rig.run(ty, 0, 0, SECTOR);
        assert_eq!(outcome.status, STATUS_UNSUPP, "type {ty}");
        assert_eq!(outcome.used_len, 1, "type {ty}");
    }
    let log = log.lock().unwrap();
    assert!(log.reads.is_empty() && log.writes.is_empty() && log.flushes == 0);
    assert!(rig.data_in(SECTOR).iter().all(|&b| b == 0));
}

#[test]
fn unsupported_request_with_out_data_is_still_answered() {
    let mut rig = Rig::memory(4096);
    assert_eq!(rig.run(REQUEST_DISCARD, 0, 16, 0).status, STATUS_UNSUPP);
}

#[test]
fn get_id_returns_the_padded_identifier() {
    let mut rig = Rig::memory(4096);
    let outcome = rig.run(REQUEST_GET_ID, 0, 0, ID_BYTES);
    assert_eq!(
        (outcome.status, outcome.used_len),
        (STATUS_OK, ID_BYTES as u32 + 1)
    );
    let mut expected = [0u8; ID_BYTES];
    expected[..ID_STRING.len()].copy_from_slice(ID_STRING);
    assert_eq!(rig.data_in(ID_BYTES), &expected);
}

#[test]
fn get_id_truncates_to_a_shorter_buffer_and_leaves_a_longer_one_alone() {
    let mut rig = Rig::memory(4096);
    let short = rig.run(REQUEST_GET_ID, 0, 0, 4);
    assert_eq!((short.status, short.used_len), (STATUS_OK, 5));
    assert_eq!(rig.data_in(4), b"uemu");
    let long = rig.run(REQUEST_GET_ID, 0, 0, 64);
    assert_eq!(long.used_len, ID_BYTES as u32 + 1);
    assert!(rig.data_in(64)[ID_BYTES..].iter().all(|&b| b == 0));
}

#[test]
fn get_id_with_no_data_buffer_only_writes_status() {
    let mut rig = Rig::memory(4096);
    let outcome = rig.run(REQUEST_GET_ID, 0, 0, 0);
    assert_eq!((outcome.status, outcome.used_len), (STATUS_OK, 1));
}

#[test]
fn read_only_disk_advertises_ro_and_rejects_writes_but_serves_reads_and_flush() {
    let (mut backend, log) = Scripted::new(8192);
    backend.read_only = true;
    let mut rig = Rig::new(Disk::from_backend(backend));
    assert_ne!(rig.dev.features() & VIRTIO_BLK_F_RO, 0);
    rig.put_out(&[1u8; SECTOR]);
    assert_eq!(rig.run(REQUEST_OUT, 0, SECTOR, 0).status, STATUS_IOERR);
    assert!(
        log.lock().unwrap().writes.is_empty(),
        "the backend never sees the write"
    );
    assert_eq!(rig.run(REQUEST_IN, 0, 0, SECTOR).status, STATUS_OK);
    assert_eq!(rig.run(REQUEST_FLUSH, 0, 0, 0).status, STATUS_OK);
}

#[test]
fn read_write_disk_does_not_advertise_ro() {
    let rig = Rig::memory(4096);
    assert_eq!(rig.dev.features() & VIRTIO_BLK_F_RO, 0);
}

#[test]
fn offered_features_are_seg_max_blk_size_and_flush() {
    let rig = Rig::memory(4096);
    assert_eq!(
        rig.dev.features(),
        VIRTIO_BLK_F_SEG_MAX | VIRTIO_BLK_F_BLK_SIZE | VIRTIO_BLK_F_FLUSH
    );
    assert_eq!(VIRTIO_BLK_F_SEG_MAX, 1 << 2);
    assert_eq!(VIRTIO_BLK_F_RO, 1 << 5);
    assert_eq!(VIRTIO_BLK_F_BLK_SIZE, 1 << 6);
    assert_eq!(VIRTIO_BLK_F_FLUSH, 1 << 9);
}

#[test]
fn device_identity_is_block_with_one_queue() {
    let rig = Rig::memory(4096);
    assert_eq!(rig.dev.device_type(), 2);
    assert_eq!(rig.dev.num_queues(), 1);
}

fn config_bytes(dev: &VirtioBlk, len: usize) -> Vec<u8> {
    (0..len).map(|off| dev.config_read(off)).collect()
}

#[test]
fn config_space_reports_capacity_seg_max_and_block_size() {
    let rig = Rig::memory(8192);
    let config = config_bytes(&rig.dev, 32);
    assert_eq!(u64::from_le_bytes(config[0..8].try_into().unwrap()), 16);
    assert_eq!(
        u32::from_le_bytes(config[8..12].try_into().unwrap()),
        0,
        "size_max not offered"
    );
    assert_eq!(u32::from_le_bytes(config[12..16].try_into().unwrap()), 126);
    assert_eq!(&config[16..20], &[0; 4], "geometry not offered");
    assert_eq!(u32::from_le_bytes(config[20..24].try_into().unwrap()), 512);
    assert_eq!(&config[24..], &[0; 8], "bytes past blk_size read zero");
}

#[test]
fn capacity_counts_whole_sectors_only() {
    for (bytes, sectors) in [
        (0usize, 0u64),
        (511, 0),
        (512, 1),
        (1023, 1),
        (1024, 2),
        (1 << 20, 2048),
    ] {
        let rig = Rig::memory(bytes);
        assert_eq!(rig.dev.capacity(), sectors, "{bytes} bytes");
        assert_eq!(rig.dev.config_read(0) as u64, sectors & 0xff);
    }
}

#[test]
fn capacity_uses_all_eight_bytes_for_huge_disks() {
    let (backend, _log) = Scripted::new(1 << 50);
    let dev = VirtioBlk::new(Disk::from_backend(backend));
    let config = config_bytes(&dev, 8);
    assert_eq!(u64::from_le_bytes(config.try_into().unwrap()), 1 << 41);
}

#[test]
fn config_reads_far_past_the_structure_return_zero() {
    let rig = Rig::memory(4096);
    for off in [24, 100, 255, 4096, usize::MAX] {
        assert_eq!(rig.dev.config_read(off), 0, "offset {off}");
    }
}

#[test]
fn config_writes_are_ignored() {
    let mut rig = Rig::memory(4096);
    let before = config_bytes(&rig.dev, 24);
    for off in 0..24 {
        rig.dev.config_write(off, 0xff);
    }
    assert_eq!(config_bytes(&rig.dev, 24), before);
}

#[test]
fn header_split_across_two_descriptors_is_reassembled() {
    let mut rig = Rig::memory(4096);
    rig.set_header(REQUEST_IN, 1);
    rig.ram[STATUS_ADDR as usize] = STATUS_SENTINEL;
    let used_len = rig.run_segments(&[
        (HEADER_ADDR, 4, false),
        (HEADER_ADDR + 4, 12, false),
        (rig.data_in, SECTOR as u32, true),
        (STATUS_ADDR, 1, true),
    ]);
    assert_eq!(used_len, SECTOR as u32 + 1);
    assert_eq!(rig.ram[STATUS_ADDR as usize], STATUS_OK);
}

#[test]
fn out_data_split_across_descriptors_is_concatenated() {
    let (backend, log) = Scripted::new(1 << 20);
    let mut rig = Rig::new(Disk::from_backend(backend));
    let payload = pattern(2 * SECTOR, 5);
    rig.put_out(&payload);
    rig.set_header(REQUEST_OUT, 1);
    let used_len = rig.run_segments(&[
        (HEADER_ADDR, REQUEST_HEADER_SIZE as u32, false),
        (DATA_OUT, 300, false),
        (DATA_OUT + 300, 400, false),
        (DATA_OUT + 700, 324, false),
        (STATUS_ADDR, 1, true),
    ]);
    assert_eq!(used_len, 1);
    assert_eq!(rig.ram[STATUS_ADDR as usize], STATUS_OK);
    assert_eq!(log.lock().unwrap().writes, vec![(SECTOR_SIZE, payload)]);
}

#[test]
fn in_data_split_across_descriptors_is_scattered_in_order() {
    let mut rig = Rig::memory(8192);
    let payload = pattern(2 * SECTOR, 11);
    rig.put_out(&payload);
    assert_eq!(rig.run(REQUEST_OUT, 4, 2 * SECTOR, 0).status, STATUS_OK);
    rig.set_header(REQUEST_IN, 4);
    let base = rig.data_in;
    let used_len = rig.run_segments(&[
        (HEADER_ADDR, REQUEST_HEADER_SIZE as u32, false),
        (base, 100, true),
        (base + 0x1000, 500, true),
        (base + 0x2000, 424, true),
        (STATUS_ADDR, 1, true),
    ]);
    assert_eq!(used_len, 2 * SECTOR as u32 + 1);
    let mut gathered = rig.ram[base as usize..base as usize + 100].to_vec();
    gathered.extend_from_slice(&rig.ram[base as usize + 0x1000..base as usize + 0x1000 + 500]);
    gathered.extend_from_slice(&rig.ram[base as usize + 0x2000..base as usize + 0x2000 + 424]);
    assert_eq!(gathered, payload);
}

#[test]
fn status_byte_is_the_last_byte_of_the_last_writable_descriptor() {
    let mut rig = Rig::memory(4096);
    rig.set_header(REQUEST_IN, 0);
    let base = rig.data_in;
    let used_len = rig.run_segments(&[
        (HEADER_ADDR, REQUEST_HEADER_SIZE as u32, false),
        (base, SECTOR as u32 + 1, true),
    ]);
    assert_eq!(used_len, SECTOR as u32 + 1);
    assert_eq!(
        rig.ram[base as usize + SECTOR],
        STATUS_OK,
        "status follows the data in the same descriptor"
    );
}

#[test]
fn status_goes_to_the_final_descriptor_even_when_it_is_larger_than_one_byte() {
    let mut rig = Rig::memory(4096);
    rig.set_header(REQUEST_FLUSH, 0);
    let used_len = rig.run_segments(&[
        (HEADER_ADDR, REQUEST_HEADER_SIZE as u32, false),
        (STATUS_ADDR, 4, true),
    ]);
    assert_eq!(used_len, 1);
    assert_eq!(rig.ram[STATUS_ADDR as usize + 3], STATUS_OK);
}

#[test]
fn chain_without_a_writable_descriptor_is_completed_with_length_zero() {
    let mut rig = Rig::memory(4096);
    rig.set_header(REQUEST_FLUSH, 0);
    let used_len = rig.run_segments(&[(HEADER_ADDR, REQUEST_HEADER_SIZE as u32, false)]);
    assert_eq!(used_len, 0);
}

#[test]
fn chain_shorter_than_a_header_is_completed_with_length_zero_and_no_status() {
    let mut rig = Rig::memory(4096);
    rig.set_header(REQUEST_FLUSH, 0);
    rig.ram[STATUS_ADDR as usize] = STATUS_SENTINEL;
    let used_len = rig.run_segments(&[(HEADER_ADDR, 15, false), (STATUS_ADDR, 1, true)]);
    assert_eq!(used_len, 0);
    assert_eq!(rig.ram[STATUS_ADDR as usize], STATUS_SENTINEL);
}

#[test]
fn every_posted_request_is_completed_by_one_kick() {
    let mut rig = Rig::memory(8192);
    // Three flush chains in separate descriptor triples.
    for request in 0..3u16 {
        let base = request * 4;
        let header = HEADER_ADDR + 0x100 * request as u64;
        rig.ram[header as usize..header as usize + 4].copy_from_slice(&REQUEST_FLUSH.to_le_bytes());
        put_desc(
            &mut rig.ram,
            DESC,
            base,
            header,
            REQUEST_HEADER_SIZE as u32,
            DESC_F_NEXT,
            base + 1,
        );
        put_desc(
            &mut rig.ram,
            DESC,
            base + 1,
            STATUS_ADDR + request as u64,
            1,
            DESC_F_WRITE,
            0,
        );
        rig.ram[STATUS_ADDR as usize + request as usize] = STATUS_SENTINEL;
        post(&mut rig.ram, AVAIL, QUEUE_SIZE, base);
    }
    let mut mem = test_mem(&mut rig.ram);
    assert!(
        rig.dev
            .kick(0, std::slice::from_mut(&mut rig.queue), &mut mem)
    );
    assert_eq!(used_idx(&rig.ram, USED), 3);
    for request in 0..3usize {
        assert_eq!(rig.ram[STATUS_ADDR as usize + request], STATUS_OK);
        assert_eq!(used_entry(&rig.ram, USED, request), (request as u32 * 4, 1));
    }
}

#[test]
fn kick_with_nothing_posted_reports_no_work() {
    let mut rig = Rig::memory(4096);
    let mut mem = test_mem(&mut rig.ram);
    assert!(
        !rig.dev
            .kick(0, std::slice::from_mut(&mut rig.queue), &mut mem)
    );
    assert_eq!(used_idx(&rig.ram, USED), 0);
}

#[test]
fn request_header_reserved_field_is_ignored() {
    let mut rig = Rig::memory(4096);
    rig.set_header(REQUEST_FLUSH, 0);
    rig.ram[HEADER_ADDR as usize + 4..HEADER_ADDR as usize + 8]
        .copy_from_slice(&0xdead_beefu32.to_le_bytes());
    let used_len = rig.run_segments(&[
        (HEADER_ADDR, REQUEST_HEADER_SIZE as u32, false),
        (STATUS_ADDR, 1, true),
    ]);
    assert_eq!(used_len, 1);
    assert_eq!(rig.ram[STATUS_ADDR as usize], STATUS_OK);
}

#[test]
fn transfer_of_exactly_the_size_limit_is_allowed() {
    let (backend, log) = Scripted::new((MAX_REQUEST_BYTES + 2 * SECTOR) as u64);
    let mut rig = Rig::with_ram(Disk::from_backend(backend), 40 << 20, 20 << 20);
    let outcome = rig.run(REQUEST_IN, 1, 0, MAX_REQUEST_BYTES);
    assert_eq!(
        (outcome.status, outcome.used_len),
        (STATUS_OK, MAX_REQUEST_BYTES as u32 + 1)
    );
    assert_eq!(rig.data_in(MAX_REQUEST_BYTES)[MAX_REQUEST_BYTES - 1], 0x5a);
    let outcome = rig.run(REQUEST_OUT, 1, MAX_REQUEST_BYTES, 0);
    assert_eq!(outcome.status, STATUS_OK);
    let log = log.lock().unwrap();
    assert_eq!(log.reads, vec![(SECTOR_SIZE, MAX_REQUEST_BYTES)]);
    assert_eq!(log.writes[0].1.len(), MAX_REQUEST_BYTES);
}

#[test]
fn oversize_request_never_reaches_the_backend() {
    let (backend, log) = Scripted::new(64 << 20);
    let mut rig = Rig::with_ram(Disk::from_backend(backend), 48 << 20, 24 << 20);
    let too_big = MAX_REQUEST_BYTES + SECTOR;
    assert_eq!(rig.run(REQUEST_IN, 0, 0, too_big).status, STATUS_IOERR);
    assert_eq!(rig.run(REQUEST_OUT, 0, too_big, 0).status, STATUS_IOERR);
    let log = log.lock().unwrap();
    assert!(log.reads.is_empty() && log.writes.is_empty());
}

#[test]
fn descriptor_pointing_outside_ram_breaks_the_queue_instead_of_panicking() {
    let mut rig = Rig::memory(4096);
    rig.set_header(REQUEST_IN, 0);
    put_desc(
        &mut rig.ram,
        DESC,
        0,
        HEADER_ADDR,
        REQUEST_HEADER_SIZE as u32,
        DESC_F_NEXT,
        1,
    );
    put_desc(
        &mut rig.ram,
        DESC,
        1,
        SMALL_RAM as u64 - 8,
        SECTOR as u32,
        DESC_F_WRITE,
        0,
    );
    post(&mut rig.ram, AVAIL, QUEUE_SIZE, 0);
    let mut mem = test_mem(&mut rig.ram);
    assert!(
        !rig.dev
            .kick(0, std::slice::from_mut(&mut rig.queue), &mut mem)
    );
    assert!(rig.queue.broken);
    assert_eq!(used_idx(&rig.ram, USED), 0);
}

#[test]
fn memory_disk_persists_across_requests_of_different_shapes() {
    let mut rig = Rig::memory(4 * SECTOR);
    rig.put_out(&pattern(SECTOR, 1));
    rig.run(REQUEST_OUT, 0, SECTOR, 0);
    rig.put_out(&pattern(SECTOR, 2));
    rig.run(REQUEST_OUT, 3, SECTOR, 0);
    assert_eq!(rig.run(REQUEST_IN, 0, 0, 4 * SECTOR).status, STATUS_OK);
    let all = rig.data_in(4 * SECTOR).to_vec();
    assert_eq!(&all[..SECTOR], &pattern(SECTOR, 1)[..]);
    assert!(all[SECTOR..3 * SECTOR].iter().all(|&b| b == 0));
    assert_eq!(&all[3 * SECTOR..], &pattern(SECTOR, 2)[..]);
}

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
