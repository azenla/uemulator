use super::*;

const RAM_BYTES: usize = 4 * (1 << PAGE_SHIFT);
const PAGE_BYTES: u64 = 1 << PAGE_SHIFT;

fn flags(pages: &[usize]) -> Vec<u8> {
    let mut map = vec![0u8; RAM_BYTES >> PAGE_SHIFT];
    for &page in pages {
        map[page] = 1;
    }
    map
}

#[test]
fn write_then_read_roundtrips_bytes() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert!(mem.write(0x10, &[1, 2, 3, 4, 5]));
    let mut out = [0u8; 5];
    assert!(mem.read(0x10, &mut out));
    assert_eq!(out, [1, 2, 3, 4, 5]);
    assert_eq!(&ram[0x10..0x15], &[1, 2, 3, 4, 5]);
}

#[test]
fn integer_accessors_are_little_endian() {
    let mut ram = vec![0u8; RAM_BYTES];
    ram[0x100..0x108].copy_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    let mut mem = test_mem(&mut ram);
    assert_eq!(mem.read_u16(0x100), 0x0201);
    assert_eq!(mem.read_u32(0x100), 0x0403_0201);
    assert_eq!(mem.read_u64(0x100), 0x0807_0605_0403_0201);
    mem.write_u16(0x200, 0xbeef);
    mem.write_u32(0x210, 0xdead_beef);
    assert_eq!(&ram[0x200..0x202], &[0xef, 0xbe]);
    assert_eq!(&ram[0x210..0x214], &[0xef, 0xbe, 0xad, 0xde]);
}

#[test]
fn unaligned_accesses_work() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    mem.write_u32(0x3, 0x1122_3344);
    assert_eq!(mem.read_u32(0x3), 0x1122_3344);
    assert_eq!(mem.read_u16(0x4), 0x2233);
}

#[test]
fn read_ending_exactly_at_ram_end_succeeds() {
    let mut ram = vec![0u8; RAM_BYTES];
    ram[RAM_BYTES - 1] = 0x7f;
    let mem = test_mem(&mut ram);
    let mut byte = [0u8; 1];
    assert!(mem.read(RAM_BYTES as u64 - 1, &mut byte));
    assert_eq!(byte, [0x7f]);
    assert_eq!(mem.read_u32(RAM_BYTES as u64 - 4), 0x7f00_0000);
}

#[test]
fn read_crossing_ram_end_fails_without_touching_the_buffer() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mem = test_mem(&mut ram);
    let mut buf = [0xaau8; 4];
    assert!(!mem.read(RAM_BYTES as u64 - 3, &mut buf));
    assert_eq!(buf, [0xaa; 4]);
}

#[test]
fn read_starting_past_ram_fails() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mem = test_mem(&mut ram);
    let mut buf = [0u8; 1];
    assert!(!mem.read(RAM_BYTES as u64, &mut buf));
    assert!(!mem.read(u64::MAX, &mut buf));
}

#[test]
fn read_with_overflowing_range_fails_instead_of_panicking() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mem = test_mem(&mut ram);
    let mut buf = [0u8; 16];
    assert!(!mem.read(u64::MAX - 7, &mut buf));
    assert!(!mem.read(usize::MAX as u64 - 7, &mut buf));
}

#[test]
fn write_crossing_ram_end_writes_nothing() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert!(!mem.write(RAM_BYTES as u64 - 2, &[1, 2, 3, 4]));
    assert_eq!(&ram[RAM_BYTES - 4..], &[0, 0, 0, 0]);
}

#[test]
fn write_past_ram_and_overflow_fail() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert!(!mem.write(RAM_BYTES as u64, &[1]));
    assert!(!mem.write(u64::MAX, &[1, 2]));
    assert!(!mem.write(u64::MAX - 1, &[1, 2, 3]));
}

#[test]
fn empty_write_inside_ram_succeeds_and_outside_fails() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert!(mem.write(0, &[]));
    assert!(mem.write(RAM_BYTES as u64, &[]));
    assert!(!mem.write(RAM_BYTES as u64 + 1, &[]));
    assert!(mem.code_pages_written.is_empty());
}

#[test]
fn out_of_range_integer_reads_return_zero_and_writes_are_dropped() {
    let mut ram = vec![0xffu8; RAM_BYTES];
    let mut mem = test_mem(&mut ram);
    assert_eq!(mem.read_u16(RAM_BYTES as u64), 0);
    assert_eq!(mem.read_u32(RAM_BYTES as u64 - 2), 0);
    assert_eq!(mem.read_u64(u64::MAX), 0);
    mem.write_u32(RAM_BYTES as u64 - 2, 0);
    assert_eq!(&ram[RAM_BYTES - 4..], &[0xff; 4]);
}

#[test]
fn contains_is_inclusive_of_the_last_byte_and_overflow_safe() {
    let mut ram = vec![0u8; RAM_BYTES];
    let mem = test_mem(&mut ram);
    assert!(mem.contains(0, RAM_BYTES as u64));
    assert!(mem.contains(RAM_BYTES as u64, 0));
    assert!(mem.contains(RAM_BYTES as u64 - 1, 1));
    assert!(!mem.contains(RAM_BYTES as u64 - 1, 2));
    assert!(!mem.contains(RAM_BYTES as u64 + 1, 0));
    assert!(!mem.contains(u64::MAX, 2));
    assert!(!mem.contains(1, u64::MAX));
}

#[test]
fn write_to_unflagged_page_records_nothing() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[2]);
    let mut mem = GuestMem::new(&mut ram, &map);
    mem.write(0x10, &[1]);
    mem.write(PAGE_BYTES, &[1]);
    assert!(mem.code_pages_written.is_empty());
}

#[test]
fn write_to_flagged_page_records_that_page() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[1]);
    let mut mem = GuestMem::new(&mut ram, &map);
    mem.write(PAGE_BYTES + 8, &[1, 2]);
    assert_eq!(mem.code_pages_written, vec![1]);
}

#[test]
fn repeated_writes_to_a_flagged_page_are_listed_once() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[3]);
    let mut mem = GuestMem::new(&mut ram, &map);
    for offset in 0..4 {
        mem.write(3 * PAGE_BYTES + offset, &[9]);
    }
    assert_eq!(mem.code_pages_written, vec![3]);
}

#[test]
fn write_spanning_two_flagged_pages_records_both() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[0, 1]);
    let mut mem = GuestMem::new(&mut ram, &map);
    mem.write(PAGE_BYTES - 2, &[1, 2, 3, 4]);
    assert_eq!(mem.code_pages_written, vec![0, 1]);
}

#[test]
fn write_ending_exactly_at_page_boundary_does_not_touch_the_next_page() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[1]);
    let mut mem = GuestMem::new(&mut ram, &map);
    mem.write(PAGE_BYTES - 4, &[1, 2, 3, 4]);
    assert!(mem.code_pages_written.is_empty());
    mem.write(PAGE_BYTES, &[1]);
    assert_eq!(mem.code_pages_written, vec![1]);
}

#[test]
fn integer_writes_track_code_pages_too() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[2]);
    let mut mem = GuestMem::new(&mut ram, &map);
    mem.write_u16(2 * PAGE_BYTES, 1);
    mem.write_u32(2 * PAGE_BYTES + 4, 1);
    assert_eq!(mem.code_pages_written, vec![2]);
}

#[test]
fn failed_write_does_not_record_a_code_page() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[3]);
    let mut mem = GuestMem::new(&mut ram, &map);
    assert!(!mem.write(4 * PAGE_BYTES - 2, &[1, 2, 3, 4]));
    assert!(mem.code_pages_written.is_empty());
}

#[test]
fn reads_never_record_code_pages() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[0, 1, 2, 3]);
    let mem = GuestMem::new(&mut ram, &map);
    let mut buf = [0u8; 64];
    mem.read(0, &mut buf);
    assert!(mem.code_pages_written.is_empty());
}

#[test]
fn page_beyond_the_flag_map_is_treated_as_not_code() {
    let mut ram = vec![0u8; RAM_BYTES];
    let map = flags(&[0]);
    let short_map = &map[..2];
    let mut mem = GuestMem::new(&mut ram, short_map);
    mem.write(3 * PAGE_BYTES, &[1]);
    assert!(mem.code_pages_written.is_empty());
}
