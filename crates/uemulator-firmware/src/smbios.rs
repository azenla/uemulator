//! A minimal SMBIOS 3.0 table: BIOS information, system information and the end marker. Xen and
//! Linux read the vendor strings from it; nothing needs more.

use alloc::vec::Vec;

const ENTRY_POINT_BYTES: usize = 24;
const ANCHOR: &[u8; 5] = b"_SM3_";
const MAJOR: u8 = 3;
const MINOR: u8 = 0;
const ENTRY_POINT_REVISION: u8 = 1;
const TYPE_BIOS: u8 = 0;
const TYPE_SYSTEM: u8 = 1;
const TYPE_END: u8 = 127;
const BIOS_LENGTH: u8 = 0x18;
const SYSTEM_LENGTH: u8 = 0x1b;
const END_LENGTH: u8 = 4;
/// "BIOS characteristics not supported".
const BIOS_CHARACTERISTICS_UNSUPPORTED: u64 = 1 << 3;
const WAKE_UP_POWER_SWITCH: u8 = 6;

/// One structure: the formatted bytes (header filled in here) and its strings.
fn structure(kind: u8, length: u8, handle: u16, body: &[u8], strings: &[&str]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[kind, length]);
    bytes.extend_from_slice(&handle.to_le_bytes());
    bytes.extend_from_slice(body);
    bytes.resize(length as usize, 0);
    if strings.is_empty() {
        bytes.extend_from_slice(&[0, 0]);
    }
    for text in strings {
        bytes.extend_from_slice(text.as_bytes());
        bytes.push(0);
    }
    if !strings.is_empty() {
        bytes.push(0);
    }
    bytes
}

pub fn build_table() -> Vec<u8> {
    let mut table = Vec::new();
    // Vendor, version, release date are strings 1-3.
    let mut bios = Vec::new();
    bios.extend_from_slice(&[1, 2]);
    bios.extend_from_slice(&0u16.to_le_bytes());
    bios.push(3);
    bios.push(0);
    bios.extend_from_slice(&BIOS_CHARACTERISTICS_UNSUPPORTED.to_le_bytes());
    table.extend(structure(TYPE_BIOS, BIOS_LENGTH, 0, &bios, &["uemulator", "1", "01/01/2026"]));
    // Manufacturer, product, version, serial, SKU, family are strings 1-6; the UUID stays zero.
    let mut system = Vec::new();
    system.extend_from_slice(&[1, 2, 3, 4]);
    system.extend_from_slice(&[0; 16]);
    system.extend_from_slice(&[WAKE_UP_POWER_SWITCH, 5, 6]);
    table.extend(structure(
        TYPE_SYSTEM,
        SYSTEM_LENGTH,
        1,
        &system,
        &["uemulator", "Virtual Machine", "1", "0", "uemulator", "uemulator"],
    ));
    table.extend(structure(TYPE_END, END_LENGTH, 2, &[], &[]));
    table
}

/// The 64-bit entry point for a table of `table_length` bytes at `table_address`.
pub fn entry_point(table_address: u64, table_length: u32) -> [u8; ENTRY_POINT_BYTES] {
    let mut entry = [0u8; ENTRY_POINT_BYTES];
    entry[..5].copy_from_slice(ANCHOR);
    entry[6] = ENTRY_POINT_BYTES as u8;
    entry[7] = MAJOR;
    entry[8] = MINOR;
    entry[10] = ENTRY_POINT_REVISION;
    entry[12..16].copy_from_slice(&table_length.to_le_bytes());
    entry[16..24].copy_from_slice(&table_address.to_le_bytes());
    let sum = entry.iter().fold(0u8, |sum, &byte| sum.wrapping_add(byte));
    entry[5] = 0u8.wrapping_sub(sum);
    entry
}

#[cfg(target_os = "none")]
pub fn install() {
    use crate::sync::Global;
    use crate::tables;
    use crate::uefi::guids;

    const SPACE: usize = 1024;
    /// In the firmware image, which the OS keeps, so the table outlives boot services.
    static STORAGE: Global<[u8; SPACE]> = Global::new([0; SPACE]);
    let table = build_table();
    let storage = STORAGE.get();
    assert!(table.len() + ENTRY_POINT_BYTES <= SPACE);
    storage[ENTRY_POINT_BYTES..ENTRY_POINT_BYTES + table.len()].copy_from_slice(&table);
    let table_address = storage[ENTRY_POINT_BYTES..].as_ptr() as u64;
    storage[..ENTRY_POINT_BYTES].copy_from_slice(&entry_point(table_address, table.len() as u32));
    tables::install_configuration(guids::SMBIOS3_TABLE, storage.as_mut_ptr().cast());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_point_checksums_to_zero() {
        let entry = entry_point(0x1234_5000, 200);
        assert_eq!(entry.iter().fold(0u8, |sum, &byte| sum.wrapping_add(byte)), 0);
        assert_eq!(&entry[..5], b"_SM3_");
        assert_eq!(u64::from_le_bytes(entry[16..24].try_into().unwrap()), 0x1234_5000);
    }

    #[test]
    fn the_table_is_a_walkable_list_ending_in_type_127() {
        let table = build_table();
        let mut at = 0;
        let mut kinds = Vec::new();
        while at < table.len() {
            let (kind, length) = (table[at], table[at + 1] as usize);
            kinds.push(kind);
            at += length;
            // Skip the string set: it ends at the first empty string.
            while !(table[at] == 0 && table[at + 1] == 0) {
                at += 1;
            }
            at += 2;
        }
        assert_eq!(kinds, [TYPE_BIOS, TYPE_SYSTEM, TYPE_END]);
        assert_eq!(at, table.len());
    }

    #[test]
    fn strings_follow_the_formatted_part() {
        let table = build_table();
        assert_eq!(&table[BIOS_LENGTH as usize..BIOS_LENGTH as usize + 10], b"uemulator\0");
    }
}
