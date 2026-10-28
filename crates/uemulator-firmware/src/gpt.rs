//! Partition tables: GPT (primary, falling back to the backup header) and plain MBR.

use alloc::vec::Vec;

use crate::crc32::crc32;
use crate::uefi::Guid;

pub trait Blocks {
    fn block_bytes(&self) -> u64;
    fn block_count(&self) -> u64;
    fn read_blocks(&mut self, first: u64, buffer: &mut [u8]) -> bool;
}

/// `C12A7328-F81F-11D2-BA4B-00A0C93EC93B`.
pub const ESP_TYPE: Guid = Guid::new(
    0xc12a7328,
    0xf81f,
    0x11d2,
    [0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9, 0x3b],
);

const HEADER_SIGNATURE: &[u8; 8] = b"EFI PART";
const HEADER_MIN_BYTES: usize = 92;
const HEADER_MAX_BYTES: usize = 4096;
const MAX_ENTRIES: u32 = 1024;
const ENTRY_MIN_BYTES: u32 = 128;
const ENTRY_MAX_BYTES: u32 = 4096;
const MBR_SIGNATURE: [u8; 2] = [0x55, 0xaa];
const MBR_PARTITION_TABLE: usize = 446;
const MBR_ENTRY_BYTES: usize = 16;
const MBR_DISK_SIGNATURE: usize = 440;
const MBR_TYPE_PROTECTIVE: u8 = 0xee;
const MBR_TYPE_EMPTY: u8 = 0;
/// MBR types worth probing for a FAT file system (FAT12/16/32, LBA variants, the EFI type).
const MBR_TYPES_FAT: [u8; 7] = [0x01, 0x04, 0x06, 0x0b, 0x0c, 0x0e, 0xef];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    /// 1-based, the slot in the table.
    pub number: u32,
    pub first_block: u64,
    pub block_count: u64,
    pub scheme: Scheme,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scheme {
    Gpt { type_guid: Guid, unique_guid: Guid },
    Mbr { kind: u8, disk_signature: u32 },
}

impl Partition {
    pub fn is_esp(&self) -> bool {
        matches!(&self.scheme, Scheme::Gpt { type_guid, .. } if *type_guid == ESP_TYPE)
    }
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}

fn u64_at(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(data[at..at + 8].try_into().unwrap())
}

fn read_bytes(disk: &mut impl Blocks, first: u64, length: usize) -> Option<Vec<u8>> {
    let block = disk.block_bytes() as usize;
    let mut buffer = alloc::vec![0u8; length.div_ceil(block) * block];
    disk.read_blocks(first, &mut buffer).then(|| {
        buffer.truncate(length);
        buffer
    })
}

fn gpt_at(disk: &mut impl Blocks, header_block: u64) -> Option<Vec<Partition>> {
    let block = disk.block_bytes() as usize;
    let header = read_bytes(disk, header_block, block)?;
    if &header[..8] != HEADER_SIGNATURE {
        return None;
    }
    let header_bytes = u32_at(&header, 12) as usize;
    if !(HEADER_MIN_BYTES..=HEADER_MAX_BYTES.min(block)).contains(&header_bytes) {
        return None;
    }
    let mut checked = header[..header_bytes].to_vec();
    let stored_crc = u32_at(&checked, 16);
    checked[16..20].fill(0);
    if crc32(&checked) != stored_crc {
        return None;
    }
    let entries_block = u64_at(&header, 72);
    let entry_count = u32_at(&header, 80);
    let entry_bytes = u32_at(&header, 84);
    if entry_count > MAX_ENTRIES
        || !(ENTRY_MIN_BYTES..=ENTRY_MAX_BYTES).contains(&entry_bytes)
        || entries_block >= disk.block_count()
    {
        return None;
    }
    let entries = read_bytes(disk, entries_block, (entry_count * entry_bytes) as usize)?;
    if crc32(&entries) != u32_at(&header, 88) {
        return None;
    }
    Some(
        entries
            .chunks_exact(entry_bytes as usize)
            .enumerate()
            .filter_map(|(index, entry)| {
                let type_guid = Guid::from_bytes(&entry[0..16]);
                let (first, last) = (u64_at(entry, 32), u64_at(entry, 40));
                if type_guid == Guid::default() || last < first || last >= disk.block_count() {
                    return None;
                }
                Some(Partition {
                    number: index as u32 + 1,
                    first_block: first,
                    block_count: last - first + 1,
                    scheme: Scheme::Gpt {
                        type_guid,
                        unique_guid: Guid::from_bytes(&entry[16..32]),
                    },
                })
            })
            .collect(),
    )
}

/// What a disk's first sectors say. `None` means "no partition table": the whole disk may be a
/// file system of its own.
pub fn partitions(disk: &mut impl Blocks) -> Option<Vec<Partition>> {
    let block = disk.block_bytes() as usize;
    let mbr = read_bytes(disk, 0, block)?;
    if mbr[510..512] != MBR_SIGNATURE {
        return None;
    }
    let entries: Vec<(u8, u64, u64)> = (0..4)
        .map(|slot| {
            let entry = &mbr[MBR_PARTITION_TABLE + slot * MBR_ENTRY_BYTES..][..MBR_ENTRY_BYTES];
            (entry[4], u32_at(entry, 8) as u64, u32_at(entry, 12) as u64)
        })
        .collect();
    if entries.iter().any(|(kind, ..)| *kind == MBR_TYPE_PROTECTIVE) {
        let backup = disk.block_count().checked_sub(1);
        return gpt_at(disk, 1).or_else(|| gpt_at(disk, backup?));
    }
    // A FAT boot sector also ends in 0x55AA; its "partition table" bytes are code.
    let disk_signature = u32_at(&mbr, MBR_DISK_SIGNATURE);
    let found: Vec<Partition> = entries
        .iter()
        .enumerate()
        .filter(|(_, (kind, first, count))| {
            *kind != MBR_TYPE_EMPTY
                && MBR_TYPES_FAT.contains(kind)
                && *count != 0
                && first + count <= disk.block_count()
        })
        .map(|(slot, (kind, first, count))| Partition {
            number: slot as u32 + 1,
            first_block: *first,
            block_count: *count,
            scheme: Scheme::Mbr {
                kind: *kind,
                disk_signature,
            },
        })
        .collect();
    (!found.is_empty()).then_some(found)
}

#[cfg(test)]
pub mod testing {
    use super::*;

    pub struct Ram {
        pub bytes: Vec<u8>,
        pub block: u64,
    }

    impl Blocks for Ram {
        fn block_bytes(&self) -> u64 {
            self.block
        }
        fn block_count(&self) -> u64 {
            self.bytes.len() as u64 / self.block
        }
        fn read_blocks(&mut self, first: u64, buffer: &mut [u8]) -> bool {
            let start = (first * self.block) as usize;
            match self.bytes.get(start..start + buffer.len()) {
                Some(source) => {
                    buffer.copy_from_slice(source);
                    true
                }
                None => false,
            }
        }
    }

    pub const BLOCK: u64 = 512;
    pub const BLOCKS: u64 = 2048;

    /// A disk with a protective MBR and a GPT of (type, first block, block count) entries.
    pub fn gpt_disk(partitions: &[(Guid, u64, u64)]) -> Ram {
        let mut bytes = alloc::vec![0u8; (BLOCKS * BLOCK) as usize];
        bytes[MBR_PARTITION_TABLE + 4] = MBR_TYPE_PROTECTIVE;
        bytes[510..512].copy_from_slice(&MBR_SIGNATURE);
        let entry_count = 128u32;
        let mut entries = alloc::vec![0u8; (entry_count * 128) as usize];
        for (index, (type_guid, first, count)) in partitions.iter().enumerate() {
            let entry = &mut entries[index * 128..][..128];
            entry[0..16].copy_from_slice(&type_guid.to_bytes());
            entry[16..32].copy_from_slice(&[index as u8 + 1; 16]);
            entry[32..40].copy_from_slice(&first.to_le_bytes());
            entry[40..48].copy_from_slice(&(first + count - 1).to_le_bytes());
        }
        let entries_block = 2usize;
        let start = entries_block * BLOCK as usize;
        bytes[start..start + entries.len()].copy_from_slice(&entries);
        let mut header = [0u8; 92];
        header[..8].copy_from_slice(HEADER_SIGNATURE);
        header[8..12].copy_from_slice(&0x0001_0000u32.to_le_bytes());
        header[12..16].copy_from_slice(&92u32.to_le_bytes());
        header[24..32].copy_from_slice(&1u64.to_le_bytes());
        header[72..80].copy_from_slice(&(entries_block as u64).to_le_bytes());
        header[80..84].copy_from_slice(&entry_count.to_le_bytes());
        header[84..88].copy_from_slice(&128u32.to_le_bytes());
        header[88..92].copy_from_slice(&crc32(&entries).to_le_bytes());
        let crc = crc32(&header);
        header[16..20].copy_from_slice(&crc.to_le_bytes());
        bytes[512..512 + 92].copy_from_slice(&header);
        Ram { bytes, block: BLOCK }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    const LINUX_DATA: Guid = Guid::new(0x0fc63daf, 0x8483, 0x4772, [0x8e, 0x79, 0x3d, 0x69, 0xd8, 0x47, 0x7d, 0xe4]);

    #[test]
    fn gpt_lists_used_slots_with_their_numbers() {
        let mut disk = gpt_disk(&[(ESP_TYPE, 64, 256), (Guid::default(), 0, 1), (LINUX_DATA, 512, 1024)]);
        // The unused middle slot is encoded as a zero entry: clear what the helper wrote.
        let start = 2 * 512 + 128;
        disk.bytes[start..start + 128].fill(0);
        // Refresh the entries checksum the header carries.
        let entries = disk.bytes[1024..1024 + 128 * 128].to_vec();
        disk.bytes[512 + 88..512 + 92].copy_from_slice(&crc32(&entries).to_le_bytes());
        let mut header = disk.bytes[512..512 + 92].to_vec();
        header[16..20].fill(0);
        let crc = crc32(&header);
        disk.bytes[512 + 16..512 + 20].copy_from_slice(&crc.to_le_bytes());
        let found = partitions(&mut disk).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].number, found[0].first_block, found[0].block_count), (1, 64, 256));
        assert!(found[0].is_esp());
        assert_eq!((found[1].number, found[1].first_block), (3, 512));
        assert!(!found[1].is_esp());
    }

    #[test]
    fn a_corrupt_header_checksum_ends_in_no_partitions() {
        let mut disk = gpt_disk(&[(ESP_TYPE, 64, 256)]);
        disk.bytes[512 + 40] ^= 1;
        assert_eq!(partitions(&mut disk), None);
    }

    #[test]
    fn the_backup_header_is_used_when_the_primary_is_damaged() {
        let mut disk = gpt_disk(&[(ESP_TYPE, 64, 256)]);
        let backup = (BLOCKS - 1) as usize * 512;
        let header = disk.bytes[512..512 + 92].to_vec();
        disk.bytes[backup..backup + 92].copy_from_slice(&header);
        disk.bytes[512] = 0;
        assert_eq!(partitions(&mut disk).unwrap().len(), 1);
    }

    #[test]
    fn a_damaged_entry_array_is_refused() {
        let mut disk = gpt_disk(&[(ESP_TYPE, 64, 256)]);
        disk.bytes[1024 + 33] ^= 0xff;
        assert_eq!(partitions(&mut disk), None);
    }

    #[test]
    fn an_mbr_lists_fat_partitions_and_ignores_the_rest() {
        let mut disk = Ram {
            bytes: alloc::vec![0u8; (BLOCKS * BLOCK) as usize],
            block: BLOCK,
        };
        disk.bytes[510..512].copy_from_slice(&MBR_SIGNATURE);
        disk.bytes[MBR_DISK_SIGNATURE..MBR_DISK_SIGNATURE + 4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        for (slot, (kind, first, count)) in [(0x83u8, 100u32, 100u32), (0x0c, 300, 400)].iter().enumerate() {
            let entry = MBR_PARTITION_TABLE + slot * MBR_ENTRY_BYTES;
            disk.bytes[entry + 4] = *kind;
            disk.bytes[entry + 8..entry + 12].copy_from_slice(&first.to_le_bytes());
            disk.bytes[entry + 12..entry + 16].copy_from_slice(&count.to_le_bytes());
        }
        let found = partitions(&mut disk).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].number, 2);
        assert_eq!(found[0].scheme, Scheme::Mbr { kind: 0x0c, disk_signature: 0xdead_beef });
    }

    #[test]
    fn a_disk_with_no_table_has_none() {
        let mut disk = Ram {
            bytes: alloc::vec![0u8; (BLOCKS * BLOCK) as usize],
            block: BLOCK,
        };
        assert_eq!(partitions(&mut disk), None);
    }
}
