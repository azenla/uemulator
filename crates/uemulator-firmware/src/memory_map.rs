//! The physical memory map: sorted, non-overlapping page ranges with a UEFI memory type each.
//! Free memory is the `CONVENTIONAL` ranges, so allocating is retyping a range and there is one
//! source of truth for what `GetMemoryMap` reports.

use crate::uefi::memory_type::CONVENTIONAL;

pub const PAGE_SIZE: u64 = 4096;
const CAPACITY: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub start: u64,
    pub pages: u64,
    pub kind: u32,
    pub attribute: u64,
}

impl Region {
    pub const EMPTY: Region = Region {
        start: 0,
        pages: 0,
        kind: 0,
        attribute: 0,
    };

    pub fn end(&self) -> u64 {
        self.start + self.pages * PAGE_SIZE
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MapError {
    /// The map has no room for another range.
    Full,
    /// `free` or `allocate_at` named memory that is not in the state the call needs.
    NotFound,
}

pub struct MemoryMap {
    regions: [Region; CAPACITY],
    len: usize,
    /// `GetMemoryMap`'s map key: any change to the map must change it, so that an application
    /// that allocated between `GetMemoryMap` and `ExitBootServices` is caught.
    key: usize,
}

impl MemoryMap {
    pub const fn new() -> Self {
        Self {
            regions: [Region::EMPTY; CAPACITY],
            len: 0,
            key: 0,
        }
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions[..self.len]
    }

    pub fn key(&self) -> usize {
        self.key
    }

    /// The type of the page holding `address`, `None` outside every range.
    pub fn kind_at(&self, address: u64) -> Option<u32> {
        self.regions()
            .iter()
            .find(|region| region.start <= address && address < region.end())
            .map(|region| region.kind)
    }

    /// Makes `[start, start + pages)` this type, replacing whatever the range held and
    /// splitting the ranges it cuts into.
    pub fn set(&mut self, start: u64, pages: u64, kind: u32, attribute: u64) -> Result<(), MapError> {
        if pages == 0 {
            return Ok(());
        }
        let end = start + pages * PAGE_SIZE;
        let mut result = [Region::EMPTY; CAPACITY + 2];
        let mut count = 0;
        let mut inserted = false;
        for region in self.regions() {
            if region.end() <= start || region.start >= end {
                if region.start >= end && !inserted {
                    result[count] = Region {
                        start,
                        pages,
                        kind,
                        attribute,
                    };
                    count += 1;
                    inserted = true;
                }
                result[count] = *region;
                count += 1;
                continue;
            }
            if region.start < start {
                result[count] = Region {
                    pages: (start - region.start) / PAGE_SIZE,
                    ..*region
                };
                count += 1;
            }
            if !inserted {
                result[count] = Region {
                    start,
                    pages,
                    kind,
                    attribute,
                };
                count += 1;
                inserted = true;
            }
            if region.end() > end {
                result[count] = Region {
                    start: end,
                    pages: (region.end() - end) / PAGE_SIZE,
                    ..*region
                };
                count += 1;
            }
        }
        if !inserted {
            result[count] = Region {
                start,
                pages,
                kind,
                attribute,
            };
            count += 1;
        }
        // Merge neighbours that now look alike; an unmerged map would fill up with the
        // fragments of a thousand small allocations.
        let mut merged = 0;
        for index in 0..count {
            let region = result[index];
            if merged > 0 {
                let previous = &mut result[merged - 1];
                if previous.end() == region.start
                    && previous.kind == region.kind
                    && previous.attribute == region.attribute
                {
                    previous.pages += region.pages;
                    continue;
                }
            }
            result[merged] = region;
            merged += 1;
        }
        if merged > CAPACITY {
            return Err(MapError::Full);
        }
        self.regions[..merged].copy_from_slice(&result[..merged]);
        self.len = merged;
        self.key += 1;
        Ok(())
    }

    /// Whether every page of the range is `CONVENTIONAL`.
    fn is_free(&self, start: u64, pages: u64) -> bool {
        let end = start + pages * PAGE_SIZE;
        self.regions()
            .iter()
            .any(|region| region.kind == CONVENTIONAL && region.start <= start && end <= region.end())
    }

    /// Takes `pages` of free memory at the highest address that fits, whose last byte is at or
    /// below `max_address` and whose start is a multiple of `alignment` (a power of two, at least
    /// the page size). Top-down like EDK2: loaders asking for the low addresses they prefer get
    /// them with `allocate_at`, and everything else stays out of their way.
    pub fn allocate_any(
        &mut self,
        pages: u64,
        max_address: u64,
        alignment: u64,
        kind: u32,
    ) -> Result<u64, MapError> {
        let length = pages * PAGE_SIZE;
        let candidate = self.regions().iter().rev().find_map(|region| {
            if region.kind != CONVENTIONAL {
                return None;
            }
            let end = region.end().min(max_address.saturating_add(1));
            let start = end.checked_sub(length)? & !(alignment - 1);
            (start >= region.start).then_some(start)
        });
        let start = candidate.ok_or(MapError::NotFound)?;
        self.set(start, pages, kind, crate::uefi::memory_attribute::WRITE_BACK)?;
        Ok(start)
    }

    pub fn allocate_at(&mut self, start: u64, pages: u64, kind: u32) -> Result<(), MapError> {
        if !self.is_free(start, pages) {
            return Err(MapError::NotFound);
        }
        self.set(start, pages, kind, crate::uefi::memory_attribute::WRITE_BACK)
    }

    /// Returns allocated pages to the free pool. Memory that is not allocated memory, or lies
    /// outside the map, is an error: freeing a free page twice is a bug in the caller.
    pub fn free(&mut self, start: u64, pages: u64) -> Result<(), MapError> {
        let end = start + pages * PAGE_SIZE;
        let allocated = self.regions().iter().any(|region| {
            region.kind != CONVENTIONAL && region.start <= start && end <= region.end()
        });
        if !allocated {
            return Err(MapError::NotFound);
        }
        self.set(start, pages, CONVENTIONAL, crate::uefi::memory_attribute::WRITE_BACK)
    }

    /// Total free pages, for `QueryVariableInfo`-style questions and diagnostics.
    pub fn free_pages(&self) -> u64 {
        self.regions()
            .iter()
            .filter(|region| region.kind == CONVENTIONAL)
            .map(|region| region.pages)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uefi::memory_attribute::WRITE_BACK;
    use crate::uefi::memory_type::{BOOT_SERVICES_DATA, LOADER_DATA, RESERVED};

    const MIB: u64 = 1 << 20;

    fn map() -> MemoryMap {
        let mut map = MemoryMap::new();
        map.set(0, 16, RESERVED, 0).unwrap();
        map.set(MIB, 15 * MIB / PAGE_SIZE, CONVENTIONAL, WRITE_BACK).unwrap();
        map
    }

    #[test]
    fn allocation_takes_the_highest_fit() {
        let mut map = map();
        let address = map.allocate_any(4, u64::MAX, PAGE_SIZE, LOADER_DATA).unwrap();
        assert_eq!(address, 16 * MIB - 4 * PAGE_SIZE);
        assert_eq!(map.kind_at(address), Some(LOADER_DATA));
        assert_eq!(map.kind_at(address - 1), Some(CONVENTIONAL));
    }

    #[test]
    fn max_address_is_the_last_usable_byte() {
        let mut map = map();
        let address = map
            .allocate_any(1, 4 * MIB - 1, PAGE_SIZE, LOADER_DATA)
            .unwrap();
        assert_eq!(address, 4 * MIB - PAGE_SIZE);
        assert_eq!(
            map.allocate_any(1, MIB / 2, PAGE_SIZE, LOADER_DATA),
            Err(MapError::NotFound)
        );
    }

    #[test]
    fn alignment_rounds_the_start_down() {
        let mut map = map();
        let address = map.allocate_any(1, u64::MAX, 2 * MIB, LOADER_DATA).unwrap();
        assert_eq!(address % (2 * MIB), 0);
        assert_eq!(address, 14 * MIB);
    }

    #[test]
    fn allocate_at_needs_free_memory() {
        let mut map = map();
        map.allocate_at(2 * MIB, 8, LOADER_DATA).unwrap();
        assert_eq!(map.allocate_at(2 * MIB + PAGE_SIZE, 1, LOADER_DATA), Err(MapError::NotFound));
        assert_eq!(map.allocate_at(0, 1, LOADER_DATA), Err(MapError::NotFound));
        assert_eq!(map.allocate_at(100 * MIB, 1, LOADER_DATA), Err(MapError::NotFound));
    }

    #[test]
    fn freeing_merges_back_and_rejects_free_memory() {
        let mut map = map();
        let before = map.regions().to_vec();
        let address = map.allocate_any(3, u64::MAX, PAGE_SIZE, BOOT_SERVICES_DATA).unwrap();
        map.free(address, 3).unwrap();
        assert_eq!(map.regions(), before);
        assert_eq!(map.free(address, 3), Err(MapError::NotFound));
        assert_eq!(map.free(0, 1), Ok(()), "reserved memory counts as allocated");
    }

    #[test]
    fn any_change_moves_the_key() {
        let mut map = map();
        let key = map.key();
        map.allocate_any(1, u64::MAX, PAGE_SIZE, LOADER_DATA).unwrap();
        assert_ne!(map.key(), key);
    }

    #[test]
    fn setting_inside_a_region_splits_it() {
        let mut map = map();
        map.set(4 * MIB, 2, LOADER_DATA, 0).unwrap();
        let kinds: Vec<u32> = map.regions().iter().map(|region| region.kind).collect();
        assert_eq!(kinds, [RESERVED, CONVENTIONAL, LOADER_DATA, CONVENTIONAL]);
        assert_eq!(map.free_pages(), 15 * MIB / PAGE_SIZE - 2);
    }

    #[test]
    fn setting_across_regions_replaces_them() {
        let mut map = map();
        map.set(2 * MIB, 1, LOADER_DATA, 0).unwrap();
        map.set(MIB, 15 * MIB / PAGE_SIZE, BOOT_SERVICES_DATA, 0).unwrap();
        assert_eq!(map.regions().len(), 2);
    }

    #[test]
    fn a_map_that_would_overflow_is_refused_and_left_alone() {
        let mut map = MemoryMap::new();
        for index in 0..CAPACITY as u64 {
            let kind = if index % 2 == 0 { RESERVED } else { LOADER_DATA };
            map.set(index * PAGE_SIZE, 1, kind, 0).unwrap();
        }
        assert_eq!(map.set(CAPACITY as u64 * PAGE_SIZE + PAGE_SIZE, 1, RESERVED, 0), Err(MapError::Full));
        assert_eq!(map.regions().len(), CAPACITY);
    }
}
