#[cfg(target_os = "uefi")]
use crate::prelude::*;

/// Guest page size in bytes is 1 << PAGE_SHIFT; must match `Cpu::page_has_code` indexing.
const PAGE_SHIFT: u32 = 12;

/// Device-side view of guest RAM for DMA (virtqueues, request buffers).
///
/// Device code cannot take `&mut Cpu`, so this borrows RAM and the `page_has_code`
/// flags separately. DMA writes bypass the CPU store path (`Cpu::phys_write`), which is
/// what normally invalidates translated code, so every write that lands on a flagged
/// page is recorded in `code_pages_written`. The caller must drain `code_pages_written` into `Cpu::code_written`
/// once the borrow ends (see `tick_devices` and `phys_write` in cpu); skipping that
/// leaves stale decoded blocks or JIT code for memory the device just overwrote.
pub(crate) struct GuestMem<'a> {
    ram: &'a mut [u8],
    /// `Cpu::page_has_code`: one byte per guest page, nonzero if a translated block covers it.
    code: &'a [u8],
    /// Page numbers written while flagged in `code`, each listed once.
    pub code_pages_written: Vec<usize>,
}

impl<'a> GuestMem<'a> {
    pub fn new(ram: &'a mut [u8], code: &'a [u8]) -> Self {
        GuestMem {
            ram,
            code,
            code_pages_written: Vec::new(),
        }
    }

    /// True if `len` bytes at `phys_addr` are all RAM. Lets a device validate a guest-supplied
    /// buffer once up front instead of finding out per access.
    pub fn contains(&self, phys_addr: u64, len: u64) -> bool {
        phys_addr
            .checked_add(len)
            .is_some_and(|end| end <= self.ram.len() as u64)
    }

    /// Out-of-range reads fail instead of faulting: a bad guest descriptor must not panic the host.
    pub fn read(&self, phys_addr: u64, buf: &mut [u8]) -> bool {
        match usize::try_from(phys_addr)
            .ok()
            .and_then(|start| self.ram.get(start..start.checked_add(buf.len())?))
        {
            Some(src) => {
                buf.copy_from_slice(src);
                true
            }
            None => false,
        }
    }

    /// Returns false, writing nothing, if any byte of the range is outside RAM.
    pub fn write(&mut self, phys_addr: u64, data: &[u8]) -> bool {
        let Ok(start) = usize::try_from(phys_addr) else {
            return false;
        };
        let Some(end) = start.checked_add(data.len()) else {
            return false;
        };
        if end > self.ram.len() || data.is_empty() {
            return end <= self.ram.len();
        }
        // A write can span pages; `end - 1` is the last byte written, not one past it.
        for page in (start >> PAGE_SHIFT)..=((end - 1) >> PAGE_SHIFT) {
            if self.code.get(page).is_some_and(|&flag| flag != 0)
                && !self.code_pages_written.contains(&page)
            {
                self.code_pages_written.push(page);
            }
        }
        self.ram[start..end].copy_from_slice(data);
        true
    }

    /// The `read_u*` accessors return 0 for unmapped addresses.
    pub fn read_u16(&self, phys_addr: u64) -> u16 {
        let mut bytes = [0u8; 2];
        self.read(phys_addr, &mut bytes);
        u16::from_le_bytes(bytes)
    }

    pub fn read_u32(&self, phys_addr: u64) -> u32 {
        let mut bytes = [0u8; 4];
        self.read(phys_addr, &mut bytes);
        u32::from_le_bytes(bytes)
    }

    pub fn read_u64(&self, phys_addr: u64) -> u64 {
        let mut bytes = [0u8; 8];
        self.read(phys_addr, &mut bytes);
        u64::from_le_bytes(bytes)
    }

    pub fn write_u16(&mut self, phys_addr: u64, value: u16) {
        self.write(phys_addr, &value.to_le_bytes());
    }

    pub fn write_u32(&mut self, phys_addr: u64, value: u32) {
        self.write(phys_addr, &value.to_le_bytes());
    }
}

#[cfg(test)]
pub(crate) fn test_mem(ram: &mut [u8]) -> GuestMem<'_> {
    GuestMem::new(ram, &[])
}

#[cfg(test)]
#[path = "guestmem_tests.rs"]
mod tests;
