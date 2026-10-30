//! Guest RAM and the per-page "translated code" flags, as handles that every vCPU of a machine
//! clones. One allocation backs all of them, so with host threads the vCPUs see each other's
//! stores, and with one thread nothing changes: the slice views below behave like a `Vec<u8>`.
//!
//! The views hand out `&[u8]` and `&mut [u8]` over memory other vCPU threads may be writing. That
//! is a data race in Rust's model, and it is what the guest asked for: a racing guest sees
//! whichever store wins, as on hardware. Code that needs more than "some value" (page-table
//! entries, atomic read-modify-write, anything a host loop polls) goes through the raw accessors
//! in `cpu/atomic.rs` instead, and never keeps a slice across a point where another thread could
//! change what it depends on.

#[cfg(freestanding)]
use crate::prelude::*;
use crate::sync::Arc;
use alloc::alloc::{Layout, alloc_zeroed};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::sync::atomic::{AtomicU8, Ordering};

/// First guest physical address of the hole below 4 GiB that holds the PCI BARs, the ECAM, the
/// I/O APIC and the local APIC. RAM that does not fit below it continues at `PCI_HOLE_END`.
pub(crate) const PCI_HOLE_START: u64 = 0xc000_0000;
pub(crate) const PCI_HOLE_END: u64 = 0x1_0000_0000;

/// Length of the flat buffer that backs `memory_bytes` of guest RAM. Guest physical address `a`
/// is byte `a` of the buffer, so with more RAM than fits below the hole the buffer spans the
/// hole as well: its `PCI_HOLE_END - PCI_HOLE_START` bytes are never touched, and the host
/// commits memory lazily, so they cost address space and nothing else. Every "is this RAM"
/// question must go through `is_ram_range` rather than compare against the buffer length, which
/// the hole would answer wrongly.
pub(crate) fn flat_bytes(memory_bytes: u64) -> u64 {
    if memory_bytes <= PCI_HOLE_START {
        memory_bytes
    } else {
        PCI_HOLE_END + (memory_bytes - PCI_HOLE_START)
    }
}

/// End of the RAM below the hole, for a buffer of `flat_len` bytes.
pub(crate) fn low_ram_end(flat_len: u64) -> u64 {
    flat_len.min(PCI_HOLE_START)
}

/// End of the RAM above 4 GiB, or 0 when there is none.
pub(crate) fn high_ram_end(flat_len: u64) -> u64 {
    if flat_len > PCI_HOLE_END { flat_len } else { 0 }
}

/// Whether all of `start..start + len` is RAM, for a buffer of `flat_len` bytes.
pub(crate) fn is_ram_range(flat_len: u64, start: u64, len: u64) -> bool {
    let Some(end) = start.checked_add(len) else {
        return false;
    };
    end <= flat_len && (end <= PCI_HOLE_START || start >= PCI_HOLE_END)
}

/// `len` zeroed bytes, or `None` when the host refuses. Large sizes come from the allocator's
/// zeroed path (`calloc`, which maps fresh pages lazily), where `vec![0; len]` would be the same
/// but aborts the process on failure and a guest's memory size is a user setting.
pub(crate) fn zeroed_bytes(len: usize) -> Option<Vec<u8>> {
    if len == 0 {
        return Some(Vec::new());
    }
    let layout = Layout::array::<u8>(len).ok()?;
    let pointer = unsafe { alloc_zeroed(layout) };
    if pointer.is_null() {
        return None;
    }
    // Allocated with the global allocator at exactly this layout, which is what `Vec<u8>` frees.
    Some(unsafe { Vec::from_raw_parts(pointer, len, len) })
}

/// The memory behind a `GuestRam`. `bytes` is never touched except to keep the allocation alive
/// and to take its address, which stays valid because a `Vec` does not move its heap buffer.
struct RamBlock {
    pointer: *mut u8,
    length: usize,
    _bytes: Vec<u8>,
}

// The block is plain memory; sharing it between threads is the point.
unsafe impl Send for RamBlock {}
unsafe impl Sync for RamBlock {}

/// Guest physical RAM starting at address 0. Cloning shares the memory.
#[derive(Clone)]
pub(crate) struct GuestRam {
    block: Arc<RamBlock>,
}

impl GuestRam {
    pub fn new(mut bytes: Vec<u8>) -> Self {
        let pointer = bytes.as_mut_ptr();
        let length = bytes.len();
        GuestRam {
            block: Arc::new(RamBlock {
                pointer,
                length,
                _bytes: bytes,
            }),
        }
    }

    /// Host address of byte 0.
    pub fn base(&self) -> *mut u8 {
        self.block.pointer
    }

    /// Whether all of `start..start + len` is RAM (not past the end, not in the PCI hole).
    pub fn contains(&self, start: u64, len: u64) -> bool {
        is_ram_range(self.block.length as u64, start, len)
    }

    /// `len` bytes at `start`, or `None` where `contains` is false.
    pub fn slice(&self, start: u64, len: usize) -> Option<&[u8]> {
        if !self.contains(start, len as u64) {
            return None;
        }
        Some(&self[start as usize..][..len])
    }
}

impl Default for GuestRam {
    fn default() -> Self {
        GuestRam::new(Vec::new())
    }
}

impl From<Vec<u8>> for GuestRam {
    fn from(bytes: Vec<u8>) -> Self {
        GuestRam::new(bytes)
    }
}

impl Deref for GuestRam {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.block.pointer, self.block.length) }
    }
}

impl DerefMut for GuestRam {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.block.pointer, self.block.length) }
    }
}

/// One byte per guest physical page: nonzero while some vCPU's decode cache or JIT holds code
/// built from that page. Cloning shares the flags.
///
/// A store to a flagged page must reach `Cpu::code_written`; the soft TLB withholds its write tag
/// for such pages so the store takes the slow path. With several threads a flag is set by one
/// vCPU and read by the others, so the accessors are atomic (relaxed: callers that need ordering
/// against guest data add the fences themselves, see `Cpu::mark_code`).
#[derive(Clone)]
pub(crate) struct CodePages {
    flags: Arc<Vec<AtomicU8>>,
}

impl CodePages {
    /// Zeroed straight from the allocator: one byte per page of a terabyte guest is hundreds of
    /// megabytes, and filling it through an iterator would commit all of it up front.
    pub fn new(pages: usize) -> Self {
        let zeroed = zeroed_bytes(pages).unwrap_or_else(|| {
            alloc::alloc::handle_alloc_error(Layout::array::<u8>(pages).unwrap())
        });
        let mut zeroed = core::mem::ManuallyDrop::new(zeroed);
        // `AtomicU8` has the size, alignment and bit validity of `u8`, so the allocation is
        // freed correctly as a `Vec<AtomicU8>`.
        let flags =
            unsafe { Vec::from_raw_parts(zeroed.as_mut_ptr().cast::<AtomicU8>(), pages, pages) };
        CodePages {
            flags: Arc::new(flags),
        }
    }

    pub fn len(&self) -> usize {
        self.flags.len()
    }

    pub fn get(&self, page: usize) -> u8 {
        self.flags[page].load(Ordering::Relaxed)
    }

    /// Like `get`, but a page number past the end reads as "no code".
    pub fn get_or_zero(&self, page: usize) -> u8 {
        self.flags
            .get(page)
            .map_or(0, |flag| flag.load(Ordering::Relaxed))
    }

    pub fn set(&self, page: usize, value: u8) {
        self.flags[page].store(value, Ordering::SeqCst);
    }

    /// The flags as plain bytes, for `GuestMem`'s read-only page check. Another thread may set a
    /// flag while the slice is alive; the check only needs a recent value, because
    /// `Cpu::mark_code` orders the flag against the guest bytes with a fence on its side.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.flags.as_ptr().cast::<u8>(), self.flags.len()) }
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        self.flags.iter().map(|flag| flag.load(Ordering::Relaxed))
    }
}

// Indexing is for tests and the single-threaded paths that already hold a `&Cpu`/`&mut Cpu`:
// the reference points at the flag's byte, which another thread may change under it. Threaded
// code uses `get` and `set`.
impl Index<usize> for CodePages {
    type Output = u8;
    fn index(&self, page: usize) -> &u8 {
        unsafe { &*self.flags[page].as_ptr() }
    }
}

impl IndexMut<usize> for CodePages {
    fn index_mut(&mut self, page: usize) -> &mut u8 {
        unsafe { &mut *self.flags[page].as_ptr() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_of_the_ram_see_each_others_writes() {
        let mut first = GuestRam::new(vec![0; 16]);
        let second = first.clone();
        first[3] = 7;
        assert_eq!(second[3], 7);
        assert_eq!(second.len(), 16);
        assert_eq!(first.base() as usize, second.as_ptr() as usize);
    }

    #[test]
    fn the_ram_keeps_its_address_when_the_handle_moves() {
        let ram = GuestRam::new(vec![1; 4096]);
        let base = ram.base();
        let moved = Box::new(ram);
        assert_eq!(moved.base(), base);
        assert_eq!(moved[100], 1);
    }

    #[test]
    fn flat_buffer_leaves_the_pci_hole_out_of_the_ram() {
        const GIB: u64 = 1 << 30;
        assert_eq!(flat_bytes(GIB), GIB);
        assert_eq!(flat_bytes(3 * GIB), 3 * GIB);
        assert_eq!(flat_bytes(4 * GIB), 5 * GIB);
        assert_eq!(flat_bytes(3 * GIB + 1), 4 * GIB + 1);
        assert_eq!(
            (low_ram_end(5 * GIB), high_ram_end(5 * GIB)),
            (3 * GIB, 5 * GIB)
        );
        assert_eq!((low_ram_end(GIB), high_ram_end(GIB)), (GIB, 0));
        // Exactly 3 GiB has no high RAM: the buffer ends where the hole starts.
        assert_eq!(high_ram_end(3 * GIB), 0);
    }

    #[test]
    fn ranges_must_avoid_the_hole_and_the_end() {
        const GIB: u64 = 1 << 30;
        let flat = flat_bytes(4 * GIB);
        assert!(is_ram_range(flat, 0, 4096));
        assert!(is_ram_range(flat, PCI_HOLE_START - 8, 8));
        assert!(!is_ram_range(flat, PCI_HOLE_START - 4, 8));
        assert!(!is_ram_range(flat, PCI_HOLE_START, 1));
        assert!(!is_ram_range(flat, PCI_HOLE_END - 1, 1));
        assert!(!is_ram_range(flat, PCI_HOLE_END - 4, 8));
        assert!(is_ram_range(flat, PCI_HOLE_END, 8));
        assert!(is_ram_range(flat, flat - 8, 8));
        assert!(!is_ram_range(flat, flat - 4, 8));
        assert!(!is_ram_range(flat, u64::MAX, 2));
        // Without high RAM the buffer ends below the hole.
        assert!(!is_ram_range(GIB, GIB, 1));
        assert!(is_ram_range(GIB, 0, 0));
    }

    #[test]
    fn the_hole_is_not_ram_for_a_handle_either() {
        let ram = GuestRam::new(zeroed_bytes(1 << 12).unwrap());
        assert!(ram.contains(0, 1 << 12));
        assert!(!ram.contains(1 << 12, 1));
        assert_eq!(ram.slice(8, 4), Some(&[0u8; 4][..]));
        assert_eq!(ram.slice(4094, 4), None);
    }

    #[test]
    fn zeroed_bytes_reports_failure_instead_of_aborting() {
        assert_eq!(zeroed_bytes(0).unwrap().len(), 0);
        assert!(zeroed_bytes(4096).unwrap().iter().all(|&byte| byte == 0));
        assert!(zeroed_bytes(isize::MAX as usize).is_none());
    }

    #[test]
    fn code_page_flags_are_shared_between_clones() {
        let flags = CodePages::new(4);
        let other = flags.clone();
        flags.set(2, 1);
        assert_eq!(other.get(2), 1);
        assert_eq!(other.get(1), 0);
        assert_eq!(other.get_or_zero(99), 0);
        assert_eq!(other.len(), 4);
    }

    #[test]
    fn code_page_flags_index_like_a_slice() {
        let mut flags = CodePages::new(2);
        flags[1] = 1;
        assert_eq!(flags.get(1), 1);
        assert_eq!(flags[1], 1);
        assert!(flags.iter().any(|flag| flag != 0));
    }
}
