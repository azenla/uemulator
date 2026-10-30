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
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::sync::atomic::{AtomicU8, Ordering};

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
    flags: Arc<[AtomicU8]>,
}

impl CodePages {
    pub fn new(pages: usize) -> Self {
        CodePages {
            flags: (0..pages).map(|_| AtomicU8::new(0)).collect(),
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
