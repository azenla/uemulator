use super::decode::{self, DecodeErr, Insn};
use super::exception::{Exception, VECTOR_GENERAL_PROTECTION, VECTOR_PAGE_FAULT};
use super::flags::*;
use super::*;
use crate::dev::guestmem::GuestMem;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessKind {
    Read,
    Write,
    Exec,
}

const PTE_PRESENT: u64 = 1 << 0;
const PTE_WRITABLE: u64 = 1 << 1;
const PTE_USER: u64 = 1 << 2;
const PTE_ACCESSED: u64 = 1 << 5;
const PTE_DIRTY: u64 = 1 << 6;
/// Makes a PDPT entry a 1 GiB leaf or a PD entry a 2 MiB leaf.
const PTE_PAGE_SIZE: u64 = 1 << 7;
const PTE_NO_EXECUTE: u64 = 1 << 63;
/// In a 2 MiB or 1 GiB leaf the PAT bit moves here (bit 7 is page size); the address bits
/// between it and the leaf's alignment are reserved.
const PTE_LARGE_PAGE_PAT: u64 = 1 << 12;
/// Physical address bits 40-51, reserved because CPUID 0x80000008 advertises 40 address bits.
const PTE_RESERVED_MASK: u64 = 0x000f_ff00_0000_0000;
const PTE_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const PTE_BYTES: u64 = 8;
const LEVEL_INDEX_BITS: u32 = 9;
const LEVEL_INDEX_MASK: u64 = (1 << LEVEL_INDEX_BITS) - 1;
/// The PML4 is level 3 and the PT is level 0; only 4-level long-mode paging is walked.
const TOP_LEVEL: u32 = 3;
/// Bit 47 and everything above must be equal (48-bit linear addresses, no LA57).
const CANONICAL_SIGN_BIT: u32 = 47;
const MAX_INSN_LEN: usize = 15;

/// Bits of the #PF error code.
const PF_PRESENT: u32 = 1 << 0;
const PF_WRITE: u32 = 1 << 1;
const PF_USER: u32 = 1 << 2;
const PF_RESERVED: u32 = 1 << 3;
const PF_INSTRUCTION_FETCH: u32 = 1 << 4;

/// `log_once` key bases for unmapped physical accesses; the page number (at most 40 bits) is
/// added to them. They sit in the top bits so no page number can reach another namespace, and
/// stay clear of the bases other callers use (opcodes in `exec.rs`, MSRs in `sys.rs`), which are
/// all below 2^34.
const LOG_KEY_UNMAPPED_READ: u64 = 1 << 62;
const LOG_KEY_UNMAPPED_WRITE: u64 = 1 << 61;

/// Slot of `linear` in `Cpu::tlb`: direct-mapped by page number, in the user half if `user`.
/// The JIT's inline probe computes the same index.
#[inline]
fn tlb_slot(linear: u64, user: bool) -> usize {
    ((user as usize) << TLB_INDEX_BITS)
        | ((linear >> PAGE_SHIFT) as usize & (TLB_ENTRIES_PER_MODE - 1))
}

/// See `Cpu::read_stack`.
fn stack_fault_for_non_canonical(exception: Exception) -> Exception {
    match exception {
        Exception::Fault {
            vector: VECTOR_GENERAL_PROTECTION,
            error_code: Some(0),
            ..
        } => Exception::stack_fault(0),
        other => other,
    }
}

impl Cpu {
    /// EFLAGS with the deferred arithmetic flags resolved.
    pub fn eflags_full(&self) -> u64 {
        let arith = flags::compute(
            self.lazy_op,
            self.lazy_size,
            self.lazy_dst,
            self.lazy_src,
            self.lazy_src2,
            self.eflags,
        );
        (self.eflags & !ARITH) | arith | RESERVED_ONE
    }

    /// Resolves the deferred flags into `eflags`. Needed before anything edits or snapshots
    /// non-arithmetic bits, since those writes would otherwise be mixed with stale lazy state.
    pub fn materialize_lazy_flags(&mut self) {
        self.eflags = self.eflags_full();
        self.lazy_op = LazyFlagsOp::Eflags;
    }

    /// Sets the `ARITH` flags eagerly from `arith`, dropping any deferred state.
    pub fn set_eflags_arith(&mut self, arith: u64) {
        self.eflags = (self.eflags & !ARITH) | (arith & ARITH) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
    }

    pub fn set_lazy_flags(&mut self, op: LazyFlagsOp, size: u8, dst: u64, src: u64, src2: u64) {
        self.lazy_op = op;
        self.lazy_size = size;
        self.lazy_dst = dst;
        self.lazy_src = src;
        self.lazy_src2 = src2;
    }

    pub fn cf(&self) -> bool {
        self.eflags_full() & CF != 0
    }

    /// Builds the #PF exception and sets CR2 right here, when the fault is raised, rather than
    /// at delivery.
    pub fn page_fault(
        &mut self,
        linear: u64,
        access: AccessKind,
        user: bool,
        present: bool,
        reserved_bit_violation: bool,
    ) -> Exception {
        self.cr2 = linear;
        let mut error_code = 0u32;
        if present {
            error_code |= PF_PRESENT;
        }
        if access == AccessKind::Write {
            error_code |= PF_WRITE;
        }
        if user {
            error_code |= PF_USER;
        }
        if reserved_bit_violation {
            error_code |= PF_RESERVED;
        }
        // The fetch bit is only reported when NX is enabled (SMEP is not advertised).
        if access == AccessKind::Exec && self.efer & EFER_NXE != 0 {
            error_code |= PF_INSTRUCTION_FETCH;
        }
        Exception::fault(VECTOR_PAGE_FAULT, Some(error_code))
    }

    fn read_ram_u64(&self, pa: u64) -> Option<u64> {
        let start = pa as usize;
        self.ram
            .get(start..start + PTE_BYTES as usize)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// 4-level page walk. Returns the physical address of the 4 KiB page containing `linear`
    /// (inside a 2 MiB/1 GiB leaf too) and the `TLB_FLAG_*` permissions, or the #PF. Page
    /// tables are read straight from RAM, never through the TLB; a table outside RAM counts
    /// as not present. Sets Accessed on every level touched, and Dirty on a write, but only
    /// after the permission check passed.
    ///
    /// Only long-mode (4-level) paging exists: CR4.PAE and EFER.LME are assumed set, whatever
    /// the guest wrote to them, and 32-bit/PAE/5-level formats are not walked. `Cpu::new`
    /// starts the guest in long mode and nothing in the emulator leaves it.
    fn walk(&mut self, linear: u64, access: AccessKind, user: bool) -> ExecResult<(u64, u8)> {
        let write = access == AccessKind::Write;
        let nxe = self.efer & EFER_NXE != 0;
        let mut table = self.cr3 & PTE_ADDRESS_MASK;
        // Effective permissions are the AND (user, write) or OR (no-execute) over all levels.
        let mut writable = true;
        let mut user_allowed = true;
        let mut no_execute = false;
        let mut level = TOP_LEVEL;
        loop {
            let index = (linear >> (PAGE_SHIFT + LEVEL_INDEX_BITS * level)) & LEVEL_INDEX_MASK;
            let pte_address = table + index * PTE_BYTES;
            let Some(mut pte) = self.read_ram_u64(pte_address) else {
                return Err(self.page_fault(linear, access, user, false, false));
            };
            if pte & PTE_PRESENT == 0 {
                return Err(self.page_fault(linear, access, user, false, false));
            }
            let leaf = level == 0 || (pte & PTE_PAGE_SIZE != 0 && level <= 2);
            // The page-size bit must be clear in a PML4 entry, and a large leaf must not have
            // address bits below its alignment (PAT aside).
            let below_leaf_alignment = (1u64 << (PAGE_SHIFT + LEVEL_INDEX_BITS * level)) - 1;
            let reserved = pte & PTE_RESERVED_MASK != 0
                || (!nxe && pte & PTE_NO_EXECUTE != 0)
                || (level == TOP_LEVEL && pte & PTE_PAGE_SIZE != 0)
                || (leaf
                    && level > 0
                    && pte & below_leaf_alignment & PTE_ADDRESS_MASK & !PTE_LARGE_PAGE_PAT != 0);
            if reserved {
                return Err(self.page_fault(linear, access, user, true, true));
            }
            writable &= pte & PTE_WRITABLE != 0;
            user_allowed &= pte & PTE_USER != 0;
            no_execute |= pte & PTE_NO_EXECUTE != 0;
            if leaf {
                let write_protect = self.cr0 & CR0_WP != 0;
                // Supervisor writes to read-only pages only fault when CR0.WP is set.
                let denied = (user && !user_allowed)
                    || (write && !writable && (user || write_protect))
                    || (access == AccessKind::Exec && no_execute && nxe);
                if denied {
                    return Err(self.page_fault(linear, access, user, true, false));
                }
                let mut updated = pte | PTE_ACCESSED;
                if write {
                    updated |= PTE_DIRTY;
                }
                if updated != pte {
                    pte = updated;
                    self.ram[pte_address as usize..][..PTE_BYTES as usize]
                        .copy_from_slice(&pte.to_le_bytes());
                }
                let leaf_shift = PAGE_SHIFT + LEVEL_INDEX_BITS * level;
                let leaf_offset_mask = (1u64 << leaf_shift) - 1;
                let leaf_base = pte & PTE_ADDRESS_MASK & !leaf_offset_mask;
                let page_pa = leaf_base | (linear & leaf_offset_mask & !PAGE_OFFSET_MASK);
                let mut tlb_flags = 0;
                if writable {
                    tlb_flags |= TLB_FLAG_WRITABLE;
                }
                if user_allowed {
                    tlb_flags |= TLB_FLAG_USER;
                }
                if no_execute && nxe {
                    tlb_flags |= TLB_FLAG_NO_EXECUTE;
                }
                if pte & PTE_DIRTY != 0 {
                    tlb_flags |= TLB_FLAG_DIRTY;
                }
                return Ok((page_pa, tlb_flags));
            }
            if pte & PTE_ACCESSED == 0 {
                pte |= PTE_ACCESSED;
                self.ram[pte_address as usize..][..PTE_BYTES as usize]
                    .copy_from_slice(&pte.to_le_bytes());
            }
            table = pte & PTE_ADDRESS_MASK;
            level -= 1;
        }
    }

    /// Linear to physical address for one access, refilling the TLB on a miss.
    ///
    /// `supervisor` marks implicit accesses by the CPU itself (descriptor tables, TSS,
    /// interrupt delivery): they ignore CPL and use the supervisor TLB half even at CPL 3.
    /// Only the 48-bit canonical check is done, and it raises #GP(0) even for stack accesses
    /// where hardware raises #SS.
    pub fn translate(
        &mut self,
        linear: u64,
        access: AccessKind,
        supervisor: bool,
    ) -> ExecResult<u64> {
        let sign_extension = (linear as i64) >> CANONICAL_SIGN_BIT;
        if sign_extension != 0 && sign_extension != -1 {
            return Err(Exception::general_protection(0));
        }
        let user = self.cpl == USER_CPL && !supervisor;
        let page = linear & !PAGE_OFFSET_MASK;
        let slot = tlb_slot(linear, user);
        let entry = self.tlb[slot];
        let tag = match access {
            AccessKind::Read => entry.tag_r,
            AccessKind::Write => entry.tag_w,
            AccessKind::Exec => entry.tag_x,
        };
        if tag == page {
            return Ok(linear
                .wrapping_add(entry.addend)
                .wrapping_sub(self.ram_base));
        }
        let (page_pa, walk_flags) = self.walk(linear, access, user)?;
        // Only RAM-backed pages are cached; MMIO has no host pointer to put in `addend`.
        if page_pa < self.ram.len() as u64 {
            let write_protect = self.cr0 & CR0_WP != 0;
            // Keep the other tags if the slot already maps this page, so a read refill does not
            // drop the page's exec tag and vice versa.
            let mut refilled = if entry.tag_r == page || entry.tag_w == page || entry.tag_x == page
            {
                entry
            } else {
                TLB_EMPTY
            };
            refilled.addend = (self.ram_base + page_pa).wrapping_sub(page);
            if !user || walk_flags & TLB_FLAG_USER != 0 {
                refilled.tag_r = page;
                if walk_flags & TLB_FLAG_NO_EXECUTE == 0 {
                    refilled.tag_x = page;
                }
            }
            // The write tag is withheld unless the store can skip both the walk (that sets
            // Dirty) and the code-page check, so the first store to such a page goes slow.
            let write_permitted = walk_flags & TLB_FLAG_DIRTY != 0
                && if user {
                    walk_flags & (TLB_FLAG_USER | TLB_FLAG_WRITABLE)
                        == (TLB_FLAG_USER | TLB_FLAG_WRITABLE)
                } else {
                    walk_flags & TLB_FLAG_WRITABLE != 0 || !write_protect
                };
            if write_permitted && self.page_has_code[(page_pa >> PAGE_SHIFT) as usize] == 0 {
                refilled.tag_w = page;
            }
            self.tlb[slot] = refilled;
        }
        Ok(page_pa | (linear & PAGE_OFFSET_MASK))
    }

    pub fn flush_tlb(&mut self) {
        for entry in self.tlb.iter_mut() {
            *entry = TLB_EMPTY;
        }
    }

    /// Empties the page's slot in both halves; whatever other page shares the slot goes too.
    pub fn invlpg(&mut self, linear: u64) {
        let slot = tlb_slot(linear, false);
        self.tlb[slot] = TLB_EMPTY;
        self.tlb[TLB_ENTRIES_PER_MODE + slot] = TLB_EMPTY;
    }

    /// Reads `size` bytes (at most 8, all inside one page) at a physical address: RAM, else
    /// MMIO, else all ones for unmapped space.
    fn phys_read(&mut self, pa: u64, size: u8) -> u64 {
        let start = pa as usize;
        let len = size as usize;
        if let Some(src) = self.ram.get(start..start + len) {
            let mut bytes = [0u8; 8];
            bytes[..len].copy_from_slice(src);
            u64::from_le_bytes(bytes)
        } else if let Some(value) = self.dev.mmio_read(pa, size) {
            value
        } else {
            self.log_once(LOG_KEY_UNMAPPED_READ + (pa >> PAGE_SHIFT), || {
                format!("read of unmapped physical address {pa:#x}")
            });
            u64::MAX >> (64 - 8 * size as u32)
        }
    }

    /// Counterpart of `phys_read`; writes to unmapped space are dropped.
    fn phys_write(&mut self, pa: u64, size: u8, val: u64) {
        let start = pa as usize;
        let len = size as usize;
        if start + len <= self.ram.len() {
            if self.page_has_code[start >> PAGE_SHIFT] != 0 {
                self.code_written(start >> PAGE_SHIFT);
            }
            self.ram[start..start + len].copy_from_slice(&val.to_le_bytes()[..len]);
        } else {
            let written_code_pages = {
                let mut mem = GuestMem::new(&mut self.ram, &self.page_has_code);
                if self.dev.mmio_write(pa, size, val, &mut mem) {
                    mem.code_pages_written
                } else {
                    self.log_once(LOG_KEY_UNMAPPED_WRITE + (pa >> PAGE_SHIFT), || {
                        format!("write to unmapped physical address {pa:#x}")
                    });
                    Vec::new()
                }
            };
            for page in written_code_pages {
                self.code_written(page);
            }
        }
    }

    /// A store or device DMA hit a page that translated code came from.
    ///
    /// With the JIT only that page's blocks are dropped; `block_exit_requested` makes the
    /// block that is currently running stop after this store instead of executing bytes it
    /// may just have overwritten. Without it the whole decode cache is flushed.
    pub fn code_written(&mut self, ppn: usize) {
        if let Some(mut jit) = self.jit.take() {
            jit.invalidate_page(ppn as u64);
            self.jit = Some(jit);
            self.page_has_code[ppn] = 0;
            self.decode_cache_generation = self.decode_cache_generation.wrapping_add(1);
            self.block_exit_requested = true;
        } else {
            self.flush_decode();
        }
    }

    pub fn flush_decode(&mut self) {
        self.flushes += 1;
        self.decode_cache_generation = self.decode_cache_generation.wrapping_add(1);
        for page in self.pages_with_code.drain(..) {
            self.page_has_code[page as usize] = 0;
        }
    }

    /// Flags the physical page as holding translated code and revokes write tags pointing at
    /// it, so the next store to it goes through `phys_write` and `code_written`.
    pub fn mark_code(&mut self, pa: u64) {
        let page = (pa >> PAGE_SHIFT) as usize;
        if page < self.page_has_code.len() && self.page_has_code[page] == 0 {
            self.page_has_code[page] = 1;
            // `code_written` clears a page's flag without un-listing it, so a page that is
            // rewritten and retranslated over and over would grow the list without bound.
            if self.pages_with_code.len() >= self.page_has_code.len() {
                let flags = &self.page_has_code;
                self.pages_with_code
                    .retain(|&listed| flags[listed as usize] != 0);
                self.pages_with_code.sort_unstable();
                self.pages_with_code.dedup();
            }
            self.pages_with_code.push(page as u32);
            let ram_base = self.ram_base;
            for entry in self.tlb.iter_mut() {
                // tag + addend is the page's host address, minus ram_base its physical address.
                if entry.tag_w != TLB_TAG_INVALID
                    && (entry
                        .tag_w
                        .wrapping_add(entry.addend)
                        .wrapping_sub(ram_base)
                        >> PAGE_SHIFT) as usize
                        == page
                {
                    entry.tag_w = TLB_TAG_INVALID;
                }
            }
        }
    }

    /// Guest load of `size` bytes (1, 2, 4 or 8). The fast path needs the whole access inside
    /// the page the tag covers: the tag only describes one page, and the next page may map
    /// anywhere, so a straddling access translates and reads each page separately.
    fn mem_read(&mut self, linear: u64, size: u8, supervisor: bool) -> ExecResult<u64> {
        let user = self.cpl == USER_CPL && !supervisor;
        let entry = &self.tlb[tlb_slot(linear, user)];
        if entry.tag_r == linear & !PAGE_OFFSET_MASK
            && (linear & PAGE_OFFSET_MASK) + size as u64 <= PAGE_SIZE
        {
            let host = linear.wrapping_add(entry.addend);
            unsafe {
                return Ok(match size {
                    1 => *(host as *const u8) as u64,
                    2 => (host as *const u16).read_unaligned() as u64,
                    4 => (host as *const u32).read_unaligned() as u64,
                    _ => (host as *const u64).read_unaligned(),
                });
            }
        }
        let offset = (linear & PAGE_OFFSET_MASK) as usize;
        if offset + size as usize <= PAGE_SIZE as usize {
            let pa = self.translate(linear, AccessKind::Read, supervisor)?;
            return Ok(self.phys_read(pa, size));
        }
        let first_len = PAGE_SIZE as usize - offset;
        let first_pa = self.translate(linear, AccessKind::Read, supervisor)?;
        let second_pa = self.translate(
            (linear & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE),
            AccessKind::Read,
            supervisor,
        )?;
        let low = self.phys_read(first_pa, first_len as u8);
        let high = self.phys_read(second_pa, size - first_len as u8);
        Ok(low | (high << (8 * first_len)))
    }

    /// Store counterpart of `mem_read`. A straddling store translates both pages before
    /// writing either, so a fault on the second page leaves memory untouched.
    fn mem_write(&mut self, linear: u64, size: u8, val: u64, supervisor: bool) -> ExecResult<()> {
        let user = self.cpl == USER_CPL && !supervisor;
        let entry = &self.tlb[tlb_slot(linear, user)];
        if entry.tag_w == linear & !PAGE_OFFSET_MASK
            && (linear & PAGE_OFFSET_MASK) + size as u64 <= PAGE_SIZE
        {
            let host = linear.wrapping_add(entry.addend);
            unsafe {
                match size {
                    1 => *(host as *mut u8) = val as u8,
                    2 => (host as *mut u16).write_unaligned(val as u16),
                    4 => (host as *mut u32).write_unaligned(val as u32),
                    _ => (host as *mut u64).write_unaligned(val),
                }
            }
            return Ok(());
        }
        let offset = (linear & PAGE_OFFSET_MASK) as usize;
        if offset + size as usize <= PAGE_SIZE as usize {
            let pa = self.translate(linear, AccessKind::Write, supervisor)?;
            self.phys_write(pa, size, val);
            return Ok(());
        }
        let first_len = PAGE_SIZE as usize - offset;
        let first_pa = self.translate(linear, AccessKind::Write, supervisor)?;
        let second_pa = self.translate(
            (linear & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE),
            AccessKind::Write,
            supervisor,
        )?;
        self.phys_write(first_pa, first_len as u8, val);
        self.phys_write(second_pa, size - first_len as u8, val >> (8 * first_len));
        Ok(())
    }

    pub fn read(&mut self, linear: u64, size: u8) -> ExecResult<u64> {
        self.mem_read(linear, size, false)
    }

    pub fn write(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.mem_write(linear, size, val, false)
    }

    /// Implicit stack accesses (push, pop, call, ret, ...). The only #GP(0) a data access can
    /// raise is for a non-canonical address, which through SS is reported as #SS(0) instead.
    pub fn read_stack(&mut self, linear: u64, size: u8) -> ExecResult<u64> {
        self.read(linear, size)
            .map_err(stack_fault_for_non_canonical)
    }

    pub fn write_stack(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.write(linear, size, val)
            .map_err(stack_fault_for_non_canonical)
    }

    /// Access by the CPU itself (see `translate`), independent of CPL.
    pub fn read_sys(&mut self, linear: u64, size: u8) -> ExecResult<u64> {
        self.mem_read(linear, size, true)
    }

    pub fn write_sys(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.mem_write(linear, size, val, true)
    }

    /// Implicit supervisor store to the stack, as interrupt delivery does: non-canonical is
    /// #SS(0), like `write_stack`.
    pub fn write_sys_stack(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.write_sys(linear, size, val)
            .map_err(stack_fault_for_non_canonical)
    }

    pub fn read128(&mut self, linear: u64) -> ExecResult<u128> {
        let low = self.read(linear, 8)?;
        let high = self.read(linear.wrapping_add(8), 8)?;
        Ok(low as u128 | ((high as u128) << 64))
    }

    pub fn write128(&mut self, linear: u64, v: u128) -> ExecResult<()> {
        // Probe both pages first: the stores below must not start if the second one faults.
        // The probe uses the page base, like a straddling `write`, so CR2 agrees.
        self.translate(linear, AccessKind::Write, false)?;
        let last_page = linear.wrapping_add(15) & !PAGE_OFFSET_MASK;
        if last_page != linear & !PAGE_OFFSET_MASK {
            self.translate(last_page, AccessKind::Write, false)?;
        }
        self.write(linear, 8, v as u64)?;
        self.write(linear.wrapping_add(8), 8, (v >> 64) as u64)
    }

    /// Decodes the instruction at `rip`, through the decode cache. Marks the page(s) it was
    /// read from as holding code so later stores invalidate the cache.
    pub fn fetch_decode(&mut self) -> ExecResult<Insn> {
        let rip = self.rip;
        let pa = self.translate(rip, AccessKind::Exec, false)?;
        let slot = (pa ^ (pa >> 16)) as usize & (DECODE_CACHE_ENTRIES - 1);
        let cached = &self.decode_cache[slot];
        if cached.gpa == pa && cached.generation == self.decode_cache_generation {
            return Ok(cached.insn);
        }
        let bytes_left_in_page = PAGE_SIZE as usize - (pa & PAGE_OFFSET_MASK) as usize;
        let len = bytes_left_in_page.min(MAX_INSN_LEN);
        let start = pa as usize;
        let mut buf = [0u8; MAX_INSN_LEN];
        // Executing from MMIO or unmapped space is reported as a not-present fetch fault.
        let Some(src) = self.ram.get(start..start + len) else {
            return Err(self.page_fault(rip, AccessKind::Exec, self.cpl == USER_CPL, false, false));
        };
        buf[..len].copy_from_slice(src);
        match decode::decode(&buf[..len]) {
            Ok(insn) => {
                self.decode_cache[slot] = DecodeCacheEntry {
                    gpa: pa,
                    generation: self.decode_cache_generation,
                    insn,
                };
                self.mark_code(pa);
                Ok(insn)
            }
            Err(DecodeErr::Invalid) => Err(Exception::invalid_opcode()),
            // The 15-byte window ended mid-instruction: it is longer than the architectural
            // limit, which is #GP(0) (an undecodable opcode stays #UD).
            Err(DecodeErr::Truncated) if len == MAX_INSN_LEN => {
                Err(Exception::general_protection(0))
            }
            // The instruction runs into the next page. It is not cached, because its validity
            // depends on two pages; both get marked as code instead.
            Err(DecodeErr::Truncated) => {
                let next_page_pa = self.translate(
                    (rip & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE),
                    AccessKind::Exec,
                    false,
                )?;
                let missing = MAX_INSN_LEN - len;
                let next_start = next_page_pa as usize;
                let Some(next_src) = self.ram.get(next_start..next_start + missing) else {
                    return Err(Exception::invalid_opcode());
                };
                buf[len..MAX_INSN_LEN].copy_from_slice(next_src);
                match decode::decode(&buf[..MAX_INSN_LEN]) {
                    Ok(insn) => {
                        self.mark_code(pa);
                        self.mark_code(next_page_pa);
                        Ok(insn)
                    }
                    Err(DecodeErr::Truncated) => Err(Exception::general_protection(0)),
                    Err(DecodeErr::Invalid) => Err(Exception::invalid_opcode()),
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "mem_tests.rs"]
mod tests;
