use std::collections::HashMap;

use super::arch;
use super::helpers;
use crate::cpu::Cpu;
use crate::cpu::decode::{self, DecodeErr, Insn};
use crate::cpu::exception::Exception;
use crate::cpu::mem::AccessKind;
use crate::sys::ExecMem;

/// The code buffer is a few segments, each starting with its own copy of the prologue (entry
/// trampoline, exit/chain stubs, memory thunks). On aarch64 every block reaches those with a
/// direct `b`/`bl` (`Asm::b_abs` asserts on out-of-range), which spans only +-128 MiB, so a
/// segment must stay below that; several of them make a buffer big enough to hold everything a
/// guest boot translates, instead of throwing all code away every 128 MiB and translating the
/// hot parts again (the kernel plus an installer userspace is ~500k blocks, ~140 MiB).
const SEGMENT_SIZE: usize = 96 << 20;
const SEGMENT_COUNT: usize = 5;
const CODE_SIZE: usize = SEGMENT_SIZE * SEGMENT_COUNT;
/// Free space required before translating a block; below it the whole cache is flushed. Assumed to
/// bound the code of one block (`MAX_INSNS` instructions); `ExecMem::write_bytes` asserts if not.
const CODE_HEADROOM: usize = 1 << 20;
const BLOCK_ALIGN: usize = 16;
/// The prologue is padded to a multiple of this before the first block (64 is a common cache
/// line size; the exact reason is not documented).
const HEADER_ALIGN: usize = 64;
/// Longest block, in guest instructions.
const MAX_INSNS: usize = 48;
const MAX_INSN_LEN: usize = 15;
const PAGE_SHIFT: u32 = 12;
const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

const JUMP_CACHE_BITS: usize = 16;

/// One slot of the direct-mapped jump cache. `#[repr(C)]` with a power-of-two size because the
/// chain stub in `arch_*.rs` indexes the table and reads `rip`, `phys_addr` and `code_addr` at
/// these offsets (`JUMP_ENTRY_*`). An empty slot has `code_addr == 0`, which the stub treats as
/// a miss.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct JumpEntry {
    pub rip: u64,
    pub phys_addr: u64,
    pub code_addr: u64,
    /// Index into `Jit::blocks` plus one, so `invalidate_page` can tell whether the slot still
    /// belongs to the block it is dropping or was since taken over by a colliding rip.
    block: u64,
}

// Read by the chain stubs in `arch_a64.rs` and `arch_x64.rs`.
pub(crate) const JUMP_ENTRY_SHIFT: u32 = std::mem::size_of::<JumpEntry>().trailing_zeros();
const _: () = assert!(std::mem::size_of::<JumpEntry>().is_power_of_two());
pub(crate) const JUMP_ENTRY_PHYS_OFFSET: u32 = std::mem::offset_of!(JumpEntry, phys_addr) as u32;
pub(crate) const JUMP_ENTRY_CODE_OFFSET: u32 = std::mem::offset_of!(JumpEntry, code_addr) as u32;

/// A patchable slot in a compiled block: the exit to the fixed guest address `target_rip` (in the
/// block's own page) is preceded by a nop of `arch::link_bytes` size that becomes a direct jump to
/// the successor's `link_addr` once it exists.
pub(crate) struct LinkSite {
    /// Offset of the slot inside the block's code.
    pub code_offset: u32,
    pub target_rip: u64,
}

/// What `arch::compile_block` produces.
pub(crate) struct CompiledBlock {
    pub code: Vec<u8>,
    /// Guest instructions lowered inline.
    pub inlined: u32,
    /// Offset of the entry the run loop and the chain stub use. Everything before it is the
    /// `link_addr` entry's checks (`icount` against `next_event`, `block_exit_requested`), which
    /// those two callers have already done; a direct link skips the stub and so must do them.
    pub body_offset: u32,
    pub links: Vec<LinkSite>,
    /// Flag records the block leaves out because a later instruction overwrites them.
    pub flag_records_skipped: u32,
}

/// A slot in the code buffer and the block it should jump to: `(rip, phys_addr)` of the
/// destination, the same key as `block_map`.
#[derive(Clone)]
struct LinkedSite {
    /// Offset into `ExecMem`.
    exec_offset: usize,
    destination: (u64, u64),
}

/// One translated block. Blocks are only ever added; an invalidated block keeps its slot (and its
/// `valid == false` flag) until `Jit::reset`, so indices held in `page_blocks` and the jump cache
/// stay stable.
pub(crate) struct Block {
    rip: u64,
    /// Physical address of the first instruction. Part of the lookup key together with `rip`: the
    /// code embeds rip-relative constants, and a remapped virtual page must not hit a stale block.
    phys_addr: u64,
    /// The body entry (run loop, chain stub, jump cache).
    code_addr: u64,
    /// The first byte of the block: the entry direct links jump to.
    link_addr: u64,
    #[cfg_attr(not(test), allow(dead_code))]
    valid: bool,
    /// Generated code calls `helper_exec` with raw pointers into this slice, so it must outlive the
    /// code. Heap storage keeps the addresses stable when `blocks` reallocates.
    _insns: Box<[Insn]>,
}

/// Block cache and code buffer. The buffer is a single read-write-execute mapping (`MAP_JIT` on
/// macOS, never flipped with mprotect; `plans/06-jit.md` describes an RW then RX path for Linux
/// that the code does not follow). All writes must go through
/// `ExecMem::write_bytes`, which toggles per-thread write protection on Apple Silicon and
/// flushes the instruction cache.
pub(crate) struct Jit {
    exec_mem: ExecMem,
    /// Bump pointer into `exec_mem`.
    used_bytes: usize,
    /// Host address of the prologue trampoline at offset 0 of the buffer; see `step_block`.
    pub enter_trampoline: u64,
    /// The addresses each segment's blocks are compiled against (its own prologue copy).
    segment_addrs: Vec<arch::Addrs>,
    blocks: Vec<Block>,
    /// Direct-mapped cache indexed by `jump_cache_hash(rip)`. Both `lookup` and the chain stub
    /// read it, so it is allocated once and never resized: the stub has its address baked in.
    jump_cache: Vec<JumpEntry>,
    /// Every live block by `(rip, phys_addr)`. The jump cache is direct-mapped and much smaller
    /// than the working set of a boot (hundreds of thousands of blocks), so without this a
    /// colliding rip would evict a block and the next visit would translate it all over again.
    block_map: HashMap<(u64, u64), u32>,
    /// Physical page number -> blocks that decoded bytes from it. A block that crosses a page is
    /// listed under both.
    page_blocks: HashMap<u64, Vec<u32>>,
    /// Slots patched to jump into a block (by its index), so invalidating the block can put them
    /// back to falling through to the chain stub.
    incoming: HashMap<u32, Vec<LinkedSite>>,
    /// Unpatched slots by the `(rip, phys_addr)` of the block they want: the destination was not
    /// translated yet, or was invalidated since.
    pending: HashMap<(u64, u64), Vec<LinkedSite>>,
    /// Offset of the first block, just past the prologue. `reset` rewinds to here.
    blocks_start: usize,
    stats: JitStats,
}

/// Translation counters, read by tests to check that the cache is used and invalidated as
/// intended (`Jit::stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct JitStats {
    /// Blocks compiled, including ones compiled again after an invalidation or a reset.
    pub translated: u64,
    /// Guest instructions lowered inline, i.e. not through `helper_exec`.
    pub inlined: u64,
    /// Guest instructions compiled, inline or not.
    pub total: u64,
    /// Times the whole code buffer was flushed because it was full.
    pub resets: u64,
    /// Block-exit slots patched into a direct jump, including relinks after an invalidation.
    pub linked: u64,
    /// Slots put back to the chain stub because their destination was invalidated.
    pub unlinked: u64,
    /// Lazy-flag records left out of compiled blocks as dead.
    pub flag_records_skipped: u64,
}

pub(crate) const fn jump_cache_mask() -> u64 {
    (1 << JUMP_CACHE_BITS) - 1
}

fn jump_cache_hash(rip: u64) -> usize {
    ((rip ^ (rip >> JUMP_CACHE_BITS)) as usize) & ((1 << JUMP_CACHE_BITS) - 1)
}

impl Jit {
    /// `chaining` lets blocks jump straight to their successor through the jump cache instead of
    /// returning to the run loop between blocks; the lockstep verifier needs one block per
    /// `step_block`, so it passes false.
    pub fn new(chaining: bool) -> Option<Jit> {
        if !crate::jit::probe() {
            return None;
        }
        let exec_mem = ExecMem::new(CODE_SIZE)?;
        let base = exec_mem.base();
        let jump_cache = vec![JumpEntry::default(); 1 << JUMP_CACHE_BITS];
        let mut segment_addrs = Vec::with_capacity(SEGMENT_COUNT);
        let mut blocks_start = 0;
        for segment in 0..SEGMENT_COUNT {
            let segment_offset = segment * SEGMENT_SIZE;
            let prologue = arch::prologue(base + segment_offset as u64, jump_cache.as_ptr() as u64);
            exec_mem.write_bytes(segment_offset, &prologue.code);
            blocks_start = (prologue.code.len() + HEADER_ALIGN - 1) & !(HEADER_ALIGN - 1);
            segment_addrs.push(arch::addrs(&prologue, chaining));
        }
        Some(Jit {
            exec_mem,
            used_bytes: blocks_start,
            enter_trampoline: base,
            segment_addrs,
            blocks: Vec::new(),
            block_map: HashMap::new(),
            jump_cache,
            page_blocks: HashMap::new(),
            incoming: HashMap::new(),
            pending: HashMap::new(),
            blocks_start,
            stats: JitStats::default(),
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stats(&self) -> JitStats {
        self.stats
    }

    #[cfg(test)]
    fn link_state_len(&self) -> (usize, usize) {
        (
            self.incoming.values().map(Vec::len).sum(),
            self.pending.values().map(Vec::len).sum(),
        )
    }

    /// Points the slot at `destination`'s block, if the jump reaches. Returns whether it did.
    fn patch_link(&mut self, site: &LinkedSite, destination: u32) -> bool {
        let site_addr = self.exec_mem.base() + site.exec_offset as u64;
        let link_addr = self.blocks[destination as usize].link_addr;
        let Some(jump) = arch::link_bytes(site_addr, link_addr) else {
            return false;
        };
        self.exec_mem.write_bytes(site.exec_offset, &jump);
        self.incoming
            .entry(destination)
            .or_default()
            .push(site.clone());
        self.stats.linked += 1;
        true
    }

    /// Called from `Cpu::code_written` when a guest write hits a page that has blocks. The
    /// blocks are only marked invalid, not freed: the block that performed the write may still be
    /// running, and `Cpu::block_exit_requested` makes it return before reaching stale code.
    pub fn invalidate_page(&mut self, ppn: u64) {
        if let Some(indices) = self.page_blocks.remove(&ppn) {
            for index in indices {
                let block = &mut self.blocks[index as usize];
                block.valid = false;
                self.block_map.remove(&(block.rip, block.phys_addr));
                let slot = &mut self.jump_cache[jump_cache_hash(block.rip)];
                if slot.block == index as u64 + 1 {
                    *slot = JumpEntry::default();
                }
                // A direct jump into the block would skip every lookup, so it goes back to
                // falling through to the chain stub, and waits for the block to be translated
                // again. The invalidated block's own slots stay patched: it never runs again.
                for site in self.incoming.remove(&index).unwrap_or_default() {
                    self.exec_mem
                        .write_bytes(site.exec_offset, &arch::unlink_bytes());
                    self.stats.unlinked += 1;
                    self.pending.entry(site.destination).or_default().push(site);
                }
            }
        }
    }

    /// Drops every block and rewinds the bump pointer, keeping the prologue. Only called from
    /// `translate`, so no generated code is running.
    fn reset(&mut self) {
        self.blocks.clear();
        self.block_map.clear();
        self.page_blocks.clear();
        self.incoming.clear();
        self.pending.clear();
        self.jump_cache.fill(JumpEntry::default());
        self.used_bytes = self.blocks_start;
        self.stats.resets += 1;
    }

    /// The block translated for this `(rip, phys_addr)`, if any. A jump cache miss falls back to
    /// the block map and refills the slot.
    pub fn lookup(&mut self, rip: u64, phys_addr: u64) -> Option<u64> {
        let slot = &self.jump_cache[jump_cache_hash(rip)];
        if slot.code_addr != 0 && slot.rip == rip && slot.phys_addr == phys_addr {
            return Some(slot.code_addr);
        }
        let index = *self.block_map.get(&(rip, phys_addr))?;
        let code_addr = self.blocks[index as usize].code_addr;
        self.jump_cache[jump_cache_hash(rip)] = JumpEntry {
            rip,
            phys_addr,
            code_addr,
            block: index as u64 + 1,
        };
        Some(code_addr)
    }

    /// Decodes up to `MAX_INSNS` instructions at `rip` (physical `phys_addr`), compiles them and
    /// returns the host address of the block. A block never spans a page boundary, except the
    /// single-instruction block for an instruction that itself straddles one. `rip` must be
    /// `cpu.rip`: that case decodes through `Cpu::fetch_decode`, which reads it.
    pub fn translate(&mut self, cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Result<u64, Exception> {
        let mut segment = self.used_bytes / SEGMENT_SIZE;
        if self.used_bytes + CODE_HEADROOM > (segment + 1) * SEGMENT_SIZE {
            if segment + 1 < SEGMENT_COUNT {
                segment += 1;
                self.used_bytes = segment * SEGMENT_SIZE + self.blocks_start;
            } else {
                self.reset();
                segment = 0;
            }
        }
        let ram_len = cpu.ram.len() as u64;
        // Code can only execute from RAM; reported as a not-present fetch fault.
        if phys_addr >= ram_len {
            return Err(cpu.page_fault(rip, AccessKind::Exec, cpu.cpl == 3, false, false));
        }
        let bytes_left_in_page = (PAGE_SIZE - (phys_addr & PAGE_MASK)) as usize;
        let mut insns: Vec<Insn> = Vec::new();
        // Guest rip of each entry in `insns`.
        let mut insn_rips: Vec<u64> = Vec::new();
        let mut offset = 0usize;
        let mut crosses_page = false;
        let mut end_rip = rip;
        while insns.len() < MAX_INSNS {
            let window = (bytes_left_in_page - offset).min(MAX_INSN_LEN);
            if window == 0 {
                break;
            }
            let ram_offset = phys_addr as usize + offset;
            match decode::decode(&cpu.ram[ram_offset..ram_offset + window]) {
                Ok(insn) => {
                    insn_rips.push(rip + offset as u64);
                    offset += insn.len as usize;
                    end_rip = rip + offset as u64;
                    insns.push(insn);
                    if helpers::ends_block(insns.last().unwrap()) {
                        break;
                    }
                }
                // Ran off the page mid-instruction. If it is the first instruction the block is
                // that one instruction, decoded through the interpreter's page-crossing path;
                // otherwise end the block here and let the next one start on the new page.
                Err(DecodeErr::Truncated) => {
                    if insns.is_empty() {
                        let insn = cpu.fetch_decode()?;
                        insn_rips.push(rip);
                        end_rip = rip + insn.len as u64;
                        insns.push(insn);
                        crosses_page = true;
                    }
                    break;
                }
                // Undecodable bytes become a one-byte marker instruction that exec turns into
                // #UD when (and only if) the guest reaches it.
                Err(DecodeErr::Invalid) => {
                    let marker = Insn {
                        op: helpers::OP_INVALID,
                        len: 1,
                        ..Default::default()
                    };
                    insn_rips.push(rip + offset as u64);
                    end_rip = rip + offset as u64 + 1;
                    insns.push(marker);
                    break;
                }
            }
        }
        // The first iteration always runs (`MAX_INSNS > 0`, at least one byte is left in the page)
        // and either pushes an instruction, returns, or takes the first-instruction `Truncated`
        // arm, so a block is never empty. An empty one would compile to "jump to `rip`" and spin.
        debug_assert!(!insns.is_empty());
        // Makes guest stores to this page take the slow path so `code_written` sees them.
        cpu.mark_code(phys_addr);
        let mut pages = vec![phys_addr >> PAGE_SHIFT];
        if crosses_page {
            let next_page_phys = cpu.translate(
                (rip & !PAGE_MASK).wrapping_add(PAGE_SIZE),
                AccessKind::Exec,
                false,
            )?;
            pages.push(next_page_phys >> PAGE_SHIFT);
        }
        let insns: Box<[Insn]> = insns.into_boxed_slice();
        self.used_bytes = (self.used_bytes + BLOCK_ALIGN - 1) & !(BLOCK_ALIGN - 1);
        let code_addr = self.exec_mem.base() + self.used_bytes as u64;
        let compiled = arch::compile_block(
            code_addr,
            &self.segment_addrs[segment],
            &insns,
            &insn_rips,
            end_rip,
        );
        self.stats.inlined += compiled.inlined as u64;
        self.stats.flag_records_skipped += compiled.flag_records_skipped as u64;
        self.stats.total += insns.len() as u64;
        let block_offset = self.used_bytes;
        self.exec_mem.write_bytes(block_offset, &compiled.code);
        self.used_bytes += compiled.code.len();
        let link_addr = code_addr;
        let code_addr = link_addr + compiled.body_offset as u64;
        let index = self.blocks.len() as u32;
        self.block_map.insert((rip, phys_addr), index);
        self.blocks.push(Block {
            rip,
            phys_addr,
            code_addr,
            link_addr,
            valid: true,
            _insns: insns,
        });
        // Blocks waiting for this one, then this block's own exits. Only same-page targets are
        // linked: the block was entered through a validated mapping of its page, which is all
        // the chain stub's TLB probe re-establishes, and a page-straddling block's exit is not
        // in the page its key assumes.
        for site in self.pending.remove(&(rip, phys_addr)).unwrap_or_default() {
            self.patch_link(&site, index);
        }
        if !crosses_page {
            let page_base = phys_addr & !PAGE_MASK;
            for link in compiled.links {
                let destination = (link.target_rip, page_base | (link.target_rip & PAGE_MASK));
                let site = LinkedSite {
                    exec_offset: block_offset + link.code_offset as usize,
                    destination,
                };
                match self.block_map.get(&destination).copied() {
                    Some(target) => {
                        self.patch_link(&site, target);
                    }
                    None => self.pending.entry(destination).or_default().push(site),
                }
            }
        }
        for page in pages {
            self.page_blocks.entry(page).or_default().push(index);
        }
        self.jump_cache[jump_cache_hash(rip)] = JumpEntry {
            rip,
            phys_addr,
            code_addr,
            block: index as u64 + 1,
        };
        self.stats.translated += 1;
        Ok(code_addr)
    }
}

impl Cpu {
    /// Runs one block (or, on a miss, translates then runs it) starting at `self.rip`.
    ///
    /// The block is entered through the prologue trampoline as
    /// `extern "C" fn(cpu: *mut Cpu, block: u64) -> u64`. The trampoline pins the `Cpu` and TLB
    /// base registers, saves the callee-saved registers and jumps to `block`. The return value
    /// is built by the exit stubs in `arch::prologue` (`arch_a64.rs`, `arch_x64.rs`), which the
    /// code from `jit/emit.rs` / `jit/emit_x64.rs` jumps to:
    ///
    /// - `r >> helpers::EXIT_COUNT_SHIFT`: instructions fully retired by the block.
    /// - `r & helpers::EXIT_STATUS_MASK`: `STATUS_EXCEPTION` if a helper stored an exception in
    ///   `self.pending`, otherwise 0 or `STATUS_EXIT_REQUESTED` (both mean success).
    ///
    /// `Cpu::run` has already added 1 to `icount` for the first instruction (the interpreter
    /// counts an instruction before executing it, so a faulting one is counted too). The updates
    /// below make the total identical to what the interpreter would produce: `done` on success,
    /// `done + 1` on an exception. `Cpu::run_verify` relies on `last_block_insns` to replay the
    /// same number of instructions in the interpreter.
    pub fn step_block(&mut self) -> Result<(), Exception> {
        let rip = self.rip;
        let phys_addr = self.translate(rip, AccessKind::Exec, false)?;
        // `translate` needs `&mut Cpu`, hence the take. The Jit is put back before the block runs
        // because stores inside it reach `Cpu::code_written`, which needs `self.jit` to invalidate.
        let mut jit = self.jit.take().unwrap();
        let code = match jit.lookup(rip, phys_addr) {
            Some(code) => Ok(code),
            None => jit.translate(self, rip, phys_addr),
        };
        let enter = jit.enter_trampoline;
        self.jit = Some(jit);
        let code = code?;
        let run_block: extern "C" fn(*mut Cpu, u64) -> u64 =
            unsafe { std::mem::transmute(enter as usize) };
        let exit_code = run_block(self as *mut Cpu, code);
        let done = exit_code >> helpers::EXIT_COUNT_SHIFT;
        self.last_block_insns = done;
        // Set by the emitted code before rdtsc / port I/O at block index k > 0: it already added
        // k to `icount` so the instruction reads the exact time, and it must not be added twice.
        let prepaid_icount = std::mem::take(&mut self.prepaid_icount);
        self.block_exit_requested = false;
        match exit_code & helpers::EXIT_STATUS_MASK {
            helpers::STATUS_EXCEPTION => {
                self.icount += done - prepaid_icount;
                Err(self.pending.take().expect("pending exception"))
            }
            _ => {
                self.icount += done.saturating_sub(1) - prepaid_icount;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Block cache, jump cache, invalidation, code-buffer management and `Cpu::step_block`
    //! accounting, tested against real translated blocks.

    use super::*;
    use crate::cpu::exception::Exception;
    use crate::jit::serialize_jit_tests;
    use crate::testkit::*;

    const PAGE_BYTES_FOR_TESTS: u64 = PAGE_SIZE;
    /// Page offset of the snippet in the harness (`CODE_BASE` is page-aligned plus this).
    const SNIPPET_PAGE_OFFSET: u64 = CODE_BASE & PAGE_MASK;

    // add rax, rcx; hlt
    const ADD_THEN_HLT: [u8; 4] = [0x48, 0x01, 0xc8, 0xf4];
    // add rax, rcx; jmp 1f; 1: hlt
    const ADD_THEN_JMP_NEXT: [u8; 6] = [0x48, 0x01, 0xc8, 0xeb, 0x00, 0xf4];
    // add rax, rcx; add rax, rcx; ud2
    const ADD_THEN_UD2: [u8; 8] = [0x48, 0x01, 0xc8, 0x48, 0x01, 0xc8, 0x0f, 0x0b];
    // mov eax, 0x11223344
    const MOV_EAX_IMM32: [u8; 5] = [0xb8, 0x44, 0x33, 0x22, 0x11];
    // mov rax, [rbx + rcx*8 + 0x12345]
    const MOV_RAX_FROM_INDEXED_MEMORY: [u8; 8] = [0x48, 0x8b, 0x84, 0xcb, 0x45, 0x23, 0x01, 0x00];
    // nop
    const NOP: u8 = 0x90;
    // ud2-free invalid opcode (push es is not valid in 64-bit mode)
    const INVALID_OPCODE: [u8; 1] = [0x06];

    fn cpu_running(code: &[u8]) -> Cpu {
        Harness::new(code).cpu
    }

    fn new_jit() -> Jit {
        Jit::new(true).expect("the host has a JIT")
    }

    /// For tests of one block's accounting: the block returns to the caller instead of chaining on.
    fn new_unchained_jit() -> Jit {
        Jit::new(false).expect("the host has a JIT")
    }

    /// Translates the block at `CODE_BASE` (identity-mapped, so physical == linear).
    fn translate_at_code_base(jit: &mut Jit, cpu: &mut Cpu) -> u64 {
        jit.translate(cpu, CODE_BASE, CODE_BASE)
            .expect("translation succeeds")
    }

    fn nops(count: usize) -> Vec<u8> {
        vec![NOP; count]
    }

    #[test]
    fn a_new_jit_places_its_first_block_after_an_aligned_prologue() {
        let _guard = serialize_jit_tests();
        let jit = new_jit();
        assert_eq!(jit.enter_trampoline, jit.exec_mem.base());
        assert_eq!(jit.used_bytes, jit.blocks_start);
        assert_eq!(jit.blocks_start % HEADER_ALIGN, 0);
        assert!(jit.blocks_start > 0);
        assert!(jit.blocks.is_empty());
        assert!(jit.jump_cache.iter().all(|slot| slot.code_addr == 0));
        assert_eq!(jit.stats(), JitStats::default());
    }

    // aarch64 `b` reaches +-128 MiB from a block back to the stubs at the start of its segment.
    const _: () = assert!(SEGMENT_SIZE <= 128 << 20);
    const _: () = assert!(CODE_HEADROOM < SEGMENT_SIZE);

    #[test]
    fn lookup_misses_before_translation_and_hits_afterwards() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        let code = translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
        assert_eq!(jit.stats().translated, 1);
    }

    #[test]
    fn translated_blocks_are_aligned_and_do_not_overlap() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(8)[..]].concat());
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.translate(&mut cpu, CODE_BASE + 5, CODE_BASE + 5)
            .unwrap();
        let (first, second) = (jit.blocks[0].link_addr, jit.blocks[1].link_addr);
        assert_eq!(first % BLOCK_ALIGN as u64, 0);
        assert_eq!(second % BLOCK_ALIGN as u64, 0);
        assert!(first >= jit.exec_mem.base() + jit.blocks_start as u64);
        assert!(second > first);
        assert!(second < jit.exec_mem.base() + jit.used_bytes as u64);
    }

    #[test]
    fn lookup_requires_the_same_physical_address() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert!(jit.lookup(CODE_BASE, CODE_BASE).is_some());
        assert_eq!(
            jit.lookup(CODE_BASE, CODE_BASE + PAGE_BYTES_FOR_TESTS),
            None,
            "the same virtual address mapped to another page is another block"
        );
    }

    #[test]
    fn lookup_requires_the_same_rip() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.lookup(CODE_BASE + 1, CODE_BASE), None);
        assert_eq!(jit.lookup(CODE_BASE ^ (1 << 40), CODE_BASE), None);
    }

    #[test]
    fn jump_cache_hash_stays_in_range_and_folds_in_the_high_bits() {
        let _guard = serialize_jit_tests();
        for rip in [0, 1, CODE_BASE, u64::MAX, 0xffff_ffff_8000_0000, 1 << 63] {
            assert!(jump_cache_hash(rip) < 1 << JUMP_CACHE_BITS, "{rip:#x}");
        }
        // The bits above the cache index take part, so a rip one 64 KiB away does not alias.
        assert_ne!(
            jump_cache_hash(0x1000),
            jump_cache_hash(0x1000 + (1 << JUMP_CACHE_BITS))
        );
        // A deliberate collision: flipping bit 0 and bit 16 cancels out in `rip ^ (rip >> 16)`.
        assert_eq!(
            jump_cache_hash(CODE_BASE),
            jump_cache_hash(CODE_BASE ^ (1 << 16) ^ 1)
        );
    }

    #[test]
    fn a_colliding_rip_evicts_the_older_block_from_the_jump_cache_but_not_from_the_block_map() {
        let _guard = serialize_jit_tests();
        let colliding_rip = CODE_BASE ^ (1 << 16) ^ 1;
        assert_eq!(jump_cache_hash(colliding_rip), jump_cache_hash(CODE_BASE));
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let first = translate_at_code_base(&mut jit, &mut cpu);
        let second = jit
            .translate(&mut cpu, colliding_rip, colliding_rip)
            .unwrap();
        assert_ne!(first, second);
        assert_eq!(
            jit.jump_cache[jump_cache_hash(CODE_BASE)].code_addr,
            second,
            "the slot belongs to the newer block"
        );
        // The older block is still found, and found again as the slot's owner, without
        // translating a second copy of it.
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(first));
        assert_eq!(jit.jump_cache[jump_cache_hash(CODE_BASE)].code_addr, first);
        assert_eq!(jit.lookup(colliding_rip, colliding_rip), Some(second));
        assert_eq!(jit.blocks.len(), 2);
        assert_eq!(jit.stats().translated, 2);
    }

    #[test]
    fn invalidating_a_page_drops_its_blocks_but_keeps_their_slots() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let page = CODE_BASE >> PAGE_SHIFT;
        assert!(jit.page_blocks.contains_key(&page));
        jit.invalidate_page(page);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        assert!(!jit.blocks[0].valid);
        assert_eq!(jit.blocks.len(), 1, "indices held elsewhere stay valid");
        assert!(!jit.page_blocks.contains_key(&page));
    }

    #[test]
    fn invalidating_an_unrelated_page_changes_nothing() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let code = translate_at_code_base(&mut jit, &mut cpu);
        jit.invalidate_page((CODE_BASE >> PAGE_SHIFT) + 1);
        jit.invalidate_page(0);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
        assert!(jit.blocks[0].valid);
    }

    #[test]
    fn invalidating_a_page_twice_is_harmless() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let page = CODE_BASE >> PAGE_SHIFT;
        jit.invalidate_page(page);
        jit.invalidate_page(page);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
    }

    #[test]
    fn a_block_translated_after_invalidation_is_valid_again() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.invalidate_page(CODE_BASE >> PAGE_SHIFT);
        let code = translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
        assert_eq!(jit.stats().translated, 2);
    }

    #[test]
    fn translating_marks_the_page_so_stores_to_it_reach_code_written() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let page = (CODE_BASE >> PAGE_SHIFT) as usize;
        assert_eq!(cpu.page_has_code[page], 0);
        translate_at_code_base(&mut jit, &mut cpu);
        assert_ne!(cpu.page_has_code[page], 0);
    }

    #[test]
    fn a_block_includes_its_terminator_and_stops_there() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(8)[..]].concat());
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let block = &jit.blocks[0];
        assert_eq!(block._insns.len(), 2, "add and the jmp that ends the block");
        assert_eq!(jit.stats().total, 2);
    }

    #[test]
    fn a_block_holds_at_most_max_insns_instructions() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&nops(MAX_INSNS * 2 + 5));
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.blocks[0]._insns.len(), MAX_INSNS);
        let next_rip = CODE_BASE + MAX_INSNS as u64;
        jit.translate(&mut cpu, next_rip, next_rip).unwrap();
        assert_eq!(jit.blocks[1]._insns.len(), MAX_INSNS);
        jit.translate(
            &mut cpu,
            next_rip + MAX_INSNS as u64,
            next_rip + MAX_INSNS as u64,
        )
        .unwrap();
        // The five remaining nops are followed by zero bytes, which decode as `add [rax], al`.
        let third = &jit.blocks[2]._insns;
        assert!(third[..5].iter().all(|insn| insn.op == NOP as u16));
        assert_eq!(third[5].op, 0x00);
    }

    #[test]
    fn a_block_stops_at_the_end_of_its_page() {
        let _guard = serialize_jit_tests();
        let bytes_to_page_end = (PAGE_SIZE - SNIPPET_PAGE_OFFSET) as usize;
        let mut cpu = cpu_running(&nops(bytes_to_page_end + 10));
        let mut jit = new_jit();
        let page_end_rip = CODE_BASE + bytes_to_page_end as u64 - 3;
        jit.translate(&mut cpu, page_end_rip, page_end_rip).unwrap();
        assert_eq!(
            jit.blocks[0]._insns.len(),
            3,
            "three nops fit before the boundary"
        );
        assert_eq!(
            jit.page_blocks.len(),
            1,
            "and only the first page is involved"
        );
    }

    #[test]
    fn an_instruction_straddling_a_page_is_a_block_of_its_own_listed_under_both_pages() {
        let _guard = serialize_jit_tests();
        let straddle_start = PAGE_SIZE - 2;
        let padding = (straddle_start - SNIPPET_PAGE_OFFSET) as usize;
        let code = [&nops(padding)[..], &MOV_EAX_IMM32[..]].concat();
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        let rip = CODE_BASE + padding as u64;
        // The straddling case decodes through `Cpu::fetch_decode`, which reads `cpu.rip`.
        cpu.rip = rip;
        jit.translate(&mut cpu, rip, rip).unwrap();
        assert_eq!(jit.blocks[0]._insns.len(), 1);
        assert_eq!(jit.blocks[0]._insns[0].len, MOV_EAX_IMM32.len() as u8);
        let first_page = rip >> PAGE_SHIFT;
        assert!(jit.page_blocks.contains_key(&first_page));
        assert!(jit.page_blocks.contains_key(&(first_page + 1)));
        assert!(
            cpu.page_has_code[first_page as usize + 1] != 0,
            "stores to the second page must invalidate it too"
        );
        jit.invalidate_page(first_page + 1);
        assert_eq!(
            jit.lookup(rip, rip),
            None,
            "invalidating either page kills the block"
        );
    }

    #[test]
    fn a_block_before_a_straddling_instruction_ends_ahead_of_it() {
        let _guard = serialize_jit_tests();
        let straddle_start = PAGE_SIZE - 2;
        let padding = (straddle_start - SNIPPET_PAGE_OFFSET) as usize;
        let code = [&nops(padding)[..], &MOV_EAX_IMM32[..]].concat();
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        let rip = CODE_BASE + padding as u64 - 3;
        jit.translate(&mut cpu, rip, rip).unwrap();
        assert_eq!(
            jit.blocks[0]._insns.len(),
            3,
            "three nops, then the next block starts at the mov"
        );
        assert_eq!(jit.page_blocks.len(), 1);
    }

    #[test]
    fn translating_outside_ram_raises_a_page_fault_and_adds_no_block() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let outside = cpu.ram.len() as u64;
        let error = jit.translate(&mut cpu, outside, outside).unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 14, .. }),
            "{error:?}"
        );
        assert_eq!(cpu.cr2, outside);
        assert!(jit.blocks.is_empty());
        assert_eq!(jit.stats().translated, 0);
    }

    #[test]
    fn undecodable_bytes_become_a_one_byte_marker_block() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&INVALID_OPCODE);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let block = &jit.blocks[0];
        assert_eq!(block._insns.len(), 1);
        assert_eq!(block._insns[0].op, helpers::OP_INVALID);
        assert_eq!(block._insns[0].len, 1);
    }

    #[test]
    fn an_undecodable_byte_ends_the_block_after_the_instructions_before_it() {
        let _guard = serialize_jit_tests();
        let code = [&ADD_THEN_HLT[..3], &INVALID_OPCODE[..], &nops(4)[..]].concat();
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.blocks[0]._insns.len(), 2);
        assert_eq!(jit.blocks[0]._insns[1].op, helpers::OP_INVALID);
    }

    #[test]
    fn the_generated_code_in_the_buffer_is_the_compiler_output_verbatim() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let body_addr = translate_at_code_base(&mut jit, &mut cpu);
        let block = &jit.blocks[0];
        let code_addr = block.link_addr;
        let rips = [CODE_BASE, CODE_BASE + 3];
        let expected = arch::compile_block(
            code_addr,
            &jit.segment_addrs[0],
            &block._insns,
            &rips,
            CODE_BASE + 4,
        );
        let written =
            unsafe { std::slice::from_raw_parts(code_addr as *const u8, expected.code.len()) };
        assert_eq!(written, expected.code.as_slice());
        assert_eq!(body_addr, code_addr + expected.body_offset as u64);
        assert_eq!(code_addr, jit.exec_mem.base() + jit.blocks_start as u64);
    }

    #[test]
    fn the_exec_memory_write_path_leaves_the_prologue_untouched_by_blocks() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let prologue_len = jit.blocks_start;
        let before =
            unsafe { std::slice::from_raw_parts(jit.exec_mem.base() as *const u8, prologue_len) }
                .to_vec();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.translate(&mut cpu, CODE_BASE + 3, CODE_BASE + 3)
            .unwrap();
        let after =
            unsafe { std::slice::from_raw_parts(jit.exec_mem.base() as *const u8, prologue_len) };
        assert_eq!(after, before.as_slice());
    }

    #[test]
    fn a_worst_case_block_fits_in_the_headroom_with_a_wide_margin() {
        let _guard = serialize_jit_tests();
        // 48 inline loads, each with a slow path emitted out of line: the largest blocks the
        // compiler produces. `CODE_HEADROOM` is the free space required before translating.
        let code = MOV_RAX_FROM_INDEXED_MEMORY.repeat(MAX_INSNS);
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        let before = jit.used_bytes;
        translate_at_code_base(&mut jit, &mut cpu);
        let block_bytes = jit.used_bytes - before;
        assert_eq!(jit.blocks[0]._insns.len(), MAX_INSNS);
        assert!(
            block_bytes * 16 < CODE_HEADROOM,
            "{block_bytes} bytes for the largest block vs {CODE_HEADROOM} bytes of headroom"
        );
    }

    #[test]
    fn reset_discards_every_block_and_rewinds_to_the_first_block_slot() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_JMP_NEXT);
        let mut jit = new_jit();
        let first = translate_at_code_base(&mut jit, &mut cpu);
        jit.translate(&mut cpu, CODE_BASE + 5, CODE_BASE + 5)
            .unwrap();
        jit.reset();
        assert!(jit.blocks.is_empty());
        assert!(jit.page_blocks.is_empty());
        assert!(jit.jump_cache.iter().all(|slot| slot.code_addr == 0));
        assert_eq!(jit.used_bytes, jit.blocks_start);
        assert_eq!(jit.stats().resets, 1);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        assert_eq!(
            translate_at_code_base(&mut jit, &mut cpu),
            first,
            "the buffer is reused from the start"
        );
    }

    #[test]
    fn a_nearly_full_code_buffer_is_flushed_before_the_next_translation() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(4)[..]].concat());
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.used_bytes = jit.exec_mem.len() - CODE_HEADROOM + 1;
        let second_rip = CODE_BASE + 5;
        let code = jit.translate(&mut cpu, second_rip, second_rip).unwrap();
        assert_eq!(jit.stats().resets, 1);
        assert_eq!(
            jit.blocks.len(),
            1,
            "only the block translated after the flush remains"
        );
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        assert_eq!(jit.lookup(second_rip, second_rip), Some(code));
        assert_eq!(
            jit.blocks[0].link_addr,
            jit.exec_mem.base() + jit.blocks_start as u64
        );
    }

    #[test]
    fn a_buffer_with_exactly_the_headroom_free_is_not_flushed() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        jit.used_bytes = jit.exec_mem.len() - CODE_HEADROOM;
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.stats().resets, 0);
    }

    #[test]
    fn step_block_runs_a_block_and_accounts_for_its_instructions() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(1)[..]].concat());
        cpu.jit = Some(Box::new(new_unchained_jit()));
        cpu.gpr[0] = 1;
        cpu.gpr[1] = 2;
        // `Cpu::run` counts the first instruction before calling in.
        cpu.icount += 1;
        cpu.step_block().unwrap();
        assert_eq!(cpu.gpr[0], 3);
        assert_eq!(cpu.last_block_insns, 2);
        assert_eq!(cpu.icount, 2, "the block retired two instructions in total");
        assert_eq!(cpu.rip, CODE_BASE + 5);
    }

    #[test]
    fn step_block_reports_an_exception_with_the_faulting_instruction_counted() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_UD2);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.icount += 1;
        let error = cpu.step_block().unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 6, .. }),
            "{error:?}"
        );
        assert_eq!(cpu.last_block_insns, 2, "two adds retired before the ud2");
        assert_eq!(
            cpu.icount, 3,
            "and the faulting ud2 counts too, as in the interpreter"
        );
        assert!(
            cpu.pending.is_none(),
            "the exception was taken out of `pending`"
        );
    }

    #[test]
    fn step_block_clears_the_exit_request_and_the_prepaid_count() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.block_exit_requested = true;
        cpu.prepaid_icount = 0;
        cpu.icount += 1;
        cpu.step_block().unwrap();
        assert!(!cpu.block_exit_requested);
        assert_eq!(cpu.prepaid_icount, 0);
    }

    #[test]
    fn step_block_keeps_the_jit_when_translation_fails() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.rip = cpu.ram.len() as u64;
        let error = cpu.step_block().unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 14, .. }),
            "{error:?}"
        );
        assert!(cpu.jit.is_some());
    }

    #[test]
    fn step_block_with_an_unmapped_rip_faults_before_touching_the_cache() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.rip = 0x1_0000_0000;
        let error = cpu.step_block().unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 14, .. }),
            "{error:?}"
        );
        assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 0);
    }

    #[test]
    fn step_block_reuses_the_cached_block_on_the_second_run() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(1)[..]].concat());
        cpu.jit = Some(Box::new(new_jit()));
        for _ in 0..3 {
            cpu.rip = CODE_BASE;
            cpu.icount += 1;
            cpu.step_block().unwrap();
        }
        assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 1);
    }

    // mov ecx, 100; 1: sub ecx, 1; jnz 1b; hlt
    const COUNTDOWN_LOOP: [u8; 11] = [
        0xb9, 0x64, 0x00, 0x00, 0x00, 0x83, 0xe9, 0x01, 0x75, 0xfb, 0xf4,
    ];
    const RSP_INDEX: usize = 4;
    /// mov (1) + 100 * (sub, jnz) + hlt (1).
    const COUNTDOWN_LOOP_INSNS: u64 = 202;
    // mov eax, 7; call 1f; add eax, 1; hlt; 1: add eax, eax; ret
    const CALL_AND_RETURN: [u8; 17] = [
        0xb8, 0x07, 0x00, 0x00, 0x00, 0xe8, 0x04, 0x00, 0x00, 0x00, 0x83, 0xc0, 0x01, 0xf4, 0x01,
        0xc0, 0xc3,
    ];

    /// `Cpu::run`'s per-step bookkeeping around one `step_block` call.
    fn step_once(cpu: &mut Cpu) {
        cpu.interrupt_shadow = false;
        cpu.icount += 1;
        cpu.step_block().unwrap();
    }

    /// Calls `step_block` until the guest halts; returns how many calls it took.
    fn steps_to_halt(cpu: &mut Cpu) -> usize {
        for calls in 1..=1000 {
            step_once(cpu);
            if cpu.halted {
                return calls;
            }
        }
        panic!("never halted: {}", cpu.dump_regs());
    }

    /// Runs the countdown loop a few iterations, so its blocks exist and chain into each other,
    /// and stops mid-loop.
    fn warm_up_loop(cpu: &mut Cpu) {
        cpu.next_event = 20;
        while cpu.icount < 20 {
            step_once(cpu);
        }
        assert!(!cpu.halted);
        cpu.next_event = u64::MAX;
    }

    #[test]
    fn chained_blocks_run_a_whole_loop_without_returning_to_the_caller() {
        let _guard = serialize_jit_tests();
        let mut chained = cpu_running(&COUNTDOWN_LOOP);
        chained.jit = Some(Box::new(new_jit()));
        chained.next_event = u64::MAX;
        // Only blocks that do not exist yet send the chain back to the caller (to be translated).
        let chained_calls = steps_to_halt(&mut chained);
        assert!(chained_calls <= 3, "{chained_calls}");
        assert_eq!(chained.gpr[1], 0);
        assert_eq!(chained.icount, COUNTDOWN_LOOP_INSNS);

        let mut unchained = cpu_running(&COUNTDOWN_LOOP);
        unchained.jit = Some(Box::new(new_unchained_jit()));
        unchained.next_event = u64::MAX;
        assert!(steps_to_halt(&mut unchained) > 100);
        assert_eq!(unchained.icount, COUNTDOWN_LOOP_INSNS);
    }

    #[test]
    fn a_chain_stops_at_the_event_deadline_and_resumes_without_losing_instructions() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        cpu.next_event = cpu.icount + 50;
        let deadline = cpu.next_event;
        step_once(&mut cpu);
        assert!(!cpu.halted);
        assert!(
            cpu.icount >= deadline && cpu.icount < deadline + 4,
            "{} vs {deadline}",
            cpu.icount
        );
        // Every retired instruction is counted once and the next one is not pre-charged: ecx says
        // how many iterations ran.
        let iterations = 100 - cpu.gpr[1];
        assert_eq!(cpu.icount, 1 + 2 * iterations);
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
    }

    #[test]
    fn a_chained_call_and_return_find_their_blocks_through_the_jump_cache() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&CALL_AND_RETURN);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[0], 15);
        // Run it again from the top: now the call, the callee and the return all hit the cache and
        // the whole sequence is a single call.
        cpu.halted = false;
        cpu.rip = CODE_BASE;
        cpu.gpr[RSP_INDEX] = STACK_TOP;
        assert_eq!(steps_to_halt(&mut cpu), 1);
        assert_eq!(cpu.gpr[0], 15);
        assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 3);
    }

    #[test]
    fn a_chain_does_not_enter_a_block_whose_page_was_invalidated() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        let translated = cpu.jit.as_ref().unwrap().stats().translated;
        cpu.jit
            .as_mut()
            .unwrap()
            .invalidate_page(CODE_BASE >> PAGE_SHIFT);
        steps_to_halt(&mut cpu);
        assert!(
            cpu.jit.as_ref().unwrap().stats().translated > translated,
            "the stale blocks were retranslated, not chained into"
        );
    }

    #[test]
    fn same_page_exits_are_linked_directly_once_the_successor_exists() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[1], 0);
        assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
        assert!(cpu.jit.as_ref().unwrap().stats().linked >= 1);
    }

    #[test]
    fn an_unchained_jit_never_links() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_unchained_jit()));
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.jit.as_ref().unwrap().stats().linked, 0);
    }

    #[test]
    fn invalidating_the_destination_unlinks_and_a_retranslation_relinks() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        let before = cpu.jit.as_ref().unwrap().stats();
        assert!(before.linked >= 1);
        cpu.jit
            .as_mut()
            .unwrap()
            .invalidate_page(CODE_BASE >> PAGE_SHIFT);
        let invalidated = cpu.jit.as_ref().unwrap().stats();
        assert_eq!(invalidated.unlinked, before.linked);
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[1], 0);
        assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
        let after = cpu.jit.as_ref().unwrap().stats();
        assert!(after.linked > before.linked, "{after:?} vs {before:?}");
    }

    #[test]
    fn reset_discards_links_and_pending_entries() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        let jit = cpu.jit.as_mut().unwrap();
        assert_ne!(jit.link_state_len(), (0, 0));
        jit.reset();
        assert_eq!(jit.link_state_len(), (0, 0));
    }

    /// Instructions of the block at `CODE_BASE` that were lowered inline, not counting the final
    /// `hlt` (which always goes through the interpreter).
    fn inlined_in(code: &[u8]) -> u64 {
        let mut cpu = cpu_running(code);
        let mut jit = new_unchained_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.stats().inlined
    }

    #[test]
    fn narrow_shifts_by_a_count_below_the_width_are_lowered_inline() {
        let _guard = serialize_jit_tests();
        // shr ax, 5 | shl ax, 1 | sar al, 3 | shr al, cl | shl ax, cl | sar ax, cl
        for shift in [
            &[0x66, 0xc1, 0xe8, 0x05][..],
            &[0x66, 0xd1, 0xe0],
            &[0xc0, 0xf8, 0x03],
            &[0xd2, 0xe8],
            &[0x66, 0xd3, 0xe0],
            &[0x66, 0xd3, 0xf8],
        ] {
            assert_eq!(inlined_in(&[shift, &[0xf4]].concat()), 1, "{shift:x?}");
        }
    }

    #[test]
    fn bmi_shifts_bzhi_andn_and_the_blsr_group_are_lowered_inline() {
        let _guard = serialize_jit_tests();
        for code in [
            // shlx rax, rcx, rbx | shrx eax, ecx, ebx | sarx rax, [rsi], rbx
            &[0xc4, 0xe2, 0xe1, 0xf7, 0xc1][..],
            &[0xc4, 0xe2, 0x63, 0xf7, 0xc1],
            &[0xc4, 0xe2, 0xe2, 0xf7, 0x06],
            // bzhi rax, rcx, rbx | bzhi eax, [rsi], ebx
            &[0xc4, 0xe2, 0xe0, 0xf5, 0xc1],
            &[0xc4, 0xe2, 0x60, 0xf5, 0x06],
            // andn rax, rbx, rcx | blsr rax, rcx | blsmsk eax, ecx | blsi rax, [rsi]
            &[0xc4, 0xe2, 0xe0, 0xf2, 0xc1],
            &[0xc4, 0xe2, 0xf8, 0xf3, 0xc9],
            &[0xc4, 0xe2, 0x78, 0xf3, 0xd1],
            &[0xc4, 0xe2, 0xf8, 0xf3, 0x1e],
        ] {
            assert_eq!(inlined_in(&[code, &[0xf4]].concat()), 1, "{code:x?}");
        }
    }

    #[test]
    fn narrow_shifts_by_a_count_at_or_above_the_width_stay_in_the_interpreter() {
        let _guard = serialize_jit_tests();
        // shr ax, 16 | shl al, 8 | shr ax, 0
        for shift in [
            &[0x66, 0xc1, 0xe8, 0x10][..],
            &[0xc0, 0xe0, 0x08],
            &[0x66, 0xc1, 0xe8, 0x00],
        ] {
            assert_eq!(inlined_in(&[shift, &[0xf4]].concat()), 0, "{shift:x?}");
        }
    }

    #[test]
    fn a_flag_write_that_a_later_instruction_overwrites_is_not_recorded() {
        let _guard = serialize_jit_tests();
        let translate = |code: &[u8]| {
            let mut cpu = cpu_running(code);
            let mut jit = new_unchained_jit();
            translate_at_code_base(&mut jit, &mut cpu);
            jit.stats().flag_records_skipped
        };
        // add rax, rcx; add rdx, rsi; hlt
        assert_eq!(translate(&[0x48, 0x01, 0xc8, 0x48, 0x01, 0xf2, 0xf4]), 1);
        // adc in between reads the first add's carry, a load in between can fault.
        assert_eq!(translate(&[0x48, 0x01, 0xc8, 0x48, 0x11, 0xf2, 0xf4]), 0);
        assert_eq!(
            translate(&[0x48, 0x01, 0xc8, 0x48, 0x8b, 0x03, 0x48, 0x01, 0xf2, 0xf4]),
            0
        );
    }

    #[test]
    fn helper_histogram_counts_the_instructions_that_reach_the_interpreter() {
        let _guard = serialize_jit_tests();
        // adc eax, ebx three times (no inline lowering), then hlt.
        let adc = [0x13, 0xc3];
        let code = [adc, adc, adc, [0xf4, 0xf4]].concat();
        let mut cpu = cpu_running(&code);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.helper_histogram = Some(Default::default());
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        let histogram = cpu.helper_histogram.as_ref().unwrap();
        let adc_key = decode::decode(&adc).unwrap().histogram_key();
        assert_eq!(histogram.get(&adc_key), Some(&3));
        let add_key = decode::decode(&[0x01, 0xd8]).unwrap().histogram_key();
        assert_eq!(histogram.get(&add_key), None, "add is lowered inline");
    }

    #[test]
    fn the_histogram_key_tells_group_opcodes_apart_but_not_unrelated_registers() {
        // shl eax, 1 and rol eax, 1 are both d1 /n.
        let shl = decode::decode(&[0xd1, 0xe0]).unwrap().histogram_key();
        let rol = decode::decode(&[0xd1, 0xc0]).unwrap().histogram_key();
        assert_ne!(shl, rol);
        // mov eax, ebx and mov ecx, ebx are one instruction as far as the histogram goes.
        let mov_a = decode::decode(&[0x89, 0xd8]).unwrap().histogram_key();
        let mov_c = decode::decode(&[0x89, 0xd9]).unwrap().histogram_key();
        assert_eq!(mov_a, mov_c);
    }

    #[test]
    fn chain_exit_after_an_rdtsc_does_not_charge_the_prepaid_instructions_twice() {
        let _guard = serialize_jit_tests();
        // mov ecx, 3; 1: sub ecx, 1; rdtsc; jnz... rdtsc clobbers eax/edx only, flags survive.
        // mov ecx, 3; 1: rdtsc; sub ecx, 1; jnz 1b; hlt
        let code = [
            0xb9, 0x03, 0x00, 0x00, 0x00, 0x0f, 0x31, 0x83, 0xe9, 0x01, 0x75, 0xf8, 0xf4,
        ];
        assert_jit_matches_interpreter(&code, |_| {});
    }

    #[test]
    fn a_full_segment_moves_on_to_the_next_one_with_its_own_prologue_copy() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(4)[..]].concat());
        let mut jit = new_jit();
        let first = translate_at_code_base(&mut jit, &mut cpu);
        jit.used_bytes = SEGMENT_SIZE - CODE_HEADROOM + 1;
        let second_rip = CODE_BASE + 5;
        let second = jit.translate(&mut cpu, second_rip, second_rip).unwrap();
        assert_eq!(jit.stats().resets, 0, "other segments are still free");
        assert_eq!(
            jit.blocks[1].link_addr,
            jit.exec_mem.base() + (SEGMENT_SIZE + jit.blocks_start) as u64
        );
        assert_eq!(jit.lookup(second_rip, second_rip), Some(second));
        assert_eq!(
            jit.lookup(CODE_BASE, CODE_BASE),
            Some(first),
            "nothing was dropped"
        );
        assert_ne!(jit.segment_addrs[0].exit_ok, jit.segment_addrs[1].exit_ok);
        assert_eq!(
            jit.segment_addrs[1].exit_ok - jit.segment_addrs[0].exit_ok,
            SEGMENT_SIZE as u64,
            "identical prologue copies"
        );
    }

    #[test]
    fn blocks_in_later_segments_run() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_unchained_jit();
        jit.used_bytes = SEGMENT_SIZE * (SEGMENT_COUNT - 1) + jit.blocks_start;
        cpu.jit = Some(Box::new(jit));
        cpu.gpr[0] = 1;
        cpu.gpr[1] = 2;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[0], 3);
    }
}
