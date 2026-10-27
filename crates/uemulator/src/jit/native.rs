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

/// One translated block. Blocks are only ever added; an invalidated block keeps its slot (and its
/// `valid == false` flag) until `Jit::reset`, so indices held in `page_blocks` and the jump cache
/// stay stable.
pub(crate) struct Block {
    rip: u64,
    /// Physical address of the first instruction. Part of the lookup key together with `rip`: the
    /// code embeds rip-relative constants, and a remapped virtual page must not hit a stale block.
    phys_addr: u64,
    code_addr: u64,
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
            blocks_start,
            stats: JitStats::default(),
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stats(&self) -> JitStats {
        self.stats
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
            }
        }
    }

    /// Drops every block and rewinds the bump pointer, keeping the prologue. Only called from
    /// `translate`, so no generated code is running.
    fn reset(&mut self) {
        self.blocks.clear();
        self.block_map.clear();
        self.page_blocks.clear();
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
        let (code, inlined) = arch::compile_block(
            code_addr,
            &self.segment_addrs[segment],
            &insns,
            &insn_rips,
            end_rip,
        );
        self.stats.inlined += inlined as u64;
        self.stats.total += insns.len() as u64;
        self.exec_mem.write_bytes(self.used_bytes, &code);
        self.used_bytes += code.len();
        let index = self.blocks.len() as u32;
        self.block_map.insert((rip, phys_addr), index);
        self.blocks.push(Block {
            rip,
            phys_addr,
            code_addr,
            valid: true,
            _insns: insns,
        });
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
#[path = "native_tests.rs"]
mod tests;
