//! The WebAssembly backend's code cache and `Cpu::step_block`. Same job as `native.rs`, much less
//! machinery: there is no executable buffer to manage, no code to patch and no second thread to
//! share it with. A block is a module of its own (`emit_wasm`, `wasm_encode`); the host turns its
//! bytes into a function in the emulator's function table (`host_jit_compile`, implemented by
//! `www/jit_host.js`) and the block is then called through that table slot like any function
//! pointer: on wasm32 a pointer to a function is its table index.
//!
//! Because compiling a module costs the engine tens of microseconds, a block is only compiled once
//! its first instruction has run `COMPILE_THRESHOLD` times in the interpreter; until then
//! `step_block` interprets one instruction. Tests build with a threshold of zero.
//!
//! The table belongs to one instance, so a second thread (a wasm worker) cannot call the blocks of
//! the first: with this backend the vCPUs always take turns on one thread (`runs_on_threads`).
//! That is also why the jump cache is one table shared by every handle, with no locking: generated
//! code (the chain stub) reads it too.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;
use core::cell::UnsafeCell;
use core::mem::{offset_of, size_of};
use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use super::blocks::{
    BlockKey, Decoded, JUMP_CACHE_BITS, NO_SECOND_PAGE, PAGE_MASK, PAGE_SHIFT, PAGE_SIZE,
    SLOT_BLOCK_BITS, SLOT_BLOCK_MASK, STRADDLER_TAG, decode_block, jump_cache_hash,
};
use super::emit_wasm::{self, Addrs, ChainTables};
use super::helpers;
use super::wasm_encode::{self, Code, op};
use crate::cpu::Cpu;
use crate::cpu::decode::Insn;
use crate::cpu::exception::Exception;
use crate::cpu::mem::AccessKind;
use crate::ram::CodePages;
use crate::sync::{Arc, Mutex};

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    /// Instantiates the module at `ptr..ptr + len` against the emulator's memory and function
    /// table and returns the table slot of its `block` export, or -1 if the engine refused it.
    fn host_jit_compile(ptr: *const u8, len: usize) -> i32;
    /// Gives a slot back; the host empties it so the engine can free the instance.
    fn host_jit_release(slot: i32);
}

/// Times the first instruction of a block must run in the interpreter before the block is
/// compiled. The engine needs tens of microseconds per module, so this is what keeps the cold
/// start-up code of a boot (most of it runs once) from being slower than interpreting it.
const COMPILE_THRESHOLD: u32 = 40;
/// A `Heat` count that means the engine refused this block: it is interpreted from now on.
const NEVER_COMPILE: u32 = u32::MAX;
const HEAT_BITS: usize = 16;

/// Most blocks one cache holds before it is flushed. Every block is a live instance in the engine,
/// so this bounds that memory; the validity table is this long.
const MAX_BLOCKS: usize = 1 << 16;
const _: () = assert!(MAX_BLOCKS as u64 <= 1 << SLOT_BLOCK_BITS);

/// What the probe module returns.
const PROBE_RESULT: i64 = 0x7e57_ab1e;

fn compile_module(module: &[u8]) -> Option<u32> {
    // SAFETY: the host only reads the bytes during the call.
    let slot = unsafe { host_jit_compile(module.as_ptr(), module.len()) };
    u32::try_from(slot).ok()
}

fn release_slot(slot: u32) {
    // SAFETY: `slot` came from `host_jit_compile` and is released once.
    unsafe { host_jit_release(slot as i32) };
}

/// Whether the host compiles and runs generated modules, by doing it with the smallest one: a
/// function that returns a constant. Cached, so the host sees one probe however many machines are
/// built.
pub(crate) fn probe() -> bool {
    static RESULT: AtomicU8 = AtomicU8::new(PROBE_UNKNOWN);
    const PROBE_UNKNOWN: u8 = 0;
    const PROBE_WORKS: u8 = 1;
    const PROBE_FAILS: u8 = 2;
    match RESULT.load(Ordering::Relaxed) {
        PROBE_WORKS => return true,
        PROBE_FAILS => return false,
        _ => {}
    }
    let mut code = Code::default();
    code.i64_const(PROBE_RESULT).op(op::END);
    let works = compile_module(&wasm_encode::block_module(&code.bytes, 0, 0)).is_some_and(|slot| {
        let block: extern "C" fn(*mut Cpu) -> u64 = unsafe { core::mem::transmute(slot as usize) };
        let result = block(core::ptr::null_mut());
        release_slot(slot);
        result == PROBE_RESULT as u64
    });
    RESULT.store(
        if works { PROBE_WORKS } else { PROBE_FAILS },
        Ordering::Relaxed,
    );
    works
}

/// A function pointer on wasm32 is an index into the function table, which is how generated
/// code names the helpers.
fn addrs(out: *mut u64, chain: u32) -> Addrs {
    Addrs {
        exec: helpers::helper_exec as *const () as usize as u32,
        ld: helpers::helper_load_via_out as *const () as usize as u32,
        st: helpers::helper_store_x64 as *const () as usize as u32,
        ld_stack: helpers::helper_load_stack_via_out as *const () as usize as u32,
        st_stack: helpers::helper_store_stack_x64 as *const () as usize as u32,
        cond: helpers::helper_cond as *const () as usize as u32,
        out: out as usize as u32,
        chain,
    }
}

/// One slot of the direct-mapped jump cache. `#[repr(C)]` with a power-of-two size because the
/// chain stub (`emit_wasm::compile_chain_stub`) indexes the table and reads these fields at the
/// offsets `ChainTables` is given. `slot == 0` is empty: table slot 0 is the null function and
/// never holds a block.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct JumpEntry {
    rip: u64,
    phys_addr: u64,
    /// Index of the block in `State::blocks`; for a straddler the physical page number of its
    /// second page sits above `SLOT_BLOCK_BITS`, as in `native.rs`. The stub reads only the low
    /// word, which is the index: a straddler's `phys_addr` is tagged, so it never gets that far.
    block: u64,
    slot: u32,
    _pad: u32,
}

const _: () = assert!(size_of::<JumpEntry>().is_power_of_two());

/// The jump cache. Read and written by `Jit` and read by generated code, none of which can run at
/// the same time as another vCPU thread (see the module docs).
struct JumpCache(Box<[UnsafeCell<JumpEntry>]>);

// SAFETY: one thread at a time uses a wasm-backed `Jit`.
unsafe impl Sync for JumpCache {}

impl JumpCache {
    fn new() -> JumpCache {
        JumpCache(
            (0..1 << JUMP_CACHE_BITS)
                .map(|_| UnsafeCell::new(JumpEntry::default()))
                .collect(),
        )
    }

    fn get(&self, rip: u64) -> JumpEntry {
        unsafe { *self.0[jump_cache_hash(rip)].get() }
    }

    fn set(&self, rip: u64, entry: JumpEntry) {
        unsafe { *self.0[jump_cache_hash(rip)].get() = entry };
    }

    fn clear(&self) {
        for entry in self.0.iter() {
            unsafe { *entry.get() = JumpEntry::default() };
        }
    }

    fn base(&self) -> u32 {
        self.0.as_ptr() as usize as u32
    }
}

/// How many times the first instruction of a block at `phys` has run in the interpreter.
#[derive(Clone, Copy, Default)]
struct Heat {
    phys: u64,
    count: u32,
}

/// One translated block. Blocks are only ever added; an invalidated one keeps its slot (cleared in
/// `SharedJit::valid`) until the next flush, so indices held in `page_blocks` and the jump caches
/// stay stable.
struct Block {
    key: BlockKey,
    /// The function table slot the block's function occupies.
    slot: u32,
    /// Generated code calls `helper_exec` with the address of one of these, so they must outlive
    /// the code. Heap storage keeps the addresses stable when `blocks` reallocates.
    _insns: Box<[Insn]>,
}

/// A vCPU's handle on the code cache. The cache itself (`SharedJit`) belongs to the machine and
/// every handle (`sibling`) shares it.
pub(crate) struct Jit {
    shared: Arc<SharedJit>,
}

struct SharedJit {
    /// Per block index: 1 while the block may run, 0 once its page was written (or after a flush).
    valid: Box<[AtomicU8]>,
    jump_cache: JumpCache,
    /// Table slot of the chain stub, or 0 without chaining.
    chain_slot: u32,
    /// Counts `invalidate_page` calls: a translation decodes outside the lock, so it compares this
    /// before and after to see a write to a page it read (see `native.rs`).
    invalidations: AtomicU64,
    /// Where the generated code's slow-path loads leave their value. One slot is enough: the
    /// helper returns before anything else reads it.
    out: Box<AtomicU64>,
    compile_threshold: AtomicU32,
    #[cfg(test)]
    test_block_limit: core::sync::atomic::AtomicUsize,
    state: Mutex<State>,
}

struct State {
    blocks: Vec<Block>,
    /// Every live block by `BlockKey`.
    block_map: BTreeMap<BlockKey, u32>,
    /// Physical page number -> blocks that decoded bytes from it.
    page_blocks: BTreeMap<u64, Vec<u32>>,
    heat: Box<[Heat]>,
    stats: JitStats,
}

/// Translation counters, read by tests to check that the cache is used and invalidated as
/// intended (`Jit::stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct JitStats {
    /// Blocks compiled, including ones compiled again after an invalidation or a flush.
    pub translated: u64,
    /// Guest instructions lowered inline, i.e. not through `helper_exec`.
    pub inlined: u64,
    /// Guest instructions compiled, inline or not.
    pub total: u64,
    /// Times the whole cache was flushed because it was full.
    pub resets: u64,
    /// Lazy-flag records left out of compiled blocks as dead.
    pub flag_records_skipped: u64,
    /// Decodes thrown away because the page was written before the block was registered.
    pub stale_decodes: u64,
    /// Blocks the host refused to instantiate; they run in the interpreter.
    pub compile_failures: u64,
    /// Guest instructions the interpreter ran because their block was not hot yet.
    pub cold_steps: u64,
}

/// What `Jit::install` found.
enum Install {
    Ready(u32),
    /// A page the decode read was written since: decode again.
    Stale,
    /// No room for another block: flush the cache, then decode again.
    Full,
    /// The host refused the module.
    Refused,
}

impl SharedJit {
    fn lock(&self) -> impl core::ops::DerefMut<Target = State> + '_ {
        self.state.lock().unwrap()
    }
}

impl State {
    /// Drops every block and gives their slots back. No generated code may be running.
    fn flush(&mut self, shared: &SharedJit) {
        for block in self.blocks.drain(..) {
            release_slot(block.slot);
        }
        for flag in shared.valid.iter() {
            flag.store(0, Ordering::Release);
        }
        // The entries name blocks that are gone, and their indices are about to be reused.
        shared.jump_cache.clear();
        self.block_map.clear();
        self.page_blocks.clear();
        self.stats.resets += 1;
    }
}

impl Drop for SharedJit {
    fn drop(&mut self) {
        // A poisoned lock means a vCPU thread panicked; its blocks still hold slots. The
        // freestanding lock cannot fail, which makes the pattern irrefutable there only.
        #[allow(irrefutable_let_patterns)]
        if let Ok(mut state) = self.state.lock() {
            for block in state.blocks.drain(..) {
                release_slot(block.slot);
            }
        }
        if self.chain_slot != 0 {
            release_slot(self.chain_slot);
        }
    }
}

fn heat_index(phys_addr: u64) -> usize {
    ((phys_addr ^ (phys_addr >> HEAT_BITS)) as usize) & ((1 << HEAT_BITS) - 1)
}

impl Jit {
    /// `chaining` lets blocks tail-call their successor through the chain stub instead of returning
    /// to the run loop between blocks; the lockstep verifier needs one block per `step_block`, so
    /// it passes false.
    pub fn new(chaining: bool) -> Option<Jit> {
        if !probe() {
            return None;
        }
        let valid: Box<[AtomicU8]> = (0..MAX_BLOCKS).map(|_| AtomicU8::new(0)).collect();
        let jump_cache = JumpCache::new();
        let chain_slot = if chaining {
            let tables = ChainTables {
                jump_cache: jump_cache.base(),
                valid: valid.as_ptr() as usize as u32,
                entry_shift: size_of::<JumpEntry>().trailing_zeros(),
                entry_phys: offset_of!(JumpEntry, phys_addr) as u32,
                entry_block: offset_of!(JumpEntry, block) as u32,
                entry_slot: offset_of!(JumpEntry, slot) as u32,
                index_bits: JUMP_CACHE_BITS as u32,
            };
            // An engine without tail calls refuses the stub; blocks then return to the run loop.
            compile_module(&emit_wasm::compile_chain_stub(&tables)).unwrap_or(0)
        } else {
            0
        };
        let shared = Arc::new(SharedJit {
            valid,
            jump_cache,
            chain_slot,
            invalidations: AtomicU64::new(0),
            out: Box::new(AtomicU64::new(0)),
            compile_threshold: AtomicU32::new(if cfg!(freestanding) {
                COMPILE_THRESHOLD
            } else {
                0
            }),
            #[cfg(test)]
            test_block_limit: core::sync::atomic::AtomicUsize::new(usize::MAX),
            state: Mutex::new(State {
                blocks: Vec::new(),
                block_map: BTreeMap::new(),
                page_blocks: BTreeMap::new(),
                heat: vec![Heat::default(); 1 << HEAT_BITS].into_boxed_slice(),
                stats: JitStats::default(),
            }),
        });
        Some(Jit { shared })
    }

    /// Another vCPU's handle on the same cache: its blocks, its jump cache and its invalidations.
    pub fn sibling(&self) -> Jit {
        Jit {
            shared: self.shared.clone(),
        }
    }

    /// Nothing to do: the vCPUs share one thread, so guest accesses need no ordering.
    pub fn set_ordered_memory(&mut self, _on: bool) {}

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stats(&self) -> JitStats {
        self.shared.lock().stats
    }

    /// Compile a block the first time it is reached instead of after `COMPILE_THRESHOLD` runs.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn set_compile_threshold(&self, runs: u32) {
        self.shared.compile_threshold.store(runs, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn set_block_limit(&self, limit: usize) {
        self.shared.test_block_limit.store(limit, Ordering::Relaxed);
    }

    /// Only the native backend has a window between decoding and registering to widen.
    #[cfg(test)]
    pub fn set_install_pause_micros(&self, _micros: u64) {}

    #[cfg(test)]
    pub fn block_count(&self) -> usize {
        self.shared.lock().blocks.len()
    }

    #[cfg(test)]
    fn block_is_valid(&self, index: usize) -> bool {
        self.shared.valid[index].load(Ordering::Acquire) != 0
    }

    /// Called from `Cpu::code_written` when a guest write hits a page that has blocks: they stop
    /// being usable for every handle at once. They are only marked invalid, not freed: the block
    /// that performed the write may still be running, and `Cpu::block_exit_requested` makes it
    /// return before reaching stale code.
    ///
    /// Also clears the page's code flag under the lock, which is what `install` checks under the
    /// same lock (see `native.rs`).
    pub fn invalidate_page(&self, ppn: u64, flags: &CodePages) {
        let shared = &*self.shared;
        let mut state = shared.lock();
        shared.invalidations.fetch_add(1, Ordering::AcqRel);
        flags.set(ppn as usize, 0);
        let Some(indices) = state.page_blocks.remove(&ppn) else {
            return;
        };
        for index in indices {
            shared.valid[index as usize].store(0, Ordering::Release);
            let key = state.blocks[index as usize].key;
            state.block_map.remove(&key);
        }
    }

    /// The block translated for the code at this physical address, if any; `rip` only picks the
    /// jump cache slot.
    pub fn lookup(&mut self, rip: u64, phys_addr: u64) -> Option<u64> {
        let entry = self.shared.jump_cache.get(rip);
        if entry.slot != 0
            && entry.rip == rip
            && entry.phys_addr == phys_addr
            && self.shared.valid[entry.block as usize].load(Ordering::Relaxed) != 0
        {
            return Some(entry.slot as u64);
        }
        let state = self.shared.lock();
        let index = *state.block_map.get(&(phys_addr, NO_SECOND_PAGE))?;
        let slot = state.blocks[index as usize].slot;
        drop(state);
        self.shared.jump_cache.set(
            rip,
            JumpEntry {
                rip,
                phys_addr,
                block: index as u64,
                slot,
                _pad: 0,
            },
        );
        Some(slot as u64)
    }

    /// The straddling block cached for `rip`, if its slot is still good: the block is valid and the
    /// page after `rip`'s is still the physical page it was decoded from.
    pub fn lookup_straddler(&mut self, cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Option<u64> {
        let entry = self.shared.jump_cache.get(rip);
        if entry.phys_addr != phys_addr | STRADDLER_TAG || entry.rip != rip {
            return None;
        }
        let (slot, block) = (entry.slot, entry.block);
        if self.shared.valid[(block & SLOT_BLOCK_MASK) as usize].load(Ordering::Relaxed) == 0 {
            return None;
        }
        let next_page = cpu
            .translate(
                (rip & !PAGE_MASK).wrapping_add(PAGE_SIZE),
                AccessKind::Exec,
                false,
            )
            .ok()?;
        ((next_page >> PAGE_SHIFT) == block >> SLOT_BLOCK_BITS).then_some(slot as u64)
    }

    /// Whether the block at `phys_addr` has run often enough in the interpreter to be worth
    /// compiling. Counts this run.
    fn is_hot(&self, phys_addr: u64) -> bool {
        let threshold = self.shared.compile_threshold.load(Ordering::Relaxed);
        if threshold == 0 {
            return true;
        }
        let mut state = self.shared.lock();
        let heat = &mut state.heat[heat_index(phys_addr)];
        if heat.phys != phys_addr {
            *heat = Heat {
                phys: phys_addr,
                count: 0,
            };
        }
        if heat.count == NEVER_COMPILE {
            return false;
        }
        heat.count += 1;
        heat.count >= threshold
    }

    /// Decodes up to `MAX_INSNS` instructions at `rip` (physical `phys_addr`), compiles them and
    /// returns the table slot of the block, or `None` when the host refuses the module (which
    /// then stays interpreted). `rip` must be `cpu.rip`: a page-straddling instruction decodes
    /// through `Cpu::fetch_decode`, which reads it.
    fn try_translate(
        &mut self,
        cpu: &mut Cpu,
        rip: u64,
        phys_addr: u64,
    ) -> Result<Option<u64>, Exception> {
        loop {
            // Code can only execute from RAM; reported as a not-present fetch fault.
            if !cpu.ram.contains(phys_addr, 1) {
                return Err(cpu.page_fault(rip, AccessKind::Exec, cpu.cpl == 3, false, false));
            }
            // Makes guest stores to this page take the slow path so `code_written` sees them,
            // before the bytes are read.
            cpu.mark_code(phys_addr);
            let generation = self.shared.invalidations.load(Ordering::Acquire);
            let decoded = decode_block(cpu, rip, phys_addr)?;
            match self.install(decoded, rip, phys_addr, generation, &cpu.page_has_code) {
                Install::Ready(slot) => return Ok(Some(slot as u64)),
                Install::Stale => {}
                Install::Full => self.flush(cpu),
                Install::Refused => {
                    let mut state = self.shared.lock();
                    state.stats.compile_failures += 1;
                    state.heat[heat_index(phys_addr)] = Heat {
                        phys: phys_addr,
                        count: NEVER_COMPILE,
                    };
                    return Ok(None);
                }
            }
        }
    }

    /// `try_translate` for tests and callers that want a block: the host refusing is a failure.
    #[cfg(test)]
    pub fn translate(&mut self, cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Result<u64, Exception> {
        Ok(self
            .try_translate(cpu, rip, phys_addr)?
            .expect("the host compiles every generated module"))
    }

    /// The part of `try_translate` under the lock: the checks that make the decoded bytes still
    /// good, then compile and register the block.
    fn install(
        &mut self,
        decoded: Decoded,
        rip: u64,
        phys_addr: u64,
        generation: u64,
        flags: &CodePages,
    ) -> Install {
        let shared = self.shared.clone();
        let mut state = shared.lock();
        if shared.invalidations.load(Ordering::Acquire) != generation {
            state.stats.stale_decodes += 1;
            return Install::Stale;
        }
        if decoded
            .pages
            .iter()
            .any(|&page| flags.get(page as usize) == 0)
        {
            state.stats.stale_decodes += 1;
            return Install::Stale;
        }
        let key = (phys_addr, decoded.second_page);
        if let Some(&index) = state.block_map.get(&key) {
            let slot = state.blocks[index as usize].slot;
            self.fill_jump_cache(rip, phys_addr, decoded.second_page, slot, index);
            return Install::Ready(slot);
        }
        #[cfg(test)]
        let limit = shared
            .test_block_limit
            .load(Ordering::Relaxed)
            .min(MAX_BLOCKS);
        #[cfg(not(test))]
        let limit = MAX_BLOCKS;
        if state.blocks.len() >= limit {
            return Install::Full;
        }
        let Decoded {
            insns,
            insn_rips,
            end_rip,
            second_page,
            pages,
            ..
        } = decoded;
        let insns: Box<[Insn]> = insns.into_boxed_slice();
        let compiled = emit_wasm::compile_block(
            &addrs(shared.out.as_ptr(), shared.chain_slot),
            &insns,
            &insn_rips,
            end_rip,
        );
        let Some(slot) = compile_module(&compiled.module) else {
            return Install::Refused;
        };
        state.stats.inlined += compiled.inlined as u64;
        state.stats.flag_records_skipped += compiled.flag_records_skipped as u64;
        state.stats.total += insns.len() as u64;
        let index = state.blocks.len() as u32;
        state.block_map.insert(key, index);
        state.blocks.push(Block {
            key,
            slot,
            _insns: insns,
        });
        shared.valid[index as usize].store(1, Ordering::Release);
        for page in pages {
            state.page_blocks.entry(page).or_default().push(index);
        }
        state.stats.translated += 1;
        self.fill_jump_cache(rip, phys_addr, second_page, slot, index);
        Install::Ready(slot)
    }

    /// A straddling block's slot is tagged (`STRADDLER_TAG`) so `lookup` never takes it and
    /// `lookup_straddler` finds it after checking the second page.
    fn fill_jump_cache(
        &mut self,
        rip: u64,
        phys_addr: u64,
        second_page: u64,
        slot: u32,
        index: u32,
    ) {
        let entry = if second_page == NO_SECOND_PAGE {
            JumpEntry {
                rip,
                phys_addr,
                block: index as u64,
                slot,
                _pad: 0,
            }
        } else {
            JumpEntry {
                rip,
                phys_addr: phys_addr | STRADDLER_TAG,
                block: index as u64 | (second_page >> PAGE_SHIFT) << SLOT_BLOCK_BITS,
                slot,
                _pad: 0,
            }
        };
        self.shared.jump_cache.set(rip, entry);
    }

    /// Flushes the whole cache, which is full. Nothing runs generated code at this point (a block
    /// is only entered from `step_block`, which is not on the stack).
    fn flush(&mut self, cpu: &mut Cpu) {
        let shared = self.shared.clone();
        cpu.with_exclusive(|_| shared.lock().flush(&shared));
    }
}

impl Cpu {
    /// Runs one block (or, on a miss, translates then runs it) starting at `self.rip`; a block that
    /// is not hot yet is not compiled, and one instruction runs in the interpreter instead.
    ///
    /// The block is a function `(cpu: *mut Cpu) -> u64` in the function table, returning
    /// `(retired_insns << helpers::EXIT_COUNT_SHIFT) | status` exactly as the native blocks do
    /// (see `native.rs`, whose accounting of `icount` this repeats).
    pub fn step_block(&mut self) -> Result<(), Exception> {
        let rip = self.rip;
        let phys_addr = self.translate(rip, AccessKind::Exec, false)?;
        // `translate` needs `&mut Cpu`, hence the take. The Jit is put back before the block runs
        // because stores inside it reach `Cpu::code_written`, which needs `self.jit` to invalidate.
        let mut jit = self.jit.take().unwrap();
        let cached = match jit.lookup_straddler(self, rip, phys_addr) {
            Some(slot) => Some(slot),
            None => jit.lookup(rip, phys_addr),
        };
        let slot = match cached {
            Some(slot) => Some(slot),
            None if !jit.is_hot(phys_addr) => None,
            None => match jit.try_translate(self, rip, phys_addr) {
                Ok(slot) => slot,
                Err(exception) => {
                    self.jit = Some(jit);
                    return Err(exception);
                }
            },
        };
        if slot.is_none() {
            jit.shared.lock().stats.cold_steps += 1;
        }
        self.jit = Some(jit);
        let Some(slot) = slot else {
            // `Cpu::run` already counted this instruction, as `step` expects.
            self.last_block_insns = 1;
            return self.step();
        };
        // SAFETY: the host put a function of this type in the slot (`wasm_encode::block_module`).
        let run_block: extern "C" fn(*mut Cpu) -> u64 =
            unsafe { core::mem::transmute(slot as usize) };
        let exit_code = run_block(self as *mut Cpu);
        let done = exit_code >> helpers::EXIT_COUNT_SHIFT;
        self.last_block_insns = done;
        // Set by `helper` before an instruction at block index k > 0 that may read the clock: it
        // already added k to `icount`, which must not be added twice.
        let prepaid_icount = core::mem::take(&mut self.prepaid_icount);
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
    //! The cache against real generated modules. They need a host that compiles them, which on the
    //! test target is `hack/wasm/run-wasi.mjs`; the differential tests in `diff_*.rs` skip
    //! silently when `Jit::new` is `None`, so these also pin that it is not.

    use super::*;
    use crate::testkit::*;

    // add rax, rcx; hlt
    const ADD_THEN_HLT: [u8; 4] = [0x48, 0x01, 0xc8, 0xf4];
    // mov ecx, 100; l: sub ecx, 1; jnz l; hlt
    const COUNTDOWN_LOOP: [u8; 11] = [
        0xb9, 0x64, 0x00, 0x00, 0x00, 0x83, 0xe9, 0x01, 0x75, 0xfb, 0xf4,
    ];

    #[test]
    fn the_host_compiles_and_runs_a_generated_module() {
        assert!(probe());
        assert!(Jit::new(true).is_some());
    }

    #[test]
    fn a_harness_with_the_jit_really_runs_translated_blocks() {
        let mut harness = Harness::new(&ADD_THEN_HLT);
        assert!(harness.enable_jit());
        harness.cpu.gpr[0] = 40;
        harness.cpu.gpr[1] = 2;
        assert_eq!(harness.run(1000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[0], 42);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, 1);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.inlined, 1, "the add is lowered, the hlt is not");
        assert_eq!(stats.cold_steps, 0);
    }

    #[test]
    fn a_loop_is_translated_once_and_run_from_the_cache() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, 3, "setup, loop body, hlt");
        assert_eq!(stats.inlined, stats.total - 1);
    }

    #[test]
    fn cold_blocks_are_interpreted_until_they_run_often_enough() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        harness.cpu.jit.as_ref().unwrap().set_compile_threshold(10);
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert!(stats.cold_steps >= 18, "{stats:?}");
        assert_eq!(stats.translated, 1, "only the loop head got hot: {stats:?}");
    }

    #[test]
    fn invalidating_a_page_makes_its_blocks_untranslated_again() {
        let mut harness = Harness::new(&ADD_THEN_HLT);
        assert!(harness.enable_jit());
        let ppn = CODE_BASE >> PAGE_SHIFT;
        let mut jit = harness.cpu.jit.take().unwrap();
        let slot = jit
            .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(slot));
        assert!(jit.block_is_valid(0));
        jit.invalidate_page(ppn, &harness.cpu.page_has_code);
        assert!(!jit.block_is_valid(0));
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        let again = jit
            .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        assert_eq!(jit.stats().translated, 2);
        assert_ne!(again, 0);
    }

    #[test]
    fn a_full_cache_is_flushed_and_translation_carries_on() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        harness.cpu.jit.as_ref().unwrap().set_block_limit(1);
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert!(stats.resets >= 1, "{stats:?}");
        assert!(harness.cpu.jit.as_ref().unwrap().block_count() <= 1);
    }

    /// Runs the loop the way `Cpu::run` does and counts the calls to `step_block`.
    fn step_block_calls_for(jit: Jit, event_budget: u64) -> (u64, Cpu) {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        harness.cpu.jit = Some(Box::new(jit));
        harness.cpu.next_event = event_budget;
        let mut calls = 0;
        while !harness.cpu.halted && calls < 1000 {
            harness.cpu.interrupt_shadow = false;
            harness.cpu.icount += 1;
            harness.cpu.step_block().unwrap();
            calls += 1;
            if harness.cpu.icount >= harness.cpu.next_event {
                harness.cpu.next_event = u64::MAX;
            }
        }
        (calls, harness.cpu)
    }

    #[test]
    fn chained_blocks_continue_without_returning_to_the_run_loop() {
        let _ = Harness::new(&[0xf4]);
        let (chained_calls, chained) = step_block_calls_for(Jit::new(true).unwrap(), u64::MAX);
        let (unchained_calls, unchained) = step_block_calls_for(Jit::new(false).unwrap(), u64::MAX);
        assert!(chained.halted && unchained.halted);
        assert_eq!(
            chained.icount, unchained.icount,
            "the accounting is the same"
        );
        assert_eq!(chained.gpr, unchained.gpr);
        assert!(
            unchained_calls > 100,
            "one call per block: {unchained_calls}"
        );
        assert!(
            chained_calls < 5,
            "the loop runs inside one call: {chained_calls}"
        );
    }

    #[test]
    fn a_chain_stops_at_the_event_budget_so_the_run_loop_can_service_events() {
        // 11 instructions of setup, then 2 per iteration: the budget ends the chain mid-loop, the
        // run loop takes over (and here lifts the budget), and the count stays exact.
        let (calls, chained) = step_block_calls_for(Jit::new(true).unwrap(), 50);
        let (_, reference) = step_block_calls_for(Jit::new(false).unwrap(), u64::MAX);
        assert!(calls >= 2, "the budget interrupted the chain: {calls}");
        assert!(chained.halted);
        assert_eq!(chained.icount, reference.icount);
    }

    #[test]
    fn siblings_share_one_cache() {
        let mut harness = Harness::new(&ADD_THEN_HLT);
        assert!(harness.enable_jit());
        let mut first = harness.cpu.jit.take().unwrap();
        let mut second = first.sibling();
        let slot = first
            .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        assert_eq!(second.lookup(CODE_BASE, CODE_BASE), Some(slot));
        assert_eq!(
            second
                .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
                .unwrap(),
            slot
        );
        assert_eq!(first.stats().translated, 1);
    }
}
