#[cfg(freestanding)]
use crate::prelude::*;
pub(crate) mod decode;
pub(crate) mod exception;
mod exec;
pub(crate) mod flags;
mod fpu;
pub(crate) mod mem;
mod sse;
mod sys;
#[cfg(test)]
pub(crate) mod test_support;

use core::sync::atomic::{AtomicBool, Ordering};

use crate::sync::Arc;

use crate::boot::BootState;
use crate::console::ConsoleHandle;
use crate::dev::guestmem::GuestMem;
use crate::dev::{DevConfig, Devices};
use crate::machine::{ExitReason, LogFn, PollFn};

use decode::Insn;
use flags::LazyFlagsOp;

pub(crate) use exception::Exception;

pub(crate) type ExecResult<T> = Result<T, Exception>;

/// Outcome of `Cpu::before_step`.
enum BeforeStep {
    /// Leave the run loop with this reason.
    Exit(ExitReason),
    /// The CPU was halted and has just idled (advanced time, ticked devices); poll again
    /// instead of executing.
    Idled,
    /// Execute the next instruction or block.
    Execute,
}

/// Virtual time per retired instruction. `now_ns`/`tsc` are derived from `icount`, which makes
/// guest time deterministic unless the clock is realtime.
pub(crate) const NS_PER_INSN: u64 = 1;
/// Instructions between `service_events` polls (stop flag, device timers). The check happens
/// at step boundaries, so with the JIT it can overshoot by up to one block.
const EVENT_INTERVAL: u64 = 4096;

pub(crate) const PAGE_SHIFT: u32 = 12;
pub(crate) const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
/// Mask of the offset within a page; `!PAGE_OFFSET_MASK` selects the page frame.
pub(crate) const PAGE_OFFSET_MASK: u64 = PAGE_SIZE - 1;

/// Indices into `Cpu::gpr`, which follows the x86 register encoding (rax=0 ... r15=15).
pub(crate) const GPR_RSP: usize = 4;
pub(crate) const GPR_RSI: usize = 6;

/// Indices into `Cpu::seg`, the same numbering as the segment-override and `mov sreg` encodings.
pub(crate) const SEG_ES: usize = 0;
pub(crate) const SEG_CS: usize = 1;
pub(crate) const SEG_SS: usize = 2;
pub(crate) const SEG_DS: usize = 3;
pub(crate) const SEG_FS: usize = 4;
pub(crate) const SEG_GS: usize = 5;

pub(crate) const USER_CPL: u8 = 3;

pub(crate) const CR0_PE: u64 = 1 << 0;
pub(crate) const CR0_MP: u64 = 1 << 1;
pub(crate) const CR0_ET: u64 = 1 << 4;
pub(crate) const CR0_NE: u64 = 1 << 5;
/// Supervisor writes honor read-only PTEs (consulted by the page walk).
pub(crate) const CR0_WP: u64 = 1 << 16;
pub(crate) const CR0_PG: u64 = 1 << 31;
pub(crate) const CR4_PAE: u64 = 1 << 5;
pub(crate) const EFER_LME: u64 = 1 << 8;
pub(crate) const EFER_LMA: u64 = 1 << 10;
/// Without NXE the PTE bit 63 is reserved, and instruction fetches never fault on it.
pub(crate) const EFER_NXE: u64 = 1 << 11;

/// Power-on PAT: PA0=WB, PA1=WT, PA2=UC-, PA3=UC, repeated for PA4..7.
const PAT_POWER_ON_DEFAULT: u64 = 0x0007_0406_0007_0406;
/// DR6 and DR7 reset values; the set bits are reserved-as-one.
const DR6_RESET: u64 = 0xffff_0ff0;
const DR7_RESET: u64 = 0x400;
/// All SSE exceptions masked, round to nearest.
const MXCSR_RESET: u32 = 0x1f80;
/// x87 control word after FINIT: all exceptions masked, 64-bit precision, round to nearest.
const FCW_RESET: u16 = 0x37f;

/// A segment register or system-segment cache: the selector plus the hidden descriptor parts.
///
/// `flags` is the upper dword of the descriptor (bytes 4-7) masked with
/// `SEG_DESCRIPTOR_FLAGS_MASK`, so the bit positions below match the in-memory descriptor.
/// `limit` is already scaled by the granularity bit. The JIT reads `base` of FS and GS with
/// `offset_of!(Seg, base)` and `size_of::<Seg>()`.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Seg {
    pub sel: u16,
    pub base: u64,
    pub limit: u32,
    pub flags: u32,
}

pub(crate) const SEG_TYPE_READ_WRITE: u32 = 1 << 9;
pub(crate) const SEG_TYPE_EXECUTABLE: u32 = 1 << 11;
/// The descriptor's S bit: set for code/data segments, clear for system segments and gates.
pub(crate) const SEG_NON_SYSTEM: u32 = 1 << 12;
pub(crate) const SEG_DPL_SHIFT: u32 = 13;
pub(crate) const SEG_DPL_MASK: u32 = 3;
pub(crate) const SEG_PRESENT: u32 = 1 << 15;
/// 64-bit code segment (CS.L).
pub(crate) const SEG_LONG: u32 = 1 << 21;
/// D/B: default operand size for code, big stack for data.
pub(crate) const SEG_DEFAULT_BIG: u32 = 1 << 22;
pub(crate) const SEG_GRANULARITY: u32 = 1 << 23;
/// The descriptor bits that `Seg::flags` keeps: type/S/DPL/P (bits 8-15) and AVL/L/D/G (20-23).
/// Base and limit are stored separately.
pub(crate) const SEG_DESCRIPTOR_FLAGS_MASK: u32 = 0x00f0_ff00;
/// Limit bits 19:16 of a descriptor sit in bits 16-19 of the upper dword; 0xf matches the
/// 4 GiB flat limit. They fall outside `SEG_DESCRIPTOR_FLAGS_MASK`, so only synthesized
/// segments carry them.
const SEG_FLAT_LIMIT_HIGH_BITS: u32 = 0xf << 16;
/// Limit of a flat 4 GiB segment.
pub(crate) const SEG_LIMIT_4GIB: u32 = 0xffff_ffff;
/// Flat 4 GiB read/write data segment, DPL 0.
pub(crate) const SEG_FLAGS_FLAT_DATA: u32 = SEG_GRANULARITY
    | SEG_DEFAULT_BIG
    | SEG_FLAT_LIMIT_HIGH_BITS
    | SEG_PRESENT
    | SEG_NON_SYSTEM
    | SEG_TYPE_READ_WRITE;
/// Execute/read 64-bit code segment, DPL 0.
pub(crate) const SEG_FLAGS_CODE64: u32 =
    SEG_LONG | SEG_PRESENT | SEG_NON_SYSTEM | SEG_TYPE_EXECUTABLE | SEG_TYPE_READ_WRITE;

impl Seg {
    pub fn dpl(&self) -> u8 {
        ((self.flags >> SEG_DPL_SHIFT) & SEG_DPL_MASK) as u8
    }
    pub fn present(&self) -> bool {
        self.flags & SEG_PRESENT != 0
    }
    /// A code segment, as opposed to data or a system descriptor.
    pub fn is_code(&self) -> bool {
        self.flags & SEG_TYPE_EXECUTABLE != 0 && self.flags & SEG_NON_SYSTEM != 0
    }
    pub fn long(&self) -> bool {
        self.flags & SEG_LONG != 0
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DescriptorTable {
    pub base: u64,
    pub limit: u16,
}

/// One slot of the direct-mapped soft TLB.
///
/// Each tag is the page-aligned linear address (or `TLB_TAG_INVALID`) for which that kind of
/// access is allowed, so a single compare proves both the translation and the permission.
/// `tag_w` is additionally withheld for pages whose PTE is not yet dirty and for pages that
/// hold translated code, which forces those stores through the slow path. MMIO and
/// out-of-RAM pages are never cached. `addend` is the host address of the page minus its
/// linear address (wrapping), so `host_ptr = la + addend`.
///
/// The JIT's inline probe (`tlb_probe` in `jit/emit.rs` and `jit/emit_x64.rs`) hard-codes this
/// layout: `tag_r` at +0, `tag_w` at +8, `addend` at +24, a 32-byte stride. `tag_x` is only
/// read by the interpreter but must stay where it is to keep `addend` at +24.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct TlbEntry {
    pub tag_r: u64,
    pub tag_w: u64,
    pub tag_x: u64,
    pub addend: u64,
}

/// Never equals a page-aligned address, so an empty tag always misses.
pub(crate) const TLB_TAG_INVALID: u64 = u64::MAX;
pub(crate) const TLB_EMPTY: TlbEntry = TlbEntry {
    tag_r: TLB_TAG_INVALID,
    tag_w: TLB_TAG_INVALID,
    tag_x: TLB_TAG_INVALID,
    addend: 0,
};

/// Permissions of a page-walk result, ANDed across all levels (`walk` in `mem.rs`). They are
/// not stored in `TlbEntry`; `translate` folds them into the per-access tags.
pub(crate) const TLB_FLAG_WRITABLE: u8 = 1;
pub(crate) const TLB_FLAG_USER: u8 = 2;
pub(crate) const TLB_FLAG_NO_EXECUTE: u8 = 4;
/// The leaf PTE already has its dirty bit, so a store can skip the walk that would set it.
pub(crate) const TLB_FLAG_DIRTY: u8 = 8;
pub(crate) const TLB_INDEX_BITS: usize = 10;
/// `Cpu::tlb` holds two halves of this many entries: supervisor first, then user. They are
/// separate because the same page can be accessible to one privilege and not the other, and a
/// CPL change then only has to swap `tlb_active_base` instead of flushing.
pub(crate) const TLB_ENTRIES_PER_MODE: usize = 1 << TLB_INDEX_BITS;

/// Decoded instruction cache slot, keyed by guest physical address. Valid only while
/// `generation` equals `Cpu::decode_cache_generation`.
#[derive(Clone, Copy)]
pub(crate) struct DecodeCacheEntry {
    pub gpa: u64,
    pub generation: u64,
    pub insn: Insn,
}

pub(crate) const DECODE_CACHE_ENTRIES: usize = 1 << 16;
/// Entries kept in `Cpu::trace` (debug only).
const TRACE_CAPACITY: usize = 64;
/// How many of the newest `trace` entries `dump` prints.
const TRACE_DUMP_ENTRIES: usize = 16;
/// Registers per line in the register dumps.
const GPRS_PER_DUMP_LINE: usize = 4;

/// All architectural and emulator state of the single vCPU.
///
/// `#[repr(C)]` keeps declaration order: the JIT writes `lazy_op` and `lazy_size` with one
/// 16-bit store, so `lazy_size` must directly follow `lazy_op`. Every other field the JIT
/// touches (`gpr`, `rip`, `icount`, `block_exit_requested`, `prepaid_icount`, `lazy_dst`,
/// `lazy_src`, `seg[].base`, `tlb_active_base`) is found with `offset_of!`.
#[repr(C)]
pub(crate) struct Cpu {
    pub gpr: [u64; 16],
    /// Address of the instruction being executed. The JIT stores it before anything that can
    /// fault, so exceptions and the faulting-instruction restart see an exact value.
    pub rip: u64,
    /// Everything but the `ARITH` flags, which are stale while `lazy_op` is not `Eflags`.
    /// Read it through `eflags_full()`.
    pub eflags: u64,
    /// Deferred arithmetic flags, see `LazyFlagsOp`. `lazy_size` is the operand size in bytes.
    pub lazy_op: LazyFlagsOp,
    pub lazy_size: u8,
    pub lazy_dst: u64,
    pub lazy_src: u64,
    pub lazy_src2: u64,
    /// Address of the half of `tlb` for the current privilege (the second half at CPL 3).
    /// The JIT latches it into a host register on block entry (`jit/arch_a64.rs`,
    /// `jit/arch_x64.rs`), so it must follow every CPL change (use `set_cpl`) and `tlb` must
    /// never be reallocated.
    pub tlb_active_base: u64,
    /// Current privilege level; change it only through `set_cpl`.
    pub cpl: u8,
    /// Set when a store hit a page holding translated code (`code_written`). Generated code
    /// polls it after stores and `helper_exec` reports it, so the running block stops before it
    /// can execute stale instructions; `step_block` clears it.
    pub block_exit_requested: bool,
    /// Instructions the JIT already added to `icount` in the middle of a block, before
    /// rdtsc/port I/O, so those read an up-to-date clock. `step_block` subtracts it from the
    /// block's total.
    pub prepaid_icount: u64,
    /// Indexed by `SEG_*`.
    pub seg: [Seg; 6],
    pub gdtr: DescriptorTable,
    pub idtr: DescriptorTable,
    pub ldtr: Seg,
    pub tr: Seg,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    /// Stored only; there is no local APIC to apply the task priority to.
    pub cr8: u64,
    pub efer: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub sfmask: u64,
    /// IA32_KERNEL_GS_BASE, exchanged with the GS base by SWAPGS.
    pub kgsbase: u64,
    pub tsc_aux: u64,
    pub pat: u64,
    /// IA32_SYSENTER_CS, _ESP, _EIP (MSRs 0x174..=0x176).
    pub sysenter: [u64; 3],
    pub dr: [u64; 8],
    /// MSRs that are only stored: a read returns the last write and writes are accepted.
    /// An MSR that is neither handled in `rdmsr`/`wrmsr` nor listed here raises #GP, which
    /// the kernel's `rdmsr_safe` probing depends on. Seeded in `Cpu::new`.
    pub msr_misc: alloc::collections::BTreeMap<u32, u64>,
    pub xmm: [u128; 16],
    pub mxcsr: u32,
    pub fcw: u16,
    /// TOP is in bits 11-13.
    pub fsw: u16,
    /// Abridged tag word: one bit per physical register, set when it holds a value.
    pub ftw: u16,
    /// x87 registers as raw 80-bit values, indexed by physical register; ST(i) is
    /// `st[(TOP + i) & 7]`.
    pub st: [[u8; 10]; 8],
    /// Set by HLT, cleared when an interrupt is delivered.
    pub halted: bool,
    /// One-instruction interrupt inhibit after `mov ss` and after an `sti` that turned IF on.
    /// `run` clears it right after `before_step` has consulted it.
    pub interrupt_shadow: bool,
    /// Guest physical RAM starting at address 0. It must never be resized or moved after
    /// `Cpu::new`, because `ram_base` and the TLB addends point into it.
    pub ram: Vec<u8>,
    /// Host address of `ram[0]`.
    pub ram_base: u64,
    /// Two halves of `TLB_ENTRIES_PER_MODE` entries, supervisor then user. Fixed size, so its
    /// address is stable for `tlb_active_base`.
    pub tlb: Vec<TlbEntry>,
    pub decode_cache: Vec<DecodeCacheEntry>,
    /// Bumping it invalidates the whole decode cache at once. Starts at 1 so the zeroed
    /// entries (generation 0) never match. 64 bits wide so it cannot wrap back to a value
    /// that stale entries still carry (a u32 would after about 4e9 flushes).
    pub decode_cache_generation: u64,
    /// One byte per guest physical page: nonzero while a cached decode or a JIT block was
    /// built from that page. Stores to such pages take the slow path and call `code_written`.
    /// Devices see it through `GuestMem` for the same purpose.
    pub page_has_code: Vec<u8>,
    /// Pages whose `page_has_code` byte is set, so `flush_decode` can clear them without
    /// scanning the whole array. May list pages already cleared by `code_written`.
    pub pages_with_code: Vec<u32>,
    pub dev: Devices,
    /// Instructions retired; guest time is derived from it. `run` adds one before an
    /// instruction executes, so an instruction that faults is counted too.
    pub icount: u64,
    /// Virtual time added while halted: the clock jumps (or sleeps, when realtime) to the next
    /// device deadline instead of executing instructions.
    pub idle_skipped_ns: u64,
    /// `icount` at which `service_events` runs next.
    pub next_event: u64,
    pub stop: Arc<AtomicBool>,
    pub log: Option<LogFn>,
    /// Host callback run on every device tick; its argument is true when the guest is idle with
    /// nothing scheduled (only needed on UEFI, where the host has to be polled for input).
    pub poll: Option<PollFn>,
    /// Keys already reported by `log_once`.
    pub logged: alloc::collections::BTreeSet<u64>,
    /// Number of `flush_decode` calls, for `dump`.
    pub flushes: u64,
    /// Wall-clock mode: halting really sleeps instead of skipping ahead.
    pub realtime: bool,
    /// Instructions the last JIT block retired before it exited; `run_verify` replays that many
    /// on the interpreter.
    pub last_block_insns: u64,
    /// Exception raised inside a JIT helper. Generated code only gets a status code back;
    /// `step_block` picks the exception up from here.
    pub pending: Option<Exception>,
    pub jit: Option<Box<crate::jit::Jit>>,
    /// Most recent instruction addresses, recorded only when debugging.
    pub trace: Option<alloc::collections::VecDeque<u64>>,
}

impl Cpu {
    /// Starts in 64-bit long mode with paging already on, as the Linux 64-bit boot protocol
    /// leaves it (`plans/02-boot.md`): there is no firmware to set any of this up.
    pub fn new(
        ram: Vec<u8>,
        boot: BootState,
        console: ConsoleHandle,
        stop: Arc<AtomicBool>,
        log: Option<LogFn>,
        dev_config: DevConfig,
    ) -> Self {
        // Stored-only MSRs; names follow Linux's msr-index.h.
        const MSR_UCODE_REV: u32 = 0x8b;
        const MSR_K7_HWCR: u32 = 0xc001_0015;
        const MSR_AMD64_NB_CFG: u32 = 0xc001_001f;
        const MSR_AMD64_LS_CFG: u32 = 0xc001_1020;
        const MSR_AMD64_DE_CFG: u32 = 0xc001_1029;
        const HWCR_INITIAL: u64 = 1 << 24;

        let pages = ram.len() >> PAGE_SHIFT;
        let flat = |sel: u16, flags: u32| Seg {
            sel,
            base: 0,
            limit: SEG_LIMIT_4GIB,
            flags,
        };
        let flat_data = flat(boot.ds, SEG_FLAGS_FLAT_DATA);
        let mut seg = [Seg::default(); 6];
        seg[SEG_ES] = flat_data;
        seg[SEG_CS] = flat(boot.cs, SEG_FLAGS_CODE64);
        seg[SEG_SS] = flat_data;
        seg[SEG_DS] = flat_data;
        let mut gpr = [0u64; 16];
        gpr[GPR_RSI] = boot.rsi;
        let mut cpu = Cpu {
            gpr,
            rip: boot.rip,
            eflags: flags::RESERVED_ONE,
            lazy_op: LazyFlagsOp::Eflags,
            lazy_size: 8,
            lazy_dst: 0,
            lazy_src: 0,
            lazy_src2: 0,
            seg,
            gdtr: DescriptorTable {
                base: boot.gdt_base,
                limit: boot.gdt_limit,
            },
            idtr: DescriptorTable::default(),
            ldtr: Seg::default(),
            tr: Seg::default(),
            cr0: CR0_PG | CR0_NE | CR0_ET | CR0_MP | CR0_PE,
            cr2: 0,
            cr3: boot.cr3,
            cr4: CR4_PAE,
            cr8: 0,
            efer: EFER_LMA | EFER_LME,
            star: 0,
            lstar: 0,
            cstar: 0,
            sfmask: 0,
            kgsbase: 0,
            tsc_aux: 0,
            pat: PAT_POWER_ON_DEFAULT,
            sysenter: [0; 3],
            dr: [0, 0, 0, 0, 0, 0, DR6_RESET, DR7_RESET],
            msr_misc: [
                (MSR_UCODE_REV, 0),
                (MSR_K7_HWCR, HWCR_INITIAL),
                (MSR_AMD64_NB_CFG, 0),
                (MSR_AMD64_LS_CFG, 0),
                (MSR_AMD64_DE_CFG, 0),
            ]
            .into_iter()
            .collect(),
            cpl: 0,
            xmm: [0; 16],
            mxcsr: MXCSR_RESET,
            fcw: FCW_RESET,
            fsw: 0,
            ftw: 0,
            st: [[0; 10]; 8],
            halted: false,
            interrupt_shadow: false,
            ram,
            tlb: vec![TLB_EMPTY; 2 * TLB_ENTRIES_PER_MODE],
            ram_base: 0,
            decode_cache: vec![
                DecodeCacheEntry {
                    gpa: 0,
                    generation: 0,
                    insn: Insn::default()
                };
                DECODE_CACHE_ENTRIES
            ],
            decode_cache_generation: 1,
            page_has_code: vec![0; pages],
            pages_with_code: Vec::new(),
            dev: Devices::new(console, dev_config),
            icount: 0,
            idle_skipped_ns: 0,
            next_event: 0,
            stop,
            log,
            poll: None,
            logged: Default::default(),
            flushes: 0,
            realtime: false,
            last_block_insns: 0,
            pending: None,
            jit: None,
            tlb_active_base: 0,
            block_exit_requested: false,
            prepaid_icount: 0,
            trace: None,
        };
        cpu.ram_base = cpu.ram.as_ptr() as u64;
        cpu.set_cpl(0);
        cpu
    }

    /// Switches the privilege level and the TLB half that goes with it.
    pub fn set_cpl(&mut self, cpl: u8) {
        self.cpl = cpl;
        let user_half_offset = TLB_ENTRIES_PER_MODE * core::mem::size_of::<TlbEntry>();
        self.tlb_active_base =
            self.tlb.as_ptr() as u64 + (cpl == USER_CPL) as u64 * user_half_offset as u64;
    }

    pub fn now_ns(&self) -> u64 {
        self.icount * NS_PER_INSN + self.idle_skipped_ns
    }

    /// The TSC is just guest time in nanoseconds.
    pub fn tsc(&self) -> u64 {
        self.now_ns()
    }

    /// Logs `msg` at most once per `key`. Callers keep their key ranges disjoint by adding the
    /// object id (page, MSR number) to a per-site base.
    pub fn log_once(&mut self, key: u64, msg: impl FnOnce() -> String) {
        if self.logged.insert(key)
            && let Some(l) = &self.log
        {
            l(&msg());
        }
    }

    pub fn tick_devices(&mut self, now: u64) {
        if let Some(poll) = &self.poll {
            poll(false);
        }
        let written_code_pages = {
            let mut mem = GuestMem::new(&mut self.ram, &self.page_has_code);
            self.dev.tick(now, &mut mem);
            mem.code_pages_written
        };
        // DMA by devices bypasses the store path, so tell the code caches about it.
        for page in written_code_pages {
            self.code_written(page);
        }
    }

    fn service_events(&mut self) -> Option<ExitReason> {
        if self.stop.load(Ordering::Relaxed) {
            return Some(ExitReason::Stopped);
        }
        let now = self.now_ns();
        self.tick_devices(now);
        if self.dev.reset_requested {
            return Some(ExitReason::Reset);
        }
        self.next_event = self.icount + EVENT_INTERVAL;
        None
    }

    fn pending_interrupt(&mut self) -> Option<u8> {
        if self.eflags & flags::IF == 0 || self.interrupt_shadow {
            return None;
        }
        self.dev.pic.ack()
    }

    /// Runs the per-step housekeeping: device/stop polling, interrupt delivery, halt handling.
    /// Interrupts are only taken here, i.e. at instruction (or JIT block) boundaries.
    fn before_step(&mut self) -> BeforeStep {
        if self.icount >= self.next_event
            && let Some(reason) = self.service_events()
        {
            return BeforeStep::Exit(reason);
        }
        // has_pending() is a cheap pre-check; pending_interrupt() does the acknowledge.
        if self.eflags & flags::IF != 0
            && !self.interrupt_shadow
            && self.dev.pic.has_pending()
            && let Some(vector) = self.pending_interrupt()
        {
            self.halted = false;
            if let Some(reason) = self.deliver_hardware_interrupt(vector) {
                return BeforeStep::Exit(reason);
            }
        }
        if self.halted {
            return match self.idle() {
                Some(reason) => BeforeStep::Exit(reason),
                None => BeforeStep::Idled,
            };
        }
        BeforeStep::Execute
    }

    /// Delivers the exception an instruction raised; `Some` means the run loop must exit.
    fn after_step(&mut self, result: ExecResult<()>) -> Option<ExitReason> {
        match result {
            Ok(()) => None,
            Err(Exception::Fault {
                vector,
                error_code,
                software,
            }) => self.deliver_event(vector, error_code, software),
            Err(Exception::Exit(reason)) => Some(reason),
        }
    }

    pub fn run(&mut self) -> ExitReason {
        loop {
            match self.before_step() {
                BeforeStep::Exit(reason) => return reason,
                BeforeStep::Idled => continue,
                BeforeStep::Execute => {}
            }
            self.interrupt_shadow = false;
            self.icount += 1;
            let result = if self.jit.is_some() {
                self.step_block()
            } else {
                self.step()
            };
            if let Some(reason) = self.after_step(result) {
                return reason;
            }
        }
    }

    /// Lockstep check of the JIT against the interpreter oracle (`--verify`): `self` runs a
    /// JIT block, then `shadow` (interpreter, own copy of RAM) executes the same number of
    /// instructions, and any difference in registers or exception stops the run.
    pub fn run_verify(&mut self, shadow: &mut Cpu) -> ExitReason {
        // Bytes of the diverging code printed from each side, and differing page offsets listed.
        const DIVERGENCE_CODE_BYTES: usize = 24;
        const MAX_DIFFERING_OFFSETS: usize = 8;

        loop {
            let (main_gate, shadow_gate) = (self.before_step(), shadow.before_step());
            match (main_gate, shadow_gate) {
                (BeforeStep::Exit(reason), _) => return reason,
                (BeforeStep::Idled, BeforeStep::Idled) => continue,
                (BeforeStep::Execute, BeforeStep::Execute) => {}
                _ => return ExitReason::Internal("verify: pre-step divergence".into()),
            }
            self.interrupt_shadow = false;
            shadow.interrupt_shadow = false;
            let main_before = (self.gpr, self.eflags_full(), self.rip);
            let shadow_before = (shadow.rip, shadow.icount, self.icount);
            let start = self.rip;
            self.icount += 1;
            shadow.icount += 1;
            let main_result = self.step_block();
            let retired = self.last_block_insns;
            // A faulting block reports the instructions retired before the fault, so the
            // interpreter needs one more step to reach the same fault.
            let attempts = if main_result.is_err() {
                retired + 1
            } else {
                retired
            };
            let mut shadow_result = Ok(());
            for attempt in 0..attempts {
                if attempt > 0 {
                    shadow.icount += 1;
                }
                shadow_result = shadow.step();
                if shadow_result.is_err() {
                    break;
                }
            }
            let same_outcome = match (&main_result, &shadow_result) {
                (Ok(()), Ok(())) => true,
                (
                    Err(Exception::Fault {
                        vector: main_vec,
                        error_code: main_err,
                        ..
                    }),
                    Err(Exception::Fault {
                        vector: shadow_vec,
                        error_code: shadow_err,
                        ..
                    }),
                ) => main_vec == shadow_vec && main_err == shadow_err,
                (Err(Exception::Exit(_)), Err(Exception::Exit(_))) => true,
                _ => false,
            };
            let mismatch = self.diff(shadow);
            if !same_outcome || mismatch.is_some() {
                let page_base = self
                    .translate(start, mem::AccessKind::Exec, true)
                    .unwrap_or(0) as usize
                    & !(PAGE_OFFSET_MASK as usize);
                // The faulting code may sit in the last RAM page or outside RAM (MMIO, where
                // `translate` fell back to 0), so every slice here is bounds-checked: a panic in
                // the diagnostics would hide the divergence being reported.
                let page = page_base..page_base + PAGE_SIZE as usize;
                let differing_offsets: Vec<usize> =
                    match (self.ram.get(page.clone()), shadow.ram.get(page.clone())) {
                        (Some(main_page), Some(shadow_page)) => (0..main_page.len())
                            .filter(|&offset| main_page[offset] != shadow_page[offset])
                            .take(MAX_DIFFERING_OFFSETS)
                            .collect(),
                        _ => Vec::new(),
                    };
                let code_offset = page_base + (start & PAGE_OFFSET_MASK) as usize;
                let code_bytes = |ram: &[u8]| {
                    ram.get(code_offset..code_offset + DIVERGENCE_CODE_BYTES)
                        .map(<[u8]>::to_vec)
                        .unwrap_or_default()
                };
                let main_bytes = code_bytes(&self.ram);
                let shadow_bytes = code_bytes(&shadow.ram);
                if let Some(l) = &self.log {
                    l(&format!(
                        "page diffs at {differing_offsets:x?}; main bytes {main_bytes:02x?}; shadow bytes {shadow_bytes:02x?}"
                    ));
                }
                let msg = format!(
                    "verify: divergence after block at {start:#x} ({retired} insns, attempts {attempts}, shadow start {shadow_before:x?}): exc main={:?} shadow={:?} diff={:?}\n--- before {main_before:x?}\n--- main\n{}--- shadow\n{}",
                    main_result.as_ref().err(),
                    shadow_result.as_ref().err(),
                    mismatch,
                    self.dump_regs(),
                    shadow.dump_regs()
                );
                return ExitReason::Internal(msg);
            }
            if let Some(reason) = self.after_step(main_result) {
                return reason;
            }
            if let Some(reason) = shadow.after_step(shadow_result) {
                return reason;
            }
        }
    }

    /// Describes how architectural state differs from `other`, or `None` if it matches.
    fn diff(&mut self, other: &mut Cpu) -> Option<String> {
        let mut differences = Vec::new();
        for index in 0..self.gpr.len() {
            if self.gpr[index] != other.gpr[index] {
                differences.push(format!(
                    "r{index}: {:#x} vs {:#x}",
                    self.gpr[index], other.gpr[index]
                ));
            }
        }
        if self.rip != other.rip {
            differences.push(format!("rip: {:#x} vs {:#x}", self.rip, other.rip));
        }
        let (our_eflags, their_eflags) = (self.eflags_full(), other.eflags_full());
        if our_eflags != their_eflags {
            differences.push(format!("eflags: {our_eflags:#x} vs {their_eflags:#x}"));
        }
        if self.cpl != other.cpl {
            differences.push("cpl".into());
        }
        for index in 0..self.seg.len() {
            if self.seg[index].sel != other.seg[index].sel
                || self.seg[index].base != other.seg[index].base
            {
                differences.push(format!("seg{index}"));
            }
        }
        for (name, ours, theirs) in [
            ("cr0", self.cr0, other.cr0),
            ("cr2", self.cr2, other.cr2),
            ("cr3", self.cr3, other.cr3),
            ("cr4", self.cr4, other.cr4),
            ("efer", self.efer, other.efer),
            ("icount", self.icount, other.icount),
        ] {
            if ours != theirs {
                differences.push(format!("{name}: {ours:#x} vs {theirs:#x}"));
            }
        }
        if self.xmm != other.xmm {
            differences.push("xmm".into());
        }
        if differences.is_empty() {
            None
        } else {
            Some(differences.join(", "))
        }
    }

    /// Handles a halted CPU: skips (or sleeps) to the next device deadline so timers can raise
    /// the interrupt that wakes it. Returns `Some` for stop, reset and power-off.
    fn idle(&mut self) -> Option<ExitReason> {
        if self.stop.load(Ordering::Relaxed) {
            return Some(ExitReason::Stopped);
        }
        // A reset written just before the `hlt` (`out 0x64, 0xfe; hlt`, with IF=0 as in every
        // kernel's reboot path) must win over the shutdown check below.
        if self.dev.reset_requested {
            return Some(ExitReason::Reset);
        }
        // `hlt` with IF=0 can only be woken by NMI, SMI, INIT or RESET (SDM vol. 3 sec. 6.8,
        // HLT), and nothing here raises the first three: the guest is dead, which is how Linux's
        // poweroff and halt end without ACPI (`stop_this_cpu`: `cli; hlt` in a loop). Only IF is
        // tested, not `interrupt_shadow` or a queued event: `sti; hlt` has IF=1 by the time
        // `hlt` runs, and a PIC line pending behind IF=0 never wakes the CPU either.
        if self.eflags & flags::IF == 0 {
            return Some(ExitReason::Shutdown);
        }
        let now = self.now_ns();
        match self.dev.next_deadline(now) {
            Some(deadline) => {
                if deadline > now {
                    #[cfg(not(freestanding))]
                    let advance_ns = if self.realtime {
                        // Short sleeps keep the stop flag and host console responsive.
                        const MAX_SLEEP_NS: u64 = 2_000_000;
                        let sleep_ns = (deadline - now).min(MAX_SLEEP_NS);
                        std::thread::sleep(std::time::Duration::from_nanos(sleep_ns));
                        sleep_ns
                    } else {
                        deadline - now
                    };
                    #[cfg(freestanding)]
                    let advance_ns = deadline - now;
                    self.idle_skipped_ns += advance_ns;
                }
                self.tick_devices(self.now_ns());
            }
            None => {
                // No timer will ever wake the guest; only host input (console, network) can.
                self.tick_devices(now);
                if self.dev.pic.has_pending() {
                    return None;
                }
                #[cfg(not(freestanding))]
                std::thread::sleep(std::time::Duration::from_millis(1));
                #[cfg(freestanding)]
                if let Some(poll) = &self.poll {
                    poll(true);
                }
            }
        }
        if self.dev.reset_requested {
            return Some(ExitReason::Reset);
        }
        None
    }

    pub(crate) fn step(&mut self) -> ExecResult<()> {
        let insn = self.fetch_decode()?;
        if let Some(trace) = &mut self.trace {
            if trace.len() >= TRACE_CAPACITY {
                trace.pop_front();
            }
            trace.push_back(self.rip);
        }
        self.exec(&insn)
    }

    pub fn dump_regs(&self) -> String {
        let mut s = format!(
            "rip={:016x} cpl={} eflags={:x}\n",
            self.rip,
            self.cpl,
            self.eflags_full()
        );
        for (i, g) in self.gpr.iter().enumerate() {
            s += &format!("r{i:<2}={g:016x} ");
            if i % GPRS_PER_DUMP_LINE == GPRS_PER_DUMP_LINE - 1 {
                s += "\n";
            }
        }
        s
    }

    /// State dump for `--debug` and fatal exits. Also scans guest RAM for the kernel's
    /// `Linux version` banner: the printk ring buffer starts with it, so the text that follows
    /// is the kernel log even though the guest never got to flush it to the serial port.
    pub fn dump(&self) -> String {
        // The banner also appears in the kernel image, hence several matches.
        const MAX_BANNER_MATCHES: usize = 6;
        // Log text extracted after each match.
        const LOG_SCAN_BYTES: usize = 6000;
        // Ring-buffer records have binary headers between the text; shorter printable runs
        // are noise from those.
        const MIN_PRINTABLE_RUN: usize = 6;
        const BANNER: &[u8] = b"Linux version";

        let mut s = format!(
            "flushes={} icount={} rip={:016x} cpl={} eflags={:x} cr0={:x} cr3={:x} cr4={:x} efer={:x}\n",
            self.flushes,
            self.icount,
            self.rip,
            self.cpl,
            self.eflags_full(),
            self.cr0,
            self.cr3,
            self.cr4,
            self.efer
        );
        for (i, g) in self.gpr.iter().enumerate() {
            s += &format!("r{i:<2}={g:016x} ");
            if i % GPRS_PER_DUMP_LINE == GPRS_PER_DUMP_LINE - 1 {
                s += "\n";
            }
        }
        s += &format!(
            "halted={} shadow={} dev: {}\n",
            self.halted,
            self.interrupt_shadow,
            self.dev.state(self.now_ns())
        );
        let mut found = 0;
        let mut search_from = 0;
        while found < MAX_BANNER_MATCHES && search_from + BANNER.len() < self.ram.len() {
            let Some(relative) = self.ram[search_from..]
                .windows(BANNER.len())
                .position(|window| window == BANNER)
            else {
                break;
            };
            let start = search_from + relative;
            let end = (start + LOG_SCAN_BYTES).min(self.ram.len());
            let mut text = String::new();
            let mut printable_run = String::new();
            for &byte in &self.ram[start..end] {
                if (b' '..0x7f).contains(&byte) {
                    printable_run.push(byte as char);
                } else {
                    if printable_run.len() >= MIN_PRINTABLE_RUN {
                        text += &printable_run;
                        text.push('\n');
                    }
                    printable_run.clear();
                }
            }
            s += &format!("--- ram match at {start:#x}\n{text}\n");
            found += 1;
            search_from = start + BANNER.len();
        }
        if let Some(trace) = &self.trace {
            s += "last rips:";
            for rip in trace.iter().rev().take(TRACE_DUMP_ENTRIES) {
                s += &format!(" {rip:x}");
            }
            s += "\n";
        }
        s
    }
}

#[cfg(test)]
mod cpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_constants_match_documented_values() {
        assert_eq!(SEG_FLAGS_FLAT_DATA, 0x00cf_9200);
        assert_eq!(SEG_FLAGS_CODE64, 0x0020_9a00);
        assert_eq!(CR0_PG | CR0_NE | CR0_ET | CR0_MP | CR0_PE, 0x8000_0033);
        assert_eq!(EFER_LMA | EFER_LME, 0x500);
        assert_eq!(CR4_PAE, 0x20);
        assert_eq!(core::mem::size_of::<TlbEntry>(), 32);
    }
}
