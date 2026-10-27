#[cfg(freestanding)]
use crate::prelude::*;
mod bmi;
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
mod xsave;

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

/// `Cpu::rng_state` until the machine overrides it with `MachineConfig::random_seed`.
pub(crate) const DEFAULT_RANDOM_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

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
    /// IA32_SPEC_CTRL (stored, no speculation to control; see `SPEC_CTRL_WRITABLE` in sys.rs).
    pub spec_ctrl: u64,
    /// XCR0, the enabled extended-state components (xgetbv/xsetbv). Bit 0 (x87) is always set.
    pub xcr0: u64,
    /// State of the generator behind rdrand/rdseed (splitmix64). Seeded by the machine.
    pub rng_state: u64,
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
            spec_ctrl: 0,
            xcr0: xsave::XCR0_X87,
            rng_state: DEFAULT_RANDOM_SEED,
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
mod cpu_tests {
    use std::sync::atomic::AtomicUsize;

    use super::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE};
    use super::flags::*;
    use super::test_support::*;
    use super::*;
    use crate::testkit::{CODE_BASE, HLT, Harness, RBX, SCRATCH, STACK_TOP};

    const OLD_RIP: u64 = 0x0000_7000_1234_5678;
    /// A handler that asks the keyboard controller for a reset and halts.
    // mov al, 0xfe; out 0x64, al; hlt
    const RESET_HANDLER: [u8; 5] = [0xb0, 0xfe, 0xe6, 0x64, 0xf4];
    const HANDLER_ADDRESS: u64 = 0x30_0000;
    const KERNEL_STACK: u64 = 0x70_0000;
    // l: jmp l
    const SPIN: [u8; 2] = [0xeb, 0xfe];
    // mov rax, 5; add rax, 7; mov al, 0xfe; out 0x64, al; l: jmp l
    const COMPUTE_THEN_RESET: [u8; 17] = [
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x83, 0xc0, 0x07, 0xb0, 0xfe, 0xe6, 0x64,
        0xeb, 0xfe,
    ];
    const PIC_MASTER_COMMAND: u16 = 0x20;
    const PIC_MASTER_DATA: u16 = 0x21;
    const PIC_SLAVE_COMMAND: u16 = 0xa0;
    const PIC_SLAVE_DATA: u16 = 0xa1;
    const PIT_CHANNEL0_PORT: u16 = 0x40;
    const PIT_COMMAND_PORT: u16 = 0x43;
    const KEYBOARD_CONTROLLER_COMMAND_PORT: u16 = 0x64;
    const KEYBOARD_CONTROLLER_RESET: u32 = 0xfe;
    /// Channel 0, low then high byte access, mode 2 (rate generator).
    const PIT_CHANNEL0_RATE_GENERATOR: u32 = 0x34;
    const IRQ_VECTOR_BASE: u8 = 0x20;
    const ICW1_INIT_WITH_ICW4: u32 = 0x11;
    const ICW4_8086: u32 = 1;
    const ICW3_SLAVE_ON_IRQ2: u32 = 0x04;
    const ICW3_SLAVE_ID_2: u32 = 0x02;

    fn harness() -> Harness {
        Harness::new(&[HLT])
    }

    fn with_idt() -> Harness {
        let mut h = harness();
        install_idt(&mut h);
        h.cpu.eflags = RESERVED_ONE | IF;
        h.cpu.rip = OLD_RIP;
        h
    }

    /// Standard PC wiring: master at `IRQ_VECTOR_BASE`, slave right after, all lines unmasked.
    fn init_pics(cpu: &mut Cpu) {
        let devices = &mut cpu.dev;
        devices.io_out(PIC_MASTER_COMMAND, 1, ICW1_INIT_WITH_ICW4, 0);
        devices.io_out(PIC_MASTER_DATA, 1, IRQ_VECTOR_BASE as u32, 0);
        devices.io_out(PIC_MASTER_DATA, 1, ICW3_SLAVE_ON_IRQ2, 0);
        devices.io_out(PIC_MASTER_DATA, 1, ICW4_8086, 0);
        devices.io_out(PIC_SLAVE_COMMAND, 1, ICW1_INIT_WITH_ICW4, 0);
        devices.io_out(PIC_SLAVE_DATA, 1, IRQ_VECTOR_BASE as u32 + 8, 0);
        devices.io_out(PIC_SLAVE_DATA, 1, ICW3_SLAVE_ID_2, 0);
        devices.io_out(PIC_SLAVE_DATA, 1, ICW4_8086, 0);
    }

    fn program_pit_channel0(cpu: &mut Cpu, count: u16) {
        cpu.dev
            .io_out(PIT_COMMAND_PORT, 1, PIT_CHANNEL0_RATE_GENERATOR, 0);
        cpu.dev
            .io_out(PIT_CHANNEL0_PORT, 1, (count & 0xff) as u32, 0);
        cpu.dev.io_out(PIT_CHANNEL0_PORT, 1, (count >> 8) as u32, 0);
    }

    fn install_reset_handler(h: &mut Harness, vector: u8) {
        h.write(HANDLER_ADDRESS, &RESET_HANDLER);
        set_gate(h, vector, Gate::interrupt(HANDLER_ADDRESS));
    }

    mod reset_state {
        use super::*;

        #[test]
        fn control_registers_describe_long_mode_with_paging_on() {
            let h = harness();
            assert_eq!(h.cpu.cr0, 0x8000_0033);
            assert_eq!(h.cpu.cr4, 0x20);
            assert_eq!(h.cpu.efer, 0x500);
            assert_eq!(h.cpu.cr2, 0);
            assert_eq!(h.cpu.cr8, 0);
            assert_eq!(h.cpu.cr3 & PTE_ADDRESS_BITS, 0x9000);
        }

        #[test]
        fn long_mode_is_enabled_without_no_execute_or_write_protect() {
            let h = harness();
            assert_eq!(h.cpu.efer & EFER_NXE, 0);
            assert_eq!(h.cpu.cr0 & CR0_WP, 0);
            assert_ne!(h.cpu.efer & EFER_LMA, 0);
            assert_ne!(h.cpu.efer & EFER_LME, 0);
            assert_ne!(h.cpu.cr4 & CR4_PAE, 0);
        }

        #[test]
        fn segments_are_flat_with_the_boot_selectors() {
            let h = harness();
            let cs = h.cpu.seg[SEG_CS];
            assert_eq!(cs.sel, KERNEL_CS);
            assert_eq!(cs.flags, 0x0020_9a00);
            assert!(cs.is_code() && cs.long() && cs.present());
            assert_eq!(cs.dpl(), 0);
            for index in [SEG_ES, SEG_SS, SEG_DS] {
                let seg = h.cpu.seg[index];
                assert_eq!(seg.sel, KERNEL_DS, "segment {index}");
                assert_eq!(seg.flags, 0x00cf_9200, "segment {index}");
                assert_eq!(seg.base, 0);
                assert_eq!(seg.limit, 0xffff_ffff);
                assert!(!seg.is_code());
            }
            for index in [SEG_FS, SEG_GS] {
                let seg = h.cpu.seg[index];
                assert_eq!((seg.sel, seg.base, seg.limit, seg.flags), (0, 0, 0, 0));
            }
        }

        #[test]
        fn descriptor_tables_point_at_the_boot_gdt_and_an_empty_idt() {
            let h = harness();
            assert_eq!(h.cpu.gdtr.base, GDT_BASE);
            assert_eq!(h.cpu.gdtr.limit, 31);
            assert_eq!((h.cpu.idtr.base, h.cpu.idtr.limit), (0, 0));
            assert_eq!((h.cpu.ldtr.sel, h.cpu.tr.sel), (0, 0));
        }

        #[test]
        fn registers_start_at_the_boot_protocol_values() {
            let h = harness();
            assert_eq!(h.cpu.rip, CODE_BASE);
            assert_eq!(h.cpu.gpr[GPR_RSI], 0x7000, "rsi points at boot_params");
            for (index, value) in h.cpu.gpr.iter().enumerate() {
                if index != GPR_RSI && index != GPR_RSP {
                    assert_eq!(*value, 0, "register {index}");
                }
            }
            assert_eq!(h.cpu.gpr[GPR_RSP], STACK_TOP);
            assert_eq!(h.cpu.eflags_full(), RESERVED_ONE);
            assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn msrs_debug_and_fpu_state_have_their_documented_power_on_values() {
            let h = harness();
            assert_eq!(h.cpu.pat, 0x0007_0406_0007_0406);
            assert_eq!(h.cpu.dr[6], 0xffff_0ff0);
            assert_eq!(h.cpu.dr[7], 0x400);
            assert_eq!(&h.cpu.dr[..6], &[0; 6]);
            assert_eq!(h.cpu.mxcsr, 0x1f80);
            assert_eq!(h.cpu.fcw, 0x37f);
            assert_eq!((h.cpu.fsw, h.cpu.ftw), (0, 0));
            assert_eq!(h.cpu.xmm, [0; 16]);
            assert_eq!(h.cpu.st, [[0; 10]; 8]);
            assert_eq!(
                (h.cpu.star, h.cpu.lstar, h.cpu.cstar, h.cpu.sfmask),
                (0, 0, 0, 0)
            );
            assert_eq!((h.cpu.kgsbase, h.cpu.tsc_aux), (0, 0));
            assert_eq!(h.cpu.sysenter, [0; 3]);
        }

        #[test]
        fn stored_only_msrs_are_seeded() {
            let h = harness();
            let expected: [(u32, u64); 5] = [
                (0x8b, 0),
                (0xc001_0015, 1 << 24),
                (0xc001_001f, 0),
                (0xc001_1020, 0),
                (0xc001_1029, 0),
            ];
            assert_eq!(h.cpu.msr_misc.len(), expected.len());
            for (msr, value) in expected {
                assert_eq!(h.cpu.msr_misc.get(&msr), Some(&value), "msr {msr:#x}");
            }
        }

        #[test]
        fn caches_and_counters_start_empty() {
            let h = harness();
            assert_eq!(h.cpu.tlb.len(), 2 * TLB_ENTRIES_PER_MODE);
            assert!(h.cpu.tlb.iter().all(|entry| entry.tag_r == TLB_TAG_INVALID
                && entry.tag_w == TLB_TAG_INVALID
                && entry.tag_x == TLB_TAG_INVALID));
            assert_eq!(h.cpu.decode_cache.len(), DECODE_CACHE_ENTRIES);
            assert!(h.cpu.decode_cache.iter().all(|entry| entry.generation == 0));
            assert_eq!(h.cpu.decode_cache_generation, 1);
            assert_eq!(h.cpu.page_has_code.len(), h.cpu.ram.len() >> PAGE_SHIFT);
            assert!(h.cpu.page_has_code.iter().all(|&flag| flag == 0));
            assert!(h.cpu.pages_with_code.is_empty());
            assert_eq!(
                (h.cpu.icount, h.cpu.idle_skipped_ns, h.cpu.next_event),
                (0, 0, 0)
            );
            assert_eq!(h.cpu.flushes, 0);
        }

        #[test]
        fn execution_flags_and_hooks_start_clear() {
            let h = harness();
            assert!(!h.cpu.halted);
            assert!(!h.cpu.interrupt_shadow);
            assert!(!h.cpu.block_exit_requested);
            assert!(!h.cpu.realtime);
            assert_eq!(h.cpu.prepaid_icount, 0);
            assert_eq!(h.cpu.last_block_insns, 0);
            assert!(h.cpu.pending.is_none());
            assert!(h.cpu.jit.is_none());
            assert!(h.cpu.trace.is_none());
            assert!(h.cpu.poll.is_none());
            assert!(h.cpu.log.is_none());
            assert!(h.cpu.logged.is_empty());
        }

        #[test]
        fn ram_base_and_the_supervisor_tlb_half_point_into_owned_buffers() {
            let h = harness();
            assert_eq!(h.cpu.ram_base, h.cpu.ram.as_ptr() as u64);
            assert_eq!(h.cpu.tlb_active_base, h.cpu.tlb.as_ptr() as u64);
            assert_eq!(h.cpu.ram.len(), crate::testkit::HARNESS_RAM_BYTES);
        }

        #[test]
        fn a_fresh_cpu_walks_the_boot_identity_map() {
            let mut h = harness();
            assert_eq!(
                h.cpu
                    .translate(0x12_3456, mem::AccessKind::Read, false)
                    .unwrap(),
                0x12_3456
            );
            assert_eq!(
                h.cpu
                    .translate(0x1fff_ffff, mem::AccessKind::Write, false)
                    .unwrap(),
                0x1fff_ffff
            );
        }
    }

    mod segments_and_sizes {
        use super::*;

        #[test]
        fn segment_accessors_read_the_descriptor_flag_bits() {
            let code = Seg {
                sel: 0x10,
                base: 0,
                limit: 0,
                flags: SEG_LONG
                    | SEG_PRESENT
                    | SEG_NON_SYSTEM
                    | SEG_TYPE_EXECUTABLE
                    | (2 << SEG_DPL_SHIFT),
            };
            assert!(code.is_code() && code.long() && code.present());
            assert_eq!(code.dpl(), 2);
            let data = Seg {
                flags: SEG_NON_SYSTEM | SEG_PRESENT | SEG_TYPE_READ_WRITE | (3 << SEG_DPL_SHIFT),
                ..code
            };
            assert!(!data.is_code() && !data.long());
            assert_eq!(data.dpl(), 3);
        }

        #[test]
        fn system_descriptor_with_the_executable_bit_pattern_is_not_code() {
            let system = Seg {
                flags: SEG_TYPE_EXECUTABLE | SEG_PRESENT,
                ..Seg::default()
            };
            assert!(!system.is_code());
        }

        #[test]
        fn default_segment_is_null_and_absent() {
            let seg = Seg::default();
            assert!(!seg.present() && !seg.is_code() && !seg.long());
            assert_eq!(seg.dpl(), 0);
        }

        #[test]
        fn tlb_entry_layout_is_what_the_jit_hardcodes() {
            assert_eq!(core::mem::size_of::<TlbEntry>(), 32);
            assert_eq!(core::mem::offset_of!(TlbEntry, tag_r), 0);
            assert_eq!(core::mem::offset_of!(TlbEntry, tag_w), 8);
            assert_eq!(core::mem::offset_of!(TlbEntry, addend), 24);
        }

        #[test]
        fn page_constants_agree() {
            assert_eq!(PAGE_SIZE, 4096);
            assert_eq!(PAGE_OFFSET_MASK, 0xfff);
            assert_eq!(TLB_ENTRIES_PER_MODE, 1024);
            assert_eq!(USER_CPL, 3);
            assert_eq!(GPR_RSP, 4);
            assert_eq!(
                [SEG_ES, SEG_CS, SEG_SS, SEG_DS, SEG_FS, SEG_GS],
                [0, 1, 2, 3, 4, 5]
            );
        }
    }

    mod time_and_logging {
        use super::*;

        #[test]
        fn now_ns_counts_retired_instructions_plus_idle_time() {
            let mut h = harness();
            assert_eq!(h.cpu.now_ns(), 0);
            h.cpu.icount = 1000;
            assert_eq!(h.cpu.now_ns(), 1000 * NS_PER_INSN);
            h.cpu.idle_skipped_ns = 5_000;
            assert_eq!(h.cpu.now_ns(), 1000 * NS_PER_INSN + 5_000);
        }

        #[test]
        fn tsc_is_the_guest_clock_in_nanoseconds() {
            let mut h = harness();
            h.cpu.icount = 77;
            h.cpu.idle_skipped_ns = 23;
            assert_eq!(h.cpu.tsc(), h.cpu.now_ns());
            assert_eq!(h.cpu.tsc(), 77 * NS_PER_INSN + 23);
        }

        #[test]
        fn tsc_is_monotonic_across_instructions() {
            let mut h = Harness::new(&[0x90, 0x90, 0x90, HLT]);
            let mut previous = h.cpu.tsc();
            for _ in 0..3 {
                h.run(1);
                assert!(h.cpu.tsc() > previous);
                previous = h.cpu.tsc();
            }
        }

        #[test]
        fn log_once_runs_the_message_closure_once_per_key() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            let built = AtomicUsize::new(0);
            for _ in 0..3 {
                h.cpu.log_once(7, || {
                    built.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    "seven".to_string()
                });
            }
            assert_eq!(built.load(core::sync::atomic::Ordering::Relaxed), 1);
            assert_eq!(*log.lock().unwrap(), ["seven"]);
        }

        #[test]
        fn log_once_keeps_distinct_keys_distinct() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.log_once(1, || "one".into());
            h.cpu.log_once(2, || "two".into());
            h.cpu.log_once(1, || "one again".into());
            assert_eq!(*log.lock().unwrap(), ["one", "two"]);
        }

        #[test]
        fn log_once_without_a_logger_is_silent_and_still_remembers_the_key() {
            let mut h = harness();
            h.cpu.log_once(9, || {
                panic!("the message must not be built without a logger")
            });
            // Installing a logger afterwards does not resurrect the key.
            let log = capture_log(&mut h.cpu);
            h.cpu.log_once(9, || "late".into());
            assert!(log.lock().unwrap().is_empty());
        }

        #[test]
        fn log_once_accepts_the_largest_key() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.log_once(u64::MAX, || "max".into());
            h.cpu.log_once(0, || "zero".into());
            assert_eq!(log.lock().unwrap().len(), 2);
        }
    }

    mod event_servicing {
        use super::*;

        #[test]
        fn stop_flag_is_honoured_by_service_events() {
            let mut h = harness();
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(h.cpu.service_events(), Some(ExitReason::Stopped)));
        }

        #[test]
        fn service_events_schedules_the_next_poll_one_interval_ahead() {
            let mut h = harness();
            h.cpu.icount = 12_345;
            assert!(h.cpu.service_events().is_none());
            assert_eq!(h.cpu.next_event, 12_345 + EVENT_INTERVAL);
        }

        #[test]
        fn a_requested_reset_surfaces_as_a_reset_exit() {
            let mut h = harness();
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(h.cpu.service_events(), Some(ExitReason::Reset)));
        }

        #[test]
        fn other_keyboard_controller_commands_do_not_reset() {
            let mut h = harness();
            h.cpu
                .dev
                .io_out(KEYBOARD_CONTROLLER_COMMAND_PORT, 1, 0xaa, 0);
            assert!(h.cpu.service_events().is_none());
        }

        #[test]
        fn tick_devices_calls_the_host_poll_hook_as_not_idle() {
            let mut h = harness();
            let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = calls.clone();
            h.cpu.poll = Some(std::sync::Arc::new(move |idle| {
                sink.lock().unwrap().push(idle)
            }));
            h.cpu.tick_devices(0);
            h.cpu.tick_devices(10);
            assert_eq!(*calls.lock().unwrap(), [false, false]);
        }

        #[test]
        fn tick_devices_raises_the_pit_interrupt_when_its_deadline_passes() {
            let mut h = harness();
            init_pics(&mut h.cpu);
            program_pit_channel0(&mut h.cpu, 1000);
            let deadline = h.cpu.dev.next_deadline(0).expect("armed timer");
            h.cpu.tick_devices(deadline - 1);
            assert!(!h.cpu.dev.pic.has_pending());
            h.cpu.tick_devices(deadline);
            assert!(h.cpu.dev.pic.has_pending());
        }

        #[test]
        fn pending_interrupt_requires_if_and_no_shadow() {
            let mut h = harness();
            init_pics(&mut h.cpu);
            h.cpu.dev.pic.raise(1);
            h.cpu.eflags = RESERVED_ONE;
            assert_eq!(h.cpu.pending_interrupt(), None);
            h.cpu.eflags = RESERVED_ONE | IF;
            h.cpu.interrupt_shadow = true;
            assert_eq!(h.cpu.pending_interrupt(), None);
            assert!(
                h.cpu.dev.pic.has_pending(),
                "refusing must not acknowledge the interrupt"
            );
            h.cpu.interrupt_shadow = false;
            assert_eq!(h.cpu.pending_interrupt(), Some(IRQ_VECTOR_BASE + 1));
            assert_eq!(h.cpu.pending_interrupt(), None, "now in service");
        }
    }

    mod before_step {
        use super::*;

        #[test]
        fn executes_when_nothing_is_pending() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
        }

        #[test]
        fn events_are_polled_only_once_the_instruction_count_reaches_next_event() {
            let mut h = with_idt();
            h.cpu.stop.store(true, Ordering::Relaxed);
            h.cpu.next_event = 100;
            h.cpu.icount = 99;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            h.cpu.icount = 100;
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Stopped)
            ));
        }

        #[test]
        fn reset_request_exits_at_the_next_poll() {
            let mut h = with_idt();
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Reset)
            ));
        }

        #[test]
        fn unmasked_interrupt_with_if_set_is_delivered_through_the_idt() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
            assert_eq!(
                h.cpu.eflags_full() & IF,
                0,
                "interrupt gates mask further interrupts"
            );
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(frame.rip, OLD_RIP);
            assert_eq!(frame.rflags & IF, IF);
            assert!(h.cpu.dev.pic.state().contains("isr 02"));
        }

        #[test]
        fn slave_interrupt_uses_the_slave_vector_base() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.dev.pic.raise(9);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 9));
        }

        #[test]
        fn interrupt_is_held_while_if_is_clear() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert_eq!(h.cpu.rip, OLD_RIP);
            assert!(h.cpu.dev.pic.has_pending());
        }

        #[test]
        fn interrupt_is_held_for_one_instruction_after_a_shadowing_instruction() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.interrupt_shadow = true;
            h.cpu.dev.pic.raise(1);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP);
            h.cpu.interrupt_shadow = false;
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
        }

        #[test]
        fn masked_interrupt_is_not_delivered() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.dev.io_out(PIC_MASTER_DATA, 1, 1 << 1, 0);
            h.cpu.dev.pic.raise(1);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn delivering_an_interrupt_wakes_a_halted_cpu() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.halted = true;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert!(!h.cpu.halted);
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
        }

        #[test]
        fn halted_cpu_with_interrupts_disabled_is_shut_down_even_with_an_irq_pending() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.halted = true;
            h.cpu.dev.pic.raise(1);
            // Only NMI, SMI, INIT and RESET wake `cli; hlt`, and none of them exists here.
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Shutdown)
            ));
            assert!(h.cpu.halted);
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn halted_cpu_with_interrupts_enabled_and_an_irq_pending_idles_without_sleeping() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            // The interrupt shadow keeps the pending IRQ from being delivered this step, but IF is
            // set, so the halt is not terminal.
            h.cpu.interrupt_shadow = true;
            h.cpu.halted = true;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert!(h.cpu.halted);
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn a_reset_request_beats_the_shutdown_of_a_halt_with_interrupts_disabled() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.halted = true;
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Reset)
            ));
        }

        #[test]
        fn a_stop_request_beats_the_shutdown_of_a_halt_with_interrupts_disabled() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.halted = true;
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Stopped)
            ));
        }

        #[test]
        fn halted_cpu_exits_on_stop_without_waiting() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.halted = true;
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Stopped)
            ));
        }

        #[test]
        fn halted_cpu_notices_a_reset_request_while_idle() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.halted = true;
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Reset)
            ));
        }

        #[test]
        fn halted_cpu_skips_virtual_time_to_the_next_timer_deadline() {
            let mut h = with_idt();
            h.cpu.next_event = u64::MAX;
            init_pics(&mut h.cpu);
            program_pit_channel0(&mut h.cpu, 1000);
            let deadline = h.cpu.dev.next_deadline(0).unwrap();
            h.cpu.halted = true;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert_eq!(h.cpu.idle_skipped_ns, deadline);
            assert_eq!(h.cpu.now_ns(), deadline);
            assert_eq!(h.cpu.icount, 0, "idling retires no instructions");
            assert!(
                h.cpu.dev.pic.has_pending(),
                "the timer fired during the skip"
            );
            // The next step takes the timer interrupt and wakes up.
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert!(!h.cpu.halted);
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE));
        }

        #[test]
        fn realtime_idle_sleeps_for_a_short_deadline_and_advances_by_it() {
            let mut h = with_idt();
            h.cpu.next_event = u64::MAX;
            h.cpu.realtime = true;
            program_pit_channel0(&mut h.cpu, 100);
            let deadline = h.cpu.dev.next_deadline(0).unwrap();
            h.cpu.halted = true;
            let started = std::time::Instant::now();
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert!(started.elapsed() >= std::time::Duration::from_nanos(deadline));
            assert_eq!(h.cpu.idle_skipped_ns, deadline);
        }

        #[test]
        fn realtime_idle_never_sleeps_longer_than_two_milliseconds_per_call() {
            const MAX_SLEEP_NS: u64 = 2_000_000;
            let mut h = with_idt();
            h.cpu.next_event = u64::MAX;
            h.cpu.realtime = true;
            // 65536 ticks of the 1.19 MHz clock is about 55 ms.
            program_pit_channel0(&mut h.cpu, 0);
            h.cpu.halted = true;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert_eq!(h.cpu.idle_skipped_ns, MAX_SLEEP_NS);
        }

        #[test]
        fn delivery_failure_while_taking_an_interrupt_ends_the_run() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.idtr.limit = 0;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::TripleFault)
            ));
        }
    }

    mod after_step {
        use super::*;

        #[test]
        fn success_continues_the_run() {
            let mut h = with_idt();
            assert!(h.cpu.after_step(Ok(())).is_none());
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn a_fault_is_delivered_through_the_idt() {
            let mut h = with_idt();
            let result = Err(Exception::fault(VECTOR_GENERAL_PROTECTION, Some(0x18)));
            assert!(h.cpu.after_step(result).is_none());
            assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], true);
            assert_eq!((frame.error_code, frame.rip), (Some(0x18), OLD_RIP));
        }

        #[test]
        fn a_fault_without_an_error_code_pushes_none() {
            let mut h = with_idt();
            assert!(h.cpu.after_step(Err(Exception::invalid_opcode())).is_none());
            assert_eq!(h.cpu.rip, handler_for(VECTOR_INVALID_OPCODE));
            assert_eq!(h.cpu.gpr[GPR_RSP], (STACK_TOP & !0xf) - 40);
        }

        #[test]
        fn software_interrupt_flag_reaches_the_privilege_check() {
            let mut h = with_idt();
            install_tss(&mut h, [KERNEL_STACK, 0, 0], [0; 7]);
            enter_user_mode(&mut h);
            let software = Exception::Fault {
                vector: 0x80,
                error_code: None,
                software: true,
            };
            assert!(h.cpu.after_step(Err(software)).is_none());
            // DPL0 gate from ring 3 raised #GP instead of entering vector 0x80.
            assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
        }

        #[test]
        fn exit_reasons_pass_straight_through() {
            let mut h = with_idt();
            let result = Err(Exception::Exit(ExitReason::Shutdown));
            assert!(matches!(
                h.cpu.after_step(result),
                Some(ExitReason::Shutdown)
            ));
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn a_fault_with_no_working_idt_is_a_triple_fault() {
            let mut h = harness();
            assert!(matches!(
                h.cpu.after_step(Err(Exception::invalid_opcode())),
                Some(ExitReason::TripleFault)
            ));
        }
    }

    mod run_loop {
        use super::*;

        #[test]
        fn stop_flag_set_before_the_run_exits_immediately() {
            let mut h = Harness::new(&SPIN);
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(h.cpu.run(), ExitReason::Stopped));
            assert_eq!(h.cpu.icount, 0);
        }

        #[test]
        fn run_executes_up_to_the_next_event_poll_then_sees_the_stop_flag() {
            let mut h = Harness::new(&SPIN);
            h.cpu.next_event = 100;
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(h.cpu.run(), ExitReason::Stopped));
            assert_eq!(h.cpu.icount, 100);
            assert_eq!(h.cpu.rip, CODE_BASE);
        }

        #[test]
        fn run_counts_one_instruction_per_step() {
            let mut h = Harness::new(&[0x90; 40]);
            h.cpu.next_event = 25;
            h.cpu.stop.store(true, Ordering::Relaxed);
            h.cpu.run();
            assert_eq!(h.cpu.icount, 25);
            assert_eq!(h.cpu.rip, CODE_BASE + 25);
        }

        #[test]
        fn run_returns_reset_as_soon_as_the_guest_writes_the_reset_port() {
            let mut h = Harness::new(&COMPUTE_THEN_RESET);
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert_eq!(h.cpu.gpr[crate::testkit::RAX], 0xfe);
            // mov, add, mov, out: the port write itself ends the run, no poll interval needed.
            assert_eq!(h.cpu.icount, 4);
        }

        #[test]
        fn run_clears_the_interrupt_shadow_before_each_instruction() {
            let mut h = Harness::new(&SPIN);
            h.cpu.interrupt_shadow = true;
            h.cpu.next_event = 3;
            h.cpu.stop.store(true, Ordering::Relaxed);
            h.cpu.run();
            assert!(!h.cpu.interrupt_shadow);
        }

        #[test]
        fn a_faulting_instruction_is_delivered_and_the_handler_runs() {
            // ud2
            let mut h = Harness::new(&[0x0f, 0x0b]);
            install_idt(&mut h);
            install_reset_handler(&mut h, VECTOR_INVALID_OPCODE);
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(
                frame.rip, CODE_BASE,
                "the saved rip is the faulting instruction"
            );
            // The run ends at the port write, two instructions into the handler.
            assert_eq!(h.cpu.rip, HANDLER_ADDRESS + 4);
        }

        #[test]
        fn a_fault_without_an_idt_ends_in_a_triple_fault() {
            // ud2
            let mut h = Harness::new(&[0x0f, 0x0b]);
            assert!(matches!(h.cpu.run(), ExitReason::TripleFault));
        }

        #[test]
        fn a_timer_interrupt_wakes_a_halted_guest_and_runs_its_handler() {
            let mut h = Harness::new(&[HLT]);
            install_idt(&mut h);
            init_pics(&mut h.cpu);
            program_pit_channel0(&mut h.cpu, 1000);
            install_reset_handler(&mut h, IRQ_VECTOR_BASE);
            h.cpu.eflags = RESERVED_ONE | IF;
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert!(h.cpu.idle_skipped_ns > 0, "the guest slept until the timer");
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(frame.rip, CODE_BASE + 1, "interrupted after the hlt");
        }

        #[test]
        fn the_jit_run_loop_reaches_the_same_end_state() {
            let mut h = Harness::new(&COMPUTE_THEN_RESET);
            if !h.enable_jit() {
                return;
            }
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert_eq!(h.cpu.gpr[crate::testkit::RAX], 0xfe);
            assert_eq!(h.cpu.icount, 4);
        }

        #[test]
        fn the_jit_run_loop_delivers_faults() {
            // ud2
            let mut h = Harness::new(&[0x0f, 0x0b]);
            if !h.enable_jit() {
                return;
            }
            install_idt(&mut h);
            install_reset_handler(&mut h, VECTOR_INVALID_OPCODE);
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert_eq!(read_frame(&h, h.cpu.gpr[GPR_RSP], false).rip, CODE_BASE);
        }
    }

    mod single_step {
        use super::*;

        #[test]
        fn step_executes_one_instruction_and_advances_rip() {
            let mut h = Harness::new(&[0x90, 0x90, HLT]);
            h.cpu.step().unwrap();
            assert_eq!(h.cpu.rip, CODE_BASE + 1);
            h.cpu.step().unwrap();
            assert_eq!(h.cpu.rip, CODE_BASE + 2);
        }

        #[test]
        fn step_reports_decode_faults_without_moving_rip() {
            let mut h = Harness::new(&[0x06]);
            assert_eq!(fault_of(h.cpu.step()), (VECTOR_INVALID_OPCODE, None));
            assert_eq!(h.cpu.rip, CODE_BASE);
        }

        #[test]
        fn trace_records_the_address_of_each_executed_instruction() {
            let mut h = Harness::new(&[0x90; 8]);
            h.cpu.trace = Some(Default::default());
            for _ in 0..3 {
                h.cpu.step().unwrap();
            }
            let trace: Vec<u64> = h.cpu.trace.as_ref().unwrap().iter().copied().collect();
            assert_eq!(trace, [CODE_BASE, CODE_BASE + 1, CODE_BASE + 2]);
        }

        #[test]
        fn trace_keeps_only_the_newest_sixty_four_entries() {
            let mut h = Harness::new(&[0x90; 100]);
            h.cpu.trace = Some(Default::default());
            for _ in 0..70 {
                h.cpu.step().unwrap();
            }
            let trace = h.cpu.trace.as_ref().unwrap();
            assert_eq!(trace.len(), TRACE_CAPACITY);
            assert_eq!(*trace.front().unwrap(), CODE_BASE + 6);
            assert_eq!(*trace.back().unwrap(), CODE_BASE + 69);
        }

        #[test]
        fn a_decode_fault_is_not_traced() {
            let mut h = Harness::new(&[0x06]);
            h.cpu.trace = Some(Default::default());
            let _ = h.cpu.step();
            assert!(h.cpu.trace.as_ref().unwrap().is_empty());
        }
    }

    mod dumps {
        use super::*;

        #[test]
        fn dump_regs_prints_rip_cpl_eflags_and_all_sixteen_registers() {
            let mut h = harness();
            h.cpu.rip = 0x1234_5678_9abc_def0;
            h.cpu.eflags = RESERVED_ONE | IF | ZF;
            for (index, reg) in h.cpu.gpr.iter_mut().enumerate() {
                *reg = 0x1111_1111_1111_1111 * (index as u64 % 16);
            }
            let dump = h.cpu.dump_regs();
            let lines: Vec<&str> = dump.lines().collect();
            assert_eq!(lines.len(), 5);
            assert_eq!(lines[0], "rip=123456789abcdef0 cpl=0 eflags=242");
            assert_eq!(
                lines[1],
                "r0 =0000000000000000 r1 =1111111111111111 r2 =2222222222222222 r3 =3333333333333333 "
            );
            assert!(lines[4].starts_with("r12=cccccccccccccccc r13=dddddddddddddddd"));
            assert!(lines[4].contains("r15=ffffffffffffffff"));
        }

        #[test]
        fn dump_regs_resolves_deferred_flags_and_shows_cpl() {
            let mut h = harness();
            h.cpu.set_cpl(3);
            h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
            let dump = h.cpu.dump_regs();
            assert!(dump.starts_with(&format!(
                "rip={:016x} cpl=3 eflags={:x}\n",
                CODE_BASE,
                RESERVED_ONE | CF | ZF | PF | AF
            )));
        }

        #[test]
        fn dump_starts_with_the_state_summary_and_ends_with_device_state() {
            let mut h = harness();
            h.cpu.flushes = 3;
            h.cpu.icount = 42;
            let dump = h.cpu.dump();
            assert!(dump.starts_with("flushes=3 icount=42 rip="), "{dump}");
            assert!(
                dump.contains("cr0=80000033 cr3=9000 cr4=20 efer=500\n"),
                "{dump}"
            );
            assert!(
                dump.contains("halted=false shadow=false dev: m(irr 00"),
                "{dump}"
            );
            assert!(!dump.contains("last rips"));
            assert!(!dump.contains("--- ram match"));
        }

        #[test]
        fn dump_shows_the_halted_and_shadow_flags() {
            let mut h = harness();
            h.cpu.halted = true;
            h.cpu.interrupt_shadow = true;
            assert!(h.cpu.dump().contains("halted=true shadow=true"));
        }

        #[test]
        fn dump_extracts_the_kernel_log_after_the_linux_version_banner() {
            let mut h = harness();
            let mut text = b"Linux version 6.1.0-test (builder) #1 SMP".to_vec();
            text.extend_from_slice(b"\0\0short\0");
            text.extend_from_slice(b"second record with enough characters\0\0");
            h.write(SCRATCH, &text);
            let dump = h.cpu.dump();
            assert!(
                dump.contains(&format!("--- ram match at {SCRATCH:#x}\n")),
                "{dump}"
            );
            assert!(dump.contains("Linux version 6.1.0-test (builder) #1 SMP\n"));
            assert!(dump.contains("second record with enough characters\n"));
            assert!(
                !dump.contains("short"),
                "runs under six characters are record headers"
            );
        }

        #[test]
        fn dump_reports_at_most_six_banner_matches() {
            let mut h = harness();
            for copy in 0..9u64 {
                h.write(SCRATCH + copy * 0x100, b"Linux version 1.2.3 banner\0");
            }
            let dump = h.cpu.dump();
            assert_eq!(dump.matches("--- ram match at").count(), 6);
        }

        #[test]
        fn dump_lists_the_newest_trace_entries_first_and_at_most_sixteen() {
            let mut h = Harness::new(&[0x90; 40]);
            h.cpu.trace = Some(Default::default());
            for _ in 0..30 {
                h.cpu.step().unwrap();
            }
            let dump = h.cpu.dump();
            let line = dump
                .lines()
                .find(|line| line.starts_with("last rips:"))
                .unwrap();
            let rips: Vec<u64> = line
                .trim_start_matches("last rips:")
                .split_whitespace()
                .map(|rip| u64::from_str_radix(rip, 16).unwrap())
                .collect();
            assert_eq!(rips.len(), TRACE_DUMP_ENTRIES);
            assert_eq!(rips[0], CODE_BASE + 29);
            assert_eq!(*rips.last().unwrap(), CODE_BASE + 14);
        }
    }

    mod verification_helpers {
        use super::*;

        fn pair(code: &[u8]) -> (Harness, Harness) {
            (Harness::new(code), Harness::new(code))
        }

        #[test]
        fn diff_of_identical_cpus_is_none() {
            let (mut a, mut b) = pair(&[HLT]);
            assert_eq!(a.cpu.diff(&mut b.cpu), None);
        }

        #[test]
        fn diff_names_each_differing_register() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.gpr[RBX] = 1;
            a.cpu.gpr[15] = 2;
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            assert!(report.contains("r3: 0x1 vs 0x0"), "{report}");
            assert!(report.contains("r15: 0x2 vs 0x0"), "{report}");
        }

        #[test]
        fn diff_reports_rip_cpl_and_segments() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.rip += 1;
            a.cpu.set_cpl(3);
            a.cpu.seg[SEG_FS].base = 0x1000;
            a.cpu.seg[SEG_SS].sel = 0x2b;
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            assert!(
                report.contains(&format!("rip: {:#x} vs {:#x}", CODE_BASE + 1, CODE_BASE)),
                "{report}"
            );
            assert!(report.contains("cpl"));
            assert!(report.contains("seg4"));
            assert!(report.contains("seg2"));
            assert!(!report.contains("seg1"));
        }

        #[test]
        fn diff_reports_control_registers_efer_and_icount() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.cr0 ^= 1 << 16;
            a.cpu.cr2 = 0x1000;
            a.cpu.cr3 ^= 0x1000;
            a.cpu.cr4 ^= 1 << 7;
            a.cpu.efer ^= EFER_NXE;
            a.cpu.icount = 9;
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            for name in ["cr0", "cr2", "cr3", "cr4", "efer", "icount: 0x9 vs 0x0"] {
                assert!(report.contains(name), "{name} missing from {report}");
            }
        }

        #[test]
        fn diff_reports_xmm_changes() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.xmm[7] = 1;
            assert_eq!(a.cpu.diff(&mut b.cpu).as_deref(), Some("xmm"));
        }

        #[test]
        fn diff_compares_resolved_flags_not_the_deferred_representation() {
            let (mut a, mut b) = pair(&[HLT]);
            // Same architectural flags, one deferred and one materialized.
            a.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
            b.cpu.set_eflags_arith(CF | ZF | PF | AF);
            assert_eq!(a.cpu.diff(&mut b.cpu), None);
            b.cpu.set_eflags_arith(CF);
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            assert!(report.contains("eflags"), "{report}");
        }

        #[test]
        fn run_verify_agrees_on_a_trivial_program() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(matches!(reason, ExitReason::Reset), "{reason:?}");
            assert_eq!(main.cpu.gpr[crate::testkit::RAX] & 0xff, 0xfe);
            assert_eq!(shadow.cpu.gpr[crate::testkit::RAX] & 0xff, 0xfe);
        }

        #[test]
        fn run_verify_reports_a_register_divergence() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            shadow.cpu.gpr[RBX] = 0xbad;
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            match reason {
                ExitReason::Internal(message) => {
                    assert!(
                        message.starts_with("verify: divergence after block at"),
                        "{message}"
                    );
                    assert!(message.contains("r3: 0x0 vs 0xbad"), "{message}");
                }
                other => panic!("expected a divergence report, got {other:?}"),
            }
        }

        #[test]
        fn run_verify_reports_diverging_code_and_logs_the_page_difference() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            let log = capture_log(&mut main.cpu);
            // The shadow adds 8 instead of 7: a one-byte difference in the same page.
            shadow.write(CODE_BASE + 10, &[0x08]);
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(matches!(reason, ExitReason::Internal(_)), "{reason:?}");
            let messages = log.lock().unwrap();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].starts_with("page diffs at "), "{}", messages[0]);
            assert!(messages[0].contains("main bytes"));
        }

        #[test]
        fn run_verify_flags_a_pre_step_divergence() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            // The main CPU idles (IF set, so the halt is not a shutdown) while the shadow is ready
            // to execute.
            main.cpu.halted = true;
            main.cpu.eflags |= IF;
            main.cpu.next_event = u64::MAX;
            shadow.cpu.next_event = u64::MAX;
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(
                matches!(&reason, ExitReason::Internal(message) if message.contains("pre-step divergence")),
                "{reason:?}"
            );
        }

        #[test]
        fn run_verify_accepts_matching_faults_and_ends_in_a_triple_fault() {
            // ud2 with no IDT: both engines raise #UD, which cannot be delivered.
            let (mut main, mut shadow) = pair(&[0x0f, 0x0b]);
            if !main.enable_jit() {
                return;
            }
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(matches!(reason, ExitReason::TripleFault), "{reason:?}");
        }
    }
}

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
