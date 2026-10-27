//! Runs the guest on the CPU with hardware virtualization (Intel VMX or AMD SVM), with the CPU's
//! nested paging in place of the emulator's soft TLB. The legacy devices still come from the
//! library: port I/O exits are forwarded to `Platform`, which also supplies interrupts.

#[cfg(target_os = "uefi")]
mod arch;
// On a hosted target (tests only) `arch` is just the constants, as the rest needs the CPU.
#[cfg(not(target_os = "uefi"))]
mod arch {
    pub use super::defs::*;
}
mod cpuid;
mod defs;
#[cfg(test)]
mod fake;
mod msr;
mod svm;
mod vmx;

#[cfg(target_os = "uefi")]
use alloc::boxed::Box;
#[cfg(target_os = "uefi")]
use alloc::format;
use alloc::string::String;
use core::time::Duration;

#[cfg(target_os = "uefi")]
use uefi::boot::{self, AllocateType, MemoryType};
#[cfg(target_os = "uefi")]
use uemulator::{ConsoleHandle, ExitReason, Platform};

#[cfg(target_os = "uefi")]
use crate::console;
#[cfg(target_os = "uefi")]
use arch::FpuState;
use arch::Gprs;

const PAGE_SIZE: usize = 4096;
const HUGE_PAGE_SIZE: usize = 2 << 20;
const GIB: usize = 1 << 30;
const MIB_SHIFT: u32 = 20;
const NANOS_PER_SEC: u128 = 1_000_000_000;
const MS_PER_SEC: u64 = 1000;
/// How often the firmware console is polled for input, in nanoseconds of host time.
const CONSOLE_POLL_NS: u64 = 1_000_000;
/// Polling interval while the guest is halted.
const HLT_POLL_INTERVAL: Duration = Duration::from_micros(50);
/// Reported unusual guest actions are capped so a guest that loops on one cannot flood the console.
const MAX_NOTES: u32 = 20;
/// The firmware tick that bounds how late the guest's timer interrupts arrive.
const FIRMWARE_TICK: Duration = Duration::from_millis(1);
/// Host TSC calibration runs for this long against the firmware's `Stall`.
const CALIBRATION_MS: u64 = 50;
const HZ_PER_KHZ: u64 = 1000;
/// Ports are 1, 2 or 4 bytes wide.
const DWORD_SIZE: u8 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Vmx,
    Svm,
}

/// Which extension this CPU offers, if any (and whether the firmware left it usable).
#[cfg(target_os = "uefi")]
pub fn detect() -> Option<Kind> {
    let features = arch::cpuid(arch::CPUID_LEAF_FEATURES, 0);
    if features[2] & arch::CPUID_1_ECX_VMX != 0 && vmx::usable() {
        return Some(Kind::Vmx);
    }
    let max_ext_leaf = arch::cpuid(arch::CPUID_LEAF_EXT_MAX, 0)[0];
    if max_ext_leaf >= arch::CPUID_LEAF_SVM
        && arch::cpuid(arch::CPUID_LEAF_EXT_FEATURES, 0)[2] & arch::CPUID_EXT1_ECX_SVM != 0
        && svm::usable()
    {
        return Some(Kind::Svm);
    }
    None
}

/// Why `Vcpu::run` returned. The instruction-specific ones leave rip at the instruction; the
/// handler calls `Vcpu::skip` when it has emulated it.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Exit {
    Io {
        port: u16,
        /// Bytes: 1, 2 or 4.
        size: u8,
        /// The guest executed OUT.
        write: bool,
    },
    Cpuid,
    Rdmsr,
    Wrmsr,
    Hlt,
    /// An instruction with no effect to emulate (INVD).
    Skip,
    /// A physical interrupt arrived; the host handles it before the guest resumes.
    HostInterrupt,
    /// The guest can take an interrupt now.
    IntWindow,
    TripleFault,
    /// Guest instruction that is not virtualized and raises `#UD`.
    Undefined,
    /// Guest access to a physical address with no RAM behind it.
    Fault {
        gpa: u64,
        write: bool,
    },
    /// Exit code or basic reason nothing handles.
    Other(u64),
}

/// One virtual CPU. Interrupt and exception injection is queued by the `inject_*` calls and takes
/// effect on the next `run`.
pub trait Vcpu {
    /// Guest registers as of the last exit; edits are loaded by the next `run`.
    fn gprs(&mut self) -> &mut Gprs;
    /// Enters the guest and returns at the next exit. `Err` is a fatal condition (rejected guest
    /// state, unsupported string I/O).
    fn run(&mut self) -> Result<Exit, String>;
    /// Moves the guest past the instruction behind the last exit.
    fn skip(&mut self);
    fn rip(&self) -> u64;
    /// The guest's IF flag, regardless of any interrupt shadow or queued event. A `hlt` with it
    /// clear can never be woken by a maskable interrupt.
    fn interrupts_enabled(&self) -> bool;
    /// The guest can take an external interrupt now: IF set, no STI or MOV SS shadow, and no other
    /// event already queued.
    fn irq_ok(&self) -> bool;
    fn inject_irq(&mut self, vector: u8);
    /// Requests (or cancels) an exit as soon as `irq_ok` would become true.
    fn window(&mut self, on: bool);
    fn inject_gp(&mut self);
    fn inject_ud(&mut self);
    /// The guest's EFER as the guest should see it (SVM hides its own SVME bit).
    fn efer(&self) -> u64;
    fn set_efer(&mut self, efer: u64);
}

/// Zeroed, page-aligned pages for hardware structures.
#[cfg(target_os = "uefi")]
pub fn alloc_pages(n: usize) -> Result<*mut u8, String> {
    let p = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, n)
        .map_err(|e| format!("out of memory: {e}"))?;
    unsafe { core::ptr::write_bytes(p.as_ptr(), 0, n * PAGE_SIZE) };
    Ok(p.as_ptr())
}

/// Guest RAM: contiguous host memory, mapped 1:1 into the guest physical address space.
pub struct GuestRam {
    ptr: *mut u8,
    /// Size the guest was configured with.
    len: usize,
    /// `len` rounded up to whole 2 MiB pages, the extent `build_tables` maps.
    mapped: usize,
}

/// Guest RAM of `mib` MiB: its size in bytes, that size rounded up to whole 2 MiB pages, and the
/// number of 4 KiB pages to allocate. One extra huge page lets the start be rounded up to a 2 MiB
/// boundary.
fn ram_layout(mib: usize) -> (usize, usize, usize) {
    let len = mib << MIB_SHIFT;
    let mapped = len.next_multiple_of(HUGE_PAGE_SIZE);
    (len, mapped, (mapped + HUGE_PAGE_SIZE) / PAGE_SIZE)
}

impl GuestRam {
    #[cfg(target_os = "uefi")]
    fn new(mib: usize) -> Result<Self, String> {
        let (len, mapped, pages) = ram_layout(mib);
        let raw = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages)
            .map_err(|e| format!("cannot allocate {mib} MiB of guest memory: {e}"))?;
        let ptr = (raw.as_ptr() as usize).next_multiple_of(HUGE_PAGE_SIZE) as *mut u8;
        unsafe { core::ptr::write_bytes(ptr, 0, mapped) };
        Ok(Self { ptr, len, mapped })
    }

    pub fn bytes(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn host_addr(&self) -> u64 {
        self.ptr as u64
    }

    /// Builds a 4-level table that maps guest physical `0..mapped` onto the RAM with 2 MiB pages.
    /// `ept` selects Intel's entry format over AMD's (which is the ordinary x86 format). Returns
    /// the physical address of the top level, for the EPT pointer or the nested CR3.
    #[cfg(target_os = "uefi")]
    pub fn build_tables(&self, ept: bool) -> Result<u64, String> {
        self.fill_tables(ept, || alloc_pages(1).map(|p| p as *mut u64))
    }

    /// `build_tables` with the source of zeroed, page-aligned table pages passed in.
    fn fill_tables(
        &self,
        ept: bool,
        mut alloc_table: impl FnMut() -> Result<*mut u64, String>,
    ) -> Result<u64, String> {
        // Bits 2:0 are read, write, execute in an EPT entry and present, writable, user in an
        // ordinary one, so one value serves as "allowed" in both.
        const ENTRY_ACCESS: u64 = 0b111;
        // EPT leaf: memory type in bits 5:3 (6 = write-back). Bit 7 makes a PD entry a 2 MiB
        // leaf in both formats.
        const EPT_MEMORY_TYPE_WRITE_BACK: u64 = 6 << 3;
        const LARGE_PAGE: u64 = 1 << 7;
        const ENTRIES_PER_TABLE: usize = PAGE_SIZE / size_of::<u64>();

        let page = &mut alloc_table;
        let table = ENTRY_ACCESS;
        let leaf = if ept {
            ENTRY_ACCESS | EPT_MEMORY_TYPE_WRITE_BACK | LARGE_PAGE
        } else {
            ENTRY_ACCESS | LARGE_PAGE
        };
        let pml4 = page()?;
        let pdpt = page()?;
        unsafe {
            *pml4 = pdpt as u64 | table;
            // One page directory (512 x 2 MiB) per GiB.
            for (pdpt_index, gib_start) in (0..self.mapped).step_by(GIB).enumerate() {
                let pd = page()?;
                *pdpt.add(pdpt_index) = pd as u64 | table;
                let entries = ((self.mapped - gib_start) / HUGE_PAGE_SIZE).min(ENTRIES_PER_TABLE);
                for i in 0..entries {
                    *pd.add(i) =
                        (self.host_addr() + (gib_start + i * HUGE_PAGE_SIZE) as u64) | leaf;
                }
            }
        }
        Ok(pml4 as u64)
    }
}

/// Host time in guest nanoseconds, from the TSC. Zero is the end of calibration.
struct Clock {
    tsc0: u64,
    /// TSC frequency measured against the firmware's `Stall`; never zero.
    hz: u64,
}

/// TSC frequency from the ticks counted during `CALIBRATION_MS`; never zero.
fn calibrated_hz(ticks: u64) -> u64 {
    (ticks * (MS_PER_SEC / CALIBRATION_MS)).max(1)
}

fn ticks_to_ns(ticks: u64, hz: u64) -> u64 {
    (ticks as u128 * NANOS_PER_SEC / hz as u128) as u64
}

#[cfg(target_os = "uefi")]
impl Clock {
    fn calibrate() -> Self {
        let t0 = arch::rdtsc();
        boot::stall(Duration::from_millis(CALIBRATION_MS));
        let t1 = arch::rdtsc();
        Self {
            tsc0: t1,
            hz: calibrated_hz(t1 - t0),
        }
    }

    fn now_ns(&self) -> u64 {
        ticks_to_ns(arch::rdtsc() - self.tsc0, self.hz)
    }
}

/// What `IN` leaves in rax: a 32-bit one zero-extends, 8- and 16-bit ones keep the rest.
fn rax_after_in(rax: u64, value: u32, size: u8) -> u64 {
    let value = value as u64;
    if size == DWORD_SIZE {
        return value;
    }
    let mask = u64::MAX >> (u64::BITS - 8 * size as u32);
    (rax & !mask) | (value & mask)
}

/// The part of rax that `OUT` sends.
fn out_value(rax: u64, size: u8) -> u32 {
    rax as u32 & (u32::MAX >> (u32::BITS - 8 * size as u32))
}

/// How RDMSR returns a value: low half in eax (zeroing the top of rax), high half in edx.
fn rdmsr_regs(value: u64) -> (u64, u64) {
    (value & u32::MAX as u64, value >> 32)
}

/// The value WRMSR writes, from edx:eax.
fn wrmsr_value(rax: u64, rdx: u64) -> u64 {
    (rdx << 32) | (rax & u32::MAX as u64)
}

/// What a halted guest does next.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum HltOutcome {
    Wake,
    Sleep,
    /// `cli; hlt`: nothing can wake it, the guest has powered off.
    PowerOff,
}

/// Whether a halted guest resumes. Like hardware, a maskable interrupt only wakes it when it could
/// be taken (IF set, no interrupt shadow, nothing else queued). A reset request always wakes it
/// (the event loop then returns it). With IF clear and no reset the guest is dead, as in the
/// emulator's `Cpu::idle`: Linux's poweroff and halt end in `cli; hlt` loops, and no NMI source
/// exists here. Only IF decides, not the shadow: `sti; hlt` has IF set once the exit ended the
/// shadow, and a device interrupt pending behind IF=0 never wakes the guest.
fn hlt_outcome(vcpu: &dyn Vcpu, irq_pending: bool, reset_requested: bool) -> HltOutcome {
    if reset_requested || (irq_pending && vcpu.irq_ok()) {
        HltOutcome::Wake
    } else if vcpu.interrupts_enabled() {
        HltOutcome::Sleep
    } else {
        HltOutcome::PowerOff
    }
}

#[cfg(target_os = "uefi")]
pub fn run(
    kind: Kind,
    kernel: &[u8],
    initrd: Option<&[u8]>,
    memory_mib: usize,
    cmdline: &str,
    unix_time: Option<u64>,
) -> Result<ExitReason, String> {
    let mut ram = GuestRam::new(memory_mib)?;
    speed_up_firmware_timer();
    let clock = Clock::calibrate();
    // The guest cannot calibrate the TSC against the emulated PIT, as exits make its readings
    // too noisy, so it is told.
    let cmdline = format!(
        "{cmdline} tsc_early_khz={} tsc=reliable",
        clock.hz / HZ_PER_KHZ
    );
    let boot =
        uemulator::load_linux(ram.bytes(), kernel, initrd, &cmdline).map_err(|e| format!("{e}"))?;
    let console = ConsoleHandle::default();
    console.on_output(console::queue);
    let mut platform = Platform::new(console.clone());
    if let Some(t) = unix_time {
        platform.set_boot_time(t);
    }
    let tables = ram.build_tables(kind == Kind::Vmx)?;
    let mut vcpu: Box<dyn Vcpu> = match kind {
        Kind::Vmx => Box::new(vmx::Vmx::new(tables, &boot)?),
        Kind::Svm => Box::new(svm::Svm::new(tables, &boot)?),
    };
    console::print(&format!(
        "uemulator: hardware virtualization ({kind:?}), {memory_mib} MiB\n"
    ));
    event_loop(&mut *vcpu, &mut ram, &mut platform, &console, &clock)
}

#[cfg(target_os = "uefi")]
fn event_loop(
    vcpu: &mut dyn Vcpu,
    ram: &mut GuestRam,
    platform: &mut Platform,
    console: &ConsoleHandle,
    clock: &Clock,
) -> Result<ExitReason, String> {
    let mut fpu = FpuState::reset();
    let mut next_poll = 0u64;
    let mut logged = 0u32;
    loop {
        // Devices only advance here, between entries. The firmware tick forces an exit at least
        // every millisecond or so (`speed_up_firmware_timer`), which bounds how late they run.
        let now = clock.now_ns();
        platform.tick(now, ram.bytes());
        if now >= next_poll {
            next_poll = now + CONSOLE_POLL_NS;
            poll_console(console);
        }
        if platform.reset_requested() {
            return Ok(ExitReason::Reset);
        }
        if platform.irq_pending() {
            if vcpu.irq_ok() {
                if let Some(v) = platform.irq_ack() {
                    vcpu.inject_irq(v);
                }
                vcpu.window(false);
            } else {
                vcpu.window(true);
            }
        } else {
            vcpu.window(false);
        }

        fpu.restore();
        let exit = vcpu.run();
        fpu.save();
        let exit = exit?;

        match exit {
            Exit::HostInterrupt => arch::host_interrupt_window(),
            // Nothing to do: the exit itself was the point, the next iteration injects.
            Exit::IntWindow => {}
            Exit::Io { port, size, write } => {
                let now = clock.now_ns();
                let g = vcpu.gprs();
                if write {
                    platform.io_out(port, size, out_value(g.rax, size), now);
                } else {
                    let v = platform.io_in(port, size, now);
                    g.rax = rax_after_in(g.rax, v, size);
                }
                vcpu.skip();
            }
            Exit::Cpuid => {
                let g = vcpu.gprs();
                let r = cpuid::guest(g.rax as u32, g.rcx as u32);
                g.rax = r[0] as u64;
                g.rbx = r[1] as u64;
                g.rcx = r[2] as u64;
                g.rdx = r[3] as u64;
                vcpu.skip();
            }
            Exit::Rdmsr => {
                let n = vcpu.gprs().rcx as u32;
                match msr::read(vcpu, n) {
                    Some(v) => {
                        let g = vcpu.gprs();
                        (g.rax, g.rdx) = rdmsr_regs(v);
                        vcpu.skip();
                    }
                    // Probing an MSR that is absent is normal; the guest handles the #GP.
                    None => vcpu.inject_gp(),
                }
            }
            Exit::Wrmsr => {
                let g = vcpu.gprs();
                let (n, v) = (g.rcx as u32, wrmsr_value(g.rax, g.rdx));
                if msr::write(vcpu, n, v) {
                    vcpu.skip();
                } else {
                    note(&mut logged, || {
                        format!("wrmsr {n:#x} = {v:#x} at {:#x}", vcpu.rip())
                    });
                    vcpu.inject_gp();
                }
            }
            Exit::Hlt => {
                vcpu.skip();
                loop {
                    let now = clock.now_ns();
                    platform.tick(now, ram.bytes());
                    poll_console(console);
                    match hlt_outcome(vcpu, platform.irq_pending(), platform.reset_requested()) {
                        HltOutcome::Wake => break,
                        HltOutcome::PowerOff => return Ok(ExitReason::Shutdown),
                        HltOutcome::Sleep => boot::stall(HLT_POLL_INTERVAL),
                    }
                }
            }
            Exit::TripleFault => return Ok(ExitReason::TripleFault),
            Exit::Skip => vcpu.skip(),
            Exit::Undefined => vcpu.inject_ud(),
            Exit::Fault { gpa, write } => {
                return Err(format!(
                    "guest touched unmapped physical address {gpa:#x} ({}) at rip {:#x}",
                    if write { "write" } else { "read" },
                    vcpu.rip()
                ));
            }
            Exit::Other(code) => {
                return Err(format!(
                    "unhandled VM exit {code:#x} at rip {:#x}",
                    vcpu.rip()
                ));
            }
        }
    }
}

/// Reports the first few unusual guest actions, so a failing boot says what it tripped on.
#[cfg(target_os = "uefi")]
fn note(count: &mut u32, msg: impl FnOnce() -> String) {
    if *count < MAX_NOTES {
        *count += 1;
        console::print(&format!("uemulator: {}\n", msg()));
    }
}

#[cfg(target_os = "uefi")]
fn poll_console(console: &ConsoleHandle) {
    console::flush();
    let input = console::read_input();
    if !input.is_empty() {
        console.write(&input);
    }
}

/// The guest only runs between physical interrupts, so the firmware's tick (10 ms by default)
/// bounds how late its timer interrupts arrive. Ask for 1 ms where the firmware allows it.
#[cfg(target_os = "uefi")]
fn speed_up_firmware_timer() {
    crate::timer::set_period(FIRMWARE_TICK);
}

#[cfg(test)]
mod tests {
    use super::fake::FakeVcpu;
    use super::*;
    use alloc::boxed::Box;
    use core::mem::size_of;

    // Port I/O.

    #[test]
    fn byte_in_replaces_only_the_low_byte_of_rax() {
        assert_eq!(
            rax_after_in(0x1122_3344_5566_7788, 0xab, 1),
            0x1122_3344_5566_77ab
        );
    }

    #[test]
    fn word_in_replaces_only_the_low_word_of_rax() {
        assert_eq!(
            rax_after_in(0x1122_3344_5566_7788, 0xabcd, 2),
            0x1122_3344_5566_abcd
        );
    }

    #[test]
    fn dword_in_zero_extends_into_rax() {
        assert_eq!(
            rax_after_in(0x1122_3344_5566_7788, 0x89ab_cdef, 4),
            0x0000_0000_89ab_cdef
        );
    }

    #[test]
    fn in_ignores_value_bits_above_the_access_size() {
        assert_eq!(rax_after_in(0, 0x1234, 1), 0x34);
        assert_eq!(rax_after_in(0, 0x12_3456, 2), 0x3456);
    }

    #[test]
    fn out_sends_only_the_bytes_of_the_access_size() {
        let rax = 0xffff_ffff_8765_4321;
        assert_eq!(out_value(rax, 1), 0x21);
        assert_eq!(out_value(rax, 2), 0x4321);
        assert_eq!(out_value(rax, 4), 0x8765_4321);
    }

    // MSR register conventions.

    #[test]
    fn rdmsr_splits_the_value_into_eax_and_edx() {
        assert_eq!(
            rdmsr_regs(0xdead_beef_0123_4567),
            (0x0123_4567, 0xdead_beef)
        );
    }

    #[test]
    fn rdmsr_of_a_small_value_leaves_edx_zero() {
        assert_eq!(rdmsr_regs(7), (7, 0));
    }

    #[test]
    fn wrmsr_joins_edx_and_eax() {
        assert_eq!(wrmsr_value(0x0123_4567, 0xdead_beef), 0xdead_beef_0123_4567);
    }

    #[test]
    fn wrmsr_ignores_the_upper_halves_of_rax_and_rdx() {
        assert_eq!(
            wrmsr_value(0xffff_ffff_0000_0001, 0xffff_ffff_0000_0002),
            0x0000_0002_0000_0001
        );
    }

    // Clock.

    #[test]
    fn a_one_ghz_tsc_counts_one_tick_per_nanosecond() {
        assert_eq!(ticks_to_ns(12_345, 1_000_000_000), 12_345);
    }

    #[test]
    fn ticks_scale_by_the_frequency() {
        assert_eq!(ticks_to_ns(2_500_000_000, 2_500_000_000), 1_000_000_000);
        assert_eq!(ticks_to_ns(3_000, 3_000_000_000), 1_000);
        assert_eq!(ticks_to_ns(0, 3_000_000_000), 0);
    }

    #[test]
    fn long_uptimes_do_not_overflow_the_conversion() {
        // About 92 hours at 3 GHz: the product with 10^9 does not fit in 64 bits.
        assert_eq!(
            ticks_to_ns(1_000_000_000_000_000, 3_000_000_000),
            333_333_333_333_333
        );
    }

    #[test]
    fn calibration_scales_the_ticks_of_the_window_to_a_second() {
        assert_eq!(calibrated_hz(150_000_000), 3_000_000_000);
        assert_eq!(MS_PER_SEC / CALIBRATION_MS, 20);
    }

    #[test]
    fn calibration_never_yields_a_zero_frequency() {
        assert_eq!(calibrated_hz(0), 1);
    }

    // Guest RAM and the nested tables.

    #[test]
    fn ram_is_rounded_up_to_whole_huge_pages() {
        let (len, mapped, _) = ram_layout(3);
        assert_eq!(len, 3 << 20);
        assert_eq!(mapped, 4 << 20);
    }

    #[test]
    fn whole_huge_page_sizes_are_not_rounded() {
        let (len, mapped, _) = ram_layout(1024);
        assert_eq!(len, 1 << 30);
        assert_eq!(mapped, 1 << 30);
    }

    #[test]
    fn ram_allocation_has_one_spare_huge_page_for_alignment() {
        let (_, mapped, pages) = ram_layout(4);
        assert_eq!(pages * PAGE_SIZE, mapped + HUGE_PAGE_SIZE);
    }

    #[test]
    fn zero_ram_still_asks_for_the_alignment_slack() {
        assert_eq!(ram_layout(0), (0, 0, HUGE_PAGE_SIZE / PAGE_SIZE));
    }

    #[repr(align(4096))]
    struct Table([u64; 512]);

    /// Hands out zeroed 4 KiB-aligned tables and keeps count.
    struct Tables {
        made: usize,
        limit: usize,
    }

    impl Tables {
        fn new() -> Self {
            Self {
                made: 0,
                limit: usize::MAX,
            }
        }

        fn alloc(&mut self) -> Result<*mut u64, String> {
            if self.made == self.limit {
                return Err("out of tables".into());
            }
            self.made += 1;
            Ok(Box::leak(Box::new(Table([0; 512]))).0.as_mut_ptr())
        }
    }

    const HOST_BASE: u64 = 0x4000_0000;
    const ACCESS: u64 = 0b111;
    const LARGE: u64 = 1 << 7;
    const WRITE_BACK: u64 = 6 << 3;

    fn ram(mapped: usize) -> GuestRam {
        GuestRam {
            ptr: HOST_BASE as *mut u8,
            len: mapped,
            mapped,
        }
    }

    fn entry(table: u64, index: usize) -> u64 {
        // Tables are plain host memory here, as under UEFI where physical equals virtual.
        unsafe { *((table & !0xfff) as *const u64).add(index) }
    }

    /// Walks the 4-level tables like the MMU: 2 MiB leaves at the page-directory level.
    fn translate(root: u64, gpa: u64) -> Option<u64> {
        let indices = [
            (gpa >> 39) & 0x1ff,
            (gpa >> 30) & 0x1ff,
            (gpa >> 21) & 0x1ff,
        ];
        let mut table = root;
        for (level, index) in indices.into_iter().enumerate() {
            let e = entry(table, index as usize);
            if e & ACCESS == 0 {
                return None;
            }
            if level == 2 {
                assert_ne!(e & LARGE, 0, "page-directory entry must be a 2 MiB leaf");
                return Some((e & 0x000f_ffff_ffe0_0000) + (gpa & 0x1f_ffff));
            }
            table = e;
        }
        None
    }

    #[test]
    fn tables_map_guest_physical_zero_onto_the_start_of_ram() {
        let mut tables = Tables::new();
        let root = ram(4 << 20).fill_tables(false, || tables.alloc()).unwrap();
        assert_eq!(translate(root, 0), Some(HOST_BASE));
        assert_eq!(translate(root, 0x1234), Some(HOST_BASE + 0x1234));
    }

    #[test]
    fn tables_map_every_huge_page_in_order() {
        let mut tables = Tables::new();
        let root = ram(8 << 20).fill_tables(false, || tables.alloc()).unwrap();
        for page in 0..4u64 {
            let gpa = page * HUGE_PAGE_SIZE as u64;
            assert_eq!(translate(root, gpa), Some(HOST_BASE + gpa), "page {page}");
            assert_eq!(
                translate(root, gpa + 0x1f_ffff),
                Some(HOST_BASE + gpa + 0x1f_ffff)
            );
        }
    }

    #[test]
    fn addresses_past_the_mapped_end_are_not_mapped() {
        let mut tables = Tables::new();
        let root = ram(4 << 20).fill_tables(false, || tables.alloc()).unwrap();
        assert_eq!(translate(root, 4 << 20), None);
        assert_eq!(translate(root, 1 << 30), None);
        assert_eq!(translate(root, 1 << 39), None);
    }

    #[test]
    fn a_small_guest_needs_one_table_per_level() {
        let mut tables = Tables::new();
        ram(4 << 20).fill_tables(false, || tables.alloc()).unwrap();
        assert_eq!(tables.made, 3, "PML4, PDPT and one page directory");
    }

    #[test]
    fn exactly_one_gib_fills_one_page_directory() {
        let mut tables = Tables::new();
        let root = ram(1 << 30).fill_tables(false, || tables.alloc()).unwrap();
        assert_eq!(tables.made, 3);
        assert_eq!(
            translate(root, (1 << 30) - 1),
            Some(HOST_BASE + (1 << 30) - 1)
        );
        assert_eq!(translate(root, 1 << 30), None);
    }

    #[test]
    fn memory_beyond_a_gib_gets_a_second_page_directory() {
        let mut tables = Tables::new();
        let root = ram((1 << 30) + (4 << 20))
            .fill_tables(false, || tables.alloc())
            .unwrap();
        assert_eq!(tables.made, 4);
        let gpa = (1u64 << 30) + 0x20_1234;
        assert_eq!(translate(root, gpa), Some(HOST_BASE + gpa));
        assert_eq!(translate(root, (1 << 30) + (4 << 20)), None);
    }

    #[test]
    fn the_top_entries_are_present_writable_and_user_tables() {
        let mut tables = Tables::new();
        let root = ram(4 << 20).fill_tables(false, || tables.alloc()).unwrap();
        let pml4e = entry(root, 0);
        assert_eq!(pml4e & 0xfff, ACCESS);
        assert_eq!(entry(root, 1), 0);
        assert_eq!(entry(pml4e, 0) & 0xfff, ACCESS);
    }

    #[test]
    fn npt_leaves_are_ordinary_large_page_entries() {
        let mut tables = Tables::new();
        let root = ram(4 << 20).fill_tables(false, || tables.alloc()).unwrap();
        let pd = entry(entry(root, 0), 0);
        assert_eq!(entry(pd, 0), HOST_BASE | ACCESS | LARGE);
        assert_eq!(entry(pd, 1), (HOST_BASE + (2 << 20)) | ACCESS | LARGE);
    }

    #[test]
    fn ept_leaves_carry_the_write_back_memory_type() {
        let mut tables = Tables::new();
        let root = ram(4 << 20).fill_tables(true, || tables.alloc()).unwrap();
        let pd = entry(entry(root, 0), 0);
        assert_eq!(entry(pd, 0), HOST_BASE | ACCESS | WRITE_BACK | LARGE);
        // Higher-level EPT entries have no memory type.
        assert_eq!(entry(root, 0) & 0xfff, ACCESS);
    }

    #[test]
    fn an_allocation_failure_is_passed_on() {
        let mut tables = Tables::new();
        tables.limit = 2;
        let result = ram(4 << 20).fill_tables(false, || tables.alloc());
        assert_eq!(result, Err("out of tables".into()));
    }

    // Idle wake-up and power-off.

    #[test]
    fn a_pending_interrupt_wakes_a_halted_guest_that_can_take_it() {
        let vcpu = FakeVcpu::with_interrupts(true);
        assert_eq!(hlt_outcome(&vcpu, true, false), HltOutcome::Wake);
    }

    #[test]
    fn a_pending_interrupt_does_not_wake_a_guest_halted_with_interrupts_off() {
        let vcpu = FakeVcpu::with_interrupts(false);
        assert_eq!(hlt_outcome(&vcpu, true, false), HltOutcome::PowerOff);
    }

    #[test]
    fn without_a_pending_interrupt_a_guest_with_interrupts_on_stays_halted() {
        let vcpu = FakeVcpu::with_interrupts(true);
        assert_eq!(hlt_outcome(&vcpu, false, false), HltOutcome::Sleep);
    }

    #[test]
    fn a_guest_halted_with_interrupts_off_and_nothing_pending_is_powered_off() {
        let vcpu = FakeVcpu::with_interrupts(false);
        assert_eq!(hlt_outcome(&vcpu, false, false), HltOutcome::PowerOff);
    }

    #[test]
    fn reset_wakes_the_guest_whatever_its_interrupt_state() {
        for interrupts in [true, false] {
            for pending in [true, false] {
                let vcpu = FakeVcpu::with_interrupts(interrupts);
                assert_eq!(hlt_outcome(&vcpu, pending, true), HltOutcome::Wake);
            }
        }
    }

    #[test]
    fn an_interrupt_shadow_defers_the_wake_up_but_is_not_a_power_off() {
        let mut vcpu = FakeVcpu::with_interrupts(true);
        vcpu.interrupt_shadow = true;
        assert_eq!(hlt_outcome(&vcpu, true, false), HltOutcome::Sleep);
    }

    #[test]
    fn sti_then_hlt_still_wakes_since_the_exit_ends_the_shadow() {
        let mut vcpu = FakeVcpu::with_interrupts(true);
        vcpu.interrupt_shadow = true;
        vcpu.skip();
        assert_eq!(hlt_outcome(&vcpu, true, false), HltOutcome::Wake);
    }

    #[test]
    fn an_event_already_queued_for_injection_defers_the_wake_up_but_is_not_a_power_off() {
        let mut vcpu = FakeVcpu::with_interrupts(true);
        vcpu.inject_ud();
        assert_eq!(hlt_outcome(&vcpu, true, false), HltOutcome::Sleep);
    }

    // Layout.

    #[test]
    fn page_sizes_are_what_the_hardware_expects() {
        assert_eq!(PAGE_SIZE, 4096);
        assert_eq!(HUGE_PAGE_SIZE, 2 * 1024 * 1024);
        assert_eq!(GIB, 1 << 30);
        assert_eq!(PAGE_SIZE / size_of::<u64>(), 512);
    }
}
