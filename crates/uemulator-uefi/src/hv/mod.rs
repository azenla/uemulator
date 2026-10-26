//! Runs the guest on the CPU with hardware virtualization (Intel VMX or AMD SVM), with the CPU's
//! nested paging in place of the emulator's soft TLB. The legacy devices still come from the
//! library: port I/O exits are forwarded to `Platform`, which also supplies interrupts.

mod arch;
mod cpuid;
mod msr;
mod svm;
mod vmx;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use core::time::Duration;

use uefi::boot::{self, AllocateType, MemoryType};
use uemulator::{ConsoleHandle, ExitReason, Platform};

use crate::console;
use arch::{FpuState, Gprs};

const PAGE: usize = 4096;
const HUGE: usize = 2 << 20;
/// How often the firmware console is polled for input, in guest-visible microseconds.
const CONSOLE_POLL_NS: u64 = 1_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Vmx,
    Svm,
}

/// Which extension this CPU offers, if any (and whether the firmware left it usable).
pub fn detect() -> Option<Kind> {
    let leaf1 = arch::cpuid(1, 0);
    if leaf1[2] & (1 << 5) != 0 && vmx::usable() {
        return Some(Kind::Vmx);
    }
    let max_ext = arch::cpuid(0x8000_0000, 0)[0];
    if max_ext >= 0x8000_000a && arch::cpuid(0x8000_0001, 0)[2] & (1 << 2) != 0 && svm::usable() {
        return Some(Kind::Svm);
    }
    None
}

pub enum Exit {
    Io {
        port: u16,
        size: u8,
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
    Fault {
        gpa: u64,
        write: bool,
    },
    Other(u64),
}

pub trait Vcpu {
    fn gprs(&mut self) -> &mut Gprs;
    fn run(&mut self) -> Result<Exit, String>;
    /// Moves the guest past the instruction behind the last exit.
    fn skip(&mut self);
    fn rip(&self) -> u64;
    fn irq_ok(&self) -> bool;
    fn inject_irq(&mut self, vector: u8);
    fn window(&mut self, on: bool);
    fn inject_gp(&mut self);
    fn inject_ud(&mut self);
    fn efer(&self) -> u64;
    fn set_efer(&mut self, efer: u64);
}

/// Zeroed, page-aligned pages for hardware structures.
pub fn alloc_pages(n: usize) -> Result<*mut u8, String> {
    let p = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, n)
        .map_err(|e| format!("out of memory: {e}"))?;
    unsafe { core::ptr::write_bytes(p.as_ptr(), 0, n * PAGE) };
    Ok(p.as_ptr())
}

/// Guest RAM: contiguous host memory, mapped 1:1 into the guest physical address space.
pub struct GuestRam {
    ptr: *mut u8,
    len: usize,
    mapped: usize,
}

impl GuestRam {
    fn new(mib: usize) -> Result<Self, String> {
        let len = mib << 20;
        let mapped = len.next_multiple_of(HUGE);
        let pages = (mapped + HUGE) / PAGE;
        let raw = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages)
            .map_err(|e| format!("cannot allocate {mib} MiB of guest memory: {e}"))?;
        let ptr = (raw.as_ptr() as usize).next_multiple_of(HUGE) as *mut u8;
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
    /// `ept` selects Intel's entry format over AMD's (which is the ordinary x86 format).
    pub fn build_tables(&self, ept: bool) -> Result<u64, String> {
        let page = || alloc_pages(1).map(|p| p as *mut u64);
        let table = 0x7;
        let leaf = if ept {
            0x7 | (6 << 3) | (1 << 7)
        } else {
            0x7 | (1 << 7)
        };
        let pml4 = page()?;
        let pdpt = page()?;
        unsafe {
            *pml4 = pdpt as u64 | table;
            for (g, gb) in (0..self.mapped).step_by(1 << 30).enumerate() {
                let pd = page()?;
                *pdpt.add(g) = pd as u64 | table;
                let n = ((self.mapped - gb) / HUGE).min(512);
                for i in 0..n {
                    *pd.add(i) = (self.host_addr() + (gb + i * HUGE) as u64) | leaf;
                }
            }
        }
        Ok(pml4 as u64)
    }
}

/// Host time in guest nanoseconds, from the TSC.
struct Clock {
    tsc0: u64,
    hz: u64,
}

impl Clock {
    fn calibrate() -> Self {
        let t0 = arch::rdtsc();
        boot::stall(Duration::from_millis(50));
        let t1 = arch::rdtsc();
        Self {
            tsc0: t1,
            hz: ((t1 - t0) * 20).max(1),
        }
    }

    fn now_ns(&self) -> u64 {
        ((arch::rdtsc() - self.tsc0) as u128 * 1_000_000_000 / self.hz as u128) as u64
    }
}

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
    let cmdline = format!("{cmdline} tsc_early_khz={} tsc=reliable", clock.hz / 1000);
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
            Exit::IntWindow => {}
            Exit::Io { port, size, write } => {
                let now = clock.now_ns();
                let g = vcpu.gprs();
                if write {
                    let v = g.rax as u32 & (u32::MAX >> (32 - 8 * size as u32));
                    platform.io_out(port, size, v, now);
                } else {
                    let v = platform.io_in(port, size, now) as u64;
                    let mask = u64::MAX >> (64 - 8 * size as u32);
                    g.rax = if size == 4 {
                        v
                    } else {
                        (g.rax & !mask) | (v & mask)
                    };
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
                        g.rax = v & 0xffff_ffff;
                        g.rdx = v >> 32;
                        vcpu.skip();
                    }
                    // Probing an MSR that is absent is normal; the guest handles the #GP.
                    None => vcpu.inject_gp(),
                }
            }
            Exit::Wrmsr => {
                let g = vcpu.gprs();
                let (n, v) = (g.rcx as u32, (g.rdx << 32) | (g.rax & 0xffff_ffff));
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
                    if platform.irq_pending() || platform.reset_requested() {
                        break;
                    }
                    boot::stall(Duration::from_micros(50));
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
fn note(count: &mut u32, msg: impl FnOnce() -> String) {
    if *count < 20 {
        *count += 1;
        console::print(&format!("uemulator: {}\n", msg()));
    }
}

fn poll_console(console: &ConsoleHandle) {
    console::flush();
    let input = console::read_input();
    if !input.is_empty() {
        console.write(&input);
    }
}

/// The guest only runs between physical interrupts, so the firmware's tick (10 ms by default)
/// bounds how late its timer interrupts arrive. Ask for 1 ms where the firmware allows it.
fn speed_up_firmware_timer() {
    crate::timer::set_period(Duration::from_millis(1));
}
