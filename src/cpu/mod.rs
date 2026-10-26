pub(crate) mod decode;
pub(crate) mod exc;
mod exec;
pub(crate) mod flags;
mod fpu;
pub(crate) mod mem;
mod sse;
mod sys;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::boot::BootState;
use crate::console::ConsoleHandle;
use crate::dev::Devices;
use crate::machine::{ExitReason, LogFn};

use decode::Insn;
use flags::Cc;

pub(crate) use exc::Exc;

pub(crate) type R<T> = Result<T, Exc>;

enum Pre {
    Exit(ExitReason),
    Again,
    Go,
}

pub(crate) const NS_PER_INSN: u64 = 1;
const EVENT_INTERVAL: u64 = 4096;

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Seg {
    pub sel: u16,
    pub base: u64,
    pub limit: u32,
    pub flags: u32,
}

impl Seg {
    pub fn dpl(&self) -> u8 {
        ((self.flags >> 13) & 3) as u8
    }
    pub fn present(&self) -> bool {
        self.flags & (1 << 15) != 0
    }
    pub fn is_code(&self) -> bool {
        self.flags & (1 << 11) != 0 && self.flags & (1 << 12) != 0
    }
    pub fn long(&self) -> bool {
        self.flags & (1 << 21) != 0
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Dt {
    pub base: u64,
    pub limit: u16,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct TlbEntry {
    pub tag_r: u64,
    pub tag_w: u64,
    pub tag_x: u64,
    pub addend: u64,
}

pub(crate) const TLB_INV: u64 = u64::MAX;
pub(crate) const TLB_EMPTY: TlbEntry = TlbEntry {
    tag_r: TLB_INV,
    tag_w: TLB_INV,
    tag_x: TLB_INV,
    addend: 0,
};

pub(crate) const TLB_W: u8 = 1;
pub(crate) const TLB_U: u8 = 2;
pub(crate) const TLB_NX: u8 = 4;
pub(crate) const TLB_D: u8 = 8;
pub(crate) const TLB_BITS: usize = 10;
pub(crate) const TLB_N: usize = 1 << TLB_BITS;

#[derive(Clone, Copy)]
pub(crate) struct DEntry {
    pub gpa: u64,
    pub gen_: u32,
    pub insn: Insn,
}

pub(crate) const DC_SIZE: usize = 1 << 16;

#[repr(C)]
pub(crate) struct Cpu {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub eflags: u64,
    pub cc: Cc,
    pub cc_size: u8,
    pub cc_dst: u64,
    pub cc_src: u64,
    pub cc_src2: u64,
    pub tlb_cur: u64,
    pub cpl: u8,
    pub exit_block: bool,
    pub prepaid: u64,
    pub seg: [Seg; 6],
    pub gdtr: Dt,
    pub idtr: Dt,
    pub ldtr: Seg,
    pub tr: Seg,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub cr8: u64,
    pub efer: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub sfmask: u64,
    pub kgsbase: u64,
    pub tsc_aux: u64,
    pub pat: u64,
    pub sysenter: [u64; 3],
    pub dr: [u64; 8],
    pub msr_misc: std::collections::HashMap<u32, u64>,
    pub xmm: [u128; 16],
    pub mxcsr: u32,
    pub fcw: u16,
    pub fsw: u16,
    pub ftw: u16,
    pub st: [[u8; 10]; 8],
    pub halted: bool,
    pub shadow: bool,
    pub ram: Vec<u8>,
    pub ram_base: u64,
    pub tlb: Vec<TlbEntry>,
    pub dcache: Vec<DEntry>,
    pub dc_gen: u32,
    pub code_page: Vec<u8>,
    pub code_list: Vec<u32>,
    pub dev: Devices,
    pub icount: u64,
    pub warp_ns: u64,
    pub next_event: u64,
    pub stop: Arc<AtomicBool>,
    pub log: Option<LogFn>,
    pub logged: std::collections::HashSet<u64>,
    pub exc_depth: u32,
    pub flushes: u64,
    pub realtime: bool,
    pub last_done: u64,
    pub pending: Option<Exc>,
    pub jit: Option<Box<crate::jit::Jit>>,
    pub trace: Option<std::collections::VecDeque<u64>>,
}

impl Cpu {
    pub fn new(
        ram: Vec<u8>,
        boot: BootState,
        console: ConsoleHandle,
        stop: Arc<AtomicBool>,
        log: Option<LogFn>,
    ) -> Self {
        let pages = ram.len() >> 12;
        let flat = |sel: u16, flags: u32| Seg {
            sel,
            base: 0,
            limit: 0xffff_ffff,
            flags,
        };
        let mut seg = [Seg::default(); 6];
        let ds = flat(boot.ds, 0x00cf_9200);
        seg[0] = ds;
        seg[1] = flat(boot.cs, 0x0020_9a00);
        seg[2] = ds;
        seg[3] = ds;
        seg[4] = Seg::default();
        seg[5] = Seg::default();
        let mut cpu = Cpu {
            gpr: {
                let mut g = [0u64; 16];
                g[6] = boot.rsi;
                g
            },
            rip: boot.rip,
            eflags: 2,
            cc: Cc::Eflags,
            cc_size: 8,
            cc_dst: 0,
            cc_src: 0,
            cc_src2: 0,
            seg,
            gdtr: Dt {
                base: boot.gdt_base,
                limit: boot.gdt_limit,
            },
            idtr: Dt::default(),
            ldtr: Seg::default(),
            tr: Seg::default(),
            cr0: 0x8000_0033,
            cr2: 0,
            cr3: boot.cr3,
            cr4: 0x20,
            cr8: 0,
            efer: 0x500,
            star: 0,
            lstar: 0,
            cstar: 0,
            sfmask: 0,
            kgsbase: 0,
            tsc_aux: 0,
            pat: 0x0007_0406_0007_0406,
            sysenter: [0; 3],
            dr: [0, 0, 0, 0, 0, 0, 0xffff_0ff0, 0x400],
            msr_misc: [
                (0x8b, 0),
                (0xc001_0015, 1 << 24),
                (0xc001_001f, 0),
                (0xc001_1020, 0),
                (0xc001_1029, 0),
            ]
            .into_iter()
            .collect(),
            cpl: 0,
            xmm: [0; 16],
            mxcsr: 0x1f80,
            fcw: 0x37f,
            fsw: 0,
            ftw: 0,
            st: [[0; 10]; 8],
            halted: false,
            shadow: false,
            ram,
            tlb: vec![TLB_EMPTY; 2 * TLB_N],
            ram_base: 0,
            dcache: vec![
                DEntry {
                    gpa: 0,
                    gen_: 0,
                    insn: Insn::default()
                };
                DC_SIZE
            ],
            dc_gen: 1,
            code_page: vec![0; pages],
            code_list: Vec::new(),
            dev: Devices::new(console),
            icount: 0,
            warp_ns: 0,
            next_event: 0,
            stop,
            log,
            logged: Default::default(),
            exc_depth: 0,
            flushes: 0,
            realtime: false,
            last_done: 0,
            pending: None,
            jit: None,
            tlb_cur: 0,
            exit_block: false,
            prepaid: 0,
            trace: None,
        };
        cpu.ram_base = cpu.ram.as_ptr() as u64;
        cpu.set_cpl(0);
        cpu
    }

    pub fn set_cpl(&mut self, c: u8) {
        self.cpl = c;
        self.tlb_cur = self.tlb.as_ptr() as u64 + (c == 3) as u64 * (TLB_N * 32) as u64;
    }

    pub fn now_ns(&self) -> u64 {
        self.icount * NS_PER_INSN + self.warp_ns
    }

    pub fn tsc(&self) -> u64 {
        self.now_ns()
    }

    pub fn log_once(&mut self, key: u64, msg: impl FnOnce() -> String) {
        if self.logged.insert(key)
            && let Some(l) = &self.log
        {
            l(&msg());
        }
    }

    fn service_events(&mut self) -> Option<ExitReason> {
        if self.stop.load(Ordering::Relaxed) {
            return Some(ExitReason::Stopped);
        }
        let now = self.now_ns();
        self.dev.tick(now);
        if self.dev.reset_requested {
            return Some(ExitReason::Reset);
        }
        self.next_event = self.icount + EVENT_INTERVAL;
        None
    }

    fn pending_interrupt(&mut self) -> Option<u8> {
        if self.eflags & flags::IF == 0 || self.shadow {
            return None;
        }
        self.dev.pic.ack()
    }

    fn pre(&mut self) -> Pre {
        if self.icount >= self.next_event
            && let Some(r) = self.service_events()
        {
            return Pre::Exit(r);
        }
        if self.eflags & flags::IF != 0
            && !self.shadow
            && self.dev.pic.has_pending()
            && let Some(v) = self.pending_interrupt()
        {
            self.halted = false;
            if let Some(r) = self.deliver_event(v, None, false) {
                return Pre::Exit(r);
            }
        }
        if self.halted {
            return match self.idle() {
                Some(r) => Pre::Exit(r),
                None => Pre::Again,
            };
        }
        Pre::Go
    }

    fn after(&mut self, r: R<()>) -> Option<ExitReason> {
        match r {
            Ok(()) => None,
            Err(Exc::Fault { vec, err, soft }) => self.deliver_event(vec, err, soft),
            Err(Exc::Exit(r)) => Some(r),
        }
    }

    pub fn run(&mut self) -> ExitReason {
        loop {
            match self.pre() {
                Pre::Exit(r) => return r,
                Pre::Again => continue,
                Pre::Go => {}
            }
            self.shadow = false;
            self.icount += 1;
            let r = if self.jit.is_some() {
                self.step_block()
            } else {
                self.step()
            };
            if let Some(r) = self.after(r) {
                return r;
            }
        }
    }

    pub fn run_verify(&mut self, sh: &mut Cpu) -> ExitReason {
        loop {
            let (a, b) = (self.pre(), sh.pre());
            match (a, b) {
                (Pre::Exit(r), _) => return r,
                (Pre::Again, Pre::Again) => continue,
                (Pre::Go, Pre::Go) => {}
                _ => return ExitReason::Internal("verify: pre-step divergence".into()),
            }
            self.shadow = false;
            sh.shadow = false;
            let before = (self.gpr, self.eflags_full(), self.rip);
            let sh_before = (sh.rip, sh.icount, self.icount);
            let start = self.rip;
            self.icount += 1;
            sh.icount += 1;
            let rm = self.step_block();
            let done = self.last_done;
            let attempts = if rm.is_err() { done + 1 } else { done };
            let mut rs = Ok(());
            for k in 0..attempts {
                if k > 0 {
                    sh.icount += 1;
                }
                rs = sh.step();
                if rs.is_err() {
                    break;
                }
            }
            let same_exc = match (&rm, &rs) {
                (Ok(()), Ok(())) => true,
                (Err(Exc::Fault { vec: a, err: ea, .. }), Err(Exc::Fault { vec: b, err: eb, .. })) => a == b && ea == eb,
                (Err(Exc::Exit(_)), Err(Exc::Exit(_))) => true,
                _ => false,
            };
            let diff = self.diff(sh);
            if !same_exc || diff.is_some() {
                let pa = self.translate(start, mem::Acc::Exec, true).unwrap_or(0) as usize & !0xfff;
                let page_diff: Vec<usize> = (0..4096)
                    .filter(|&k| self.ram[pa + k] != sh.ram[pa + k])
                    .take(8)
                    .collect();
                let bytes_m = self.ram[pa + (start as usize & 0xfff)..][..24].to_vec();
                let bytes_s = sh.ram[pa + (start as usize & 0xfff)..][..24].to_vec();
                if let Some(l) = &self.log {
                    l(&format!("page diffs at {page_diff:x?}; main bytes {bytes_m:02x?}; shadow bytes {bytes_s:02x?}"));
                }
                let msg = format!(
                    "verify: divergence after block at {start:#x} ({done} insns, attempts {attempts}, shadow start {sh_before:x?}): exc main={:?} shadow={:?} diff={:?}\n--- before {before:x?}\n--- main\n{}--- shadow\n{}",
                    rm.as_ref().err(),
                    rs.as_ref().err(),
                    diff,
                    self.dump_regs(),
                    sh.dump_regs()
                );
                return ExitReason::Internal(msg);
            }
            if let Some(r) = self.after(rm) {
                return r;
            }
            if let Some(r) = sh.after(rs) {
                return r;
            }
        }
    }

    fn diff(&mut self, o: &mut Cpu) -> Option<String> {
        let mut d = Vec::new();
        for k in 0..16 {
            if self.gpr[k] != o.gpr[k] {
                d.push(format!("r{k}: {:#x} vs {:#x}", self.gpr[k], o.gpr[k]));
            }
        }
        if self.rip != o.rip {
            d.push(format!("rip: {:#x} vs {:#x}", self.rip, o.rip));
        }
        let (fa, fb) = (self.eflags_full(), o.eflags_full());
        if fa != fb {
            d.push(format!("eflags: {fa:#x} vs {fb:#x}"));
        }
        if self.cpl != o.cpl {
            d.push("cpl".into());
        }
        for k in 0..6 {
            if self.seg[k].sel != o.seg[k].sel || self.seg[k].base != o.seg[k].base {
                d.push(format!("seg{k}"));
            }
        }
        for (n, a, b) in [
            ("cr0", self.cr0, o.cr0),
            ("cr2", self.cr2, o.cr2),
            ("cr3", self.cr3, o.cr3),
            ("cr4", self.cr4, o.cr4),
            ("efer", self.efer, o.efer),
            ("icount", self.icount, o.icount),
        ] {
            if a != b {
                d.push(format!("{n}: {a:#x} vs {b:#x}"));
            }
        }
        if self.xmm != o.xmm {
            d.push("xmm".into());
        }
        if d.is_empty() { None } else { Some(d.join(", ")) }
    }

    fn idle(&mut self) -> Option<ExitReason> {
        if self.stop.load(Ordering::Relaxed) {
            return Some(ExitReason::Stopped);
        }
        let now = self.now_ns();
        match self.dev.next_deadline(now) {
            Some(t) => {
                if t > now {
                    let mut dt = t - now;
                    if self.realtime {
                        dt = dt.min(2_000_000);
                        std::thread::sleep(std::time::Duration::from_nanos(dt));
                    }
                    self.warp_ns += dt;
                }
                self.dev.tick(self.now_ns());
            }
            None => {
                if self.dev.poll_console() {
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        if self.dev.reset_requested {
            return Some(ExitReason::Reset);
        }
        None
    }

    fn step(&mut self) -> R<()> {
        let insn = self.fetch_decode()?;
        if let Some(t) = &mut self.trace {
            if t.len() >= 64 {
                t.pop_front();
            }
            t.push_back(self.rip);
        }
        self.exec(&insn)
    }

    pub fn dump_regs(&self) -> String {
        let mut s = format!("rip={:016x} cpl={} eflags={:x}\n", self.rip, self.cpl, self.eflags_full());
        for (i, g) in self.gpr.iter().enumerate() {
            s += &format!("r{i:<2}={g:016x} ");
            if i % 4 == 3 {
                s += "\n";
            }
        }
        s
    }

    pub fn dump(&self) -> String {
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
            if i % 4 == 3 {
                s += "\n";
            }
        }
        let needle = b"Linux version";
        let mut found = 0;
        let mut at = 0;
        while found < 6 && at + needle.len() < self.ram.len() {
            let Some(p) = self.ram[at..]
                .windows(needle.len())
                .position(|w| w == needle)
            else {
                break;
            };
            let st = at + p;
            let end = (st + 6000).min(self.ram.len());
            let mut txt = String::new();
            let mut run = String::new();
            for &b in &self.ram[st..end] {
                if (0x20..0x7f).contains(&b) {
                    run.push(b as char);
                } else {
                    if run.len() >= 6 {
                        txt += &run;
                        txt.push('\n');
                    }
                    run.clear();
                }
            }
            s += &format!("--- ram match at {st:#x}\n{txt}\n");
            found += 1;
            at = st + needle.len();
        }
        if let Some(t) = &self.trace {
            s += "last rips:";
            for r in t.iter().rev().take(16) {
                s += &format!(" {r:x}");
            }
            s += "\n";
        }
        s
    }
}
