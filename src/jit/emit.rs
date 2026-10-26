use std::mem::{offset_of, size_of};

use super::a64::*;
use super::helpers::*;
use crate::cpu::decode::{Insn, NO_REG, NO_SEG};
use crate::cpu::flags::Cc;
use crate::cpu::{Cpu, Seg, TLB_BITS, TLB_N};

const GPR: u32 = offset_of!(Cpu, gpr) as u32;
const RIP: u32 = offset_of!(Cpu, rip) as u32;
const ICOUNT: u32 = offset_of!(Cpu, icount) as u32;
const EXIT_BLOCK: u32 = offset_of!(Cpu, exit_block) as u32;
const PREPAID: u32 = offset_of!(Cpu, prepaid) as u32;
const CC_OFF: u32 = offset_of!(Cpu, cc) as u32;
const CC_DST: u32 = offset_of!(Cpu, cc_dst) as u32;
const CC_SRC: u32 = offset_of!(Cpu, cc_src) as u32;
const SEG_BASE: u32 = (offset_of!(Cpu, seg) + offset_of!(Seg, base)) as u32;
const SEG_SIZE: u32 = size_of::<Seg>() as u32;

const CPU: u32 = 19;
const TLB: u32 = 20;
const SAVE_LA: u32 = 22;
const SAVE_NZCV: u32 = 23;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Add,
    Sub,
    Logic,
}

#[derive(Clone, Copy)]
struct Live {
    kind: Kind,
}

enum SlowKind {
    Ld { dst: u32, size: u8, signed: bool },
    St { size: u8 },
}

struct Slow {
    slow: Label,
    resume: Label,
    kind: SlowKind,
    k: u32,
    rip: u64,
}

#[derive(Clone, Copy)]
enum Dst {
    Reg(u8),
    Mem,
}

#[derive(Clone, Copy)]
enum Src {
    Reg(u8),
    Mem,
    Imm(u64),
}

enum CondMap {
    Cond(u32),
    Always,
    Never,
}

fn cond_map(kind: Kind, cc: u8) -> Option<CondMap> {
    use CondMap::*;
    let c = |x| Some(Cond(x));
    let idx = cc as usize & 15;
    match kind {
        Kind::Sub => [
            c(VS),
            c(VC),
            c(CC),
            c(CS),
            c(EQ),
            c(NE),
            c(LS),
            c(HI),
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][idx]
            .take(),
        Kind::Add => [
            c(VS),
            c(VC),
            c(CS),
            c(CC),
            c(EQ),
            c(NE),
            None,
            None,
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][idx]
            .take(),
        Kind::Logic => [
            Some(Never),
            Some(Always),
            Some(Never),
            Some(Always),
            c(EQ),
            c(NE),
            c(EQ),
            c(NE),
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][idx]
            .take(),
    }
}

pub(crate) struct Addrs {
    pub exec: u64,
    pub ld: u64,
    pub st: u64,
    pub cond: u64,
    pub exit_ok: u64,
    pub exit_fail: u64,
}

pub(crate) struct Gen<'a> {
    pub a: Asm,
    ad: &'a Addrs,
    insns: &'a [Insn],
    slows: Vec<Slow>,
    smc: Vec<(Label, u32, u64)>,
    live: Option<Live>,
    pub inlined: u32,
}

pub(crate) fn is_term(i: &Insn) -> bool {
    match i.op {
        0x70..=0x7f
        | 0xe0..=0xe3
        | 0xe8
        | 0xe9
        | 0xeb
        | 0xc2
        | 0xc3
        | 0xca
        | 0xcb
        | 0xcc
        | 0xcd
        | 0xcf
        | 0xf4 => true,
        0x6c..=0x6f
        | 0xa4..=0xa7
        | 0xaa..=0xaf
        | 0xe4..=0xe7
        | 0xec..=0xef
        | 0xfa
        | 0xfb
        | 0x9d
        | 0x8e
        | 0xc8 => true,
        0xff => matches!(i.reg & 7, 2..=5),
        0x105..=0x107
        | 0x10b
        | 0x180..=0x18f
        | 0x1a1
        | 0x1a9
        | 0x120..=0x123
        | 0x130
        | 0x100
        | 0x101
        | 0xffff => true,
        _ => false,
    }
}

impl<'a> Gen<'a> {
    pub fn new(base: u64, ad: &'a Addrs, insns: &'a [Insn]) -> Self {
        Gen {
            a: Asm::new(base),
            ad,
            insns,
            slows: Vec::new(),
            smc: Vec::new(),
            live: None,
            inlined: 0,
        }
    }

    fn ld_gpr(&mut self, rd: u32, r: u8) {
        self.a.ldr(rd, CPU, GPR + 8 * r as u32);
    }

    fn st_gpr(&mut self, r: u8, rs: u32) {
        self.a.str(rs, CPU, GPR + 8 * r as u32);
    }

    fn call(&mut self, addr: u64) {
        self.a.mov_imm(16, addr);
        self.a.blr(16);
    }

    fn set_rip(&mut self, rip: u64) {
        self.a.mov_imm(1, rip);
        self.a.str(1, CPU, RIP);
    }

    fn exit_ok(&mut self, n: u32) {
        self.a.mov_imm(0, (n as u64) << 8);
        self.a.b_abs(self.ad.exit_ok);
    }

    fn fail_exit(&mut self, k: u32) {
        self.a.mov_imm(1, k as u64);
        self.a.b_abs(self.ad.exit_fail);
    }

    fn helper(&mut self, k: usize, rip: u64, terminal: bool) {
        let insn = &self.insns[k] as *const Insn as u64;
        if k > 0 && matches!(self.insns[k].op, 0x131 | 0xe4..=0xe7 | 0xec..=0xef) {
            self.a.ldr(4, CPU, ICOUNT);
            self.a.add_imm(4, 4, k as u32, true);
            self.a.str(4, CPU, ICOUNT);
            self.a.mov_imm(4, k as u64);
            self.a.str(4, CPU, PREPAID);
        }
        self.a.mov(0, CPU);
        self.a.mov_imm(1, insn);
        self.a.mov_imm(2, rip);
        self.call(self.ad.exec);
        let ok = self.a.label();
        self.a.cbz(0, ok, true);
        self.fail_exit(k as u32);
        self.a.bind(ok);
        if terminal {
            self.exit_ok(k as u32 + 1);
        }
        self.live = None;
    }

    fn ea(&mut self, i: &Insn, next: u64, rd: u32) {
        if i.rip_rel {
            self.a.mov_imm(rd, next.wrapping_add(i.disp as u64));
        } else {
            let mut have = false;
            if i.base != NO_REG {
                self.ld_gpr(rd, i.base);
                have = true;
            }
            if i.index != NO_REG {
                if have {
                    self.ld_gpr(3, i.index);
                    self.a.add_lsl(rd, rd, 3, i.scale as u32);
                } else {
                    self.ld_gpr(rd, i.index);
                    if i.scale != 0 {
                        self.a.lsl_imm(rd, rd, i.scale as u32);
                    }
                    have = true;
                }
            }
            if !have {
                self.a.mov_imm(rd, i.disp as u64);
            } else if i.disp > 0 && i.disp < 4096 {
                self.a.add_imm(rd, rd, i.disp as u32, true);
            } else if i.disp < 0 && i.disp > -4096 {
                self.a.sub_imm(rd, rd, (-i.disp) as u32, true);
            } else if i.disp != 0 {
                self.a.mov_imm(3, i.disp as u64);
                self.a.add(rd, rd, 3, true);
            }
        }
        if i.seg == 4 || i.seg == 5 {
            self.a.ldr(3, CPU, SEG_BASE + SEG_SIZE * i.seg as u32);
            self.a.add(rd, rd, 3, true);
        }
    }

    fn tlb_probe(&mut self, size: u8, tag_off: u32, slow: Label) {
        self.a.lsr_imm(3, 2, 12);
        self.a.and_imm(3, 3, (TLB_N - 1) as u64, true);
        self.a.add_lsl(3, TLB, 3, 5);
        self.a.ldr(4, 3, tag_off);
        if size == 1 {
            self.a.and_imm(5, 2, !0xfff, true);
        } else {
            self.a.add_imm(5, 2, size as u32 - 1, true);
            self.a.and_imm(5, 5, !0xfff, true);
        }
        self.a.eor(5, 4, 5, true);
        self.a.cbnz(5, slow, true);
        let _ = TLB_BITS;
    }

    fn mem_load(&mut self, dst: u32, size: u8, signed: bool, k: u32, rip: u64) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, 0, slow);
        self.a.ldr(4, 3, 24);
        self.a.ldr_r(dst, 2, 4, size, signed);
        self.a.bind(resume);
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Ld { dst, size, signed },
            k,
            rip,
        });
    }

    fn mem_store(&mut self, val: u32, size: u8, k: u32, rip: u64) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, 8, slow);
        self.a.ldr(4, 3, 24);
        self.a.str_r(val, 2, 4, size);
        self.a.bind(resume);
        let _ = val;
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::St { size },
            k,
            rip,
        });
    }

    fn smc_check(&mut self, k: u32, next: u64) {
        let l = self.a.label();
        self.a.ldrb(4, CPU, EXIT_BLOCK);
        self.a.cbnz(4, l, false);
        self.smc.push((l, k, next));
    }

    fn emit_slows(&mut self) {
        let smc = std::mem::take(&mut self.smc);
        for (l, k, next) in smc {
            self.a.bind(l);
            self.set_rip(next);
            self.exit_ok(k + 1);
        }
        let slows = std::mem::take(&mut self.slows);
        for s in slows {
            self.a.bind(s.slow);
            self.a.mrs_nzcv(SAVE_NZCV);
            self.a.mov(SAVE_LA, 2);
            let fail = self.a.label();
            match s.kind {
                SlowKind::Ld { dst, size, signed } => {
                    self.a.mov(0, CPU);
                    self.a.mov(1, 2);
                    self.a.mov_imm(2, size as u64);
                    self.a.mov_imm(3, s.rip);
                    self.call(self.ad.ld as u64);
                    self.a.cbnz(1, fail, true);
                    if signed {
                        match size {
                            1 => self.a.sxtb(dst, 0, true),
                            2 => self.a.sxth(dst, 0, true),
                            4 => self.a.sxtw(dst, 0),
                            _ => self.a.mov(dst, 0),
                        }
                    } else {
                        self.a.mov(dst, 0);
                    }
                    self.a.msr_nzcv(SAVE_NZCV);
                    self.a.mov(2, SAVE_LA);
                    self.a.b(s.resume);
                    self.a.bind(fail);
                    self.a.mov(0, 1);
                    self.fail_exit(s.k);
                }
                SlowKind::St { size } => {
                    self.a.mov(3, 6);
                    self.a.mov(1, 2);
                    self.a.mov_imm(2, size as u64);
                    self.a.mov_imm(4, s.rip);
                    self.a.mov(0, CPU);
                    self.call(self.ad.st as u64);
                    self.a.cbnz(0, fail, true);
                    self.a.msr_nzcv(SAVE_NZCV);
                    self.a.mov(2, SAVE_LA);
                    self.a.b(s.resume);
                    self.a.bind(fail);
                    self.fail_exit(s.k);
                }
            }
        }
        self.a.finish();
    }

    fn store_cc(&mut self, kind: Kind, osz: u8) {
        let cc = match kind {
            Kind::Add => Cc::Add,
            Kind::Sub => Cc::Sub,
            Kind::Logic => Cc::Logic,
        } as u32;
        self.a.mov_imm(4, (cc | (osz as u32) << 8) as u64);
        self.a.strh(4, CPU, CC_OFF);
        self.a.str(0, CPU, CC_DST);
        self.a.str(1, CPU, CC_SRC);
        self.live = Some(Live { kind });
    }

    fn mem_ok(i: &Insn) -> bool {
        !i.addr32 && (i.seg == NO_SEG || i.seg <= 5) && !i.lock
    }

    #[allow(clippy::too_many_arguments)]
    fn alu(
        &mut self,
        i: &Insn,
        k: u8,
        dst: Dst,
        src: Src,
        osz: u8,
        next: u64,
        kidx: u32,
        rip: u64,
    ) {
        let is64 = osz == 8;
        match (dst, src) {
            (Dst::Mem, Src::Reg(s)) => {
                self.ea(i, next, 2);
                self.mem_load(0, osz, false, kidx, rip);
                self.ld_gpr(1, s);
            }
            (Dst::Mem, Src::Imm(v)) => {
                self.ea(i, next, 2);
                self.mem_load(0, osz, false, kidx, rip);
                self.a.mov_imm(1, v);
            }
            (Dst::Reg(d), Src::Mem) => {
                self.ea(i, next, 2);
                self.mem_load(1, osz, false, kidx, rip);
                self.ld_gpr(0, d);
            }
            (Dst::Reg(d), Src::Reg(s)) => {
                self.ld_gpr(0, d);
                self.ld_gpr(1, s);
            }
            (Dst::Reg(d), Src::Imm(v)) => {
                self.ld_gpr(0, d);
                self.a.mov_imm(1, v);
            }
            (Dst::Mem, Src::Mem) => unreachable!(),
        }
        let kind = match k {
            0 => {
                self.a.adds(0, 0, 1, is64);
                Kind::Add
            }
            1 => {
                self.a.orr(0, 0, 1, is64);
                self.a.ands(XZR, 0, 0, is64);
                Kind::Logic
            }
            4 | 8 => {
                self.a.ands(0, 0, 1, is64);
                Kind::Logic
            }
            5 | 7 => {
                self.a.subs(0, 0, 1, is64);
                Kind::Sub
            }
            _ => {
                self.a.eor(0, 0, 1, is64);
                self.a.ands(XZR, 0, 0, is64);
                Kind::Logic
            }
        };
        if k != 7 && k != 8 {
            if let Dst::Reg(d) = dst {
                self.st_gpr(d, 0);
            }
        }
        self.store_cc(kind, osz);
    }

    fn imm_for(osz: u8, imm: i64) -> u64 {
        if osz == 8 {
            imm as u64
        } else {
            imm as u32 as u64
        }
    }

    fn jcc(&mut self, cc: u8, target: u64, next: u64, n: u32) {
        let taken = self.a.label();
        match self.live.and_then(|l| cond_map(l.kind, cc)) {
            Some(CondMap::Cond(c)) => self.a.b_cond(c, taken),
            Some(CondMap::Always) => self.a.b(taken),
            Some(CondMap::Never) => {}
            None => {
                self.a.mov(0, CPU);
                self.a.mov_imm(1, cc as u64);
                self.call(self.ad.cond);
                self.a.cbnz(0, taken, true);
            }
        }
        self.set_rip(next);
        self.exit_ok(n);
        self.a.bind(taken);
        self.set_rip(target);
        self.exit_ok(n);
    }

    fn try_inline(&mut self, k: usize, rip: u64, next: u64) -> bool {
        let i = &self.insns[k];
        let i: Insn = *i;
        let kidx = k as u32;
        let n = kidx + 1;
        let osz = i.opsize;
        let std = !i.addr32 && !i.lock && !i.p66 && (osz == 4 || osz == 8);
        let mem = i.md != 3;
        let mem_ok = Self::mem_ok(&i);
        let rex = i.rex_present;
        let mut store = false;
        match i.op {
            0x00..=0x3f if i.op & 7 < 6 && std => {
                let kk = ((i.op >> 3) & 7) as u8;
                if kk == 2 || kk == 3 {
                    return false;
                }
                match i.op & 7 {
                    1 => {
                        if mem && (kk != 7 || !mem_ok) {
                            return false;
                        }
                        let dst = if mem { Dst::Mem } else { Dst::Reg(i.rm) };
                        self.alu(&i, kk, dst, Src::Reg(i.reg), osz, next, kidx, rip);
                    }
                    3 => {
                        if mem && !mem_ok {
                            return false;
                        }
                        let src = if mem { Src::Mem } else { Src::Reg(i.rm) };
                        self.alu(&i, kk, Dst::Reg(i.reg), src, osz, next, kidx, rip);
                    }
                    5 => {
                        let v = Self::imm_for(osz, i.imm);
                        self.alu(&i, kk, Dst::Reg(0), Src::Imm(v), osz, next, kidx, rip);
                    }
                    _ => return false,
                }
            }
            0x81 | 0x83 if std => {
                let kk = i.reg & 7;
                if kk == 2 || kk == 3 || (mem && (kk != 7 || !mem_ok)) {
                    return false;
                }
                let v = Self::imm_for(osz, i.imm);
                let dst = if mem { Dst::Mem } else { Dst::Reg(i.rm) };
                self.alu(&i, kk, dst, Src::Imm(v), osz, next, kidx, rip);
            }
            0x85 if std => {
                if mem && !mem_ok {
                    return false;
                }
                let dst = if mem { Dst::Mem } else { Dst::Reg(i.rm) };
                self.alu(&i, 8, dst, Src::Reg(i.reg), osz, next, kidx, rip);
            }
            0xa9 if std => {
                let v = Self::imm_for(osz, i.imm);
                self.alu(&i, 8, Dst::Reg(0), Src::Imm(v), osz, next, kidx, rip);
            }
            0xf7 if std && i.reg & 7 == 0 => {
                if mem && !mem_ok {
                    return false;
                }
                let v = Self::imm_for(osz, i.imm);
                let dst = if mem { Dst::Mem } else { Dst::Reg(i.rm) };
                self.alu(&i, 8, dst, Src::Imm(v), osz, next, kidx, rip);
            }
            0x89 if std => {
                if mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&i, next, 2);
                    self.ld_gpr(6, i.reg);
                    self.mem_store(6, osz, kidx, rip);
                    store = true;
                } else {
                    self.ld_gpr(6, i.reg);
                    if osz == 4 {
                        self.a.uxtw(6, 6);
                    }
                    self.st_gpr(i.rm, 6);
                }
            }
            0x8b if std => {
                if mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&i, next, 2);
                    self.mem_load(6, osz, false, kidx, rip);
                } else {
                    self.ld_gpr(6, i.rm);
                    if osz == 4 {
                        self.a.uxtw(6, 6);
                    }
                }
                self.st_gpr(i.reg, 6);
            }
            0x88 if mem && mem_ok && !i.p66 => {
                if !rex && (4..8).contains(&i.reg) {
                    self.ld_gpr(6, i.reg - 4);
                    self.a.lsr_imm(6, 6, 8);
                } else {
                    self.ld_gpr(6, i.reg);
                }
                self.ea(&i, next, 2);
                self.mem_store(6, 1, kidx, rip);
                store = true;
            }
            0xc7 if std && i.reg & 7 == 0 => {
                let v = Self::imm_for(osz, i.imm);
                if mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&i, next, 2);
                    self.a.mov_imm(6, v);
                    self.mem_store(6, osz, kidx, rip);
                    store = true;
                } else {
                    self.a.mov_imm(6, v);
                    self.st_gpr(i.rm, 6);
                }
            }
            0xb8..=0xbf if std => {
                let r = (i.op as u8 & 7) | ((i.rex & 1) << 3);
                self.a.mov_imm(6, Self::imm_for(osz, i.imm));
                self.st_gpr(r, 6);
            }
            0x8d if std && mem => {
                if i.seg == 4 || i.seg == 5 {
                    return false;
                }
                self.ea(&i, next, 2);
                if osz == 4 {
                    self.a.uxtw(2, 2);
                }
                self.st_gpr(i.reg, 2);
            }
            0x63 if osz == 8 && !i.addr32 && !i.lock && !i.p66 && (!mem || mem_ok) => {
                if mem {
                    self.ea(&i, next, 2);
                    self.mem_load(6, 4, true, kidx, rip);
                } else {
                    self.ld_gpr(6, i.rm);
                    self.a.sxtw(6, 6);
                }
                self.st_gpr(i.reg, 6);
            }
            0x1b6 | 0x1b7 | 0x1be | 0x1bf if std && (!mem || mem_ok) => {
                let ssz: u8 = if i.op & 1 == 0 { 1 } else { 2 };
                let signed = i.op >= 0x1b8;
                if mem {
                    self.ea(&i, next, 2);
                    self.mem_load(6, ssz, signed, kidx, rip);
                } else {
                    if ssz == 1 && !rex && (4..8).contains(&i.rm) {
                        return false;
                    }
                    self.ld_gpr(6, i.rm);
                    match (ssz, signed) {
                        (1, false) => self.a.uxtb(6, 6),
                        (2, false) => self.a.uxth(6, 6),
                        (1, true) => self.a.sxtb(6, 6, true),
                        _ => self.a.sxth(6, 6, true),
                    }
                }
                if osz == 4 {
                    self.a.uxtw(6, 6);
                }
                self.st_gpr(i.reg, 6);
            }
            0x50..=0x57 if !i.p66 => {
                let r = (i.op as u8 & 7) | ((i.rex & 1) << 3);
                self.ld_gpr(2, 4);
                self.a.sub_imm(2, 2, 8, true);
                self.ld_gpr(6, r);
                self.mem_store(6, 8, kidx, rip);
                self.st_gpr(4, 2);
                store = true;
            }
            0x58..=0x5f if !i.p66 => {
                let r = (i.op as u8 & 7) | ((i.rex & 1) << 3);
                self.ld_gpr(2, 4);
                self.mem_load(6, 8, false, kidx, rip);
                self.a.add_imm(2, 2, 8, true);
                self.st_gpr(4, 2);
                self.st_gpr(r, 6);
            }
            0x90 | 0x10d | 0x118..=0x11f => {
                if i.op == 0x90 && i.rex & 1 != 0 {
                    return false;
                }
            }
            0x70..=0x7f | 0x180..=0x18f => {
                let target = next.wrapping_add(i.imm as u64);
                self.jcc((i.op & 15) as u8, target, next, n);
                self.inlined += 1;
                return true;
            }
            0xe9 | 0xeb => {
                self.set_rip(next.wrapping_add(i.imm as u64));
                self.exit_ok(n);
                self.inlined += 1;
                return true;
            }
            0xe8 => {
                self.ld_gpr(2, 4);
                self.a.sub_imm(2, 2, 8, true);
                self.a.mov_imm(6, next);
                self.mem_store(6, 8, kidx, rip);
                self.st_gpr(4, 2);
                self.set_rip(next.wrapping_add(i.imm as u64));
                self.exit_ok(n);
                self.inlined += 1;
                return true;
            }
            0xc3 => {
                self.ld_gpr(2, 4);
                self.mem_load(6, 8, false, kidx, rip);
                self.a.add_imm(2, 2, 8, true);
                self.st_gpr(4, 2);
                self.a.str(6, CPU, RIP);
                self.exit_ok(n);
                self.inlined += 1;
                return true;
            }
            _ => return false,
        }
        if store {
            self.smc_check(kidx, next);
        }
        self.inlined += 1;
        true
    }

    pub fn compile(&mut self, rips: &[u64], end_rip: u64) {
        let n = self.insns.len();
        for k in 0..n {
            let rip = rips[k];
            let next = if k + 1 < n { rips[k + 1] } else { end_rip };
            let term = is_term(&self.insns[k]);
            if self.try_inline(k, rip, next) {
                if is_inline_term(&self.insns[k]) {
                    self.emit_slows();
                    return;
                }
                continue;
            }
            self.helper(k, rip, term);
            if term {
                self.emit_slows();
                return;
            }
        }
        self.set_rip(end_rip);
        self.exit_ok(n as u32);
        self.emit_slows();
    }
}

fn is_inline_term(i: &Insn) -> bool {
    matches!(i.op, 0x70..=0x7f | 0x180..=0x18f | 0xe9 | 0xeb | 0xe8 | 0xc3)
}

#[allow(dead_code)]
fn _use(_: Pair) {}
