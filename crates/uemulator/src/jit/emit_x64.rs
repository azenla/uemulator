//! x86-64 block generator. Mirrors `emit.rs`; the guest flags map straight onto host flags, so a
//! conditional jump right after an inlined ALU op tests them directly. Anything that clobbers host
//! flags (TLB probes, helper calls) ends that window.

use std::mem::{offset_of, size_of};

use super::helpers::*;
use super::x64::*;
use crate::cpu::decode::{Insn, NO_REG, NO_SEG};
use crate::cpu::flags::Cc;
use crate::cpu::{Cpu, Seg, TLB_N};

const GPR: i32 = offset_of!(Cpu, gpr) as i32;
const RIP: i32 = offset_of!(Cpu, rip) as i32;
const ICOUNT: i32 = offset_of!(Cpu, icount) as i32;
const EXIT_BLOCK: i32 = offset_of!(Cpu, exit_block) as i32;
const PREPAID: i32 = offset_of!(Cpu, prepaid) as i32;
const CC_OFF: i32 = offset_of!(Cpu, cc) as i32;
const CC_DST: i32 = offset_of!(Cpu, cc_dst) as i32;
const CC_SRC: i32 = offset_of!(Cpu, cc_src) as i32;
const SEG_BASE: i32 = (offset_of!(Cpu, seg) + offset_of!(Seg, base)) as i32;
const SEG_SIZE: i32 = size_of::<Seg>() as i32;

// Host registers; RBX, RBP, R12 and R13 survive helper calls.
const CPU: u8 = RBX;
const TLB: u8 = RBP;
const V0: u8 = RAX;
const V1: u8 = RCX;
const LA: u8 = RDX;
const T3: u8 = RSI;
const T4: u8 = RDI;
const T5: u8 = R8;
const VAL: u8 = R9;
const T10: u8 = R10;
const CALL: u8 = R11;
const SAVE_LA: u8 = R12;
const SAVE_VAL: u8 = R13;

/// Frame below the saved registers: 32 bytes of Win64 shadow space, then an 8-byte load result slot.
pub(crate) const FRAME: i32 = 56;
const OUT: i32 = 32;

enum SlowKind {
    Ld { dst: u8, size: u8, signed: bool },
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
    /// Host flags currently equal the guest flags (set by the last inlined ALU op).
    live: bool,
    pub inlined: u32,
}

impl<'a> Gen<'a> {
    pub fn new(base: u64, ad: &'a Addrs, insns: &'a [Insn]) -> Self {
        Gen {
            a: Asm::new(base),
            ad,
            insns,
            slows: Vec::new(),
            smc: Vec::new(),
            live: false,
            inlined: 0,
        }
    }

    fn ld_gpr(&mut self, rd: u8, r: u8) {
        self.a.ld(rd, mem(CPU, GPR + 8 * r as i32));
    }

    fn st_gpr(&mut self, r: u8, rs: u8) {
        self.a.st(mem(CPU, GPR + 8 * r as i32), rs, 8);
    }

    fn call(&mut self, addr: u64) {
        self.a.mov_imm(CALL, addr);
        self.a.call_reg(CALL);
    }

    fn set_rip(&mut self, rip: u64) {
        self.a.mov_imm(T10, rip);
        self.a.st(mem(CPU, RIP), T10, 8);
    }

    fn exit_ok(&mut self, n: u32) {
        self.a.mov_imm(V0, (n as u64) << 8);
        self.a.jmp_abs(self.ad.exit_ok);
    }

    fn fail_exit(&mut self, k: u32) {
        self.a.mov_imm(V1, k as u64);
        self.a.jmp_abs(self.ad.exit_fail);
    }

    fn helper(&mut self, k: usize, rip: u64, terminal: bool) {
        let insn = &self.insns[k] as *const Insn as u64;
        if k > 0 && matches!(self.insns[k].op, 0x131 | 0xe4..=0xe7 | 0xec..=0xef) {
            self.a.ld(T10, mem(CPU, ICOUNT));
            self.a.alu_ri(ADD, T10, k as i32, true);
            self.a.st(mem(CPU, ICOUNT), T10, 8);
            self.a.mov_imm(T10, k as u64);
            self.a.st(mem(CPU, PREPAID), T10, 8);
        }
        self.a.mov_rr(ARGS[0], CPU, true);
        self.a.mov_imm(ARGS[1], insn);
        self.a.mov_imm(ARGS[2], rip);
        self.call(self.ad.exec);
        let ok = self.a.label();
        self.a.test_rr(V0, V0, true);
        self.a.jcc(4, ok);
        self.fail_exit(k as u32);
        self.a.bind(ok);
        if terminal {
            self.exit_ok(k as u32 + 1);
        }
        self.live = false;
    }

    // Only lea and moves here: the effective address must not disturb live host flags.
    fn ea(&mut self, i: &Insn, next: u64) {
        let disp = i.disp as i32;
        if i.rip_rel {
            self.a.mov_imm(LA, next.wrapping_add(i.disp as u64));
        } else if i.index != NO_REG {
            if i.base != NO_REG {
                self.ld_gpr(LA, i.base);
            } else {
                self.a.mov_imm(LA, 0);
            }
            self.ld_gpr(T3, i.index);
            self.a.lea(LA, mem_idx(LA, T3, i.scale, disp));
        } else if i.base != NO_REG {
            self.ld_gpr(LA, i.base);
            if disp != 0 {
                self.a.lea(LA, mem(LA, disp));
            }
        } else {
            self.a.mov_imm(LA, i.disp as u64);
        }
        if i.seg == 4 || i.seg == 5 {
            self.a.ld(T3, mem(CPU, SEG_BASE + SEG_SIZE * i.seg as i32));
            self.a.lea(LA, mem_idx(LA, T3, 0, 0));
        }
    }

    fn tlb_probe(&mut self, size: u8, tag_off: i32, slow: Label) {
        self.live = false;
        self.a.mov_rr(T3, LA, true);
        self.a.shift_ri(SHR, T3, 12, true);
        self.a.alu_ri(AND, T3, (TLB_N - 1) as i32, true);
        self.a.shift_ri(SHL, T3, 5, true);
        self.a.lea(T3, mem_idx(TLB, T3, 0, 0));
        if size == 1 {
            self.a.mov_rr(T5, LA, true);
        } else {
            self.a.lea(T5, mem(LA, size as i32 - 1));
        }
        self.a.alu_ri(AND, T5, !0xfff, true);
        self.a.cmp_mr(mem(T3, tag_off), T5);
        self.a.jcc(NE, slow);
    }

    fn mem_load(&mut self, dst: u8, size: u8, signed: bool, k: u32, rip: u64) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, 0, slow);
        self.a.ld(T4, mem(T3, 24));
        self.a.ld_ext(dst, mem_idx(LA, T4, 0, 0), size, signed);
        self.a.bind(resume);
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Ld { dst, size, signed },
            k,
            rip,
        });
    }

    fn mem_store(&mut self, size: u8, k: u32, rip: u64) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, 8, slow);
        self.a.ld(T4, mem(T3, 24));
        self.a.st(mem_idx(LA, T4, 0, 0), VAL, size);
        self.a.bind(resume);
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
        self.a.cmp_m8_imm(mem(CPU, EXIT_BLOCK), 0);
        self.a.jcc(NE, l);
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
            self.a.mov_rr(SAVE_LA, LA, true);
            self.set_rip(s.rip);
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_rr(ARGS[1], SAVE_LA, true);
            let fail = self.a.label();
            match s.kind {
                SlowKind::Ld { dst, size, signed } => {
                    self.a.mov_imm(ARGS[2], size as u64);
                    self.a.lea(ARGS[3], mem(RSP, OUT));
                    self.call(self.ad.ld);
                    self.a.test_rr(V0, V0, true);
                    self.a.jcc(NE, fail);
                    self.a.ld_ext(dst, mem(RSP, OUT), size, signed);
                }
                SlowKind::St { size } => {
                    self.a.mov_rr(SAVE_VAL, VAL, true);
                    self.a.mov_imm(ARGS[2], size as u64);
                    self.a.mov_rr(ARGS[3], SAVE_VAL, true);
                    self.call(self.ad.st);
                    self.a.test_rr(V0, V0, true);
                    self.a.jcc(NE, fail);
                }
            }
            self.a.mov_rr(LA, SAVE_LA, true);
            self.a.jmp(s.resume);
            self.a.bind(fail);
            self.fail_exit(s.k);
        }
        self.a.finish();
    }

    fn store_cc(&mut self, cc: Cc, osz: u8) {
        self.a
            .st_imm16(mem(CPU, CC_OFF), (cc as u32 | (osz as u32) << 8) as u16);
        self.a.st(mem(CPU, CC_DST), V0, 8);
        self.a.st(mem(CPU, CC_SRC), V1, 8);
        self.live = true;
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
        let w = osz == 8;
        match (dst, src) {
            (Dst::Mem, Src::Reg(s)) => {
                self.ea(i, next);
                self.mem_load(V0, osz, false, kidx, rip);
                self.ld_gpr(V1, s);
            }
            (Dst::Mem, Src::Imm(v)) => {
                self.ea(i, next);
                self.mem_load(V0, osz, false, kidx, rip);
                self.a.mov_imm(V1, v);
            }
            (Dst::Reg(d), Src::Mem) => {
                self.ea(i, next);
                self.mem_load(V1, osz, false, kidx, rip);
                self.ld_gpr(V0, d);
            }
            (Dst::Reg(d), Src::Reg(s)) => {
                self.ld_gpr(V0, d);
                self.ld_gpr(V1, s);
            }
            (Dst::Reg(d), Src::Imm(v)) => {
                self.ld_gpr(V0, d);
                self.a.mov_imm(V1, v);
            }
            (Dst::Mem, Src::Mem) => unreachable!(),
        }
        // cmp and test run as sub and and on the scratch copy: the result feeds the lazy flags
        let (op, cc) = match k {
            0 => (ADD, Cc::Add),
            1 => (OR, Cc::Logic),
            4 | 8 => (AND, Cc::Logic),
            5 | 7 => (SUB, Cc::Sub),
            _ => (XOR, Cc::Logic),
        };
        self.a.alu_rr(op, V0, V1, w);
        if k != 7
            && k != 8
            && let Dst::Reg(d) = dst
        {
            self.st_gpr(d, V0);
        }
        self.store_cc(cc, osz);
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
        if self.live {
            self.a.jcc(cc & 15, taken);
        } else {
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_imm(ARGS[1], cc as u64);
            self.call(self.ad.cond);
            self.a.test_rr(V0, V0, true);
            self.a.jcc(NE, taken);
        }
        self.set_rip(next);
        self.exit_ok(n);
        self.a.bind(taken);
        self.set_rip(target);
        self.exit_ok(n);
    }

    fn try_inline(&mut self, k: usize, rip: u64, next: u64) -> bool {
        let i: Insn = self.insns[k];
        let kidx = k as u32;
        let n = kidx + 1;
        let osz = i.opsize;
        let std = !i.addr32 && !i.lock && !i.p66 && (osz == 4 || osz == 8);
        let is_mem = i.md != 3;
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
                        if is_mem && (kk != 7 || !mem_ok) {
                            return false;
                        }
                        let dst = if is_mem { Dst::Mem } else { Dst::Reg(i.rm) };
                        self.alu(&i, kk, dst, Src::Reg(i.reg), osz, next, kidx, rip);
                    }
                    3 => {
                        if is_mem && !mem_ok {
                            return false;
                        }
                        let src = if is_mem { Src::Mem } else { Src::Reg(i.rm) };
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
                if kk == 2 || kk == 3 || (is_mem && (kk != 7 || !mem_ok)) {
                    return false;
                }
                let v = Self::imm_for(osz, i.imm);
                let dst = if is_mem { Dst::Mem } else { Dst::Reg(i.rm) };
                self.alu(&i, kk, dst, Src::Imm(v), osz, next, kidx, rip);
            }
            0x85 if std => {
                if is_mem && !mem_ok {
                    return false;
                }
                let dst = if is_mem { Dst::Mem } else { Dst::Reg(i.rm) };
                self.alu(&i, 8, dst, Src::Reg(i.reg), osz, next, kidx, rip);
            }
            0xa9 if std => {
                let v = Self::imm_for(osz, i.imm);
                self.alu(&i, 8, Dst::Reg(0), Src::Imm(v), osz, next, kidx, rip);
            }
            0xf7 if std && i.reg & 7 == 0 => {
                if is_mem && !mem_ok {
                    return false;
                }
                let v = Self::imm_for(osz, i.imm);
                let dst = if is_mem { Dst::Mem } else { Dst::Reg(i.rm) };
                self.alu(&i, 8, dst, Src::Imm(v), osz, next, kidx, rip);
            }
            0x89 if std => {
                if is_mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&i, next);
                    self.ld_gpr(VAL, i.reg);
                    self.mem_store(osz, kidx, rip);
                    store = true;
                } else {
                    self.ld_gpr(VAL, i.reg);
                    if osz == 4 {
                        self.a.mov_rr(VAL, VAL, false);
                    }
                    self.st_gpr(i.rm, VAL);
                }
            }
            0x8b if std => {
                if is_mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&i, next);
                    self.mem_load(VAL, osz, false, kidx, rip);
                } else {
                    self.ld_gpr(VAL, i.rm);
                    if osz == 4 {
                        self.a.mov_rr(VAL, VAL, false);
                    }
                }
                self.st_gpr(i.reg, VAL);
            }
            0x88 if is_mem && mem_ok && !i.p66 => {
                if !rex && (4..8).contains(&i.reg) {
                    self.ld_gpr(VAL, i.reg - 4);
                    self.a.shift_ri(SHR, VAL, 8, true);
                } else {
                    self.ld_gpr(VAL, i.reg);
                }
                self.ea(&i, next);
                self.mem_store(1, kidx, rip);
                store = true;
            }
            0xc7 if std && i.reg & 7 == 0 => {
                let v = Self::imm_for(osz, i.imm);
                if is_mem {
                    if !mem_ok {
                        return false;
                    }
                    self.ea(&i, next);
                    self.a.mov_imm(VAL, v);
                    self.mem_store(osz, kidx, rip);
                    store = true;
                } else {
                    self.a.mov_imm(VAL, v);
                    self.st_gpr(i.rm, VAL);
                }
            }
            0xb8..=0xbf if std => {
                let r = (i.op as u8 & 7) | ((i.rex & 1) << 3);
                self.a.mov_imm(VAL, Self::imm_for(osz, i.imm));
                self.st_gpr(r, VAL);
            }
            0x8d if std && is_mem => {
                if i.seg == 4 || i.seg == 5 {
                    return false;
                }
                self.ea(&i, next);
                if osz == 4 {
                    self.a.mov_rr(LA, LA, false);
                }
                self.st_gpr(i.reg, LA);
            }
            0x63 if osz == 8 && !i.addr32 && !i.lock && !i.p66 && (!is_mem || mem_ok) => {
                if is_mem {
                    self.ea(&i, next);
                    self.mem_load(VAL, 4, true, kidx, rip);
                } else {
                    self.ld_gpr(VAL, i.rm);
                    self.a.ext_rr(VAL, VAL, 4, true);
                }
                self.st_gpr(i.reg, VAL);
            }
            0x1b6 | 0x1b7 | 0x1be | 0x1bf if std && (!is_mem || mem_ok) => {
                let ssz: u8 = if i.op & 1 == 0 { 1 } else { 2 };
                let signed = i.op >= 0x1b8;
                if is_mem {
                    self.ea(&i, next);
                    self.mem_load(VAL, ssz, signed, kidx, rip);
                } else {
                    if ssz == 1 && !rex && (4..8).contains(&i.rm) {
                        return false;
                    }
                    self.ld_gpr(VAL, i.rm);
                    self.a.ext_rr(VAL, VAL, ssz, signed);
                }
                if osz == 4 {
                    self.a.mov_rr(VAL, VAL, false);
                }
                self.st_gpr(i.reg, VAL);
            }
            0x50..=0x57 if !i.p66 => {
                let r = (i.op as u8 & 7) | ((i.rex & 1) << 3);
                self.ld_gpr(LA, 4);
                self.a.lea(LA, mem(LA, -8));
                self.ld_gpr(VAL, r);
                self.mem_store(8, kidx, rip);
                self.st_gpr(4, LA);
                store = true;
            }
            0x58..=0x5f if !i.p66 => {
                let r = (i.op as u8 & 7) | ((i.rex & 1) << 3);
                self.ld_gpr(LA, 4);
                self.mem_load(VAL, 8, false, kidx, rip);
                self.a.lea(LA, mem(LA, 8));
                self.st_gpr(4, LA);
                self.st_gpr(r, VAL);
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
                self.ld_gpr(LA, 4);
                self.a.lea(LA, mem(LA, -8));
                self.a.mov_imm(VAL, next);
                self.mem_store(8, kidx, rip);
                self.st_gpr(4, LA);
                self.set_rip(next.wrapping_add(i.imm as u64));
                self.exit_ok(n);
                self.inlined += 1;
                return true;
            }
            0xc3 => {
                self.ld_gpr(LA, 4);
                self.mem_load(VAL, 8, false, kidx, rip);
                self.a.lea(LA, mem(LA, 8));
                self.st_gpr(4, LA);
                self.a.st(mem(CPU, RIP), VAL, 8);
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
