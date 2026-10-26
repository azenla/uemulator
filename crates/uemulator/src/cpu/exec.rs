use super::decode::*;
use super::exc::Exc;
use super::flags::*;
use super::*;

#[derive(Clone, Copy)]
pub(crate) enum Loc {
    Reg(u8),
    Mem(u64),
}

impl Cpu {
    pub fn rd_reg(&self, r: u8, size: u8, rex: bool) -> u64 {
        let g = self.gpr[r as usize & 15];
        match size {
            1 => {
                if !rex && (4..8).contains(&r) {
                    (self.gpr[r as usize - 4] >> 8) & 0xff
                } else {
                    g & 0xff
                }
            }
            2 => g & 0xffff,
            4 => g & 0xffff_ffff,
            _ => g,
        }
    }

    pub fn wr_reg(&mut self, r: u8, size: u8, rex: bool, v: u64) {
        match size {
            1 => {
                if !rex && (4..8).contains(&r) {
                    let g = &mut self.gpr[r as usize - 4];
                    *g = (*g & !0xff00) | ((v & 0xff) << 8);
                } else {
                    let g = &mut self.gpr[r as usize];
                    *g = (*g & !0xff) | (v & 0xff);
                }
            }
            2 => {
                let g = &mut self.gpr[r as usize];
                *g = (*g & !0xffff) | (v & 0xffff);
            }
            4 => self.gpr[r as usize] = v & 0xffff_ffff,
            _ => self.gpr[r as usize] = v,
        }
    }

    pub fn ea(&self, i: &Insn, next: u64) -> u64 {
        let mut a = i.disp as u64;
        if i.rip_rel {
            a = a.wrapping_add(next);
        } else {
            if i.base != NO_REG {
                a = a.wrapping_add(self.gpr[i.base as usize]);
            }
            if i.index != NO_REG {
                a = a.wrapping_add(self.gpr[i.index as usize] << i.scale);
            }
        }
        if i.addr32 {
            a &= 0xffff_ffff;
        }
        a
    }

    pub fn lin(&self, i: &Insn, a: u64) -> u64 {
        match i.seg {
            4 => a.wrapping_add(self.seg[4].base),
            5 => a.wrapping_add(self.seg[5].base),
            _ => a,
        }
    }

    pub fn rm_loc(&self, i: &Insn, next: u64) -> Loc {
        if i.md == 3 {
            Loc::Reg(i.rm)
        } else {
            Loc::Mem(self.lin(i, self.ea(i, next)))
        }
    }

    pub fn ld(&mut self, loc: Loc, size: u8, i: &Insn) -> R<u64> {
        match loc {
            Loc::Reg(r) => Ok(self.rd_reg(r, size, i.rex_present)),
            Loc::Mem(a) => self.read(a, size),
        }
    }

    pub fn st(&mut self, loc: Loc, size: u8, i: &Insn, v: u64) -> R<()> {
        match loc {
            Loc::Reg(r) => {
                self.wr_reg(r, size, i.rex_present, v);
                Ok(())
            }
            Loc::Mem(a) => self.write(a, size, v),
        }
    }

    pub fn push(&mut self, size: u8, v: u64) -> R<()> {
        let sp = self.gpr[4].wrapping_sub(size as u64);
        self.write(sp, size, v)?;
        self.gpr[4] = sp;
        Ok(())
    }

    pub fn pop(&mut self, size: u8) -> R<u64> {
        let v = self.read(self.gpr[4], size)?;
        self.gpr[4] = self.gpr[4].wrapping_add(size as u64);
        Ok(v)
    }

    pub fn cond(&self, cc: u8) -> bool {
        let f = self.eflags_full();
        let sf = f & SF != 0;
        let of = f & OF != 0;
        let r = match cc >> 1 {
            0 => of,
            1 => f & CF != 0,
            2 => f & ZF != 0,
            3 => f & (CF | ZF) != 0,
            4 => sf,
            5 => f & PF != 0,
            6 => sf != of,
            _ => f & ZF != 0 || sf != of,
        };
        r ^ (cc & 1 == 1)
    }

    fn alu(&mut self, k: u8, a: u64, b: u64, size: u8) -> u64 {
        let m = mask(size);
        let (a, b) = (a & m, b & m);
        match k {
            0 => {
                let r = a.wrapping_add(b) & m;
                self.set_cc(Cc::Add, size, r, b, 0);
                r
            }
            1 => {
                let r = a | b;
                self.set_cc(Cc::Logic, size, r, 0, 0);
                r
            }
            2 => {
                let c = self.cf() as u64;
                let r = a.wrapping_add(b).wrapping_add(c) & m;
                self.set_cc(Cc::Adc, size, r, b, c);
                r
            }
            3 => {
                let c = self.cf() as u64;
                let r = a.wrapping_sub(b).wrapping_sub(c) & m;
                self.set_cc(Cc::Sbb, size, r, b, c);
                r
            }
            4 => {
                let r = a & b;
                self.set_cc(Cc::Logic, size, r, 0, 0);
                r
            }
            5 | 7 => {
                let r = a.wrapping_sub(b) & m;
                self.set_cc(Cc::Sub, size, r, b, 0);
                r
            }
            _ => {
                let r = a ^ b;
                self.set_cc(Cc::Logic, size, r, 0, 0);
                r
            }
        }
    }

    fn shift(&mut self, k: u8, a: u64, cnt: u64, size: u8) -> u64 {
        let bits = size as u32 * 8;
        let m = mask(size);
        let c = (cnt & if size == 8 { 0x3f } else { 0x1f }) as u32;
        if c == 0 {
            return a & m;
        }
        let a = a & m;
        let fl = self.eflags_full();
        let msb = |v: u64| (v >> (bits - 1)) & 1;
        let (r, cf, of, szp) = match k {
            0 => {
                let c2 = c % bits;
                let r = if c2 == 0 {
                    a
                } else {
                    ((a << c2) | (a >> (bits - c2))) & m
                };
                let cf = r & 1;
                (r, cf, msb(r) ^ cf, false)
            }
            1 => {
                let c2 = c % bits;
                let r = if c2 == 0 {
                    a
                } else {
                    ((a >> c2) | (a << (bits - c2))) & m
                };
                (r, msb(r), msb(r) ^ ((r >> (bits - 2)) & 1), false)
            }
            2 => {
                let mut r = a;
                let mut cf = fl & CF;
                for _ in 0..c % (bits + 1) {
                    let out = msb(r);
                    r = ((r << 1) | cf) & m;
                    cf = out;
                }
                (r, cf, msb(r) ^ cf, false)
            }
            3 => {
                let mut r = a;
                let mut cf = fl & CF;
                for _ in 0..c % (bits + 1) {
                    let out = r & 1;
                    r = (r >> 1) | (cf << (bits - 1));
                    cf = out;
                }
                (r, cf, msb(r) ^ ((r >> (bits - 2)) & 1), false)
            }
            4 | 6 => {
                let r = if c >= bits { 0 } else { (a << c) & m };
                let cf = if c <= bits { (a >> (bits - c)) & 1 } else { 0 };
                (r, cf, msb(r) ^ cf, true)
            }
            5 => {
                let r = if c >= bits { 0 } else { a >> c };
                let cf = if c <= bits { (a >> (c - 1)) & 1 } else { 0 };
                (r, cf, msb(a), true)
            }
            _ => {
                let sh = 64 - bits;
                let sa = ((a << sh) as i64) >> sh;
                let r = (sa >> c.min(63)) as u64 & m;
                let cf = ((sa >> (c - 1).min(63)) & 1) as u64;
                (r, cf, 0, true)
            }
        };
        let mut f = if szp {
            szp_flags(r, size)
        } else {
            fl & (PF | AF | ZF | SF)
        };
        if cf != 0 {
            f |= CF;
        }
        if of != 0 {
            f |= OF;
        }
        self.set_eflags_arith(f);
        r
    }

    fn mul_div(&mut self, i: &Insn, next: u64, size: u8) -> R<()> {
        let loc = self.rm_loc(i, next);
        let kind = i.reg & 7;
        let src = self.ld(loc, size, i)?;
        let bits = size as u32 * 8;
        let m = mask(size);
        let rax = self.gpr[0] & m;
        let rdx = self.gpr[2] & m;
        match kind {
            4 | 5 => {
                let (lo, hi, ovf) = if kind == 4 {
                    let p = rax as u128 * src as u128;
                    let lo = p as u64 & m;
                    let hi = (p >> bits) as u64 & m;
                    (lo, hi, hi != 0)
                } else {
                    let sh = 64 - bits;
                    let sx = |v: u64| ((v << sh) as i64) >> sh;
                    let p = sx(rax) as i128 * sx(src) as i128;
                    let lo = p as u64 & m;
                    let hi = (p >> bits) as u64 & m;
                    (lo, hi, p != sx(lo) as i128)
                };
                if size == 1 {
                    self.wr_reg(0, 2, true, lo | (hi << 8));
                } else {
                    self.wr_reg(0, size, true, lo);
                    self.wr_reg(2, size, true, hi);
                }
                let mut f = szp_flags(lo, size);
                if ovf {
                    f |= CF | OF;
                }
                self.set_eflags_arith(f);
            }
            _ => {
                if src == 0 {
                    return Err(Exc::de());
                }
                let (q, r);
                if kind == 6 {
                    let n = if size == 1 {
                        self.gpr[0] as u128 & 0xffff
                    } else {
                        (rdx as u128) << bits | rax as u128
                    };
                    let (qq, rr) = (n / src as u128, n % src as u128);
                    if qq > m as u128 {
                        return Err(Exc::de());
                    }
                    q = qq as u64;
                    r = rr as u64;
                } else {
                    let sh = 64 - bits;
                    let sx = |v: u64| ((v << sh) as i64) >> sh;
                    let n: i128 = if size == 1 {
                        (self.gpr[0] as u16 as i16) as i128
                    } else {
                        (((rdx as u128) << bits | rax as u128) << (128 - 2 * bits)) as i128
                            >> (128 - 2 * bits)
                    };
                    let d = sx(src) as i128;
                    let (qq, rr) = (n / d, n % d);
                    let lim = 1i128 << (bits - 1);
                    if qq >= lim || qq < -lim {
                        return Err(Exc::de());
                    }
                    q = qq as u64 & m;
                    r = rr as u64 & m;
                }
                if size == 1 {
                    self.wr_reg(0, 2, true, q | (r << 8));
                } else {
                    self.wr_reg(0, size, true, q);
                    self.wr_reg(2, size, true, r);
                }
            }
        }
        Ok(())
    }

    fn imul3(&mut self, a: u64, b: u64, size: u8) -> u64 {
        let bits = size as u32 * 8;
        let sh = 64 - bits;
        let sx = |v: u64| ((v << sh) as i64) >> sh;
        let p = sx(a) as i128 * sx(b) as i128;
        let r = p as u64 & mask(size);
        let mut f = szp_flags(r, size);
        if p != sx(r) as i128 {
            f |= CF | OF;
        }
        self.set_eflags_arith(f);
        r
    }

    fn bit_op(&mut self, i: &Insn, next: u64, kind: u8, imm: bool) -> R<()> {
        let size = i.opsize;
        let bits = size as i64 * 8;
        let off = if imm {
            i.imm as u64
        } else {
            self.rd_reg(i.reg, size, true)
        };
        let loc = if i.md == 3 {
            Loc::Reg(i.rm)
        } else {
            let mut a = self.ea(i, next);
            if !imm {
                let sh = 64 - bits as u32;
                let so = ((off << sh) as i64) >> sh;
                a = a.wrapping_add((so.div_euclid(bits) * size as i64) as u64);
            }
            Loc::Mem(self.lin(i, a))
        };
        let bit = if imm || i.md == 3 {
            (off as i64).rem_euclid(bits) as u32
        } else {
            let sh = 64 - bits as u32;
            (((off << sh) as i64) >> sh).rem_euclid(bits) as u32
        };
        let v = self.ld(loc, size, i)?;
        let old = (v >> bit) & 1;
        let nv = match kind {
            1 => v | (1 << bit),
            2 => v & !(1 << bit),
            3 => v ^ (1 << bit),
            _ => v,
        };
        if kind != 0 {
            self.st(loc, size, i, nv)?;
        }
        let f = self.eflags_full();
        self.set_eflags_arith((f & !CF) | old);
        Ok(())
    }

    fn string_op(&mut self, i: &Insn, next: u64) -> R<()> {
        let op = i.op as u8;
        let size = if op & 1 == 0 { 1 } else { i.opsize };
        let amask = if i.addr32 { 0xffff_ffff } else { u64::MAX };
        let step: u64 = if self.eflags & DF != 0 {
            (size as u64).wrapping_neg()
        } else {
            size as u64
        };
        let rep = i.rep != 0;
        let cmp = matches!(op, 0xa6 | 0xa7 | 0xae | 0xaf);
        let mut n = 0;
        loop {
            if rep && self.gpr[1] & amask == 0 {
                break;
            }
            let src = self.lin(i, self.gpr[6] & amask);
            let dst = self.gpr[7] & amask;
            match op {
                0xa4 | 0xa5 => {
                    let v = self.read(src, size)?;
                    self.write(dst, size, v)?;
                    self.gpr[6] = self.gpr[6].wrapping_add(step);
                    self.gpr[7] = self.gpr[7].wrapping_add(step);
                }
                0xaa | 0xab => {
                    let v = self.rd_reg(0, size, true);
                    self.write(dst, size, v)?;
                    self.gpr[7] = self.gpr[7].wrapping_add(step);
                }
                0xac | 0xad => {
                    let v = self.read(src, size)?;
                    self.wr_reg(0, size, true, v);
                    self.gpr[6] = self.gpr[6].wrapping_add(step);
                }
                0xa6 | 0xa7 => {
                    let a = self.read(src, size)?;
                    let b = self.read(dst, size)?;
                    self.set_cc(Cc::Sub, size, a.wrapping_sub(b) & mask(size), b, 0);
                    self.gpr[6] = self.gpr[6].wrapping_add(step);
                    self.gpr[7] = self.gpr[7].wrapping_add(step);
                }
                _ => {
                    let a = self.rd_reg(0, size, true);
                    let b = self.read(dst, size)?;
                    self.set_cc(Cc::Sub, size, a.wrapping_sub(b) & mask(size), b, 0);
                    self.gpr[7] = self.gpr[7].wrapping_add(step);
                }
            }
            if i.addr32 {
                self.gpr[6] &= 0xffff_ffff;
                self.gpr[7] &= 0xffff_ffff;
            }
            if !rep {
                break;
            }
            let c = self.gpr[1].wrapping_sub(1);
            self.gpr[1] = if i.addr32 { c & 0xffff_ffff } else { c };
            n += 1;
            if cmp {
                let zf = self.eflags_full() & ZF != 0;
                if (i.rep == 1 && !zf) || (i.rep == 2 && zf) {
                    break;
                }
            }
            if n >= 4096 && self.gpr[1] & amask != 0 {
                self.icount += n;
                return Ok(());
            }
        }
        self.icount += n;
        self.rip = next;
        Ok(())
    }

    pub fn load_data_seg(&mut self, idx: usize, sel: u16) -> R<()> {
        if sel & 0xfffc == 0 {
            if idx == 2 && self.cpl == 3 {
                return Err(Exc::gp(0));
            }
            self.seg[idx] = Seg {
                sel,
                ..Default::default()
            };
            return Ok(());
        }
        let s = self.load_desc_seg(sel, false)?;
        if s.flags & (1 << 12) == 0 || (s.is_code() && s.flags & (1 << 9) == 0) {
            return Err(Exc::gp(sel as u32 & 0xfffc));
        }
        if !s.present() {
            return Err(Exc::fault(
                if idx == 2 { 12 } else { 11 },
                Some(sel as u32 & 0xfffc),
            ));
        }
        self.seg[idx] = s;
        Ok(())
    }

    fn do_iret(&mut self, i: &Insn) -> R<()> {
        if i.opsize != 8 {
            return Err(Exc::ud());
        }
        let sp = self.gpr[4];
        let rip = self.read(sp, 8)?;
        let cs = self.read(sp + 8, 8)? as u16;
        let rfl = self.read(sp + 16, 8)?;
        let rsp = self.read(sp + 24, 8)?;
        let ss = self.read(sp + 32, 8)? as u16;
        let ncs = self.load_desc_seg(cs, true)?;
        if !ncs.is_code() || !ncs.long() {
            return Err(Exc::gp(cs as u32 & 0xfffc));
        }
        let ncpl = (cs & 3) as u8;
        if ncpl < self.cpl {
            return Err(Exc::gp(cs as u32 & 0xfffc));
        }
        let old_cpl = self.cpl;
        self.seg[1] = ncs;
        self.seg[1].sel = cs;
        self.set_cpl(ncpl);
        if let Err(e) = self.load_data_seg(2, ss) {
            self.set_cpl(old_cpl);
            return Err(e);
        }
        let mut m = CF | PF | AF | ZF | SF | TF | DF | OF | NT | AC | (1 << 21);
        if old_cpl <= ((self.eflags_full() >> 12) & 3) as u8 {
            m |= IF;
        }
        if old_cpl == 0 {
            m |= IOPL;
        }
        let cur = self.eflags_full();
        self.eflags = ((cur & !m) | (rfl & m)) | 2;
        self.cc = Cc::Eflags;
        self.gpr[4] = rsp;
        self.rip = rip;
        if ncpl != 0 {
            for k in [0usize, 3, 4, 5] {
                let s = self.seg[k];
                if s.sel & 0xfffc != 0 && s.dpl() < ncpl && s.flags & (1 << 11) == 0 {
                    self.seg[k] = Seg::default();
                }
            }
        }
        Ok(())
    }

    fn do_retf(&mut self, i: &Insn, imm: u64) -> R<()> {
        if i.opsize != 8 {
            return Err(Exc::ud());
        }
        let sp = self.gpr[4];
        let rip = self.read(sp, 8)?;
        let cs = self.read(sp + 8, 8)? as u16;
        let ncs = self.load_desc_seg(cs, true)?;
        if !ncs.is_code() || !ncs.long() {
            return Err(Exc::gp(cs as u32 & 0xfffc));
        }
        let ncpl = (cs & 3) as u8;
        if ncpl < self.cpl {
            return Err(Exc::gp(cs as u32 & 0xfffc));
        }
        if ncpl != self.cpl {
            let rsp = self.read(sp + 16 + imm, 8)?;
            let ss = self.read(sp + 24 + imm, 8)? as u16;
            self.seg[1] = ncs;
            self.set_cpl(ncpl);
            self.load_data_seg(2, ss)?;
            self.gpr[4] = rsp;
        } else {
            self.seg[1] = ncs;
            self.gpr[4] = sp.wrapping_add(16 + imm);
        }
        self.rip = rip;
        Ok(())
    }

    pub fn exec(&mut self, i: &Insn) -> R<()> {
        let next = self.rip.wrapping_add(i.len as u64);
        let osz = i.opsize;
        let op = i.op;
        let rex = i.rex_present;
        match op {
            0x00..=0x3f if op & 7 < 6 => {
                let k = ((op >> 3) & 7) as u8;
                let form = op & 7;
                let size = if form & 1 == 0 { 1 } else { osz };
                match form {
                    0 | 1 => {
                        let loc = self.rm_loc(i, next);
                        let a = self.ld(loc, size, i)?;
                        let b = self.rd_reg(i.reg, size, rex);
                        let r = self.alu(k, a, b, size);
                        if k != 7 {
                            self.st(loc, size, i, r)?;
                        }
                    }
                    2 | 3 => {
                        let loc = self.rm_loc(i, next);
                        let b = self.ld(loc, size, i)?;
                        let a = self.rd_reg(i.reg, size, rex);
                        let r = self.alu(k, a, b, size);
                        if k != 7 {
                            self.wr_reg(i.reg, size, rex, r);
                        }
                    }
                    _ => {
                        let a = self.rd_reg(0, size, true);
                        let r = self.alu(k, a, i.imm as u64, size);
                        if k != 7 {
                            self.wr_reg(0, size, true, r);
                        }
                    }
                }
            }
            0x50..=0x57 => {
                let r = (op as u8 & 7) | ((i.rex & 1) << 3);
                let sz = if i.p66 { 2 } else { 8 };
                let v = self.gpr[r as usize];
                self.push(sz, v)?;
            }
            0x58..=0x5f => {
                let r = (op as u8 & 7) | ((i.rex & 1) << 3);
                let sz = if i.p66 { 2 } else { 8 };
                let v = self.pop(sz)?;
                self.wr_reg(r, sz, true, v);
            }
            0x63 => {
                let loc = self.rm_loc(i, next);
                let v = self.ld(loc, 4, i)?;
                if osz == 8 {
                    self.gpr[i.reg as usize] = v as u32 as i32 as i64 as u64;
                } else {
                    self.wr_reg(i.reg, osz, true, v);
                }
            }
            0x68 | 0x6a => {
                let sz = if i.p66 { 2 } else { 8 };
                self.push(sz, i.imm as u64)?;
            }
            0x69 | 0x6b => {
                let loc = self.rm_loc(i, next);
                let a = self.ld(loc, osz, i)?;
                let r = self.imul3(a, i.imm as u64, osz);
                self.wr_reg(i.reg, osz, true, r);
            }
            0x70..=0x7f => {
                if self.cond((op & 15) as u8) {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            0x80 | 0x81 | 0x83 => {
                let size = if op == 0x80 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let a = self.ld(loc, size, i)?;
                let k = i.reg & 7;
                let r = self.alu(k, a, i.imm as u64, size);
                if k != 7 {
                    self.st(loc, size, i, r)?;
                }
            }
            0x84 | 0x85 => {
                let size = if op == 0x84 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let a = self.ld(loc, size, i)?;
                let b = self.rd_reg(i.reg, size, rex);
                self.set_cc(Cc::Logic, size, a & b, 0, 0);
            }
            0x86 | 0x87 => {
                let size = if op == 0x86 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let a = self.ld(loc, size, i)?;
                let b = self.rd_reg(i.reg, size, rex);
                self.st(loc, size, i, b)?;
                self.wr_reg(i.reg, size, rex, a);
            }
            0x88 | 0x89 => {
                let size = if op == 0x88 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let v = self.rd_reg(i.reg, size, rex);
                self.st(loc, size, i, v)?;
            }
            0x8a | 0x8b => {
                let size = if op == 0x8a { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let v = self.ld(loc, size, i)?;
                self.wr_reg(i.reg, size, rex, v);
            }
            0x8c => {
                if i.reg > 5 {
                    return Err(Exc::ud());
                }
                let v = self.seg[i.reg as usize].sel as u64;
                let loc = self.rm_loc(i, next);
                let size = if i.md == 3 { osz } else { 2 };
                self.st(loc, size, i, v)?;
            }
            0x8d => {
                if i.md == 3 {
                    return Err(Exc::ud());
                }
                let a = self.ea(i, next);
                self.wr_reg(i.reg, osz, true, a);
            }
            0x8e => {
                let idx = (i.reg & 7) as usize;
                if idx == 1 || idx > 5 {
                    return Err(Exc::ud());
                }
                let loc = self.rm_loc(i, next);
                let sel = self.ld(loc, 2, i)? as u16;
                self.load_data_seg(idx, sel)?;
                if idx == 2 {
                    self.shadow = true;
                }
            }
            0x8f => {
                let sz = if i.p66 { 2 } else { 8 };
                let loc = self.rm_loc(i, next);
                let sp = self.gpr[4];
                let v = self.read(sp, sz)?;
                self.gpr[4] = sp.wrapping_add(sz as u64);
                let loc = match loc {
                    Loc::Mem(a) if i.base == 4 => Loc::Mem(a.wrapping_add(sz as u64)),
                    l => l,
                };
                if let Err(e) = self.st(loc, sz, i, v) {
                    self.gpr[4] = sp;
                    return Err(e);
                }
            }
            0x90 => {
                if i.rex & 1 != 0 {
                    let r = 8;
                    let a = self.rd_reg(0, osz, true);
                    let b = self.rd_reg(r, osz, true);
                    self.wr_reg(0, osz, true, b);
                    self.wr_reg(r, osz, true, a);
                }
            }
            0x91..=0x97 => {
                let r = (op as u8 & 7) | ((i.rex & 1) << 3);
                let a = self.rd_reg(0, osz, true);
                let b = self.rd_reg(r, osz, true);
                self.wr_reg(0, osz, true, b);
                self.wr_reg(r, osz, true, a);
            }
            0x98 => match osz {
                8 => self.gpr[0] = self.gpr[0] as u32 as i32 as i64 as u64,
                4 => self.gpr[0] = self.gpr[0] as u16 as i16 as i32 as u32 as u64,
                _ => {
                    let v = self.gpr[0] as u8 as i8 as i16 as u16 as u64;
                    self.wr_reg(0, 2, true, v)
                }
            },
            0x99 => match osz {
                8 => self.gpr[2] = ((self.gpr[0] as i64) >> 63) as u64,
                4 => self.gpr[2] = (((self.gpr[0] as u32 as i32) >> 31) as u32) as u64,
                _ => {
                    let v = (((self.gpr[0] as u16 as i16) >> 15) as u16) as u64;
                    self.wr_reg(2, 2, true, v)
                }
            },
            0x9b => {
                if self.cr0 & 0xa == 0xa {
                    return Err(Exc::nm());
                }
            }
            0x9c => {
                let sz = if i.p66 { 2 } else { 8 };
                let f = self.eflags_full() & !(RF | VM);
                self.push(sz, f)?;
            }
            0x9d => {
                let sz = if i.p66 { 2 } else { 8 };
                let v = self.pop(sz)?;
                let iopl = ((self.eflags_full() >> 12) & 3) as u8;
                let mut m = CF | PF | AF | ZF | SF | TF | DF | OF | NT | AC | (1 << 21);
                if self.cpl <= iopl {
                    m |= IF;
                }
                if self.cpl == 0 {
                    m |= IOPL;
                }
                if sz == 2 {
                    m &= 0xffff;
                }
                let cur = self.eflags_full();
                self.eflags = ((cur & !m) | (v & m)) | 2;
                self.cc = Cc::Eflags;
            }
            0x9e => {
                let ah = (self.gpr[0] >> 8) & 0xd5;
                let f = self.eflags_full();
                self.set_eflags_arith((f & !0xd5) | ah);
            }
            0x9f => {
                let f = self.eflags_full() & 0xd5 | 2;
                self.wr_reg(4, 1, false, f);
            }
            0xa0..=0xa3 => {
                let size = if op & 1 == 0 { 1 } else { osz };
                let a = self.lin(
                    i,
                    if i.addr32 {
                        i.imm as u64 & 0xffff_ffff
                    } else {
                        i.imm as u64
                    },
                );
                if op < 0xa2 {
                    let v = self.read(a, size)?;
                    self.wr_reg(0, size, true, v);
                } else {
                    let v = self.rd_reg(0, size, true);
                    self.write(a, size, v)?;
                }
            }
            0xa4..=0xa7 | 0xaa..=0xaf => return self.string_op(i, next),
            0xa8 | 0xa9 => {
                let size = if op == 0xa8 { 1 } else { osz };
                let a = self.rd_reg(0, size, true);
                self.set_cc(Cc::Logic, size, a & i.imm as u64 & mask(size), 0, 0);
            }
            0xb0..=0xb7 => {
                let r = (op as u8 & 7) | ((i.rex & 1) << 3);
                self.wr_reg(r, 1, rex, i.imm as u64);
            }
            0xb8..=0xbf => {
                let r = (op as u8 & 7) | ((i.rex & 1) << 3);
                self.wr_reg(r, osz, true, i.imm as u64);
            }
            0xc0 | 0xc1 | 0xd0..=0xd3 => {
                let size = if op & 1 == 0 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let cnt = match op {
                    0xc0 | 0xc1 => i.imm as u64,
                    0xd0 | 0xd1 => 1,
                    _ => self.gpr[1] & 0xff,
                };
                let a = self.ld(loc, size, i)?;
                let r = self.shift(i.reg & 7, a, cnt, size);
                self.st(loc, size, i, r)?;
            }
            0xc2 | 0xc3 => {
                let t = self.read(self.gpr[4], 8)?;
                let extra = if op == 0xc2 { i.imm as u64 } else { 0 };
                self.gpr[4] = self.gpr[4].wrapping_add(8 + extra);
                self.rip = t;
                return Ok(());
            }
            0xc6 | 0xc7 => {
                let size = if op == 0xc6 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                self.st(loc, size, i, i.imm as u64)?;
            }
            0xc8 => {
                if i.imm2 & 31 != 0 {
                    return Err(self.unimpl(i));
                }
                let bp = self.gpr[5];
                self.push(8, bp)?;
                self.gpr[5] = self.gpr[4];
                self.gpr[4] = self.gpr[4].wrapping_sub(i.imm as u64);
            }
            0xc9 => {
                let bp = self.gpr[5];
                let v = self.read(bp, 8)?;
                self.gpr[4] = bp.wrapping_add(8);
                self.gpr[5] = v;
            }
            0xca | 0xcb => {
                let imm = if op == 0xca { i.imm as u64 } else { 0 };
                return self.do_retf(i, imm);
            }
            0xcc => {
                self.rip = next;
                return Err(Exc::Fault {
                    vec: 3,
                    err: None,
                    soft: true,
                });
            }
            0xcd => {
                self.rip = next;
                return Err(Exc::Fault {
                    vec: i.imm as u8,
                    err: None,
                    soft: true,
                });
            }
            0xcf => return self.do_iret(i),
            0xd7 => {
                let a = self.gpr[3].wrapping_add(self.gpr[0] & 0xff);
                let v = self.read(self.lin(i, a), 1)?;
                self.wr_reg(0, 1, true, v);
            }
            0xd8..=0xdf => return self.x87(i, next),
            0xe0..=0xe3 => {
                let amask = if i.addr32 { 0xffff_ffff } else { u64::MAX };
                let take = if op == 0xe3 {
                    self.gpr[1] & amask == 0
                } else {
                    let c = self.gpr[1].wrapping_sub(1) & amask;
                    self.gpr[1] = c;
                    let zf = self.eflags_full() & ZF != 0;
                    c != 0
                        && match op {
                            0xe0 => !zf,
                            0xe1 => zf,
                            _ => true,
                        }
                };
                if take {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            0xe4..=0xe7 | 0xec..=0xef => return self.port_io(i, next),
            0xe8 => {
                self.push(8, next)?;
                self.rip = next.wrapping_add(i.imm as u64);
                return Ok(());
            }
            0xe9 | 0xeb => {
                self.rip = next.wrapping_add(i.imm as u64);
                return Ok(());
            }
            0xf4 => {
                if self.cpl != 0 {
                    return Err(Exc::gp(0));
                }
                self.halted = true;
            }
            0xf5 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f ^ CF);
            }
            0xf6 | 0xf7 => {
                let size = if op == 0xf6 { 1 } else { osz };
                match i.reg & 7 {
                    0 | 1 => {
                        let loc = self.rm_loc(i, next);
                        let a = self.ld(loc, size, i)?;
                        self.set_cc(Cc::Logic, size, a & i.imm as u64 & mask(size), 0, 0);
                    }
                    2 => {
                        let loc = self.rm_loc(i, next);
                        let a = self.ld(loc, size, i)?;
                        self.st(loc, size, i, !a)?;
                    }
                    3 => {
                        let loc = self.rm_loc(i, next);
                        let a = self.ld(loc, size, i)?;
                        let r = 0u64.wrapping_sub(a) & mask(size);
                        self.set_cc(Cc::Sub, size, r, a & mask(size), 0);
                        self.st(loc, size, i, r)?;
                    }
                    _ => self.mul_div(i, next, size)?,
                }
            }
            0xf8 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f & !CF);
            }
            0xf9 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f | CF);
            }
            0xfa | 0xfb => {
                let iopl = ((self.eflags >> 12) & 3) as u8;
                if self.cpl > iopl {
                    return Err(Exc::gp(0));
                }
                if op == 0xfa {
                    self.eflags &= !IF;
                } else {
                    if self.eflags & IF == 0 {
                        self.shadow = true;
                    }
                    self.eflags |= IF;
                }
            }
            0xfc => self.eflags &= !DF,
            0xfd => self.eflags |= DF,
            0xfe | 0xff => {
                let size = if op == 0xfe { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                match i.reg & 7 {
                    0 | 1 => {
                        let a = self.ld(loc, size, i)?;
                        let cf = self.cf() as u64;
                        let (r, cc) = if i.reg & 7 == 0 {
                            (a.wrapping_add(1) & mask(size), Cc::Inc)
                        } else {
                            (a.wrapping_sub(1) & mask(size), Cc::Dec)
                        };
                        self.set_cc(cc, size, r, cf, 0);
                        self.st(loc, size, i, r)?;
                    }
                    2 if op == 0xff => {
                        let t = self.ld(loc, 8, i)?;
                        self.push(8, next)?;
                        self.rip = t;
                        return Ok(());
                    }
                    4 if op == 0xff => {
                        self.rip = self.ld(loc, 8, i)?;
                        return Ok(());
                    }
                    6 if op == 0xff => {
                        let sz = if i.p66 { 2 } else { 8 };
                        let v = self.ld(loc, sz, i)?;
                        self.push(sz, v)?;
                    }
                    _ => return Err(self.unimpl(i)),
                }
            }
            _ => return self.exec_0f(i, next),
        }
        self.rip = next;
        Ok(())
    }

    fn port_io(&mut self, i: &Insn, next: u64) -> R<()> {
        let iopl = ((self.eflags >> 12) & 3) as u8;
        if self.cpl > iopl {
            return Err(Exc::gp(0));
        }
        let op = i.op;
        let size = if op & 1 == 0 {
            1
        } else if i.p66 {
            2
        } else {
            4
        };
        let port = if op < 0xe8 {
            i.imm as u16 & 0xff
        } else {
            self.gpr[2] as u16
        };
        let now = self.now_ns();
        if op & 2 == 0 {
            let v = self.dev.io_in(port, size, now);
            self.wr_reg(0, size, true, v as u64);
        } else {
            let v = self.rd_reg(0, size, true) as u32;
            self.dev.io_out(port, size, v, now);
            if self.dev.reset_requested {
                self.rip = next;
                return Err(Exc::Exit(ExitReason::Reset));
            }
        }
        self.rip = next;
        Ok(())
    }

    pub fn unimpl(&mut self, i: &Insn) -> Exc {
        let rip = self.rip;
        self.log_once(0x4000_0000 + i.op as u64 * 8 + (i.reg as u64 & 7), || {
            format!(
                "unimplemented opcode {:#05x} /{} at rip {:#x} (modrm {:#04x})",
                i.op,
                i.reg & 7,
                rip,
                i.modrm
            )
        });
        Exc::ud()
    }

    fn exec_0f(&mut self, i: &Insn, next: u64) -> R<()> {
        let op = i.op;
        let osz = i.opsize;
        let rex = i.rex_present;
        match op {
            0x100 | 0x101 => return self.sys_group(i, next),
            0x105 => return self.do_syscall(next),
            0x107 => return self.do_sysret(i),
            0x106 => {
                self.need_cpl0()?;
                self.cr0 &= !8;
            }
            0x108 | 0x109 => self.need_cpl0()?,
            0x10b => return Err(Exc::ud()),
            0x10d | 0x118..=0x11f => {}
            0x120..=0x123 => self.mov_cr_dr(i)?,
            0x130 => {
                self.need_cpl0()?;
                let v = (self.gpr[0] & 0xffff_ffff) | (self.gpr[2] << 32);
                self.wrmsr(self.gpr[1] as u32, v)?;
            }
            0x131 => {
                if self.cr4 & 4 != 0 && self.cpl != 0 {
                    return Err(Exc::gp(0));
                }
                let t = self.tsc();
                self.gpr[0] = t & 0xffff_ffff;
                self.gpr[2] = t >> 32;
            }
            0x132 => {
                self.need_cpl0()?;
                let v = self.rdmsr(self.gpr[1] as u32)?;
                self.gpr[0] = v & 0xffff_ffff;
                self.gpr[2] = v >> 32;
            }
            0x140..=0x14f => {
                let loc = self.rm_loc(i, next);
                let v = self.ld(loc, osz, i)?;
                if self.cond((i.op & 15) as u8) {
                    self.wr_reg(i.reg, osz, true, v);
                } else if osz == 4 {
                    self.gpr[i.reg as usize] &= 0xffff_ffff;
                }
            }
            0x180..=0x18f => {
                if self.cond((i.op & 15) as u8) {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            0x190..=0x19f => {
                let loc = self.rm_loc(i, next);
                let v = self.cond((i.op & 15) as u8) as u64;
                self.st(loc, 1, i, v)?;
            }
            0x1a0 | 0x1a8 => {
                let sel = self.seg[if op == 0x1a0 { 4 } else { 5 }].sel as u64;
                let sz = if i.p66 { 2 } else { 8 };
                self.push(sz, sel)?;
            }
            0x1a1 | 0x1a9 => {
                let sz = if i.p66 { 2 } else { 8 };
                let v = self.pop(sz)? as u16;
                if let Err(e) = self.load_data_seg(if op == 0x1a1 { 4 } else { 5 }, v) {
                    self.gpr[4] = self.gpr[4].wrapping_sub(sz as u64);
                    return Err(e);
                }
            }
            0x1a2 => self.cpuid(),
            0x1a3 => self.bit_op(i, next, 0, false)?,
            0x1ab => self.bit_op(i, next, 1, false)?,
            0x1b3 => self.bit_op(i, next, 2, false)?,
            0x1bb => self.bit_op(i, next, 3, false)?,
            0x1ba => {
                let k = i.reg & 7;
                if k < 4 {
                    return Err(Exc::ud());
                }
                self.bit_op(i, next, k - 4, true)?
            }
            0x1a4 | 0x1a5 | 0x1ac | 0x1ad => {
                let loc = self.rm_loc(i, next);
                let a = self.ld(loc, osz, i)?;
                let b = self.rd_reg(i.reg, osz, rex);
                let cnt = if op & 1 == 0 {
                    i.imm as u64
                } else {
                    self.gpr[1] & 0xff
                };
                let bits = osz as u32 * 8;
                let c = (cnt & if osz == 8 { 0x3f } else { 0x1f }) as u32;
                if c != 0 {
                    let m = mask(osz);
                    let (r, cf) = if op < 0x1a8 {
                        let r = ((a << c) | (b >> (bits - c))) & m;
                        (r, (a >> (bits - c)) & 1)
                    } else {
                        let r = ((a >> c) | (b << (bits - c))) & m;
                        (r, (a >> (c - 1)) & 1)
                    };
                    let of = ((r ^ a) >> (bits - 1)) & 1;
                    let mut f = szp_flags(r, osz);
                    if cf != 0 {
                        f |= CF;
                    }
                    if of != 0 {
                        f |= OF;
                    }
                    self.set_eflags_arith(f);
                    self.st(loc, osz, i, r)?;
                }
            }
            0x1ae => return self.group_ae(i, next),
            0x1af => {
                let loc = self.rm_loc(i, next);
                let b = self.ld(loc, osz, i)?;
                let a = self.rd_reg(i.reg, osz, true);
                let r = self.imul3(a, b, osz);
                self.wr_reg(i.reg, osz, true, r);
            }
            0x1b0 | 0x1b1 => {
                let size = if op == 0x1b0 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let d = self.ld(loc, size, i)?;
                let acc = self.rd_reg(0, size, true);
                let m = mask(size);
                self.set_cc(Cc::Sub, size, acc.wrapping_sub(d) & m, d, 0);
                if acc == d {
                    let s = self.rd_reg(i.reg, size, rex);
                    self.st(loc, size, i, s)?;
                } else {
                    self.st(loc, size, i, d)?;
                    self.wr_reg(0, size, true, d);
                }
            }
            0x1c0 | 0x1c1 => {
                let size = if op == 0x1c0 { 1 } else { osz };
                let loc = self.rm_loc(i, next);
                let d = self.ld(loc, size, i)?;
                let s = self.rd_reg(i.reg, size, rex);
                let r = self.alu(0, d, s, size);
                self.st(loc, size, i, r)?;
                self.wr_reg(i.reg, size, rex, d);
            }
            0x1c3 => {
                if i.md == 3 {
                    return Err(Exc::ud());
                }
                let loc = self.rm_loc(i, next);
                let v = self.rd_reg(i.reg, osz, true);
                self.st(loc, osz, i, v)?;
            }
            0x1c7 => {
                if i.md == 3 || i.reg & 7 != 1 || osz == 8 {
                    return Err(self.unimpl(i));
                }
                let a = self.lin(i, self.ea(i, next));
                let cur = self.read(a, 8)?;
                let cmp = (self.gpr[0] & 0xffff_ffff) | (self.gpr[2] << 32);
                let f = self.eflags_full();
                if cur == cmp {
                    let nv = (self.gpr[3] & 0xffff_ffff) | (self.gpr[1] << 32);
                    self.write(a, 8, nv)?;
                    self.set_eflags_arith(f | ZF);
                } else {
                    self.gpr[0] = cur & 0xffff_ffff;
                    self.gpr[2] = cur >> 32;
                    self.set_eflags_arith(f & !ZF);
                }
            }
            0x1c8..=0x1cf => {
                let r = (op as u8 & 7) | ((i.rex & 1) << 3);
                if osz == 8 {
                    self.gpr[r as usize] = self.gpr[r as usize].swap_bytes();
                } else {
                    self.gpr[r as usize] = (self.gpr[r as usize] as u32).swap_bytes() as u64;
                }
            }
            0x1b6 | 0x1b7 | 0x1be | 0x1bf => {
                let ssz = if op & 1 == 0 { 1 } else { 2 };
                let loc = self.rm_loc(i, next);
                let v = self.ld(loc, ssz, i)?;
                let v = if op < 0x1b8 {
                    v
                } else if ssz == 1 {
                    v as u8 as i8 as i64 as u64
                } else {
                    v as u16 as i16 as i64 as u64
                };
                self.wr_reg(i.reg, osz, true, v);
            }
            0x1bc | 0x1bd => {
                let loc = self.rm_loc(i, next);
                let v = self.ld(loc, osz, i)? & mask(osz);
                let f = self.eflags_full();
                if v == 0 {
                    self.set_eflags_arith(f | ZF);
                } else {
                    let idx = if op == 0x1bc {
                        v.trailing_zeros()
                    } else {
                        63 - v.leading_zeros()
                    };
                    self.wr_reg(i.reg, osz, true, idx as u64);
                    self.set_eflags_arith(f & !ZF);
                }
            }
            _ => {
                if self.sse(i, next)? {
                    self.rip = next;
                    return Ok(());
                }
                return Err(self.unimpl(i));
            }
        }
        self.rip = next;
        Ok(())
    }
}
