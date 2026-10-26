#![allow(dead_code)]

pub(crate) const XZR: u32 = 31;
pub(crate) const SP: u32 = 31;

pub(crate) const EQ: u32 = 0;
pub(crate) const NE: u32 = 1;
pub(crate) const CS: u32 = 2;
pub(crate) const CC: u32 = 3;
pub(crate) const MI: u32 = 4;
pub(crate) const PL: u32 = 5;
pub(crate) const VS: u32 = 6;
pub(crate) const VC: u32 = 7;
pub(crate) const HI: u32 = 8;
pub(crate) const LS: u32 = 9;
pub(crate) const GE: u32 = 10;
pub(crate) const LT: u32 = 11;
pub(crate) const GT: u32 = 12;
pub(crate) const LE: u32 = 13;

pub(crate) type Label = usize;

#[derive(Clone, Copy)]
enum Fix {
    B26,
    B19,
}

pub(crate) struct Asm {
    pub code: Vec<u32>,
    base: u64,
    labels: Vec<Option<usize>>,
    fixes: Vec<(usize, Label, Fix)>,
}

pub(crate) fn encode_bitmask(val: u64, is64: bool) -> Option<u32> {
    let reg: u32 = if is64 { 64 } else { 32 };
    let mut imm = if is64 { val } else { val & 0xffff_ffff };
    if imm == 0 || imm == if is64 { u64::MAX } else { 0xffff_ffff } {
        return None;
    }
    if !is64 {
        imm |= imm << 32;
    }
    let mut size = reg;
    loop {
        size /= 2;
        let mask = (1u64 << size) - 1;
        if (imm & mask) != ((imm >> size) & mask) {
            size *= 2;
            break;
        }
        if size <= 2 {
            break;
        }
    }
    let mask = u64::MAX >> (64 - size);
    imm &= mask;
    let shifted = |v: u64| {
        v != 0 && {
            let x = (v - 1) | v;
            x & x.wrapping_add(1) == 0
        }
    };
    let (i, cto);
    if shifted(imm) {
        i = imm.trailing_zeros();
        cto = (imm >> i).trailing_ones();
    } else {
        imm |= !mask;
        if !shifted(!imm) {
            return None;
        }
        let clo = imm.leading_ones();
        i = 64 - clo;
        cto = clo + imm.trailing_ones() - (64 - size);
    }
    let immr = (size - i) & (size - 1);
    let mut nimms = !(size - 1) << 1;
    nimms |= cto - 1;
    let n = ((nimms >> 6) & 1) ^ 1;
    Some((n << 12) | (immr << 6) | (nimms & 0x3f))
}

impl Asm {
    pub fn new(base: u64) -> Asm {
        Asm {
            code: Vec::new(),
            base,
            labels: Vec::new(),
            fixes: Vec::new(),
        }
    }

    pub fn here(&self) -> u64 {
        self.base + self.code.len() as u64 * 4
    }

    pub fn emit(&mut self, w: u32) {
        self.code.push(w);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        self.labels.len() - 1
    }

    pub fn bind(&mut self, l: Label) {
        self.labels[l] = Some(self.code.len());
    }

    pub fn finish(&mut self) {
        for &(at, l, k) in &self.fixes {
            let off = self.labels[l].expect("unbound label") as i64 - at as i64;
            match k {
                Fix::B26 => self.code[at] |= (off as u32) & 0x03ff_ffff,
                Fix::B19 => self.code[at] |= ((off as u32) & 0x7_ffff) << 5,
            }
        }
        self.fixes.clear();
    }

    pub fn b(&mut self, l: Label) {
        self.fixes.push((self.code.len(), l, Fix::B26));
        self.emit(0x1400_0000);
    }

    pub fn b_cond(&mut self, cond: u32, l: Label) {
        self.fixes.push((self.code.len(), l, Fix::B19));
        self.emit(0x5400_0000 | cond);
    }

    pub fn cbz(&mut self, rt: u32, l: Label, is64: bool) {
        self.fixes.push((self.code.len(), l, Fix::B19));
        self.emit(if is64 { 0xb400_0000 } else { 0x3400_0000 } | rt);
    }

    pub fn cbnz(&mut self, rt: u32, l: Label, is64: bool) {
        self.fixes.push((self.code.len(), l, Fix::B19));
        self.emit(if is64 { 0xb500_0000 } else { 0x3500_0000 } | rt);
    }

    pub fn b_abs(&mut self, target: u64) {
        let off = (target as i64 - self.here() as i64) / 4;
        assert!(off.abs() < (1 << 25), "branch out of range");
        self.emit(0x1400_0000 | (off as u32 & 0x03ff_ffff));
    }

    pub fn ret(&mut self) {
        self.emit(0xd65f_03c0);
    }

    pub fn blr(&mut self, rn: u32) {
        self.emit(0xd63f_0000 | rn << 5);
    }

    pub fn br(&mut self, rn: u32) {
        self.emit(0xd61f_0000 | rn << 5);
    }

    pub fn movz(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0xd280_0000 | hw << 21 | (imm & 0xffff) << 5 | rd);
    }

    pub fn movk(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0xf280_0000 | hw << 21 | (imm & 0xffff) << 5 | rd);
    }

    pub fn movn(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0x9280_0000 | hw << 21 | (imm & 0xffff) << 5 | rd);
    }

    pub fn mov_imm(&mut self, rd: u32, v: u64) {
        if v == 0 {
            self.movz(rd, 0, 0);
            return;
        }
        if !v >> 16 == 0 {
            self.movn(rd, (!v) as u32, 0);
            return;
        }
        let mut first = true;
        for hw in 0..4 {
            let part = ((v >> (16 * hw)) & 0xffff) as u32;
            if part != 0 {
                if first {
                    self.movz(rd, part, hw);
                    first = false;
                } else {
                    self.movk(rd, part, hw);
                }
            }
        }
    }

    pub fn mov(&mut self, rd: u32, rm: u32) {
        self.emit(0xaa00_03e0 | rm << 16 | rd);
    }

    pub fn mov_sp(&mut self, rd: u32, rn: u32) {
        self.emit(0x9100_0000 | rn << 5 | rd);
    }

    fn addsub_imm(&mut self, op: u32, rd: u32, rn: u32, imm: u32, is64: bool) {
        let sf = if is64 { 0x8000_0000 } else { 0 };
        assert!(imm < 4096);
        self.emit(sf | op | imm << 10 | rn << 5 | rd);
    }

    pub fn add_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x1100_0000, rd, rn, imm, is64);
    }
    pub fn sub_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x5100_0000, rd, rn, imm, is64);
    }
    pub fn adds_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x3100_0000, rd, rn, imm, is64);
    }
    pub fn subs_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x7100_0000, rd, rn, imm, is64);
    }

    fn reg3(&mut self, op: u32, rd: u32, rn: u32, rm: u32, is64: bool) {
        let sf = if is64 { 0x8000_0000 } else { 0 };
        self.emit(sf | op | rm << 16 | rn << 5 | rd);
    }

    pub fn add(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x0b00_0000, rd, rn, rm, is64);
    }
    pub fn adds(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x2b00_0000, rd, rn, rm, is64);
    }
    pub fn sub(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x4b00_0000, rd, rn, rm, is64);
    }
    pub fn subs(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x6b00_0000, rd, rn, rm, is64);
    }
    pub fn and(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x0a00_0000, rd, rn, rm, is64);
    }
    pub fn ands(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x6a00_0000, rd, rn, rm, is64);
    }
    pub fn orr(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x2a00_0000, rd, rn, rm, is64);
    }
    pub fn eor(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x4a00_0000, rd, rn, rm, is64);
    }
    pub fn mul(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1b00_7c00, rd, rn, rm, is64);
    }
    pub fn lslv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2000, rd, rn, rm, is64);
    }
    pub fn lsrv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2400, rd, rn, rm, is64);
    }
    pub fn asrv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2800, rd, rn, rm, is64);
    }

    pub fn add_lsl(&mut self, rd: u32, rn: u32, rm: u32, sh: u32) {
        self.emit(0x8b00_0000 | rm << 16 | sh << 10 | rn << 5 | rd);
    }

    pub fn cmp(&mut self, rn: u32, rm: u32, is64: bool) {
        self.subs(XZR, rn, rm, is64);
    }
    pub fn cmp_imm(&mut self, rn: u32, imm: u32, is64: bool) {
        self.subs_imm(XZR, rn, imm, is64);
    }
    pub fn tst(&mut self, rn: u32, rm: u32, is64: bool) {
        self.ands(XZR, rn, rm, is64);
    }

    fn logic_imm(&mut self, op: u32, rd: u32, rn: u32, imm: u64, is64: bool) -> bool {
        match encode_bitmask(imm, is64) {
            Some(e) => {
                let sf = if is64 { 0x8000_0000 } else { 0 };
                self.emit(sf | op | e << 10 | rn << 5 | rd);
                true
            }
            None => false,
        }
    }

    pub fn and_imm(&mut self, rd: u32, rn: u32, imm: u64, is64: bool) -> bool {
        self.logic_imm(0x1200_0000, rd, rn, imm, is64)
    }

    pub fn ubfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let w = if is64 { 0xd340_0000 } else { 0x5300_0000 };
        self.emit(w | immr << 16 | imms << 10 | rn << 5 | rd);
    }
    pub fn sbfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let w = if is64 { 0x9340_0000 } else { 0x1300_0000 };
        self.emit(w | immr << 16 | imms << 10 | rn << 5 | rd);
    }
    pub fn bfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let w = if is64 { 0xb340_0000 } else { 0x3300_0000 };
        self.emit(w | immr << 16 | imms << 10 | rn << 5 | rd);
    }

    pub fn lsl_imm(&mut self, rd: u32, rn: u32, s: u32) {
        self.ubfm(rd, rn, (64 - s) & 63, 63 - s, true);
    }
    pub fn lsr_imm(&mut self, rd: u32, rn: u32, s: u32) {
        self.ubfm(rd, rn, s, 63, true);
    }
    pub fn asr_imm(&mut self, rd: u32, rn: u32, s: u32) {
        self.sbfm(rd, rn, s, 63, true);
    }
    pub fn uxtb(&mut self, rd: u32, rn: u32) {
        self.ubfm(rd, rn, 0, 7, false);
    }
    pub fn uxth(&mut self, rd: u32, rn: u32) {
        self.ubfm(rd, rn, 0, 15, false);
    }
    pub fn uxtw(&mut self, rd: u32, rn: u32) {
        self.emit(0x2a00_03e0 | rn << 16 | rd);
    }
    pub fn sxtb(&mut self, rd: u32, rn: u32, is64: bool) {
        self.sbfm(rd, rn, 0, 7, is64);
    }
    pub fn sxth(&mut self, rd: u32, rn: u32, is64: bool) {
        self.sbfm(rd, rn, 0, 15, is64);
    }
    pub fn sxtw(&mut self, rd: u32, rn: u32) {
        self.sbfm(rd, rn, 0, 31, true);
    }

    pub fn bfi(&mut self, rd: u32, rn: u32, lsb: u32, width: u32, is64: bool) {
        let m = if is64 { 63 } else { 31 };
        self.bfm(rd, rn, (m + 1 - lsb) & m, width - 1, is64);
    }

    pub fn cset(&mut self, rd: u32, cond: u32) {
        self.emit(0x9a9f_07e0 | ((cond ^ 1) << 12) | rd);
    }

    pub fn csel(&mut self, rd: u32, rn: u32, rm: u32, cond: u32) {
        self.emit(0x9a80_0000 | rm << 16 | cond << 12 | rn << 5 | rd);
    }

    fn ldst_imm(&mut self, base: u32, scale: u32, rt: u32, rn: u32, off: u32) {
        assert!(
            off.is_multiple_of(1 << scale) && (off >> scale) < 4096,
            "bad ldst offset {off}"
        );
        self.emit(base | (off >> scale) << 10 | rn << 5 | rt);
    }

    pub fn ldr(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xf940_0000, 3, rt, rn, off);
    }
    pub fn str(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xf900_0000, 3, rt, rn, off);
    }
    pub fn ldr_w(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xb940_0000, 2, rt, rn, off);
    }
    pub fn str_w(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xb900_0000, 2, rt, rn, off);
    }
    pub fn ldrh(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x7940_0000, 1, rt, rn, off);
    }
    pub fn strh(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x7900_0000, 1, rt, rn, off);
    }
    pub fn ldrb(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x3940_0000, 0, rt, rn, off);
    }
    pub fn strb(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x3900_0000, 0, rt, rn, off);
    }

    fn ldst_reg(&mut self, base: u32, rt: u32, rn: u32, rm: u32) {
        self.emit(base | rm << 16 | rn << 5 | rt);
    }

    pub fn ldr_r(&mut self, rt: u32, rn: u32, rm: u32, size: u8, signed: bool) {
        let base = match (size, signed) {
            (1, false) => 0x3860_6800,
            (2, false) => 0x7860_6800,
            (4, false) => 0xb860_6800,
            (8, _) => 0xf860_6800,
            (1, true) => 0x38a0_6800,
            (2, true) => 0x78a0_6800,
            _ => 0xb8a0_6800,
        };
        self.ldst_reg(base, rt, rn, rm);
    }

    pub fn str_r(&mut self, rt: u32, rn: u32, rm: u32, size: u8) {
        let base = match size {
            1 => 0x3820_6800,
            2 => 0x7820_6800,
            4 => 0xb820_6800,
            _ => 0xf820_6800,
        };
        self.ldst_reg(base, rt, rn, rm);
    }

    pub fn stp_pre(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa980_0000 | (((imm / 8) as u32) & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt);
    }
    pub fn stp_off(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa900_0000 | (((imm / 8) as u32) & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt);
    }
    pub fn ldp_off(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa940_0000 | (((imm / 8) as u32) & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt);
    }
    pub fn cinc_eq(&mut self, rd: u32, rn: u32) {
        self.emit(0x9a80_0400 | rn << 16 | (NE << 12) | rn << 5 | rd);
    }
    pub fn mrs_nzcv(&mut self, rt: u32) {
        self.emit(0xd53b_4200 | rt);
    }
    pub fn msr_nzcv(&mut self, rt: u32) {
        self.emit(0xd51b_4200 | rt);
    }
    pub fn ldp_post(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa8c0_0000 | (((imm / 8) as u32) & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::sys::ExecMem;

    fn run(f: impl FnOnce(&mut Asm), a0: u64, a1: u64) -> u64 {
        let m = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(m.base());
        f(&mut a);
        a.ret();
        a.finish();
        m.write(0, &a.code);
        let fun: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(m.base() as usize) };
        fun(a0, a1)
    }

    #[test]
    fn arithmetic() {
        assert_eq!(run(|a| a.add(0, 0, 1, true), 40, 2), 42);
        assert_eq!(run(|a| a.sub(0, 0, 1, true), 40, 2), 38);
        assert_eq!(
            run(|a| a.mov_imm(0, 0x1234_5678_9abc_def0), 0, 0),
            0x1234_5678_9abc_def0
        );
        assert_eq!(run(|a| a.mov_imm(0, u64::MAX - 5), 0, 0), u64::MAX - 5);
        assert_eq!(
            run(|a| a.mov_imm(0, 0xffff_0000_0000_1234), 0, 0),
            0xffff_0000_0000_1234
        );
        assert_eq!(run(|a| a.lsl_imm(0, 0, 4), 3, 0), 48);
        assert_eq!(run(|a| a.lsr_imm(0, 0, 4), 0xf0, 0), 0xf);
        assert_eq!(
            run(|a| a.asr_imm(0, 0, 4), (-32i64) as u64, 0),
            (-2i64) as u64
        );
        assert_eq!(run(|a| a.mul(0, 0, 1, true), 6, 7), 42);
    }

    #[test]
    fn logic_immediates() {
        for &v in &[
            0xffu64,
            0x3ff,
            0xffff_ffff,
            !0xfffu64,
            0xff00,
            0x5555_5555_5555_5555,
            0x8000_0000_0000_0001,
        ] {
            let got = run(
                |a| {
                    assert!(a.and_imm(0, 0, v, true), "{v:#x}");
                },
                u64::MAX,
                0,
            );
            assert_eq!(got, v, "{v:#x}");
        }
        assert!(encode_bitmask(0x1234, true).is_none());
        assert!(encode_bitmask(0, true).is_none());
    }

    #[test]
    fn branches_and_flags() {
        let r = run(
            |a| {
                let l = a.label();
                a.cmp(0, 1, true);
                a.mov_imm(0, 1);
                a.b_cond(LT, l);
                a.mov_imm(0, 2);
                a.bind(l);
            },
            1,
            5,
        );
        assert_eq!(r, 1);
        assert_eq!(
            run(
                |a| {
                    a.adds(2, 0, 1, true);
                    a.cset(0, CS);
                },
                u64::MAX,
                1
            ),
            1
        );
    }

    #[test]
    fn loads_stores() {
        let mut buf = [0u64; 4];
        let p = buf.as_mut_ptr() as u64;
        run(
            |a| {
                a.mov_imm(2, 0xdead_beef_1234_5678);
                a.str(2, 0, 8);
            },
            p,
            0,
        );
        assert_eq!(buf[1], 0xdead_beef_1234_5678);
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 8);
                    a.ldr_r(0, 0, 2, 4, false);
                },
                p,
                0
            ),
            0x1234_5678
        );
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 8);
                    a.ldr_r(0, 0, 2, 1, true);
                },
                p,
                0
            ),
            0x78
        );
        run(
            |a| {
                a.mov_imm(2, 0xab);
                a.mov_imm(3, 16);
                a.str_r(2, 0, 3, 1);
            },
            p,
            0,
        );
        assert_eq!(buf[2] & 0xff, 0xab);
    }
}
