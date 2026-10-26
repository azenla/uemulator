use super::decode::*;
use super::exc::Exc;
use super::exec::Loc;
use super::flags::*;
use super::*;
#[cfg(target_os = "uefi")]
use crate::math::{Float64Ext, FloatExt};

pub(crate) fn ext_to_f64(b: [u8; 10]) -> f64 {
    let m = u64::from_le_bytes(b[..8].try_into().unwrap());
    let se = u16::from_le_bytes([b[8], b[9]]);
    let neg = se & 0x8000 != 0;
    let e = (se & 0x7fff) as i32;
    let v = if e == 0x7fff {
        if m << 1 == 0 { f64::INFINITY } else { f64::NAN }
    } else if m == 0 {
        0.0
    } else {
        let n = e - 16383 - 63;
        let (a, c) = (n / 2, n - n / 2);
        (m as f64) * 2f64.powi(a) * 2f64.powi(c)
    };
    if neg { -v } else { v }
}

pub(crate) fn f64_to_ext(v: f64) -> [u8; 10] {
    let bits = v.to_bits();
    let neg = bits >> 63 != 0;
    let e = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1 << 52) - 1);
    let (se, m): (u16, u64) = if e == 0x7ff {
        (0x7fff, (1 << 63) | (frac << 11))
    } else if e == 0 {
        if frac == 0 {
            (0, 0)
        } else {
            let sh = frac.leading_zeros();
            (((16383 - 1022 - (sh as i32 - 11)) as u16), frac << sh)
        }
    } else {
        ((e - 1023 + 16383) as u16, (1 << 63) | (frac << 11))
    };
    let mut out = [0u8; 10];
    out[..8].copy_from_slice(&m.to_le_bytes());
    out[8..].copy_from_slice(&(se | if neg { 0x8000 } else { 0 }).to_le_bytes());
    out
}

fn round_mode(v: f64, mode: u16) -> f64 {
    match mode & 3 {
        0 => v.round_ties_even(),
        1 => v.floor(),
        2 => v.ceil(),
        _ => v.trunc(),
    }
}

impl Cpu {
    pub fn fpu_check(&self) -> R<()> {
        if self.cr0 & 0xc != 0 {
            return Err(Exc::nm());
        }
        Ok(())
    }

    fn top(&self) -> usize {
        ((self.fsw >> 11) & 7) as usize
    }

    fn set_top(&mut self, t: usize) {
        self.fsw = (self.fsw & !0x3800) | ((t as u16 & 7) << 11);
    }

    fn st_raw(&self, i: usize) -> [u8; 10] {
        self.st[(self.top() + i) & 7]
    }

    fn st_val(&self, i: usize) -> f64 {
        ext_to_f64(self.st_raw(i))
    }

    fn set_st_raw(&mut self, i: usize, v: [u8; 10]) {
        let p = (self.top() + i) & 7;
        self.st[p] = v;
        self.ftw |= 1 << p;
    }

    fn set_st(&mut self, i: usize, v: f64) {
        self.set_st_raw(i, f64_to_ext(v));
    }

    fn fpush_raw(&mut self, v: [u8; 10]) {
        let t = (self.top() + 7) & 7;
        self.set_top(t);
        self.st[t] = v;
        self.ftw |= 1 << t;
    }

    fn fpush(&mut self, v: f64) {
        self.fpush_raw(f64_to_ext(v));
    }

    fn fpop(&mut self) {
        let t = self.top();
        self.ftw &= !(1 << t);
        self.set_top((t + 1) & 7);
    }

    fn set_cc_fsw(&mut self, c0: bool, c2: bool, c3: bool) {
        self.fsw =
            (self.fsw & !0x4500) | ((c0 as u16) << 8) | ((c2 as u16) << 10) | ((c3 as u16) << 14);
    }

    fn fcom_val(&mut self, a: f64, b: f64) {
        if a.is_nan() || b.is_nan() {
            self.set_cc_fsw(true, true, true);
        } else if a > b {
            self.set_cc_fsw(false, false, false);
        } else if a < b {
            self.set_cc_fsw(true, false, false);
        } else {
            self.set_cc_fsw(false, false, true);
        }
    }

    fn fcomi_val(&mut self, a: f64, b: f64) {
        let f = if a.is_nan() || b.is_nan() {
            ZF | PF | CF
        } else if a > b {
            0
        } else if a < b {
            CF
        } else {
            ZF
        };
        let cur = self.eflags_full();
        self.set_eflags_arith((cur & !(ZF | PF | CF)) | f);
    }

    fn arith(r: u8, a: f64, b: f64) -> f64 {
        match r & 7 {
            0 => a + b,
            1 => a * b,
            4 => a - b,
            5 => b - a,
            6 => a / b,
            _ => b / a,
        }
    }

    fn to_int(&self, v: f64, bits: u32) -> i64 {
        let r = round_mode(v, self.fcw >> 10);
        let lim = (1u128 << (bits - 1)) as f64;
        if r.is_nan() || r >= lim || r < -lim {
            i64::MIN >> (64 - bits)
        } else {
            r as i64
        }
    }

    pub fn x87(&mut self, i: &Insn, next: u64) -> R<()> {
        self.fpu_check()?;
        let op = (i.op & 0xff) as u8;
        let r = i.reg & 7;
        let sti = (i.rm & 7) as usize;
        if i.md == 3 {
            match op {
                0xd8 => {
                    if r == 2 || r == 3 {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcom_val(a, b);
                        if r == 3 {
                            self.fpop();
                        }
                    } else {
                        let v = Self::arith(r, self.st_val(0), self.st_val(sti));
                        self.set_st(0, v);
                    }
                }
                0xd9 => match i.modrm {
                    0xc0..=0xc7 => {
                        let v = self.st_raw(sti);
                        self.fpush_raw(v);
                    }
                    0xc8..=0xcf => {
                        let (a, b) = (self.st_raw(0), self.st_raw(sti));
                        self.set_st_raw(0, b);
                        self.set_st_raw(sti, a);
                    }
                    0xd0 => {}
                    0xe0 => {
                        let mut v = self.st_raw(0);
                        v[9] ^= 0x80;
                        self.set_st_raw(0, v);
                    }
                    0xe1 => {
                        let mut v = self.st_raw(0);
                        v[9] &= 0x7f;
                        self.set_st_raw(0, v);
                    }
                    0xe4 => {
                        let a = self.st_val(0);
                        self.fcom_val(a, 0.0);
                    }
                    0xe5 => {
                        let v = self.st_val(0);
                        let neg = self.st_raw(0)[9] & 0x80 != 0;
                        let empty = self.ftw & (1 << self.top()) == 0;
                        let (c3, c2, c0) = if empty {
                            (true, false, true)
                        } else if v.is_nan() {
                            (false, false, true)
                        } else if v.is_infinite() {
                            (false, true, true)
                        } else if v == 0.0 {
                            (true, false, false)
                        } else {
                            (false, true, false)
                        };
                        self.set_cc_fsw(c0, c2, c3);
                        self.fsw = (self.fsw & !0x200) | ((neg as u16) << 9);
                    }
                    0xe8 => self.fpush(1.0),
                    0xe9 => self.fpush(core::f64::consts::LOG2_10),
                    0xea => self.fpush(core::f64::consts::LOG2_E),
                    0xeb => self.fpush(core::f64::consts::PI),
                    0xec => self.fpush(core::f64::consts::LOG10_2),
                    0xed => self.fpush(core::f64::consts::LN_2),
                    0xee => self.fpush(0.0),
                    0xf0 => {
                        let x = self.st_val(0);
                        self.set_st(0, (x * core::f64::consts::LN_2).exp_m1());
                    }
                    0xf1 => {
                        let v = self.st_val(1) * self.st_val(0).log2();
                        self.set_st(1, v);
                        self.fpop();
                    }
                    0xf2 => {
                        let x = self.st_val(0).tan();
                        self.set_st(0, x);
                        self.fpush(1.0);
                        self.set_cc_fsw(false, false, false);
                    }
                    0xf3 => {
                        let v = self.st_val(1).atan2(self.st_val(0));
                        self.set_st(1, v);
                        self.fpop();
                    }
                    0xf5 | 0xf8 => {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        let q = if i.modrm == 0xf8 {
                            (a / b).trunc()
                        } else {
                            (a / b).round_ties_even()
                        };
                        self.set_st(0, a - q * b);
                        let qi = q.abs() as u64;
                        self.set_cc_fsw(qi & 4 != 0, false, qi & 2 != 0);
                        self.fsw = (self.fsw & !0x200) | (((qi & 1) as u16) << 9);
                    }
                    0xf6 => {
                        let t = self.top();
                        self.set_top((t + 7) & 7);
                    }
                    0xf7 => {
                        let t = self.top();
                        self.set_top((t + 1) & 7);
                    }
                    0xf9 => {
                        let v = self.st_val(1) * (1.0 + self.st_val(0)).log2();
                        self.set_st(1, v);
                        self.fpop();
                    }
                    0xfa => {
                        let v = self.st_val(0).sqrt();
                        self.set_st(0, v);
                    }
                    0xfb => {
                        let x = self.st_val(0);
                        self.set_st(0, x.sin());
                        self.fpush(x.cos());
                        self.set_cc_fsw(false, false, false);
                    }
                    0xfc => {
                        let v = round_mode(self.st_val(0), self.fcw >> 10);
                        self.set_st(0, v);
                    }
                    0xfd => {
                        let v = self.st_val(0) * 2f64.powf(self.st_val(1).trunc());
                        self.set_st(0, v);
                    }
                    0xfe => {
                        let v = self.st_val(0).sin();
                        self.set_st(0, v);
                        self.set_cc_fsw(false, false, false);
                    }
                    0xff => {
                        let v = self.st_val(0).cos();
                        self.set_st(0, v);
                        self.set_cc_fsw(false, false, false);
                    }
                    _ => return Err(self.unimpl(i)),
                },
                0xda => match i.modrm {
                    0xc0..=0xdf => {
                        let f = self.eflags_full();
                        let c = match r {
                            0 => f & CF != 0,
                            1 => f & ZF != 0,
                            2 => f & (CF | ZF) != 0,
                            _ => f & PF != 0,
                        };
                        if c {
                            let v = self.st_raw(sti);
                            self.set_st_raw(0, v);
                        }
                    }
                    0xe9 => {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        self.fcom_val(a, b);
                        self.fpop();
                        self.fpop();
                    }
                    _ => return Err(self.unimpl(i)),
                },
                0xdb => match i.modrm {
                    0xc0..=0xdf => {
                        let f = self.eflags_full();
                        let c = match r {
                            0 => f & CF == 0,
                            1 => f & ZF == 0,
                            2 => f & (CF | ZF) == 0,
                            _ => f & PF == 0,
                        };
                        if c {
                            let v = self.st_raw(sti);
                            self.set_st_raw(0, v);
                        }
                    }
                    0xe2 => self.fsw &= 0x7f00,
                    0xe3 => {
                        self.fcw = 0x37f;
                        self.fsw = 0;
                        self.ftw = 0;
                    }
                    0xe8..=0xf7 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcomi_val(a, b);
                    }
                    _ => return Err(self.unimpl(i)),
                },
                0xdc => {
                    if r == 2 || r == 3 {
                        return Err(self.unimpl(i));
                    }
                    let (a, b) = (self.st_val(sti), self.st_val(0));
                    let v = match r {
                        4 => b - a,
                        5 => a - b,
                        6 => b / a,
                        7 => a / b,
                        _ => Self::arith(r, a, b),
                    };
                    self.set_st(sti, v);
                }
                0xdd => match r {
                    0 => self.ftw &= !(1 << ((self.top() + sti) & 7)),
                    2 | 3 => {
                        let v = self.st_raw(0);
                        self.set_st_raw(sti, v);
                        if r == 3 {
                            self.fpop();
                        }
                    }
                    4 | 5 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcom_val(a, b);
                        if r == 5 {
                            self.fpop();
                        }
                    }
                    _ => return Err(self.unimpl(i)),
                },
                0xde => {
                    if i.modrm == 0xd9 {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        self.fcom_val(a, b);
                        self.fpop();
                    } else if r == 2 || r == 3 {
                        return Err(self.unimpl(i));
                    } else {
                        let (a, b) = (self.st_val(sti), self.st_val(0));
                        let v = match r {
                            4 => b - a,
                            5 => a - b,
                            6 => b / a,
                            7 => a / b,
                            _ => Self::arith(r, a, b),
                        };
                        self.set_st(sti, v);
                    }
                    self.fpop();
                }
                _ => match i.modrm {
                    0xe0 => {
                        let v = self.fsw as u64;
                        self.wr_reg(0, 2, true, v);
                    }
                    0xe8..=0xf7 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcomi_val(a, b);
                        self.fpop();
                    }
                    0xc0..=0xc7 => {
                        self.ftw &= !(1 << ((self.top() + sti) & 7));
                        self.fpop();
                    }
                    _ => return Err(self.unimpl(i)),
                },
            }
            self.rip = next;
            return Ok(());
        }
        let a = match self.rm_loc(i, next) {
            Loc::Mem(a) => a,
            _ => unreachable!(),
        };
        match op {
            0xd8 | 0xdc => {
                let v = if op == 0xd8 {
                    f32::from_bits(self.read(a, 4)? as u32) as f64
                } else {
                    f64::from_bits(self.read(a, 8)?)
                };
                self.arith_mem(r, v);
            }
            0xda | 0xde => {
                let v = if op == 0xda {
                    self.read(a, 4)? as u32 as i32 as f64
                } else {
                    self.read(a, 2)? as u16 as i16 as f64
                };
                self.arith_mem(r, v);
            }
            0xd9 => match r {
                0 => {
                    let v = f32::from_bits(self.read(a, 4)? as u32) as f64;
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.st_val(0) as f32;
                    self.write(a, 4, v.to_bits() as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                4 => self.fldenv(a)?,
                5 => self.fcw = self.read(a, 2)? as u16,
                6 => self.fnstenv(a)?,
                _ => self.write(a, 2, self.fcw as u64)?,
            },
            0xdb => match r {
                0 => {
                    let v = self.read(a, 4)? as u32 as i32 as f64;
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.to_int(self.st_val(0), 32) as u32;
                    self.write(a, 4, v as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                5 => {
                    let mut raw = [0u8; 10];
                    for (k, b) in raw.iter_mut().enumerate() {
                        *b = self.read(a + k as u64, 1)? as u8;
                    }
                    self.fpush_raw(raw);
                }
                7 => {
                    let raw = self.st_raw(0);
                    for (k, b) in raw.iter().enumerate() {
                        self.write(a + k as u64, 1, *b as u64)?;
                    }
                    self.fpop();
                }
                _ => return Err(self.unimpl(i)),
            },
            0xdd => match r {
                0 => {
                    let v = f64::from_bits(self.read(a, 8)?);
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.st_val(0);
                    self.write(a, 8, v.to_bits())?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                4 => self.frstor(a)?,
                6 => self.fnsave(a)?,
                7 => self.write(a, 2, self.fsw as u64)?,
                _ => return Err(self.unimpl(i)),
            },
            _ => match r {
                0 => {
                    let v = self.read(a, 2)? as u16 as i16 as f64;
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.to_int(self.st_val(0), 16) as u16;
                    self.write(a, 2, v as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                5 => {
                    let v = self.read(a, 8)? as i64 as f64;
                    self.fpush(v);
                }
                7 => {
                    let v = self.to_int(self.st_val(0), 64);
                    self.write(a, 8, v as u64)?;
                    self.fpop();
                }
                _ => return Err(self.unimpl(i)),
            },
        }
        self.rip = next;
        Ok(())
    }

    fn arith_mem(&mut self, r: u8, v: f64) {
        let a = self.st_val(0);
        if r == 2 || r == 3 {
            self.fcom_val(a, v);
            if r == 3 {
                self.fpop();
            }
        } else {
            let res = Self::arith(r, a, v);
            self.set_st(0, res);
        }
    }

    fn full_tag(&self) -> u16 {
        let mut t = 0u16;
        for p in 0..8 {
            let tag = if self.ftw & (1 << p) == 0 {
                3
            } else {
                let v = ext_to_f64(self.st[p]);
                if v == 0.0 {
                    1
                } else if v.is_finite() {
                    0
                } else {
                    2
                }
            };
            t |= tag << (2 * p);
        }
        t
    }

    fn fnstenv(&mut self, a: u64) -> R<()> {
        self.write(a, 4, self.fcw as u64)?;
        self.write(a + 4, 4, self.fsw as u64)?;
        let t = self.full_tag();
        self.write(a + 8, 4, t as u64)?;
        for k in 3..7 {
            self.write(a + k * 4, 4, 0)?;
        }
        self.fcw |= 0x3f;
        Ok(())
    }

    fn fldenv(&mut self, a: u64) -> R<()> {
        self.fcw = self.read(a, 2)? as u16;
        self.fsw = self.read(a + 4, 2)? as u16;
        let t = self.read(a + 8, 2)? as u16;
        self.ftw = (0..8)
            .filter(|p| (t >> (2 * p)) & 3 != 3)
            .fold(0, |m, p| m | (1 << p));
        Ok(())
    }

    fn fnsave(&mut self, a: u64) -> R<()> {
        self.fnstenv(a)?;
        for k in 0..8 {
            let raw = self.st_raw(k);
            for (j, b) in raw.iter().enumerate() {
                self.write(a + 28 + k as u64 * 10 + j as u64, 1, *b as u64)?;
            }
        }
        self.fcw = 0x37f;
        self.fsw = 0;
        self.ftw = 0;
        Ok(())
    }

    fn frstor(&mut self, a: u64) -> R<()> {
        self.fldenv(a)?;
        for k in 0..8 {
            let mut raw = [0u8; 10];
            for (j, b) in raw.iter_mut().enumerate() {
                *b = self.read(a + 28 + k as u64 * 10 + j as u64, 1)? as u8;
            }
            let p = (self.top() + k) & 7;
            self.st[p] = raw;
        }
        Ok(())
    }

    pub fn fxsave(&mut self, a: u64) -> R<()> {
        if self.cr0 & 4 != 0 {
            return Err(Exc::ud());
        }
        self.fpu_check()?;
        self.translate(a, mem::Acc::Write, false)?;
        self.translate(a + 511, mem::Acc::Write, false)?;
        self.write(a, 2, self.fcw as u64)?;
        self.write(a + 2, 2, self.fsw as u64)?;
        self.write(a + 4, 1, self.ftw as u64 & 0xff)?;
        self.write(a + 24, 4, self.mxcsr as u64)?;
        self.write(a + 28, 4, 0xffff)?;
        for k in 0..8 {
            let b = self.st_raw(k);
            for (j, byte) in b.iter().enumerate() {
                self.write(a + 32 + k as u64 * 16 + j as u64, 1, *byte as u64)?;
            }
        }
        for k in 0..16 {
            self.write128(a + 160 + k as u64 * 16, self.xmm[k])?;
        }
        Ok(())
    }

    pub fn fxrstor(&mut self, a: u64) -> R<()> {
        if self.cr0 & 4 != 0 {
            return Err(Exc::ud());
        }
        self.fpu_check()?;
        let fcw = self.read(a, 2)? as u16;
        let fsw = self.read(a + 2, 2)? as u16;
        let ftw = self.read(a + 4, 1)? as u16;
        let mxcsr = self.read(a + 24, 4)? as u32;
        if mxcsr & !0xffff != 0 {
            return Err(Exc::gp(0));
        }
        let mut st = [[0u8; 10]; 8];
        let top = ((fsw >> 11) & 7) as usize;
        for k in 0..8 {
            for (j, b) in st[(top + k) & 7].iter_mut().enumerate() {
                *b = self.read(a + 32 + k as u64 * 16 + j as u64, 1)? as u8;
            }
        }
        for k in 0..16 {
            self.xmm[k] = self.read128(a + 160 + k as u64 * 16)?;
        }
        self.st = st;
        self.fcw = fcw;
        self.fsw = fsw;
        self.ftw = ftw;
        self.mxcsr = mxcsr;
        Ok(())
    }
}
