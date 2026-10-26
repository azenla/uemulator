use super::decode::*;
use super::exc::Exc;
use super::flags::*;
use super::*;
#[cfg(target_os = "uefi")]
use crate::math::FloatExt;

trait Lane: Copy {
    const N: usize;
    fn get(v: u128, i: usize) -> Self;
    fn put(v: &mut u128, i: usize, x: Self);
}

macro_rules! lane {
    ($t:ty, $n:expr) => {
        impl Lane for $t {
            const N: usize = $n;
            fn get(v: u128, i: usize) -> Self {
                (v >> (i * (128 / $n))) as $t
            }
            fn put(v: &mut u128, i: usize, x: Self) {
                *v |= (x as u128) << (i * (128 / $n));
            }
        }
    };
}
lane!(u8, 16);
lane!(u16, 8);
lane!(u32, 4);
lane!(u64, 2);

fn zip<T: Lane>(a: u128, b: u128, f: impl Fn(T, T) -> T) -> u128 {
    let mut r = 0;
    for i in 0..T::N {
        T::put(&mut r, i, f(T::get(a, i), T::get(b, i)));
    }
    r
}

fn unpack<T: Lane>(a: u128, b: u128, hi: bool) -> u128 {
    let mut r = 0;
    let off = if hi { T::N / 2 } else { 0 };
    for i in 0..T::N / 2 {
        T::put(&mut r, 2 * i, T::get(a, off + i));
        T::put(&mut r, 2 * i + 1, T::get(b, off + i));
    }
    r
}

fn f32s(v: u128) -> [f32; 4] {
    [0, 1, 2, 3].map(|i| f32::from_bits(u32::get(v, i)))
}
fn from_f32s(a: [f32; 4]) -> u128 {
    let mut r = 0;
    for (i, x) in a.iter().enumerate() {
        u32::put(&mut r, i, x.to_bits());
    }
    r
}
fn f64s(v: u128) -> [f64; 2] {
    [0, 1].map(|i| f64::from_bits(u64::get(v, i)))
}
fn from_f64s(a: [f64; 2]) -> u128 {
    a[0].to_bits() as u128 | ((a[1].to_bits() as u128) << 64)
}

fn pack_sat(a: u128, b: u128, wide: usize, signed_out: bool) -> u128 {
    let mut r = 0u128;
    let get = |v: u128, i: usize| -> i32 {
        if wide == 16 {
            u16::get(v, i) as i16 as i32
        } else {
            u32::get(v, i) as i32
        }
    };
    let n = 128 / wide;
    for k in 0..2 * n {
        let x = get(if k < n { a } else { b }, k % n);
        if wide == 16 {
            let y = if signed_out {
                x.clamp(-128, 127) as i8 as u8
            } else {
                x.clamp(0, 255) as u8
            };
            u8::put(&mut r, k, y);
        } else {
            let y = if signed_out {
                x.clamp(-32768, 32767) as i16 as u16
            } else {
                x.clamp(0, 65535) as u16
            };
            u16::put(&mut r, k, y);
        }
    }
    r
}

fn cvt_i32(v: f64, trunc: bool, mxcsr: u32) -> i32 {
    let r = if trunc { v.trunc() } else { round_rc(v, mxcsr) };
    if r.is_nan() || !(-2147483648.0..2147483648.0).contains(&r) {
        i32::MIN
    } else {
        r as i32
    }
}

fn cvt_i64(v: f64, trunc: bool, mxcsr: u32) -> i64 {
    let r = if trunc { v.trunc() } else { round_rc(v, mxcsr) };
    if r.is_nan() || !(-9.223_372_036_854_776e18..9.223_372_036_854_776e18).contains(&r) {
        i64::MIN
    } else {
        r as i64
    }
}

fn round_rc(v: f64, mxcsr: u32) -> f64 {
    match (mxcsr >> 13) & 3 {
        0 => v.round_ties_even(),
        1 => v.floor(),
        2 => v.ceil(),
        _ => v.trunc(),
    }
}

fn cmp_pred(p: u8, a: f64, b: f64) -> bool {
    let un = a.is_nan() || b.is_nan();
    match p & 7 {
        0 => a == b,
        1 => a < b,
        2 => a <= b,
        3 => un,
        4 => a != b || un,
        5 => a.partial_cmp(&b) != Some(core::cmp::Ordering::Less),
        6 => !matches!(
            a.partial_cmp(&b),
            Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
        ),
        _ => !un,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum P {
    N,
    O66,
    F3,
    F2,
}

impl Cpu {
    fn sse_gate(&self) -> R<()> {
        if self.cr0 & 4 != 0 || self.cr4 & 0x200 == 0 {
            return Err(Exc::ud());
        }
        if self.cr0 & 8 != 0 {
            return Err(Exc::nm());
        }
        Ok(())
    }

    fn fp_binop(&self, op: u8, a: f64, b: f64) -> f64 {
        match op {
            0x51 => b.sqrt(),
            0x58 => a + b,
            0x59 => a * b,
            0x5c => a - b,
            0x5d => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            0x5e => a / b,
            _ => {
                if a > b {
                    a
                } else {
                    b
                }
            }
        }
    }

    pub fn sse(&mut self, i: &Insn, next: u64) -> R<bool> {
        let op = i.op;
        if op >> 8 != 1 {
            return Ok(false);
        }
        let low = (op & 0xff) as u8;
        let p = match (i.rep, i.p66) {
            (1, _) => P::F3,
            (2, _) => P::F2,
            (_, true) => P::O66,
            _ => P::N,
        };
        if !allowed(low, p) {
            return Ok(false);
        }
        if low == 0x77 {
            self.fpu_check()?;
            self.ftw = 0;
            return Ok(true);
        }
        self.sse_gate()?;
        let d = i.reg as usize;
        let rmx = i.rm as usize;
        let mem = if i.md == 3 {
            None
        } else {
            Some(self.lin(i, self.ea(i, next)))
        };
        macro_rules! src {
            ($sz:expr) => {
                match mem {
                    None => self.xmm[rmx],
                    Some(a) => {
                        if $sz == 16 {
                            self.read128(a)?
                        } else {
                            self.read(a, $sz)? as u128
                        }
                    }
                }
            };
        }
        macro_rules! aligned {
            () => {
                if let Some(a) = mem {
                    if a & 15 != 0 {
                        return Err(Exc::gp(0));
                    }
                }
            };
        }
        let xd = self.xmm[d];
        let mx = self.mxcsr;
        let m32: u128 = 0xffff_ffff;
        let m64: u128 = u64::MAX as u128;
        let res: Option<u128> = match low {
            0x10 => Some(match p {
                P::N | P::O66 => src!(16),
                P::F3 => {
                    if mem.is_some() {
                        src!(4)
                    } else {
                        (xd & !m32) | (self.xmm[rmx] & m32)
                    }
                }
                P::F2 => {
                    if mem.is_some() {
                        src!(8)
                    } else {
                        (xd & !m64) | (self.xmm[rmx] & m64)
                    }
                }
            }),
            0x11 | 0x29 | 0x2b | 0x7f | 0xe7 => {
                if matches!(low, 0x29 | 0x2b) || (low == 0x7f && p == P::O66) || low == 0xe7 {
                    aligned!();
                }
                let sz = match (low, p) {
                    (0x11, P::F3) => 4,
                    (0x11, P::F2) => 8,
                    _ => 16,
                };
                match mem {
                    Some(a) => match sz {
                        16 => self.write128(a, xd)?,
                        s => self.write(a, s, xd as u64)?,
                    },
                    None => {
                        let cur = self.xmm[rmx];
                        self.xmm[rmx] = match sz {
                            4 => (cur & !m32) | (xd & m32),
                            8 => (cur & !m64) | (xd & m64),
                            _ => xd,
                        };
                    }
                }
                None
            }
            0x12 | 0x16 => {
                let lo = low == 0x12;
                let v = if mem.is_none() {
                    if lo {
                        self.xmm[rmx] >> 64
                    } else {
                        self.xmm[rmx] & m64
                    }
                } else {
                    src!(8)
                } & m64;
                Some(if lo {
                    (xd & !m64) | v
                } else {
                    (xd & m64) | (v << 64)
                })
            }
            0x13 | 0x17 => {
                let a = mem.ok_or_else(Exc::ud)?;
                let v = if low == 0x13 {
                    xd as u64
                } else {
                    (xd >> 64) as u64
                };
                self.write(a, 8, v)?;
                None
            }
            0x14 | 0x15 => {
                let s = src!(16);
                let hi = low == 0x15;
                Some(if p == P::N {
                    unpack::<u32>(xd, s, hi)
                } else {
                    unpack::<u64>(xd, s, hi)
                })
            }
            0x28 => {
                aligned!();
                Some(src!(16))
            }
            0x2a => {
                let sz = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem {
                    None => self.rd_reg(i.rm, sz, true),
                    Some(a) => self.read(a, sz)?,
                };
                let f = if sz == 8 {
                    v as i64 as f64
                } else {
                    v as u32 as i32 as f64
                };
                Some(if p == P::F3 {
                    (xd & !m32) | (f as f32).to_bits() as u128
                } else {
                    (xd & !m64) | f.to_bits() as u128
                })
            }
            0x2c | 0x2d => {
                let sz = if p == P::F3 { 4 } else { 8 };
                let v = src!(sz) & if sz == 4 { m32 } else { m64 };
                let f = if sz == 4 {
                    f32::from_bits(v as u32) as f64
                } else {
                    f64::from_bits(v as u64)
                };
                let t = low == 0x2c;
                if i.opsize == 8 {
                    self.gpr[d] = cvt_i64(f, t, mx) as u64;
                } else {
                    self.gpr[d] = cvt_i32(f, t, mx) as u32 as u64;
                }
                None
            }
            0x2e | 0x2f => {
                let (a, b) = if p == P::N {
                    let s = src!(4) as u32;
                    (f32::from_bits(xd as u32) as f64, f32::from_bits(s) as f64)
                } else {
                    let s = src!(8) as u64;
                    (f64::from_bits(xd as u64), f64::from_bits(s))
                };
                let f = if a.is_nan() || b.is_nan() {
                    ZF | PF | CF
                } else if a > b {
                    0
                } else if a < b {
                    CF
                } else {
                    ZF
                };
                self.set_eflags_arith(f);
                None
            }
            0x50 => {
                if mem.is_some() {
                    return Err(Exc::ud());
                }
                let x = self.xmm[rmx];
                self.gpr[d] = if p == P::N {
                    (0..4).fold(0, |m, k| m | (((u32::get(x, k) >> 31) as u64) << k))
                } else {
                    (0..2).fold(0, |m, k| m | ((u64::get(x, k) >> 63) << k))
                };
                None
            }
            0x51 | 0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f => {
                let o = low;
                Some(match p {
                    P::N => {
                        let s = f32s(src!(16));
                        let a = f32s(xd);
                        from_f32s(
                            [0, 1, 2, 3].map(|k| self.fp_binop(o, a[k] as f64, s[k] as f64) as f32),
                        )
                    }
                    P::O66 => {
                        let s = f64s(src!(16));
                        let a = f64s(xd);
                        from_f64s([0, 1].map(|k| self.fp_binop(o, a[k], s[k])))
                    }
                    P::F3 => {
                        let s = f32::from_bits(src!(4) as u32);
                        let r = self.fp_binop(o, f32::from_bits(xd as u32) as f64, s as f64) as f32;
                        (xd & !m32) | r.to_bits() as u128
                    }
                    P::F2 => {
                        let s = f64::from_bits(src!(8) as u64);
                        let r = self.fp_binop(o, f64::from_bits(xd as u64), s);
                        (xd & !m64) | r.to_bits() as u128
                    }
                })
            }
            0x52 | 0x53 => {
                let f = |x: f32| if low == 0x52 { 1.0 / x.sqrt() } else { 1.0 / x };
                Some(if p == P::F3 {
                    let s = f32::from_bits(src!(4) as u32);
                    (xd & !m32) | f(s).to_bits() as u128
                } else {
                    from_f32s(f32s(src!(16)).map(f))
                })
            }
            0x54..=0x57 => {
                let s = src!(16);
                Some(match low {
                    0x54 => xd & s,
                    0x55 => !xd & s,
                    0x56 => xd | s,
                    _ => xd ^ s,
                })
            }
            0x5a => Some(match p {
                P::N => {
                    let s = f32s(src!(8));
                    from_f64s([s[0] as f64, s[1] as f64])
                }
                P::O66 => {
                    let s = f64s(src!(16));
                    from_f32s([s[0] as f32, s[1] as f32, 0.0, 0.0])
                }
                P::F3 => {
                    let s = f32::from_bits(src!(4) as u32);
                    (xd & !m64) | (s as f64).to_bits() as u128
                }
                P::F2 => {
                    let s = f64::from_bits(src!(8) as u64);
                    (xd & !m32) | (s as f32).to_bits() as u128
                }
            }),
            0x5b => {
                let s = src!(16);
                Some(match p {
                    P::N => from_f32s([0, 1, 2, 3].map(|k| u32::get(s, k) as i32 as f32)),
                    _ => {
                        let t = p == P::F3;
                        let mut r = 0;
                        for (k, f) in f32s(s).iter().enumerate() {
                            u32::put(&mut r, k, cvt_i32(*f as f64, t, mx) as u32);
                        }
                        r
                    }
                })
            }
            0x60..=0x6d => {
                let s = src!(16);
                Some(match low {
                    0x60 => unpack::<u8>(xd, s, false),
                    0x61 => unpack::<u16>(xd, s, false),
                    0x62 => unpack::<u32>(xd, s, false),
                    0x63 => pack_sat(xd, s, 16, true),
                    0x64 => zip::<u8>(xd, s, |a, b| if (a as i8) > (b as i8) { 0xff } else { 0 }),
                    0x65 => zip::<u16>(
                        xd,
                        s,
                        |a, b| if (a as i16) > (b as i16) { 0xffff } else { 0 },
                    ),
                    0x66 => zip::<u32>(
                        xd,
                        s,
                        |a, b| if (a as i32) > (b as i32) { u32::MAX } else { 0 },
                    ),
                    0x67 => pack_sat(xd, s, 16, false),
                    0x68 => unpack::<u8>(xd, s, true),
                    0x69 => unpack::<u16>(xd, s, true),
                    0x6a => unpack::<u32>(xd, s, true),
                    0x6b => pack_sat(xd, s, 32, true),
                    0x6c => unpack::<u64>(xd, s, false),
                    _ => unpack::<u64>(xd, s, true),
                })
            }
            0x6e => {
                let sz = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem {
                    None => self.rd_reg(i.rm, sz, true),
                    Some(a) => self.read(a, sz)?,
                };
                Some(v as u128)
            }
            0x6f => {
                if p == P::O66 {
                    aligned!();
                }
                Some(src!(16))
            }
            0x70 => {
                let s = src!(16);
                let im = i.imm as u8;
                Some(match p {
                    P::O66 => {
                        let mut r = 0;
                        for k in 0..4 {
                            u32::put(&mut r, k, u32::get(s, ((im >> (2 * k)) & 3) as usize));
                        }
                        r
                    }
                    P::F2 => {
                        let mut r = s & !m64;
                        for k in 0..4 {
                            u16::put(&mut r, k, u16::get(s, ((im >> (2 * k)) & 3) as usize));
                        }
                        r
                    }
                    _ => {
                        let mut r = s & m64;
                        for k in 0..4 {
                            u16::put(
                                &mut r,
                                4 + k,
                                u16::get(s, 4 + ((im >> (2 * k)) & 3) as usize),
                            );
                        }
                        r
                    }
                })
            }
            0x71..=0x73 => {
                if mem.is_some() {
                    return Err(Exc::ud());
                }
                let x = self.xmm[rmx];
                let c = i.imm as u8 as u64;
                let k = i.reg & 7;
                let r = match (low, k) {
                    (0x71, 2) => shl_r::<u16>(x, c, false, 16),
                    (0x71, 4) => sra::<u16>(x, c, 16),
                    (0x71, 6) => shl_r::<u16>(x, c, true, 16),
                    (0x72, 2) => shl_r::<u32>(x, c, false, 32),
                    (0x72, 4) => sra::<u32>(x, c, 32),
                    (0x72, 6) => shl_r::<u32>(x, c, true, 32),
                    (0x73, 2) => shl_r::<u64>(x, c, false, 64),
                    (0x73, 6) => shl_r::<u64>(x, c, true, 64),
                    (0x73, 3) => {
                        if c > 15 {
                            0
                        } else {
                            x >> (c * 8)
                        }
                    }
                    (0x73, 7) => {
                        if c > 15 {
                            0
                        } else {
                            x << (c * 8)
                        }
                    }
                    _ => return Err(self.unimpl(i)),
                };
                self.xmm[rmx] = r;
                None
            }
            0x74 => {
                let s = src!(16);
                Some(zip::<u8>(xd, s, |a, b| if a == b { 0xff } else { 0 }))
            }
            0x75 => {
                let s = src!(16);
                Some(zip::<u16>(xd, s, |a, b| if a == b { 0xffff } else { 0 }))
            }
            0x76 => {
                let s = src!(16);
                Some(zip::<u32>(xd, s, |a, b| if a == b { u32::MAX } else { 0 }))
            }
            0x7e => {
                if p == P::F3 {
                    Some(src!(8) & m64)
                } else {
                    let sz = if i.opsize == 8 { 8 } else { 4 };
                    let v = xd as u64 & if sz == 8 { u64::MAX } else { 0xffff_ffff };
                    match mem {
                        None => self.wr_reg(i.rm, sz, true, v),
                        Some(a) => self.write(a, sz, v)?,
                    }
                    None
                }
            }
            0xc2 => {
                let pr = i.imm as u8;
                Some(match p {
                    P::N => {
                        let (a, s) = (f32s(xd), f32s(src!(16)));
                        let mut r = 0;
                        for k in 0..4 {
                            u32::put(
                                &mut r,
                                k,
                                if cmp_pred(pr, a[k] as f64, s[k] as f64) {
                                    u32::MAX
                                } else {
                                    0
                                },
                            );
                        }
                        r
                    }
                    P::O66 => {
                        let (a, s) = (f64s(xd), f64s(src!(16)));
                        let mut r = 0;
                        for k in 0..2 {
                            u64::put(
                                &mut r,
                                k,
                                if cmp_pred(pr, a[k], s[k]) {
                                    u64::MAX
                                } else {
                                    0
                                },
                            );
                        }
                        r
                    }
                    P::F3 => {
                        let s = f32::from_bits(src!(4) as u32) as f64;
                        let t = cmp_pred(pr, f32::from_bits(xd as u32) as f64, s);
                        (xd & !m32) | if t { m32 } else { 0 }
                    }
                    P::F2 => {
                        let s = f64::from_bits(src!(8) as u64);
                        let t = cmp_pred(pr, f64::from_bits(xd as u64), s);
                        (xd & !m64) | if t { m64 } else { 0 }
                    }
                })
            }
            0xc4 => {
                let v = match mem {
                    None => self.rd_reg(i.rm, 2, true),
                    Some(a) => self.read(a, 2)?,
                } as u16;
                let k = (i.imm as usize) & 7;
                let sh = k * 16;
                Some((xd & !(0xffffu128 << sh)) | ((v as u128) << sh))
            }
            0xc5 => {
                if mem.is_some() {
                    return Err(Exc::ud());
                }
                let k = (i.imm as usize) & 7;
                self.gpr[d] = u16::get(self.xmm[rmx], k) as u64;
                None
            }
            0xc6 => {
                let s = src!(16);
                let im = i.imm as u8 as usize;
                Some(if p == P::N {
                    let mut r = 0;
                    u32::put(&mut r, 0, u32::get(xd, im & 3));
                    u32::put(&mut r, 1, u32::get(xd, (im >> 2) & 3));
                    u32::put(&mut r, 2, u32::get(s, (im >> 4) & 3));
                    u32::put(&mut r, 3, u32::get(s, (im >> 6) & 3));
                    r
                } else {
                    u64::get(xd, im & 1) as u128 | ((u64::get(s, (im >> 1) & 1) as u128) << 64)
                })
            }
            0xd1..=0xd3 | 0xe1 | 0xe2 | 0xf1..=0xf3 => {
                let cs = src!(16);
                let c = if cs >> 64 != 0 { u64::MAX } else { cs as u64 };
                Some(match low {
                    0xd1 => shl_r::<u16>(xd, c, false, 16),
                    0xd2 => shl_r::<u32>(xd, c, false, 32),
                    0xd3 => shl_r::<u64>(xd, c, false, 64),
                    0xe1 => sra::<u16>(xd, c, 16),
                    0xe2 => sra::<u32>(xd, c, 32),
                    0xf1 => shl_r::<u16>(xd, c, true, 16),
                    0xf2 => shl_r::<u32>(xd, c, true, 32),
                    _ => shl_r::<u64>(xd, c, true, 64),
                })
            }
            0xd6 => {
                let v = xd & m64;
                match mem {
                    None => self.xmm[rmx] = v,
                    Some(a) => self.write(a, 8, v as u64)?,
                }
                None
            }
            0xd7 => {
                if mem.is_some() {
                    return Err(Exc::ud());
                }
                let x = self.xmm[rmx];
                self.gpr[d] = (0..16).fold(0, |m, k| m | (((u8::get(x, k) >> 7) as u64) << k));
                None
            }
            0xe6 => {
                let t = p == P::O66;
                Some(match p {
                    P::F3 => {
                        let s = src!(8);
                        from_f64s([u32::get(s, 0) as i32 as f64, u32::get(s, 1) as i32 as f64])
                    }
                    _ => {
                        let s = f64s(src!(16));
                        let mut r = 0;
                        u32::put(&mut r, 0, cvt_i32(s[0], t, mx) as u32);
                        u32::put(&mut r, 1, cvt_i32(s[1], t, mx) as u32);
                        r
                    }
                })
            }
            0xd4
            | 0xd5
            | 0xd8..=0xdf
            | 0xe0
            | 0xe3..=0xe5
            | 0xe8..=0xef
            | 0xf4..=0xf6
            | 0xf8..=0xfe => {
                let s = src!(16);
                Some(match low {
                    0xd4 => zip::<u64>(xd, s, |a, b| a.wrapping_add(b)),
                    0xd5 => zip::<u16>(xd, s, |a, b| (a as i16).wrapping_mul(b as i16) as u16),
                    0xd8 => zip::<u8>(xd, s, |a, b| a.saturating_sub(b)),
                    0xd9 => zip::<u16>(xd, s, |a, b| a.saturating_sub(b)),
                    0xda => zip::<u8>(xd, s, |a, b| a.min(b)),
                    0xdb => xd & s,
                    0xdc => zip::<u8>(xd, s, |a, b| a.saturating_add(b)),
                    0xdd => zip::<u16>(xd, s, |a, b| a.saturating_add(b)),
                    0xde => zip::<u8>(xd, s, |a, b| a.max(b)),
                    0xdf => !xd & s,
                    0xe0 => zip::<u8>(xd, s, |a, b| ((a as u16 + b as u16 + 1) >> 1) as u8),
                    0xe3 => zip::<u16>(xd, s, |a, b| ((a as u32 + b as u32 + 1) >> 1) as u16),
                    0xe4 => zip::<u16>(xd, s, |a, b| ((a as u32 * b as u32) >> 16) as u16),
                    0xe5 => zip::<u16>(xd, s, |a, b| {
                        ((a as i16 as i32 * b as i16 as i32) >> 16) as u16
                    }),
                    0xe8 => zip::<u8>(xd, s, |a, b| (a as i8).saturating_sub(b as i8) as u8),
                    0xe9 => zip::<u16>(xd, s, |a, b| (a as i16).saturating_sub(b as i16) as u16),
                    0xea => zip::<u16>(xd, s, |a, b| (a as i16).min(b as i16) as u16),
                    0xeb => xd | s,
                    0xec => zip::<u8>(xd, s, |a, b| (a as i8).saturating_add(b as i8) as u8),
                    0xed => zip::<u16>(xd, s, |a, b| (a as i16).saturating_add(b as i16) as u16),
                    0xee => zip::<u16>(xd, s, |a, b| (a as i16).max(b as i16) as u16),
                    0xef => xd ^ s,
                    0xf4 => {
                        let mut r = 0;
                        for k in 0..2 {
                            u64::put(
                                &mut r,
                                k,
                                (u32::get(xd, 2 * k) as u64) * (u32::get(s, 2 * k) as u64),
                            );
                        }
                        r
                    }
                    0xf5 => {
                        let mut r = 0;
                        for k in 0..4 {
                            let lo = (u16::get(xd, 2 * k) as i16 as i32)
                                * (u16::get(s, 2 * k) as i16 as i32);
                            let hi = (u16::get(xd, 2 * k + 1) as i16 as i32)
                                * (u16::get(s, 2 * k + 1) as i16 as i32);
                            u32::put(&mut r, k, lo.wrapping_add(hi) as u32);
                        }
                        r
                    }
                    0xf6 => {
                        let mut r = 0;
                        for k in 0..2 {
                            let sum: u64 = (0..8)
                                .map(|j| {
                                    (u8::get(xd, 8 * k + j) as i32 - u8::get(s, 8 * k + j) as i32)
                                        .unsigned_abs() as u64
                                })
                                .sum();
                            u64::put(&mut r, k, sum);
                        }
                        r
                    }
                    0xf8 => zip::<u8>(xd, s, |a, b| a.wrapping_sub(b)),
                    0xf9 => zip::<u16>(xd, s, |a, b| a.wrapping_sub(b)),
                    0xfa => zip::<u32>(xd, s, |a, b| a.wrapping_sub(b)),
                    0xfb => zip::<u64>(xd, s, |a, b| a.wrapping_sub(b)),
                    0xfc => zip::<u8>(xd, s, |a, b| a.wrapping_add(b)),
                    0xfd => zip::<u16>(xd, s, |a, b| a.wrapping_add(b)),
                    _ => zip::<u32>(xd, s, |a, b| a.wrapping_add(b)),
                })
            }
            _ => return Ok(false),
        };
        if let Some(v) = res {
            self.xmm[d] = v;
        }
        Ok(true)
    }
}

fn allowed(low: u8, p: P) -> bool {
    use P::*;
    match low {
        0x10 | 0x11 | 0x51 | 0x58 | 0x59 | 0x5a | 0x5c..=0x5f | 0xc2 => true,
        0x12
        | 0x13
        | 0x14
        | 0x15
        | 0x16
        | 0x17
        | 0x28
        | 0x29
        | 0x2b
        | 0x2e
        | 0x2f
        | 0x50
        | 0x54..=0x57
        | 0xc6 => {
            matches!(p, N | O66)
        }
        0x2a | 0x2c | 0x2d => matches!(p, F3 | F2),
        0x52 | 0x53 => matches!(p, N | F3),
        0x5b => p != F2,
        0x6f | 0x7f | 0x7e => matches!(p, O66 | F3) && !(low == 0x7f && false),
        0x70 | 0xe6 => p != N,
        0x77 => p == N,
        0x60..=0x6e
        | 0x71..=0x76
        | 0xc4
        | 0xc5
        | 0xd1..=0xd6
        | 0xd7..=0xdf
        | 0xe0..=0xe5
        | 0xe7..=0xef
        | 0xf1..=0xf6
        | 0xf8..=0xfe => p == O66,
        _ => false,
    }
}

fn shl_r<T: Lane>(x: u128, c: u64, left: bool, width: u32) -> u128 {
    if c >= width as u64 {
        return 0;
    }
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        let o = if left { v << c } else { v >> c };
        put_u64::<T>(&mut r, k, o, width);
    }
    r
}

fn sra<T: Lane>(x: u128, c: u64, width: u32) -> u128 {
    let c = c.min(width as u64 - 1);
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        let sh = 64 - width;
        let o = (((v << sh) as i64) >> sh) >> c;
        put_u64::<T>(&mut r, k, o as u64, width);
    }
    r
}

fn lane_u64<T: Lane>(x: u128, k: usize) -> u64 {
    let w = 128 / T::N;
    ((x >> (k * w)) as u64) & if w == 64 { u64::MAX } else { (1u64 << w) - 1 }
}

fn put_u64<T: Lane>(r: &mut u128, k: usize, v: u64, width: u32) {
    let m = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    *r |= ((v & m) as u128) << (k * width as usize);
}
