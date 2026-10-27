use super::decode::*;
use super::exception::Exception;
use super::flags::*;
use super::sys::{CR0_EM, CR0_TS, CR4_OSFXSR};
use super::*;
#[cfg(target_os = "uefi")]
use crate::math::FloatExt;

/// A vector register viewed as `N` lanes of this type; lane 0 is the least significant.
trait Lane: Copy {
    const N: usize;
    fn get(v: u128, i: usize) -> Self;
    /// ORs into the lane, so `v` must start zeroed (or the lane must be known to be clear).
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

fn zip_lanes<T: Lane>(a: u128, b: u128, f: impl Fn(T, T) -> T) -> u128 {
    let mut r = 0;
    for i in 0..T::N {
        T::put(&mut r, i, f(T::get(a, i), T::get(b, i)));
    }
    r
}

/// punpckl*/punpckh*: alternates lanes of `a` and `b` taken from the low or high half of each.
fn interleave<T: Lane>(a: u128, b: u128, hi: bool) -> u128 {
    let mut r = 0;
    let off = if hi { T::N / 2 } else { 0 };
    for i in 0..T::N / 2 {
        T::put(&mut r, 2 * i, T::get(a, off + i));
        T::put(&mut r, 2 * i + 1, T::get(b, off + i));
    }
    r
}

fn f32_lanes(v: u128) -> [f32; 4] {
    [0, 1, 2, 3].map(|i| f32::from_bits(u32::get(v, i)))
}
fn from_f32_lanes(a: [f32; 4]) -> u128 {
    let mut r = 0;
    for (i, x) in a.iter().enumerate() {
        u32::put(&mut r, i, x.to_bits());
    }
    r
}
fn f64_lanes(v: u128) -> [f64; 2] {
    [0, 1].map(|i| f64::from_bits(u64::get(v, i)))
}
fn from_f64_lanes(a: [f64; 2]) -> u128 {
    a[0].to_bits() as u128 | ((a[1].to_bits() as u128) << 64)
}

/// packsswb/packuswb (`source_bits` 16) and packssdw (`source_bits` 32): saturating narrow of the
/// signed lanes of `a` into the low half of the result, then those of `b` into the high half.
fn pack_sat(a: u128, b: u128, source_bits: usize, signed_out: bool) -> u128 {
    let mut r = 0u128;
    let get = |v: u128, i: usize| -> i32 {
        if source_bits == 16 {
            u16::get(v, i) as i16 as i32
        } else {
            u32::get(v, i) as i32
        }
    };
    let n = 128 / source_bits;
    for k in 0..2 * n {
        let x = get(if k < n { a } else { b }, k % n);
        if source_bits == 16 {
            let y = if signed_out {
                x.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8 as u8
            } else {
                x.clamp(0, i32::from(u8::MAX)) as u8
            };
            u8::put(&mut r, k, y);
        } else {
            let y = if signed_out {
                x.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16 as u16
            } else {
                x.clamp(0, i32::from(u16::MAX)) as u16
            };
            u16::put(&mut r, k, y);
        }
    }
    r
}

/// MXCSR.RC field, bits 14:13: 0 nearest-even, 1 down, 2 up, 3 truncate.
const MXCSR_RC_SHIFT: u32 = 13;
const MXCSR_RC_MASK: u32 = 3;
/// 2^31 and 2^63 as f64; values in [-LIMIT, LIMIT) fit the integer type after rounding.
const I32_LIMIT: f64 = 2_147_483_648.0;
const I64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// cvtss2si/cvtsd2si and friends. NaN and out-of-range inputs give the "integer indefinite"
/// value (the minimum), as on hardware. The invalid-operation flag in MXCSR is not set.
/// `trunc` selects the cvtt* forms, which ignore MXCSR.RC.
fn cvt_i32(v: f64, trunc: bool, mxcsr: u32) -> i32 {
    let r = if trunc { v.trunc() } else { round_rc(v, mxcsr) };
    if r.is_nan() || !(-I32_LIMIT..I32_LIMIT).contains(&r) {
        i32::MIN
    } else {
        r as i32
    }
}

fn cvt_i64(v: f64, trunc: bool, mxcsr: u32) -> i64 {
    let r = if trunc { v.trunc() } else { round_rc(v, mxcsr) };
    if r.is_nan() || !(-I64_LIMIT..I64_LIMIT).contains(&r) {
        i64::MIN
    } else {
        r as i64
    }
}

fn round_rc(v: f64, mxcsr: u32) -> f64 {
    match (mxcsr >> MXCSR_RC_SHIFT) & MXCSR_RC_MASK {
        0 => v.round_ties_even(),
        1 => v.floor(),
        2 => v.ceil(),
        _ => v.trunc(),
    }
}

/// cmpps/cmppd/cmpss/cmpsd imm8 predicates 0-7: eq, lt, le, unord, neq, nlt, nle, ord. Only the
/// low three bits exist before AVX. The n* forms are true for unordered operands.
fn cmp_pred(predicate: u8, a: f64, b: f64) -> bool {
    let un = a.is_nan() || b.is_nan();
    match predicate & 7 {
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

/// The mandatory prefix that selects between the packed-single, packed-double, scalar-single and
/// scalar-double (or MMX vs. XMM integer) forms of an 0F opcode. `Np` = none, `P66` = operand-size
/// prefix, `F3`/`F2` = rep/repne; if several are present F3/F2 win (see `Insn::rep`).
#[derive(Clone, Copy, PartialEq)]
enum Prefix {
    Np,
    P66,
    F3,
    F2,
}

/// `Insn::op` is `(map << 8) | opcode_byte`; map 1 is the `0f xx` two-byte map.
const OPCODE_MAP_0F: u16 = 1;
const MODRM_MOD_REGISTER: u8 = 3;
const LOW_32: u128 = 0xffff_ffff;
const LOW_64: u128 = u64::MAX as u128;

impl Cpu {
    /// SDM SSE exception rules: #UD if CR0.EM is set or the OS has not enabled SSE (CR4.OSFXSR),
    /// #NM if CR0.TS is set (lazy FPU switching). #UD takes priority over #NM.
    fn sse_gate(&self) -> ExecResult<()> {
        if self.cr0 & CR0_EM != 0 || self.cr4 & CR4_OSFXSR == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.cr0 & CR0_TS != 0 {
            return Err(Exception::device_not_available());
        }
        Ok(())
    }

    /// One lane of sqrt/add/mul/sub/min/max/div, `a` = destination, `b` = source (sqrt uses only
    /// `b`). Computed on the host FPU in f64 and rounded to f32 for single precision; double
    /// rounding is harmless for these operations. Host default rounding applies: MXCSR.RC, DAZ,
    /// FTZ, the exception masks and the exception flags are not modelled. min/max return `b` when
    /// the operands are unordered or both zero, matching the SDM. NaN payloads and the NaN made
    /// by invalid operations come from the host FPU, not the x86 rules; single-precision signaling
    /// NaNs are quieted by the f32 to f64 widening.
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

    /// Executes an SSE/SSE2 instruction from the `0f xx` map. Returns `Ok(false)` for anything
    /// else, which the caller turns into #UD; on `Ok(true)` the caller sets rip to `next`.
    /// CPUID (`cpu/sys.rs`) advertises SSE and SSE2 and this covers those: no SSE3 and later, no
    /// VEX, and the MMX forms (no mandatory prefix on 0x60-0xff) are rejected except `emms`.
    ///
    /// Vector registers are `u128` in `self.xmm`. The `src!` operand is the whole register in the
    /// register form, so scalar arms truncate it to the lane they use.
    pub fn sse(&mut self, i: &Insn, next: u64) -> ExecResult<bool> {
        let map = i.op >> 8;
        if map != OPCODE_MAP_0F {
            return Ok(false);
        }
        let opcode = (i.op & 0xff) as u8;
        let prefix = match (i.rep, i.p66) {
            (1, _) => Prefix::F3,
            (2, _) => Prefix::F2,
            (_, true) => Prefix::P66,
            _ => Prefix::Np,
        };
        if !allowed(opcode, prefix) {
            return Ok(false);
        }
        // emms is x87 state management, so it uses the FPU's #NM rule, not the SSE one. ftw is the
        // abridged tag word of FXSAVE: 0 marks all eight registers empty.
        if opcode == 0x77 {
            self.fpu_check()?;
            self.ftw = 0;
            return Ok(true);
        }
        self.sse_gate()?;
        // ModRM.reg and .rm already include REX.R/REX.B. `dst_reg` is a general register for the
        // instructions that write one (cvt*2si, movmskp*, pmovmskb, pextrw).
        let dst_reg = i.reg as usize;
        let rm_reg = i.rm as usize;
        // Linear address of the memory operand, `None` for the register form (ModRM.mod == 3).
        let mem_addr = if i.md == MODRM_MOD_REGISTER {
            None
        } else {
            Some(self.linear_address(i, self.effective_address(i, next)))
        };
        macro_rules! src {
            ($size:expr) => {
                match mem_addr {
                    None => self.xmm[rm_reg],
                    Some(addr) => {
                        if $size == 16 {
                            self.read128(addr)?
                        } else {
                            self.read(addr, $size)? as u128
                        }
                    }
                }
            };
        }
        // #GP(0) for a misaligned 16-byte operand. Only the explicitly aligned moves (movaps,
        // movapd, movdqa, movntps/pd/dq) check this; hardware also faults on legacy-SSE
        // arithmetic with a misaligned m128 operand, which is not enforced here.
        macro_rules! aligned {
            () => {
                if let Some(addr) = mem_addr {
                    if addr & 15 != 0 {
                        return Err(Exception::general_protection(0));
                    }
                }
            };
        }
        let dst = self.xmm[dst_reg];
        let mxcsr = self.mxcsr;
        let result: Option<u128> = match opcode {
            // movups, movupd, movss, movsd load. The memory form of movss/movsd zeroes the rest of
            // the register; the register form merges into the low lane.
            0x10 => Some(match prefix {
                Prefix::Np | Prefix::P66 => src!(16),
                Prefix::F3 => {
                    if mem_addr.is_some() {
                        src!(4)
                    } else {
                        (dst & !LOW_32) | (self.xmm[rm_reg] & LOW_32)
                    }
                }
                Prefix::F2 => {
                    if mem_addr.is_some() {
                        src!(8)
                    } else {
                        (dst & !LOW_64) | (self.xmm[rm_reg] & LOW_64)
                    }
                }
            }),
            // stores: movups/movupd/movss/movsd (0x11), movaps/movapd (0x29), movntps/movntpd
            // (0x2b), movdqa/movdqu (0x7f), movntdq (0xe7). Scalar register destinations merge into
            // the low lane.
            0x11 | 0x29 | 0x2b | 0x7f | 0xe7 => {
                if matches!(opcode, 0x29 | 0x2b)
                    || (opcode == 0x7f && prefix == Prefix::P66)
                    || opcode == 0xe7
                {
                    aligned!();
                }
                let size = match (opcode, prefix) {
                    (0x11, Prefix::F3) => 4,
                    (0x11, Prefix::F2) => 8,
                    _ => 16,
                };
                match mem_addr {
                    Some(addr) => match size {
                        16 => self.write128(addr, dst)?,
                        s => self.write(addr, s, dst as u64)?,
                    },
                    None => {
                        let cur = self.xmm[rm_reg];
                        self.xmm[rm_reg] = match size {
                            4 => (cur & !LOW_32) | (dst & LOW_32),
                            8 => (cur & !LOW_64) | (dst & LOW_64),
                            _ => dst,
                        };
                    }
                }
                None
            }
            // movlps/movlpd (0x12) and movhps/movhpd (0x16) loads; the register forms are movhlps
            // and movlhps.
            0x12 | 0x16 => {
                let low_half = opcode == 0x12;
                let v = if mem_addr.is_none() {
                    if low_half {
                        self.xmm[rm_reg] >> 64
                    } else {
                        self.xmm[rm_reg] & LOW_64
                    }
                } else {
                    src!(8)
                } & LOW_64;
                Some(if low_half {
                    (dst & !LOW_64) | v
                } else {
                    (dst & LOW_64) | (v << 64)
                })
            }
            // movlps/movlpd and movhps/movhpd stores; memory destination only.
            0x13 | 0x17 => {
                let addr = mem_addr.ok_or_else(Exception::invalid_opcode)?;
                let v = if opcode == 0x13 {
                    dst as u64
                } else {
                    (dst >> 64) as u64
                };
                self.write(addr, 8, v)?;
                None
            }
            // unpcklps/unpcklpd (0x14), unpckhps/unpckhpd (0x15)
            0x14 | 0x15 => {
                let s = src!(16);
                let hi = opcode == 0x15;
                Some(if prefix == Prefix::Np {
                    interleave::<u32>(dst, s, hi)
                } else {
                    interleave::<u64>(dst, s, hi)
                })
            }
            // movaps/movapd load
            0x28 => {
                aligned!();
                Some(src!(16))
            }
            // cvtsi2ss/cvtsi2sd from a 32-bit or (REX.W) 64-bit integer. The 64-bit case goes
            // through f64 before narrowing to f32, so cvtsi2ss of values above 2^53 can round
            // twice.
            0x2a => {
                let size = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, size, true),
                    Some(addr) => self.read(addr, size)?,
                };
                let f = if size == 8 {
                    v as i64 as f64
                } else {
                    v as u32 as i32 as f64
                };
                Some(if prefix == Prefix::F3 {
                    (dst & !LOW_32) | (f as f32).to_bits() as u128
                } else {
                    (dst & !LOW_64) | f.to_bits() as u128
                })
            }
            // cvttss2si/cvttsd2si (0x2c, truncates) and cvtss2si/cvtsd2si (0x2d, rounds per
            // MXCSR.RC)
            0x2c | 0x2d => {
                let size = if prefix == Prefix::F3 { 4 } else { 8 };
                let v = src!(size) & if size == 4 { LOW_32 } else { LOW_64 };
                let f = if size == 4 {
                    f32::from_bits(v as u32) as f64
                } else {
                    f64::from_bits(v as u64)
                };
                let truncate = opcode == 0x2c;
                if i.opsize == 8 {
                    self.gpr[dst_reg] = cvt_i64(f, truncate, mxcsr) as u64;
                } else {
                    self.gpr[dst_reg] = cvt_i32(f, truncate, mxcsr) as u32 as u64;
                }
                None
            }
            // ucomiss/ucomisd, comiss/comisd. Unordered sets ZF, PF and CF; OF, SF and AF are
            // cleared. The invalid-operation exception comiss raises for QNaN is not modelled.
            0x2e | 0x2f => {
                let (a, b) = if prefix == Prefix::Np {
                    let s = src!(4) as u32;
                    (f32::from_bits(dst as u32) as f64, f32::from_bits(s) as f64)
                } else {
                    let s = src!(8) as u64;
                    (f64::from_bits(dst as u64), f64::from_bits(s))
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
            // movmskps/movmskpd: sign bits into a general register
            0x50 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let x = self.xmm[rm_reg];
                self.gpr[dst_reg] = if prefix == Prefix::Np {
                    (0..4).fold(0, |m, k| m | (((u32::get(x, k) >> 31) as u64) << k))
                } else {
                    (0..2).fold(0, |m, k| m | ((u64::get(x, k) >> 63) << k))
                };
                None
            }
            // sqrt, add, mul, sub, min, div, max in packed and scalar forms; scalar forms merge
            // into the low lane.
            0x51 | 0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f => Some(match prefix {
                Prefix::Np => {
                    let s = f32_lanes(src!(16));
                    let a = f32_lanes(dst);
                    from_f32_lanes(
                        [0, 1, 2, 3]
                            .map(|k| self.fp_binop(opcode, a[k] as f64, s[k] as f64) as f32),
                    )
                }
                Prefix::P66 => {
                    let s = f64_lanes(src!(16));
                    let a = f64_lanes(dst);
                    from_f64_lanes([0, 1].map(|k| self.fp_binop(opcode, a[k], s[k])))
                }
                Prefix::F3 => {
                    let s = f32::from_bits(src!(4) as u32);
                    let r =
                        self.fp_binop(opcode, f32::from_bits(dst as u32) as f64, s as f64) as f32;
                    (dst & !LOW_32) | r.to_bits() as u128
                }
                Prefix::F2 => {
                    let s = f64::from_bits(src!(8) as u64);
                    let r = self.fp_binop(opcode, f64::from_bits(dst as u64), s);
                    (dst & !LOW_64) | r.to_bits() as u128
                }
            }),
            // rsqrtps/rsqrtss (0x52), rcpps/rcpss (0x53). Computed at full f32 precision; hardware
            // returns an approximation with relative error up to 1.5 * 2^-12, so low bits differ.
            0x52 | 0x53 => {
                let f = |x: f32| {
                    if opcode == 0x52 {
                        1.0 / x.sqrt()
                    } else {
                        1.0 / x
                    }
                };
                Some(if prefix == Prefix::F3 {
                    let s = f32::from_bits(src!(4) as u32);
                    (dst & !LOW_32) | f(s).to_bits() as u128
                } else {
                    from_f32_lanes(f32_lanes(src!(16)).map(f))
                })
            }
            // andps/andpd, andnps/andnpd, orps/orpd, xorps/xorpd
            0x54..=0x57 => {
                let s = src!(16);
                Some(match opcode {
                    0x54 => dst & s,
                    0x55 => !dst & s,
                    0x56 => dst | s,
                    _ => dst ^ s,
                })
            }
            // cvtps2pd, cvtpd2ps, cvtss2sd, cvtsd2ss
            0x5a => Some(match prefix {
                Prefix::Np => {
                    let s = f32_lanes(src!(8));
                    from_f64_lanes([s[0] as f64, s[1] as f64])
                }
                Prefix::P66 => {
                    let s = f64_lanes(src!(16));
                    from_f32_lanes([s[0] as f32, s[1] as f32, 0.0, 0.0])
                }
                Prefix::F3 => {
                    let s = f32::from_bits(src!(4) as u32);
                    (dst & !LOW_64) | (s as f64).to_bits() as u128
                }
                Prefix::F2 => {
                    let s = f64::from_bits(src!(8) as u64);
                    (dst & !LOW_32) | (s as f32).to_bits() as u128
                }
            }),
            // cvtdq2ps, cvtps2dq, cvttps2dq. Deviation: cvtdq2ps always rounds to nearest, where
            // hardware honors MXCSR.RC; likewise cvtpd2ps/cvtsd2ss (0x5a) and cvtsi2ss/sd (0x2a).
            0x5b => {
                let s = src!(16);
                Some(match prefix {
                    Prefix::Np => {
                        from_f32_lanes([0, 1, 2, 3].map(|k| u32::get(s, k) as i32 as f32))
                    }
                    _ => {
                        let truncate = prefix == Prefix::F3;
                        let mut r = 0;
                        for (k, f) in f32_lanes(s).iter().enumerate() {
                            u32::put(&mut r, k, cvt_i32(*f as f64, truncate, mxcsr) as u32);
                        }
                        r
                    }
                })
            }
            // 0x60-0x62 punpckl{bw,wd,dq}, 0x63 packsswb, 0x64-0x66 pcmpgt{b,w,d}, 0x67 packuswb,
            // 0x68-0x6a punpckh{bw,wd,dq}, 0x6b packssdw, 0x6c/0x6d punpcklqdq/punpckhqdq
            0x60..=0x6d => {
                let s = src!(16);
                Some(match opcode {
                    0x60 => interleave::<u8>(dst, s, false),
                    0x61 => interleave::<u16>(dst, s, false),
                    0x62 => interleave::<u32>(dst, s, false),
                    0x63 => pack_sat(dst, s, 16, true),
                    0x64 => {
                        zip_lanes::<u8>(dst, s, |a, b| if (a as i8) > (b as i8) { 0xff } else { 0 })
                    }
                    0x65 => {
                        zip_lanes::<u16>(
                            dst,
                            s,
                            |a, b| if (a as i16) > (b as i16) { 0xffff } else { 0 },
                        )
                    }
                    0x66 => {
                        zip_lanes::<u32>(
                            dst,
                            s,
                            |a, b| if (a as i32) > (b as i32) { u32::MAX } else { 0 },
                        )
                    }
                    0x67 => pack_sat(dst, s, 16, false),
                    0x68 => interleave::<u8>(dst, s, true),
                    0x69 => interleave::<u16>(dst, s, true),
                    0x6a => interleave::<u32>(dst, s, true),
                    0x6b => pack_sat(dst, s, 32, true),
                    0x6c => interleave::<u64>(dst, s, false),
                    _ => interleave::<u64>(dst, s, true),
                })
            }
            // movd/movq xmm, r/m; zero-extends into the whole register
            0x6e => {
                let size = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, size, true),
                    Some(addr) => self.read(addr, size)?,
                };
                Some(v as u128)
            }
            // movdqa (alignment checked) and movdqu load
            0x6f => {
                if prefix == Prefix::P66 {
                    aligned!();
                }
                Some(src!(16))
            }
            // pshufd, pshuflw, pshufhw: imm8 holds four 2-bit source lane indices, lowest field
            // first; pshuflw/pshufhw copy the other half of the register unchanged.
            0x70 => {
                let s = src!(16);
                let imm8 = i.imm as u8;
                Some(match prefix {
                    Prefix::P66 => {
                        let mut r = 0;
                        for k in 0..4 {
                            u32::put(&mut r, k, u32::get(s, ((imm8 >> (2 * k)) & 3) as usize));
                        }
                        r
                    }
                    Prefix::F2 => {
                        let mut r = s & !LOW_64;
                        for k in 0..4 {
                            u16::put(&mut r, k, u16::get(s, ((imm8 >> (2 * k)) & 3) as usize));
                        }
                        r
                    }
                    _ => {
                        let mut r = s & LOW_64;
                        for k in 0..4 {
                            u16::put(
                                &mut r,
                                4 + k,
                                u16::get(s, 4 + ((imm8 >> (2 * k)) & 3) as usize),
                            );
                        }
                        r
                    }
                })
            }
            // Shifts by immediate; ModRM.reg selects the operation: /2 psrl, /4 psra, /6 psll, and
            // for 0x73 /3 psrldq, /7 pslldq (whole-register byte shifts; a count above 15 clears
            // the register).
            0x71..=0x73 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let x = self.xmm[rm_reg];
                let count = i.imm as u8 as u64;
                let group = i.reg & 7;
                let r = match (opcode, group) {
                    (0x71, 2) => shift_logical::<u16>(x, count, false, 16),
                    (0x71, 4) => sra::<u16>(x, count, 16),
                    (0x71, 6) => shift_logical::<u16>(x, count, true, 16),
                    (0x72, 2) => shift_logical::<u32>(x, count, false, 32),
                    (0x72, 4) => sra::<u32>(x, count, 32),
                    (0x72, 6) => shift_logical::<u32>(x, count, true, 32),
                    (0x73, 2) => shift_logical::<u64>(x, count, false, 64),
                    (0x73, 6) => shift_logical::<u64>(x, count, true, 64),
                    (0x73, 3) => {
                        if count > 15 {
                            0
                        } else {
                            x >> (count * 8)
                        }
                    }
                    (0x73, 7) => {
                        if count > 15 {
                            0
                        } else {
                            x << (count * 8)
                        }
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                };
                self.xmm[rm_reg] = r;
                None
            }
            // pcmpeqb
            0x74 => {
                let s = src!(16);
                Some(zip_lanes::<u8>(
                    dst,
                    s,
                    |a, b| if a == b { 0xff } else { 0 },
                ))
            }
            // pcmpeqw
            0x75 => {
                let s = src!(16);
                Some(zip_lanes::<u16>(
                    dst,
                    s,
                    |a, b| if a == b { 0xffff } else { 0 },
                ))
            }
            // pcmpeqd
            0x76 => {
                let s = src!(16);
                Some(zip_lanes::<u32>(
                    dst,
                    s,
                    |a, b| if a == b { u32::MAX } else { 0 },
                ))
            }
            // movq xmm, xmm/m64 (F3, zero-extends) and movd/movq r/m, xmm (66)
            0x7e => {
                if prefix == Prefix::F3 {
                    Some(src!(8) & LOW_64)
                } else {
                    let size = if i.opsize == 8 { 8 } else { 4 };
                    let v = dst as u64 & if size == 8 { u64::MAX } else { 0xffff_ffff };
                    match mem_addr {
                        None => self.write_gpr(i.rm, size, true, v),
                        Some(addr) => self.write(addr, size, v)?,
                    }
                    None
                }
            }
            // cmpps/cmppd/cmpss/cmpsd: all ones or zero per lane
            0xc2 => {
                let predicate = i.imm as u8;
                Some(match prefix {
                    Prefix::Np => {
                        let (a, s) = (f32_lanes(dst), f32_lanes(src!(16)));
                        let mut r = 0;
                        for k in 0..4 {
                            u32::put(
                                &mut r,
                                k,
                                if cmp_pred(predicate, a[k] as f64, s[k] as f64) {
                                    u32::MAX
                                } else {
                                    0
                                },
                            );
                        }
                        r
                    }
                    Prefix::P66 => {
                        let (a, s) = (f64_lanes(dst), f64_lanes(src!(16)));
                        let mut r = 0;
                        for k in 0..2 {
                            u64::put(
                                &mut r,
                                k,
                                if cmp_pred(predicate, a[k], s[k]) {
                                    u64::MAX
                                } else {
                                    0
                                },
                            );
                        }
                        r
                    }
                    Prefix::F3 => {
                        let s = f32::from_bits(src!(4) as u32) as f64;
                        let matched = cmp_pred(predicate, f32::from_bits(dst as u32) as f64, s);
                        (dst & !LOW_32) | if matched { LOW_32 } else { 0 }
                    }
                    Prefix::F2 => {
                        let s = f64::from_bits(src!(8) as u64);
                        let matched = cmp_pred(predicate, f64::from_bits(dst as u64), s);
                        (dst & !LOW_64) | if matched { LOW_64 } else { 0 }
                    }
                })
            }
            // pinsrw: imm8 & 7 selects the word
            0xc4 => {
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, 2, true),
                    Some(addr) => self.read(addr, 2)?,
                } as u16;
                let k = (i.imm as usize) & 7;
                let sh = k * 16;
                Some((dst & !(0xffffu128 << sh)) | ((v as u128) << sh))
            }
            // pextrw: register source only; zero-extends into the general register
            0xc5 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let k = (i.imm as usize) & 7;
                self.gpr[dst_reg] = u16::get(self.xmm[rm_reg], k) as u64;
                None
            }
            // shufps: result lanes 0-1 come from dst, 2-3 from the source. shufpd: imm8 bit 0 picks
            // the low lane from dst, bit 1 the high lane from the source.
            0xc6 => {
                let s = src!(16);
                let imm8 = i.imm as u8 as usize;
                Some(if prefix == Prefix::Np {
                    let mut r = 0;
                    u32::put(&mut r, 0, u32::get(dst, imm8 & 3));
                    u32::put(&mut r, 1, u32::get(dst, (imm8 >> 2) & 3));
                    u32::put(&mut r, 2, u32::get(s, (imm8 >> 4) & 3));
                    u32::put(&mut r, 3, u32::get(s, (imm8 >> 6) & 3));
                    r
                } else {
                    u64::get(dst, imm8 & 1) as u128 | ((u64::get(s, (imm8 >> 1) & 1) as u128) << 64)
                })
            }
            // Shifts by the low 64 bits of an xmm/m128 count: psrlw/d/q (0xd1-0xd3), psraw/d
            // (0xe1/0xe2), psllw/d/q (0xf1-0xf3). A count with any upper bit set is treated as out
            // of range.
            0xd1..=0xd3 | 0xe1 | 0xe2 | 0xf1..=0xf3 => {
                let count_src = src!(16);
                let count = if count_src >> 64 != 0 {
                    u64::MAX
                } else {
                    count_src as u64
                };
                Some(match opcode {
                    0xd1 => shift_logical::<u16>(dst, count, false, 16),
                    0xd2 => shift_logical::<u32>(dst, count, false, 32),
                    0xd3 => shift_logical::<u64>(dst, count, false, 64),
                    0xe1 => sra::<u16>(dst, count, 16),
                    0xe2 => sra::<u32>(dst, count, 32),
                    0xf1 => shift_logical::<u16>(dst, count, true, 16),
                    0xf2 => shift_logical::<u32>(dst, count, true, 32),
                    _ => shift_logical::<u64>(dst, count, true, 64),
                })
            }
            // movq xmm/m64, xmm; the register form zeroes the upper half
            0xd6 => {
                let v = dst & LOW_64;
                match mem_addr {
                    None => self.xmm[rm_reg] = v,
                    Some(addr) => self.write(addr, 8, v as u64)?,
                }
                None
            }
            // pmovmskb
            0xd7 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let x = self.xmm[rm_reg];
                self.gpr[dst_reg] =
                    (0..16).fold(0, |m, k| m | (((u8::get(x, k) >> 7) as u64) << k));
                None
            }
            // cvtdq2pd (F3), cvtpd2dq (F2, MXCSR.RC), cvttpd2dq (66, truncates). The integer forms
            // fill only the low two dwords and zero the upper half.
            0xe6 => {
                let truncate = prefix == Prefix::P66;
                Some(match prefix {
                    Prefix::F3 => {
                        let s = src!(8);
                        from_f64_lanes([u32::get(s, 0) as i32 as f64, u32::get(s, 1) as i32 as f64])
                    }
                    _ => {
                        let s = f64_lanes(src!(16));
                        let mut r = 0;
                        u32::put(&mut r, 0, cvt_i32(s[0], truncate, mxcsr) as u32);
                        u32::put(&mut r, 1, cvt_i32(s[1], truncate, mxcsr) as u32);
                        r
                    }
                })
            }
            // SSE2 integer arithmetic and logic, 66-prefixed. By opcode: d4 paddq, d5 pmullw, d8/d9
            // psubusb/w, da pminub, db pand, dc/dd paddusb/w, de pmaxub, df pandn, e0 pavgb, e3
            // pavgw, e4 pmulhuw, e5 pmulhw, e8/e9 psubsb/w, ea pminsw, eb por, ec/ed paddsb/w, ee
            // pmaxsw, ef pxor, f4 pmuludq, f5 pmaddwd, f6 psadbw, f8-fb psubb/w/d/q, fc-fe
            // paddb/w/d.
            0xd4
            | 0xd5
            | 0xd8..=0xdf
            | 0xe0
            | 0xe3..=0xe5
            | 0xe8..=0xef
            | 0xf4..=0xf6
            | 0xf8..=0xfe => {
                let s = src!(16);
                Some(match opcode {
                    0xd4 => zip_lanes::<u64>(dst, s, |a, b| a.wrapping_add(b)),
                    0xd5 => {
                        zip_lanes::<u16>(dst, s, |a, b| (a as i16).wrapping_mul(b as i16) as u16)
                    }
                    0xd8 => zip_lanes::<u8>(dst, s, |a, b| a.saturating_sub(b)),
                    0xd9 => zip_lanes::<u16>(dst, s, |a, b| a.saturating_sub(b)),
                    0xda => zip_lanes::<u8>(dst, s, |a, b| a.min(b)),
                    0xdb => dst & s,
                    0xdc => zip_lanes::<u8>(dst, s, |a, b| a.saturating_add(b)),
                    0xdd => zip_lanes::<u16>(dst, s, |a, b| a.saturating_add(b)),
                    0xde => zip_lanes::<u8>(dst, s, |a, b| a.max(b)),
                    0xdf => !dst & s,
                    0xe0 => zip_lanes::<u8>(dst, s, |a, b| ((a as u16 + b as u16 + 1) >> 1) as u8),
                    0xe3 => {
                        zip_lanes::<u16>(dst, s, |a, b| ((a as u32 + b as u32 + 1) >> 1) as u16)
                    }
                    0xe4 => zip_lanes::<u16>(dst, s, |a, b| ((a as u32 * b as u32) >> 16) as u16),
                    0xe5 => zip_lanes::<u16>(dst, s, |a, b| {
                        ((a as i16 as i32 * b as i16 as i32) >> 16) as u16
                    }),
                    0xe8 => zip_lanes::<u8>(dst, s, |a, b| (a as i8).saturating_sub(b as i8) as u8),
                    0xe9 => {
                        zip_lanes::<u16>(dst, s, |a, b| (a as i16).saturating_sub(b as i16) as u16)
                    }
                    0xea => zip_lanes::<u16>(dst, s, |a, b| (a as i16).min(b as i16) as u16),
                    0xeb => dst | s,
                    0xec => zip_lanes::<u8>(dst, s, |a, b| (a as i8).saturating_add(b as i8) as u8),
                    0xed => {
                        zip_lanes::<u16>(dst, s, |a, b| (a as i16).saturating_add(b as i16) as u16)
                    }
                    0xee => zip_lanes::<u16>(dst, s, |a, b| (a as i16).max(b as i16) as u16),
                    0xef => dst ^ s,
                    0xf4 => {
                        let mut r = 0;
                        for k in 0..2 {
                            u64::put(
                                &mut r,
                                k,
                                (u32::get(dst, 2 * k) as u64) * (u32::get(s, 2 * k) as u64),
                            );
                        }
                        r
                    }
                    0xf5 => {
                        let mut r = 0;
                        for k in 0..4 {
                            let lo = (u16::get(dst, 2 * k) as i16 as i32)
                                * (u16::get(s, 2 * k) as i16 as i32);
                            let hi = (u16::get(dst, 2 * k + 1) as i16 as i32)
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
                                    (u8::get(dst, 8 * k + j) as i32 - u8::get(s, 8 * k + j) as i32)
                                        .unsigned_abs() as u64
                                })
                                .sum();
                            u64::put(&mut r, k, sum);
                        }
                        r
                    }
                    0xf8 => zip_lanes::<u8>(dst, s, |a, b| a.wrapping_sub(b)),
                    0xf9 => zip_lanes::<u16>(dst, s, |a, b| a.wrapping_sub(b)),
                    0xfa => zip_lanes::<u32>(dst, s, |a, b| a.wrapping_sub(b)),
                    0xfb => zip_lanes::<u64>(dst, s, |a, b| a.wrapping_sub(b)),
                    0xfc => zip_lanes::<u8>(dst, s, |a, b| a.wrapping_add(b)),
                    0xfd => zip_lanes::<u16>(dst, s, |a, b| a.wrapping_add(b)),
                    _ => zip_lanes::<u32>(dst, s, |a, b| a.wrapping_add(b)),
                })
            }
            _ => return Ok(false),
        };
        if let Some(v) = result {
            self.xmm[dst_reg] = v;
        }
        Ok(true)
    }
}

/// Which mandatory prefixes each implemented opcode accepts. This is the only prefix validation:
/// `sse` assumes any (opcode, prefix) pair that gets past here is handled by its arm.
fn allowed(opcode: u8, prefix: Prefix) -> bool {
    use Prefix::*;
    match opcode {
        // movups/movss/movupd/movsd (load and store), sqrt, add, mul, cvtps2pd family, sub, min,
        // div, max, cmp: all four prefixes
        0x10 | 0x11 | 0x51 | 0x58 | 0x59 | 0x5a | 0x5c..=0x5f | 0xc2 => true,
        // movlps/movlpd, movhps/movhpd (and movhlps/movlhps), unpck, movaps/movapd, movnt,
        // (u)comiss/sd, movmsk, and/andn/or/xor, shufps/shufpd: packed only
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
            matches!(prefix, Np | P66)
        }
        // cvtsi2ss/sd, cvttss2si/cvttsd2si, cvtss2si/cvtsd2si: scalar only
        0x2a | 0x2c | 0x2d => matches!(prefix, F3 | F2),
        // rsqrtps/rsqrtss, rcpps/rcpss
        0x52 | 0x53 => matches!(prefix, Np | F3),
        // cvtdq2ps, cvtps2dq, cvttps2dq
        0x5b => prefix != F2,
        // movdqa/movdqu load, movdqa/movdqu store, movd/movq store and movq load
        0x6f | 0x7f | 0x7e => matches!(prefix, P66 | F3),
        // pshufd/pshufhw/pshuflw, cvtdq2pd/cvtpd2dq/cvttpd2dq
        0x70 | 0xe6 => prefix != Np,
        // emms (the only MMX instruction implemented)
        0x77 => prefix == Np,
        // SSE2 integer instructions, 66-prefixed XMM forms only (the unprefixed ones are MMX)
        0x60..=0x6e
        | 0x71..=0x76
        | 0xc4
        | 0xc5
        | 0xd1..=0xd6
        | 0xd7..=0xdf
        | 0xe0..=0xe5
        | 0xe7..=0xef
        | 0xf1..=0xf6
        | 0xf8..=0xfe => prefix == P66,
        _ => false,
    }
}

/// psrl*/psll* on every lane of type `T` (`width` bits). Counts of `width` or more clear the
/// register, as the SDM specifies; Rust's own shifts would panic or wrap there.
fn shift_logical<T: Lane>(x: u128, count: u64, left: bool, width: u32) -> u128 {
    if count >= width as u64 {
        return 0;
    }
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        let shifted = if left { v << count } else { v >> count };
        put_u64::<T>(&mut r, k, shifted, width);
    }
    r
}

/// psra*: arithmetic right shift. Counts of `width` or more behave as `width - 1` (all sign bits).
fn sra<T: Lane>(x: u128, count: u64, width: u32) -> u128 {
    let count = count.min(width as u64 - 1);
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        // Left-align the lane in an i64 so the arithmetic shift sign-extends it.
        let align = 64 - width;
        let shifted = (((v << align) as i64) >> align) >> count;
        put_u64::<T>(&mut r, k, shifted as u64, width);
    }
    r
}

/// Lane `k` zero-extended to u64; `Lane::get` returns the lane type itself, which the generic
/// shift code cannot do arithmetic on.
fn lane_u64<T: Lane>(x: u128, k: usize) -> u64 {
    let w = 128 / T::N;
    ((x >> (k * w)) as u64) & if w == 64 { u64::MAX } else { (1u64 << w) - 1 }
}

/// ORs the low `width` bits of `v` into lane `k` of `r`.
fn put_u64<T: Lane>(r: &mut u128, k: usize, v: u64, width: u32) {
    let m = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    *r |= ((v & m) as u128) << (k * width as usize);
}
