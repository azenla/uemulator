//! SSE3, SSSE3, SSE4.1, SSE4.2, AES-NI and PCLMULQDQ for the interpreter, as a child of `sse` so it
//! shares that file's soft-float engine and operand helpers. Dispatch comes from `Cpu::sse`:
//! everything outside the `0F` map and every `0F` opcode that `sse::allowed` rejects arrives here.
//!
//! Only the legacy (non-VEX) encodings exist, so every vector instruction needs its mandatory
//! prefix and a 16-byte aligned `m128` operand except where noted. All of them obey the usual SSE
//! enable rules (`Cpu::sse_gate`). CRC32 and MOVBE are integer instructions that live in the same
//! opcode map and skip the gate.
//!
//! The MMX-register forms of SSSE3 (`0F 38 xx` / `0F 3A 0F` without a prefix) are deliberately
//! left as #UD (logged once as unimplemented): glibc and the kernel only use the XMM forms, and
//! every MMX form would need its own lane layout. CPUID still advertises SSSE3, as the bit
//! describes the XMM forms for all software that matters.

use super::*;

/// Opcode maps of `Insn::op` (the high byte) that this file adds to the `0F` map of `sse.rs`.
const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;

/// The operation of an instruction, decoded from (map, opcode, prefix).
#[derive(Clone, Copy)]
enum Op {
    Movddup,
    Movsldup,
    Movshdup,
    Lddqu,
    /// addsubps/addsubpd: subtract in the even lanes, add in the odd ones.
    AddSub {
        double: bool,
    },
    /// haddps/haddpd/hsubps/hsubpd: pairwise sums or differences of adjacent lanes.
    Horizontal {
        double: bool,
        subtract: bool,
    },
    /// `dst = f(dst, src)` on whole registers.
    Binary(fn(u128, u128) -> u128),
    /// `dst = f(src)`: the old destination is ignored.
    Unary(fn(u128) -> u128),
    /// Binary with the imm8 that follows the ModRM bytes.
    WithImm(fn(u128, u128, u8) -> u128),
}

/// How an operation reads its second operand.
#[derive(Clone, Copy)]
enum Source {
    /// 128 bits; a memory operand must be 16-byte aligned.
    Aligned,
    /// 128 bits; any alignment.
    Unaligned,
    /// The low `n` bytes of the register, or `n` bytes of memory (no alignment requirement).
    Low(u8),
}

impl Op {
    fn source(self) -> Source {
        match self {
            Op::Movddup => Source::Low(8),
            Op::Lddqu => Source::Unaligned,
            _ => Source::Aligned,
        }
    }

    /// Instructions that exist only with a memory operand: the register form is #UD.
    fn memory_only(self) -> bool {
        matches!(self, Op::Lddqu)
    }
}

fn decode_op(map: u16, opcode: u8, prefix: Prefix) -> Option<Op> {
    use Prefix::*;
    Some(match (map, opcode, prefix) {
        (OPCODE_MAP_0F, 0x12, F2) => Op::Movddup,
        (OPCODE_MAP_0F, 0x12, F3) => Op::Movsldup,
        (OPCODE_MAP_0F, 0x16, F3) => Op::Movshdup,
        (OPCODE_MAP_0F, 0xf0, F2) => Op::Lddqu,
        (OPCODE_MAP_0F, 0xd0, P66) => Op::AddSub { double: true },
        (OPCODE_MAP_0F, 0xd0, F2) => Op::AddSub { double: false },
        (OPCODE_MAP_0F, 0x7c, P66) => Op::Horizontal {
            double: true,
            subtract: false,
        },
        (OPCODE_MAP_0F, 0x7c, F2) => Op::Horizontal {
            double: false,
            subtract: false,
        },
        (OPCODE_MAP_0F, 0x7d, P66) => Op::Horizontal {
            double: true,
            subtract: true,
        },
        (OPCODE_MAP_0F, 0x7d, F2) => Op::Horizontal {
            double: false,
            subtract: true,
        },
        _ => return decode_integer_op(map, opcode, prefix),
    })
}

/// The packed-integer instructions of the 66-prefixed `0F 38` and `0F 3A` maps. Without the 66
/// prefix these opcodes are the MMX forms (SSSE3) or other instructions, so no match.
fn decode_integer_op(map: u16, opcode: u8, prefix: Prefix) -> Option<Op> {
    if prefix != Prefix::P66 {
        return None;
    }
    Some(match (map, opcode) {
        (OPCODE_MAP_0F38, 0x00) => Op::Binary(pshufb),
        (OPCODE_MAP_0F38, 0x01) => {
            Op::Binary(|a, b| horizontal_integer::<u16>(a, b, u16::wrapping_add))
        }
        (OPCODE_MAP_0F38, 0x02) => {
            Op::Binary(|a, b| horizontal_integer::<u32>(a, b, u32::wrapping_add))
        }
        (OPCODE_MAP_0F38, 0x03) => Op::Binary(|a, b| {
            horizontal_integer::<u16>(a, b, |x, y| (x as i16).saturating_add(y as i16) as u16)
        }),
        (OPCODE_MAP_0F38, 0x04) => Op::Binary(pmaddubsw),
        (OPCODE_MAP_0F38, 0x05) => {
            Op::Binary(|a, b| horizontal_integer::<u16>(a, b, u16::wrapping_sub))
        }
        (OPCODE_MAP_0F38, 0x06) => {
            Op::Binary(|a, b| horizontal_integer::<u32>(a, b, u32::wrapping_sub))
        }
        (OPCODE_MAP_0F38, 0x07) => Op::Binary(|a, b| {
            horizontal_integer::<u16>(a, b, |x, y| (x as i16).saturating_sub(y as i16) as u16)
        }),
        (OPCODE_MAP_0F38, 0x08) => Op::Binary(|a, b| {
            zip_lanes::<u8>(a, b, |x, y| {
                apply_sign((x as i8) as i32, (y as i8) as i32) as u8
            })
        }),
        (OPCODE_MAP_0F38, 0x09) => Op::Binary(|a, b| {
            zip_lanes::<u16>(a, b, |x, y| {
                apply_sign((x as i16) as i32, (y as i16) as i32) as u16
            })
        }),
        (OPCODE_MAP_0F38, 0x0a) => {
            Op::Binary(|a, b| zip_lanes::<u32>(a, b, |x, y| apply_sign(x as i32, y as i32) as u32))
        }
        (OPCODE_MAP_0F38, 0x0b) => Op::Binary(|a, b| {
            zip_lanes::<u16>(a, b, |x, y| {
                let product = (x as i16 as i32) * (y as i16 as i32);
                (((product >> 14) + 1) >> 1) as u16
            })
        }),
        (OPCODE_MAP_0F38, 0x1c) => {
            Op::Unary(|b| zip_lanes::<u8>(b, 0, |x, _| (x as i8).unsigned_abs()))
        }
        (OPCODE_MAP_0F38, 0x1d) => {
            Op::Unary(|b| zip_lanes::<u16>(b, 0, |x, _| (x as i16).unsigned_abs()))
        }
        (OPCODE_MAP_0F38, 0x1e) => {
            Op::Unary(|b| zip_lanes::<u32>(b, 0, |x, _| (x as i32).unsigned_abs()))
        }
        (OPCODE_MAP_0F3A, 0x0f) => Op::WithImm(palignr),
        _ => return None,
    })
}

/// pshufb: each result byte is the destination byte selected by the low four bits of the
/// source byte, or zero when the source byte's top bit is set.
fn pshufb(dst: u128, selector: u128) -> u128 {
    const ZEROING_BIT: u8 = 0x80;
    const INDEX_MASK: u8 = 0x0f;
    zip_lanes::<u8>(dst, selector, |_, select| {
        if select & ZEROING_BIT != 0 {
            0
        } else {
            u8::get(dst, (select & INDEX_MASK) as usize)
        }
    })
}

/// phadd*/phsub*: `combine` applied to each (even, odd) lane pair of the destination, giving the
/// low half of the result, then to those of the source, giving the high half.
fn horizontal_integer<T: Lane>(dst: u128, src: u128, combine: impl Fn(T, T) -> T) -> u128 {
    let half = T::N / 2;
    let mut result = 0;
    for pair in 0..half {
        T::put(
            &mut result,
            pair,
            combine(T::get(dst, 2 * pair), T::get(dst, 2 * pair + 1)),
        );
        T::put(
            &mut result,
            half + pair,
            combine(T::get(src, 2 * pair), T::get(src, 2 * pair + 1)),
        );
    }
    result
}

/// pmaddubsw: unsigned destination bytes times signed source bytes, adjacent products added with
/// signed 16-bit saturation.
fn pmaddubsw(dst: u128, src: u128) -> u128 {
    let mut result = 0;
    for word in 0..8 {
        let product = |byte: usize| u8::get(dst, byte) as i32 * (u8::get(src, byte) as i8) as i32;
        let sum = product(2 * word) + product(2 * word + 1);
        u16::put(
            &mut result,
            word,
            sum.clamp(i16::MIN as i32, i16::MAX as i32) as u16,
        );
    }
    result
}

/// psign*: the destination lane negated, zeroed or kept as the source lane is negative, zero or
/// positive. Negating the most negative value wraps to itself, which the caller's truncating
/// cast preserves.
fn apply_sign(value: i32, sign_source: i32) -> i32 {
    match sign_source.signum() {
        -1 => value.wrapping_neg(),
        0 => 0,
        _ => value,
    }
}

/// palignr: the 32-byte value `dst:src` shifted right by `count` bytes, low 16 bytes kept.
fn palignr(dst: u128, src: u128, count: u8) -> u128 {
    const REGISTER_BYTES: u32 = 16;
    let count = count as u32;
    if count == 0 {
        src
    } else if count < REGISTER_BYTES {
        (src >> (count * 8)) | (dst << ((REGISTER_BYTES - count) * 8))
    } else if count < 2 * REGISTER_BYTES {
        dst >> ((count - REGISTER_BYTES) * 8)
    } else {
        0
    }
}

/// Applies `f` to each lane of `format` of the two operands, in order, and packs the results.
fn map_float_lanes(
    format: Format,
    dst: u128,
    src: u128,
    mut f: impl FnMut(usize, u64, u64) -> u64,
) -> u128 {
    let mut result = 0;
    for lane in 0..format.lanes() {
        let bits = f(
            lane,
            lane_get(dst, format, lane),
            lane_get(src, format, lane),
        );
        lane_put(&mut result, format, lane, bits);
    }
    result
}

/// haddp*/hsubp*: the low half of the result is the pairwise combination of the destination's
/// lanes, the high half that of the source's. The left operand of each pair is the lower lane.
fn horizontal(fp: &mut FpEnv, format: Format, subtract: bool, dst: u128, src: u128) -> u128 {
    let operation = if subtract {
        Arithmetic::Subtract
    } else {
        Arithmetic::Add
    };
    let half = format.lanes() / 2;
    let mut result = 0;
    for pair in 0..format.lanes() {
        let from = if pair < half { dst } else { src };
        let lane = (pair % half) * 2;
        let bits = arithmetic(
            fp,
            format,
            operation,
            lane_get(from, format, lane),
            lane_get(from, format, lane + 1),
        );
        lane_put(&mut result, format, pair, bits);
    }
    result
}

impl Cpu {
    /// Executes an instruction of the extension sets above. `Ok(false)` means "not mine": the
    /// caller raises the logged #UD for an unimplemented opcode. On `Ok(true)` the caller sets
    /// rip to `next`.
    pub(super) fn sse_ext(&mut self, i: &Insn, next: u64) -> ExecResult<bool> {
        let map = i.op >> 8;
        let opcode = (i.op & 0xff) as u8;
        let prefix = match (i.rep, i.p66) {
            (1, _) => Prefix::F3,
            (2, _) => Prefix::F2,
            (_, true) => Prefix::P66,
            _ => Prefix::Np,
        };
        let Some(op) = decode_op(map, opcode, prefix) else {
            return Ok(false);
        };
        let is_register = i.md == MODRM_MOD_REGISTER;
        // As in `sse`: an invalid ModRM form is #UD even with CR0.TS set.
        if is_register && op.memory_only() {
            return Err(Exception::invalid_opcode());
        }
        self.sse_gate()?;
        let reg = i.reg as usize;
        let mem_addr = if is_register {
            None
        } else {
            Some(self.linear_address(i, self.effective_address(i, next)))
        };
        let src = match (op.source(), mem_addr) {
            (_, None) => {
                let v = self.xmm[i.rm as usize];
                match op.source() {
                    Source::Low(bytes) => v & mask_bytes(bytes),
                    _ => v,
                }
            }
            (Source::Aligned, Some(addr)) => {
                if addr & (VECTOR_ALIGNMENT - 1) != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.read128(addr)?
            }
            (Source::Unaligned, Some(addr)) => self.read128(addr)?,
            (Source::Low(bytes), Some(addr)) => self.read(addr, bytes)? as u128,
        };
        let dst = self.xmm[reg];
        let mut fp = FpEnv::new(self.mxcsr);
        let result = match op {
            Op::Movddup => (src & LOW_64) | (src << 64),
            Op::Movsldup => {
                let (even0, even2) = (src & LOW_32, (src >> 64) & LOW_32);
                even0 | (even0 << 32) | (even2 << 64) | (even2 << 96)
            }
            Op::Movshdup => {
                let (odd1, odd3) = ((src >> 32) & LOW_32, src >> 96);
                odd1 | (odd1 << 32) | (odd3 << 64) | (odd3 << 96)
            }
            Op::Lddqu => src,
            Op::AddSub { double } => {
                let format = if double { DOUBLE } else { SINGLE };
                map_float_lanes(format, dst, src, |lane, a, b| {
                    let operation = if lane % 2 == 0 {
                        Arithmetic::Subtract
                    } else {
                        Arithmetic::Add
                    };
                    arithmetic(&mut fp, format, operation, a, b)
                })
            }
            Op::Horizontal { double, subtract } => {
                let format = if double { DOUBLE } else { SINGLE };
                horizontal(&mut fp, format, subtract, dst, src)
            }
            Op::Binary(f) => f(dst, src),
            Op::Unary(f) => f(src),
            Op::WithImm(f) => f(dst, src, i.imm as u8),
        };
        self.fp_commit(&fp)?;
        self.xmm[reg] = result;
        Ok(true)
    }
}

/// A mask of the low `bytes` bytes of a vector register.
fn mask_bytes(bytes: u8) -> u128 {
    if bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (bytes as u32 * 8)) - 1
    }
}

#[cfg(test)]
#[path = "sse_ext_tests.rs"]
mod tests;
