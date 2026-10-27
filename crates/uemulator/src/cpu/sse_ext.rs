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

#[path = "crypto.rs"]
mod crypto;
#[path = "string_compare.rs"]
mod string_compare;

const GPR_RAX: usize = 0;
const GPR_RCX: usize = 1;
const GPR_RDX: usize = 2;

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
    /// `dst = f(dst, src, xmm0)`: the implicit xmm0 operand of the variable blends.
    WithMask(fn(u128, u128, u128) -> u128),
    /// movntdqa: an aligned 128-bit load that exists only with a memory operand.
    LoadAligned,
    /// pmovsx*/pmovzx*: widen the low lanes of the source from `from_bits` to `to_bits`.
    Widen {
        signed: bool,
        from_bits: u32,
        to_bits: u32,
    },
    /// ptest: ZF and CF from `dst & src` and `!dst & src`.
    Ptest,
    /// roundps/roundpd/roundss/roundsd.
    Round {
        double: bool,
        scalar: bool,
    },
    /// dpps/dppd.
    DotProduct {
        double: bool,
    },
    Extract(LaneKind),
    Insert(LaneKind),
    /// insertps: one dword of the source (register form) or the m32 into one dword of the
    /// destination, then zeroing by mask.
    Insertps,
    /// pcmpistri/pcmpistrm (lengths from the first zero element) and pcmpestri/pcmpestrm
    /// (lengths in EAX and EDX, or RAX and RDX under REX.W).
    StringCompare {
        explicit: bool,
        mask_output: bool,
    },
}

/// The element a pextr*/pinsr*/extractps instruction moves between a vector register and a
/// general register or memory.
#[derive(Clone, Copy)]
enum LaneKind {
    Byte,
    Word,
    /// pextrd/pinsrd, or the 64-bit pextrq/pinsrq under REX.W.
    DwordOrQword,
    /// extractps: the dword is moved as raw bits.
    Single,
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
    /// The whole register, but only `n` bytes of memory (insertps picks a dword out of the
    /// register form and takes an m32).
    WholeOrMemory(u8),
    /// No vector source: the instruction handles its ModRM operand itself.
    None,
}

impl Op {
    fn source(self) -> Source {
        match self {
            Op::Movddup => Source::Low(8),
            Op::Lddqu => Source::Unaligned,
            Op::Widen {
                from_bits, to_bits, ..
            } => Source::Low((from_bits * (VECTOR_BITS / to_bits) / 8) as u8),
            Op::Round { double, scalar } if scalar => Source::Low(if double {
                DOUBLE.bytes()
            } else {
                SINGLE.bytes()
            }),
            Op::Insertps => Source::WholeOrMemory(SINGLE.bytes()),
            Op::Extract(_) | Op::Insert(_) => Source::None,
            _ => Source::Aligned,
        }
    }

    /// Instructions that exist only with a memory operand: the register form is #UD.
    fn memory_only(self) -> bool {
        matches!(self, Op::Lddqu | Op::LoadAligned)
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
        _ => return decode_sse41_op(map, opcode),
    })
}

/// SSE4.1 (all 66-prefixed). The 0F38 widening moves are numbered 0x20-0x25 (sign) and 0x30-0x35
/// (zero) in the order bw, bd, bq, wd, wq, dq.
fn decode_sse41_op(map: u16, opcode: u8) -> Option<Op> {
    const WIDENINGS: [(u32, u32); 6] = [(8, 16), (8, 32), (8, 64), (16, 32), (16, 64), (32, 64)];
    Some(match (map, opcode) {
        (OPCODE_MAP_0F38, 0x10) => Op::WithMask(blend_variable::<u8>),
        (OPCODE_MAP_0F38, 0x14) => Op::WithMask(blend_variable::<u32>),
        (OPCODE_MAP_0F38, 0x15) => Op::WithMask(blend_variable::<u64>),
        (OPCODE_MAP_0F38, 0x17) => Op::Ptest,
        (OPCODE_MAP_0F38, 0x20..=0x25) => {
            let (from_bits, to_bits) = WIDENINGS[(opcode - 0x20) as usize];
            Op::Widen {
                signed: true,
                from_bits,
                to_bits,
            }
        }
        (OPCODE_MAP_0F38, 0x30..=0x35) => {
            let (from_bits, to_bits) = WIDENINGS[(opcode - 0x30) as usize];
            Op::Widen {
                signed: false,
                from_bits,
                to_bits,
            }
        }
        (OPCODE_MAP_0F38, 0x28) => Op::Binary(pmuldq),
        (OPCODE_MAP_0F38, 0x29) => {
            Op::Binary(|a, b| zip_lanes::<u64>(a, b, |x, y| if x == y { u64::MAX } else { 0 }))
        }
        (OPCODE_MAP_0F38, 0x2a) => Op::LoadAligned,
        (OPCODE_MAP_0F38, 0x2b) => Op::Binary(|a, b| pack_sat(a, b, 32, false)),
        (OPCODE_MAP_0F38, 0x38) => {
            Op::Binary(|a, b| zip_lanes::<u8>(a, b, |x, y| (x as i8).min(y as i8) as u8))
        }
        (OPCODE_MAP_0F38, 0x39) => {
            Op::Binary(|a, b| zip_lanes::<u32>(a, b, |x, y| (x as i32).min(y as i32) as u32))
        }
        (OPCODE_MAP_0F38, 0x3a) => Op::Binary(|a, b| zip_lanes::<u16>(a, b, u16::min)),
        (OPCODE_MAP_0F38, 0x3b) => Op::Binary(|a, b| zip_lanes::<u32>(a, b, u32::min)),
        (OPCODE_MAP_0F38, 0x3c) => {
            Op::Binary(|a, b| zip_lanes::<u8>(a, b, |x, y| (x as i8).max(y as i8) as u8))
        }
        (OPCODE_MAP_0F38, 0x3d) => {
            Op::Binary(|a, b| zip_lanes::<u32>(a, b, |x, y| (x as i32).max(y as i32) as u32))
        }
        (OPCODE_MAP_0F38, 0x3e) => Op::Binary(|a, b| zip_lanes::<u16>(a, b, u16::max)),
        (OPCODE_MAP_0F38, 0x3f) => Op::Binary(|a, b| zip_lanes::<u32>(a, b, u32::max)),
        (OPCODE_MAP_0F38, 0x40) => Op::Binary(|a, b| zip_lanes::<u32>(a, b, u32::wrapping_mul)),
        (OPCODE_MAP_0F38, 0x41) => Op::Unary(phminposuw),
        (OPCODE_MAP_0F3A, 0x08) => Op::Round {
            double: false,
            scalar: false,
        },
        (OPCODE_MAP_0F3A, 0x09) => Op::Round {
            double: true,
            scalar: false,
        },
        (OPCODE_MAP_0F3A, 0x0a) => Op::Round {
            double: false,
            scalar: true,
        },
        (OPCODE_MAP_0F3A, 0x0b) => Op::Round {
            double: true,
            scalar: true,
        },
        (OPCODE_MAP_0F3A, 0x0c) => Op::WithImm(blend_immediate::<u32>),
        (OPCODE_MAP_0F3A, 0x0d) => Op::WithImm(blend_immediate::<u64>),
        (OPCODE_MAP_0F3A, 0x0e) => Op::WithImm(blend_immediate::<u16>),
        (OPCODE_MAP_0F3A, 0x14) => Op::Extract(LaneKind::Byte),
        (OPCODE_MAP_0F3A, 0x15) => Op::Extract(LaneKind::Word),
        (OPCODE_MAP_0F3A, 0x16) => Op::Extract(LaneKind::DwordOrQword),
        (OPCODE_MAP_0F3A, 0x17) => Op::Extract(LaneKind::Single),
        (OPCODE_MAP_0F3A, 0x20) => Op::Insert(LaneKind::Byte),
        (OPCODE_MAP_0F3A, 0x21) => Op::Insertps,
        (OPCODE_MAP_0F3A, 0x22) => Op::Insert(LaneKind::DwordOrQword),
        (OPCODE_MAP_0F3A, 0x40) => Op::DotProduct { double: false },
        (OPCODE_MAP_0F3A, 0x41) => Op::DotProduct { double: true },
        (OPCODE_MAP_0F3A, 0x42) => Op::WithImm(mpsadbw),
        _ => return decode_sse42_op(map, opcode),
    })
}

/// SSE4.2's vector instructions (CRC32 is dispatched separately: it has no vector operands).
fn decode_sse42_op(map: u16, opcode: u8) -> Option<Op> {
    Some(match (map, opcode) {
        (OPCODE_MAP_0F38, 0x37) => Op::Binary(|a, b| {
            zip_lanes::<u64>(
                a,
                b,
                |x, y| if (x as i64) > (y as i64) { u64::MAX } else { 0 },
            )
        }),
        (OPCODE_MAP_0F3A, 0x60) => Op::StringCompare {
            explicit: true,
            mask_output: true,
        },
        (OPCODE_MAP_0F3A, 0x61) => Op::StringCompare {
            explicit: true,
            mask_output: false,
        },
        (OPCODE_MAP_0F3A, 0x62) => Op::StringCompare {
            explicit: false,
            mask_output: true,
        },
        (OPCODE_MAP_0F3A, 0x63) => Op::StringCompare {
            explicit: false,
            mask_output: false,
        },
        _ => return decode_crypto_op(map, opcode),
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

/// AES-NI and PCLMULQDQ.
fn decode_crypto_op(map: u16, opcode: u8) -> Option<Op> {
    Some(match (map, opcode) {
        (OPCODE_MAP_0F38, 0xdb) => Op::Unary(crypto::aes_inverse_mix_columns),
        (OPCODE_MAP_0F38, 0xdc) => {
            Op::Binary(|state, key| crypto::aes_encrypt_round(state, key, false))
        }
        (OPCODE_MAP_0F38, 0xdd) => {
            Op::Binary(|state, key| crypto::aes_encrypt_round(state, key, true))
        }
        (OPCODE_MAP_0F38, 0xde) => {
            Op::Binary(|state, key| crypto::aes_decrypt_round(state, key, false))
        }
        (OPCODE_MAP_0F38, 0xdf) => {
            Op::Binary(|state, key| crypto::aes_decrypt_round(state, key, true))
        }
        // The old destination does not take part: the immediate is the round constant.
        (OPCODE_MAP_0F3A, 0xdf) => {
            Op::WithImm(|_, source, constant| crypto::aes_key_generation_assist(source, constant))
        }
        (OPCODE_MAP_0F3A, 0x44) => Op::WithImm(pclmulqdq),
        _ => return None,
    })
}

/// pclmulqdq immediate: bit 0 picks the high quadword of the destination operand, bit 4 that of
/// the source; the other bits are ignored (the pclmul{l,h}q{l,h}qdq aliases are 0x00/0x01/0x10/0x11).
const PCLMUL_DESTINATION_HIGH: u8 = 1 << 0;
const PCLMUL_SOURCE_HIGH: u8 = 1 << 4;

fn pclmulqdq(dst: u128, src: u128, selector: u8) -> u128 {
    let quadword = |value: u128, high: bool| {
        if high {
            (value >> 64) as u64
        } else {
            value as u64
        }
    };
    crypto::carry_less_multiply(
        quadword(dst, selector & PCLMUL_DESTINATION_HIGH != 0),
        quadword(src, selector & PCLMUL_SOURCE_HIGH != 0),
    )
}

/// pmuldq: signed product of dwords 0 and 2 of each operand, as two qwords.
fn pmuldq(dst: u128, src: u128) -> u128 {
    let mut result = 0;
    for quad in 0..2 {
        let product =
            (u32::get(dst, 2 * quad) as i32 as i64) * (u32::get(src, 2 * quad) as i32 as i64);
        u64::put(&mut result, quad, product as u64);
    }
    result
}

/// pblendvb/blendvps/blendvpd: each lane comes from `src` when the top bit of the matching lane
/// of the implicit xmm0 mask is set, else it stays.
fn blend_variable<T: Lane>(dst: u128, src: u128, mask: u128) -> u128 {
    let lane_bits = VECTOR_BITS as usize / T::N;
    let mut result = 0;
    for lane in 0..T::N {
        let take_source = (mask >> (lane * lane_bits + lane_bits - 1)) & 1 != 0;
        T::put(
            &mut result,
            lane,
            T::get(if take_source { src } else { dst }, lane),
        );
    }
    result
}

/// pblendw/blendps/blendpd: bit `k` of the immediate selects lane `k` of `src`; immediate bits
/// beyond the lane count are ignored.
fn blend_immediate<T: Lane>(dst: u128, src: u128, selector: u8) -> u128 {
    let mut result = 0;
    for lane in 0..T::N {
        let take_source = (selector >> lane) & 1 != 0;
        T::put(
            &mut result,
            lane,
            T::get(if take_source { src } else { dst }, lane),
        );
    }
    result
}

/// pmovsx*/pmovzx*: the low `128 / to_bits` lanes of `from_bits` bits, extended.
fn widen(src: u128, signed: bool, from_bits: u32, to_bits: u32) -> u128 {
    let from_mask = (1u128 << from_bits) - 1;
    let to_mask = (1u128 << to_bits) - 1;
    let mut result = 0;
    for lane in 0..VECTOR_BITS / to_bits {
        let raw = (src >> (lane * from_bits)) & from_mask;
        let negative = (raw >> (from_bits - 1)) & 1 != 0;
        let value = if signed && negative {
            raw | !from_mask
        } else {
            raw
        };
        result |= (value & to_mask) << (lane * to_bits);
    }
    result
}

/// phminposuw: the smallest unsigned word (first one on a tie) in bits 15:0 and its index in
/// bits 18:16; everything above is zero.
fn phminposuw(src: u128) -> u128 {
    let mut minimum = u16::get(src, 0);
    let mut index = 0;
    for lane in 1..u16::N {
        let value = u16::get(src, lane);
        if value < minimum {
            minimum = value;
            index = lane;
        }
    }
    minimum as u128 | ((index as u128) << 16)
}

/// mpsadbw: eight sums of absolute differences between a sliding 4-byte window of the
/// destination and one fixed 4-byte block of the source. imm8 bits 1:0 pick the source block and
/// bit 2 the destination's first window start, both in units of four bytes.
fn mpsadbw(dst: u128, src: u128, selector: u8) -> u128 {
    const BLOCK_BYTES: usize = 4;
    let source_start = (selector & 3) as usize * BLOCK_BYTES;
    let destination_start = ((selector >> 2) & 1) as usize * BLOCK_BYTES;
    let mut result = 0;
    for window in 0..u16::N {
        let sum: u16 = (0..BLOCK_BYTES)
            .map(|k| {
                let a = u8::get(dst, destination_start + window + k) as i16;
                let b = u8::get(src, source_start + k) as i16;
                (a - b).unsigned_abs()
            })
            .sum();
        u16::put(&mut result, window, sum);
    }
    result
}

/// roundps/pd/ss/sd immediate: bit 2 selects MXCSR.RC instead of bits 1:0, bit 3 suppresses the
/// precision exception.
const ROUND_USE_MXCSR: u8 = 1 << 2;
const ROUND_SUPPRESS_PRECISION: u8 = 1 << 3;
const ROUND_MODE_MASK: u8 = 3;

/// Rounds one value to an integral value in the same format. A signaling NaN is quieted and
/// raises invalid; zeros, infinities and quiet NaNs pass through; a subnormal input is not
/// reported as a denormal operand (the SDM lists only invalid and precision for these
/// instructions) but DAZ still turns it into a zero. `quiet_precision` hides the inexact flag.
fn round_integral(
    fp: &mut FpEnv,
    format: Format,
    bits: u64,
    rounding: u32,
    quiet_precision: bool,
) -> u64 {
    let denormal_before = fp.flags & FLAG_DENORMAL;
    let value = unpack(fp, format, bits);
    fp.flags = (fp.flags & !FLAG_DENORMAL) | denormal_before;
    match value.kind {
        Kind::SignalingNan => {
            fp.flags |= FLAG_INVALID;
            return value.bits | format.quiet_bit();
        }
        Kind::QuietNan | Kind::Infinity | Kind::Zero => return value.bits,
        Kind::Finite => {}
    }
    // Magnitudes of 2^fraction_bits and up have no fractional bits.
    if value.exponent >= format.fraction_bits as i32 {
        return value.bits;
    }
    let shift = (format.fraction_bits as i32 - value.exponent) as u32;
    // Beyond this shift the whole significand is below half of the last kept place, so only the
    // sticky remainder matters.
    const NEGLIGIBLE_SHIFT: u32 = 100;
    let significand = value.significand as u128;
    let (mut kept, remainder, half) = if shift > NEGLIGIBLE_SHIFT {
        (0, 1, 2)
    } else {
        (
            significand >> shift,
            significand & ((1u128 << shift) - 1),
            1u128 << (shift - 1),
        )
    };
    if rounds_up(rounding, value.sign, kept, remainder, half) {
        kept += 1;
    }
    if remainder != 0 && !quiet_precision {
        fp.flags |= FLAG_PRECISION;
    }
    if kept == 0 {
        return format.zero(value.sign);
    }
    let leading = leading_bit(kept);
    let sign_bits = if value.sign { format.sign_bit() } else { 0 };
    let exponent_field = (leading as i32 + format.bias()) as u64;
    let fraction = ((kept << (format.fraction_bits - leading)) as u64) & format.fraction_mask();
    sign_bits | (exponent_field << format.fraction_bits) | fraction
}

fn round_lanes(
    fp: &mut FpEnv,
    format: Format,
    scalar: bool,
    dst: u128,
    src: u128,
    selector: u8,
) -> u128 {
    let rounding = if selector & ROUND_USE_MXCSR != 0 {
        fp.rounding
    } else {
        (selector & ROUND_MODE_MASK) as u32
    };
    let quiet_precision = selector & ROUND_SUPPRESS_PRECISION != 0;
    // The scalar forms keep the rest of the destination.
    let (lanes, mut result) = if scalar {
        (1, dst & !format.lane_mask())
    } else {
        (format.lanes(), 0)
    };
    for lane in 0..lanes {
        let bits = round_integral(
            fp,
            format,
            lane_get(src, format, lane),
            rounding,
            quiet_precision,
        );
        lane_put(&mut result, format, lane, bits);
    }
    result
}

/// dpps/dppd: imm8 bits 7:4 pick which products take part, bits 3:0 which result lanes receive the
/// sum (the others are zeroed). The sum is `(p0 + p1) + (p2 + p3)` with unselected products as
/// +0.0, the order the SDM pseudocode gives.
fn dot_product(fp: &mut FpEnv, double: bool, dst: u128, src: u128, selector: u8) -> u128 {
    let format = if double { DOUBLE } else { SINGLE };
    let product_mask = selector >> 4;
    let output_mask = selector & 0xf;
    let mut products = [format.zero(false); 4];
    for (lane, product) in products.iter_mut().enumerate().take(format.lanes()) {
        if (product_mask >> lane) & 1 != 0 {
            *product = arithmetic(
                fp,
                format,
                Arithmetic::Multiply,
                lane_get(dst, format, lane),
                lane_get(src, format, lane),
            );
        }
    }
    let sum = if double {
        arithmetic(fp, format, Arithmetic::Add, products[0], products[1])
    } else {
        let low = arithmetic(fp, format, Arithmetic::Add, products[0], products[1]);
        let high = arithmetic(fp, format, Arithmetic::Add, products[2], products[3]);
        arithmetic(fp, format, Arithmetic::Add, low, high)
    };
    let mut result = 0;
    for lane in 0..format.lanes() {
        if (output_mask >> lane) & 1 != 0 {
            lane_put(&mut result, format, lane, sum);
        }
    }
    result
}

/// pinsrb/pinsrd/pinsrq: replaces one lane of `dst`; imm8 is taken modulo the lane count.
fn insert_lane(dst: u128, kind: LaneKind, selector: u8, value: u64, wide: bool) -> u128 {
    let selector = selector as usize;
    let (lane_bits, lane_count) = match kind {
        LaneKind::Byte => (8, u8::N),
        _ if wide => (64, u64::N),
        _ => (32, u32::N),
    };
    let shift = (selector % lane_count) as u32 * lane_bits;
    let lane_mask = ((1u128 << lane_bits) - 1) << shift;
    (dst & !lane_mask) | (((value as u128) << shift) & lane_mask)
}

/// insertps: imm8 bits 7:6 pick the source dword (register form only, the m32 form always uses
/// its one dword), bits 5:4 the destination dword it replaces, bits 3:0 zero result dwords.
fn insertps(dst: u128, src: u128, selector: u8, from_memory: bool) -> u128 {
    let source_lane = if from_memory {
        0
    } else {
        (selector >> 6) as usize
    };
    let destination_lane = ((selector >> 4) & 3) as u32;
    let zero_mask = selector & 0xf;
    let mut result = dst & !(LOW_32 << (destination_lane * 32));
    result |= (u32::get(src, source_lane) as u128) << (destination_lane * 32);
    for lane in 0..u32::N {
        if (zero_mask >> lane) & 1 != 0 {
            result &= !(LOW_32 << (lane * 32));
        }
    }
    result
}

impl Cpu {
    /// MOVBE (0F 38 F0 load, F1 store): a general register and a memory operand with the bytes
    /// swapped; 16, 32 or 64 bits by operand size. There is no register form (#UD), and no flags
    /// change.
    fn movbe(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        if i.md == MODRM_MOD_REGISTER {
            return Err(Exception::invalid_opcode());
        }
        let size = i.opsize;
        let swap = |value: u64| match size {
            2 => (value as u16).swap_bytes() as u64,
            4 => (value as u32).swap_bytes() as u64,
            _ => value.swap_bytes(),
        };
        let addr = self.linear_address(i, self.effective_address(i, next));
        if i.op & 0xff == 0xf0 {
            let value = swap(self.read(addr, size)?);
            self.write_gpr(i.reg, size, true, value);
        } else {
            let value = swap(self.read_gpr(i.reg, size, true));
            self.write(addr, size, value)?;
        }
        Ok(())
    }

    /// CRC32 (F2 0F 38 F0/F1): accumulates the source operand into the CRC-32C in the low 32 bits
    /// of the destination register. The source is a byte (F0) or 16/32/64 bits by operand size
    /// (F1); the destination is 32 bits, or 64 under REX.W (the upper half is then zero). No SSE
    /// enable rules apply.
    fn crc32(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let source_bytes = if i.op & 0xff == 0xf0 { 1 } else { i.opsize };
        let value = if i.md == MODRM_MOD_REGISTER {
            self.read_gpr(i.rm, source_bytes, i.rex_present)
        } else {
            let addr = self.linear_address(i, self.effective_address(i, next));
            self.read(addr, source_bytes)?
        };
        let crc = self.read_gpr(i.reg, 4, true) as u32;
        let result = crypto::crc32c(crc, value, source_bytes);
        let destination_bytes = if i.opsize == 8 { 8 } else { 4 };
        self.write_gpr(i.reg, destination_bytes, true, result as u64);
        Ok(())
    }

    /// pcmpistri/pcmpistrm/pcmpestri/pcmpestrm. Explicit lengths are the absolute values of
    /// EAX (first string) and EDX (second), or of RAX and RDX under REX.W, saturated to the
    /// element count. Flags: CF = result nonzero, ZF = second string shorter than a full register,
    /// SF = first string shorter, OF = result bit 0; AF and PF are cleared.
    fn string_compare(
        &mut self,
        wide: bool,
        explicit: bool,
        mask_output: bool,
        first: u128,
        second: u128,
        control: u8,
    ) {
        let count = string_compare::element_count(control);
        let (first_length, second_length) = if explicit {
            let clamp = |register: u64| -> usize {
                let magnitude = if wide {
                    (register as i64).unsigned_abs()
                } else {
                    (register as u32 as i32).unsigned_abs() as u64
                };
                magnitude.min(count as u64) as usize
            };
            (clamp(self.gpr[GPR_RAX]), clamp(self.gpr[GPR_RDX]))
        } else {
            (
                string_compare::implicit_length(first, control),
                string_compare::implicit_length(second, control),
            )
        };
        let comparison =
            string_compare::compare(control, first, first_length, second, second_length);
        let mut flags = 0;
        for (set, flag) in [
            (comparison.carry, CF),
            (comparison.zero, ZF),
            (comparison.sign, SF),
            (comparison.overflow, OF),
        ] {
            if set {
                flags |= flag;
            }
        }
        self.set_eflags_arith(flags);
        if mask_output {
            self.xmm[0] = string_compare::result_mask(control, comparison.result);
        } else {
            let index = string_compare::result_index(control, comparison.result);
            self.write_gpr(GPR_RCX as u8, 4, true, index as u64);
        }
    }

    /// pextrb/pextrw/pextrd/pextrq/extractps: the lane of `source` that imm8 selects goes to the
    /// ModRM.rm general register (zero-extended) or memory. imm8 is taken modulo the lane count.
    fn extract_lane(
        &mut self,
        i: &Insn,
        mem_addr: Option<u64>,
        kind: LaneKind,
        source: u128,
    ) -> ExecResult<()> {
        let selector = i.imm as usize;
        let (value, size) = match kind {
            LaneKind::Byte => (u8::get(source, selector % u8::N) as u64, 1),
            LaneKind::Word => (u16::get(source, selector % u16::N) as u64, 2),
            LaneKind::DwordOrQword if i.opsize == 8 => (u64::get(source, selector % u64::N), 8),
            LaneKind::DwordOrQword | LaneKind::Single => {
                (u32::get(source, selector % u32::N) as u64, 4)
            }
        };
        match mem_addr {
            Some(addr) => self.write(addr, size, value),
            None => {
                self.write_gpr(i.rm, size.max(4), true, value);
                Ok(())
            }
        }
    }

    /// The value pinsrb/pinsrd/pinsrq insert: from memory, or from the low bits of the ModRM.rm
    /// general register.
    fn insert_source(
        &mut self,
        i: &Insn,
        mem_addr: Option<u64>,
        kind: LaneKind,
    ) -> ExecResult<u64> {
        let size = match kind {
            LaneKind::Byte => 1,
            _ if i.opsize == 8 => 8,
            _ => 4,
        };
        match mem_addr {
            Some(addr) => self.read(addr, size),
            None => Ok(self.read_gpr(i.rm, size.max(4), true) & mask(size)),
        }
    }

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
        if map == OPCODE_MAP_0F38 && matches!(opcode, 0xf0 | 0xf1) {
            return match prefix {
                Prefix::F2 => self.crc32(i, next).map(|()| true),
                Prefix::Np | Prefix::P66 => self.movbe(i, next).map(|()| true),
                Prefix::F3 => Ok(false),
            };
        }
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
            (Source::None, None) => 0,
            (source, None) => {
                let v = self.xmm[i.rm as usize];
                match source {
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
            (Source::Low(bytes) | Source::WholeOrMemory(bytes), Some(addr)) => {
                self.read(addr, bytes)? as u128
            }
            (Source::None, Some(_)) => 0,
        };
        let dst = self.xmm[reg];
        let selector = i.imm as u8;
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
            Op::WithImm(f) => f(dst, src, selector),
            Op::WithMask(f) => f(dst, src, self.xmm[0]),
            Op::LoadAligned => src,
            Op::Widen {
                signed,
                from_bits,
                to_bits,
            } => widen(src, signed, from_bits, to_bits),
            Op::Ptest => {
                let mut flags = 0;
                if dst & src == 0 {
                    flags |= ZF;
                }
                if !dst & src == 0 {
                    flags |= CF;
                }
                self.set_eflags_arith(flags);
                return Ok(true);
            }
            Op::Round { double, scalar } => {
                let format = if double { DOUBLE } else { SINGLE };
                round_lanes(&mut fp, format, scalar, dst, src, selector)
            }
            Op::DotProduct { double } => dot_product(&mut fp, double, dst, src, selector),
            Op::Extract(kind) => {
                self.extract_lane(i, mem_addr, kind, dst)?;
                return Ok(true);
            }
            Op::Insert(kind) => {
                let value = self.insert_source(i, mem_addr, kind)?;
                insert_lane(dst, kind, selector, value, i.opsize == 8)
            }
            Op::Insertps => insertps(dst, src, selector, mem_addr.is_some()),
            Op::StringCompare {
                explicit,
                mask_output,
            } => {
                self.string_compare(i.opsize == 8, explicit, mask_output, dst, src, selector);
                return Ok(true);
            }
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
