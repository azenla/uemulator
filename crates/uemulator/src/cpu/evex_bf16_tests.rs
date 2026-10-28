//! AVX512_BF16. References are written from the SDM pseudo-code differently from the
//! implementation: the conversion by comparing the dropped bits against the halfway point, the
//! dot product on the host's binary32 arithmetic with denormals flushed by hand.

use super::*;
use crate::testkit::assert_jit_matches_interpreter;

fn reference_bfloat16(bits: u32) -> u64 {
    let (kept, dropped) = (bits >> 16, bits & 0xffff);
    let exponent = bits >> 23 & 0xff;
    if exponent == 0xff && bits & 0x7f_ffff != 0 {
        return (kept | 0x40) as u64;
    }
    if exponent == 0 {
        return (kept & 0x8000) as u64;
    }
    let round_up = dropped > 0x8000 || (dropped == 0x8000 && kept & 1 == 1);
    (kept + round_up as u32) as u64 & 0xffff
}

/// Floats around the interesting spots: ties, NaNs of both kinds with payloads, denormals, the
/// largest finite values (which round up to infinity) and ordinary numbers.
fn conversion_inputs(rng: &mut Rng) -> Vector {
    build(64, 4, |_| {
        let r = rng.next();
        let sign = (r as u32 & 1) << 31;
        let bits = match r >> 8 & 7 {
            0 => sign | 0x7f80_0000 | (r >> 20) as u32 & 0x7f_ffff,
            1 => sign | (r >> 20) as u32 & 0x7f_ffff,
            2 => sign | 0x7f7f_0000 | (r >> 20) as u32 & 0xffff,
            3 => sign | (r >> 16) as u32 & 0x7fff_0000 | 0x8000,
            4 => sign | 0x7f80_0000,
            _ => (r >> 20) as u32,
        };
        bits as u64
    })
}

#[test]
fn vcvtneps2bf16_rounds_to_nearest_even_and_flushes_denormals() {
    check_convert(
        "vcvtneps2bf16",
        &VCVTNEPS2BF16,
        4,
        2,
        false,
        &conversion_inputs,
        |bits| reference_bfloat16(bits as u32),
    );
}

#[test]
fn vcvtneps2bf16_broadcasts_a_dword() {
    let source = f32::to_bits(1.00390625 + 3.0);
    let h = run(VCVTNEPS2BF16_BROADCAST, |h| {
        h.write(DATA, &source.to_le_bytes());
    });
    let want = build(32, 2, |_| reference_bfloat16(source));
    assert_eq!(h.cpu.zmm(0), want);
}

/// The result of an instruction with sources `first` (vvvv) and `second` (r/m), per shape.
fn check_two_source(
    name: &str,
    variants: &Variants,
    element: usize,
    result: impl Fn(&Vector, &Vector, &Vector, usize, usize) -> u64,
    generate: &dyn Fn(&mut Rng) -> Vector,
) {
    let mut rng = Rng(0xbf16 ^ name.len() as u64);
    for trial in 0..6 {
        let (a, b, old) = (generate(&mut rng), generate(&mut rng), generate(&mut rng));
        let k1 = rng.next();
        for (index, length) in LENGTHS.into_iter().enumerate() {
            for (shape, code) in [
                ("plain", variants.p[index]),
                ("merge", variants.k[index]),
                ("zero", variants.kz[index]),
                ("memory", variants.m[index]),
                ("memory merge", variants.mk[index]),
                ("broadcast", variants.b[index]),
                ("broadcast merge", variants.bk[index]),
            ] {
                if code.is_empty() {
                    continue;
                }
                let masked = matches!(shape, "merge" | "zero" | "memory merge" | "broadcast merge");
                let h = run(code, |h| {
                    h.cpu.set_zmm(0, old);
                    h.cpu.set_zmm(1, a);
                    h.cpu.set_zmm(2, b);
                    h.cpu.opmask[1] = k1;
                    h.write(DATA, &bytes_of(&b));
                });
                let second = if shape.starts_with("broadcast") {
                    let first_dword = super::element(&b, 4, 0);
                    build(64, 4, |_| first_dword)
                } else {
                    b
                };
                let want = expected(
                    length,
                    element,
                    masked.then_some((k1, shape == "zero")),
                    &old,
                    |k| result(&old, &a, &second, length, k),
                );
                assert_eq!(
                    bytes_of(&h.cpu.zmm(0)),
                    bytes_of(&want),
                    "{name} {shape} {length} trial {trial}"
                );
            }
        }
    }
}

#[test]
fn vcvtne2ps2bf16_packs_the_r_m_source_below_the_vvvv_source() {
    check_two_source(
        "vcvtne2ps2bf16",
        &VCVTNE2PS2BF16,
        2,
        |_, first, second, length, k| {
            let half = length / 4;
            let (source, index) = if k < half {
                (second, k)
            } else {
                (first, k - half)
            };
            reference_bfloat16(element(source, 4, index) as u32)
        },
        &conversion_inputs,
    );
}

/// bfloat16 pairs and accumulators with exponents that keep products finite, plus denormals,
/// zeros and products small enough to underflow.
fn dot_product_inputs(rng: &mut Rng) -> Vector {
    build(64, 4, |_| {
        let word = |rng: &mut Rng| -> u32 {
            let r = rng.next();
            let sign = (r as u32 & 1) << 15;
            let fraction = (r >> 8) as u32 & 0x7f;
            match r >> 16 & 15 {
                0 => sign | fraction,
                1 => sign,
                _ => sign | ((127 - 70 + (r >> 24) as u32 % 131) << 7) | fraction,
            }
        };
        (word(rng) as u64) | (word(rng) as u64) << 16
    })
}

fn flush(x: f32) -> f32 {
    if x.is_subnormal() {
        f32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        x
    }
}

fn reference_dot_product(accumulator: u32, first: u32, second: u32) -> u32 {
    let widen = |pair: u32, half: u32| flush(f32::from_bits((pair >> (16 * half) & 0xffff) << 16));
    let mut sum = flush(f32::from_bits(accumulator));
    for half in [1, 0] {
        sum = flush(sum + flush(widen(first, half) * widen(second, half)));
    }
    sum.to_bits()
}

#[test]
fn vdpbf16ps_accumulates_pairs_into_binary32() {
    check_two_source(
        "vdpbf16ps",
        &VDPBF16PS,
        4,
        |old, first, second, _, k| {
            reference_dot_product(
                element(old, 4, k) as u32,
                element(first, 4, k) as u32,
                element(second, 4, k) as u32,
            ) as u64
        },
        &|rng| {
            // accumulators and pairs share the generator: a pair read as binary32 is a normal or
            // tiny number, which is what the accumulator should be too
            let v = dot_product_inputs(rng);
            build(64, 4, |k| {
                let pair = element(&v, 4, k);
                if k % 3 == 0 { pair << 16 } else { pair }
            })
        },
    );
}

/// The odd pair is added before the even one, with a rounding in between: from 2^24 adding 1.0
/// is a tie that stays at 2^24 and then 2.0 gives 2^24 + 2, whereas 2.0 first and then 1.0 would
/// end at 2^24 + 4.
#[test]
fn vdpbf16ps_adds_the_odd_pair_first() {
    let h = run(VDPBF16PS.p[2], |h| {
        h.cpu.set_zmm(0, build(64, 4, |_| 0x4b80_0000));
        h.cpu.set_zmm(1, build(64, 4, |_| 0x3f80_4000));
        h.cpu.set_zmm(2, build(64, 4, |_| 0x3f80_3f80));
    });
    assert_eq!(h.cpu.zmm(0), build(64, 4, |_| 0x4b80_0001));
}

#[test]
fn bfloat16_instructions_ignore_mxcsr_and_leave_its_flags_alone() {
    // Round toward zero, no DAZ/FTZ, every exception unmasked: none of it applies.
    const MXCSR_ROUND_TOWARD_ZERO_UNMASKED: u32 = 3 << 13;
    // 2^-126 * 2^-126 underflows and (1 + 2^-8) * (1 + 2^-8) would raise precision in binary32
    let tiny = 0x0080_0080u32;
    for code in [VDPBF16PS.p[2], VCVTNE2PS2BF16.p[2], VCVTNEPS2BF16.p[2]] {
        let setup = |h: &mut Harness| {
            h.cpu.mxcsr = MXCSR_ROUND_TOWARD_ZERO_UNMASKED;
            h.cpu.set_zmm(0, build(64, 4, |_| 0x3f80_0000));
            h.cpu.set_zmm(1, build(64, 4, |_| tiny as u64));
            h.cpu.set_zmm(2, build(64, 4, |_| tiny as u64));
        };
        let h = run(code, setup);
        assert_eq!(h.cpu.mxcsr, MXCSR_ROUND_TOWARD_ZERO_UNMASKED, "{code:02x?}");
    }
    // The 0x00010001 pair is two bfloat16 denormals: they read as zero, the sum stays the accumulator.
    let h = run(VDPBF16PS.p[2], |h| {
        h.cpu.set_zmm(0, build(64, 4, |_| 0x3f80_0000));
        h.cpu.set_zmm(1, build(64, 4, |_| 0x0001_0001));
        h.cpu.set_zmm(2, build(64, 4, |_| 0x4000_4000));
    });
    assert_eq!(h.cpu.zmm(0), build(64, 4, |_| 0x3f80_0000));
}

#[test]
fn the_jit_matches_the_interpreter() {
    for code in [
        VDPBF16PS.p[2],
        VDPBF16PS.k[1],
        VCVTNE2PS2BF16.p[2],
        VCVTNE2PS2BF16.kz[0],
        VCVTNEPS2BF16.p[2],
        VCVTNEPS2BF16.mk[1],
    ] {
        assert_jit_matches_interpreter(code, |h| {
            let mut rng = Rng(0xbf16_0001);
            h.cpu.cr4 |= CR4_OSXSAVE;
            h.cpu.xcr0 = XCR0_SUPPORTED;
            h.cpu.opmask[1] = rng.next();
            for register in 0..3 {
                h.cpu.set_zmm(register, conversion_inputs(&mut rng));
            }
            h.cpu.gpr[RSI] = DATA;
            h.cpu.gpr[RDI] = OUT;
            h.write(DATA, &bytes_of(&conversion_inputs(&mut rng)));
        });
    }
}
