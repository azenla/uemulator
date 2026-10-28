//! Reference-model tests for the integer instructions of `avx512_int.rs`, `avx512_perm.rs` and
//! `avx512_misc.rs`. Each expectation is a plain loop over elements written from the SDM.

use super::*;

/// Runs every shape of a compare-into-mask `variants` against `f(a, b)` per element.
pub(in crate::cpu) fn check_compare(
    name: &str,
    variants: &Variants,
    element: usize,
    broadcast: usize,
    f: impl Fn(u64, u64) -> bool,
) {
    assert!(!variants.p[2].is_empty(), "{name}: no 512-bit encoding");
    let mut rng = Rng(0x1357_9bdf ^ name.len() as u64);
    for trial in 0..6 {
        let mut a = rng.vector();
        let mut b = rng.vector();
        // make some elements equal so the equality-sensitive predicates see both outcomes
        for k in 0..64 / element {
            if rng.next().is_multiple_of(3) {
                let v = super::element(&a, element, k);
                set_element(&mut b, element, k, v);
            }
        }
        // and some neighbours differ in the sign bit
        let _ = &mut a;
        let k1 = rng.next();
        for (index, length) in LENGTHS.into_iter().enumerate() {
            for (shape, codes) in [
                ("plain", &variants.p),
                ("merge", &variants.k),
                ("memory", &variants.m),
                ("memory merge", &variants.mk),
                ("broadcast", &variants.b),
                ("broadcast merge", &variants.bk),
            ] {
                let code = codes[index];
                if code.is_empty() {
                    continue;
                }
                let masked = matches!(shape, "merge" | "memory merge" | "broadcast merge");
                let broadcasting = shape.starts_with("broadcast");
                let h = run(code, |h| {
                    h.cpu.set_zmm(1, a);
                    h.cpu.set_zmm(2, b);
                    h.cpu.opmask[1] = k1;
                    h.cpu.opmask[2] = u64::MAX;
                    h.write(DATA, &bytes_of(&b));
                });
                let source = if broadcasting {
                    let e = super::element(&b, broadcast, 0);
                    let mut v = [0u128; 4];
                    for k in 0..64 / broadcast {
                        set_element(&mut v, broadcast, k, e);
                    }
                    v
                } else {
                    b
                };
                let mut want = 0u64;
                for k in 0..length / element {
                    let selected = !masked || k1 >> k & 1 == 1;
                    if selected
                        && f(
                            super::element(&a, element, k),
                            super::element(&source, element, k),
                        )
                    {
                        want |= 1 << k;
                    }
                }
                assert_eq!(
                    h.cpu.opmask[2], want,
                    "{name} {shape} {length} bytes trial {trial}"
                );
            }
        }
        let h = run(variants.high, |h| {
            h.cpu.set_zmm(21, a);
            h.cpu.set_zmm(22, b);
            h.cpu.opmask[3] = u64::MAX;
        });
        let mut want = 0u64;
        for k in 0..64 / element {
            if f(
                super::element(&a, element, k),
                super::element(&b, element, k),
            ) {
                want |= 1 << k;
            }
        }
        assert_eq!(h.cpu.opmask[3], want, "{name} high registers");
    }
}

fn s(v: u64, size: usize) -> i64 {
    signed(v, size)
}

mod element_wise {
    use super::*;

    macro_rules! binary {
        ($name:ident, $variants:expr, $element:expr, $broadcast:expr, $f:expr) => {
            #[test]
            fn $name() {
                let mask = size_mask($element);
                check_elementwise(
                    stringify!($name),
                    &$variants,
                    spec($element, $broadcast),
                    |a, b| {
                        let f: fn(u64, u64) -> u64 = $f;
                        f(a, b) & mask
                    },
                );
            }
        };
    }

    macro_rules! unary {
        ($name:ident, $variants:expr, $element:expr, $broadcast:expr, $f:expr) => {
            #[test]
            fn $name() {
                let mask = size_mask($element);
                check_elementwise(
                    stringify!($name),
                    &$variants,
                    unary_spec($element, $broadcast),
                    |_, b| {
                        let f: fn(u64) -> u64 = $f;
                        f(b) & mask
                    },
                );
            }
        };
    }

    binary!(vpminsq, VPMINSQ, 8, 8, |a, b| (a as i64).min(b as i64)
        as u64);
    binary!(vpmaxsq, VPMAXSQ, 8, 8, |a, b| (a as i64).max(b as i64)
        as u64);
    binary!(vpminuq, VPMINUQ, 8, 8, |a, b| a.min(b));
    binary!(vpmaxuq, VPMAXUQ, 8, 8, |a, b| a.max(b));
    binary!(vpmullq, VPMULLQ, 8, 8, |a, b| a.wrapping_mul(b));
    binary!(vpandnd, VPANDND, 4, 4, |a, b| !a & b);
    binary!(vpxord, VPXORD, 4, 4, |a, b| a ^ b);
    binary!(vporq, VPORQ, 8, 8, |a, b| a | b);

    // Variable shifts: the count is the whole element; counts at or above the width shift
    // everything out (or fill with the sign).
    binary!(vpsrlvd, VPSRLVD, 4, 4, |a, b| if b >= 32 {
        0
    } else {
        a >> b
    });
    binary!(vpsrlvq, VPSRLVQ, 8, 8, |a, b| if b >= 64 {
        0
    } else {
        a >> b
    });
    binary!(vpsllvd, VPSLLVD, 4, 4, |a, b| if b >= 32 {
        0
    } else {
        a << b
    });
    binary!(vpsllvq, VPSLLVQ, 8, 8, |a, b| if b >= 64 {
        0
    } else {
        a << b
    });
    binary!(vpsravd, VPSRAVD, 4, 4, |a, b| (s(a, 4) >> b.min(31)) as u64);
    binary!(vpsravq, VPSRAVQ, 8, 8, |a, b| (s(a, 8) >> b.min(63)) as u64);
    binary!(vpsrlvw, VPSRLVW, 2, 0, |a, b| if b >= 16 {
        0
    } else {
        a >> b
    });
    binary!(vpsllvw, VPSLLVW, 2, 0, |a, b| if b >= 16 {
        0
    } else {
        a << b
    });
    binary!(vpsravw, VPSRAVW, 2, 0, |a, b| (s(a, 2) >> b.min(15)) as u64);
    binary!(
        vprorvd,
        VPRORVD,
        4,
        4,
        |a, b| (a as u32).rotate_right(b as u32 & 31) as u64
    );
    binary!(
        vprolvd,
        VPROLVD,
        4,
        4,
        |a, b| (a as u32).rotate_left(b as u32 & 31) as u64
    );
    binary!(vprorvq, VPRORVQ, 8, 8, |a, b| a.rotate_right(b as u32 & 63));
    binary!(vprolvq, VPROLVQ, 8, 8, |a, b| a.rotate_left(b as u32 & 63));

    unary!(vpabsq, VPABSQ, 8, 8, |b| (b as i64).unsigned_abs());
    unary!(vpabsb, VPABSB, 1, 0, |b| (b as u8 as i8).unsigned_abs()
        as u64);
    unary!(vpabsw, VPABSW, 2, 0, |b| (b as u16 as i16).unsigned_abs()
        as u64);
    unary!(vplzcntd, VPLZCNTD, 4, 4, |b| (b as u32).leading_zeros()
        as u64);
    unary!(vplzcntq, VPLZCNTQ, 8, 8, |b| b.leading_zeros() as u64);
    unary!(vpopcntd, VPOPCNTD, 4, 4, |b| (b as u32).count_ones() as u64);
    unary!(vpopcntq, VPOPCNTQ, 8, 8, |b| b.count_ones() as u64);
    unary!(vpopcntb, VPOPCNTB, 1, 0, |b| (b as u8).count_ones() as u64);
    unary!(vpopcntw, VPOPCNTW, 2, 0, |b| (b as u16).count_ones() as u64);

    #[test]
    fn immediate_shifts_and_rotates_write_vvvv_from_the_source() {
        macro_rules! immediate {
            ($variants:expr, $element:expr, $broadcast:expr, $f:expr) => {{
                let mask = size_mask($element);
                check_elementwise(
                    "immediate shift",
                    &$variants,
                    unary_spec($element, $broadcast),
                    |_, b| {
                        let f: fn(u64) -> u64 = $f;
                        f(b) & mask
                    },
                );
            }};
        }
        immediate!(VPSRLD_I, 4, 4, |b| b >> 5);
        immediate!(VPSLLD_I, 4, 4, |b| b << 7);
        immediate!(VPSRAD_I, 4, 4, |b| (s(b, 4) >> 9) as u64);
        immediate!(VPSRAQ_I, 8, 8, |b| (s(b, 8) >> 3) as u64);
        immediate!(VPSRLQ_I, 8, 8, |b| b >> 11);
        immediate!(VPSLLQ_I, 8, 8, |b| b << 63);
        immediate!(VPROLD_I, 4, 4, |b| (b as u32).rotate_left(5) as u64);
        immediate!(VPRORD_I, 4, 4, |b| (b as u32).rotate_right(9) as u64);
        immediate!(VPROLQ_I, 8, 8, |b| b.rotate_left(17));
        immediate!(VPRORQ_I, 8, 8, |b| b.rotate_right(40));
        immediate!(VPSRLW_I, 2, 0, |b| b >> 3);
        immediate!(VPSRAW_I, 2, 0, |b| (s(b, 2) >> 15) as u64);
        immediate!(VPSLLW_I, 2, 0, |b| b << 4);
        // counts beyond the width: logical shifts give 0, arithmetic ones fill with the sign
        immediate!(VPSLLD_BIG_I, 4, 4, |_| 0);
        immediate!(VPSRAD_BIG_I, 4, 4, |b| (s(b, 4) >> 31) as u64);
    }

    #[test]
    fn byte_shifts_work_per_lane() {
        let mut rng = Rng(0x88);
        let v = rng.vector();
        let bytes = bytes_of(&v);
        let h = run(VPSLLDQ_Z, |h| h.cpu.set_zmm(2, v));
        let mut want = [0u8; 64];
        for lane in 0..4 {
            for k in 5..16 {
                want[lane * 16 + k] = bytes[lane * 16 + k - 5];
            }
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
        let h = run(VPSRLDQ_Z, |h| h.cpu.set_zmm(2, v));
        let mut want = [0u8; 64];
        for lane in 0..4 {
            for k in 0..13 {
                want[lane * 16 + k] = bytes[lane * 16 + k + 3];
            }
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
        let h = run(VPSLLDQ_Y_M, |h| h.write(DATA, &bytes));
        let mut want = [0u8; 64];
        for lane in 0..2 {
            for k in 4..16 {
                want[lane * 16 + k] = bytes[lane * 16 + k - 4];
            }
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
    }

    #[test]
    fn quadword_arithmetic_shift_by_a_count() {
        let mut rng = Rng(0x99);
        let (value, count) = (rng.vector(), 37u128);
        let h = run(VPSRAQ_COUNT_Z, |h| {
            h.cpu.set_zmm(1, value);
            h.cpu.set_zmm(2, [count, u128::MAX, 0, 0]);
        });
        let want = build(64, 8, |k| (s(element(&value, 8, k), 8) >> 37) as u64);
        assert_eq!(h.cpu.zmm(0), want);
        // counts of 64 and more fill with the sign bit
        let h = run(VPSRAQ_COUNT_Z, |h| {
            h.cpu.set_zmm(1, value);
            h.cpu.set_zmm(2, [200, 0, 0, 0]);
        });
        let want = build(64, 8, |k| (s(element(&value, 8, k), 8) >> 63) as u64);
        assert_eq!(h.cpu.zmm(0), want);
        // a memory count reads 16 bytes
        let h = run(VPSRAQ_COUNT_Z_M, |h| {
            h.cpu.set_zmm(1, value);
            h.write(DATA, &5u128.to_le_bytes());
        });
        let want = build(64, 8, |k| (s(element(&value, 8, k), 8) >> 5) as u64);
        assert_eq!(h.cpu.zmm(0), want);
        // masked, 256 bits
        let old = rng.vector();
        let h = run(VPSRAQ_COUNT_Y_K, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, value);
            h.cpu.set_zmm(2, [3, 0, 0, 0]);
            h.cpu.opmask[1] = 0b0110;
        });
        let got = h.cpu.zmm(0);
        for k in 0..4 {
            let want = if 0b0110 >> k & 1 == 1 {
                (s(element(&value, 8, k), 8) >> 3) as u64
            } else {
                element(&old, 8, k)
            };
            assert_eq!(element(&got, 8, k), want, "element {k}");
        }
        assert_eq!(&got[2..], &[0, 0], "above 256 bits is cleared");
    }

    #[test]
    fn word_and_byte_instructions_with_lane_semantics() {
        check_elementwise("vpmullw", &VPMULLW, spec(2, 0), |a, b| {
            a.wrapping_mul(b) & 0xffff
        });
        check_elementwise("vpmulhuw", &VPMULHUW, spec(2, 0), |a, b| (a * b) >> 16);
        check_elementwise("vpminsw", &VPMINSW, spec(2, 0), |a, b| {
            (s(a, 2).min(s(b, 2))) as u64 & 0xffff
        });
        check_elementwise("vpmaxub", &VPMAXUB, spec(1, 0), |a, b| a.max(b));
        check_elementwise("vpminuw", &VPMINUW, spec(2, 0), |a, b| a.min(b));
        check_elementwise("vpmaxsb", &VPMAXSB, spec(1, 0), |a, b| {
            (s(a, 1).max(s(b, 1))) as u64 & 0xff
        });
        check_elementwise("vpavgb", &VPAVGB, spec(1, 0), |a, b| (a + b + 1) >> 1);
        check_elementwise("vpsubusw", &VPSUBUSW, spec(2, 0), |a, b| {
            a.saturating_sub(b)
        });
        check_elementwise("vpaddusb", &VPADDUSB, spec(1, 0), |a, b| (a + b).min(0xff));
        check_elementwise("vpaddsb", &VPADDSB, spec(1, 0), |a, b| {
            (s(a, 1) + s(b, 1)).clamp(-128, 127) as u64 & 0xff
        });
        check_elementwise("vpsubsb", &VPSUBSB, spec(1, 0), |a, b| {
            (s(a, 1) - s(b, 1)).clamp(-128, 127) as u64 & 0xff
        });
        check_elementwise("vpmulhrsw", &VPMULHRSW, spec(2, 0), |a, b| {
            (((s(a, 2) * s(b, 2)) >> 14) + 1) as u64 >> 1 & 0xffff
        });
        check_elementwise("vpabsd", &VPABSD, unary_spec(4, 4), |_, b| {
            s(b, 4).unsigned_abs() & 0xffff_ffff
        });
    }

    #[test]
    fn multiplies_that_widen_use_the_low_dword_of_each_quadword() {
        check_elementwise("vpmuludq", &VPMULUDQ, spec(8, 8), |a, b| {
            (a & 0xffff_ffff) * (b & 0xffff_ffff)
        });
        check_elementwise("vpmuldq", &VPMULDQ, spec(8, 8), |a, b| {
            (s(a & 0xffff_ffff, 4) * s(b & 0xffff_ffff, 4)) as u64
        });
    }

    #[test]
    fn pack_unpack_and_madd_work_within_lanes() {
        let mut rng = Rng(0x4242);
        let (a, b) = (rng.vector(), rng.vector());
        let (ab, bb) = (bytes_of(&a), bytes_of(&b));
        // vpunpcklbw: interleave the low 8 bytes of each lane
        let h = run(VPUNPCKLBW.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let mut want = [0u8; 64];
        for lane in 0..4 {
            for k in 0..8 {
                want[lane * 16 + 2 * k] = ab[lane * 16 + k];
                want[lane * 16 + 2 * k + 1] = bb[lane * 16 + k];
            }
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
        // vpunpckhwd
        let h = run(VPUNPCKHWD.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let mut want = [0u8; 64];
        for lane in 0..4 {
            for k in 0..4 {
                let from = lane * 16 + 8 + 2 * k;
                want[lane * 16 + 4 * k..lane * 16 + 4 * k + 2].copy_from_slice(&ab[from..from + 2]);
                want[lane * 16 + 4 * k + 2..lane * 16 + 4 * k + 4]
                    .copy_from_slice(&bb[from..from + 2]);
            }
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
        // vpackuswb: signed words saturate to unsigned bytes, a then b, per lane
        let h = run(VPACKUSWB.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let word = |bytes: &[u8], k: usize| i16::from_le_bytes([bytes[2 * k], bytes[2 * k + 1]]);
        let mut want = [0u8; 64];
        for lane in 0..4 {
            for k in 0..8 {
                want[lane * 16 + k] = word(&ab, lane * 8 + k).clamp(0, 255) as u8;
                want[lane * 16 + 8 + k] = word(&bb, lane * 8 + k).clamp(0, 255) as u8;
            }
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
        // vpmaddwd
        let h = run(VPMADDWD.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let want = build(64, 4, |k| {
            let product =
                |j: usize| i32::from(word(&ab, 2 * k + j)) * i32::from(word(&bb, 2 * k + j));
            product(0).wrapping_add(product(1)) as u32 as u64
        });
        assert_eq!(h.cpu.zmm(0), want);
        // vpmaddubsw: unsigned bytes of the first source times signed bytes of the second,
        // pairs added with signed saturation
        let h = run(VPMADDUBSW.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let want = build(64, 2, |k| {
            let product = |j: usize| i32::from(ab[2 * k + j]) * i32::from(bb[2 * k + j] as i8);
            (product(0) + product(1)).clamp(-32768, 32767) as u16 as u64
        });
        assert_eq!(h.cpu.zmm(0), want);
        // vpackssdw
        let h = run(VPACKSSDW.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let dword = |bytes: &[u8], k: usize| {
            i32::from_le_bytes(bytes[4 * k..4 * k + 4].try_into().unwrap())
        };
        let want = build(64, 2, |k| {
            let (lane, slot) = (k / 8, k % 8);
            let source = if slot < 4 { &ab } else { &bb };
            dword(source, lane * 4 + slot % 4).clamp(-32768, 32767) as u16 as u64
        });
        assert_eq!(h.cpu.zmm(0), want);
    }
}

mod compares {
    use super::*;

    macro_rules! equal_and_greater {
        ($name:ident, $variants:expr, $element:expr, $broadcast:expr, $f:expr) => {
            #[test]
            fn $name() {
                check_compare(stringify!($name), &$variants, $element, $broadcast, $f);
            }
        };
    }

    equal_and_greater!(vpcmpeqb, VPCMPEQB, 1, 0, |a, b| a == b);
    equal_and_greater!(vpcmpeqw, VPCMPEQW, 2, 0, |a, b| a == b);
    equal_and_greater!(vpcmpeqd, VPCMPEQD, 4, 4, |a, b| a == b);
    equal_and_greater!(vpcmpeqq, VPCMPEQQ, 8, 8, |a, b| a == b);
    equal_and_greater!(vpcmpgtb, VPCMPGTB, 1, 0, |a, b| s(a, 1) > s(b, 1));
    equal_and_greater!(vpcmpgtw, VPCMPGTW, 2, 0, |a, b| s(a, 2) > s(b, 2));
    equal_and_greater!(vpcmpgtd, VPCMPGTD, 4, 4, |a, b| s(a, 4) > s(b, 4));
    equal_and_greater!(vpcmpgtq, VPCMPGTQ, 8, 8, |a, b| s(a, 8) > s(b, 8));
    equal_and_greater!(vptestmd, VPTESTMD, 4, 4, |a, b| a & b & 0xffff_ffff != 0);
    equal_and_greater!(vptestmq, VPTESTMQ, 8, 8, |a, b| a & b != 0);
    equal_and_greater!(vptestmb, VPTESTMB, 1, 0, |a, b| a & b & 0xff != 0);
    equal_and_greater!(vptestmw, VPTESTMW, 2, 0, |a, b| a & b & 0xffff != 0);
    equal_and_greater!(vptestnmd, VPTESTNMD, 4, 4, |a, b| a & b & 0xffff_ffff == 0);
    equal_and_greater!(vptestnmq, VPTESTNMQ, 8, 8, |a, b| a & b == 0);
    equal_and_greater!(vptestnmb, VPTESTNMB, 1, 0, |a, b| a & b & 0xff == 0);

    /// The eight integer predicates, as the SDM tabulates them.
    fn predicate(p: usize, less: bool, equal: bool) -> bool {
        match p {
            0 => equal,
            1 => less,
            2 => less || equal,
            3 => false,
            4 => !equal,
            5 => !less,
            6 => !(less || equal),
            _ => true,
        }
    }

    #[test]
    fn vpcmp_and_vpcmpu_apply_every_predicate() {
        // (variants for predicates 0..8, element, broadcast, signed)
        let sets: Vec<([&Variants; 8], usize, usize, bool)> = vec![
            (
                [
                    &VPCMPB_0, &VPCMPB_1, &VPCMPB_2, &VPCMPB_3, &VPCMPB_4, &VPCMPB_5, &VPCMPB_6,
                    &VPCMPB_7,
                ],
                1,
                0,
                true,
            ),
            (
                [
                    &VPCMPW_0, &VPCMPW_1, &VPCMPW_2, &VPCMPW_3, &VPCMPW_4, &VPCMPW_5, &VPCMPW_6,
                    &VPCMPW_7,
                ],
                2,
                0,
                true,
            ),
            (
                [
                    &VPCMPD_0, &VPCMPD_1, &VPCMPD_2, &VPCMPD_3, &VPCMPD_4, &VPCMPD_5, &VPCMPD_6,
                    &VPCMPD_7,
                ],
                4,
                4,
                true,
            ),
            (
                [
                    &VPCMPQ_0, &VPCMPQ_1, &VPCMPQ_2, &VPCMPQ_3, &VPCMPQ_4, &VPCMPQ_5, &VPCMPQ_6,
                    &VPCMPQ_7,
                ],
                8,
                8,
                true,
            ),
            (
                [
                    &VPCMPUB_0, &VPCMPUB_1, &VPCMPUB_2, &VPCMPUB_3, &VPCMPUB_4, &VPCMPUB_5,
                    &VPCMPUB_6, &VPCMPUB_7,
                ],
                1,
                0,
                false,
            ),
            (
                [
                    &VPCMPUW_0, &VPCMPUW_1, &VPCMPUW_2, &VPCMPUW_3, &VPCMPUW_4, &VPCMPUW_5,
                    &VPCMPUW_6, &VPCMPUW_7,
                ],
                2,
                0,
                false,
            ),
            (
                [
                    &VPCMPUD_0, &VPCMPUD_1, &VPCMPUD_2, &VPCMPUD_3, &VPCMPUD_4, &VPCMPUD_5,
                    &VPCMPUD_6, &VPCMPUD_7,
                ],
                4,
                4,
                false,
            ),
            (
                [
                    &VPCMPUQ_0, &VPCMPUQ_1, &VPCMPUQ_2, &VPCMPUQ_3, &VPCMPUQ_4, &VPCMPUQ_5,
                    &VPCMPUQ_6, &VPCMPUQ_7,
                ],
                8,
                8,
                false,
            ),
        ];
        for (variants, element, broadcast, signed_compare) in sets {
            for (p, v) in variants.into_iter().enumerate() {
                check_compare("vpcmp", v, element, broadcast, |a, b| {
                    let (less, equal) = if signed_compare {
                        (s(a, element) < s(b, element), a == b)
                    } else {
                        (a < b, a == b)
                    };
                    predicate(p, less, equal)
                });
            }
        }
    }
}
