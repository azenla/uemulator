//! Conversion tests: every element is checked against the host's casts, which round to nearest
//! even, and against the x86 "integer indefinite" rules for out-of-range results.

use super::*;

fn random_ints(rng: &mut Rng) -> Vector {
    rng.vector()
}

/// Floats spread over small fractions, ordinary magnitudes, huge values and specials.
fn mixed_singles(rng: &mut Rng) -> Vector {
    build(64, 4, |_| {
        let r = rng.next();
        let x = match r % 8 {
            0 => (r >> 8) as f32 / 1e6,
            1 => -((r >> 8) as f32 / 1e3),
            2 => f32::from_bits((r >> 16) as u32 & 0x7fff_ffff | 0x4e00_0000),
            3 => (r >> 8) as u32 as f32,
            4 => [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                0.5,
                1.5,
                -0.5,
            ][(r >> 8) as usize % 8],
            5 => ((r >> 8) % 100000) as f32 + 0.5,
            6 => -(((r >> 8) % 100000) as f32) - 0.25,
            _ => ((r >> 8) % 1000) as f32 / 8.0,
        };
        x.to_bits() as u64
    })
}

fn mixed_doubles(rng: &mut Rng) -> Vector {
    build(64, 8, |_| {
        let r = rng.next();
        let x = match r % 8 {
            0 => (r >> 8) as f64 / 1e9,
            1 => -((r >> 8) as f64 / 1e3),
            2 => f64::from_bits(r >> 1 & 0x7fff_ffff_ffff_ffff | 0x43e0_0000_0000_0000),
            3 => (r >> 4) as f64,
            4 => [
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                0.5,
                1.5,
                -0.5,
            ][(r >> 8) as usize % 8],
            5 => ((r >> 8) % 1000000) as f64 + 0.5,
            6 => -(((r >> 8) % 1000000) as f64) - 0.25,
            _ => 9.3e18 + (r >> 12) as f64 * 1024.0,
        };
        x.to_bits()
    })
}

fn rounded(x: f64, truncate: bool) -> f64 {
    if truncate {
        x.trunc()
    } else {
        x.round_ties_even()
    }
}

fn to_i64(x: f64, truncate: bool) -> u64 {
    let r = rounded(x, truncate);
    if x.is_nan() || r >= 9.223372036854775808e18 || r < -9.223372036854775808e18 {
        0x8000_0000_0000_0000
    } else {
        r as i64 as u64
    }
}

fn to_i32(x: f64, truncate: bool) -> u64 {
    let r = rounded(x, truncate);
    if x.is_nan() || r >= 2147483648.0 || r < -2147483648.0 {
        0x8000_0000
    } else {
        r as i32 as u32 as u64
    }
}

fn to_u64(x: f64, truncate: bool) -> u64 {
    let r = rounded(x, truncate);
    if x.is_nan() || r >= 18446744073709551616.0 || (r < 0.0 && r != 0.0) {
        u64::MAX
    } else {
        r as u64
    }
}

fn to_u32(x: f64, truncate: bool) -> u64 {
    let r = rounded(x, truncate);
    if x.is_nan() || r >= 4294967296.0 || (r < 0.0 && r != 0.0) {
        0xffff_ffff
    } else {
        r as u32 as u64
    }
}

fn single(bits: u64) -> f64 {
    f32::from_bits(bits as u32) as f64
}

fn double(bits: u64) -> f64 {
    f64::from_bits(bits)
}

#[test]
fn integer_to_float_conversions() {
    check_convert("vcvtudq2ps", &VCVTUDQ2PS, 4, 4, true, &random_ints, |v| {
        (v as u32 as f32).to_bits() as u64
    });
    check_convert("vcvtqq2pd", &VCVTQQ2PD, 8, 8, true, &random_ints, |v| {
        (v as i64 as f64).to_bits()
    });
    check_convert("vcvtuqq2pd", &VCVTUQQ2PD, 8, 8, true, &random_ints, |v| {
        (v as f64).to_bits()
    });
    check_convert("vcvtdq2pd", &VCVTDQ2PD, 4, 8, false, &random_ints, |v| {
        (v as u32 as i32 as f64).to_bits()
    });
    check_convert("vcvtudq2pd", &VCVTUDQ2PD, 4, 8, false, &random_ints, |v| {
        (v as u32 as f64).to_bits()
    });
    check_convert("vcvtqq2ps", &VCVTQQ2PS, 8, 4, true, &random_ints, |v| {
        (v as i64 as f32).to_bits() as u64
    });
    check_convert("vcvtuqq2ps", &VCVTUQQ2PS, 8, 4, true, &random_ints, |v| {
        (v as f32).to_bits() as u64
    });
}

#[test]
fn float_to_integer_conversions() {
    check_convert("vcvtps2dq", &VCVTPS2DQ, 4, 4, true, &mixed_singles, |v| {
        to_i32(single(v), false)
    });
    check_convert("vcvttps2dq", &VCVTTPS2DQ, 4, 4, true, &mixed_singles, |v| {
        to_i32(single(v), true)
    });
    check_convert("vcvtps2udq", &VCVTPS2UDQ, 4, 4, true, &mixed_singles, |v| {
        to_u32(single(v), false)
    });
    check_convert(
        "vcvttps2udq",
        &VCVTTPS2UDQ,
        4,
        4,
        true,
        &mixed_singles,
        |v| to_u32(single(v), true),
    );
    check_convert("vcvtpd2qq", &VCVTPD2QQ, 8, 8, true, &mixed_doubles, |v| {
        to_i64(double(v), false)
    });
    check_convert("vcvttpd2qq", &VCVTTPD2QQ, 8, 8, true, &mixed_doubles, |v| {
        to_i64(double(v), true)
    });
    check_convert("vcvtpd2uqq", &VCVTPD2UQQ, 8, 8, true, &mixed_doubles, |v| {
        to_u64(double(v), false)
    });
    check_convert(
        "vcvttpd2uqq",
        &VCVTTPD2UQQ,
        8,
        8,
        true,
        &mixed_doubles,
        |v| to_u64(double(v), true),
    );
    check_convert("vcvtps2qq", &VCVTPS2QQ, 4, 8, false, &mixed_singles, |v| {
        to_i64(single(v), false)
    });
    check_convert(
        "vcvttps2uqq",
        &VCVTTPS2UQQ,
        4,
        8,
        false,
        &mixed_singles,
        |v| to_u64(single(v), true),
    );
    check_convert("vcvtpd2dq", &VCVTPD2DQ, 8, 4, true, &mixed_doubles, |v| {
        to_i32(double(v), false)
    });
    check_convert("vcvttpd2dq", &VCVTTPD2DQ, 8, 4, true, &mixed_doubles, |v| {
        to_i32(double(v), true)
    });
    check_convert(
        "vcvttpd2udq",
        &VCVTTPD2UDQ,
        8,
        4,
        true,
        &mixed_doubles,
        |v| to_u32(double(v), true),
    );
}

#[test]
fn float_width_conversions() {
    check_convert("vcvtps2pd", &VCVTPS2PD, 4, 8, false, &mixed_singles, |v| {
        let x = f32::from_bits(v as u32);
        if x.is_nan() {
            // the payload shifts left and the quiet bit is forced
            let sign = (v & 0x8000_0000) << 32;
            return sign | 0x7ff8_0000_0000_0000 | (v & 0x007f_ffff) << 29;
        }
        (x as f64).to_bits()
    });
    check_convert("vcvtpd2ps", &VCVTPD2PS, 8, 4, true, &mixed_doubles, |v| {
        let x = f64::from_bits(v);
        if x.is_nan() {
            let sign = (v >> 32) & 0x8000_0000;
            return sign | 0x7fc0_0000 | (v >> 29) & 0x007f_ffff;
        }
        (x as f32).to_bits() as u64
    });
}

#[test]
fn invalid_conversions_raise_the_invalid_flag_and_the_indefinite() {
    let h = run(VCVTTPS2UDQ.p[2], |h| {
        h.cpu
            .set_zmm(2, singles(&[-1.5, 5.0e9, f32::NAN, -0.25, 3.7]))
    });
    let v = h.cpu.zmm(0);
    assert_eq!(element(&v, 4, 0), 0xffff_ffff, "negative");
    assert_eq!(element(&v, 4, 1), 0xffff_ffff, "too big");
    assert_eq!(element(&v, 4, 2), 0xffff_ffff, "NaN");
    assert_eq!(element(&v, 4, 3), 0, "-0.25 truncates to zero");
    assert_eq!(element(&v, 4, 4), 3);
    assert_eq!(h.cpu.mxcsr & 0x21, 0x21, "invalid and precision");
    // the non-truncating forms follow MXCSR.RC
    let h = run(VCVTPS2UDQ.p[2], |h| {
        h.cpu.mxcsr = 0x1f80 | 3 << 13;
        h.cpu.set_zmm(2, singles(&[2.9, 2.5, 1.1]));
    });
    assert_eq!(element(&h.cpu.zmm(0), 4, 0), 2, "round toward zero");
    // embedded rounding on the scalar form
    let h = run(VCVTSS2SI_L_RU, |h| h.cpu.set_zmm(2, singles(&[2.1])));
    assert_eq!(h.cpu.gpr[0], 3);
    let h = run(VCVTSS2SI_L_RU, |h| h.cpu.set_zmm(2, singles(&[-2.9])));
    assert_eq!(h.cpu.gpr[0] as u32, (-2i32) as u32);
}

#[test]
fn half_precision_conversions() {
    let halves: [u16; 8] = [
        0x3c00, 0xc000, 0x7bff, 0x0001, 0x0400, 0x3555, 0x8000, 0x0000,
    ];
    let h = run(VCVTPH2PS.p[2], |h| {
        h.cpu.set_zmm(2, build(32, 2, |k| halves[k % 8] as u64));
    });
    let want = [
        1.0f32,
        -2.0,
        65504.0,
        f32::powi(2.0, -24),
        f32::powi(2.0, -14),
        0.333251953125,
        -0.0,
        0.0,
    ];
    for k in 0..16 {
        assert_eq!(
            f32_at(&h.cpu.zmm(0), k).to_bits(),
            want[k % 8].to_bits(),
            "element {k}"
        );
    }
    // masked widening zeroes the rest
    let h = run(VCVTPH2PS.kz[2], |h| {
        h.cpu.set_zmm(2, build(32, 2, |_| 0x3c00));
        h.cpu.opmask[1] = 0x00f0;
    });
    for k in 0..16 {
        let want = if (0x00f0u64 >> k) & 1 == 1 {
            1.0f32
        } else {
            0.0
        };
        assert_eq!(f32_at(&h.cpu.zmm(0), k), want);
    }
    // narrowing with imm 0 (nearest even) and imm 2 (round up)
    let values = [
        1.0f32, -2.0, 65504.0, 1.0e-8, 70000.0, 0.3333333, -0.0, 1.0e-5,
    ];
    let h = run(VCVTPS2PH.p[2], |h| h.cpu.set_zmm(1, singles(&values)));
    let got = h.cpu.zmm(0);
    let want = [
        0x3c00u64, 0xc000, 0x7bff, 0x0000, 0x7c00, 0x3555, 0x8000, 0x00a8,
    ];
    for k in 0..8 {
        assert_eq!(element(&got, 2, k), want[k], "vcvtps2ph nearest {k}");
    }
    assert_eq!(
        &got[2..],
        &[0, 0],
        "a 512-bit source gives a 256-bit result"
    );
    let h = run(VCVTPS2PH_UP.p[2], |h| h.cpu.set_zmm(1, singles(&values)));
    assert_eq!(
        element(&h.cpu.zmm(0), 2, 3),
        1,
        "1e-8 rounds up to the smallest half"
    );
    assert_eq!(h.cpu.mxcsr & 0x30, 0x30, "underflow and precision");
    // memory destination
    let h = run(VCVTPS2PH.m[2], |h| {
        h.cpu.set_zmm(1, singles(&values));
        h.write(OUT, &[0xee; 64]);
    });
    let stored = h.read(OUT, 64);
    assert_eq!(&stored[..2], &0x3c00u16.to_le_bytes());
    assert_eq!(&stored[14..16], &0x00a8u16.to_le_bytes());
    assert!(stored[32..].iter().all(|&b| b == 0xee));
}

#[test]
fn scalar_integer_to_float() {
    let h = run(VCVTSI2SS_Q, |h| {
        h.cpu.gpr[0] = (-1234567890123i64) as u64;
        h.cpu.set_zmm(1, singles(&[0.0, 5.0, 6.0]));
    });
    assert_eq!(f32_at(&h.cpu.zmm(0), 0), -1234567890123i64 as f32);
    assert_eq!(f32_at(&h.cpu.zmm(0), 1), 5.0, "upper elements from vvvv");
    let h = run(VCVTSI2SD_L, |h| {
        h.cpu.gpr[0] = 0xffff_fff0;
        h.cpu.set_zmm(1, doubles(&[0.0, 7.0]));
    });
    assert_eq!(
        f64_at(&h.cpu.zmm(0), 0),
        -16.0,
        "the 32-bit source is signed"
    );
    assert_eq!(f64_at(&h.cpu.zmm(0), 1), 7.0);
    let h = run(VCVTUSI2SD_Q, |h| {
        h.cpu.gpr[0] = u64::MAX;
        h.cpu.set_zmm(1, doubles(&[0.0, 7.0]));
    });
    assert_eq!(f64_at(&h.cpu.zmm(0), 0), u64::MAX as f64);
    let h = run(VCVTUSI2SS_L, |h| h.cpu.gpr[0] = 0xffff_fff0);
    assert_eq!(f32_at(&h.cpu.zmm(0), 0), 0xffff_fff0u32 as f32);
    let h = run(VCVTSI2SS_L_RD, |h| h.cpu.gpr[0] = 16777217);
    assert_eq!(f32_at(&h.cpu.zmm(0), 0), 16777216.0, "round down");
    let h = run(VCVTSI2SD_M, |h| h.write(DATA, &(-5i32).to_le_bytes()));
    assert_eq!(f64_at(&h.cpu.zmm(0), 0), -5.0);
}

#[test]
fn scalar_float_to_integer() {
    let h = run(VCVTSS2SI_Q, |h| h.cpu.set_zmm(2, singles(&[-2.5, 9.0])));
    assert_eq!(h.cpu.gpr[0], (-2i64) as u64, "nearest even");
    let h = run(VCVTTSD2SI_L, |h| {
        h.cpu.gpr[0] = u64::MAX;
        h.cpu.set_zmm(2, doubles(&[-7.9]));
    });
    assert_eq!(
        h.cpu.gpr[0],
        (-7i32) as u32 as u64,
        "a 32-bit result zero-extends"
    );
    let h = run(VCVTTSD2SI_Q, |h| h.cpu.set_zmm(2, doubles(&[1.0e30])));
    assert_eq!(h.cpu.gpr[0], 0x8000_0000_0000_0000);
    assert_eq!(h.cpu.mxcsr & 1, 1);
    let h = run(VCVTSD2SI_Q, |h| h.cpu.set_zmm(2, doubles(&[3.5])));
    assert_eq!(h.cpu.gpr[0], 4);
    let h = run(VCVTSS2USI_Q, |h| h.cpu.set_zmm(2, singles(&[1.8446744e19])));
    assert_eq!(h.cpu.gpr[0], 1.8446744e19f32 as u64);
    let h = run(VCVTTSD2USI_Q, |h| h.cpu.set_zmm(2, doubles(&[-1.0])));
    assert_eq!(h.cpu.gpr[0], u64::MAX, "negative is invalid");
    let h = run(VCVTTSS2USI_L, |h| {
        h.cpu.set_zmm(2, singles(&[4000000000.0]))
    });
    assert_eq!(h.cpu.gpr[0], 4000000000);
    let h = run(VCVTSD2USI_L, |h| h.cpu.set_zmm(2, doubles(&[2.5])));
    assert_eq!(h.cpu.gpr[0], 2);
}

#[test]
fn scalar_float_to_float() {
    let h = run(VCVTSS2SD, |h| {
        h.cpu.set_zmm(1, doubles(&[0.0, 8.0]));
        h.cpu.set_zmm(2, singles(&[1.5]));
    });
    assert_eq!(f64_at(&h.cpu.zmm(0), 0), 1.5);
    assert_eq!(f64_at(&h.cpu.zmm(0), 1), 8.0);
    let h = run(VCVTSD2SS, |h| {
        h.cpu.set_zmm(1, singles(&[0.0, 4.0, 5.0, 6.0]));
        h.cpu.set_zmm(2, doubles(&[0.1]));
    });
    assert_eq!(f32_at(&h.cpu.zmm(0), 0), 0.1f32);
    assert_eq!(f32_at(&h.cpu.zmm(0), 1), 4.0);
    let h = run(VCVTSD2SS_RD, |h| h.cpu.set_zmm(2, doubles(&[0.1])));
    assert_eq!(
        f32_at(&h.cpu.zmm(0), 0),
        f32::from_bits(0.1f32.to_bits() - 1),
        "0.1 rounded down"
    );
    let h = run(VCVTSS2SD_K, |h| {
        h.cpu.set_zmm(0, doubles(&[9.0, 9.0]));
        h.cpu.set_zmm(1, doubles(&[0.0, 8.0]));
        h.cpu.set_zmm(2, singles(&[1.5]));
        h.cpu.opmask[1] = 0;
    });
    assert_eq!(h.cpu.zmm(0)[0] as u64, 0, "zeroed element");
    assert_eq!(f64_at(&h.cpu.zmm(0), 1), 8.0);
}

#[test]
fn masked_conversions_do_not_compute_masked_out_elements() {
    // element 1 would be invalid but is masked out: no flag, no #XM
    let h = run(VCVTTPS2UDQ.kz[2], |h| {
        h.cpu.set_zmm(2, singles(&[1.0, f32::NAN, 2.0]));
        h.cpu.opmask[1] = 0b101;
    });
    assert_eq!(h.cpu.mxcsr & 0x3f, 0);
    assert_eq!(element(&h.cpu.zmm(0), 4, 0), 1);
    assert_eq!(element(&h.cpu.zmm(0), 4, 1), 0);
}
