//! Tests for the floating-point instructions of `avx512_fp.rs` and `avx512_convert.rs`. Ordinary
//! values are checked bit-exactly against the host FPU; special values against the SDM tables.

use super::*;

#[path = "evex_conv_tests.rs"]
mod conversions;

fn singles(values: &[f32]) -> Vector {
    build(64, 4, |k| values[k % values.len()].to_bits() as u64)
}

fn doubles(values: &[f64]) -> Vector {
    build(64, 8, |k| values[k % values.len()].to_bits())
}

fn f32_at(v: &Vector, k: usize) -> f32 {
    f32::from_bits(element(v, 4, k) as u32)
}

fn f64_at(v: &Vector, k: usize) -> f64 {
    f64::from_bits(element(v, 8, k))
}

/// Random finite values with moderate exponents, so host arithmetic never meets NaNs, overflow or
/// denormals.
fn ordinary_singles(rng: &mut Rng) -> Vector {
    build(64, 4, |_| {
        let magnitude = 0.25 + (rng.next() % 4096) as f32 / 64.0;
        let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
        (sign * magnitude).to_bits() as u64
    })
}

fn ordinary_doubles(rng: &mut Rng) -> Vector {
    build(64, 8, |_| {
        let magnitude = 0.25 + (rng.next() % 65536) as f64 / 1024.0;
        let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
        (sign * magnitude).to_bits()
    })
}

const QUIET_NAN_SINGLE: u32 = 0x7fc0_0000;
const SIGNALING_NAN_SINGLE: u32 = 0x7f80_0001;
const DEFAULT_NAN_SINGLE: u32 = 0xffc0_0000;

/// Runs every shape of a conversion `variants` and compares each element with
/// `f(source element)`; `source` and `destination` are the element sizes in bytes.
fn check_convert(
    name: &str,
    variants: &Variants,
    source: usize,
    destination: usize,
    broadcast: bool,
    sources: &dyn Fn(&mut Rng) -> Vector,
    f: impl Fn(u64) -> u64,
) {
    assert!(!variants.p[2].is_empty(), "{name}: no 512-bit encoding");
    let mut rng = Rng(0xc0de ^ name.len() as u64);
    for trial in 0..4 {
        let (old, b) = (rng.vector(), sources(&mut rng));
        let k1 = rng.next();
        for (index, length) in LENGTHS.into_iter().enumerate() {
            let count = length / source.max(destination);
            let result_bytes = count * destination;
            for (shape, code) in [
                ("plain", variants.p[index]),
                ("merge", variants.k[index]),
                ("zero", variants.kz[index]),
                ("memory", variants.m[index]),
                ("memory merge", variants.mk[index]),
                ("broadcast", if broadcast { variants.b[index] } else { &[] }),
                (
                    "broadcast merge",
                    if broadcast { variants.bk[index] } else { &[] },
                ),
            ] {
                if code.is_empty() {
                    continue;
                }
                let masked = matches!(shape, "merge" | "zero" | "memory merge" | "broadcast merge");
                let h = run(code, |h| {
                    h.cpu.set_zmm(0, old);
                    h.cpu.set_zmm(2, b);
                    h.cpu.opmask[1] = k1;
                    h.write(DATA, &bytes_of(&b));
                });
                let want = expected(
                    result_bytes,
                    destination,
                    masked.then_some((k1, shape == "zero")),
                    &old,
                    |k| {
                        let index = if shape.starts_with("broadcast") { 0 } else { k };
                        f(element(&b, source, index))
                    },
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

mod fused_multiply_add {
    use super::*;

    #[derive(Clone, Copy)]
    enum Form {
        F132,
        F213,
        F231,
    }

    /// (x, y, z) such that the result is x*y ± z, for operands (dest, vvvv, rm) = (a, b, c).
    fn roles(form: Form, a: f64, b: f64, c: f64) -> (f64, f64, f64) {
        match form {
            Form::F132 => (a, c, b),
            Form::F213 => (b, a, c),
            Form::F231 => (b, c, a),
        }
    }

    #[test]
    fn packed_forms_match_the_host_fused_operation() {
        macro_rules! forms {
            ($($form:expr, $ps:ident, $pd:ident, $negate:expr, $subtract:expr, $alternate:expr);*) => {$(
                let mut rng = Rng(0xf3a ^ stringify!($ps).len() as u64);
                let (a, b, c) = (ordinary_singles(&mut rng), ordinary_singles(&mut rng), ordinary_singles(&mut rng));
                let h = run($ps.p[2], |h| {
                    h.cpu.set_zmm(0, a);
                    h.cpu.set_zmm(1, b);
                    h.cpu.set_zmm(2, c);
                });
                for k in 0..16 {
                    let (x, y, z) = roles($form, f_at(&a, k), f_at(&b, k), f_at(&c, k));
                    let product_sign = if $negate { -1.0f64 } else { 1.0 };
                    let subtract = if $alternate == 0 {
                        $subtract
                    } else if $alternate == 1 {
                        // vfmaddsub: even lanes subtract
                        k % 2 == 0
                    } else {
                        // vfmsubadd: even lanes add
                        k % 2 == 1
                    };
                    let addend = if subtract { -z } else { z };
                    let want = ((product_sign as f32) * (x as f32)).mul_add(y as f32, addend as f32);
                    assert_eq!(f32_at(&h.cpu.zmm(0), k).to_bits(), want.to_bits(), "{} lane {k}", stringify!($ps));
                }
                let mut rng = Rng(0xf3b ^ stringify!($pd).len() as u64);
                let (a, b, c) = (ordinary_doubles(&mut rng), ordinary_doubles(&mut rng), ordinary_doubles(&mut rng));
                let h = run($pd.p[2], |h| {
                    h.cpu.set_zmm(0, a);
                    h.cpu.set_zmm(1, b);
                    h.cpu.set_zmm(2, c);
                });
                for k in 0..8 {
                    let (x, y, z) = roles($form, f64_at(&a, k), f64_at(&b, k), f64_at(&c, k));
                    let product_sign = if $negate { -1.0f64 } else { 1.0 };
                    let subtract = if $alternate == 0 {
                        $subtract
                    } else if $alternate == 1 {
                        k % 2 == 0
                    } else {
                        k % 2 == 1
                    };
                    let addend = if subtract { -z } else { z };
                    let want = (product_sign * x).mul_add(y, addend);
                    assert_eq!(f64_at(&h.cpu.zmm(0), k).to_bits(), want.to_bits(), "{} lane {k}", stringify!($pd));
                }
            )*};
        }
        fn f_at(v: &Vector, k: usize) -> f64 {
            f32_at(v, k) as f64
        }
        forms!(
            Form::F132, VFMADD132PS, VFMADD132PD, false, false, 0;
            Form::F132, VFMSUB132PS, VFMSUB132PD, false, true, 0;
            Form::F132, VFNMADD132PS, VFNMADD132PD, true, false, 0;
            Form::F132, VFNMSUB132PS, VFNMSUB132PD, true, true, 0;
            Form::F132, VFMADDSUB132PS, VFMADDSUB132PD, false, false, 1;
            Form::F132, VFMSUBADD132PS, VFMSUBADD132PD, false, false, 2;
            Form::F213, VFMADD213PS, VFMADD213PD, false, false, 0;
            Form::F213, VFMSUB213PS, VFMSUB213PD, false, true, 0;
            Form::F213, VFNMADD213PS, VFNMADD213PD, true, false, 0;
            Form::F213, VFNMSUB213PS, VFNMSUB213PD, true, true, 0;
            Form::F213, VFMADDSUB213PS, VFMADDSUB213PD, false, false, 1;
            Form::F213, VFMSUBADD213PS, VFMSUBADD213PD, false, false, 2;
            Form::F231, VFMADD231PS, VFMADD231PD, false, false, 0;
            Form::F231, VFMSUB231PS, VFMSUB231PD, false, true, 0;
            Form::F231, VFNMADD231PS, VFNMADD231PD, true, false, 0;
            Form::F231, VFNMSUB231PS, VFNMSUB231PD, true, true, 0;
            Form::F231, VFMADDSUB231PS, VFMADDSUB231PD, false, false, 1;
            Form::F231, VFMSUBADD231PS, VFMSUBADD231PD, false, false, 2
        );
    }

    #[test]
    fn the_product_is_not_rounded_before_the_add() {
        // (1 + 2^-12)^2 - (1 + 2^-11) = 2^-24: only a fused operation keeps the low product bits
        let x = 1.0f32 + f32::powi(2.0, -12);
        let (a, b, c) = (
            singles(&[x]),
            singles(&[x]),
            singles(&[-(1.0 + f32::powi(2.0, -11))]),
        );
        let h = run(VFMADD231PS.p[2], |h| {
            h.cpu.set_zmm(0, c);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), f32::powi(2.0, -24));
    }

    #[test]
    fn masking_and_the_scalar_forms() {
        let mut rng = Rng(0x5ca);
        let (old, a, b) = (
            ordinary_singles(&mut rng),
            ordinary_singles(&mut rng),
            ordinary_singles(&mut rng),
        );
        let h = run(VFMADD231SS_K, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
            h.cpu.opmask[1] = 1;
        });
        let want = f32_at(&a, 0).mul_add(f32_at(&b, 0), f32_at(&old, 0));
        let got = h.cpu.zmm(0);
        assert_eq!(f32_at(&got, 0).to_bits(), want.to_bits());
        assert_eq!(
            got[0] >> 32,
            old[0] >> 32,
            "the upper elements are the destination's"
        );
        assert_eq!(&got[1..], &[0, 0, 0]);
        let h = run(VFMADD231SS_KZ, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
            h.cpu.opmask[1] = 0;
        });
        assert_eq!(h.cpu.zmm(0)[0], old[0] >> 32 << 32);
        // the three scalar forms use op1 as the destination
        let h = run(VFMADD132SS, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let want = f32_at(&old, 0).mul_add(f32_at(&b, 0), f32_at(&a, 0));
        assert_eq!(f32_at(&h.cpu.zmm(0), 0).to_bits(), want.to_bits());
        let h = run(VFNMSUB213SD, |h| {
            h.cpu.set_zmm(0, doubles(&[1.5, 9.0]));
            h.cpu.set_zmm(1, doubles(&[2.0, 7.0]));
            h.cpu.set_zmm(2, doubles(&[0.25, 5.0]));
        });
        // -(op2 * op1) - op3 = -(2 * 1.5) - 0.25
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), -3.25);
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 9.0);
        let h = run(VFMADD213SS_M, |h| {
            h.cpu.set_zmm(0, singles(&[2.0]));
            h.cpu.set_zmm(1, singles(&[3.0]));
            h.write(DATA, &4.0f32.to_le_bytes());
        });
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), 10.0);
    }

    #[test]
    fn embedded_rounding_applies_to_the_fused_result() {
        let tiny = f32::powi(2.0, -30);
        for (code, want) in [
            (VFMADD231PS_RZ, 1.0f32),
            (VFMADD231PS_RU, f32::from_bits(1.0f32.to_bits() + 1)),
        ] {
            let h = run(code, |h| {
                h.cpu.set_zmm(0, singles(&[1.0]));
                h.cpu.set_zmm(1, singles(&[tiny]));
                h.cpu.set_zmm(2, singles(&[1.0]));
            });
            assert_eq!(f32_at(&h.cpu.zmm(0), 0), want);
        }
        let h = run(VFMADD231SD_RD, |h| {
            h.cpu.set_zmm(0, doubles(&[-1.0]));
            h.cpu.set_zmm(1, doubles(&[f64::powi(2.0, -60)]));
            h.cpu.set_zmm(2, doubles(&[1.0]));
        });
        // -1 + 2^-60 is exact in neither direction... rounding down leaves it at the exact value
        // since it is representable; the point is that the instruction runs with {rd-sae}
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), -1.0 + f64::powi(2.0, -60));
    }
}

mod approximations {
    use super::*;

    #[test]
    fn reciprocal_estimates_are_within_two_to_the_minus_14() {
        let mut rng = Rng(0xa11);
        for (variants, double, rsqrt) in [
            (&VRCP14PS, false, false),
            (&VRCP14PD, true, false),
            (&VRSQRT14PS, false, true),
            (&VRSQRT14PD, true, true),
        ] {
            let source = if double {
                ordinary_doubles(&mut rng)
            } else {
                ordinary_singles(&mut rng)
            };
            let h = run(variants.p[2], |h| {
                // rsqrt needs positive inputs
                let v = if double {
                    build(64, 8, |k| f64_at(&source, k).abs().to_bits())
                } else {
                    build(64, 4, |k| f32_at(&source, k).abs().to_bits() as u64)
                };
                h.cpu.set_zmm(2, v);
            });
            for k in 0..if double { 8 } else { 16 } {
                let x = if double {
                    f64_at(&source, k).abs()
                } else {
                    f32_at(&source, k).abs() as f64
                };
                let exact = if rsqrt { 1.0 / x.sqrt() } else { 1.0 / x };
                let got = if double {
                    f64_at(&h.cpu.zmm(0), k)
                } else {
                    f32_at(&h.cpu.zmm(0), k) as f64
                };
                assert!(
                    ((got - exact) / exact).abs() < 2f64.powi(-14),
                    "{x}: {got} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn special_values_of_the_estimates() {
        let h = run(VRCP14PS.p[2], |h| {
            h.cpu
                .set_zmm(2, singles(&[0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY]))
        });
        let v = h.cpu.zmm(0);
        assert_eq!(f32_at(&v, 0), f32::INFINITY);
        assert_eq!(f32_at(&v, 1), f32::NEG_INFINITY);
        assert_eq!(f32_at(&v, 2).to_bits(), 0);
        assert_eq!(f32_at(&v, 3).to_bits(), 0x8000_0000);
        let h = run(VRSQRT14PS.p[2], |h| {
            h.cpu.set_zmm(2, singles(&[0.0, f32::INFINITY, 4.0]))
        });
        let v = h.cpu.zmm(0);
        assert_eq!(f32_at(&v, 0), f32::INFINITY);
        assert_eq!(f32_at(&v, 1), 0.0);
        assert!((f32_at(&v, 2) - 0.5).abs() < 1e-4);
        // scalar: the upper elements come from vvvv
        let h = run(VRCP14SS, |h| {
            h.cpu.set_zmm(1, singles(&[9.0, 8.0, 7.0, 6.0]));
            h.cpu.set_zmm(2, singles(&[4.0, 1.0, 1.0, 1.0]));
        });
        let v = h.cpu.zmm(0);
        assert!((f32_at(&v, 0) - 0.25).abs() < 1e-4);
        assert_eq!(
            (f32_at(&v, 1), f32_at(&v, 2), f32_at(&v, 3)),
            (8.0, 7.0, 6.0)
        );
        let h = run(VRSQRT14SD, |h| {
            h.cpu.set_zmm(1, doubles(&[9.0, 8.0]));
            h.cpu.set_zmm(2, doubles(&[16.0, 1.0]));
        });
        assert!((f64_at(&h.cpu.zmm(0), 0) - 0.25).abs() < 1e-4);
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 8.0);
    }
}

mod exponent_and_mantissa {
    use super::*;

    #[test]
    fn getexp_is_the_unbiased_exponent_as_a_float() {
        let values = [1.0f32, 2.0, 3.9, 0.5, 0.3, -8.0, 1.0e-40, 1.0e30];
        let h = run(VGETEXPPS.p[2], |h| h.cpu.set_zmm(2, singles(&values)));
        let v = h.cpu.zmm(0);
        for (k, &x) in values.iter().enumerate() {
            let want = x.abs().log2().floor();
            // the subnormal 1e-40 has exponent -133
            let want = if x == 1.0e-40 { -133.0 } else { want };
            assert_eq!(f32_at(&v, k), want, "getexp({x})");
        }
        let h = run(VGETEXPPS.p[2], |h| {
            h.cpu.set_zmm(
                2,
                singles(&[
                    0.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::from_bits(QUIET_NAN_SINGLE),
                ]),
            )
        });
        let v = h.cpu.zmm(0);
        assert_eq!(f32_at(&v, 0), f32::NEG_INFINITY, "getexp(0) = -inf");
        assert_eq!(f32_at(&v, 1), f32::INFINITY);
        assert_eq!(f32_at(&v, 2), f32::INFINITY, "getexp(-inf) = +inf");
        assert_eq!(element(&v, 4, 3), QUIET_NAN_SINGLE as u64);
        let h = run(VGETEXPPD.p[2], |h| {
            h.cpu.set_zmm(2, doubles(&[1.0, 1024.0, 0.1]))
        });
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), 0.0);
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 10.0);
        assert_eq!(f64_at(&h.cpu.zmm(0), 2), -4.0);
        // a signaling NaN is quieted and raises invalid
        let h = run(VGETEXPPS.p[2], |h| {
            h.cpu.set_zmm(2, [SIGNALING_NAN_SINGLE as u128; 4])
        });
        assert_eq!(element(&h.cpu.zmm(0), 4, 0), 0x7fc0_0001);
        assert_eq!(h.cpu.mxcsr & 1, 1);
        let h = run(VGETEXPSD, |h| {
            h.cpu.set_zmm(1, doubles(&[9.0, 8.0]));
            h.cpu.set_zmm(2, doubles(&[40.0, 1.0]));
        });
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), 5.0);
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 8.0);
        let h = run(VGETEXPSS_K, |h| {
            h.cpu.set_zmm(0, singles(&[77.0]));
            h.cpu.set_zmm(1, singles(&[9.0, 8.0]));
            h.cpu.set_zmm(2, singles(&[40.0]));
            h.cpu.opmask[1] = 0;
        });
        assert_eq!(
            f32_at(&h.cpu.zmm(0), 0),
            77.0,
            "masked scalar keeps the old element"
        );
    }

    #[test]
    fn getmant_normalises_into_the_selected_interval() {
        let values = [12.0f32, 0.75, -6.5, 1.0, 100.0, 0.1, 1.5, 1.4];
        // interval [1, 2), sign from the source
        let h = run(VGETMANTPS.p[2], |h| h.cpu.set_zmm(2, singles(&values)));
        for (k, &x) in values.iter().enumerate() {
            let exponent = x.abs().log2().floor() as i32;
            let want = x / f32::powi(2.0, exponent);
            assert_eq!(f32_at(&h.cpu.zmm(0), k), want, "[1,2) getmant({x})");
        }
        // interval [1/2, 1)
        let h = run(VGETMANTPD_HALF.p[2], |h| {
            h.cpu.set_zmm(2, doubles(&[12.0, 0.75, 3.0]))
        });
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), 0.75, "[1/2, 2): 12 = 0.75 * 16");
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 0.75);
        assert_eq!(f64_at(&h.cpu.zmm(0), 2), 0.75);
        // interval [3/4, 3/2)
        let h = run(VGETMANTPS_34.p[2], |h| {
            h.cpu.set_zmm(2, singles(&[12.0, 1.4, 0.8, 7.0]))
        });
        let v = h.cpu.zmm(0);
        assert_eq!(f32_at(&v, 0), 0.75);
        assert_eq!(f32_at(&v, 1), 1.4);
        assert_eq!(f32_at(&v, 2), 0.8);
        assert_eq!(f32_at(&v, 3), 0.875);
        // sign control: always positive
        let h = run(VGETMANTPS_NOSIGN.p[2], |h| {
            h.cpu.set_zmm(2, singles(&[-12.0, 3.0]))
        });
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), 1.5);
        // sign control: negative input is the default NaN with invalid
        let h = run(VGETMANTPS_NANNEG.p[2], |h| {
            h.cpu.set_zmm(2, singles(&[-12.0, 12.0]))
        });
        assert_eq!(element(&h.cpu.zmm(0), 4, 0), DEFAULT_NAN_SINGLE as u64);
        assert_eq!(f32_at(&h.cpu.zmm(0), 1), 1.5);
        assert_eq!(h.cpu.mxcsr & 1, 1);
        let h = run(VGETMANTSS, |h| {
            h.cpu.set_zmm(1, singles(&[0.0, 5.0]));
            h.cpu.set_zmm(2, singles(&[20.0]));
        });
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), 1.25);
        assert_eq!(f32_at(&h.cpu.zmm(0), 1), 5.0);
    }

    #[test]
    fn scalef_multiplies_by_two_to_the_floor_of_the_second_operand() {
        let a = [1.5f32, -3.0, 0.25, 7.0, 1.0, 1.0, 1.0e30, 1.0e-30];
        let b = [3.0f32, 2.5, -2.0, -0.5, 200.0, -200.0, 100.0, -100.0];
        let h = run(VSCALEFPS.p[2], |h| {
            h.cpu.set_zmm(1, singles(&a));
            h.cpu.set_zmm(2, singles(&b));
        });
        for k in 0..8 {
            let n = b[k].floor() as i32;
            // powi(2, 200) overflows f32, so scale in two steps through f64
            let want = ((a[k] as f64) * 2f64.powi(n)) as f32;
            assert_eq!(f32_at(&h.cpu.zmm(0), k), want, "scalef({}, {})", a[k], b[k]);
        }
        // special values
        let a = [
            0.0f32,
            f32::INFINITY,
            1.0,
            0.0,
            f32::NEG_INFINITY,
            2.0,
            f32::from_bits(QUIET_NAN_SINGLE),
            3.0,
        ];
        let b = [
            5.0,
            -5.0,
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            1.0,
            f32::from_bits(QUIET_NAN_SINGLE),
        ];
        let h = run(VSCALEFPS.p[2], |h| {
            h.cpu.set_zmm(1, singles(&a));
            h.cpu.set_zmm(2, singles(&b));
        });
        let v = h.cpu.zmm(0);
        assert_eq!(f32_at(&v, 0), 0.0);
        assert_eq!(f32_at(&v, 1), f32::INFINITY);
        assert_eq!(f32_at(&v, 2), f32::INFINITY);
        assert_eq!(
            element(&v, 4, 3),
            DEFAULT_NAN_SINGLE as u64,
            "0 * 2^inf is invalid"
        );
        assert_eq!(
            element(&v, 4, 4),
            DEFAULT_NAN_SINGLE as u64,
            "inf * 2^-inf is invalid"
        );
        assert_eq!(f32_at(&v, 5), 0.0);
        assert!(f32_at(&v, 6).is_nan());
        assert!(f32_at(&v, 7).is_nan());
        let h = run(VSCALEFPD.p[2], |h| {
            h.cpu.set_zmm(1, doubles(&[3.0, 0.5, 1.0]));
            h.cpu.set_zmm(2, doubles(&[4.9, -1.1, 1100.0]));
        });
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), 48.0);
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 0.125);
        assert_eq!(f64_at(&h.cpu.zmm(0), 2), f64::INFINITY);
        let h = run(VSCALEFSS, |h| {
            h.cpu.set_zmm(1, singles(&[3.0, 9.0]));
            h.cpu.set_zmm(2, singles(&[2.0]));
        });
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), 12.0);
        assert_eq!(f32_at(&h.cpu.zmm(0), 1), 9.0);
        let h = run(VSCALEFSD, |h| {
            h.cpu.set_zmm(1, doubles(&[3.0]));
            h.cpu.set_zmm(2, doubles(&[-1.0]));
        });
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), 1.5);
    }
}

mod rounding_instructions {
    use super::*;

    fn values() -> Vec<f64> {
        vec![
            0.5, 1.5, 2.5, -0.5, -1.5, 3.7, -3.7, 100.25, 0.1, -0.1, 7.0, 123456.789,
        ]
    }

    #[test]
    fn rndscale_rounds_to_a_multiple_of_a_power_of_two() {
        // imm 0x0b: truncate, precision exception suppressed
        let h = run(VRNDSCALEPS.p[2], |h| {
            h.cpu.set_zmm(
                2,
                singles(&values().iter().map(|&v| v as f32).collect::<Vec<_>>()),
            )
        });
        for (k, &v) in values().iter().enumerate() {
            assert_eq!(f32_at(&h.cpu.zmm(0), k), (v as f32).trunc(), "trunc({v})");
        }
        assert_eq!(h.cpu.mxcsr & 0x20, 0, "suppressed precision exception");
        // imm 0x41: round down to multiples of 1/16
        let h = run(VRNDSCALEPS_SCALE.p[2], |h| {
            h.cpu.set_zmm(
                2,
                singles(&values().iter().map(|&v| v as f32).collect::<Vec<_>>()),
            )
        });
        for (k, &v) in values().iter().enumerate() {
            let want = ((v as f32) * 16.0).floor() / 16.0;
            assert_eq!(f32_at(&h.cpu.zmm(0), k), want, "floor to 1/16 of {v}");
        }
        assert_eq!(h.cpu.mxcsr & 0x20, 0x20, "precision is reported");
        // imm 0x09: floor, precision suppressed; imm 0x04: current rounding mode (nearest even)
        let h = run(VRNDSCALEPD.p[2], |h| {
            h.cpu.set_zmm(2, doubles(&values()[..8]))
        });
        for (k, &v) in values()[..8].iter().enumerate() {
            assert_eq!(f64_at(&h.cpu.zmm(0), k), v.floor());
        }
        let h = run(VRNDSCALEPD_MXCSR.p[2], |h| {
            h.cpu.set_zmm(2, doubles(&values()[..8]))
        });
        for (k, &v) in values()[..8].iter().enumerate() {
            assert_eq!(f64_at(&h.cpu.zmm(0), k), v.round_ties_even());
        }
        // scalar forms keep vvvv's upper element and honour the mask
        let h = run(VRNDSCALESD, |h| {
            h.cpu.set_zmm(1, doubles(&[0.0, 42.0]));
            h.cpu.set_zmm(2, doubles(&[-2.9]));
        });
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), -2.0);
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 42.0);
        let h = run(VRNDSCALESS_K, |h| {
            h.cpu.set_zmm(0, singles(&[5.0]));
            h.cpu.set_zmm(1, singles(&[0.0, 1.0]));
            h.cpu.set_zmm(2, singles(&[2.9]));
            h.cpu.opmask[1] = 0;
        });
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), 5.0);
    }

    #[test]
    fn rndscale_leaves_special_values_alone() {
        let specials = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::from_bits(QUIET_NAN_SINGLE),
            1.0e20,
        ];
        let h = run(VRNDSCALEPS.p[2], |h| h.cpu.set_zmm(2, singles(&specials)));
        for (k, &v) in specials.iter().enumerate() {
            let got = f32_at(&h.cpu.zmm(0), k);
            assert!(
                got.to_bits() == v.to_bits() || (got.is_nan() && v.is_nan()),
                "{v}"
            );
        }
    }

    #[test]
    fn reduce_returns_what_rndscale_removes() {
        let h = run(VREDUCEPS.p[2], |h| {
            h.cpu.set_zmm(
                2,
                singles(&values().iter().map(|&v| v as f32).collect::<Vec<_>>()),
            )
        });
        for (k, &v) in values().iter().enumerate() {
            let x = v as f32;
            assert_eq!(f32_at(&h.cpu.zmm(0), k), x - x.trunc(), "reduce({v})");
        }
        let h = run(VREDUCEPD.p[2], |h| {
            h.cpu.set_zmm(2, doubles(&values()[..8]))
        });
        for (k, &v) in values()[..8].iter().enumerate() {
            assert_eq!(
                f64_at(&h.cpu.zmm(0), k),
                v - v.round_ties_even(),
                "reduce nearest of {v}"
            );
        }
        let h = run(VREDUCESD, |h| {
            h.cpu.set_zmm(1, doubles(&[0.0, 8.0]));
            h.cpu.set_zmm(2, doubles(&[2.75]));
        });
        assert_eq!(f64_at(&h.cpu.zmm(0), 0), 0.75);
        assert_eq!(f64_at(&h.cpu.zmm(0), 1), 8.0);
    }
}

mod range_and_class {
    use super::*;

    #[test]
    fn range_picks_min_or_max_with_a_sign_control() {
        let a = [1.0f32, -5.0, 3.0, -2.0];
        let b = [2.0f32, 4.0, -3.5, -2.5];
        // imm 0: min, sign from the selected value
        let h = run(VRANGEPS_MIN.p[2], |h| {
            h.cpu.set_zmm(1, singles(&a));
            h.cpu.set_zmm(2, singles(&b));
        });
        for k in 0..4 {
            assert_eq!(f32_at(&h.cpu.zmm(0), k), a[k].min(b[k]));
        }
        // imm 0x06: min by magnitude, sign of the first operand
        let h = run(VRANGEPS_MINABS.p[2], |h| {
            h.cpu.set_zmm(1, singles(&a));
            h.cpu.set_zmm(2, singles(&b));
        });
        for k in 0..4 {
            let magnitude = a[k].abs().min(b[k].abs());
            assert_eq!(f32_at(&h.cpu.zmm(0), k), magnitude.copysign(a[k]));
        }
        // imm 0x0f: max by magnitude, sign set
        let h = run(VRANGEPD_MAXABS_NEG.p[2], |h| {
            h.cpu.set_zmm(1, doubles(&[1.0, -5.0, 3.0, -2.0]));
            h.cpu.set_zmm(2, doubles(&[2.0, 4.0, -3.5, -2.5]));
        });
        let want = [-2.0, -5.0, -3.5, -2.5];
        for (k, w) in want.into_iter().enumerate() {
            assert_eq!(f64_at(&h.cpu.zmm(0), k), w);
        }
        let h = run(VRANGESS, |h| {
            h.cpu.set_zmm(1, singles(&[1.0, 7.0]));
            h.cpu.set_zmm(2, singles(&[2.0]));
        });
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), 2.0, "imm 1 is max");
        assert_eq!(f32_at(&h.cpu.zmm(0), 1), 7.0);
    }

    fn class_bits(bits: u32, selector: u8) -> bool {
        let x = f32::from_bits(bits);
        let negative = bits >> 31 == 1;
        let mut classes = 0u8;
        if x.is_nan() {
            classes |= if bits & 0x0040_0000 != 0 { 1 } else { 0x80 };
        }
        if x == 0.0 {
            classes |= if negative { 4 } else { 2 };
        }
        if x.is_infinite() {
            classes |= if negative { 0x10 } else { 8 };
        }
        if x.is_subnormal() {
            classes |= 0x20;
        }
        if negative && x != 0.0 && !x.is_nan() && !x.is_infinite() {
            classes |= 0x40;
        }
        classes & selector != 0
    }

    #[test]
    fn fpclass_reports_membership_in_the_selected_classes() {
        let patterns: [u32; 16] = [
            0,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            QUIET_NAN_SINGLE,
            SIGNALING_NAN_SINGLE,
            0xffc0_0000,
            1,
            0x8000_0001,
            0x3f80_0000,
            0xbf80_0000,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x4000_0000,
        ];
        let v = build(64, 4, |k| patterns[k] as u64);
        for (variants, selector) in [
            (&VFPCLASSPS, 0xffu8),
            (&VFPCLASSPS_INF, 0x18),
            (&VFPCLASSPS_NEG, 0x56),
        ] {
            let h = run(variants.p[2], |h| h.cpu.set_zmm(2, v));
            let mut want = 0u64;
            for (k, &bits) in patterns.iter().enumerate() {
                if class_bits(bits, selector) {
                    want |= 1 << k;
                }
            }
            assert_eq!(h.cpu.opmask[2], want, "selector {selector:#x}");
        }
        // doubles: QNaN or SNaN
        let nan = f64::from_bits(0x7ff8_0000_0000_0000);
        let snan = f64::from_bits(0x7ff0_0000_0000_0001);
        let h = run(VFPCLASSPD_NAN.p[2], |h| {
            h.cpu.set_zmm(2, doubles(&[1.0, nan, snan, f64::INFINITY]))
        });
        assert_eq!(
            h.cpu.opmask[2], 0b0110_0110,
            "the four values repeat across the vector"
        );
        // selector 0xc3 = QNaN, +0, negative finite, SNaN: -inf is none of them, -3.0 is
        let h = run(VFPCLASSSD, |h| {
            h.cpu.set_zmm(2, doubles(&[f64::NEG_INFINITY]))
        });
        assert_eq!(h.cpu.opmask[2], 0);
        let h = run(VFPCLASSSD, |h| h.cpu.set_zmm(2, doubles(&[-3.0])));
        assert_eq!(h.cpu.opmask[2], 1);
    }

    #[test]
    fn fpclass_scalar_from_memory() {
        let h = run(VFPCLASSSS_M, |h| {
            h.write(DATA, &f32::INFINITY.to_le_bytes())
        });
        assert_eq!(h.cpu.opmask[2], 1);
        let h = run(VFPCLASSSS_M, |h| h.write(DATA, &1.0f32.to_le_bytes()));
        assert_eq!(h.cpu.opmask[2], 0);
    }

    #[test]
    fn fixupimm_replaces_special_values_by_table_responses() {
        // Table nibbles, indexed by token: QNaN 0, SNaN 1, zero 2, one 3, -inf 4, +inf 5,
        // negative 6, positive 7. Responses: 5 = +inf, 10 = +1, 9 = -1, 8 = +0, 14 = max float,
        // 1 = keep the input, 0 = keep the destination.
        let table: u32 = 0x1e5a_9850; // token0=0, 1=5, 2=8, 3=9, 4=a, 5=5, 6=e, 7=1
        let tokens = [
            f32::from_bits(QUIET_NAN_SINGLE),
            f32::from_bits(SIGNALING_NAN_SINGLE),
            0.0,
            1.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -2.0,
            2.0,
        ];
        let old = singles(&[100.0]);
        let h = run(VFIXUPIMMPS.p[2], |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, singles(&tokens));
            h.cpu.set_zmm(
                2,
                [table as u128
                    | (table as u128) << 32
                    | (table as u128) << 64
                    | (table as u128) << 96; 4],
            );
        });
        let v = h.cpu.zmm(0);
        assert_eq!(f32_at(&v, 0), 100.0, "response 0 keeps the destination");
        assert_eq!(f32_at(&v, 1), f32::INFINITY, "response 5");
        assert_eq!(f32_at(&v, 2), 0.0, "response 8");
        assert_eq!(f32_at(&v, 3), -1.0, "response 9");
        assert_eq!(f32_at(&v, 4), 1.0, "response a");
        assert_eq!(f32_at(&v, 5), f32::INFINITY);
        assert_eq!(f32_at(&v, 6), f32::MAX, "response e");
        assert_eq!(f32_at(&v, 7), 2.0, "response 1 returns the input");
        // the immediate selects which tokens raise exceptions
        let h = run(VFIXUPIMMPD_FLAGS.p[2], |h| {
            h.cpu.set_zmm(1, doubles(&[0.0]));
            h.cpu.set_zmm(2, [0; 4]);
        });
        assert_eq!(
            h.cpu.mxcsr & 0x3f,
            0x01 | 0x04,
            "invalid and divide-by-zero for a zero token"
        );
        let h = run(VFIXUPIMMSS, |h| {
            h.cpu.set_zmm(0, singles(&[3.0, 4.0]));
            h.cpu.set_zmm(1, singles(&[0.0, 6.0]));
            h.cpu.set_zmm(2, singles(&[f32::from_bits(0x0000_0a00)]));
        });
        // token 2 (zero) -> response 0xa = +1.0; the upper element is the destination's
        assert_eq!(f32_at(&h.cpu.zmm(0), 0), 1.0);
        assert_eq!(f32_at(&h.cpu.zmm(0), 1), 4.0);
    }
}

mod compares {
    use super::*;

    /// The 32 vcmp predicates over (a < b, a == b, unordered).
    fn predicate(p: u8, a: f64, b: f64) -> bool {
        let unordered = a.is_nan() || b.is_nan();
        let (less, equal) = (a < b, a == b);
        let greater = a > b;
        match p & 15 {
            0 => equal,
            1 => less,
            2 => less || equal,
            3 => unordered,
            4 => !equal,
            5 => !less,
            6 => !(less || equal),
            7 => !unordered,
            8 => equal || unordered,
            9 => less || unordered,
            10 => less || equal || unordered,
            11 => false,
            12 => !unordered && !equal,
            13 => greater || equal,
            14 => greater,
            _ => true,
        }
    }

    fn operands() -> (Vec<f64>, Vec<f64>) {
        let a = vec![1.0, 2.0, 3.0, f64::NAN, 5.0, -0.0, f64::INFINITY, 2.0];
        let b = vec![1.0, 3.0, 2.0, 1.0, f64::NAN, 0.0, f64::INFINITY, 2.5];
        (a, b)
    }

    #[test]
    fn vcmpps_and_vcmppd_implement_every_predicate() {
        let (a, b) = operands();
        for (p, ps, pd) in [
            (0u8, &VCMPPS_0, &VCMPPD_0),
            (1, &VCMPPS_1, &VCMPPD_1),
            (2, &VCMPPS_2, &VCMPPD_2),
            (3, &VCMPPS_3, &VCMPPD_3),
            (4, &VCMPPS_4, &VCMPPD_4),
            (5, &VCMPPS_5, &VCMPPD_5),
            (6, &VCMPPS_6, &VCMPPD_6),
            (7, &VCMPPS_7, &VCMPPD_7),
            (8, &VCMPPS_8, &VCMPPD_8),
            (12, &VCMPPS_12, &VCMPPD_12),
            (16, &VCMPPS_16, &VCMPPD_16),
            (17, &VCMPPS_17, &VCMPPD_17),
            (21, &VCMPPS_21, &VCMPPD_21),
            (24, &VCMPPS_24, &VCMPPD_24),
            (29, &VCMPPS_29, &VCMPPD_29),
            (31, &VCMPPS_31, &VCMPPD_31),
        ] {
            let h = run(pd.p[2], |h| {
                h.cpu.set_zmm(1, doubles(&a));
                h.cpu.set_zmm(2, doubles(&b));
            });
            let mut want = 0u64;
            for k in 0..8 {
                if predicate(p, a[k], b[k]) {
                    want |= 1 << k;
                }
            }
            assert_eq!(h.cpu.opmask[2], want, "vcmppd predicate {p}");
            let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
            let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
            let h = run(ps.p[2], |h| {
                h.cpu.set_zmm(1, singles(&a32));
                h.cpu.set_zmm(2, singles(&b32));
            });
            let mut want = 0u64;
            for k in 0..16 {
                if predicate(p, a[k % 8], b[k % 8]) {
                    want |= 1 << k;
                }
            }
            assert_eq!(h.cpu.opmask[2], want, "vcmpps predicate {p}");
        }
    }

    #[test]
    fn compare_results_follow_the_vector_length_and_the_write_mask() {
        let (a, b) = operands();
        let h = run(VCMPPD_1.pk(), |h| {
            h.cpu.set_zmm(1, doubles(&a));
            h.cpu.set_zmm(2, doubles(&b));
            h.cpu.opmask[1] = 0b1010_1010;
            h.cpu.opmask[2] = u64::MAX;
        });
        let mut want = 0u64;
        for k in 0..8 {
            if 0b1010_1010u64 >> k & 1 == 1 && a[k] < b[k] {
                want |= 1 << k;
            }
        }
        assert_eq!(h.cpu.opmask[2], want);
        // 256-bit: four elements, the upper mask bits are cleared
        let h = run(VCMPPD_7.y_plain(), |h| {
            h.cpu.set_zmm(1, doubles(&a));
            h.cpu.set_zmm(2, doubles(&b));
            h.cpu.opmask[2] = u64::MAX;
        });
        assert_eq!(h.cpu.opmask[2] >> 4, 0);
    }

    #[test]
    fn signaling_predicates_raise_invalid_for_quiet_nans() {
        let (a, _) = operands();
        let nan = doubles(&[f64::NAN]);
        // predicate 1 (LT_OS) signals on a QNaN; predicate 17 (LT_OQ) does not
        let h = run(VCMPPD_1.p[2], |h| {
            h.cpu.set_zmm(1, nan);
            h.cpu.set_zmm(2, doubles(&a));
        });
        assert_eq!(h.cpu.mxcsr & 1, 1);
        let h = run(VCMPPD_17.p[2], |h| {
            h.cpu.set_zmm(1, nan);
            h.cpu.set_zmm(2, doubles(&a));
        });
        assert_eq!(h.cpu.mxcsr & 1, 0);
    }

    #[test]
    fn scalar_compares_and_comis() {
        let h = run(VCMPSS_1, |h| {
            h.cpu.set_zmm(1, singles(&[1.0, 9.0]));
            h.cpu.set_zmm(2, singles(&[2.0, 0.0]));
            h.cpu.opmask[2] = u64::MAX;
        });
        assert_eq!(h.cpu.opmask[2], 1);
        let h = run(VCMPSD_12_K, |h| {
            h.cpu.set_zmm(1, doubles(&[1.0]));
            h.cpu.set_zmm(2, doubles(&[2.0]));
            h.cpu.opmask[1] = 0;
            h.cpu.opmask[2] = u64::MAX;
        });
        assert_eq!(h.cpu.opmask[2], 0, "a masked-out element compares as false");
        let eflags = |h: &Harness| h.cpu.eflags_full() & 0x45;
        let h = run(VCOMISS_SAE, |h| {
            h.cpu.set_zmm(1, singles(&[1.0]));
            h.cpu.set_zmm(2, singles(&[2.0]));
        });
        assert_eq!(eflags(&h), 0x01, "less: CF");
        let h = run(VUCOMISD, |h| {
            h.cpu.set_zmm(1, doubles(&[3.0]));
            h.cpu.set_zmm(2, doubles(&[3.0]));
        });
        assert_eq!(eflags(&h), 0x40, "equal: ZF");
        let h = run(VCOMISD_M, |h| {
            h.cpu.set_zmm(1, doubles(&[f64::NAN]));
            h.write(DATA, &1.0f64.to_le_bytes());
        });
        assert_eq!(eflags(&h), 0x45, "unordered: ZF, PF, CF");
    }
}

trait ShapeAccess {
    fn pk(&self) -> &'static [u8];
    fn y_plain(&self) -> &'static [u8];
}

impl ShapeAccess for Variants {
    /// The 512-bit merge-masked shape.
    fn pk(&self) -> &'static [u8] {
        self.k[2]
    }
    fn y_plain(&self) -> &'static [u8] {
        self.p[1]
    }
}
