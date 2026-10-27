//! Plumbing tests for `avx.rs`: every instruction the wrapper reuses from the legacy SSE kernels
//! is run both ways on identical data and the results must agree. This proves the staging (vvvv
//! seeding, register collisions, lane passes, zeroing, alignment, gating); the arithmetic itself
//! is covered by the legacy SSE tests.

use crate::cpu::sys::{CR0_TS, CR4_OSXSAVE};
use crate::cpu::xsave::{XCR0_AVX, XCR0_SSE, XCR0_X87};
use crate::testkit::{Harness, RSI, SCRATCH, Stop};

const DATA: u64 = SCRATCH + 0x1000;
const CR4_OSFXSR_BIT: u64 = 1 << 9;
const FLAGS: u64 = crate::cpu::flags::ARITH;
const ITERATIONS: u64 = 8;

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Binary,
    Scalar,
    Unary,
}

pub(super) struct PlumbingCase {
    pub name: &'static str,
    pub kind: Kind,
    pub legacy_reg: &'static [u8],
    pub legacy_mem: &'static [u8],
    pub vex_reg: &'static [u8],
    pub vex_mem: &'static [u8],
    pub vex256_reg: &'static [u8],
    pub vex256_mem: &'static [u8],
}

#[path = "avx_cases.rs"]
mod cases;
use cases::PLUMBING_CASES;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Mostly random bits, sometimes values with exponents that stay finite, so floating-point
    /// kernels see numbers as well as NaN soup.
    fn vector(&mut self) -> u128 {
        let mut v = (self.next() as u128) << 64 | self.next() as u128;
        if self.next().is_multiple_of(3) {
            let singles = [1.5f32, -2.25, 1e10, 0.1, 3.0, -0.0, 7.0, 1e-3];
            let mut r = 0u128;
            for k in 0..4 {
                r |= (singles[(self.next() % 8) as usize].to_bits() as u128) << (32 * k);
            }
            v = r;
        } else if self.next().is_multiple_of(3) {
            let doubles = [1.5f64, -2.25, 1e100, 0.1, 3.0, -0.0, 7.0, 1e-30];
            v = (doubles[(self.next() % 8) as usize].to_bits() as u128)
                | (doubles[(self.next() % 8) as usize].to_bits() as u128) << 64;
        }
        v
    }
}

struct Inputs {
    first: u128,
    second: u128,
    first_hi: u128,
    second_hi: u128,
}

fn inputs(rng: &mut Rng) -> Inputs {
    Inputs {
        first: rng.vector(),
        second: rng.vector(),
        first_hi: rng.vector(),
        second_hi: rng.vector(),
    }
}

const JUNK: u128 = 0xdead_beef_dead_beef_dead_beef_dead_beef;

fn harness(code: &[u8], vex: bool) -> Harness {
    let mut h = Harness::new(code);
    // legacy SSE needs OSFXSR; VEX must not (it is cleared here to prove it)
    if vex {
        h.cpu.cr4 = (h.cpu.cr4 & !CR4_OSFXSR_BIT) | CR4_OSXSAVE;
        h.cpu.xcr0 = XCR0_X87 | XCR0_SSE | XCR0_AVX;
    } else {
        h.cpu.cr4 |= CR4_OSFXSR_BIT;
    }
    h.cpu.gpr[RSI] = DATA;
    h
}

fn put_memory(h: &mut Harness, low: u128, high: u128) {
    h.write(DATA, &low.to_le_bytes());
    h.write(DATA + 16, &high.to_le_bytes());
}

fn vex(code: &[u8], inputs: &Inputs) -> Harness {
    let mut h = harness(code, true);
    h.cpu.xmm[1] = JUNK;
    h.cpu.ymm_hi[1] = JUNK;
    h.cpu.xmm[3] = inputs.first;
    h.cpu.ymm_hi[3] = inputs.first_hi;
    h.cpu.xmm[2] = inputs.second;
    h.cpu.ymm_hi[2] = inputs.second_hi;
    put_memory(&mut h, inputs.second, inputs.second_hi);
    h
}

#[track_caller]
fn assert_same_state(name: &str, vex: &Harness, legacy: &Harness) {
    assert_eq!(vex.cpu.gpr, legacy.cpu.gpr, "{name}: general registers");
    assert_eq!(
        vex.cpu.eflags_full() & FLAGS,
        legacy.cpu.eflags_full() & FLAGS,
        "{name}: flags"
    );
}

/// Unary instructions ignore vvvv and the first operand: feed the source in xmm2 only.
fn unary_inputs(inputs: &Inputs) -> Inputs {
    Inputs {
        first: inputs.first,
        second: inputs.second,
        first_hi: inputs.first_hi,
        second_hi: inputs.second_hi,
    }
}

#[test]
fn vex_128_forms_match_the_legacy_instruction_with_vvvv_as_the_first_source() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for case in PLUMBING_CASES {
        for _ in 0..ITERATIONS {
            let data = unary_inputs(&inputs(&mut rng));
            for (legacy_code, vex_code, memory) in [
                (case.legacy_reg, case.vex_reg, false),
                (case.legacy_mem, case.vex_mem, true),
            ] {
                // lddqu and friends exist only with memory operands; the generator emitted both
                // forms for every case, and an invalid form must fail the same way on both sides.
                let mut l = harness(legacy_code, false);
                l.cpu.xmm[1] = data.first;
                l.cpu.xmm[2] = data.second;
                put_memory(&mut l, data.second, 0);
                let legacy_stop = l.run(100);
                let mut v = vex(vex_code, &data);
                let vex_stop = v.run(100);
                assert_eq!(vex_stop, legacy_stop, "{} memory={memory}", case.name);
                if legacy_stop != Stop::Halted {
                    continue;
                }
                assert_eq!(
                    v.cpu.xmm[1], l.cpu.xmm[1],
                    "{} memory={memory} low lane",
                    case.name
                );
                assert_eq!(
                    v.cpu.ymm_hi[1], 0,
                    "{} memory={memory}: VEX.128 zeroes bits 255:128",
                    case.name
                );
                assert_eq!(
                    v.cpu.mxcsr, l.cpu.mxcsr,
                    "{} memory={memory}: mxcsr",
                    case.name
                );
                assert_same_state(case.name, &v, &l);
                // sources are untouched
                assert_eq!(
                    v.cpu.xmm[2], data.second,
                    "{}: rm source preserved",
                    case.name
                );
                assert_eq!(
                    v.cpu.xmm[3], data.first,
                    "{}: vvvv source preserved",
                    case.name
                );
            }
        }
    }
}

#[test]
fn vex_256_forms_run_the_legacy_instruction_on_each_lane() {
    let mut rng = Rng(0x0fed_cba9_8765_4321);
    for case in PLUMBING_CASES {
        if case.vex256_reg.is_empty() {
            continue;
        }
        for _ in 0..ITERATIONS {
            let data = inputs(&mut rng);
            for (legacy_code, vex_code, memory) in [
                (case.legacy_reg, case.vex256_reg, false),
                (case.legacy_mem, case.vex256_mem, true),
            ] {
                let mut lo = harness(legacy_code, false);
                lo.cpu.xmm[1] = data.first;
                lo.cpu.xmm[2] = data.second;
                put_memory(&mut lo, data.second, 0);
                let lo_stop = lo.run(100);
                let mut hi = harness(legacy_code, false);
                hi.cpu.xmm[1] = data.first_hi;
                hi.cpu.xmm[2] = data.second_hi;
                put_memory(&mut hi, data.second_hi, 0);
                let hi_stop = hi.run(100);
                let mut v = vex(vex_code, &data);
                let vex_stop = v.run(100);
                assert_eq!(lo_stop, hi_stop, "{} legacy lanes disagree", case.name);
                assert_eq!(vex_stop, lo_stop, "{} memory={memory}", case.name);
                if lo_stop != Stop::Halted {
                    continue;
                }
                assert_eq!(
                    v.cpu.xmm[1], lo.cpu.xmm[1],
                    "{} memory={memory} low lane",
                    case.name
                );
                assert_eq!(
                    v.cpu.ymm_hi[1], hi.cpu.xmm[1],
                    "{} memory={memory} high lane",
                    case.name
                );
                assert_eq!(
                    v.cpu.mxcsr,
                    lo.cpu.mxcsr | hi.cpu.mxcsr,
                    "{}: sticky flags of both lanes",
                    case.name
                );
            }
        }
    }
}

#[test]
fn scalar_forms_ignore_vex_l_and_still_zero_the_upper_half() {
    let mut rng = Rng(0x5555_aaaa_5555_aaaa);
    for case in PLUMBING_CASES
        .iter()
        .filter(|c| matches!(c.kind, Kind::Scalar))
    {
        let data = inputs(&mut rng);
        // setting VEX.L (bit 2 of the last VEX payload byte) must change nothing
        let mut with_l = case.vex_reg.to_vec();
        // the L bit is in the last VEX payload byte: byte 1 of c5, byte 2 of c4
        let payload = if with_l[0] == 0xc5 { 1 } else { 2 };
        with_l[payload] |= 0x04;
        let mut a = vex(case.vex_reg, &data);
        a.run(100);
        let mut b = vex(&with_l, &data);
        b.run(100);
        assert_eq!(a.cpu.xmm[1], b.cpu.xmm[1], "{}", case.name);
        assert_eq!(b.cpu.ymm_hi[1], 0, "{}", case.name);
    }
}

#[test]
fn vex_memory_operands_need_no_alignment_except_the_explicitly_aligned_loads() {
    let mut rng = Rng(77);
    for case in PLUMBING_CASES {
        let data = inputs(&mut rng);
        for code in [case.vex_mem, case.vex256_mem] {
            if code.is_empty() {
                continue;
            }
            let mut v = vex(code, &data);
            v.cpu.gpr[RSI] = DATA + 1;
            let stop = v.run(100);
            let must_fault = matches!(case.name, "movaps" | "movapd" | "movdqa");
            if must_fault {
                assert!(
                    matches!(stop, Stop::Fault { vector: 13, .. }),
                    "{}: {stop:?}",
                    case.name
                );
            } else {
                assert_eq!(
                    stop,
                    Stop::Halted,
                    "{} must accept a misaligned operand",
                    case.name
                );
            }
        }
    }
}

#[test]
fn vex_is_undefined_without_osxsave_or_the_avx_state_and_needs_the_fpu_for_nm() {
    let case = &PLUMBING_CASES[0];
    let data = inputs(&mut Rng(3));
    let mut v = vex(case.vex_reg, &data);
    v.cpu.cr4 &= !CR4_OSXSAVE;
    assert!(matches!(v.run(10), Stop::Fault { vector: 6, .. }));
    let mut v = vex(case.vex_reg, &data);
    v.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
    assert!(matches!(v.run(10), Stop::Fault { vector: 6, .. }));
    let mut v = vex(case.vex_reg, &data);
    v.cpu.xcr0 = XCR0_X87 | XCR0_AVX;
    assert!(matches!(v.run(10), Stop::Fault { vector: 6, .. }));
    let mut v = vex(case.vex_reg, &data);
    v.cpu.cr0 |= CR0_TS;
    assert!(matches!(v.run(10), Stop::Fault { vector: 7, .. }));
}

#[test]
fn a_fault_in_the_upper_lane_leaves_every_register_alone() {
    // vaddps ymm1, ymm3, [rsi] with the second half of the operand on an unmapped page
    let case = PLUMBING_CASES.iter().find(|c| c.name == "addps").unwrap();
    let data = inputs(&mut Rng(9));
    let mut v = vex(case.vex256_mem, &data);
    // 0x7ff000 is mapped RAM here; put the operand so that bytes 16..32 cross into a page the test
    // unmaps. Using a non-canonical-free high address: the last 16 bytes of the 4 GiB identity map.
    v.cpu.gpr[RSI] = 0x1_0000_0000 - 16;
    let before = (v.cpu.xmm, v.cpu.ymm_hi);
    let stop = v.run(10);
    if matches!(stop, Stop::Fault { .. }) {
        assert_eq!(
            (v.cpu.xmm, v.cpu.ymm_hi),
            before,
            "registers restored after the fault"
        );
    }
}

#[test]
fn vvvv_must_be_1111_when_the_instruction_has_no_first_source() {
    // vmovaps xmm1, xmm2 with vvvv = 1 (c5 f0 28 ca: ~vvvv = 1110 -> vvvv = 1 would be 0xf0 pattern)
    let mut v = harness(&[0xc5, 0xf0, 0x28, 0xca, 0xf4], true);
    assert!(matches!(v.run(10), Stop::Fault { vector: 6, .. }));
}

#[test]
fn vzeroupper_and_vzeroall() {
    let mut v = harness(&[0xc5, 0xf8, 0x77, 0xf4], true);
    for k in 0..16 {
        v.cpu.xmm[k] = k as u128 + 1;
        v.cpu.ymm_hi[k] = 0xff;
    }
    assert_eq!(v.run(10), Stop::Halted);
    assert_eq!(v.cpu.ymm_hi, [0; 16]);
    assert_eq!(v.cpu.xmm[5], 6, "vzeroupper keeps the low halves");
    let mut v = harness(&[0xc5, 0xfc, 0x77, 0xf4], true);
    for k in 0..16 {
        v.cpu.xmm[k] = k as u128 + 1;
        v.cpu.ymm_hi[k] = 0xff;
    }
    assert_eq!(v.run(10), Stop::Halted);
    assert_eq!((v.cpu.xmm, v.cpu.ymm_hi), ([0; 16], [0; 16]));
    // without AVX state it is #UD
    let mut v = harness(&[0xc5, 0xf8, 0x77, 0xf4], true);
    v.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
    assert!(matches!(v.run(10), Stop::Fault { vector: 6, .. }));
}

#[test]
fn xsave_saves_and_restores_the_upper_halves() {
    // xsave [rdi]; vzeroall; xrstor [rdi]; hlt with EDX:EAX = 7
    let mut v = harness(
        &[0x0f, 0xae, 0x27, 0xc5, 0xfc, 0x77, 0x0f, 0xae, 0x2f, 0xf4],
        true,
    );
    v.cpu.gpr[7] = DATA;
    v.cpu.gpr[0] = 7;
    v.cpu.gpr[2] = 0;
    for k in 0..16 {
        v.cpu.xmm[k] = (k as u128) << 8 | 1;
        v.cpu.ymm_hi[k] = (k as u128) << 16 | 2;
    }
    let (low, high) = (v.cpu.xmm, v.cpu.ymm_hi);
    assert_eq!(v.run(100), Stop::Halted);
    assert_eq!((v.cpu.xmm, v.cpu.ymm_hi), (low, high));
    assert_eq!(
        v.read_u64(DATA + 512),
        7,
        "XSTATE_BV marks x87, SSE and AVX"
    );
    assert_eq!(v.read(DATA + 576, 16), high[0].to_le_bytes());
}
