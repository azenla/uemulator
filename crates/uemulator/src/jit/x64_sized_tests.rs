//! Encoder tests for the sized, unary and conditional forms the block generator uses at every
//! operand size. Expected bytes come from `hack/asm.sh` (the assembly is in each case). Included
//! as a child of `x64_tests`, which has the case runner and `run`.

use super::*;

/// Condition code 6 (BE); the generator never names it, so the encoder has no constant.
const CONDITION_BE: u8 = 6;
const CONDITION_L: u8 = 0xc;
const CONDITION_G: u8 = 0xf;

#[rustfmt::skip]
#[test]
fn sized_unary_and_conditional_forms_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("add al, cl", |a| a.alu_rr_sized(ADD, RAX, RCX, 1), &[0x00, 0xc8]),
        ("add ax, cx", |a| a.alu_rr_sized(ADD, RAX, RCX, 2), &[0x66, 0x01, 0xc8]),
        ("add eax, ecx", |a| a.alu_rr_sized(ADD, RAX, RCX, 4), &[0x01, 0xc8]),
        ("add rax, rcx", |a| a.alu_rr_sized(ADD, RAX, RCX, 8), &[0x48, 0x01, 0xc8]),
        ("sub sil, dil", |a| a.alu_rr_sized(SUB, RSI, RDI, 1), &[0x40, 0x28, 0xfe]),
        ("xor r9b, r10b", |a| a.alu_rr_sized(XOR, R9, R10, 1), &[0x45, 0x30, 0xd1]),
        ("cmp r8w, r9w", |a| a.alu_rr_sized(CMP, R8, R9, 2), &[0x66, 0x45, 0x39, 0xc8]),
        ("and r9d, r10d", |a| a.alu_rr_sized(AND, R9, R10, 4), &[0x45, 0x21, 0xd1]),
        ("or r11, r12", |a| a.alu_rr_sized(OR, R11, R12, 8), &[0x4d, 0x09, 0xe3]),
        ("test ecx, 4", |a| a.test_ri(RCX, 4, false), &[0xf7, 0xc1, 0x04, 0x00, 0x00, 0x00]),
        ("test rcx, 4", |a| a.test_ri(RCX, 4, true), &[0x48, 0xf7, 0xc1, 0x04, 0x00, 0x00, 0x00]),
        ("test r9d, 0x12345678", |a| a.test_ri(R9, 0x1234_5678, false), &[0x41, 0xf7, 0xc1, 0x78, 0x56, 0x34, 0x12]),
        ("not al", |a| a.not_r(RAX, 1), &[0xf6, 0xd0]),
        ("not cx", |a| a.not_r(RCX, 2), &[0x66, 0xf7, 0xd1]),
        ("not ecx", |a| a.not_r(RCX, 4), &[0xf7, 0xd1]),
        ("not r9", |a| a.not_r(R9, 8), &[0x49, 0xf7, 0xd1]),
        ("neg sil", |a| a.neg_r(RSI, 1), &[0x40, 0xf6, 0xde]),
        ("neg ax", |a| a.neg_r(RAX, 2), &[0x66, 0xf7, 0xd8]),
        ("neg r10d", |a| a.neg_r(R10, 4), &[0x41, 0xf7, 0xda]),
        ("neg rdx", |a| a.neg_r(RDX, 8), &[0x48, 0xf7, 0xda]),
        ("inc al", |a| a.inc_r(RAX, 1), &[0xfe, 0xc0]),
        ("inc cx", |a| a.inc_r(RCX, 2), &[0x66, 0xff, 0xc1]),
        ("inc r9d", |a| a.inc_r(R9, 4), &[0x41, 0xff, 0xc1]),
        ("inc rax", |a| a.inc_r(RAX, 8), &[0x48, 0xff, 0xc0]),
        ("dec dil", |a| a.dec_r(RDI, 1), &[0x40, 0xfe, 0xcf]),
        ("dec r8w", |a| a.dec_r(R8, 2), &[0x66, 0x41, 0xff, 0xc8]),
        ("dec ecx", |a| a.dec_r(RCX, 4), &[0xff, 0xc9]),
        ("dec r10", |a| a.dec_r(R10, 8), &[0x49, 0xff, 0xca]),
        ("bt ecx, 0", |a| a.bt_ri(RCX, 0, false), &[0x0f, 0xba, 0xe1, 0x00]),
        ("bt r9, 5", |a| a.bt_ri(R9, 5, true), &[0x49, 0x0f, 0xba, 0xe1, 0x05]),
        ("imul eax, ecx", |a| a.imul_rr(RAX, RCX, false), &[0x0f, 0xaf, 0xc1]),
        ("imul rax, r9", |a| a.imul_rr(RAX, R9, true), &[0x49, 0x0f, 0xaf, 0xc1]),
        ("imul r10d, edx", |a| a.imul_rr(R10, RDX, false), &[0x44, 0x0f, 0xaf, 0xd2]),
        ("cmove eax, ecx", |a| a.cmov_rr(E, RAX, RCX, false), &[0x0f, 0x44, 0xc1]),
        ("cmovne rax, r9", |a| a.cmov_rr(NE, RAX, R9, true), &[0x49, 0x0f, 0x45, 0xc1]),
        ("cmovl r10d, edx", |a| a.cmov_rr(CONDITION_L, R10, RDX, false), &[0x44, 0x0f, 0x4c, 0xd2]),
        ("cmovg r8, r15", |a| a.cmov_rr(CONDITION_G, R8, R15, true), &[0x4d, 0x0f, 0x4f, 0xc7]),
        ("shl eax, cl", |a| a.shift_cl(SHL, RAX, false), &[0xd3, 0xe0]),
        ("shr rdx, cl", |a| a.shift_cl(SHR, RDX, true), &[0x48, 0xd3, 0xea]),
        ("sar r9d, cl", |a| a.shift_cl(SAR, R9, false), &[0x41, 0xd3, 0xf9]),
        ("sar r10, cl", |a| a.shift_cl(SAR, R10, true), &[0x49, 0xd3, 0xfa]),
        ("sar eax, 31", |a| a.shift_ri(SAR, RAX, 31, false), &[0xc1, 0xf8, 0x1f]),
        ("sar rdx, 63", |a| a.shift_ri(SAR, RDX, 63, true), &[0x48, 0xc1, 0xfa, 0x3f]),
        ("seto r8b", |a| a.setcc(O, R8), &[0x41, 0x0f, 0x90, 0xc0]),
        ("setb sil", |a| a.setcc(B, RSI), &[0x40, 0x0f, 0x92, 0xc6]),
        ("setae al", |a| a.setcc(AE, RAX), &[0x0f, 0x93, 0xc0]),
        ("setne r10b", |a| a.setcc(NE, R10), &[0x41, 0x0f, 0x95, 0xc2]),
        ("setbe dil", |a| a.setcc(CONDITION_BE, RDI), &[0x40, 0x0f, 0x96, 0xc7]),
        ("cdq", |a| a.cdq(false), &[0x99]),
        ("cqo", |a| a.cdq(true), &[0x48, 0x99]),
    ];
    check_encodings(cases);
}

#[test]
fn sized_operations_compute_at_their_width_and_set_that_widths_flags() {
    // a byte add that overflows leaves the upper bits of rax alone
    assert_eq!(
        run(|a| a.alu_rr_sized(ADD, RAX, RCX, 1), 0x1234_5680, 0x80),
        0x1234_5600
    );
    assert_eq!(
        run(|a| a.alu_rr_sized(ADD, RAX, RCX, 2), 0x1234_ffff, 2),
        0x1234_0001
    );
    // and sets CF for the byte, not for the register
    assert_eq!(
        run(
            |a| {
                a.alu_rr_sized(ADD, RAX, RCX, 1);
                a.setcc(B, RDX);
                a.ext_rr(RAX, RDX, 1, false);
            },
            0x80,
            0x80
        ),
        1
    );
    assert_eq!(run(|a| a.neg_r(RAX, 4), 1, 0), 0xffff_ffff);
    assert_eq!(run(|a| a.not_r(RAX, 1), 0xff00, 0), 0xffff);
    assert_eq!(run(|a| a.inc_r(RAX, 1), 0xff, 0), 0);
    assert_eq!(run(|a| a.dec_r(RAX, 2), 0x1_0000, 0), 0x1_ffff);
    assert_eq!(
        run(|a| a.imul_rr(RAX, RCX, false), 0xffff_fffe, 3),
        0xffff_fffa
    );
    assert_eq!(
        run(|a| a.shift_cl(SAR, RAX, true), !0u64 << 8, 4),
        !0u64 << 4
    );
}

#[test]
fn inc_keeps_the_carry_that_bt_put_in() {
    let carry_after = |value: u64| {
        run(
            |a| {
                a.bt_ri(RCX, 0, false);
                a.inc_r(RAX, 4);
                a.setcc(B, RDX);
                a.ext_rr(RAX, RDX, 1, false);
            },
            0,
            value,
        )
    };
    assert_eq!(carry_after(1), 1);
    assert_eq!(carry_after(2), 0);
}

#[test]
fn cmov_moves_only_when_the_condition_holds_and_zero_extends_either_way() {
    let cmov = |condition: u8| {
        run(
            |a| {
                a.bt_ri(RCX, 0, false);
                a.cmov_rr(condition, RAX, RCX, false);
            },
            0xffff_ffff_0000_0005,
            1,
        )
    };
    assert_eq!(cmov(B), 1, "moved");
    assert_eq!(cmov(AE), 5, "not moved, upper half still cleared");
}

#[test]
fn cdq_and_cqo_sign_extend_into_rdx() {
    let rdx_after = |w: bool, value: u64| {
        run(
            |a| {
                a.mov_rr(RDX, RCX, true);
                a.cdq(w);
                a.mov_rr(RAX, RDX, true);
            },
            value,
            u64::MAX,
        )
    };
    assert_eq!(rdx_after(false, 0x8000_0000), 0xffff_ffff);
    assert_eq!(
        rdx_after(false, 0x7fff_ffff),
        0,
        "the upper half is cleared"
    );
    assert_eq!(rdx_after(true, 1 << 63), u64::MAX);
    assert_eq!(rdx_after(true, 1 << 62), 0);
}

#[test]
fn cdq_does_not_touch_the_flags() {
    for w in [false, true] {
        let zero_flag = run(
            |a| {
                a.alu_rr(CMP, RAX, RAX, true);
                a.cdq(w);
                a.mov_imm(RCX, 0);
                a.setcc(E, RCX);
                a.mov_rr(RAX, RCX, true);
            },
            5,
            0,
        );
        assert_eq!(zero_flag, 1);
    }
}
