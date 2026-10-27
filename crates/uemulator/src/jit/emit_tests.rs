//! The pure helpers of the aarch64 block compiler: the x86-condition to NZCV mapping (checked
//! against a model of both flag definitions), the constant-mask fallback, and small decoders.

use super::*;
use crate::cpu::decode::NO_SEG;
use crate::cpu::flags::{self, CF, OF, PF, SF, ZF};
use crate::sys::ExecMem;

fn size_in_bits(size: u8) -> u32 {
    size as u32 * 8
}

/// aarch64 NZCV after the instruction `kind` corresponds to (`adds`, `subs`, `ands`) at `size`.
struct Nzcv {
    n: bool,
    z: bool,
    c: bool,
    v: bool,
}

fn nzcv_after(kind: FlagKind, size: u8, a: u64, b: u64) -> (u64, Nzcv) {
    let bits = size_in_bits(size);
    let mask = flags::mask(size);
    let sign = 1u64 << (bits - 1);
    let (a, b) = (a & mask, b & mask);
    let (result, c, v) = match kind {
        FlagKind::Add => {
            let sum = a as u128 + b as u128;
            let result = (sum as u64) & mask;
            (
                result,
                sum >> bits != 0,
                (!(a ^ b) & (a ^ result)) & sign != 0,
            )
        }
        FlagKind::Sub => {
            let result = a.wrapping_sub(b) & mask;
            // aarch64 carry is "no borrow"
            (result, a >= b, ((a ^ b) & (a ^ result)) & sign != 0)
        }
        FlagKind::Logic => ((a & b) & mask, false, false),
        // The host op is `adds`/`subs` with the immediate 1; the caller passes b = 1.
        FlagKind::Inc => {
            let result = a.wrapping_add(1) & mask;
            (result, false, (!(a ^ 1) & (a ^ result)) & sign != 0)
        }
        FlagKind::Dec => {
            let result = a.wrapping_sub(1) & mask;
            (result, false, ((a ^ 1) & (a ^ result)) & sign != 0)
        }
    };
    (
        result,
        Nzcv {
            n: result & sign != 0,
            z: result == 0,
            c,
            v,
        },
    )
}

fn aarch64_condition_holds(condition: u32, f: &Nzcv) -> bool {
    match condition {
        EQ => f.z,
        NE => !f.z,
        CS => f.c,
        CC => !f.c,
        MI => f.n,
        PL => !f.n,
        VS => f.v,
        VC => !f.v,
        HI => f.c && !f.z,
        LS => !(f.c && !f.z),
        GE => f.n == f.v,
        LT => f.n != f.v,
        GT => !f.z && f.n == f.v,
        LE => f.z || f.n != f.v,
        other => panic!("condition {other}"),
    }
}

/// The x86 condition codes (low nibble of Jcc) evaluated from EFLAGS.
fn x86_condition_holds(condition: u8, eflags: u64) -> bool {
    let (of, cf, zf, sf, pf) = (
        eflags & OF != 0,
        eflags & CF != 0,
        eflags & ZF != 0,
        eflags & SF != 0,
        eflags & PF != 0,
    );
    match condition {
        0 => of,
        1 => !of,
        2 => cf,
        3 => !cf,
        4 => zf,
        5 => !zf,
        6 => cf || zf,
        7 => !(cf || zf),
        8 => sf,
        9 => !sf,
        10 => pf,
        11 => !pf,
        12 => sf != of,
        13 => sf == of,
        14 => zf || sf != of,
        _ => !zf && sf == of,
    }
}

const OPERAND_VALUES: [u64; 12] = [
    0,
    1,
    2,
    0x7f,
    0x80,
    0xff,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    u64::MAX,
];

fn lazy_op(kind: FlagKind) -> LazyFlagsOp {
    match kind {
        FlagKind::Add => LazyFlagsOp::Add,
        FlagKind::Sub => LazyFlagsOp::Sub,
        FlagKind::Logic => LazyFlagsOp::Logic,
        FlagKind::Inc => LazyFlagsOp::Inc,
        FlagKind::Dec => LazyFlagsOp::Dec,
    }
}

#[test]
fn every_mapped_condition_matches_the_x86_flags_the_same_operation_produces() {
    for kind in [FlagKind::Add, FlagKind::Sub, FlagKind::Logic] {
        for size in [4u8, 8] {
            for a in OPERAND_VALUES {
                for b in OPERAND_VALUES {
                    let (result, nzcv) = nzcv_after(kind, size, a, b);
                    let eflags = flags::compute(lazy_op(kind), size, result, b, 0, 0);
                    for condition in 0..16u8 {
                        let expected = x86_condition_holds(condition, eflags);
                        match cond_map(kind, condition) {
                            Some(CondMap::Cond(host)) => assert_eq!(
                                aarch64_condition_holds(host, &nzcv),
                                expected,
                                "{kind:?} size {size} {a:#x}, {b:#x}: condition {condition}"
                            ),
                            Some(CondMap::Always) => assert!(expected, "{kind:?} cond {condition}"),
                            Some(CondMap::Never) => assert!(!expected, "{kind:?} cond {condition}"),
                            None => {}
                        }
                    }
                }
            }
        }
    }
}

/// `inc`/`dec` keep CF whatever it was, so every condition the map offers must hold for both
/// incoming carries; the ones that read CF must not be offered at all.
#[test]
fn inc_and_dec_conditions_are_right_for_either_incoming_carry() {
    for kind in [FlagKind::Inc, FlagKind::Dec] {
        for size in [4u8, 8] {
            for a in OPERAND_VALUES {
                let (result, nzcv) = nzcv_after(kind, size, a, 1);
                for carry in [0, 1] {
                    let eflags = flags::compute(lazy_op(kind), size, result, carry, 0, 0);
                    for condition in 0..16u8 {
                        let expected = x86_condition_holds(condition, eflags);
                        match cond_map(kind, condition) {
                            Some(CondMap::Cond(host)) => assert_eq!(
                                aarch64_condition_holds(host, &nzcv),
                                expected,
                                "{kind:?} size {size} {a:#x} carry {carry}: condition {condition}"
                            ),
                            Some(_) => panic!("{kind:?} condition {condition} is not constant"),
                            None => {}
                        }
                    }
                }
            }
        }
        for condition in [2, 3, 6, 7, 10, 11] {
            assert!(cond_map(kind, condition).is_none(), "{kind:?} {condition}");
        }
    }
}

#[test]
fn parity_and_add_below_or_equal_have_no_host_condition() {
    for kind in [FlagKind::Add, FlagKind::Sub, FlagKind::Logic] {
        assert!(cond_map(kind, 10).is_none(), "{kind:?} p");
        assert!(cond_map(kind, 11).is_none(), "{kind:?} np");
    }
    assert!(
        cond_map(FlagKind::Add, 6).is_none(),
        "add: be needs CF | ZF"
    );
    assert!(
        cond_map(FlagKind::Add, 7).is_none(),
        "add: a needs !CF & !ZF"
    );
    assert!(cond_map(FlagKind::Sub, 6).is_some());
    assert!(cond_map(FlagKind::Logic, 6).is_some());
}

#[test]
fn logic_never_overflows_or_carries() {
    assert!(
        matches!(cond_map(FlagKind::Logic, 0), Some(CondMap::Never)),
        "o"
    );
    assert!(
        matches!(cond_map(FlagKind::Logic, 1), Some(CondMap::Always)),
        "no"
    );
    assert!(
        matches!(cond_map(FlagKind::Logic, 2), Some(CondMap::Never)),
        "b"
    );
    assert!(
        matches!(cond_map(FlagKind::Logic, 3), Some(CondMap::Always)),
        "ae"
    );
}

#[test]
fn cond_map_uses_only_the_low_four_bits_of_the_condition() {
    for condition in 0..16u8 {
        let high = condition | 0x70;
        assert_eq!(
            format!(
                "{:?}",
                cond_map(FlagKind::Sub, condition).map(|m| matches!(m, CondMap::Cond(_)))
            ),
            format!(
                "{:?}",
                cond_map(FlagKind::Sub, high).map(|m| matches!(m, CondMap::Cond(_)))
            ),
        );
    }
}

#[test]
fn the_flag_kinds_map_to_the_lazy_ops_the_interpreter_understands() {
    // `store_lazy_op` writes these discriminants into `Cpu::lazy_op`; they must stay the
    // `LazyFlagsOp` values the interpreter decodes (Add = 1, Sub = 3, Logic = 5).
    assert_eq!(LazyFlagsOp::Add as u8, 1);
    assert_eq!(LazyFlagsOp::Sub as u8, 3);
    assert_eq!(LazyFlagsOp::Logic as u8, 5);
}

fn insn_with(op: u16, rex: u8, seg: u8) -> Insn {
    Insn {
        op,
        rex,
        seg,
        ..Default::default()
    }
}

#[test]
fn opcode_register_combines_the_opcode_bits_with_rex_b() {
    for low in 0..8u16 {
        assert_eq!(opcode_register(&insn_with(0x50 + low, 0, 0xff)), low as u8);
        assert_eq!(
            opcode_register(&insn_with(0x58 + low, REX_B, 0xff)),
            8 + low as u8
        );
    }
    assert_eq!(
        opcode_register(&insn_with(0xbf, 0x4f & 0xf, 0xff)),
        15,
        "REX.W/R/X do not matter, only B"
    );
}

#[test]
fn only_fs_and_gs_overrides_add_a_segment_base() {
    for seg in [0u8, 1, 2, 3, NO_SEG] {
        assert!(!has_segment_base(&insn_with(0, 0, seg)), "seg {seg}");
    }
    assert!(has_segment_base(&insn_with(0, 0, SEG_FS as u8)));
    assert!(has_segment_base(&insn_with(0, 0, SEG_GS as u8)));
}

#[test]
fn immediates_are_sign_extended_for_64_bit_operations_and_truncated_for_32_bit() {
    assert_eq!(Gen::immediate_for(8, -1), u64::MAX);
    assert_eq!(Gen::immediate_for(4, -1), 0xffff_ffff);
    assert_eq!(Gen::immediate_for(8, 0x7fff_ffff), 0x7fff_ffff);
    assert_eq!(Gen::immediate_for(4, i32::MIN as i64), 0x8000_0000);
}

#[test]
fn memory_operands_with_address_size_override_or_lock_stay_out_of_line() {
    let plain = Insn::default();
    assert!(Gen::memory_inlinable(&plain));
    assert!(!Gen::memory_inlinable(&Insn {
        addr32: true,
        ..Default::default()
    }));
    assert!(!Gen::memory_inlinable(&Insn {
        lock: true,
        ..Default::default()
    }));
    for seg in [0u8, 3, 4, 5, NO_SEG] {
        assert!(
            Gen::memory_inlinable(&Insn {
                seg,
                ..Default::default()
            }),
            "seg {seg}"
        );
    }
}

const DUMMY_ADDRESSES: Addrs = Addrs {
    exec: 0,
    cond: 0,
    exit_ok: 0,
    exit_fail: 0,
    chain: 0,
    load: [0; 4],
    load_signed: [0; 3],
    load_stack: 0,
    store: [0; 4],
    store_keep: [0; 4],
    store_stack: 0,
};

fn and_constant_result(constant: u64, input: u64) -> (u64, usize) {
    let _guard = crate::jit::serialize_jit_tests();
    let memory = ExecMem::new(16384).unwrap();
    let insns: [Insn; 0] = [];
    let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
    generator.and_constant(0, 1, constant);
    let words_for_mask = generator.asm.code.len();
    generator.asm.ret();
    generator.asm.finish();
    memory.write(0, &generator.asm.code);
    let function: extern "C" fn(u64, u64) -> u64 =
        unsafe { std::mem::transmute(memory.base() as usize) };
    (function(0, input), words_for_mask)
}

#[test]
fn and_constant_uses_one_instruction_for_an_encodable_mask() {
    let (result, words) = and_constant_result(0x3ff, 0xffff_ffff_ffff_ffff);
    assert_eq!(result, 0x3ff);
    assert_eq!(words, 1);
    let (result, words) = and_constant_result(!0xfff, 0x1234_5678_9abc_def0);
    assert_eq!(result, 0x1234_5678_9abc_d000);
    assert_eq!(words, 1);
}

#[test]
fn and_constant_falls_back_to_a_register_for_a_mask_that_is_not_a_bitmask_immediate() {
    // 0x1234 and 0xdead_beef are not rotated runs of ones, so `and_imm` would emit nothing.
    for mask in [0x1234u64, 0xdead_beef, 0x0123_4567_89ab_cdef, 0] {
        let (result, words) = and_constant_result(mask, 0xffff_ffff_0f0f_0f0f);
        assert_eq!(result, mask & 0xffff_ffff_0f0f_0f0f, "mask {mask:#x}");
        assert!(words > 1, "mask {mask:#x} needs the constant materialised");
    }
}

#[test]
fn and_constant_does_not_disturb_the_value_register_when_destination_differs() {
    let _guard = crate::jit::serialize_jit_tests();
    let memory = ExecMem::new(16384).unwrap();
    let insns: [Insn; 0] = [];
    let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
    generator.and_constant(2, 1, 0x1234);
    generator.asm.mov(0, 1);
    generator.asm.add(0, 0, 2, true);
    generator.asm.ret();
    generator.asm.finish();
    memory.write(0, &generator.asm.code);
    let function: extern "C" fn(u64, u64) -> u64 =
        unsafe { std::mem::transmute(memory.base() as usize) };
    assert_eq!(function(0, 0xffff), 0xffff + 0x1234);
}
