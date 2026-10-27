use super::*;

fn decoded(bytes: &[u8]) -> Insn {
    decode(bytes).unwrap_or_else(|error| panic!("{bytes:02x?} failed to decode: {error:?}"))
}

/// The assembler's byte count is the independent length oracle: the decoder must consume exactly
/// the bytes of one instruction, and every shorter prefix of it must report `Truncated` (x86
/// instruction encodings are prefix-free, so the decoder can never settle early).
#[track_caller]
fn assert_length(bytes: &[u8]) {
    assert_eq!(decoded(bytes).len as usize, bytes.len(), "{bytes:02x?}");
    for shorter in 0..bytes.len() {
        assert_eq!(
            decode(&bytes[..shorter]).err(),
            Some(DecodeErr::Truncated),
            "{:02x?} cut to {shorter} bytes",
            bytes
        );
    }
}

/// Trailing bytes after the instruction must not influence the result.
#[track_caller]
fn assert_ignores_trailing_bytes(bytes: &[u8]) {
    let mut longer = bytes.to_vec();
    longer.extend_from_slice(&[0x90; 4]);
    assert_eq!(decoded(&longer).len as usize, bytes.len(), "{bytes:02x?}");
}

type PrefixCheck = fn(&Insn) -> bool;

const SEG_NAMES_AND_PREFIXES: [(u8, usize); 6] = [
    (0x26, SEG_ES),
    (0x2e, SEG_CS),
    (0x36, SEG_SS),
    (0x3e, SEG_DS),
    (0x64, SEG_FS),
    (0x65, SEG_GS),
];

mod prefixes {
    use super::*;

    #[test]
    fn no_prefix_defaults() {
        let nop = decoded(&[0x90]);
        assert_eq!(nop.len, 1);
        assert_eq!(nop.op, 0x90);
        assert_eq!(nop.rep, 0);
        assert!(!nop.lock);
        assert!(!nop.p66);
        assert_eq!(nop.rex, 0);
        assert!(!nop.rex_present);
        assert_eq!(nop.opsize, 4);
        assert!(!nop.addr32);
        assert_eq!(nop.seg, NO_SEG);
        assert!(!nop.has_modrm);
        assert_eq!(nop.base, NO_REG);
        assert_eq!(nop.index, NO_REG);
    }

    #[test]
    fn operand_size_prefix_selects_16_bits() {
        let i = decoded(&[0x66, 0x90]);
        assert!(i.p66);
        assert_eq!(i.opsize, 2);
        assert_eq!(i.len, 2);
    }

    #[test]
    fn address_size_prefix_is_recorded() {
        let i = decoded(&[0x67, 0x90]);
        assert!(i.addr32);
        assert_eq!(i.opsize, 4);
    }

    #[test]
    fn lock_prefix_is_recorded() {
        let i = decoded(&[0xf0, 0x90]);
        assert!(i.lock);
        assert_eq!(i.rep, 0);
    }

    #[test]
    fn rep_prefixes_select_their_code() {
        assert_eq!(decoded(&[0xf3, 0x90]).rep, 1);
        assert_eq!(decoded(&[0xf2, 0x90]).rep, 2);
    }

    #[test]
    fn last_repeat_prefix_wins() {
        assert_eq!(decoded(&[0xf3, 0xf2, 0x90]).rep, 2);
        assert_eq!(decoded(&[0xf2, 0xf3, 0x90]).rep, 1);
    }

    #[test]
    fn each_segment_override_maps_to_its_register_index() {
        for (prefix, expected) in SEG_NAMES_AND_PREFIXES {
            let i = decoded(&[prefix, 0x90]);
            assert_eq!(i.seg as usize, expected, "prefix {prefix:#x}");
        }
    }

    #[test]
    fn last_segment_override_wins() {
        assert_eq!(decoded(&[0x64, 0x65, 0x90]).seg as usize, SEG_GS);
        assert_eq!(decoded(&[0x65, 0x64, 0x90]).seg as usize, SEG_FS);
        assert_eq!(decoded(&[0x2e, 0x64, 0x3e, 0x90]).seg as usize, SEG_DS);
    }

    #[test]
    fn legacy_prefixes_are_order_independent() {
        let orders: [[u8; 5]; 4] = [
            [0x66, 0x67, 0xf0, 0xf3, 0x64],
            [0x64, 0xf3, 0xf0, 0x67, 0x66],
            [0xf0, 0x64, 0x66, 0xf3, 0x67],
            [0x67, 0x66, 0x64, 0xf0, 0xf3],
        ];
        for order in orders {
            let mut bytes = order.to_vec();
            bytes.push(0x90);
            let i = decoded(&bytes);
            assert!(i.p66 && i.addr32 && i.lock, "{order:02x?}");
            assert_eq!(i.rep, 1, "{order:02x?}");
            assert_eq!(i.seg as usize, SEG_FS, "{order:02x?}");
            assert_eq!(i.len, 6, "{order:02x?}");
        }
    }

    #[test]
    fn repeated_operand_size_prefixes_still_mean_16_bits() {
        let i = decoded(&[0x66, 0x66, 0x66, 0x90]);
        assert_eq!(i.opsize, 2);
        assert_eq!(i.len, 4);
    }

    #[test]
    fn prefix_before_two_byte_opcode_keeps_the_map() {
        let i = decoded(&[0x66, 0x0f, 0xef, 0xc1]);
        assert_eq!(i.op, 0x1ef);
        assert!(i.p66);
        assert_eq!(i.len, 4);
    }

    #[test]
    fn rex_must_immediately_precede_the_opcode() {
        // 66 48 89 c0: operand-size prefix first, so REX.W applies and wins over 66.
        let i = decoded(&[0x66, 0x48, 0x89, 0xc0]);
        assert_eq!(i.rex, 0x48);
        assert!(i.rex_present);
        assert_eq!(i.opsize, 8);
        assert!(
            i.p66,
            "66 stays visible to push/pop and SSE even when REX.W overrides it"
        );
        assert_eq!(i.op, 0x89);
        assert_eq!(i.len, 4);
    }

    #[test]
    fn rex_followed_by_operand_size_prefix_is_dropped() {
        // 48 66 89 c0 is `mov ax, ax`: the REX is not adjacent to the opcode, so it is ignored.
        let i = decoded(&[0x48, 0x66, 0x89, 0xc0]);
        assert_eq!(i.op, 0x89);
        assert_eq!(i.rex, 0);
        assert!(!i.rex_present);
        assert_eq!(i.opsize, 2);
        assert!(i.p66);
        assert_eq!(i.len, 4);
        assert!(i.has_modrm);
        assert_eq!((i.reg, i.rm), (0, 0));
    }

    #[test]
    fn rex_followed_by_each_legacy_prefix_is_dropped_and_prefix_applies() {
        let cases: [(u8, PrefixCheck); 11] = [
            (0x66, |i| i.p66),
            (0x67, |i| i.addr32),
            (0xf0, |i| i.lock),
            (0xf2, |i| i.rep == 2),
            (0xf3, |i| i.rep == 1),
            (0x26, |i| i.seg as usize == SEG_ES),
            (0x2e, |i| i.seg as usize == SEG_CS),
            (0x36, |i| i.seg as usize == SEG_SS),
            (0x3e, |i| i.seg as usize == SEG_DS),
            (0x64, |i| i.seg as usize == SEG_FS),
            (0x65, |i| i.seg as usize == SEG_GS),
        ];
        for (prefix, applied) in cases {
            // REX.WRXB then the legacy prefix then `add eax, ebx` (01 d8).
            let i = decoded(&[0x4f, prefix, 0x01, 0xd8]);
            assert!(applied(&i), "prefix {prefix:#x} was not applied");
            assert_eq!(i.rex, 0, "prefix {prefix:#x}");
            assert!(!i.rex_present, "prefix {prefix:#x}");
            assert_eq!(
                (i.reg, i.rm),
                (3, 0),
                "prefix {prefix:#x}: REX.R/B must not leak"
            );
            assert_eq!(i.len, 4, "prefix {prefix:#x}");
        }
    }

    #[test]
    fn dropped_rex_does_not_widen_the_operand_or_extend_registers() {
        // 4c 66 01 d8: REX.WR dropped, so this is `add ax, bx`, not `add r8, rbx`.
        let i = decoded(&[0x4c, 0x66, 0x01, 0xd8]);
        assert_eq!(i.opsize, 2);
        assert_eq!(i.reg, 3);
        assert_eq!(i.rm, 0);
    }

    #[test]
    fn rex_before_legacy_prefix_before_a_new_rex_uses_the_new_rex() {
        // 48 66 4c 01 c8: only the REX adjacent to the opcode (4c: W and R) counts.
        let i = decoded(&[0x48, 0x66, 0x4c, 0x01, 0xc8]);
        assert_eq!(i.rex, 0x4c);
        assert_eq!(i.opsize, 8);
        assert_eq!(i.reg, 9);
        assert_eq!(i.rm, 0);
        assert_eq!(i.len, 5);
    }

    #[test]
    fn consecutive_rex_bytes_use_the_last_one() {
        let i = decoded(&[0x48, 0x41, 0x01, 0xd8]);
        assert_eq!(i.rex, 0x41);
        assert_eq!(
            i.opsize, 4,
            "the earlier REX.W is not adjacent to the opcode"
        );
        assert_eq!(i.rm, 8);
    }

    #[test]
    fn dropped_rex_still_counts_toward_the_15_byte_limit() {
        // 14 prefix bytes (REX then 13 x 66) plus the opcode make exactly 15 bytes.
        let mut bytes = vec![0x48];
        bytes.extend(std::iter::repeat_n(0x66, 13));
        bytes.push(0x90);
        assert_eq!(bytes.len(), 15);
        let i = decoded(&bytes);
        assert_eq!(i.len, 15);
        assert!(!i.rex_present);
        bytes.insert(1, 0x66);
        assert_eq!(decode(&bytes).err(), Some(DecodeErr::Invalid));
    }

    #[test]
    fn mandatory_prefix_then_rex_keeps_the_rex() {
        // F3 48 0F B8 C1 (popcnt rax, rcx style): mandatory prefix before REX is the valid order.
        let i = decoded(&[0xf3, 0x48, 0x0f, 0xb8, 0xc1]);
        assert_eq!(i.rep, 1);
        assert_eq!(i.rex, 0x48);
        assert_eq!(i.op, 0x1b8);
    }

    #[test]
    fn rex_before_mandatory_prefix_drops_the_rex() {
        let i = decoded(&[0x48, 0xf3, 0x0f, 0xb8, 0xc1]);
        assert_eq!(i.rep, 1);
        assert_eq!(i.rex, 0);
        assert_eq!(i.opsize, 4);
    }
}

mod rex {
    use super::*;

    #[test]
    fn bare_rex_is_present_without_bits() {
        let i = decoded(&[0x40, 0x88, 0xe6]);
        assert!(i.rex_present);
        assert_eq!(i.rex, 0x40);
        assert_eq!(i.opsize, 4);
        assert_eq!((i.reg, i.rm), (4, 6));
    }

    #[test]
    fn every_rex_value_is_a_prefix_never_inc_or_dec() {
        for rex in 0x40..=0x4fu8 {
            let i = decoded(&[rex, 0x90]);
            assert_eq!(i.rex, rex);
            assert!(i.rex_present);
            assert_eq!(i.op, 0x90);
            assert_eq!(i.len, 2);
            assert_eq!(i.opsize, if rex & 8 != 0 { 8 } else { 4 });
        }
    }

    #[test]
    fn rex_w_wins_over_operand_size_prefix() {
        let i = decoded(&[0x66, 0x48, 0x90]);
        assert_eq!(i.opsize, 8);
        assert!(i.p66);
    }

    #[test]
    fn rex_r_extends_modrm_reg() {
        // mov rax, r9
        let bytes = &[0x4c, 0x89, 0xc8];
        let i = decoded(bytes);
        assert_eq!(i.reg, 9);
        assert_eq!(i.rm, 0);
        assert_eq!(i.rex & REX_R, REX_R);
    }

    #[test]
    fn rex_b_extends_modrm_rm_in_register_form() {
        // mov r9, rax
        let bytes = &[0x49, 0x89, 0xc1];
        let i = decoded(bytes);
        assert_eq!(i.rm, 9);
        assert_eq!(i.reg, 0);
    }

    #[test]
    fn rex_b_extends_the_memory_base() {
        // mov rax, [r9 + 8]
        let bytes = &[0x49, 0x8b, 0x41, 0x08];
        let i = decoded(bytes);
        assert_eq!(i.base, 9);
        assert_eq!(i.index, NO_REG);
        assert_eq!(i.disp, 8);
    }

    #[test]
    fn rex_x_extends_the_sib_index() {
        // mov rax, [rbx + r10*4]
        let bytes = &[0x4a, 0x8b, 0x04, 0x93];
        let i = decoded(bytes);
        assert_eq!(i.base, 3);
        assert_eq!(i.index, 10);
        assert_eq!(i.scale, 2);
    }

    #[test]
    fn rex_b_extends_the_sib_base() {
        // mov rax, [r11 + rcx*2]
        let bytes = &[0x49, 0x8b, 0x04, 0x4b];
        let i = decoded(bytes);
        assert_eq!(i.base, 11);
        assert_eq!(i.index, 1);
        assert_eq!(i.scale, 1);
    }

    #[test]
    fn all_rex_bits_together() {
        // mov r15, [r14 + r13*8 + 0x20]
        let bytes = &[0x4f, 0x8b, 0x7c, 0xee, 0x20];
        let i = decoded(bytes);
        assert_eq!(i.reg, 15);
        assert_eq!(i.base, 14);
        assert_eq!(i.index, 13);
        assert_eq!(i.scale, 3);
        assert_eq!(i.disp, 0x20);
        assert_eq!(i.opsize, 8);
    }

    #[test]
    fn rex_before_two_byte_opcode() {
        let i = decoded(&[0x4c, 0x0f, 0xaf, 0xc1]);
        assert_eq!(i.op, 0x1af);
        assert_eq!(i.reg, 8);
        assert_eq!(i.rm, 1);
        assert_eq!(i.opsize, 8);
    }

    #[test]
    fn rex_before_three_byte_opcode() {
        let i = decoded(&[0x48, 0x0f, 0x38, 0xf0, 0xc1]);
        assert_eq!(i.op, 0x2f0);
        assert_eq!(i.rex, 0x48);
    }

    #[test]
    fn low_byte_register_numbers_need_rex_to_mean_spl_through_dil() {
        // Both `mov dh, al` forms are 88 /r with reg = 6; only the bare REX changes the meaning,
        // which the executor reads from rex_present.
        let without = decoded(&[0x88, 0xc6]);
        let with = decoded(&[0x40, 0x88, 0xc6]);
        assert!(!without.rex_present);
        assert!(with.rex_present);
        assert_eq!(without.rm, with.rm);
    }
}

mod opcode_maps {
    use super::*;

    #[test]
    fn one_byte_map_has_map_zero() {
        assert_eq!(decoded(&[0x90]).op >> 8, 0);
        assert_eq!(decoded(&[0xff, 0xc0]).op, 0xff);
    }

    #[test]
    fn two_byte_map_sets_map_one() {
        let i = decoded(&[0x0f, 0xa2]);
        assert_eq!(i.op, 0x1a2);
        assert_eq!(i.len, 2);
        assert!(!i.has_modrm);
    }

    #[test]
    fn three_byte_0f38_map_sets_map_two() {
        let i = decoded(&[0x66, 0x0f, 0x38, 0x00, 0xc1]);
        assert_eq!(i.op, 0x200);
        assert!(i.has_modrm);
        assert_eq!(i.len, 5);
    }

    #[test]
    fn three_byte_0f3a_map_sets_map_three_and_takes_an_imm8() {
        let i = decoded(&[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x05]);
        assert_eq!(i.op, 0x30f);
        assert_eq!(i.imm, 5);
        assert_eq!(i.len, 6);
    }

    #[test]
    fn every_0f3a_opcode_has_modrm_and_imm8() {
        for opcode in 0..=0xffu8 {
            let i = decoded(&[0x66, 0x0f, 0x3a, opcode, 0xc1, 0x7f]);
            assert_eq!(i.len, 6, "0f 3a {opcode:#x}");
            assert_eq!(i.imm, 0x7f, "0f 3a {opcode:#x}");
        }
    }

    #[test]
    fn every_0f38_opcode_has_modrm_and_no_immediate() {
        for opcode in 0..=0xffu8 {
            let i = decoded(&[0x66, 0x0f, 0x38, opcode, 0xc1]);
            assert_eq!(i.len, 5, "0f 38 {opcode:#x}");
            assert_eq!(i.imm, 0, "0f 38 {opcode:#x}");
        }
    }

    #[test]
    fn two_byte_opcodes_without_modrm() {
        let no_modrm: Vec<u8> = (0x05..=0x09)
            .chain([0x0b, 0x0e])
            .chain(0x30..=0x37)
            .chain([0x77])
            .chain(0xa0..=0xa2)
            .chain(0xa8..=0xaa)
            .chain(0xc8..=0xcf)
            .collect();
        for opcode in no_modrm {
            let i = decoded(&[0x0f, opcode, 0x90, 0x90, 0x90, 0x90]);
            assert_eq!(i.len, 2, "0f {opcode:#x}");
            assert!(!i.has_modrm, "0f {opcode:#x}");
        }
    }

    #[test]
    fn two_byte_opcodes_with_modrm() {
        for opcode in [
            0x00u8, 0x01, 0x10, 0x20, 0x28, 0x40, 0x6f, 0x7f, 0xa3, 0xaf, 0xb6, 0xc7,
        ] {
            let i = decoded(&[0x0f, opcode, 0xc1, 0x90, 0x90, 0x90]);
            assert!(i.has_modrm, "0f {opcode:#x}");
            assert_eq!(i.modrm, 0xc1);
        }
    }

    #[test]
    fn jcc_rel32_takes_four_bytes() {
        for cc in 0x80..=0x8fu8 {
            let i = decoded(&[0x0f, cc, 0xfe, 0xff, 0xff, 0xff]);
            assert_eq!(i.len, 6);
            assert_eq!(i.imm, -2);
        }
    }

    #[test]
    fn bswap_has_no_modrm_and_keeps_rex_b() {
        // bswap r9
        let bytes = &[0x49, 0x0f, 0xc9];
        let i = decoded(bytes);
        assert_eq!(i.op, 0x1c9);
        assert_eq!(i.len as usize, bytes.len());
        assert!(!i.has_modrm);
        assert_eq!(i.rex & REX_B, REX_B);
    }

    #[test]
    fn escape_alone_is_truncated() {
        assert_eq!(decode(&[0x0f]).err(), Some(DecodeErr::Truncated));
        assert_eq!(decode(&[0x0f, 0x38]).err(), Some(DecodeErr::Truncated));
        assert_eq!(decode(&[0x0f, 0x3a]).err(), Some(DecodeErr::Truncated));
        assert_eq!(
            decode(&[0x0f, 0x38, 0x00]).err(),
            Some(DecodeErr::Truncated)
        );
        assert_eq!(
            decode(&[0x0f, 0x3a, 0x0f, 0xc1]).err(),
            Some(DecodeErr::Truncated)
        );
    }
}

mod modrm_and_sib {
    use super::*;

    #[test]
    fn register_direct_form() {
        // add rbx, rcx
        let bytes = &[0x48, 0x01, 0xcb];
        let i = decoded(bytes);
        assert!(i.has_modrm);
        assert_eq!(i.md, 3);
        assert_eq!(i.modrm >> 6, 3);
        assert_eq!(i.base, NO_REG);
        assert_eq!(i.index, NO_REG);
        assert!(!i.rip_rel);
        assert_eq!(i.disp, 0);
    }

    #[test]
    fn base_only() {
        // mov rax, [rbx]
        let bytes = &[0x48, 0x8b, 0x03];
        let i = decoded(bytes);
        assert_eq!((i.md, i.base, i.index, i.disp), (0, 3, NO_REG, 0));
        assert_eq!(i.len, 3);
    }

    #[test]
    fn base_plus_disp8_sign_extends() {
        // mov rax, [rbx - 0x80]
        let bytes = &[0x48, 0x8b, 0x43, 0x80];
        let i = decoded(bytes);
        assert_eq!((i.md, i.base), (1, 3));
        assert_eq!(i.disp, -0x80);
        // mov rax, [rbx + 0x7f]
        let positive = decoded(&[0x48, 0x8b, 0x43, 0x7f]);
        assert_eq!(positive.disp, 0x7f);
    }

    #[test]
    fn base_plus_disp32_sign_extends() {
        // mov rax, [rbx - 0x12345]
        let bytes = &[0x48, 0x8b, 0x83, 0xbb, 0xdc, 0xfe, 0xff];
        let i = decoded(bytes);
        assert_eq!((i.md, i.base), (2, 3));
        assert_eq!(i.disp, -0x12345);
        // mov rax, [rbx + 0x7fffffff]
        let high = decoded(&[0x48, 0x8b, 0x83, 0xff, 0xff, 0xff, 0x7f]);
        assert_eq!(high.disp, 0x7fff_ffff);
    }

    #[test]
    fn rsp_base_needs_a_sib_byte() {
        // mov rax, [rsp]
        let bytes = &[0x48, 0x8b, 0x04, 0x24];
        let i = decoded(bytes);
        assert_eq!(i.base, 4);
        assert_eq!(i.index, NO_REG);
        assert_eq!(i.len, 4);
        // mov rax, [rsp + 0x10]
        let with_disp = decoded(&[0x48, 0x8b, 0x44, 0x24, 0x10]);
        assert_eq!(
            (with_disp.base, with_disp.disp, with_disp.len),
            (4, 0x10, 5)
        );
    }

    #[test]
    fn r12_base_needs_a_sib_byte_like_rsp() {
        // mov rax, [r12]
        let bytes = &[0x49, 0x8b, 0x04, 0x24];
        let i = decoded(bytes);
        assert_eq!(i.base, 12);
        assert_eq!(i.index, NO_REG);
        assert_eq!(i.len, 4);
    }

    #[test]
    fn rbp_base_is_encoded_with_a_zero_disp8() {
        // mov rax, [rbp]
        let bytes = &[0x48, 0x8b, 0x45, 0x00];
        let i = decoded(bytes);
        assert_eq!((i.md, i.base, i.disp), (1, 5, 0));
        assert!(!i.rip_rel);
    }

    #[test]
    fn r13_base_is_encoded_with_a_zero_disp8() {
        // mov rax, [r13]
        let bytes = &[0x49, 0x8b, 0x45, 0x00];
        let i = decoded(bytes);
        assert_eq!((i.md, i.base, i.disp), (1, 13, 0));
        assert!(!i.rip_rel);
    }

    #[test]
    fn rip_relative_has_no_base_and_a_signed_disp32() {
        // mov rax, [rip + 0x100]
        let bytes = &[0x48, 0x8b, 0x05, 0x00, 0x01, 0x00, 0x00];
        let i = decoded(bytes);
        assert!(i.rip_rel);
        assert_eq!(i.base, NO_REG);
        assert_eq!(i.index, NO_REG);
        assert_eq!(i.disp, 0x100);
        assert_eq!(i.md, 0);
        // mov rax, [rip - 0x20]
        let backwards = decoded(&[0x48, 0x8b, 0x05, 0xe0, 0xff, 0xff, 0xff]);
        assert_eq!(backwards.disp, -0x20);
        assert!(backwards.rip_rel);
    }

    #[test]
    fn rip_relative_ignores_rex_b() {
        // mod=00 rm=101 is RIP-relative even when REX.B would make the register r13.
        let i = decoded(&[0x49, 0x8b, 0x05, 0x10, 0, 0, 0]);
        assert!(i.rip_rel);
        assert_eq!(i.base, NO_REG);
        assert_eq!(i.len, 7);
    }

    #[test]
    fn rip_relative_with_a_trailing_immediate_counts_both() {
        // add dword ptr [rip + 0x10], 5
        let bytes = &[0x83, 0x05, 0x10, 0x00, 0x00, 0x00, 0x05];
        let i = decoded(bytes);
        assert!(i.rip_rel);
        assert_eq!(i.imm, 5);
        assert_eq!(i.len as usize, bytes.len());
    }

    #[test]
    fn absolute_disp32_through_sib() {
        // mov eax, [0x12345678]
        let bytes = &[0x8b, 0x04, 0x25, 0x78, 0x56, 0x34, 0x12];
        let i = decoded(bytes);
        assert!(!i.rip_rel);
        assert_eq!(i.base, NO_REG);
        assert_eq!(i.index, NO_REG);
        assert_eq!(i.disp, 0x1234_5678);
    }

    #[test]
    fn scaled_index_without_base_keeps_the_index() {
        // mov eax, [rcx*8 + 0x1000]
        let bytes = &[0x8b, 0x04, 0xcd, 0x00, 0x10, 0x00, 0x00];
        let i = decoded(bytes);
        assert_eq!(i.base, NO_REG);
        assert_eq!(i.index, 1);
        assert_eq!(i.scale, 3);
        assert_eq!(i.disp, 0x1000);
    }

    #[test]
    fn sib_base_5_with_mod_0_has_no_base_even_for_r13() {
        // 4b 8b 04 ed disp32: REX.XB, SIB base field 5 with mod 0 means "no base".
        let i = decoded(&[0x4b, 0x8b, 0x04, 0xed, 0x10, 0, 0, 0]);
        assert_eq!(i.base, NO_REG);
        assert_eq!(i.index, 13);
        assert_eq!(i.scale, 3);
        assert_eq!(i.disp, 0x10);
    }

    #[test]
    fn sib_base_5_with_mod_1_is_rbp() {
        // mov rax, [rbp + rcx*2 + 8]
        let bytes = &[0x48, 0x8b, 0x44, 0x4d, 0x08];
        let i = decoded(bytes);
        assert_eq!((i.md, i.base, i.index, i.scale, i.disp), (1, 5, 1, 1, 8));
    }

    #[test]
    fn sib_index_4_means_none_but_r12_index_is_real() {
        // .byte 0x48, 0x8b, 0x04, 0x20
        let none = decoded(&[0x48, 0x8b, 0x04, 0x20]);
        assert_eq!(none.index, NO_REG);
        // mov rax, [rax + r12*2]
        let r12 = decoded(&[0x4a, 0x8b, 0x04, 0x60]);
        assert_eq!(r12.index, 12);
        assert_eq!(r12.scale, 1);
    }

    #[test]
    fn all_four_scales() {
        for log2 in 0..4u8 {
            let sib = (log2 << 6) | (1 << 3) | 3;
            let i = decoded(&[0x48, 0x8b, 0x04, sib]);
            assert_eq!(i.scale, log2);
            assert_eq!((i.base, i.index), (3, 1));
        }
    }

    #[test]
    fn modrm_reg_field_is_the_group_extension() {
        // ff /4 (jmp r/m64) and ff /6 (push r/m64): reg & 7 selects the operation.
        // jmp rax
        let jmp = decoded(&[0xff, 0xe0]);
        assert_eq!(jmp.reg & 7, 4);
        // push qword ptr [rax]
        let push = decoded(&[0xff, 0x30]);
        assert_eq!(push.reg & 7, 6);
    }

    #[test]
    fn modrm_fields_are_all_exposed() {
        let i = decoded(&[0x8b, 0b10_011_100, 0b11_010_001, 1, 0, 0, 0]);
        assert_eq!(i.modrm, 0b10_011_100);
        assert_eq!(i.md, 2);
        assert_eq!(i.reg, 3);
        assert_eq!(i.base, 1);
        assert_eq!(i.index, 2);
        assert_eq!(i.scale, 3);
        assert_eq!(i.disp, 1);
    }

    #[test]
    fn address_size_prefix_does_not_change_the_encoding_length() {
        assert_eq!(decoded(&[0x67, 0x8b, 0x04, 0x24]).len, 4);
        assert_eq!(decoded(&[0x67, 0x8b, 0x05, 1, 0, 0, 0]).len, 7);
    }
}

mod immediates {
    use super::*;

    #[test]
    fn imm8_is_sign_extended() {
        // add rax, -1
        let i = decoded(&[0x48, 0x83, 0xc0, 0xff]);
        assert_eq!(i.imm, -1);
        let max = decoded(&[0x83, 0xc0, 0x7f]);
        assert_eq!(max.imm, 0x7f);
        let min = decoded(&[0x83, 0xc0, 0x80]);
        assert_eq!(min.imm, -128);
    }

    #[test]
    fn iz_is_32_bits_without_prefix_and_sign_extended() {
        // add eax, 0x12345678
        let i = decoded(&[0x05, 0x78, 0x56, 0x34, 0x12]);
        assert_eq!(i.imm, 0x1234_5678);
        assert_eq!(i.len, 5);
        let negative = decoded(&[0x05, 0x00, 0x00, 0x00, 0x80]);
        assert_eq!(negative.imm, i64::from(i32::MIN));
    }

    #[test]
    fn iz_is_16_bits_with_operand_size_prefix() {
        // add ax, 0x1234
        let bytes = &[0x66, 0x05, 0x34, 0x12];
        let i = decoded(bytes);
        assert_eq!(i.imm, 0x1234);
        assert_eq!(i.len, 4);
        let negative = decoded(&[0x66, 0x05, 0x00, 0x80]);
        assert_eq!(negative.imm, -0x8000);
    }

    #[test]
    fn iz_stays_32_bits_with_rex_w_and_sign_extends_to_64() {
        // add rax, 0x12345678
        let bytes = &[0x48, 0x05, 0x78, 0x56, 0x34, 0x12];
        let i = decoded(bytes);
        assert_eq!(i.imm, 0x1234_5678);
        assert_eq!(i.len, 6);
        // mov rax, -2
        let negative = decoded(&[0x48, 0xc7, 0xc0, 0xfe, 0xff, 0xff, 0xff]);
        assert_eq!(negative.imm, -2);
    }

    #[test]
    fn rex_w_with_operand_size_prefix_uses_a_32_bit_immediate() {
        // opsize is 8 under REX.W even with 66, so Iz is 4 bytes.
        let i = decoded(&[0x66, 0x48, 0x05, 1, 0, 0, 0]);
        assert_eq!(i.len, 7);
        assert_eq!(i.imm, 1);
    }

    #[test]
    fn mov_r64_imm64() {
        // movabs rax, 0x1122334455667788
        let bytes = &[0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11];
        let i = decoded(bytes);
        assert_eq!(i.len, 10);
        assert_eq!(i.imm, 0x1122_3344_5566_7788);
        assert_eq!(i.op, 0xb8);
    }

    #[test]
    fn mov_r32_imm32_and_r16_imm16() {
        // mov ecx, 0xdeadbeef
        let dword = decoded(&[0xb9, 0xef, 0xbe, 0xad, 0xde]);
        assert_eq!(dword.len, 5);
        assert_eq!(dword.imm, i64::from(0xdead_beefu32 as i32));
        // mov cx, 0x1234
        let word = decoded(&[0x66, 0xb9, 0x34, 0x12]);
        assert_eq!(word.len, 4);
        assert_eq!(word.imm, 0x1234);
    }

    #[test]
    fn mov_r8_imm8_for_all_byte_registers() {
        for opcode in 0xb0..=0xb7u8 {
            let i = decoded(&[opcode, 0x80]);
            assert_eq!(i.len, 2);
            assert_eq!(i.imm, -128);
        }
    }

    #[test]
    fn mov_r_imm_with_rex_b_reaches_r8_to_r15() {
        // mov r8d, 5
        let bytes = &[0x41, 0xb8, 0x05, 0x00, 0x00, 0x00];
        let i = decoded(bytes);
        assert_eq!(i.rex & REX_B, REX_B);
        assert_eq!(i.op, 0xb8);
        assert_eq!(i.imm, 5);
    }

    #[test]
    fn moffs_is_eight_bytes_in_64_bit_mode() {
        // movabs al, [0x1122334455667788]
        let bytes = &[0xa0, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11];
        let i = decoded(bytes);
        assert_eq!(i.op, 0xa0);
        assert_eq!(i.len, 9);
        assert_eq!(i.imm, 0x1122_3344_5566_7788);
        // movabs [0x1122334455667788], rax
        let store = decoded(&[0x48, 0xa3, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
        assert_eq!(store.op, 0xa3);
        assert_eq!(store.len, 10);
    }

    #[test]
    fn moffs_shrinks_to_four_bytes_with_the_address_size_prefix() {
        let i = decoded(&[0x67, 0xa1, 0x78, 0x56, 0x34, 0x12]);
        assert!(i.addr32);
        assert_eq!(i.len, 6);
        assert_eq!(i.imm, 0x1234_5678);
        let with_rex = decoded(&[0x48, 0x67, 0xa1, 0x78, 0x56, 0x34, 0x12]);
        assert_eq!(with_rex.len, 7, "the dropped REX still occupies a byte");
        assert_eq!(with_rex.imm, 0x1234_5678);
        assert!(!with_rex.rex_present);
    }

    #[test]
    fn ret_imm16_is_zero_extended() {
        // ret 0xffff
        let i = decoded(&[0xc2, 0xff, 0xff]);
        assert_eq!(i.op, 0xc2);
        assert_eq!(i.imm, 0xffff);
        assert_eq!(i.len, 3);
        let far = decoded(&[0xca, 0x00, 0x80]);
        assert_eq!(far.imm, 0x8000);
    }

    #[test]
    fn enter_has_a_zero_extended_size_and_a_nesting_level() {
        let i = decoded(&[0xc8, 0x00, 0x80, 0x1f]);
        assert_eq!(i.len, 4);
        assert_eq!(i.imm, 0x8000);
        assert_eq!(i.imm2, 0x1f);
        let max_level = decoded(&[0xc8, 0x10, 0x00, 0xff]);
        assert_eq!(max_level.imm2, 0xff);
    }

    #[test]
    fn relative_branches_are_signed() {
        let backward = decoded(&[0xeb, 0xfe]);
        assert_eq!(backward.imm, -2);
        let call = decoded(&[0xe8, 0xfb, 0xff, 0xff, 0xff]);
        assert_eq!((call.op, call.imm, call.len), (0xe8, -5, 5));
        let jmp = decoded(&[0xe9, 0x00, 0x01, 0x00, 0x00]);
        assert_eq!((jmp.imm, jmp.len), (0x100, 5));
    }

    #[test]
    fn jcc_short_and_loop_forms_take_rel8() {
        for opcode in (0x70..=0x7fu8).chain(0xe0..=0xe3) {
            let i = decoded(&[opcode, 0x80]);
            assert_eq!(i.len, 2, "{opcode:#x}");
            assert_eq!(i.imm, -128, "{opcode:#x}");
        }
    }

    #[test]
    fn rel32_ignores_the_operand_size_prefix() {
        let i = decoded(&[0x66, 0xe8, 1, 0, 0, 0]);
        assert_eq!(i.len, 6);
        assert_eq!(i.imm, 1);
    }

    #[test]
    fn push_imm_forms() {
        // push 0x12345678
        let imm32 = decoded(&[0x68, 0x78, 0x56, 0x34, 0x12]);
        assert_eq!((imm32.op, imm32.len, imm32.imm), (0x68, 5, 0x1234_5678));
        // push -2
        let imm8 = decoded(&[0x6a, 0xfe]);
        assert_eq!((imm8.op, imm8.len, imm8.imm), (0x6a, 2, -2));
        let imm16 = decoded(&[0x66, 0x68, 0x34, 0x12]);
        assert_eq!((imm16.len, imm16.imm), (4, 0x1234));
    }

    #[test]
    fn imul_imm_forms() {
        // imul rax, rbx, 0x1000
        let imm32 = decoded(&[0x48, 0x69, 0xc3, 0x00, 0x10, 0x00, 0x00]);
        assert_eq!(imm32.op, 0x69);
        assert_eq!(imm32.imm, 0x1000);
        // imul rax, rbx, 5
        let imm8 = decoded(&[0x48, 0x6b, 0xc3, 0x05]);
        assert_eq!(imm8.op, 0x6b);
        assert_eq!(imm8.imm, 5);
    }

    #[test]
    fn group1_immediate_forms() {
        // add byte ptr [rax], 1
        let eighty = decoded(&[0x80, 0x00, 0x01]);
        assert_eq!((eighty.op, eighty.imm), (0x80, 1));
        // add dword ptr [rax], 0x1000
        let eighty_one = decoded(&[0x81, 0x00, 0x00, 0x10, 0x00, 0x00]);
        assert_eq!((eighty_one.op, eighty_one.imm), (0x81, 0x1000));
        // add dword ptr [rax], -1
        let eighty_three = decoded(&[0x83, 0x00, 0xff]);
        assert_eq!((eighty_three.op, eighty_three.imm), (0x83, -1));
    }

    #[test]
    fn shift_group_immediates() {
        // shl rax, 5
        let by_imm = decoded(&[0x48, 0xc1, 0xe0, 0x05]);
        assert_eq!((by_imm.op, by_imm.imm, by_imm.len), (0xc1, 5, 4));
        // shl al, 3
        let byte = decoded(&[0xc0, 0xe0, 0x03]);
        assert_eq!((byte.op, byte.imm), (0xc0, 3));
        // shl rax, 1
        let by_one = decoded(&[0x48, 0xd1, 0xe0]);
        assert_eq!(by_one.op, 0xd1);
        assert_eq!(by_one.len, 3);
        // shl rax, cl
        let by_cl = decoded(&[0x48, 0xd3, 0xe0]);
        assert_eq!(by_cl.op, 0xd3);
        assert_eq!(by_cl.len, 3);
    }

    #[test]
    fn mov_rm_imm_forms() {
        // mov byte ptr [rax], 7
        let byte = decoded(&[0xc6, 0x00, 0x07]);
        assert_eq!((byte.op, byte.imm, byte.len), (0xc6, 7, 3));
        // mov dword ptr [rax], 0x11223344
        let dword = decoded(&[0xc7, 0x00, 0x44, 0x33, 0x22, 0x11]);
        assert_eq!((dword.op, dword.imm, dword.len), (0xc7, 0x1122_3344, 6));
        // mov word ptr [rax], 0x1122
        let word = decoded(&[0x66, 0xc7, 0x00, 0x22, 0x11]);
        assert_eq!((word.imm, word.len), (0x1122, 5));
    }

    #[test]
    fn test_group3_takes_an_immediate_only_for_slash_0_and_1() {
        // test byte ptr [rax], 0x55
        let test_byte = decoded(&[0xf6, 0x00, 0x55]);
        assert_eq!(
            (test_byte.op, test_byte.imm, test_byte.len),
            (0xf6, 0x55, 3)
        );
        // test dword ptr [rax], 0x55667788
        let test_dword = decoded(&[0xf7, 0x00, 0x88, 0x77, 0x66, 0x55]);
        assert_eq!(
            (test_dword.op, test_dword.imm, test_dword.len),
            (0xf7, 0x5566_7788, 6)
        );
        let test_word = decoded(&[0x66, 0xf7, 0x00, 0x34, 0x12]);
        assert_eq!((test_word.imm, test_word.len), (0x1234, 5));
        // /1 is an undocumented alias of /0 and also carries the immediate.
        let alias = decoded(&[0xf6, 0x08, 0x55]);
        assert_eq!((alias.imm, alias.len), (0x55, 3));
        for slash in 2..8u8 {
            let modrm = 0xc0 | (slash << 3);
            let byte = decoded(&[0xf6, modrm]);
            assert_eq!(byte.len, 2, "f6 /{slash}");
            assert_eq!(byte.imm, 0, "f6 /{slash}");
            let dword = decoded(&[0xf7, modrm]);
            assert_eq!(dword.len, 2, "f7 /{slash}");
        }
    }

    #[test]
    fn accumulator_alu_immediates() {
        for row in 0..8u8 {
            let base = row << 3;
            let byte = decoded(&[base + 4, 0x80]);
            assert_eq!((byte.len, byte.imm), (2, -128), "{base:#x}+4");
            assert!(!byte.has_modrm);
            let dword = decoded(&[base + 5, 1, 0, 0, 0]);
            assert_eq!((dword.len, dword.imm), (5, 1), "{base:#x}+5");
            assert!(!dword.has_modrm);
        }
    }

    #[test]
    fn test_al_and_eax_immediates() {
        assert_eq!(decoded(&[0xa8, 0x01]).len, 2);
        assert_eq!(decoded(&[0xa9, 1, 0, 0, 0]).len, 5);
        assert_eq!(decoded(&[0x66, 0xa9, 1, 0]).len, 4);
    }

    #[test]
    fn io_immediate_port_forms() {
        for opcode in [0xe4u8, 0xe5, 0xe6, 0xe7] {
            let i = decoded(&[opcode, 0x80]);
            assert_eq!(i.len, 2);
            assert_eq!(i.imm, -128);
        }
        assert_eq!(decoded(&[0xec]).len, 1);
    }

    #[test]
    fn int_imm8() {
        // int 0x80
        let i = decoded(&[0xcd, 0x80]);
        assert_eq!((i.op, i.len), (0xcd, 2));
        // int3
        assert_eq!(decoded(&[0xcc]).len, 1);
    }

    #[test]
    fn two_byte_map_immediates() {
        // pshufd xmm0, xmm1, 0x1b
        let pshufd = decoded(&[0x66, 0x0f, 0x70, 0xc1, 0x1b]);
        assert_eq!((pshufd.op, pshufd.imm), (0x170, 0x1b));
        // psllw xmm0, 3
        let shift = decoded(&[0x66, 0x0f, 0x71, 0xf0, 0x03]);
        assert_eq!((shift.op, shift.imm), (0x171, 3));
        // shld rax, rbx, 4
        let shld = decoded(&[0x48, 0x0f, 0xa4, 0xd8, 0x04]);
        assert_eq!((shld.op, shld.imm), (0x1a4, 4));
        // shld rax, rbx, cl
        let shld_cl = decoded(&[0x48, 0x0f, 0xa5, 0xd8]);
        assert_eq!((shld_cl.op, shld_cl.len), (0x1a5, 4));
        // shrd rax, rbx, 4
        let shrd = decoded(&[0x48, 0x0f, 0xac, 0xd8, 0x04]);
        assert_eq!((shrd.op, shrd.imm), (0x1ac, 4));
        // bt rax, 3
        let bt = decoded(&[0x48, 0x0f, 0xba, 0xe0, 0x03]);
        assert_eq!((bt.op, bt.imm), (0x1ba, 3));
        // cmpps xmm0, xmm1, 2
        let cmpps = decoded(&[0x0f, 0xc2, 0xc1, 0x02]);
        assert_eq!((cmpps.op, cmpps.imm), (0x1c2, 2));
        // pinsrw xmm0, eax, 3
        let pinsrw = decoded(&[0x66, 0x0f, 0xc4, 0xc0, 0x03]);
        assert_eq!((pinsrw.op, pinsrw.imm), (0x1c4, 3));
        // pextrw eax, xmm1, 3
        let pextrw = decoded(&[0x66, 0x0f, 0xc5, 0xc1, 0x03]);
        assert_eq!((pextrw.op, pextrw.imm), (0x1c5, 3));
        // shufps xmm0, xmm1, 0xe4
        let shufps = decoded(&[0x0f, 0xc6, 0xc1, 0xe4]);
        assert_eq!(shufps.op, 0x1c6);
        assert_eq!(shufps.imm, -28, "0xe4 sign-extended");
    }

    #[test]
    fn two_byte_opcodes_next_to_immediate_ones_have_none() {
        for opcode in [0x6fu8, 0x74, 0x75, 0x76, 0xa5, 0xad, 0xbb, 0xc1, 0xc3, 0xc7] {
            let i = decoded(&[0x0f, opcode, 0xc1, 0x55]);
            assert_eq!(i.len, 3, "0f {opcode:#x}");
            assert_eq!(i.imm, 0, "0f {opcode:#x}");
        }
    }

    #[test]
    fn memory_operand_before_the_immediate() {
        // mov dword ptr [rbx + rcx*4 + 0x10], 0x11223344
        let bytes = &[0xc7, 0x44, 0x8b, 0x10, 0x44, 0x33, 0x22, 0x11];
        let i = decoded(bytes);
        assert_eq!(i.disp, 0x10);
        assert_eq!(i.imm, 0x1122_3344);
        assert_eq!(i.len as usize, bytes.len());
    }
}

mod invalid_encodings {
    use super::*;

    const INVALID_IN_64_BIT_MODE: [u8; 23] = [
        0x06, 0x07, 0x0e, 0x16, 0x17, 0x1e, 0x1f, 0x27, 0x2f, 0x37, 0x3f, 0x60, 0x61, 0x62, 0x82,
        0x9a, 0xc4, 0xc5, 0xce, 0xd4, 0xd5, 0xd6, 0xea,
    ];

    #[test]
    fn opcodes_removed_in_long_mode_are_invalid() {
        for opcode in &INVALID_IN_64_BIT_MODE {
            assert_eq!(
                decode(&[*opcode, 0xc0, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]).err(),
                Some(DecodeErr::Invalid),
                "{opcode:#x}"
            );
        }
    }

    #[test]
    fn invalid_opcodes_stay_invalid_behind_prefixes() {
        for prefix in [&[0x66u8][..], &[0x48], &[0xf0], &[0x67, 0x66], &[0x2e]] {
            let mut bytes = prefix.to_vec();
            bytes.extend_from_slice(&[0x06, 0x90, 0x90, 0x90]);
            assert_eq!(
                decode(&bytes).err(),
                Some(DecodeErr::Invalid),
                "{prefix:02x?}"
            );
        }
    }

    #[test]
    fn invalid_one_byte_opcodes_are_valid_in_the_two_byte_map() {
        // 0f 06 is clts, 0f 07 sysret, 0f 0e femms, 0f 60 punpcklbw, 0f 82 jb: only the one-byte
        // map entries were removed.
        for opcode in [
            0x06u8, 0x07, 0x0e, 0x60, 0x61, 0x62, 0x82, 0xc4, 0xc5, 0xd4, 0xd5, 0xd6,
        ] {
            assert!(
                decode(&[0x0f, opcode, 0xc0, 0x90, 0x90, 0x90, 0x90, 0x90]).is_ok(),
                "0f {opcode:#x}"
            );
        }
    }

    #[test]
    fn vex_and_evex_escapes_are_rejected_because_avx_is_not_advertised() {
        let vex3 = [0xc4, 0xe2, 0x7d, 0x18, 0xc1];
        let vex2 = [0xc5, 0xf8, 0x77];
        let evex = [0x62, 0xf1, 0x7c, 0x48, 0x28, 0xc1];
        for bytes in [&vex3[..], &vex2[..], &evex[..]] {
            assert_eq!(
                decode(bytes).err(),
                Some(DecodeErr::Invalid),
                "{bytes:02x?}"
            );
        }
    }

    #[test]
    fn neighbouring_valid_opcodes_still_decode() {
        for opcode in [0x63u8, 0x9b, 0xcc, 0xcf, 0xd7, 0xd8, 0xe8] {
            let result = decode(&[opcode, 0xc0, 0, 0, 0, 0, 0, 0]);
            assert!(result.is_ok(), "{opcode:#x} unexpectedly {result:?}");
        }
    }
}

mod length_limit_and_truncation {
    use super::*;

    #[test]
    fn empty_input_is_truncated() {
        assert_eq!(decode(&[]).err(), Some(DecodeErr::Truncated));
    }

    #[test]
    fn prefix_only_is_truncated() {
        for prefix in [
            0x66u8, 0x67, 0xf0, 0xf2, 0xf3, 0x26, 0x2e, 0x36, 0x3e, 0x64, 0x65, 0x40, 0x4f,
        ] {
            assert_eq!(
                decode(&[prefix]).err(),
                Some(DecodeErr::Truncated),
                "{prefix:#x}"
            );
        }
    }

    #[test]
    fn missing_modrm_sib_disp_and_immediate_bytes_are_truncated() {
        let cases: &[&[u8]] = &[
            &[0x8b],
            &[0x8b, 0x04],
            &[0x8b, 0x44, 0x24],
            &[0x8b, 0x84, 0x24, 0, 0, 0],
            &[0x8b, 0x05, 0, 0, 0],
            &[0x83, 0xc0],
            &[0x81, 0xc0, 1, 0, 0],
            &[0x66, 0x81, 0xc0, 1],
            &[0xb8, 1, 0, 0],
            &[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7],
            &[0xa1, 1, 2, 3, 4, 5, 6, 7],
            &[0xc2, 0x10],
            &[0xc8, 0x10, 0x00],
            &[0xe9, 0, 0, 0],
            &[0xeb],
            &[0x0f, 0x84, 0, 0, 0],
            &[0xf6, 0x00],
            &[0xf7, 0xc0, 1, 0, 0],
        ];
        for bytes in cases {
            assert_eq!(
                decode(bytes).err(),
                Some(DecodeErr::Truncated),
                "{bytes:02x?}"
            );
        }
    }

    #[test]
    fn exactly_fifteen_bytes_decode() {
        // fs: addr32 lock add qword ptr [rax*8 + 0x12345678], 1 -- REX, opcode, modrm, SIB,
        // disp32, imm32 plus three legacy prefixes.
        let bytes = [
            0x64, 0x67, 0xf0, 0x48, 0x81, 0x84, 0xc0, 0x78, 0x56, 0x34, 0x12, 0x01, 0x00, 0x00,
            0x00,
        ];
        assert_eq!(bytes.len(), 15);
        let i = decoded(&bytes);
        assert_eq!(i.len, 15);
        assert!(i.lock && i.addr32);
        assert_eq!(i.seg as usize, SEG_FS);
        assert_eq!(i.disp, 0x1234_5678);
        assert_eq!(i.imm, 1);
    }

    #[test]
    fn a_sixteenth_byte_makes_it_invalid() {
        let mut bytes = vec![
            0x2e, 0x64, 0x67, 0xf0, 0x48, 0x81, 0x84, 0xc0, 0x78, 0x56, 0x34, 0x12, 1, 0, 0, 0,
        ];
        assert_eq!(bytes.len(), 16);
        assert_eq!(decode(&bytes).err(), Some(DecodeErr::Invalid));
        bytes.truncate(15);
        assert_eq!(decode(&bytes).err(), Some(DecodeErr::Truncated));
    }

    #[test]
    fn fifteen_prefixes_need_a_sixteenth_byte_to_become_invalid() {
        let mut bytes = vec![0x66u8; 15];
        assert_eq!(decode(&bytes).err(), Some(DecodeErr::Truncated));
        bytes.push(0x90);
        assert_eq!(decode(&bytes).err(), Some(DecodeErr::Invalid));
    }

    #[test]
    fn fourteen_prefixes_and_an_opcode_is_the_longest_nop() {
        let mut bytes = vec![0x66u8; 14];
        bytes.push(0x90);
        assert_eq!(decoded(&bytes).len, 15);
    }

    #[test]
    fn decoding_stops_at_the_instruction_end() {
        assert_eq!(decoded(&[0x90, 0xff, 0xff, 0xff]).len, 1);
        assert_eq!(decoded(&[0x48, 0x89, 0xe5, 0xff, 0xff]).len, 3);
        assert_eq!(decoded(&[0xc3, 0x00]).len, 1);
    }
}

mod instruction_lengths {
    use super::*;

    /// Encodings assembled by clang; the decoder must agree with its byte counts.
    const ASSEMBLED: &[&[u8]] = &[
        // nop
        &[0x90],
        // ret
        &[0xc3],
        // hlt
        &[0xf4],
        // cli
        &[0xfa],
        // sti
        &[0xfb],
        // pause
        &[0xf3, 0x90],
        // leave
        &[0xc9],
        // cpuid
        &[0x0f, 0xa2],
        // rdtsc
        &[0x0f, 0x31],
        // syscall
        &[0x0f, 0x05],
        // sysretq
        &[0x48, 0x0f, 0x07],
        // swapgs
        &[0x0f, 0x01, 0xf8],
        // iretq
        &[0x48, 0xcf],
        // push rax
        &[0x50],
        // push r15
        &[0x41, 0x57],
        // pop rbx
        &[0x5b],
        // pop r12
        &[0x41, 0x5c],
        // mov rax, rbx
        &[0x48, 0x89, 0xd8],
        // mov eax, ebx
        &[0x89, 0xd8],
        // mov ax, bx
        &[0x66, 0x89, 0xd8],
        // mov al, bl
        &[0x88, 0xd8],
        // mov sil, dil
        &[0x40, 0x88, 0xfe],
        // mov r8b, r15b
        &[0x45, 0x88, 0xf8],
        // mov rax, [rbx]
        &[0x48, 0x8b, 0x03],
        // mov [rbx], rax
        &[0x48, 0x89, 0x03],
        // mov rax, [rbx + 8]
        &[0x48, 0x8b, 0x43, 0x08],
        // mov rax, [rbx + 0x1000]
        &[0x48, 0x8b, 0x83, 0x00, 0x10, 0x00, 0x00],
        // mov rax, [rbx + rcx]
        &[0x48, 0x8b, 0x04, 0x0b],
        // mov rax, [rbx + rcx*8 + 0x10]
        &[0x48, 0x8b, 0x44, 0xcb, 0x10],
        // mov rax, [rbx + rcx*8 + 0x1000]
        &[0x48, 0x8b, 0x84, 0xcb, 0x00, 0x10, 0x00, 0x00],
        // mov rax, fs:[0x28]
        &[0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00],
        // mov rax, gs:[rax]
        &[0x65, 0x48, 0x8b, 0x00],
        // mov eax, [rip + 0x1234]
        &[0x8b, 0x05, 0x34, 0x12, 0x00, 0x00],
        // lea rax, [rip + 0x1234]
        &[0x48, 0x8d, 0x05, 0x34, 0x12, 0x00, 0x00],
        // lea r9, [r10 + r11*4 - 8]
        &[0x4f, 0x8d, 0x4c, 0x9a, 0xf8],
        // movzx eax, byte ptr [rax]
        &[0x0f, 0xb6, 0x00],
        // movzx eax, word ptr [rax]
        &[0x0f, 0xb7, 0x00],
        // movsx rax, byte ptr [rax]
        &[0x48, 0x0f, 0xbe, 0x00],
        // movsxd rax, dword ptr [rax]
        &[0x48, 0x63, 0x00],
        // add rax, 1
        &[0x48, 0x83, 0xc0, 0x01],
        // add rax, 0x1000
        &[0x48, 0x05, 0x00, 0x10, 0x00, 0x00],
        // sub dword ptr [rsp + 8], 1
        &[0x83, 0x6c, 0x24, 0x08, 0x01],
        // and eax, 0xffff
        &[0x25, 0xff, 0xff, 0x00, 0x00],
        // or byte ptr [rax], 0x80
        &[0x80, 0x08, 0x80],
        // xor eax, eax
        &[0x31, 0xc0],
        // cmp rax, [rbx]
        &[0x48, 0x3b, 0x03],
        // test rax, rax
        &[0x48, 0x85, 0xc0],
        // test al, 1
        &[0xa8, 0x01],
        // test eax, 0x80000000
        &[0xa9, 0x00, 0x00, 0x00, 0x80],
        // inc qword ptr [rax]
        &[0x48, 0xff, 0x00],
        // dec ecx
        &[0xff, 0xc9],
        // neg rax
        &[0x48, 0xf7, 0xd8],
        // not rax
        &[0x48, 0xf7, 0xd0],
        // mul rbx
        &[0x48, 0xf7, 0xe3],
        // imul rax, rbx
        &[0x48, 0x0f, 0xaf, 0xc3],
        // imul rax, [rbx], 10
        &[0x48, 0x6b, 0x03, 0x0a],
        // div rcx
        &[0x48, 0xf7, 0xf1],
        // idiv dword ptr [rcx]
        &[0xf7, 0x39],
        // shl rax, cl
        &[0x48, 0xd3, 0xe0],
        // shr rax, 3
        &[0x48, 0xc1, 0xe8, 0x03],
        // sar eax, 1
        &[0xd1, 0xf8],
        // rol rax, 8
        &[0x48, 0xc1, 0xc0, 0x08],
        // ror dword ptr [rax], cl
        &[0xd3, 0x08],
        // shld rax, rbx, 4
        &[0x48, 0x0f, 0xa4, 0xd8, 0x04],
        // bt rax, rbx
        &[0x48, 0x0f, 0xa3, 0xd8],
        // bts qword ptr [rax], 5
        &[0x48, 0x0f, 0xba, 0x28, 0x05],
        // bsf rax, rbx
        &[0x48, 0x0f, 0xbc, 0xc3],
        // bsr rax, [rbx]
        &[0x48, 0x0f, 0xbd, 0x03],
        // bswap eax
        &[0x0f, 0xc8],
        // bswap rax
        &[0x48, 0x0f, 0xc8],
        // xchg rax, rbx
        &[0x48, 0x93],
        // xchg [rax], rbx
        &[0x48, 0x87, 0x18],
        // xadd [rax], rbx
        &[0x48, 0x0f, 0xc1, 0x18],
        // cmpxchg [rax], rbx
        &[0x48, 0x0f, 0xb1, 0x18],
        // lock cmpxchg [rax], rbx
        &[0xf0, 0x48, 0x0f, 0xb1, 0x18],
        // cmpxchg8b [rax]
        &[0x0f, 0xc7, 0x08],
        // cmovne rax, rbx
        &[0x48, 0x0f, 0x45, 0xc3],
        // sete al
        &[0x0f, 0x94, 0xc0],
        // call rax
        &[0xff, 0xd0],
        // call [rax + 8]
        &[0xff, 0x50, 0x08],
        // jmp [rax*8 + 0x1000]
        &[0xff, 0x24, 0xc5, 0x00, 0x10, 0x00, 0x00],
        // push qword ptr [rax]
        &[0xff, 0x30],
        // push 5
        &[0x6a, 0x05],
        // push 0x1000
        &[0x68, 0x00, 0x10, 0x00, 0x00],
        // ret 8
        &[0xc2, 0x08, 0x00],
        // enter 0x10, 2
        &[0xc8, 0x10, 0x00, 0x02],
        // int 0x80
        &[0xcd, 0x80],
        // rep movsb
        &[0xf3, 0xa4],
        // rep stosq
        &[0xf3, 0x48, 0xab],
        // repne scasb
        &[0xf2, 0xae],
        // movsq
        &[0x48, 0xa5],
        // cmpsb
        &[0xa6],
        // lodsd
        &[0xad],
        // in al, dx
        &[0xec],
        // out dx, al
        &[0xee],
        // in eax, 0x60
        &[0xe5, 0x60],
        // out 0x80, al
        &[0xe6, 0x80],
        // cwde
        &[0x98],
        // cdqe
        &[0x48, 0x98],
        // cdq
        &[0x99],
        // cqo
        &[0x48, 0x99],
        // lahf
        &[0x9f],
        // sahf
        &[0x9e],
        // pushfq
        &[0x9c],
        // popfq
        &[0x9d],
        // std
        &[0xfd],
        // cld
        &[0xfc],
        // clc
        &[0xf8],
        // stc
        &[0xf9],
        // cmc
        &[0xf5],
        // lgdt [rax]
        &[0x0f, 0x01, 0x10],
        // lidt [rax]
        &[0x0f, 0x01, 0x18],
        // sgdt [rax]
        &[0x0f, 0x01, 0x00],
        // ltr ax
        &[0x0f, 0x00, 0xd8],
        // lldt word ptr [rax]
        &[0x0f, 0x00, 0x10],
        // verr ax
        &[0x0f, 0x00, 0xe0],
        // invlpg [rax]
        &[0x0f, 0x01, 0x38],
        // mov cr3, rax
        &[0x0f, 0x22, 0xd8],
        // mov rax, cr0
        &[0x0f, 0x20, 0xc0],
        // mov r9, cr8
        &[0x45, 0x0f, 0x20, 0xc1],
        // mov dr7, rax
        &[0x0f, 0x23, 0xf8],
        // rdmsr
        &[0x0f, 0x32],
        // wrmsr
        &[0x0f, 0x30],
        // rdpmc
        &[0x0f, 0x33],
        // wbinvd
        &[0x0f, 0x09],
        // lmsw ax
        &[0x0f, 0x01, 0xf0],
        // smsw eax
        &[0x0f, 0x01, 0xe0],
        // clflush [rax]
        &[0x0f, 0xae, 0x38],
        // mfence
        &[0x0f, 0xae, 0xf0],
        // lfence
        &[0x0f, 0xae, 0xe8],
        // sfence
        &[0x0f, 0xae, 0xf8],
        // prefetcht0 [rax]
        &[0x0f, 0x18, 0x08],
        // endbr64
        &[0xf3, 0x0f, 0x1e, 0xfa],
        // ud2
        &[0x0f, 0x0b],
        // fld1
        &[0xd9, 0xe8],
        // fldz
        &[0xd9, 0xee],
        // fld qword ptr [rax]
        &[0xdd, 0x00],
        // fld tbyte ptr [rax + 0x10]
        &[0xdb, 0x68, 0x10],
        // fstp st(3)
        &[0xdd, 0xdb],
        // fadd st, st(2)
        &[0xd8, 0xc2],
        // fxch st(1)
        &[0xd9, 0xc9],
        // fnstcw [rax]
        &[0xd9, 0x38],
        // fldcw [rax]
        &[0xd9, 0x28],
        // fnstsw ax
        &[0xdf, 0xe0],
        // fxsave [rax]
        &[0x0f, 0xae, 0x00],
        // fxrstor [rax + rcx*8]
        &[0x0f, 0xae, 0x0c, 0xc8],
        // fucomip st, st(1)
        &[0xdf, 0xe9],
        // fcmovb st, st(1)
        &[0xda, 0xc1],
        // movaps xmm0, [rax]
        &[0x0f, 0x28, 0x00],
        // movdqu xmm1, [rax + 0x10]
        &[0xf3, 0x0f, 0x6f, 0x48, 0x10],
        // movdqa [rax], xmm15
        &[0x66, 0x44, 0x0f, 0x7f, 0x38],
        // movups xmm8, xmm9
        &[0x45, 0x0f, 0x10, 0xc1],
        // movss xmm0, [rax]
        &[0xf3, 0x0f, 0x10, 0x00],
        // movsd xmm0, [rax]
        &[0xf2, 0x0f, 0x10, 0x00],
        // movq xmm0, rax
        &[0x66, 0x48, 0x0f, 0x6e, 0xc0],
        // movd eax, xmm0
        &[0x66, 0x0f, 0x7e, 0xc0],
        // pxor xmm0, xmm0
        &[0x66, 0x0f, 0xef, 0xc0],
        // pshufd xmm0, xmm1, 0
        &[0x66, 0x0f, 0x70, 0xc1, 0x00],
        // pcmpeqb xmm0, [rax]
        &[0x66, 0x0f, 0x74, 0x00],
        // pmovmskb eax, xmm0
        &[0x66, 0x0f, 0xd7, 0xc0],
        // addsd xmm0, xmm1
        &[0xf2, 0x0f, 0x58, 0xc1],
        // cvtsi2sd xmm0, rax
        &[0xf2, 0x48, 0x0f, 0x2a, 0xc0],
        // cvttsd2si rax, xmm0
        &[0xf2, 0x48, 0x0f, 0x2c, 0xc0],
        // ucomisd xmm0, xmm1
        &[0x66, 0x0f, 0x2e, 0xc1],
        // sqrtpd xmm0, xmm1
        &[0x66, 0x0f, 0x51, 0xc1],
        // ldmxcsr [rax]
        &[0x0f, 0xae, 0x10],
        // stmxcsr [rax]
        &[0x0f, 0xae, 0x18],
        // movnti [rax], ecx
        &[0x0f, 0xc3, 0x08],
        // movntdq [rax], xmm0
        &[0x66, 0x0f, 0xe7, 0x00],
        // emms
        &[0x0f, 0x77],
        // pxor mm0, mm1
        &[0x0f, 0xef, 0xc1],
        // popcnt rax, rbx
        &[0xf3, 0x48, 0x0f, 0xb8, 0xc3],
        // lzcnt rax, rbx
        &[0xf3, 0x48, 0x0f, 0xbd, 0xc3],
        // tzcnt eax, ebx
        &[0xf3, 0x0f, 0xbc, 0xc3],
        // crc32 eax, bl
        &[0xf2, 0x0f, 0x38, 0xf0, 0xc3],
        // pshufb xmm0, xmm1
        &[0x66, 0x0f, 0x38, 0x00, 0xc1],
        // palignr xmm0, xmm1, 4
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x04],
        // movbe eax, [rax]
        &[0x0f, 0x38, 0xf0, 0x00],
        // xgetbv
        &[0x0f, 0x01, 0xd0],
        // rdtscp
        &[0x0f, 0x01, 0xf9],
        // monitor
        &[0x0f, 0x01, 0xc8],
        // mwait
        &[0x0f, 0x01, 0xc9],
        // lock add dword ptr [rax], 1
        &[0xf0, 0x83, 0x00, 0x01],
        // data16 nop word ptr cs:[rax + rax]
        &[0x66, 0x2e, 0x66, 0x0f, 0x1f, 0x04, 0x00],
        // nop dword ptr [rax + rax]
        &[0x0f, 0x1f, 0x04, 0x00],
        // nop word ptr [rax + rax]
        &[0x66, 0x0f, 0x1f, 0x04, 0x00],
    ];

    #[test]
    fn every_assembled_instruction_decodes_to_exactly_its_length() {
        for bytes in ASSEMBLED {
            assert_length(bytes);
        }
    }

    #[test]
    fn trailing_bytes_are_never_consumed() {
        for bytes in ASSEMBLED {
            assert_ignores_trailing_bytes(bytes);
        }
    }

    #[test]
    fn decoding_is_deterministic() {
        for bytes in ASSEMBLED {
            let first = decoded(bytes);
            let second = decoded(bytes);
            assert_eq!(first.len, second.len);
            assert_eq!(first.op, second.op);
            assert_eq!(first.imm, second.imm);
            assert_eq!(first.disp, second.disp);
        }
    }
}

mod exhaustive {
    use super::*;

    /// Padding long enough for any instruction: ModRM c0 (register form) then zeros.
    fn padded(prefix: &[u8], opcode: &[u8]) -> Vec<u8> {
        let mut bytes = prefix.to_vec();
        bytes.extend_from_slice(opcode);
        bytes.push(0xc0);
        bytes.resize(MAX_INSN_LEN, 0);
        bytes
    }

    /// Opcode bytes that are prefixes or escapes rather than opcodes.
    fn is_prefix_or_escape(opcode: u8) -> bool {
        matches!(
            opcode,
            0x26 | 0x2e | 0x36 | 0x3e | 0x40
                ..=0x4f | 0x64 | 0x65 | 0x66 | 0x67 | 0x0f | 0xf0 | 0xf2 | 0xf3
        )
    }

    #[test]
    fn exactly_the_documented_one_byte_opcodes_are_invalid_in_long_mode() {
        let invalid: [u8; 23] = [
            0x06, 0x07, 0x0e, 0x16, 0x17, 0x1e, 0x1f, 0x27, 0x2f, 0x37, 0x3f, 0x60, 0x61, 0x62,
            0x82, 0x9a, 0xc4, 0xc5, 0xce, 0xd4, 0xd5, 0xd6, 0xea,
        ];
        for opcode in (0..=0xffu8).filter(|&opcode| !is_prefix_or_escape(opcode)) {
            let result = decode(&padded(&[], &[opcode]));
            assert_eq!(
                result.is_err(),
                invalid.contains(&opcode),
                "opcode {opcode:#04x}: {result:?}"
            );
        }
    }

    #[test]
    fn every_two_byte_opcode_decodes() {
        for opcode in 0..=0xffu8 {
            if opcode == 0x38 || opcode == 0x3a {
                continue;
            }
            let result = decode(&padded(&[], &[0x0f, opcode]));
            assert!(result.is_ok(), "0f {opcode:#04x}: {result:?}");
        }
    }

    #[test]
    fn every_opcode_decodes_the_same_with_every_legacy_prefix_in_front() {
        for prefix in [0x66u8, 0x67, 0xf0, 0xf2, 0xf3, 0x2e, 0x64, 0x65] {
            for opcode in (0..=0xffu8).filter(|&opcode| !is_prefix_or_escape(opcode)) {
                let plain = decode(&padded(&[], &[opcode]));
                let prefixed = decode(&padded(&[prefix], &[opcode]));
                match (plain, prefixed) {
                    (Err(a), Err(b)) => assert_eq!(a, b),
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a.op, b.op, "{prefix:#x} {opcode:#x}");
                        // 66 resizes immediates and 67 resizes the moffs operand; nothing else moves
                        let moffs = (0xa0..=0xa3).contains(&opcode);
                        if prefix != 0x66 && !(prefix == 0x67 && moffs) {
                            assert_eq!(a.len + 1, b.len, "{prefix:#x} {opcode:#x}");
                        }
                    }
                    (a, b) => panic!("{prefix:#x} {opcode:#x}: {a:?} vs {b:?}"),
                }
            }
        }
    }

    /// Length of `8b /r` (mov r32, r/m32) computed from the encoding rules alone.
    fn expected_mov_length(modrm: u8, sib: u8) -> usize {
        let (mode, rm) = (modrm >> 6, modrm & 7);
        if mode == 3 {
            return 2;
        }
        let has_sib = rm == 4;
        let displacement = match mode {
            1 => 1,
            2 => 4,
            _ if rm == 5 => 4,
            _ if has_sib && sib & 7 == 5 => 4,
            _ => 0,
        };
        2 + has_sib as usize + displacement
    }

    #[test]
    fn every_modrm_and_sib_combination_has_the_architectural_length() {
        for modrm in 0..=0xffu8 {
            for sib in 0..=0xffu8 {
                let mut bytes = vec![0x8b, modrm, sib];
                bytes.extend_from_slice(&[0x11; 4]);
                let i = decode(&bytes).unwrap();
                assert_eq!(
                    i.len as usize,
                    expected_mov_length(modrm, sib),
                    "modrm {modrm:#04x} sib {sib:#04x}"
                );
                if modrm >> 6 != 3 && modrm & 7 == 4 {
                    assert_eq!(i.scale, sib >> 6);
                }
            }
        }
    }

    #[test]
    fn modrm_forms_are_independent_of_rex_bits_for_length() {
        for rex in 0x40..=0x4fu8 {
            for modrm in 0..=0xffu8 {
                let mut bytes = vec![rex, 0x8b, modrm, 0x24];
                bytes.extend_from_slice(&[0x22; 4]);
                let i = decode(&bytes).unwrap();
                assert_eq!(
                    i.len as usize,
                    1 + expected_mov_length(modrm, 0x24),
                    "rex {rex:#04x} modrm {modrm:#04x}"
                );
            }
        }
    }

    #[test]
    fn every_truncation_of_every_modrm_form_is_truncated_not_invalid() {
        for modrm in 0..=0xffu8 {
            let full = [0x8b, modrm, 0x25, 0x10, 0x20, 0x30, 0x40];
            let length = expected_mov_length(modrm, 0x25);
            for cut in 0..length {
                assert_eq!(
                    decode(&full[..cut]).err(),
                    Some(DecodeErr::Truncated),
                    "modrm {modrm:#04x} cut at {cut}"
                );
            }
        }
    }
}
