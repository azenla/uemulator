//! Tests for broadcasts, block inserts and extracts, element inserts and extracts, the half-lane
//! moves, gathers and scatters.

use super::*;

const K1: u64 = 0b1010_0110_1100_0101;

#[test]
fn broadcasts_replicate_the_source_element_under_every_mask() {
    for (name, variants, size) in [
        ("vpbroadcastd", &VPBROADCASTD, 4usize),
        ("vpbroadcastq", &VPBROADCASTQ, 8),
        ("vpbroadcastb", &VPBROADCASTB, 1),
        ("vpbroadcastw", &VPBROADCASTW, 2),
        ("vbroadcastss", &VBROADCASTSS, 4),
        ("vbroadcastsd", &VBROADCASTSD, 8),
    ] {
        let mut rng = Rng(0xb0 ^ name.len() as u64);
        let (old, source) = (rng.vector(), rng.vector());
        for (index, length) in LENGTHS.into_iter().enumerate() {
            for (shape, code) in [
                ("plain", variants.p[index]),
                ("merge", variants.k[index]),
                ("zero", variants.kz[index]),
                ("memory", variants.m[index]),
                ("memory merge", variants.mk[index]),
            ] {
                if code.is_empty() {
                    continue;
                }
                let masked = shape != "plain" && shape != "memory";
                let h = run(code, |h| {
                    h.cpu.set_zmm(0, old);
                    h.cpu.set_zmm(2, source);
                    h.cpu.opmask[1] = K1;
                    h.write(DATA, &bytes_of(&source));
                });
                let first = element(&source, size, 0);
                let want = expected(
                    length,
                    size,
                    masked.then_some((K1, shape == "zero")),
                    &old,
                    |_| first,
                );
                assert_eq!(h.cpu.zmm(0), want, "{name} {shape} {length}");
            }
        }
    }
}

#[test]
fn broadcast_of_a_pair_of_dwords_masks_per_dword() {
    let mut rng = Rng(0xb1);
    let (old, source) = (rng.vector(), rng.vector());
    let h = run(VBROADCASTF32X2.k[2], |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(2, source);
        h.cpu.opmask[1] = K1;
    });
    let want = expected(64, 4, Some((K1, false)), &old, |k| {
        element(&source, 4, k % 2)
    });
    assert_eq!(h.cpu.zmm(0), want);
    let h = run(VBROADCASTI32X2.p[1], |h| h.cpu.set_zmm(2, source));
    let want = expected(32, 4, None, &[0; 4], |k| element(&source, 4, k % 2));
    assert_eq!(h.cpu.zmm(0), want);
}

#[test]
fn broadcasts_from_general_registers() {
    let mut rng = Rng(0xb2);
    let old = rng.vector();
    let h = run(VPBROADCASTD_GPR, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.gpr[0] = 0xdead_beef_1234_5678;
        h.cpu.opmask[1] = K1;
    });
    assert_eq!(
        h.cpu.zmm(0),
        expected(64, 4, Some((K1, false)), &old, |_| 0x1234_5678)
    );
    let h = run(VPBROADCASTQ_GPR, |h| h.cpu.gpr[0] = 0x0123_4567_89ab_cdef);
    assert_eq!(
        h.cpu.zmm(0),
        expected(64, 8, None, &[0; 4], |_| 0x0123_4567_89ab_cdef)
    );
    let h = run(VPBROADCASTB_GPR, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.gpr[0] = 0xabcd;
        h.cpu.opmask[1] = 0xff00_ff00;
    });
    assert_eq!(
        h.cpu.zmm(0),
        expected(32, 1, Some((0xff00_ff00, true)), &old, |_| 0xcd)
    );
    let h = run(VPBROADCASTW_GPR, |h| h.cpu.gpr[0] = 0x1_8001);
    assert_eq!(h.cpu.zmm(0), expected(16, 2, None, &[0; 4], |_| 0x8001));
}

#[test]
fn block_broadcasts_repeat_a_memory_block() {
    let mut rng = Rng(0xb3);
    let (old, memory) = (rng.vector(), rng.vector());
    let bytes = bytes_of(&memory);
    let h = run(VBROADCASTF32X4, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.opmask[1] = K1;
        h.write(DATA, &bytes);
    });
    let want = expected(64, 4, Some((K1, false)), &old, |k| {
        element(&memory, 4, k % 4)
    });
    assert_eq!(h.cpu.zmm(0), want);
    let h = run(VBROADCASTF32X4_Y, |h| h.write(DATA, &bytes));
    assert_eq!(h.cpu.zmm(0), [memory[0], memory[0], 0, 0]);
    let h = run(VBROADCASTI64X2, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.opmask[1] = 0x0f;
        h.write(DATA, &bytes);
    });
    let want = expected(64, 8, Some((0x0f, true)), &old, |k| {
        element(&memory, 8, k % 2)
    });
    assert_eq!(h.cpu.zmm(0), want);
    let h = run(VBROADCASTF64X4, |h| h.write(DATA, &bytes));
    assert_eq!(h.cpu.zmm(0), [memory[0], memory[1], memory[0], memory[1]]);
    let h = run(VBROADCASTI32X8, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.opmask[1] = 0xffff;
        h.write(DATA, &bytes);
    });
    assert_eq!(h.cpu.zmm(0), [memory[0], memory[1], memory[0], memory[1]]);
    // disp8 scales by the block size: [rsi+32] is disp8 2
    let h = run(VBROADCASTI32X4_DISP8, |h| h.write(DATA + 32, &bytes));
    assert_eq!(h.cpu.zmm(0), [memory[0]; 4]);
}

#[test]
fn non_temporal_load_needs_alignment() {
    let mut rng = Rng(0xb4);
    let memory = rng.vector();
    let h = run(VMOVNTDQA_Z, |h| h.write(DATA, &bytes_of(&memory)));
    assert_eq!(h.cpu.zmm(0), memory);
    assert!(matches!(
        stop_of(VMOVNTDQA_Z, |h| h.cpu.gpr[RSI] = DATA + 16),
        Stop::Fault { vector: 13, .. }
    ));
}

#[test]
fn block_inserts_replace_one_lane_group() {
    let mut rng = Rng(0xb5);
    let (old, a, b) = (rng.vector(), rng.vector(), rng.vector());
    let h = run(VINSERTF32X4_Z, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    assert_eq!(h.cpu.zmm(0), [a[0], a[1], b[0], a[3]]);
    let h = run(VINSERTF32X4_ZK, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
        h.cpu.opmask[1] = K1;
    });
    let full = [a[0], a[1], a[2], b[0]];
    assert_eq!(
        h.cpu.zmm(0),
        expected(64, 4, Some((K1, true)), &old, |k| element(&full, 4, k))
    );
    let h = run(VINSERTI64X2_ZM, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(1, a);
        h.cpu.opmask[1] = K1;
        h.write(DATA, &bytes_of(&b));
    });
    let full = [a[0], b[0], a[2], a[3]];
    assert_eq!(
        h.cpu.zmm(0),
        expected(64, 8, Some((K1, false)), &old, |k| element(&full, 8, k))
    );
    let h = run(VINSERTI32X4_Y, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    assert_eq!(h.cpu.zmm(0), [a[0], b[0], 0, 0]);
    let h = run(VINSERTF64X4_Z, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    assert_eq!(h.cpu.zmm(0), [a[0], a[1], b[0], b[1]]);
    let h = run(VINSERTI32X8_ZK, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
        h.cpu.opmask[1] = 0xff00;
    });
    let full = [b[0], b[1], a[2], a[3]];
    assert_eq!(
        h.cpu.zmm(0),
        expected(64, 4, Some((0xff00, false)), &old, |k| element(&full, 4, k))
    );
}

#[test]
fn block_extracts_read_one_lane_group() {
    let mut rng = Rng(0xb6);
    let (old, a) = (rng.vector(), rng.vector());
    let h = run(VEXTRACTF32X4_Z, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(1, a);
    });
    assert_eq!(h.cpu.zmm(0), [a[3], 0, 0, 0]);
    let h = run(VEXTRACTF32X4_ZK, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(1, a);
        h.cpu.opmask[1] = 0b0101;
    });
    let part = [a[2], 0, 0, 0];
    assert_eq!(
        h.cpu.zmm(0),
        expected(16, 4, Some((0b0101, true)), &old, |k| element(&part, 4, k))
    );
    let h = run(VEXTRACTI64X2_ZMEM, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.opmask[1] = 0b10;
        h.write(OUT, &[0xee; 32]);
    });
    let stored = h.read(OUT, 32);
    assert!(
        stored[..8].iter().all(|&b| b == 0xee),
        "masked-out quadword is not written"
    );
    assert_eq!(&stored[8..16], &element(&a, 8, 3).to_le_bytes());
    assert!(stored[16..].iter().all(|&b| b == 0xee));
    let h = run(VEXTRACTI32X4_Y, |h| h.cpu.set_zmm(1, a));
    assert_eq!(h.cpu.zmm(0), [a[1], 0, 0, 0]);
    let h = run(VEXTRACTF64X4_Z, |h| h.cpu.set_zmm(1, a));
    assert_eq!(h.cpu.zmm(0), [a[2], a[3], 0, 0]);
    let h = run(VEXTRACTI64X4_ZMEM, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.opmask[1] = 0b1001;
        h.write(OUT, &[0xee; 64]);
    });
    let stored = h.read(OUT, 64);
    assert_eq!(&stored[..8], &element(&a, 8, 0).to_le_bytes());
    assert!(stored[8..24].iter().all(|&b| b == 0xee));
    assert_eq!(&stored[24..32], &element(&a, 8, 3).to_le_bytes());
    let h = run(VEXTRACTI32X8_ZK, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(1, a);
        h.cpu.opmask[1] = 0xf0;
    });
    let part = [a[2], a[3], 0, 0];
    assert_eq!(
        h.cpu.zmm(0),
        expected(32, 4, Some((0xf0, false)), &old, |k| element(&part, 4, k))
    );
}

#[test]
fn element_extracts_and_inserts() {
    let mut rng = Rng(0xb7);
    let v = rng.vector();
    let h = run(VPEXTRB_EAX, |h| {
        h.cpu.set_zmm(1, v);
        h.cpu.gpr[0] = u64::MAX;
    });
    assert_eq!(h.cpu.gpr[0], element(&v, 1, 5));
    let h = run(VPEXTRB_MEM, |h| {
        h.cpu.set_zmm(1, v);
        h.write(OUT, &[0xee; 4]);
    });
    assert_eq!(h.read(OUT, 3), vec![element(&v, 1, 9) as u8, 0xee, 0xee]);
    let h = run(VPEXTRW_EAX, |h| {
        h.cpu.set_zmm(1, v);
        h.cpu.gpr[0] = u64::MAX;
    });
    assert_eq!(h.cpu.gpr[0], element(&v, 2, 3));
    let h = run(VPEXTRW_MEM, |h| {
        h.cpu.set_zmm(1, v);
        h.write(OUT, &[0xee; 4]);
    });
    assert_eq!(
        h.read(OUT, 3),
        vec![
            element(&v, 2, 6) as u8,
            (element(&v, 2, 6) >> 8) as u8,
            0xee
        ]
    );
    let h = run(VPEXTRD_EAX, |h| {
        h.cpu.set_zmm(1, v);
        h.cpu.gpr[0] = u64::MAX;
    });
    assert_eq!(h.cpu.gpr[0], element(&v, 4, 2));
    let h = run(VPEXTRQ_RAX, |h| h.cpu.set_zmm(1, v));
    assert_eq!(h.cpu.gpr[0], element(&v, 8, 1));
    let h = run(VPEXTRQ_MEM, |h| h.cpu.set_zmm(1, v));
    assert_eq!(h.read_u64(OUT), element(&v, 8, 0));
    let h = run(VEXTRACTPS_EAX, |h| h.cpu.set_zmm(1, v));
    assert_eq!(h.cpu.gpr[0], element(&v, 4, 2));
    // inserts into the copy of vvvv, zeroing everything above the low lane
    let (a, g) = (rng.vector(), 0x1234_5678_9abc_def0u64);
    for (code, size, index) in [
        (VPINSRB_HIGH, 1usize, 7usize),
        (VPINSRW_HIGH, 2, 4),
        (VPINSRD_HIGH, 4, 3),
        (VPINSRQ_HIGH, 8, 1),
    ] {
        let h = run(code, |h| {
            h.cpu.set_zmm(17, a);
            h.cpu.gpr[0] = g;
        });
        let mut want = [a[0], 0, 0, 0];
        set_element(&mut want, size, index, g);
        assert_eq!(h.cpu.zmm(16), want, "vpinsr size {size}");
    }
    let h = run(VPINSRD_MEM, |h| {
        h.cpu.set_zmm(1, a);
        h.write(DATA, &0xcafe_f00du32.to_le_bytes());
    });
    let mut want = [a[0], 0, 0, 0];
    set_element(&mut want, 4, 2, 0xcafe_f00d);
    assert_eq!(h.cpu.zmm(0), want);
}

#[test]
fn insertps_selects_an_element_and_zeroes_by_mask() {
    let mut rng = Rng(0xb8);
    let (a, b) = (rng.vector(), rng.vector());
    // imm 0x9d: source element 2, destination element 1, zero mask 0b1101
    let h = run(VINSERTPS_REG, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    let mut want = [a[0], 0, 0, 0];
    set_element(&mut want, 4, 1, element(&b, 4, 2));
    for k in 0..4 {
        if 0xd >> k & 1 == 1 {
            set_element(&mut want, 4, k, 0);
        }
    }
    assert_eq!(h.cpu.zmm(0), want);
    // a memory source is always the m32 operand: imm 0x20 puts it in element 2
    let h = run(VINSERTPS_MEM, |h| {
        h.cpu.set_zmm(1, a);
        h.write(DATA, &0x4242_4242u32.to_le_bytes());
    });
    let mut want = [a[0], 0, 0, 0];
    set_element(&mut want, 4, 2, 0x4242_4242);
    assert_eq!(h.cpu.zmm(0), want);
}

#[test]
fn half_lane_moves() {
    let mut rng = Rng(0xb9);
    let (a, b) = (rng.vector(), rng.vector());
    let h = run(VMOVLPS_LOAD, |h| {
        h.cpu.set_zmm(1, a);
        h.write(DATA, &0x1122_3344_5566_7788u64.to_le_bytes());
    });
    assert_eq!(
        h.cpu.zmm(0),
        [a[0] & !(u64::MAX as u128) | 0x1122_3344_5566_7788, 0, 0, 0]
    );
    let h = run(VMOVHPD_LOAD, |h| {
        h.cpu.set_zmm(1, a);
        h.write(DATA, &0x1122_3344_5566_7788u64.to_le_bytes());
    });
    assert_eq!(
        h.cpu.zmm(0),
        [
            a[0] & u64::MAX as u128 | 0x1122_3344_5566_7788 << 64,
            0,
            0,
            0
        ]
    );
    let h = run(VMOVLPD_STORE, |h| h.cpu.set_zmm(1, a));
    assert_eq!(h.read_u64(OUT), a[0] as u64);
    let h = run(VMOVHPS_STORE, |h| h.cpu.set_zmm(1, a));
    assert_eq!(h.read_u64(OUT), (a[0] >> 64) as u64);
    let h = run(VMOVHLPS, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    assert_eq!(
        h.cpu.zmm(0),
        [a[0] & !(u64::MAX as u128) | b[0] >> 64, 0, 0, 0]
    );
    let h = run(VMOVLHPS, |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    assert_eq!(
        h.cpu.zmm(0),
        [a[0] & u64::MAX as u128 | b[0] << 64, 0, 0, 0]
    );
}

#[test]
fn psadbw_and_aes_and_clmul_work_per_lane() {
    let mut rng = Rng(0xba);
    let (a, b) = (rng.vector(), rng.vector());
    let h = run(VPSADBW.p[2], |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    let want = build(64, 8, |k| {
        (0..8)
            .map(|j| {
                (element(&a, 1, 8 * k + j) as i64 - element(&b, 1, 8 * k + j) as i64).unsigned_abs()
            })
            .sum()
    });
    assert_eq!(h.cpu.zmm(0), want);
    // each 128-bit lane goes through the same AES round: compare lane 0 with the legacy
    // instruction on that lane, and check the other lanes agree with lane 0 on equal inputs
    let state = rng.next() as u128 | (rng.next() as u128) << 64;
    let key = rng.next() as u128 | (rng.next() as u128) << 64;
    let h = run(VAESENC.p[2], |h| {
        h.cpu.set_zmm(1, [state; 4]);
        h.cpu.set_zmm(2, [key; 4]);
    });
    let got = h.cpu.zmm(0);
    assert_eq!(got[0], got[1]);
    assert_eq!(got[1], got[2]);
    assert_eq!(got[2], got[3]);
    // legacy aesenc xmm0, xmm1 as the reference for one lane
    let legacy = run(&[0x66, 0x0f, 0x38, 0xdc, 0xc1, 0xf4], |h| {
        h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
        h.cpu.set_zmm(0, [state, 0, 0, 0]);
        h.cpu.set_zmm(1, [key, 0, 0, 0]);
    });
    assert_eq!(got[0], legacy.cpu.zmm(0)[0]);
    let h = run(VPCLMULQDQ.p[2], |h| {
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
    });
    // imm 0x11: high quadwords of each lane, carry-less product
    let clmul = |x: u64, y: u64| -> u128 {
        let mut r = 0u128;
        for bit in 0..64 {
            if y >> bit & 1 == 1 {
                r ^= (x as u128) << bit;
            }
        }
        r
    };
    for lane in 0..4 {
        assert_eq!(
            h.cpu.zmm(0)[lane],
            clmul((a[lane] >> 64) as u64, (b[lane] >> 64) as u64),
            "lane {lane}"
        );
    }
}

mod gathers_and_scatters {
    use super::*;

    const TABLE: u64 = DATA;

    fn table() -> Vec<u8> {
        (0..4096u32)
            .map(|k| (k.wrapping_mul(2654435761) >> 13) as u8)
            .collect()
    }

    fn at(table: &[u8], offset: u64, size: usize) -> u64 {
        let mut v = 0u64;
        for b in 0..size {
            v |= (table[offset as usize + b] as u64) << (8 * b);
        }
        v
    }

    /// Runs a gather and returns (destination, mask register).
    fn gather(
        code: &[u8],
        index_size: usize,
        indices: &[u64],
        k: u64,
        old: Vector,
        high: bool,
    ) -> (Harness, Vector) {
        let mut idx = [0u128; 4];
        for (n, &v) in indices.iter().enumerate().take(64 / index_size) {
            set_element(&mut idx, index_size, n, v);
        }
        let t = table();
        let h = run(code, |h| {
            h.write(TABLE, &t);
            h.cpu.set_zmm(if high { 20 } else { 2 }, idx);
            h.cpu.set_zmm(if high { 16 } else { 0 }, old);
            h.cpu.opmask[1] = k;
        });
        (h, idx)
    }

    #[test]
    fn dword_gathers_load_the_selected_elements_and_clear_the_mask() {
        let mut rng = Rng(0xc1);
        let old = rng.vector();
        let indices: Vec<u64> = (0..16).map(|k| (k * 37 % 100) as u64).collect();
        let t = table();
        let k = 0b1011_0101_1100_0011u64;
        let (h, _) = gather(VPGATHERDD_Z, 4, &indices, k, old, false);
        let want = expected(64, 4, Some((k, false)), &old, |n| at(&t, indices[n] * 4, 4));
        assert_eq!(h.cpu.zmm(0), want);
        assert_eq!(h.cpu.opmask[1], 0, "the mask register is consumed");
        // 256- and 128-bit forms zero everything above the gathered elements
        let (h, _) = gather(VPGATHERDD_Y, 4, &indices, 0xff, old, false);
        let want = expected(32, 4, None, &old, |n| at(&t, indices[n] * 4, 4));
        assert_eq!(h.cpu.zmm(0), want);
        let (h, _) = gather(VPGATHERDD_X, 4, &indices, 0b0101, old, false);
        let want = expected(16, 4, Some((0b0101, false)), &old, |n| {
            at(&t, indices[n] * 4, 4)
        });
        assert_eq!(h.cpu.zmm(0), want);
        // a displacement scales by the element size: [rsi + zmm2*4 + 8] is disp8 2
        let (h, _) = gather(VPGATHERDD_DISP8, 4, &indices, 0xffff, old, false);
        let want = expected(64, 4, None, &old, |n| at(&t, indices[n] * 4 + 8, 4));
        assert_eq!(h.cpu.zmm(0), want);
        // registers above 15, index scaled by 2
        let (h, _) = gather(VPGATHERDD_HIGH, 4, &indices, 0xffff, old, true);
        let want = expected(64, 4, None, &old, |n| at(&t, indices[n] * 2, 4));
        assert_eq!(h.cpu.zmm(16), want);
    }

    #[test]
    fn index_and_data_sizes_combine_in_every_way() {
        let mut rng = Rng(0xc2);
        let old = rng.vector();
        let t = table();
        let indices: Vec<u64> = (0..16).map(|k| (k * 29 % 80) as u64).collect();
        // dq: dword indices (the low half of the index register), qword data
        let (h, _) = gather(VPGATHERDQ_Z, 4, &indices, 0xff, old, false);
        assert_eq!(
            h.cpu.zmm(0),
            expected(64, 8, None, &old, |n| at(&t, indices[n] * 8, 8))
        );
        // qd: qword indices, dword data in a 256-bit destination
        let (h, _) = gather(VPGATHERQD_Z, 8, &indices, 0xff, old, false);
        assert_eq!(
            h.cpu.zmm(0),
            expected(32, 4, None, &old, |n| at(&t, indices[n] * 4, 4))
        );
        // qq
        let (h, _) = gather(VPGATHERQQ_Z, 8, &indices, 0b1010_0101, old, false);
        assert_eq!(
            h.cpu.zmm(0),
            expected(64, 8, Some((0b1010_0101, false)), &old, |n| at(
                &t,
                indices[n] * 8,
                8
            ))
        );
        // float forms are the same loads
        let (h, _) = gather(VGATHERDPS_Z, 4, &indices, 0xffff, old, false);
        assert_eq!(
            h.cpu.zmm(0),
            expected(64, 4, None, &old, |n| at(&t, indices[n] * 4, 4))
        );
        let (h, _) = gather(VGATHERQPD_Z, 8, &indices, 0xff, old, false);
        assert_eq!(
            h.cpu.zmm(0),
            expected(64, 8, None, &old, |n| at(&t, indices[n] * 8, 8))
        );
        let (h, _) = gather(VGATHERQPS_Z, 8, &indices, 0xff, old, false);
        assert_eq!(
            h.cpu.zmm(0),
            expected(32, 4, None, &old, |n| at(&t, indices[n] * 4, 4))
        );
    }

    #[test]
    fn negative_indices_are_sign_extended() {
        let mut rng = Rng(0xc3);
        let old = rng.vector();
        let t = table();
        let indices: Vec<u64> = (0..16)
            .map(|k| (-(k as i64) & 0xffff_ffff) as u64)
            .collect();
        let h = run(VPGATHERDD_Z, |h| {
            h.write(TABLE - 128, &[0x5a; 128]);
            h.write(TABLE, &t);
            let mut idx = [0u128; 4];
            for (n, &v) in indices.iter().enumerate() {
                set_element(&mut idx, 4, n, v);
            }
            h.cpu.set_zmm(2, idx);
            h.cpu.set_zmm(0, old);
            h.cpu.opmask[1] = 0xffff;
        });
        let want = expected(64, 4, None, &old, |n| {
            if n == 0 { at(&t, 0, 4) } else { 0x5a5a_5a5a }
        });
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn a_faulting_gather_keeps_the_completed_elements_and_their_cleared_mask_bits() {
        let t = table();
        let mut rng = Rng(0xc4);
        let old = rng.vector();
        let unmapped = 0x180_0000u64;
        let mut h = rig(VPGATHERDD_Z);
        h.write(TABLE, &t);
        // element 5 reads from the unmapped region
        let mut idx = [0u128; 4];
        for k in 0..16 {
            set_element(
                &mut idx,
                4,
                k,
                if k == 5 {
                    (unmapped - TABLE) / 4
                } else {
                    k as u64
                },
            );
        }
        crate::cpu::test_support::unmap_boot_identity_2m(&mut h, unmapped);
        h.cpu.set_zmm(2, idx);
        h.cpu.set_zmm(0, old);
        h.cpu.opmask[1] = 0xffff;
        assert!(matches!(h.run(10), Stop::Fault { vector: 14, .. }));
        assert_eq!(h.cpu.opmask[1], 0xffff & !0b11111, "elements 0-4 completed");
        let got = h.cpu.zmm(0);
        for k in 0..16 {
            let want = if k < 5 {
                at(&t, 4 * k as u64, 4)
            } else {
                element(&old, 4, k)
            };
            assert_eq!(element(&got, 4, k), want, "element {k}");
        }
        // restarting after the fault is repaired finishes the job
        h.cpu.rip = crate::testkit::CODE_BASE;
        h.cpu.interrupt_shadow = false;
        set_element(&mut idx, 4, 5, 5);
        h.cpu.set_zmm(2, idx);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.opmask[1], 0);
        for k in 0..16 {
            assert_eq!(
                element(&h.cpu.zmm(0), 4, k),
                at(&t, 4 * k as u64, 4),
                "element {k}"
            );
        }
    }

    #[test]
    fn invalid_gather_encodings() {
        let ud = Stop::Fault {
            vector: 6,
            error_code: None,
            software: false,
        };
        // destination equal to the index register
        let mut code = VPGATHERDD_Z.to_vec();
        code[5] = (code[5] & !0x38) | (2 << 3);
        assert_eq!(stop_of(&code, |h| h.cpu.opmask[1] = 1), ud);
        // no mask register
        let mut code = VPGATHERDD_Z.to_vec();
        code[3] &= !7;
        assert_eq!(stop_of(&code, |_| {}), ud);
    }

    #[test]
    fn scatters_store_the_selected_elements_in_order() {
        let mut rng = Rng(0xc5);
        let data = rng.vector();
        let indices: Vec<u64> = (0..16).map(|k| (k * 3) as u64).collect();
        let mut idx = [0u128; 4];
        for (n, &v) in indices.iter().enumerate() {
            set_element(&mut idx, 4, n, v);
        }
        let k = 0b1100_1010_0101_0011u64;
        let h = run(VPSCATTERDD_Z, |h| {
            h.cpu.set_zmm(0, data);
            h.cpu.set_zmm(2, idx);
            h.cpu.opmask[1] = k;
            h.write(OUT, &[0xee; 256]);
        });
        let stored = h.read(OUT, 256);
        let mut want = vec![0xeeu8; 256];
        for n in 0..16 {
            if k >> n & 1 == 1 {
                let at = indices[n] as usize * 4;
                want[at..at + 4].copy_from_slice(&(element(&data, 4, n) as u32).to_le_bytes());
            }
        }
        assert_eq!(stored, want);
        assert_eq!(h.cpu.opmask[1], 0);
        // dq: dword indices, qword data
        let h = run(VPSCATTERDQ_Z, |h| {
            h.cpu.set_zmm(0, data);
            h.cpu.set_zmm(2, idx);
            h.cpu.opmask[1] = 0xff;
            h.write(OUT, &[0xee; 256]);
        });
        for n in 0..8 {
            assert_eq!(h.read_u64(OUT + indices[n] * 8), element(&data, 8, n));
        }
        // qd: qword indices, dword data from a 256-bit register
        let mut qidx = [0u128; 4];
        for n in 0..8 {
            set_element(&mut qidx, 8, n, indices[n]);
        }
        let h = run(VPSCATTERQD_Z, |h| {
            h.cpu.set_zmm(0, data);
            h.cpu.set_zmm(2, qidx);
            h.cpu.opmask[1] = 0xff;
            h.write(OUT, &[0xee; 256]);
        });
        for n in 0..8 {
            assert_eq!(
                h.read(OUT + indices[n] * 4, 4),
                (element(&data, 4, n) as u32).to_le_bytes()
            );
        }
        let h = run(VSCATTERQPD_Z, |h| {
            h.cpu.set_zmm(0, data);
            h.cpu.set_zmm(2, qidx);
            h.cpu.opmask[1] = 0b0110_0110;
            h.write(OUT, &[0xee; 256]);
        });
        for n in 0..8 {
            let written = 0b0110_0110u64 >> n & 1 == 1;
            let got = h.read_u64(OUT + indices[n] * 8);
            if written {
                assert_eq!(got, element(&data, 8, n));
            }
        }
        let h = run(VSCATTERDPS_Z, |h| {
            h.cpu.set_zmm(0, data);
            h.cpu.set_zmm(2, idx);
            h.cpu.opmask[1] = 0xffff;
            h.write(OUT, &[0xee; 256]);
        });
        assert_eq!(h.read(OUT, 4), (element(&data, 4, 0) as u32).to_le_bytes());
    }

    #[test]
    fn overlapping_scatter_elements_store_lowest_first() {
        let mut idx = [0u128; 4];
        for k in 0..16 {
            set_element(&mut idx, 4, k, 0);
        }
        let data = build(64, 4, |k| k as u64 + 100);
        let h = run(VPSCATTERDD_Z, |h| {
            h.cpu.set_zmm(0, data);
            h.cpu.set_zmm(2, idx);
            h.cpu.opmask[1] = 0xffff;
        });
        assert_eq!(
            h.read(OUT, 4),
            115u32.to_le_bytes(),
            "the highest element lands last"
        );
    }
}
