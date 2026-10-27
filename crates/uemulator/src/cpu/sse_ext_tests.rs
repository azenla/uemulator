use super::*;
use crate::testkit::*;

const MXCSR_RESET: u32 = 0x1f80;
/// Where a snippet's `[rsi]` points unless a test moves it: 16-byte aligned scratch RAM.
const DATA: u64 = SCRATCH;
const ARITH_FLAGS: u64 = ARITH;
const SINGLE_SNAN: u32 = 0x7f80_0001;
const SINGLE_QUIET_BIT: u32 = 0x0040_0000;
/// 2^-24, half an ulp of 1.0 in single precision.
const TWO_POW_MINUS_24_SINGLE: u32 = 0x3380_0000;
const DEFAULT_NAN_SINGLE: u32 = 0xffc0_0000;

fn code(asm: &str) -> &'static [u8] {
    SNIPPETS
        .iter()
        .find(|(key, _)| *key == asm)
        .unwrap_or_else(|| panic!("no assembled snippet for {asm:?}"))
        .1
}

fn fault(vector: u8) -> Stop {
    Stop::Fault {
        vector,
        error_code: None,
        software: false,
    }
}
fn ud() -> Stop {
    fault(6)
}
fn nm() -> Stop {
    fault(7)
}
fn gp0() -> Stop {
    Stop::Fault {
        vector: 13,
        error_code: Some(0),
        software: false,
    }
}
fn xm() -> Stop {
    fault(VECTOR_SIMD_FLOATING_POINT)
}

fn lanes8(v: [u8; 16]) -> u128 {
    u128::from_le_bytes(v)
}
fn lanes32(v: [u32; 4]) -> u128 {
    v.iter().rev().fold(0, |acc, &x| (acc << 32) | x as u128)
}
fn lanes64(v: [u64; 2]) -> u128 {
    ((v[1] as u128) << 64) | v[0] as u128
}
fn singles(v: [f32; 4]) -> u128 {
    lanes32(v.map(f32::to_bits))
}
fn doubles(v: [f64; 2]) -> u128 {
    lanes64(v.map(f64::to_bits))
}

/// One CPU reused across many runs, so table-driven tests do not pay for a fresh 32 MiB RAM each
/// time. `run` puts the machine back into the state a fresh harness has, with SSE enabled.
struct Rig {
    h: Harness,
    cr0: u64,
    cr4: u64,
}

impl Rig {
    fn new() -> Rig {
        let mut h = Harness::new(&[HLT]);
        h.cpu.cr4 |= CR4_OSFXSR;
        let (cr0, cr4) = (h.cpu.cr0, h.cpu.cr4);
        Rig { h, cr0, cr4 }
    }

    fn reset(&mut self) {
        let cpu = &mut self.h.cpu;
        cpu.cr0 = self.cr0;
        cpu.cr4 = self.cr4;
        cpu.mxcsr = MXCSR_RESET;
        cpu.halted = false;
        cpu.gpr[RSI] = DATA;
        cpu.gpr[RSP] = STACK_TOP;
        cpu.set_eflags_arith(0);
        cpu.decode_cache_generation = cpu.decode_cache_generation.wrapping_add(1);
    }

    /// Runs `asm` from the current register state with the harness defaults restored first.
    fn run(&mut self, asm: &str) -> Stop {
        self.reset();
        self.exec(asm)
    }

    /// Like `run` but keeps CR0, CR4 and MXCSR as the test left them.
    fn exec(&mut self, asm: &str) -> Stop {
        self.exec_bytes(code(asm))
    }

    fn exec_bytes(&mut self, bytes: &[u8]) -> Stop {
        self.h.write(CODE_BASE, bytes);
        self.h.cpu.rip = CODE_BASE;
        self.h.cpu.halted = false;
        self.h.cpu.decode_cache_generation = self.h.cpu.decode_cache_generation.wrapping_add(1);
        self.h.run(100)
    }

    fn run_ok(&mut self, asm: &str) -> &mut Self {
        let stop = self.run(asm);
        assert_eq!(stop, Stop::Halted, "{asm}: {}", self.h.cpu.dump_regs());
        self
    }

    /// xmm0 = `dst`, xmm1 = `src`; returns xmm0 after `asm`.
    fn op(&mut self, asm: &str, dst: u128, src: u128) -> u128 {
        self.h.cpu.xmm[0] = dst;
        self.h.cpu.xmm[1] = src;
        self.run_ok(asm);
        self.h.cpu.xmm[0]
    }

    /// xmm0 = `dst`, memory at `DATA` = `src` (16 bytes); returns xmm0 after `asm`.
    fn op_mem(&mut self, asm: &str, dst: u128, src: u128) -> u128 {
        self.h.cpu.xmm[0] = dst;
        self.h.write(DATA, &src.to_le_bytes());
        self.run_ok(asm);
        self.h.cpu.xmm[0]
    }

    fn gpr(&self, index: usize) -> u64 {
        self.h.cpu.gpr[index]
    }
    fn flags(&self) -> u64 {
        self.h.cpu.eflags_full() & ARITH_FLAGS
    }
    fn memory(&self, address: u64) -> u128 {
        u128::from_le_bytes(self.h.read(address, 16).try_into().unwrap())
    }
}

mod sse3 {
    use super::*;

    const DST: u128 = 0x1111_1111_2222_2222_3333_3333_4444_4444;
    const SRC: u128 = 0xaaaa_aaaa_bbbb_bbbb_cccc_cccc_dddd_dddd;

    #[test]
    fn movddup_duplicates_the_low_quadword() {
        let mut rig = Rig::new();
        let want = lanes64([SRC as u64, SRC as u64]);
        assert_eq!(rig.op("movddup xmm0, xmm1", DST, SRC), want);
        // The memory form reads only 8 bytes and needs no 16-byte alignment.
        rig.h.write(DATA + 8, &SRC.to_le_bytes());
        rig.h.cpu.gpr[RSI] = DATA + 8;
        rig.h.cpu.xmm[0] = DST;
        assert_eq!(rig.exec("movddup xmm0, [rsi]"), Stop::Halted);
        assert_eq!(rig.h.cpu.xmm[0], want);
    }

    #[test]
    fn movsldup_and_movshdup_duplicate_the_even_and_odd_dwords() {
        let mut rig = Rig::new();
        let src = lanes32([1, 2, 3, 4]);
        assert_eq!(
            rig.op("movsldup xmm0, xmm1", DST, src),
            lanes32([1, 1, 3, 3])
        );
        assert_eq!(
            rig.op("movshdup xmm0, xmm1", DST, src),
            lanes32([2, 2, 4, 4])
        );
        assert_eq!(
            rig.op_mem("movsldup xmm0, [rsi]", DST, src),
            lanes32([1, 1, 3, 3])
        );
        assert_eq!(
            rig.op_mem("movshdup xmm0, [rsi]", DST, src),
            lanes32([2, 2, 4, 4])
        );
    }

    #[test]
    fn memory_forms_need_16_byte_alignment_except_movddup_and_lddqu() {
        let mut rig = Rig::new();
        for asm in [
            "movsldup xmm0, [rsi]",
            "movshdup xmm0, [rsi]",
            "addsubps xmm0, [rsi]",
            "addsubpd xmm0, [rsi]",
            "haddps xmm0, [rsi]",
            "haddpd xmm0, [rsi]",
            "hsubps xmm0, [rsi]",
            "hsubpd xmm0, [rsi]",
        ] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 8;
            assert_eq!(rig.exec(asm), gp0(), "{asm}");
        }
    }

    #[test]
    fn lddqu_loads_unaligned_and_has_no_register_form() {
        let mut rig = Rig::new();
        let bytes: Vec<u8> = (0..32).collect();
        rig.h.write(DATA, &bytes);
        rig.h.cpu.xmm[0] = DST;
        rig.run_ok("lddqu xmm0, [rsi + 1]");
        assert_eq!(rig.h.cpu.xmm[0], lanes8(bytes[1..17].try_into().unwrap()));
        assert_eq!(rig.run(".byte 0xf2, 0x0f, 0xf0, 0xc1"), ud());
        // The register form is #UD even with CR0.TS set.
        rig.h.cpu.cr0 |= CR0_TS;
        assert_eq!(rig.exec(".byte 0xf2, 0x0f, 0xf0, 0xc1"), ud());
    }

    #[test]
    fn addsub_subtracts_in_even_lanes_and_adds_in_odd_ones() {
        let mut rig = Rig::new();
        let dst = singles([10.0, 20.0, 30.0, 40.0]);
        let src = singles([1.0, 2.0, 3.0, 4.0]);
        let want = singles([9.0, 22.0, 27.0, 44.0]);
        assert_eq!(rig.op("addsubps xmm0, xmm1", dst, src), want);
        assert_eq!(rig.op_mem("addsubps xmm0, [rsi]", dst, src), want);
        let dst = doubles([10.0, 20.0]);
        let src = doubles([1.0, 2.0]);
        let want = doubles([9.0, 22.0]);
        assert_eq!(rig.op("addsubpd xmm0, xmm1", dst, src), want);
        assert_eq!(rig.op_mem("addsubpd xmm0, [rsi]", dst, src), want);
    }

    #[test]
    fn horizontal_ops_combine_adjacent_lanes_of_dest_then_source() {
        let mut rig = Rig::new();
        let dst = singles([1.0, 2.0, 4.0, 8.0]);
        let src = singles([16.0, 32.0, 64.0, 128.0]);
        assert_eq!(
            rig.op("haddps xmm0, xmm1", dst, src),
            singles([3.0, 12.0, 48.0, 192.0])
        );
        assert_eq!(
            rig.op("hsubps xmm0, xmm1", dst, src),
            singles([-1.0, -4.0, -16.0, -64.0])
        );
        assert_eq!(
            rig.op_mem("haddps xmm0, [rsi]", dst, src),
            singles([3.0, 12.0, 48.0, 192.0])
        );
        let dst = doubles([5.0, 7.0]);
        let src = doubles([100.0, 30.0]);
        assert_eq!(
            rig.op("haddpd xmm0, xmm1", dst, src),
            doubles([12.0, 130.0])
        );
        assert_eq!(rig.op("hsubpd xmm0, xmm1", dst, src), doubles([-2.0, 70.0]));
        assert_eq!(
            rig.op_mem("hsubpd xmm0, [rsi]", dst, src),
            doubles([-2.0, 70.0])
        );
    }

    #[test]
    fn horizontal_ops_round_in_the_current_mode_and_flag_inexact_results() {
        let mut rig = Rig::new();
        // 1 + 2^-24 is a tie in single precision: round-to-nearest-even keeps 1 and round-up does
        // not; the precision flag is raised either way.
        let tie = f32::from_bits(TWO_POW_MINUS_24_SINGLE);
        let dst = singles([1.0, tie, 0.0, 0.0]);
        assert_eq!(rig.op("haddps xmm0, xmm1", dst, 0) as u32, 1.0f32.to_bits());
        assert_eq!(rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS, FLAG_PRECISION);
        rig.h.cpu.xmm[0] = dst;
        rig.h.cpu.xmm[1] = 0;
        rig.reset();
        rig.h.cpu.mxcsr |= ROUND_UP << MXCSR_RC_SHIFT;
        assert_eq!(rig.exec("haddps xmm0, xmm1"), Stop::Halted);
        assert_eq!(rig.h.cpu.xmm[0] as u32, 1.0f32.to_bits() + 1);
    }

    #[test]
    fn nan_and_invalid_operations_follow_the_arithmetic_rules() {
        let mut rig = Rig::new();
        let snan = f32::from_bits(SINGLE_SNAN);
        // inf - inf is invalid: the default NaN in that lane, and the flag is set.
        let dst = singles([f32::INFINITY, 0.0, 0.0, 0.0]);
        let src = singles([f32::INFINITY, 0.0, 0.0, 0.0]);
        let result = rig.op("addsubps xmm0, xmm1", dst, src);
        assert_eq!(result as u32, DEFAULT_NAN_SINGLE);
        assert_eq!(rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID);
        // A signalling NaN is quieted.
        let result = rig.op("haddps xmm0, xmm1", singles([snan, 1.0, 0.0, 0.0]), 0);
        assert_eq!(result as u32, SINGLE_SNAN | SINGLE_QUIET_BIT);
    }

    #[test]
    fn an_unmasked_exception_faults_and_leaves_the_destination_alone() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = singles([f32::INFINITY, 0.0, 0.0, 0.0]);
        rig.h.cpu.xmm[1] = singles([f32::INFINITY, 0.0, 0.0, 0.0]);
        let before = rig.h.cpu.xmm[0];
        rig.reset();
        rig.h.cpu.mxcsr &= !(1 << MXCSR_MASK_SHIFT);
        // Without CR4.OSXMMEXCPT the exception is #UD, with it #XM.
        assert_eq!(rig.exec("addsubps xmm0, xmm1"), ud());
        assert_eq!(rig.h.cpu.xmm[0], before);
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        assert_eq!(rig.exec("addsubps xmm0, xmm1"), xm());
        assert_eq!(rig.h.cpu.xmm[0], before);
    }

    #[test]
    fn prefix_and_enable_rules() {
        let mut rig = Rig::new();
        // The unprefixed and F3 forms of hadd/addsub and the 66 form of lddqu do not exist.
        for asm in [
            ".byte 0x0f, 0x7c, 0xc1",
            ".byte 0x0f, 0xd0, 0xc1",
            ".byte 0xf3, 0x0f, 0xd0, 0xc1",
            ".byte 0xf3, 0x0f, 0xf0, 0x06",
            ".byte 0x66, 0x0f, 0xf0, 0x06",
            ".byte 0x66, 0x0f, 0x12, 0xc1",
        ] {
            assert_eq!(rig.run(asm), ud(), "{asm}");
        }
        for asm in [
            "haddps xmm0, xmm1",
            "movddup xmm0, xmm1",
            "movsldup xmm0, xmm1",
            "lddqu xmm0, [rsi]",
        ] {
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            assert_eq!(rig.exec(asm), ud(), "{asm} without OSFXSR");
            rig.reset();
            rig.h.cpu.cr0 |= CR0_EM;
            assert_eq!(rig.exec(asm), ud(), "{asm} with CR0.EM");
            rig.reset();
            rig.h.cpu.cr0 |= CR0_TS;
            assert_eq!(rig.exec(asm), nm(), "{asm} with CR0.TS");
        }
    }

    #[test]
    fn jit_matches_the_interpreter() {
        for asm in [
            "haddps xmm0, xmm1",
            "addsubpd xmm0, [rsi]",
            "movshdup xmm0, xmm1",
            "lddqu xmm0, [rsi + 1]",
        ] {
            assert_jit_matches_interpreter(code(asm), |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                h.cpu.gpr[RSI] = DATA;
                h.cpu.xmm[0] = singles([1.5, -2.25, 1e30, 7.0]);
                h.cpu.xmm[1] = singles([0.5, 3.0, 1e30, -7.0]);
                h.write(DATA, &(0..32).collect::<Vec<u8>>());
            });
        }
    }
}

mod integer {
    use super::*;

    fn regs(rig: &mut Rig, asm: &str, rbx: u64, rax: u64) {
        rig.h.cpu.gpr[RBX] = rbx;
        rig.h.cpu.gpr[RAX] = rax;
        rig.reset();
        rig.h.cpu.set_eflags_arith(ARITH_FLAGS);
        let stop = rig.exec(asm);
        assert_eq!(stop, Stop::Halted, "{asm}");
    }

    #[test]
    fn popcnt_counts_bits_within_the_operand_size() {
        let mut rig = Rig::new();
        let value = 0xf0f0_0000_0000_ffff_u64;
        regs(&mut rig, "popcnt rax, rbx", value, u64::MAX);
        assert_eq!(rig.gpr(RAX), 24);
        regs(&mut rig, "popcnt eax, ebx", value, u64::MAX);
        assert_eq!(rig.gpr(RAX), 16, "32-bit form zero-extends");
        regs(&mut rig, "popcnt ax, bx", value, u64::MAX);
        assert_eq!(rig.gpr(RAX), 0xffff_ffff_ffff_0010, "16-bit form merges");
        regs(&mut rig, "popcnt rax, rbx", u64::MAX, 0);
        assert_eq!(rig.gpr(RAX), 64);
    }

    #[test]
    fn popcnt_flags_report_only_a_zero_source() {
        let mut rig = Rig::new();
        regs(&mut rig, "popcnt rax, rbx", 0, 5);
        assert_eq!(rig.gpr(RAX), 0);
        assert_eq!(rig.flags(), ZF, "CF OF SF AF PF cleared, ZF set");
        regs(&mut rig, "popcnt rax, rbx", 1 << 63, 5);
        assert_eq!(rig.flags(), 0, "everything cleared for a nonzero source");
        // A 16-bit popcnt only looks at the low word.
        regs(&mut rig, "popcnt ax, bx", 0x1_0000, 5);
        assert_eq!(rig.flags(), ZF);
    }

    #[test]
    fn popcnt_memory_form_reads_the_operand_size() {
        let mut rig = Rig::new();
        rig.h.write(DATA, &u64::MAX.to_le_bytes());
        rig.run_ok("popcnt rax, [rsi]");
        assert_eq!(rig.gpr(RAX), 64);
        rig.run_ok("popcnt eax, [rsi]");
        assert_eq!(rig.gpr(RAX), 32);
        rig.run_ok("popcnt ax, [rsi]");
        assert_eq!(rig.gpr(RAX) & 0xffff, 16);
    }

    #[test]
    fn popcnt_needs_the_f3_prefix() {
        let mut rig = Rig::new();
        assert_eq!(
            rig.run(".byte 0x0f, 0xb8, 0xc3"),
            ud(),
            "jmpe does not exist"
        );
        assert_eq!(rig.run(".byte 0xf2, 0x0f, 0xb8, 0xc3"), ud());
    }

    #[test]
    fn lzcnt_counts_leading_zeros_within_the_operand_size() {
        let mut rig = Rig::new();
        regs(&mut rig, "lzcnt rax, rbx", 0x100, u64::MAX);
        assert_eq!(rig.gpr(RAX), 55);
        regs(&mut rig, "lzcnt eax, ebx", 0x100, u64::MAX);
        assert_eq!(rig.gpr(RAX), 23, "32-bit form zero-extends");
        regs(&mut rig, "lzcnt ax, bx", 0x100, u64::MAX);
        assert_eq!(rig.gpr(RAX), 0xffff_ffff_ffff_0007, "16-bit form merges");
        regs(&mut rig, "lzcnt rax, rbx", 1, 0);
        assert_eq!(rig.gpr(RAX), 63);
        regs(&mut rig, "lzcnt eax, ebx", 0xffff_ffff_0000_0001, 0);
        assert_eq!(rig.gpr(RAX), 31, "bits above the operand size are ignored");
    }

    #[test]
    fn lzcnt_flags_and_zero_source() {
        let mut rig = Rig::new();
        regs(&mut rig, "lzcnt rax, rbx", 0, 9);
        assert_eq!(rig.gpr(RAX), 64, "unlike bsr, the destination is written");
        assert_eq!(rig.flags(), CF);
        regs(&mut rig, "lzcnt ax, bx", 0x1_0000, 9);
        assert_eq!(rig.gpr(RAX) & 0xffff, 16);
        assert_eq!(rig.flags(), CF);
        regs(&mut rig, "lzcnt rax, rbx", 1 << 63, 9);
        assert_eq!(rig.gpr(RAX), 0);
        assert_eq!(rig.flags(), ZF, "ZF reports a zero result");
        regs(&mut rig, "lzcnt rax, rbx", 1, 9);
        assert_eq!(rig.flags(), 0);
        rig.h.write(DATA, &0x80u64.to_le_bytes());
        rig.run_ok("lzcnt rax, [rsi]");
        assert_eq!(rig.gpr(RAX), 56);
    }

    const OPERAND: u64 = DATA;

    fn put128(rig: &mut Rig, address: u64, low: u64, high: u64) {
        rig.h.write(address, &lanes64([low, high]).to_le_bytes());
    }

    #[test]
    fn cmpxchg16b_stores_rcx_rbx_on_a_match() {
        let mut rig = Rig::new();
        put128(&mut rig, OPERAND, 0x1111, 0x2222);
        rig.h.cpu.gpr[RAX] = 0x1111;
        rig.h.cpu.gpr[RDX] = 0x2222;
        rig.h.cpu.gpr[RBX] = 0xaaaa;
        rig.h.cpu.gpr[RCX] = 0xbbbb;
        rig.run_ok("cmpxchg16b [rsi]");
        assert_eq!(rig.memory(OPERAND), lanes64([0xaaaa, 0xbbbb]));
        assert_eq!(rig.flags(), ZF);
        assert_eq!(rig.gpr(RAX), 0x1111, "registers are untouched on a match");
        assert_eq!(rig.gpr(RDX), 0x2222);
    }

    #[test]
    fn cmpxchg16b_loads_rdx_rax_on_a_mismatch_and_clears_zf() {
        let mut rig = Rig::new();
        for (rax, rdx) in [(0x1110, 0x2222), (0x1111, 0x2223)] {
            put128(&mut rig, OPERAND, 0x1111, 0x2222);
            rig.h.cpu.gpr[RAX] = rax;
            rig.h.cpu.gpr[RDX] = rdx;
            rig.h.cpu.gpr[RBX] = 0xaaaa;
            rig.h.cpu.gpr[RCX] = 0xbbbb;
            rig.reset();
            rig.h.cpu.set_eflags_arith(ZF | CF);
            assert_eq!(rig.exec("cmpxchg16b [rsi]"), Stop::Halted);
            assert_eq!(rig.memory(OPERAND), lanes64([0x1111, 0x2222]));
            assert_eq!(rig.gpr(RAX), 0x1111);
            assert_eq!(rig.gpr(RDX), 0x2222);
            assert_eq!(rig.flags(), CF, "only ZF changes");
        }
    }

    #[test]
    fn cmpxchg16b_misaligned_operand_is_general_protection() {
        let mut rig = Rig::new();
        for offset in [1, 4, 8, 15] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + offset;
            assert_eq!(rig.exec("cmpxchg16b [rsi]"), gp0(), "offset {offset}");
        }
        // A lock prefix changes nothing; the base address with offset 16 is aligned again.
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA;
        rig.h.cpu.gpr[RAX] = 1;
        assert_eq!(rig.exec("lock cmpxchg16b [rsi]"), Stop::Halted);
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 8;
        assert_eq!(rig.exec("cmpxchg16b [rsi + 8]"), Stop::Halted);
    }

    #[test]
    fn cmpxchg16b_register_form_is_invalid_and_cmpxchg8b_is_unchanged() {
        let mut rig = Rig::new();
        assert_eq!(rig.run(".byte 0x48, 0x0f, 0xc7, 0xce"), ud());
        put128(&mut rig, OPERAND, 0x1111_1111_2222_2222, 0);
        rig.h.cpu.gpr[RAX] = 0x2222_2222;
        rig.h.cpu.gpr[RDX] = 0x1111_1111;
        rig.h.cpu.gpr[RBX] = 5;
        rig.h.cpu.gpr[RCX] = 6;
        rig.run_ok("cmpxchg8b [rsi]");
        assert_eq!(rig.memory(OPERAND) as u64, 0x6_0000_0005);
        assert_eq!(rig.flags(), ZF);
    }

    #[test]
    fn jit_matches_the_interpreter() {
        for (asm, rbx) in [
            ("popcnt rax, rbx", 0x1234_5678_9abc_def0_u64),
            ("popcnt ax, bx", 0),
            ("lzcnt rax, rbx", 0x1234),
            ("lzcnt eax, ebx", 0),
        ] {
            assert_jit_matches_interpreter(code(asm), |h| h.cpu.gpr[RBX] = rbx);
        }
        for operand in [DATA, DATA + 8] {
            assert_jit_matches_interpreter(code("cmpxchg16b [rsi]"), |h| {
                h.cpu.gpr[RSI] = operand;
                h.cpu.gpr[RAX] = 0x0807_0605_0403_0201;
                h.cpu.gpr[RDX] = 0x100f_0e0d_0c0b_0a09;
                h.cpu.gpr[RBX] = 77;
                h.write(DATA, &(1..=16).collect::<Vec<u8>>());
            });
        }
    }
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn vector(&mut self) -> u128 {
        ((self.next() as u128) << 64) | self.next() as u128
    }
}

/// Edge-case registers followed by random ones: the operands every table-driven test runs over.
fn vectors() -> Vec<u128> {
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    let mut all = vec![
        0,
        u128::MAX,
        lanes8([0x80; 16]),
        lanes8([0x7f; 16]),
        lanes16([0x8000; 8]),
        lanes16([0x7fff, 0x8000, 0xffff, 1, 0, 2, 0x4000, 0xc000]),
        lanes32([0x8000_0000, 0x7fff_ffff, 0xffff_ffff, 1]),
        lanes8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        lanes8([
            15, 14, 13, 12, 11, 10, 9, 8, 0x87, 6, 0xff, 4, 3, 2, 1, 0x80,
        ]),
    ];
    for _ in 0..24 {
        all.push(rng.vector());
    }
    all
}

fn bytes_of(v: u128) -> [u8; 16] {
    v.to_le_bytes()
}
fn words_of(v: u128) -> [u16; 8] {
    std::array::from_fn(|k| (v >> (16 * k)) as u16)
}
fn dwords_of(v: u128) -> [u32; 4] {
    std::array::from_fn(|k| (v >> (32 * k)) as u32)
}
fn qwords_of(v: u128) -> [u64; 2] {
    [v as u64, (v >> 64) as u64]
}
fn lanes16(v: [u16; 8]) -> u128 {
    v.iter().rev().fold(0, |acc, &x| (acc << 16) | x as u128)
}

/// Runs `name xmm0, xmm1` and `name xmm0, [rsi]` over pairs of `vectors()` (a stride keeps the
/// pair count modest) and compares with `reference(dst, src)`.
#[track_caller]
fn check_binary(rig: &mut Rig, name: &str, reference: impl Fn(u128, u128) -> u128) {
    let register = format!("{name} xmm0, xmm1");
    let memory = format!("{name} xmm0, [rsi]");
    let all = vectors();
    for (index, &dst) in all.iter().enumerate() {
        for &src in all.iter().skip(index % 3).step_by(3) {
            let want = reference(dst, src);
            assert_eq!(
                rig.op(&register, dst, src),
                want,
                "{register} {dst:#x} {src:#x}"
            );
            assert_eq!(
                rig.op_mem(&memory, dst, src),
                want,
                "{memory} {dst:#x} {src:#x}"
            );
        }
    }
}

mod ssse3 {
    use super::*;

    type UnaryReference = fn(u128) -> u128;

    #[test]
    fn pshufb_selects_bytes_of_the_destination_or_zero() {
        let mut rig = Rig::new();
        check_binary(&mut rig, "pshufb", |dst, selector| {
            let (data, select) = (bytes_of(dst), bytes_of(selector));
            lanes8(std::array::from_fn(|k| {
                if select[k] & 0x80 != 0 {
                    0
                } else {
                    data[(select[k] & 0xf) as usize]
                }
            }))
        });
        // A concrete case: reverse the bytes, zero the odd ones.
        let data = lanes8(std::array::from_fn(|k| 0x10 + k as u8));
        let selector = lanes8(std::array::from_fn(|k| {
            if k % 2 == 1 { 0x80 } else { 15 - k as u8 }
        }));
        let want = lanes8(std::array::from_fn(|k| {
            if k % 2 == 1 { 0 } else { 0x10 + (15 - k) as u8 }
        }));
        assert_eq!(rig.op("pshufb xmm0, xmm1", data, selector), want);
    }

    fn horizontal_words(dst: u128, src: u128, f: impl Fn(i16, i16) -> i16) -> u128 {
        let (d, s) = (words_of(dst), words_of(src));
        lanes16(std::array::from_fn(|k| {
            let (from, pair) = if k < 4 { (d, k) } else { (s, k - 4) };
            f(from[2 * pair] as i16, from[2 * pair + 1] as i16) as u16
        }))
    }

    fn horizontal_dwords(dst: u128, src: u128, f: impl Fn(i32, i32) -> i32) -> u128 {
        let (d, s) = (dwords_of(dst), dwords_of(src));
        lanes32(std::array::from_fn(|k| {
            let (from, pair) = if k < 2 { (d, k) } else { (s, k - 2) };
            f(from[2 * pair] as i32, from[2 * pair + 1] as i32) as u32
        }))
    }

    #[test]
    fn horizontal_add_and_subtract_pair_adjacent_lanes() {
        let mut rig = Rig::new();
        check_binary(&mut rig, "phaddw", |a, b| {
            horizontal_words(a, b, i16::wrapping_add)
        });
        check_binary(&mut rig, "phsubw", |a, b| {
            horizontal_words(a, b, i16::wrapping_sub)
        });
        check_binary(&mut rig, "phaddsw", |a, b| {
            horizontal_words(a, b, |x, y| {
                (x as i32 + y as i32).clamp(-32768, 32767) as i16
            })
        });
        check_binary(&mut rig, "phsubsw", |a, b| {
            horizontal_words(a, b, |x, y| {
                (x as i32 - y as i32).clamp(-32768, 32767) as i16
            })
        });
        check_binary(&mut rig, "phaddd", |a, b| {
            horizontal_dwords(a, b, i32::wrapping_add)
        });
        check_binary(&mut rig, "phsubd", |a, b| {
            horizontal_dwords(a, b, i32::wrapping_sub)
        });
    }

    #[test]
    fn horizontal_ops_put_destination_pairs_low_and_the_lower_lane_first() {
        let mut rig = Rig::new();
        let dst = lanes16([1, 2, 3, 4, 5, 6, 7, 8]);
        let src = lanes16([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(
            rig.op("phaddw xmm0, xmm1", dst, src),
            lanes16([3, 7, 11, 15, 30, 70, 110, 150])
        );
        let minus = |v: i16| v as u16;
        assert_eq!(
            rig.op("phsubw xmm0, xmm1", dst, src),
            lanes16([
                minus(-1),
                minus(-1),
                minus(-1),
                minus(-1),
                minus(-10),
                minus(-10),
                minus(-10),
                minus(-10)
            ])
        );
    }

    #[test]
    fn saturating_horizontal_ops_clamp_at_the_signed_limits() {
        let mut rig = Rig::new();
        let dst = lanes16([0x7fff, 1, 0x8000, 0xffff, 0x7fff, 0x7fff, 0x8000, 0x8000]);
        assert_eq!(
            rig.op("phaddsw xmm0, xmm1", dst, 0),
            lanes16([0x7fff, 0x8000, 0x7fff, 0x8000, 0, 0, 0, 0])
        );
        let dst = lanes16([0x7fff, 0xffff, 0x8000, 1, 0, 0, 0, 0]);
        assert_eq!(
            rig.op("phsubsw xmm0, xmm1", dst, 0),
            lanes16([0x7fff, 0x8000, 0, 0, 0, 0, 0, 0])
        );
    }

    #[test]
    fn pmaddubsw_multiplies_unsigned_by_signed_and_saturates_the_pair_sum() {
        let mut rig = Rig::new();
        check_binary(&mut rig, "pmaddubsw", |dst, src| {
            let (u, s) = (bytes_of(dst), bytes_of(src));
            lanes16(std::array::from_fn(|k| {
                let sum = u[2 * k] as i32 * (s[2 * k] as i8) as i32
                    + u[2 * k + 1] as i32 * (s[2 * k + 1] as i8) as i32;
                sum.clamp(-32768, 32767) as u16
            }))
        });
        // 255*127 twice saturates high; 255*-128 twice saturates low; the rest is exact.
        let signed = lanes8([
            127, 127, 0x80, 0x80, 1, 2, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        assert_eq!(
            rig.op("pmaddubsw xmm0, xmm1", lanes8([255; 16]), signed),
            lanes16([0x7fff, 0x8000, 255 * 3, (-510i16) as u16, 0, 0, 0, 0])
        );
    }

    #[test]
    fn pmulhrsw_rounds_the_scaled_product() {
        let mut rig = Rig::new();
        check_binary(&mut rig, "pmulhrsw", |dst, src| {
            let (a, b) = (words_of(dst), words_of(src));
            lanes16(std::array::from_fn(|k| {
                let product = a[k] as i16 as i32 * b[k] as i16 as i32;
                (((product >> 14) + 1) >> 1) as u16
            }))
        });
        // 0.5 * 0.5 = 0.25 in Q15; -1 * -1 overflows back to -1.
        assert_eq!(
            rig.op(
                "pmulhrsw xmm0, xmm1",
                lanes16([0x4000; 8]),
                lanes16([0x4000; 8])
            ),
            lanes16([0x2000; 8])
        );
        assert_eq!(
            rig.op(
                "pmulhrsw xmm0, xmm1",
                lanes16([0x8000; 8]),
                lanes16([0x8000; 8])
            ),
            lanes16([0x8000; 8])
        );
    }

    #[test]
    fn psign_negates_zeroes_or_keeps_by_the_source_sign() {
        let mut rig = Rig::new();
        check_binary(&mut rig, "psignb", |dst, src| {
            let (a, b) = (bytes_of(dst), bytes_of(src));
            lanes8(std::array::from_fn(|k| match (b[k] as i8).signum() {
                -1 => (a[k] as i8).wrapping_neg() as u8,
                0 => 0,
                _ => a[k],
            }))
        });
        check_binary(&mut rig, "psignw", |dst, src| {
            let (a, b) = (words_of(dst), words_of(src));
            lanes16(std::array::from_fn(|k| match (b[k] as i16).signum() {
                -1 => (a[k] as i16).wrapping_neg() as u16,
                0 => 0,
                _ => a[k],
            }))
        });
        check_binary(&mut rig, "psignd", |dst, src| {
            let (a, b) = (dwords_of(dst), dwords_of(src));
            lanes32(std::array::from_fn(|k| match (b[k] as i32).signum() {
                -1 => (a[k] as i32).wrapping_neg() as u32,
                0 => 0,
                _ => a[k],
            }))
        });
        // The most negative value negates to itself.
        assert_eq!(
            rig.op("psignb xmm0, xmm1", lanes8([0x80; 16]), lanes8([0xff; 16])),
            lanes8([0x80; 16])
        );
    }

    #[test]
    fn pabs_takes_the_absolute_value_of_the_source() {
        let mut rig = Rig::new();
        let references: [(&str, UnaryReference); 3] = [
            ("pabsb", |v| {
                lanes8(bytes_of(v).map(|x| (x as i8).unsigned_abs()))
            }),
            ("pabsw", |v| {
                lanes16(words_of(v).map(|x| (x as i16).unsigned_abs()))
            }),
            ("pabsd", |v| {
                lanes32(dwords_of(v).map(|x| (x as i32).unsigned_abs()))
            }),
        ];
        for (name, reference) in references {
            for &src in &vectors() {
                let junk = !src;
                assert_eq!(
                    rig.op(&format!("{name} xmm0, xmm1"), junk, src),
                    reference(src),
                    "{name} {src:#x}: the old destination is ignored"
                );
                assert_eq!(
                    rig.op_mem(&format!("{name} xmm0, [rsi]"), junk, src),
                    reference(src),
                    "{name} memory {src:#x}"
                );
            }
        }
        // abs of the most negative value is its own unsigned magnitude.
        assert_eq!(
            rig.op("pabsb xmm0, xmm1", 0, lanes8([0x80; 16])),
            lanes8([0x80; 16])
        );
        assert_eq!(
            rig.op("pabsd xmm0, xmm1", 0, lanes32([0x8000_0000; 4])),
            lanes32([0x8000_0000; 4])
        );
    }

    #[test]
    fn palignr_shifts_the_concatenation_right_by_bytes() {
        let mut rig = Rig::new();
        let all = vectors();
        for count in [0u8, 1, 4, 15, 16, 17, 31, 32, 255] {
            let asm = format!("palignr xmm0, xmm1, {count}");
            for (index, &dst) in all.iter().enumerate() {
                let src = all[(index * 7 + 3) % all.len()];
                // dst is the high half of the 32-byte value, src the low half; the bytes beyond
                // the pair read as zero.
                let mut joined = [0u8; 48];
                joined[..16].copy_from_slice(&bytes_of(src));
                joined[16..32].copy_from_slice(&bytes_of(dst));
                let want = lanes8(std::array::from_fn(|k| {
                    joined[(k + count as usize).min(47)]
                }));
                assert_eq!(rig.op(&asm, dst, src), want, "{asm} {dst:#x} {src:#x}");
            }
        }
        let (dst, src) = (lanes8([0xdd; 16]), lanes8([0xcc; 16]));
        let mut want = [0xcc; 16];
        want[13..].fill(0xdd);
        assert_eq!(rig.op_mem("palignr xmm0, [rsi], 3", dst, src), lanes8(want));
    }

    #[test]
    fn memory_operands_must_be_aligned() {
        let mut rig = Rig::new();
        for asm in [
            "pshufb xmm0, [rsi]",
            "phaddd xmm0, [rsi]",
            "pmaddubsw xmm0, [rsi]",
            "pabsw xmm0, [rsi]",
            "palignr xmm0, [rsi], 3",
        ] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 1;
            assert_eq!(rig.exec(asm), gp0(), "{asm}");
        }
    }

    #[test]
    fn mmx_register_forms_and_foreign_prefixes_are_invalid() {
        // The MMX forms (no prefix) are deliberately not implemented: see the module comment of
        // sse_ext.rs. They are #UD, as are the F2/F3-prefixed encodings.
        let mut rig = Rig::new();
        for asm in [
            ".byte 0x0f, 0x38, 0x00, 0xc1",
            ".byte 0x0f, 0x38, 0x1c, 0xc1",
            ".byte 0x0f, 0x3a, 0x0f, 0xc1, 0x04",
            ".byte 0xf2, 0x0f, 0x38, 0x00, 0xc1",
            ".byte 0xf3, 0x0f, 0x38, 0x00, 0xc1",
        ] {
            assert_eq!(rig.run(asm), ud(), "{asm}");
        }
    }

    #[test]
    fn enable_rules_apply() {
        let mut rig = Rig::new();
        for asm in [
            "pshufb xmm0, xmm1",
            "palignr xmm0, xmm1, 4",
            "pabsb xmm0, [rsi]",
        ] {
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            assert_eq!(rig.exec(asm), ud(), "{asm} without OSFXSR");
            rig.reset();
            rig.h.cpu.cr0 |= CR0_EM;
            assert_eq!(rig.exec(asm), ud(), "{asm} with CR0.EM");
            rig.reset();
            rig.h.cpu.cr0 |= CR0_TS;
            assert_eq!(rig.exec(asm), nm(), "{asm} with CR0.TS");
        }
    }

    #[test]
    fn jit_matches_the_interpreter() {
        for asm in [
            "pshufb xmm0, xmm1",
            "phaddsw xmm0, [rsi]",
            "pmaddubsw xmm0, xmm1",
            "psignw xmm0, xmm1",
            "pabsd xmm0, xmm1",
            "palignr xmm0, xmm1, 4",
        ] {
            assert_jit_matches_interpreter(code(asm), |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                h.cpu.gpr[RSI] = DATA;
                h.cpu.xmm[0] = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
                h.cpu.xmm[1] = 0x8070_6050_4030_2010_00ff_7f80_0102_0304;
                h.write(DATA, &h.cpu.xmm[1].to_le_bytes());
            });
        }
    }
}

#[test]
fn unassigned_extension_opcodes_are_invalid() {
    let mut rig = Rig::new();
    for asm in [
        ".byte 0x66, 0x0f, 0x38, 0xff, 0xc1",
        ".byte 0x66, 0x0f, 0x3a, 0xff, 0xc1, 0x00",
    ] {
        assert_eq!(rig.run(asm), ud(), "{asm}");
    }
}

mod sse41 {
    use super::*;

    /// Runs a snippet whose last byte before the `hlt` is imm8, with that byte replaced.
    fn exec_imm(rig: &mut Rig, asm: &str, imm: u8) -> Stop {
        let mut bytes = code(asm).to_vec();
        let last = bytes.len() - 2;
        bytes[last] = imm;
        rig.exec_bytes(&bytes)
    }

    /// xmm0 = `dst`, xmm1 = `src`; returns xmm0 after `asm` with immediate `imm`.
    fn op_imm(rig: &mut Rig, asm: &str, imm: u8, dst: u128, src: u128) -> u128 {
        rig.h.cpu.xmm[0] = dst;
        rig.h.cpu.xmm[1] = src;
        rig.reset();
        let stop = exec_imm(rig, asm, imm);
        assert_eq!(
            stop,
            Stop::Halted,
            "{asm} {imm:#x}: {}",
            rig.h.cpu.dump_regs()
        );
        rig.h.cpu.xmm[0]
    }

    /// Like `op_imm` with the source in memory at `DATA`.
    fn op_mem_imm(rig: &mut Rig, asm: &str, imm: u8, dst: u128, src: u128) -> u128 {
        rig.h.cpu.xmm[0] = dst;
        rig.h.write(DATA, &src.to_le_bytes());
        rig.reset();
        let stop = exec_imm(rig, asm, imm);
        assert_eq!(
            stop,
            Stop::Halted,
            "{asm} {imm:#x}: {}",
            rig.h.cpu.dump_regs()
        );
        rig.h.cpu.xmm[0]
    }

    fn lane_values<const N: usize>(v: u128, lane_bits: usize) -> [u128; N] {
        std::array::from_fn(|k| (v >> (k * lane_bits)) & ((1u128 << lane_bits) - 1))
    }

    fn from_lane_values(values: &[u128], lane_bits: usize) -> u128 {
        values
            .iter()
            .enumerate()
            .fold(0, |acc, (k, &x)| acc | (x << (k * lane_bits)))
    }

    #[test]
    fn variable_blends_select_by_the_mask_lane_sign_bit() {
        let mut rig = Rig::new();
        for (name, lane_bits) in [("pblendvb", 8), ("blendvps", 32), ("blendvpd", 64)] {
            let register = format!("{name} xmm1, xmm2, xmm0");
            let memory = format!("{name} xmm1, [rsi], xmm0");
            let all = vectors();
            for (index, &mask) in all.iter().enumerate() {
                let dst = all[(index + 5) % all.len()];
                let src = all[(index + 11) % all.len()];
                let lanes = 128 / lane_bits;
                let want = from_lane_values(
                    &(0..lanes)
                        .map(|k| {
                            let pick =
                                |v: u128| (v >> (k * lane_bits)) & ((1u128 << lane_bits) - 1);
                            if pick(mask) >> (lane_bits - 1) & 1 == 1 {
                                pick(src)
                            } else {
                                pick(dst)
                            }
                        })
                        .collect::<Vec<_>>(),
                    lane_bits,
                );
                rig.h.cpu.xmm[0] = mask;
                rig.h.cpu.xmm[1] = dst;
                rig.h.cpu.xmm[2] = src;
                rig.run_ok(&register);
                assert_eq!(rig.h.cpu.xmm[1], want, "{register} {mask:#x}");
                assert_eq!(rig.h.cpu.xmm[0], mask, "the mask register is not written");
                rig.h.cpu.xmm[0] = mask;
                rig.h.cpu.xmm[1] = dst;
                rig.h.write(DATA, &src.to_le_bytes());
                rig.run_ok(&memory);
                assert_eq!(rig.h.cpu.xmm[1], want, "{memory} {mask:#x}");
            }
        }
    }

    #[test]
    fn immediate_blends_pick_lanes_by_the_selector_bits() {
        let mut rig = Rig::new();
        let dst = lanes8(std::array::from_fn(|k| k as u8));
        let src = lanes8(std::array::from_fn(|k| 0x80 + k as u8));
        for (name, lane_bits) in [("pblendw", 16), ("blendps", 32), ("blendpd", 64)] {
            let register = format!("{name} xmm0, xmm1, 0");
            let memory = format!("{name} xmm0, [rsi], 0");
            let lanes = 128 / lane_bits;
            for imm in [0u8, 1, 2, 0x55, 0xaa, 0x0f, 0xf0, 0xff, 0xfe] {
                let d = lane_values::<8>(dst, lane_bits);
                let s = lane_values::<8>(src, lane_bits);
                let want = from_lane_values(
                    &(0..lanes)
                        .map(|k| if imm >> k & 1 == 1 { s[k] } else { d[k] })
                        .collect::<Vec<_>>(),
                    lane_bits,
                );
                assert_eq!(
                    op_imm(&mut rig, &register, imm, dst, src),
                    want,
                    "{name} {imm:#x}"
                );
                assert_eq!(
                    op_mem_imm(&mut rig, &memory, imm, dst, src),
                    want,
                    "{name} memory {imm:#x}"
                );
            }
        }
    }

    #[test]
    fn ptest_reports_and_and_andn_zero_and_clears_the_other_flags() {
        let mut rig = Rig::new();
        let cases = [
            (0u128, 0u128, ZF | CF),
            (0xff00, 0x00ff, ZF),
            (0xf0f0, 0xf0f0, CF),
            (0x00f0, 0xf0f0, 0),
            (0xf0f0, 0x00f0, CF),
            (u128::MAX, 1 << 127, CF),
            (1 << 127, 1 << 127, CF),
            (0, 1, ZF),
        ];
        for (dst, src, want) in cases {
            for asm in ["ptest xmm0, xmm1", "ptest xmm0, [rsi]"] {
                rig.h.cpu.xmm[0] = dst;
                rig.h.cpu.xmm[1] = src;
                rig.h.write(DATA, &src.to_le_bytes());
                rig.reset();
                rig.h.cpu.set_eflags_arith(ARITH_FLAGS & !(ZF | CF));
                assert_eq!(rig.exec(asm), Stop::Halted);
                assert_eq!(rig.flags(), want, "{asm} {dst:#x} {src:#x}");
                assert_eq!(rig.h.cpu.xmm[0], dst, "ptest writes no register");
            }
        }
        // Every status flag is preset: OF, SF, AF and PF must be cleared, ZF and CF rewritten.
        rig.h.cpu.xmm[0] = 0x00f0;
        rig.h.cpu.xmm[1] = 0xf0f0;
        rig.reset();
        rig.h.cpu.set_eflags_arith(ARITH_FLAGS);
        rig.exec("ptest xmm0, xmm1");
        assert_eq!(rig.flags(), 0);
    }

    #[test]
    fn pmovsx_and_pmovzx_extend_the_low_lanes() {
        let mut rig = Rig::new();
        let source = lanes8([
            0x80, 0x7f, 0xff, 0x01, 0xfe, 0x02, 0x00, 0x81, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 1,
            2,
        ]);
        let forms: [(&str, u32, u32); 6] = [
            ("bw", 8, 16),
            ("bd", 8, 32),
            ("bq", 8, 64),
            ("wd", 16, 32),
            ("wq", 16, 64),
            ("dq", 32, 64),
        ];
        for (suffix, from_bits, to_bits) in forms {
            for (kind, signed) in [("sx", true), ("zx", false)] {
                let lanes = 128 / to_bits as usize;
                let from = from_bits as usize;
                let want = from_lane_values(
                    &(0..lanes)
                        .map(|k| {
                            let raw = lane_values::<16>(source, from)[k];
                            let extended = if signed && raw >> (from - 1) & 1 == 1 {
                                raw | (u128::MAX << from)
                            } else {
                                raw
                            };
                            extended & ((1u128 << to_bits) - 1)
                        })
                        .collect::<Vec<_>>(),
                    to_bits as usize,
                );
                let register = format!("pmov{kind}{suffix} xmm0, xmm1");
                let memory = format!("pmov{kind}{suffix} xmm0, [rsi]");
                assert_eq!(rig.op(&register, !0, source), want, "{register}");
                assert_eq!(rig.op_mem(&memory, !0, source), want, "{memory}");
            }
        }
    }

    #[test]
    fn pmovx_memory_forms_need_no_alignment() {
        let mut rig = Rig::new();
        rig.h
            .write(DATA + 1, &[0x80, 0x7f, 1, 0, 2, 0, 3, 0, 0xff, 0xff]);
        for (asm, want) in [
            ("pmovsxbq xmm0, [rsi]", lanes64([u64::MAX - 0x7f, 0x7f])),
            ("pmovzxbd xmm0, [rsi]", lanes32([0x80, 0x7f, 1, 0])),
            ("pmovsxwd xmm0, [rsi]", lanes32([0x7f80, 1, 2, 3])),
        ] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 1;
            assert_eq!(rig.exec(asm), Stop::Halted, "{asm}");
            assert_eq!(rig.h.cpu.xmm[0], want, "{asm}");
        }
    }

    #[test]
    fn dword_and_word_arithmetic_matches_per_lane_references() {
        let mut rig = Rig::new();
        check_binary(&mut rig, "pmuldq", |a, b| {
            let (x, y) = (dwords_of(a), dwords_of(b));
            lanes64(std::array::from_fn(|k| {
                ((x[2 * k] as i32 as i64) * (y[2 * k] as i32 as i64)) as u64
            }))
        });
        check_binary(&mut rig, "pmulld", |a, b| {
            let (x, y) = (dwords_of(a), dwords_of(b));
            lanes32(std::array::from_fn(|k| x[k].wrapping_mul(y[k])))
        });
        check_binary(&mut rig, "pminsb", |a, b| {
            let (x, y) = (bytes_of(a), bytes_of(b));
            lanes8(std::array::from_fn(|k| (x[k] as i8).min(y[k] as i8) as u8))
        });
        check_binary(&mut rig, "pmaxsb", |a, b| {
            let (x, y) = (bytes_of(a), bytes_of(b));
            lanes8(std::array::from_fn(|k| (x[k] as i8).max(y[k] as i8) as u8))
        });
        check_binary(&mut rig, "pminsd", |a, b| {
            let (x, y) = (dwords_of(a), dwords_of(b));
            lanes32(std::array::from_fn(|k| {
                (x[k] as i32).min(y[k] as i32) as u32
            }))
        });
        check_binary(&mut rig, "pmaxsd", |a, b| {
            let (x, y) = (dwords_of(a), dwords_of(b));
            lanes32(std::array::from_fn(|k| {
                (x[k] as i32).max(y[k] as i32) as u32
            }))
        });
        check_binary(&mut rig, "pminuw", |a, b| {
            let (x, y) = (words_of(a), words_of(b));
            lanes16(std::array::from_fn(|k| x[k].min(y[k])))
        });
        check_binary(&mut rig, "pmaxuw", |a, b| {
            let (x, y) = (words_of(a), words_of(b));
            lanes16(std::array::from_fn(|k| x[k].max(y[k])))
        });
        check_binary(&mut rig, "pminud", |a, b| {
            let (x, y) = (dwords_of(a), dwords_of(b));
            lanes32(std::array::from_fn(|k| x[k].min(y[k])))
        });
        check_binary(&mut rig, "pmaxud", |a, b| {
            let (x, y) = (dwords_of(a), dwords_of(b));
            lanes32(std::array::from_fn(|k| x[k].max(y[k])))
        });
        check_binary(&mut rig, "pcmpeqq", |a, b| {
            let (x, y) = (qwords_of(a), qwords_of(b));
            lanes64(std::array::from_fn(
                |k| if x[k] == y[k] { u64::MAX } else { 0 },
            ))
        });
        check_binary(&mut rig, "packusdw", |a, b| {
            let lanes: Vec<i32> = dwords_of(a)
                .into_iter()
                .chain(dwords_of(b))
                .map(|x| x as i32)
                .collect();
            lanes16(std::array::from_fn(|k| lanes[k].clamp(0, 0xffff) as u16))
        });
    }

    #[test]
    fn pmuldq_uses_the_even_dwords_signed() {
        let mut rig = Rig::new();
        let a = lanes32([0xffff_ffff, 99, 3, 99]);
        let b = lanes32([2, 99, 0x8000_0000, 99]);
        assert_eq!(
            rig.op("pmuldq xmm0, xmm1", a, b),
            lanes64([(-2i64) as u64, (3i64 * i32::MIN as i64) as u64])
        );
    }

    #[test]
    fn phminposuw_finds_the_first_minimum_word() {
        let mut rig = Rig::new();
        for &src in &vectors() {
            let words = words_of(src);
            let (index, minimum) = words
                .iter()
                .enumerate()
                .fold(
                    (0, words[0]),
                    |best, (k, &w)| if w < best.1 { (k, w) } else { best },
                );
            let want = minimum as u128 | ((index as u128) << 16);
            assert_eq!(rig.op("phminposuw xmm0, xmm1", !0, src), want, "{src:#x}");
            assert_eq!(rig.op_mem("phminposuw xmm0, [rsi]", !0, src), want);
        }
        // ties pick the lowest index
        assert_eq!(
            rig.op(
                "phminposuw xmm0, xmm1",
                0,
                lanes16([5, 3, 9, 3, 3, 8, 7, 6])
            ),
            3 | (1 << 16)
        );
    }

    #[test]
    fn mpsadbw_sums_absolute_differences_of_sliding_windows() {
        let mut rig = Rig::new();
        let all = vectors();
        for imm in 0..8u8 {
            for (index, &dst) in all.iter().enumerate() {
                let src = all[(index * 5 + 2) % all.len()];
                let (d, s) = (bytes_of(dst), bytes_of(src));
                let dst_start = ((imm >> 2) & 1) as usize * 4;
                let src_start = (imm & 3) as usize * 4;
                let want = lanes16(std::array::from_fn(|window| {
                    (0..4)
                        .map(|k| {
                            (d[dst_start + window + k] as i32 - s[src_start + k] as i32)
                                .unsigned_abs()
                        })
                        .sum::<u32>() as u16
                }));
                assert_eq!(
                    op_imm(&mut rig, "mpsadbw xmm0, xmm1, 0", imm, dst, src),
                    want,
                    "imm {imm} {dst:#x} {src:#x}"
                );
            }
        }
        // The memory form and the ignored upper immediate bits.
        let (dst, src) = (all[3], all[7]);
        assert_eq!(
            op_mem_imm(&mut rig, "mpsadbw xmm0, [rsi], 0", 0xfd, dst, src),
            op_imm(&mut rig, "mpsadbw xmm0, xmm1, 0", 0x05, dst, src)
        );
    }

    const EXTRACT_SOURCE: u128 = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100;

    #[test]
    fn pextr_stores_the_selected_lane_zero_extended() {
        let mut rig = Rig::new();
        let source = lanes8([
            0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19, 0x2a, 0x3b, 0x4c, 0x5d,
            0x6e, 0x7f,
        ]);
        let bytes = bytes_of(source);
        for imm in 0..=0x1fu8 {
            rig.h.cpu.xmm[1] = source;
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.reset();
            assert_eq!(exec_imm(&mut rig, "pextrb eax, xmm1, 0", imm), Stop::Halted);
            assert_eq!(
                rig.gpr(RAX),
                bytes[(imm & 15) as usize] as u64,
                "pextrb {imm}"
            );
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.reset();
            assert_eq!(
                exec_imm(&mut rig, ".byte 0x66, 0x0f, 0x3a, 0x15, 0xc8, 0x00", imm),
                Stop::Halted
            );
            let word = u16::from_le_bytes([
                bytes[2 * (imm & 7) as usize],
                bytes[2 * (imm & 7) as usize + 1],
            ]);
            assert_eq!(rig.gpr(RAX), word as u64, "pextrw {imm}");
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.reset();
            assert_eq!(exec_imm(&mut rig, "pextrd eax, xmm1, 0", imm), Stop::Halted);
            assert_eq!(
                rig.gpr(RAX),
                dwords_of(source)[(imm & 3) as usize] as u64,
                "pextrd {imm}"
            );
            rig.h.cpu.gpr[RAX] = 0;
            rig.reset();
            assert_eq!(exec_imm(&mut rig, "pextrq rax, xmm1, 0", imm), Stop::Halted);
            assert_eq!(
                rig.gpr(RAX),
                qwords_of(source)[(imm & 1) as usize],
                "pextrq {imm}"
            );
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.reset();
            assert_eq!(
                exec_imm(&mut rig, "extractps eax, xmm1, 0", imm),
                Stop::Halted
            );
            assert_eq!(
                rig.gpr(RAX),
                dwords_of(source)[(imm & 3) as usize] as u64,
                "extractps {imm}"
            );
        }
        // Extended registers on both sides.
        rig.h.cpu.xmm[9] = EXTRACT_SOURCE;
        rig.reset();
        assert_eq!(exec_imm(&mut rig, "pextrq r9, xmm9, 0", 1), Stop::Halted);
        assert_eq!(rig.gpr(R9), 0x0f0e_0d0c_0b0a_0908);
        rig.reset();
        assert_eq!(exec_imm(&mut rig, "pextrb r9d, xmm9, 0", 5), Stop::Halted);
        assert_eq!(rig.gpr(R9), 5);
    }

    #[test]
    fn pextr_memory_forms_write_only_their_lane_size() {
        let mut rig = Rig::new();
        for (asm, size, imm) in [
            ("pextrb [rsi], xmm1, 0", 1usize, 3u8),
            ("pextrw [rsi], xmm1, 0", 2, 5),
            ("pextrd [rsi], xmm1, 0", 4, 2),
            ("pextrq [rsi], xmm1, 0", 8, 1),
            ("extractps [rsi], xmm1, 0", 4, 3),
        ] {
            rig.h.cpu.xmm[1] = EXTRACT_SOURCE;
            rig.h.write(DATA, &[0xaa; 16]);
            rig.reset();
            // Misaligned on purpose: pextr stores need no alignment.
            rig.h.cpu.gpr[RSI] = DATA + 1;
            assert_eq!(exec_imm(&mut rig, asm, imm), Stop::Halted, "{asm}");
            let written = rig.h.read(DATA + 1, 16);
            let lane_start = size * imm as usize;
            let want = &EXTRACT_SOURCE.to_le_bytes()[lane_start..lane_start + size];
            assert_eq!(&written[..size], want, "{asm}");
            assert!(
                written[size..15].iter().all(|&b| b == 0xaa),
                "{asm} wrote too much"
            );
            assert_eq!(rig.h.read(DATA, 1), [0xaa]);
        }
    }

    #[test]
    fn pinsr_replaces_the_selected_lane() {
        let mut rig = Rig::new();
        let dst = EXTRACT_SOURCE;
        for imm in 0..=0x1fu8 {
            rig.h.cpu.xmm[0] = dst;
            rig.h.cpu.gpr[RAX] = 0xffff_ffff_ffff_ffa5;
            rig.reset();
            assert_eq!(exec_imm(&mut rig, "pinsrb xmm0, eax, 0", imm), Stop::Halted);
            let mut want = bytes_of(dst);
            want[(imm & 15) as usize] = 0xa5;
            assert_eq!(rig.h.cpu.xmm[0], lanes8(want), "pinsrb {imm}");
            rig.h.cpu.xmm[0] = dst;
            rig.h.cpu.gpr[RAX] = 0xdead_beef_1234_5678;
            rig.reset();
            assert_eq!(exec_imm(&mut rig, "pinsrd xmm0, eax, 0", imm), Stop::Halted);
            let mut want = dwords_of(dst);
            want[(imm & 3) as usize] = 0x1234_5678;
            assert_eq!(rig.h.cpu.xmm[0], lanes32(want), "pinsrd {imm}");
            rig.h.cpu.xmm[0] = dst;
            rig.reset();
            assert_eq!(exec_imm(&mut rig, "pinsrq xmm0, rax, 0", imm), Stop::Halted);
            let mut want = qwords_of(dst);
            want[(imm & 1) as usize] = 0xdead_beef_1234_5678;
            assert_eq!(rig.h.cpu.xmm[0], lanes64(want), "pinsrq {imm}");
        }
    }

    #[test]
    fn pinsr_memory_forms_read_their_lane_size_unaligned() {
        let mut rig = Rig::new();
        let end = SCRATCH + SCRATCH_BYTES;
        rig.h
            .write(end - 8, &0x1122_3344_5566_7788u64.to_le_bytes());
        for (asm, size, imm, expected) in [
            ("pinsrb xmm0, [rsi], 0", 1usize, 4u8, {
                let mut b = bytes_of(EXTRACT_SOURCE);
                b[4] = 0x88;
                lanes8(b)
            }),
            ("pinsrd xmm0, [rsi], 0", 4, 1, {
                let mut d = dwords_of(EXTRACT_SOURCE);
                d[1] = 0x5566_7788;
                lanes32(d)
            }),
            ("pinsrq xmm0, [rsi], 0", 8, 0, {
                let mut q = qwords_of(EXTRACT_SOURCE);
                q[0] = 0x1122_3344_5566_7788;
                lanes64(q)
            }),
        ] {
            rig.h.cpu.xmm[0] = EXTRACT_SOURCE;
            rig.reset();
            rig.h.cpu.gpr[RSI] = end - size as u64;
            // The little-endian operand is the low `size` bytes of the qword written above.
            rig.h.write(
                end - size as u64,
                &0x1122_3344_5566_7788u64.to_le_bytes()[..size],
            );
            assert_eq!(exec_imm(&mut rig, asm, imm), Stop::Halted, "{asm}");
            assert_eq!(rig.h.cpu.xmm[0], expected, "{asm}");
        }
        // Extended registers.
        rig.h.cpu.xmm[9] = 0;
        rig.h.cpu.gpr[R8] = 0x77;
        rig.reset();
        assert_eq!(exec_imm(&mut rig, "pinsrq xmm9, r8, 0", 1), Stop::Halted);
        assert_eq!(rig.h.cpu.xmm[9], (0x77u128) << 64);
    }

    #[test]
    fn insertps_selects_source_and_destination_dwords_and_zeroes_by_mask() {
        let mut rig = Rig::new();
        let dst = lanes32([0xd0, 0xd1, 0xd2, 0xd3]);
        let src = lanes32([0x50, 0x51, 0x52, 0x53]);
        for imm in 0..=0xffu8 {
            let source_lane = (imm >> 6) as usize;
            let destination_lane = ((imm >> 4) & 3) as usize;
            let mut want = dwords_of(dst);
            want[destination_lane] = dwords_of(src)[source_lane];
            for (k, lane) in want.iter_mut().enumerate() {
                if imm >> k & 1 == 1 {
                    *lane = 0;
                }
            }
            assert_eq!(
                op_imm(&mut rig, "insertps xmm0, xmm1, 0", imm, dst, src),
                lanes32(want),
                "imm {imm:#x}"
            );
            // The m32 form always reads its one dword: count_s is ignored.
            let mut want_memory = dwords_of(dst);
            want_memory[destination_lane] = 0x50;
            for (k, lane) in want_memory.iter_mut().enumerate() {
                if imm >> k & 1 == 1 {
                    *lane = 0;
                }
            }
            assert_eq!(
                op_mem_imm(&mut rig, "insertps xmm0, [rsi], 0", imm, dst, src),
                lanes32(want_memory),
                "memory imm {imm:#x}"
            );
        }
    }

    const MODE_NAMES: [&str; 4] = ["nearest even", "down", "up", "toward zero"];

    fn round_single(x: f32, mode: u8) -> f32 {
        match mode {
            0 => x.round_ties_even(),
            1 => x.floor(),
            2 => x.ceil(),
            _ => x.trunc(),
        }
    }

    fn round_double(x: f64, mode: u8) -> f64 {
        match mode {
            0 => x.round_ties_even(),
            1 => x.floor(),
            2 => x.ceil(),
            _ => x.trunc(),
        }
    }

    const SINGLE_VALUES: [f32; 20] = [
        0.0,
        -0.0,
        0.3,
        -0.3,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.5,
        8_388_607.5,
        -8_388_607.5,
        8_388_608.0,
        1e10,
        -1e10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        123_456.75,
        1.0,
    ];

    const DOUBLE_VALUES: [f64; 18] = [
        0.0,
        -0.0,
        0.3,
        -0.3,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        4_503_599_627_370_495.5,
        -4_503_599_627_370_495.5,
        4_503_599_627_370_496.0,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        123_456.75,
        1.0,
    ];

    #[test]
    fn roundps_and_roundss_round_every_mode_and_flag_inexact_results() {
        let mut rig = Rig::new();
        for mode in 0..4u8 {
            for &x in &SINGLE_VALUES {
                let want_bits = round_single(x, mode).to_bits();
                let inexact = round_single(x, mode) != x;
                let src = singles([x, x, x, x]);
                let packed = op_imm(&mut rig, "roundps xmm0, xmm1, 0", mode, 0, src);
                assert_eq!(
                    dwords_of(packed),
                    [want_bits; 4],
                    "roundps {x} {}",
                    MODE_NAMES[mode as usize]
                );
                let flags = rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS;
                assert_eq!(
                    flags,
                    if inexact { FLAG_PRECISION } else { 0 },
                    "flags for {x}"
                );
                // The scalar form leaves the upper lanes of the destination alone.
                let marker = lanes32([0, 0x1111_1111, 0x2222_2222, 0x3333_3333]);
                let scalar = op_imm(&mut rig, "roundss xmm0, xmm1, 0", mode, marker, src);
                assert_eq!(
                    dwords_of(scalar),
                    [want_bits, 0x1111_1111, 0x2222_2222, 0x3333_3333],
                    "roundss {x}"
                );
                // Suppressing the precision exception hides the flag but not the rounding.
                let quiet = op_imm(&mut rig, "roundps xmm0, xmm1, 0", mode | 8, 0, src);
                assert_eq!(dwords_of(quiet), [want_bits; 4]);
                assert_eq!(rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS, 0, "suppressed {x}");
            }
        }
    }

    #[test]
    fn roundpd_and_roundsd_round_every_mode_and_flag_inexact_results() {
        let mut rig = Rig::new();
        for mode in 0..4u8 {
            for &x in &DOUBLE_VALUES {
                let want_bits = round_double(x, mode).to_bits();
                let inexact = round_double(x, mode) != x;
                let src = doubles([x, x]);
                let packed = op_imm(&mut rig, "roundpd xmm0, xmm1, 0", mode, 0, src);
                assert_eq!(
                    qwords_of(packed),
                    [want_bits; 2],
                    "roundpd {x} {}",
                    MODE_NAMES[mode as usize]
                );
                let flags = rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS;
                assert_eq!(
                    flags,
                    if inexact { FLAG_PRECISION } else { 0 },
                    "flags for {x}"
                );
                let scalar = op_imm(
                    &mut rig,
                    "roundsd xmm0, xmm1, 0",
                    mode,
                    lanes64([0, 0x1234]),
                    src,
                );
                assert_eq!(qwords_of(scalar), [want_bits, 0x1234], "roundsd {x}");
            }
        }
    }

    #[test]
    fn round_with_the_mxcsr_bit_uses_mxcsr_rc_and_ignores_immediate_mode_bits() {
        let mut rig = Rig::new();
        let src = singles([2.5, -2.5, 0.3, -0.3]);
        for mode in 0..4u8 {
            rig.h.cpu.xmm[1] = src;
            rig.reset();
            rig.h.cpu.mxcsr |= (mode as u32) << MXCSR_RC_SHIFT;
            // imm bit 2 set; the low mode bits (here 3) must be ignored.
            assert_eq!(
                exec_imm(&mut rig, "roundps xmm0, xmm1, 0", 0b111),
                Stop::Halted
            );
            let want = singles([2.5f32, -2.5, 0.3, -0.3].map(|x| round_single(x, mode)));
            assert_eq!(
                rig.h.cpu.xmm[0], want,
                "mxcsr rc {}",
                MODE_NAMES[mode as usize]
            );
        }
    }

    #[test]
    fn round_nan_inputs_and_denormals() {
        let mut rig = Rig::new();
        let qnan = f32::from_bits(0x7fc0_0123);
        let snan = f32::from_bits(0x7f80_0001);
        let result = op_imm(
            &mut rig,
            "roundps xmm0, xmm1, 0",
            1,
            0,
            singles([qnan, snan, 1.0, 0.0]),
        );
        assert_eq!(
            dwords_of(result)[0],
            0x7fc0_0123,
            "a quiet NaN passes through"
        );
        assert_eq!(
            dwords_of(result)[1],
            0x7fc0_0001,
            "a signaling NaN is quieted"
        );
        assert_eq!(rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID);
        // A subnormal rounds to 0 (nearest) or to 1.0 (up) and raises only precision, never the
        // denormal flag.
        let subnormal = f32::from_bits(1);
        let result = op_imm(
            &mut rig,
            "roundss xmm0, xmm1, 0",
            2,
            0,
            singles([subnormal, 0.0, 0.0, 0.0]),
        );
        assert_eq!(dwords_of(result)[0], 1.0f32.to_bits());
        assert_eq!(rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS, FLAG_PRECISION);
        // With DAZ the subnormal is a zero: nothing to round up, nothing inexact.
        rig.h.cpu.xmm[1] = singles([subnormal, 0.0, 0.0, 0.0]);
        rig.reset();
        rig.h.cpu.mxcsr |= MXCSR_DAZ;
        assert_eq!(exec_imm(&mut rig, "roundss xmm0, xmm1, 0", 2), Stop::Halted);
        assert_eq!(dwords_of(rig.h.cpu.xmm[0])[0], 0);
        assert_eq!(rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS, 0);
    }

    #[test]
    fn an_unmasked_precision_exception_faults_unless_suppressed() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = 0x5555;
        rig.h.cpu.xmm[1] = singles([2.5, 0.0, 0.0, 0.0]);
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        rig.h.cpu.mxcsr &= !(FLAG_PRECISION << MXCSR_MASK_SHIFT);
        assert_eq!(exec_imm(&mut rig, "roundss xmm0, xmm1, 0", 0), xm());
        assert_eq!(
            rig.h.cpu.xmm[0], 0x5555,
            "the destination is untouched by a fault"
        );
        assert_eq!(rig.h.cpu.mxcsr & FLAG_PRECISION, FLAG_PRECISION);
        rig.h.cpu.mxcsr &= !FLAG_PRECISION;
        assert_eq!(exec_imm(&mut rig, "roundss xmm0, xmm1, 0", 8), Stop::Halted);
        assert_eq!(dwords_of(rig.h.cpu.xmm[0])[0], 2.0f32.to_bits());
    }

    #[test]
    fn round_memory_forms_follow_the_operand_size_and_alignment_rules() {
        let mut rig = Rig::new();
        // roundss reads 4 bytes, roundsd 8: any alignment.
        rig.h.write(DATA + 1, &2.5f32.to_le_bytes());
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 1;
        assert_eq!(
            exec_imm(&mut rig, "roundss xmm0, [rsi], 0", 0),
            Stop::Halted
        );
        assert_eq!(dwords_of(rig.h.cpu.xmm[0])[0], 2.0f32.to_bits());
        // roundps needs an aligned m128.
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 4;
        assert_eq!(exec_imm(&mut rig, "roundps xmm0, [rsi], 0", 0), gp0());
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 8;
        assert_eq!(exec_imm(&mut rig, "roundpd xmm0, [rsi], 0", 0), gp0());
    }

    #[test]
    fn dpps_multiplies_selected_lanes_and_broadcasts_the_sum_to_selected_outputs() {
        let mut rig = Rig::new();
        let dst = singles([1.0, 2.0, 3.0, 4.0]);
        let src = singles([5.0, 6.0, 7.0, 8.0]);
        for imm in 0..=0xffu8 {
            let products = [5.0f32, 12.0, 21.0, 32.0];
            let selected = |k: usize| {
                if imm >> (4 + k) & 1 == 1 {
                    products[k]
                } else {
                    0.0
                }
            };
            let sum = (selected(0) + selected(1)) + (selected(2) + selected(3));
            let want = singles(std::array::from_fn(|k| {
                if imm >> k & 1 == 1 { sum } else { 0.0 }
            }));
            assert_eq!(
                op_imm(&mut rig, "dpps xmm0, xmm1, 0", imm, dst, src),
                want,
                "dpps {imm:#x}"
            );
            assert_eq!(
                op_mem_imm(&mut rig, "dpps xmm0, [rsi], 0", imm, dst, src),
                want,
                "dpps memory {imm:#x}"
            );
        }
    }

    #[test]
    fn dppd_multiplies_selected_lanes_and_broadcasts_the_sum_to_selected_outputs() {
        let mut rig = Rig::new();
        let dst = doubles([1.5, 2.0]);
        let src = doubles([4.0, -3.0]);
        for imm in 0..=0xffu8 {
            let products = [6.0f64, -6.0];
            let selected = |k: usize| {
                if imm >> (4 + k) & 1 == 1 {
                    products[k]
                } else {
                    0.0
                }
            };
            let sum = selected(0) + selected(1);
            let want = doubles(std::array::from_fn(|k| {
                if imm >> k & 1 == 1 { sum } else { 0.0 }
            }));
            assert_eq!(
                op_imm(&mut rig, "dppd xmm0, xmm1, 0", imm, dst, src),
                want,
                "dppd {imm:#x}"
            );
        }
        assert_eq!(
            op_mem_imm(&mut rig, "dppd xmm0, [rsi], 0", 0x11, dst, src),
            doubles([6.0, 0.0])
        );
    }

    #[test]
    fn dpps_propagates_nans_and_raises_exceptions_like_the_arithmetic_ops() {
        let mut rig = Rig::new();
        let nan = f32::from_bits(0x7fc0_0042);
        let result = op_imm(
            &mut rig,
            "dpps xmm0, xmm1, 0",
            0xff,
            singles([1.0, nan, 1.0, 1.0]),
            singles([1.0, 1.0, 1.0, 1.0]),
        );
        assert_eq!(dwords_of(result), [0x7fc0_0042; 4]);
        // inf * 0 is invalid: the default NaN, with the flag.
        let result = op_imm(
            &mut rig,
            "dpps xmm0, xmm1, 0",
            0x11,
            singles([f32::INFINITY, 0.0, 0.0, 0.0]),
            singles([0.0, 0.0, 0.0, 0.0]),
        );
        assert_eq!(dwords_of(result)[0], DEFAULT_NAN_SINGLE);
        assert_eq!(rig.h.cpu.mxcsr & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID);
    }

    #[test]
    fn movntdqa_loads_aligned_memory_only() {
        let mut rig = Rig::new();
        let value = vectors()[10];
        assert_eq!(rig.op_mem("movntdqa xmm0, [rsi]", 0, value), value);
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 8;
        assert_eq!(rig.exec("movntdqa xmm0, [rsi]"), gp0());
        rig.h.cpu.cr0 |= CR0_TS;
        assert_eq!(
            rig.exec(".byte 0x66, 0x0f, 0x38, 0x2a, 0xc1"),
            ud(),
            "no register form, even with CR0.TS set"
        );
    }

    #[test]
    fn sse41_memory_operands_must_be_aligned() {
        let mut rig = Rig::new();
        for asm in [
            "pblendvb xmm1, [rsi], xmm0",
            "pblendw xmm0, [rsi], 0",
            "ptest xmm0, [rsi]",
            "pmulld xmm0, [rsi]",
            "pminsb xmm0, [rsi]",
            "pcmpeqq xmm0, [rsi]",
            "packusdw xmm0, [rsi]",
            "phminposuw xmm0, [rsi]",
            "dpps xmm0, [rsi], 0",
            "mpsadbw xmm0, [rsi], 0",
            "blendpd xmm0, [rsi], 0",
        ] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 1;
            assert_eq!(rig.exec(asm), gp0(), "{asm}");
        }
    }

    #[test]
    fn enable_rules_apply() {
        let mut rig = Rig::new();
        for asm in [
            "pblendvb xmm1, xmm2, xmm0",
            "ptest xmm0, xmm1",
            "pmovsxbw xmm0, xmm1",
            "roundss xmm0, xmm1, 0",
            "pextrd eax, xmm1, 0",
            "pinsrb xmm0, eax, 0",
            "insertps xmm0, xmm1, 0",
        ] {
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            assert_eq!(rig.exec(asm), ud(), "{asm} without OSFXSR");
            rig.reset();
            rig.h.cpu.cr0 |= CR0_EM;
            assert_eq!(rig.exec(asm), ud(), "{asm} with CR0.EM");
            rig.reset();
            rig.h.cpu.cr0 |= CR0_TS;
            assert_eq!(rig.exec(asm), nm(), "{asm} with CR0.TS");
        }
    }

    #[test]
    fn jit_matches_the_interpreter() {
        for asm in [
            "pblendvb xmm1, xmm2, xmm0",
            "blendvps xmm1, [rsi], xmm0",
            "ptest xmm0, xmm1",
            "pmovsxbd xmm0, xmm1",
            "pmuldq xmm0, xmm1",
            "pminud xmm0, [rsi]",
            "phminposuw xmm0, xmm1",
            "mpsadbw xmm0, xmm1, 0",
            "pextrq rax, xmm1, 0",
            "pinsrb xmm0, eax, 0",
            "insertps xmm0, xmm1, 0",
            "roundps xmm0, xmm1, 0",
            "roundsd xmm0, [rsi], 0",
            "dpps xmm0, xmm1, 0",
            "packusdw xmm0, xmm1",
        ] {
            assert_jit_matches_interpreter(code(asm), |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                h.cpu.gpr[RSI] = DATA;
                h.cpu.gpr[RAX] = 0x1234_5678_9abc_def0;
                h.cpu.xmm[0] = singles([1.5, -2.25, 1e30, 7.0]);
                h.cpu.xmm[1] = singles([2.5, 3.5, -1e30, -7.5]);
                h.cpu.xmm[2] = 0x8070_6050_4030_2010_00ff_7f80_0102_0304;
                h.write(DATA, &h.cpu.xmm[1].to_le_bytes());
            });
        }
    }
}

const SNIPPETS: &[(&str, &[u8])] = &[
    ("movddup xmm0, xmm1", &[0xf2, 0x0f, 0x12, 0xc1, 0xf4]),
    ("movddup xmm0, [rsi]", &[0xf2, 0x0f, 0x12, 0x06, 0xf4]),
    ("movsldup xmm0, xmm1", &[0xf3, 0x0f, 0x12, 0xc1, 0xf4]),
    ("movsldup xmm0, [rsi]", &[0xf3, 0x0f, 0x12, 0x06, 0xf4]),
    ("movshdup xmm0, xmm1", &[0xf3, 0x0f, 0x16, 0xc1, 0xf4]),
    ("movshdup xmm0, [rsi]", &[0xf3, 0x0f, 0x16, 0x06, 0xf4]),
    ("lddqu xmm0, [rsi]", &[0xf2, 0x0f, 0xf0, 0x06, 0xf4]),
    (
        "lddqu xmm0, [rsi + 1]",
        &[0xf2, 0x0f, 0xf0, 0x46, 0x01, 0xf4],
    ),
    (
        ".byte 0xf2, 0x0f, 0xf0, 0xc1",
        &[0xf2, 0x0f, 0xf0, 0xc1, 0xf4],
    ),
    ("addsubps xmm0, xmm1", &[0xf2, 0x0f, 0xd0, 0xc1, 0xf4]),
    ("addsubps xmm0, [rsi]", &[0xf2, 0x0f, 0xd0, 0x06, 0xf4]),
    ("addsubpd xmm0, xmm1", &[0x66, 0x0f, 0xd0, 0xc1, 0xf4]),
    ("addsubpd xmm0, [rsi]", &[0x66, 0x0f, 0xd0, 0x06, 0xf4]),
    ("haddps xmm0, xmm1", &[0xf2, 0x0f, 0x7c, 0xc1, 0xf4]),
    ("haddps xmm0, [rsi]", &[0xf2, 0x0f, 0x7c, 0x06, 0xf4]),
    ("haddpd xmm0, xmm1", &[0x66, 0x0f, 0x7c, 0xc1, 0xf4]),
    ("haddpd xmm0, [rsi]", &[0x66, 0x0f, 0x7c, 0x06, 0xf4]),
    ("hsubps xmm0, xmm1", &[0xf2, 0x0f, 0x7d, 0xc1, 0xf4]),
    ("hsubps xmm0, [rsi]", &[0xf2, 0x0f, 0x7d, 0x06, 0xf4]),
    ("hsubpd xmm0, xmm1", &[0x66, 0x0f, 0x7d, 0xc1, 0xf4]),
    ("hsubpd xmm0, [rsi]", &[0x66, 0x0f, 0x7d, 0x06, 0xf4]),
    (".byte 0x0f, 0x7c, 0xc1", &[0x0f, 0x7c, 0xc1, 0xf4]),
    (".byte 0x0f, 0xd0, 0xc1", &[0x0f, 0xd0, 0xc1, 0xf4]),
    (
        ".byte 0xf3, 0x0f, 0xd0, 0xc1",
        &[0xf3, 0x0f, 0xd0, 0xc1, 0xf4],
    ),
    (
        ".byte 0xf3, 0x0f, 0xf0, 0x06",
        &[0xf3, 0x0f, 0xf0, 0x06, 0xf4],
    ),
    (
        ".byte 0x66, 0x0f, 0xf0, 0x06",
        &[0x66, 0x0f, 0xf0, 0x06, 0xf4],
    ),
    (
        ".byte 0x66, 0x0f, 0x12, 0xc1",
        &[0x66, 0x0f, 0x12, 0xc1, 0xf4],
    ),
    ("fisttp word ptr [rsi]", &[0xdf, 0x0e, 0xf4]),
    ("fisttp dword ptr [rsi]", &[0xdb, 0x0e, 0xf4]),
    ("fisttp qword ptr [rsi]", &[0xdd, 0x0e, 0xf4]),
    ("popcnt rax, rbx", &[0xf3, 0x48, 0x0f, 0xb8, 0xc3, 0xf4]),
    ("popcnt eax, ebx", &[0xf3, 0x0f, 0xb8, 0xc3, 0xf4]),
    ("popcnt ax, bx", &[0x66, 0xf3, 0x0f, 0xb8, 0xc3, 0xf4]),
    ("popcnt rax, [rsi]", &[0xf3, 0x48, 0x0f, 0xb8, 0x06, 0xf4]),
    ("popcnt eax, [rsi]", &[0xf3, 0x0f, 0xb8, 0x06, 0xf4]),
    ("popcnt ax, [rsi]", &[0x66, 0xf3, 0x0f, 0xb8, 0x06, 0xf4]),
    ("lzcnt rax, rbx", &[0xf3, 0x48, 0x0f, 0xbd, 0xc3, 0xf4]),
    ("lzcnt eax, ebx", &[0xf3, 0x0f, 0xbd, 0xc3, 0xf4]),
    ("lzcnt ax, bx", &[0x66, 0xf3, 0x0f, 0xbd, 0xc3, 0xf4]),
    ("lzcnt rax, [rsi]", &[0xf3, 0x48, 0x0f, 0xbd, 0x06, 0xf4]),
    (".byte 0x0f, 0xb8, 0xc3", &[0x0f, 0xb8, 0xc3, 0xf4]),
    (
        ".byte 0xf2, 0x0f, 0xb8, 0xc3",
        &[0xf2, 0x0f, 0xb8, 0xc3, 0xf4],
    ),
    ("cmpxchg16b [rsi]", &[0x48, 0x0f, 0xc7, 0x0e, 0xf4]),
    (
        "cmpxchg16b [rsi + 8]",
        &[0x48, 0x0f, 0xc7, 0x4e, 0x08, 0xf4],
    ),
    (
        "lock cmpxchg16b [rsi]",
        &[0xf0, 0x48, 0x0f, 0xc7, 0x0e, 0xf4],
    ),
    (
        ".byte 0x48, 0x0f, 0xc7, 0xce",
        &[0x48, 0x0f, 0xc7, 0xce, 0xf4],
    ),
    ("cmpxchg8b [rsi]", &[0x0f, 0xc7, 0x0e, 0xf4]),
    ("pshufb xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x00, 0xc1, 0xf4]),
    ("pshufb xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x00, 0x06, 0xf4]),
    ("phaddw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x01, 0xc1, 0xf4]),
    ("phaddw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x01, 0x06, 0xf4]),
    ("phaddd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x02, 0xc1, 0xf4]),
    ("phaddd xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x02, 0x06, 0xf4]),
    ("phaddsw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x03, 0xc1, 0xf4]),
    ("phaddsw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x03, 0x06, 0xf4]),
    (
        "pmaddubsw xmm0, xmm1",
        &[0x66, 0x0f, 0x38, 0x04, 0xc1, 0xf4],
    ),
    (
        "pmaddubsw xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x04, 0x06, 0xf4],
    ),
    ("phsubw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x05, 0xc1, 0xf4]),
    ("phsubw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x05, 0x06, 0xf4]),
    ("phsubd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x06, 0xc1, 0xf4]),
    ("phsubd xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x06, 0x06, 0xf4]),
    ("phsubsw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x07, 0xc1, 0xf4]),
    ("phsubsw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x07, 0x06, 0xf4]),
    ("psignb xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x08, 0xc1, 0xf4]),
    ("psignb xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x08, 0x06, 0xf4]),
    ("psignw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x09, 0xc1, 0xf4]),
    ("psignw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x09, 0x06, 0xf4]),
    ("psignd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x0a, 0xc1, 0xf4]),
    ("psignd xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x0a, 0x06, 0xf4]),
    ("pmulhrsw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x0b, 0xc1, 0xf4]),
    (
        "pmulhrsw xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x0b, 0x06, 0xf4],
    ),
    ("pabsb xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x1c, 0xc1, 0xf4]),
    ("pabsb xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x1c, 0x06, 0xf4]),
    ("pabsw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x1d, 0xc1, 0xf4]),
    ("pabsw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x1d, 0x06, 0xf4]),
    ("pabsd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x1e, 0xc1, 0xf4]),
    ("pabsd xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x1e, 0x06, 0xf4]),
    (
        "palignr xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x00, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 1",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x01, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 4",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x04, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 15",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x0f, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 16",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x10, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 17",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x11, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 31",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x1f, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 32",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x20, 0xf4],
    ),
    (
        "palignr xmm0, xmm1, 255",
        &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0xff, 0xf4],
    ),
    (
        "palignr xmm0, [rsi], 3",
        &[0x66, 0x0f, 0x3a, 0x0f, 0x06, 0x03, 0xf4],
    ),
    (
        ".byte 0x0f, 0x38, 0x00, 0xc1",
        &[0x0f, 0x38, 0x00, 0xc1, 0xf4],
    ),
    (
        ".byte 0x0f, 0x38, 0x1c, 0xc1",
        &[0x0f, 0x38, 0x1c, 0xc1, 0xf4],
    ),
    (
        ".byte 0x0f, 0x3a, 0x0f, 0xc1, 0x04",
        &[0x0f, 0x3a, 0x0f, 0xc1, 0x04, 0xf4],
    ),
    (
        ".byte 0xf2, 0x0f, 0x38, 0x00, 0xc1",
        &[0xf2, 0x0f, 0x38, 0x00, 0xc1, 0xf4],
    ),
    (
        ".byte 0xf3, 0x0f, 0x38, 0x00, 0xc1",
        &[0xf3, 0x0f, 0x38, 0x00, 0xc1, 0xf4],
    ),
    (
        ".byte 0x66, 0x0f, 0x38, 0xff, 0xc1",
        &[0x66, 0x0f, 0x38, 0xff, 0xc1, 0xf4],
    ),
    (
        ".byte 0x66, 0x0f, 0x3a, 0xff, 0xc1, 0x00",
        &[0x66, 0x0f, 0x3a, 0xff, 0xc1, 0x00, 0xf4],
    ),
    (
        "pblendvb xmm1, xmm2, xmm0",
        &[0x66, 0x0f, 0x38, 0x10, 0xca, 0xf4],
    ),
    (
        "pblendvb xmm1, [rsi], xmm0",
        &[0x66, 0x0f, 0x38, 0x10, 0x0e, 0xf4],
    ),
    (
        "blendvps xmm1, xmm2, xmm0",
        &[0x66, 0x0f, 0x38, 0x14, 0xca, 0xf4],
    ),
    (
        "blendvps xmm1, [rsi], xmm0",
        &[0x66, 0x0f, 0x38, 0x14, 0x0e, 0xf4],
    ),
    (
        "blendvpd xmm1, xmm2, xmm0",
        &[0x66, 0x0f, 0x38, 0x15, 0xca, 0xf4],
    ),
    (
        "blendvpd xmm1, [rsi], xmm0",
        &[0x66, 0x0f, 0x38, 0x15, 0x0e, 0xf4],
    ),
    (
        "pblendw xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x0e, 0xc1, 0x00, 0xf4],
    ),
    (
        "pblendw xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x0e, 0x06, 0x00, 0xf4],
    ),
    (
        "blendps xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x0c, 0xc1, 0x00, 0xf4],
    ),
    (
        "blendps xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x0c, 0x06, 0x00, 0xf4],
    ),
    (
        "blendpd xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x0d, 0xc1, 0x00, 0xf4],
    ),
    (
        "blendpd xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x0d, 0x06, 0x00, 0xf4],
    ),
    (
        "mpsadbw xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x42, 0xc1, 0x00, 0xf4],
    ),
    (
        "mpsadbw xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x42, 0x06, 0x00, 0xf4],
    ),
    (
        "dpps xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x40, 0xc1, 0x00, 0xf4],
    ),
    (
        "dpps xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x40, 0x06, 0x00, 0xf4],
    ),
    (
        "dppd xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x41, 0xc1, 0x00, 0xf4],
    ),
    (
        "dppd xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x41, 0x06, 0x00, 0xf4],
    ),
    (
        "roundps xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x08, 0xc1, 0x00, 0xf4],
    ),
    (
        "roundps xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x08, 0x06, 0x00, 0xf4],
    ),
    (
        "roundpd xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x09, 0xc1, 0x00, 0xf4],
    ),
    (
        "roundpd xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x09, 0x06, 0x00, 0xf4],
    ),
    (
        "roundss xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x0a, 0xc1, 0x00, 0xf4],
    ),
    (
        "roundss xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x0a, 0x06, 0x00, 0xf4],
    ),
    (
        "roundsd xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x0b, 0xc1, 0x00, 0xf4],
    ),
    (
        "roundsd xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x0b, 0x06, 0x00, 0xf4],
    ),
    ("ptest xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x17, 0xc1, 0xf4]),
    ("ptest xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x17, 0x06, 0xf4]),
    ("pmovsxbw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x20, 0xc1, 0xf4]),
    (
        "pmovsxbw xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x20, 0x06, 0xf4],
    ),
    ("pmovzxbw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x30, 0xc1, 0xf4]),
    (
        "pmovzxbw xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x30, 0x06, 0xf4],
    ),
    ("pmovsxbd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x21, 0xc1, 0xf4]),
    (
        "pmovsxbd xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x21, 0x06, 0xf4],
    ),
    ("pmovzxbd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x31, 0xc1, 0xf4]),
    (
        "pmovzxbd xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x31, 0x06, 0xf4],
    ),
    ("pmovsxbq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x22, 0xc1, 0xf4]),
    (
        "pmovsxbq xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x22, 0x06, 0xf4],
    ),
    ("pmovzxbq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x32, 0xc1, 0xf4]),
    (
        "pmovzxbq xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x32, 0x06, 0xf4],
    ),
    ("pmovsxwd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x23, 0xc1, 0xf4]),
    (
        "pmovsxwd xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x23, 0x06, 0xf4],
    ),
    ("pmovzxwd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x33, 0xc1, 0xf4]),
    (
        "pmovzxwd xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x33, 0x06, 0xf4],
    ),
    ("pmovsxwq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x24, 0xc1, 0xf4]),
    (
        "pmovsxwq xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x24, 0x06, 0xf4],
    ),
    ("pmovzxwq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x34, 0xc1, 0xf4]),
    (
        "pmovzxwq xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x34, 0x06, 0xf4],
    ),
    ("pmovsxdq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x25, 0xc1, 0xf4]),
    (
        "pmovsxdq xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x25, 0x06, 0xf4],
    ),
    ("pmovzxdq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x35, 0xc1, 0xf4]),
    (
        "pmovzxdq xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x35, 0x06, 0xf4],
    ),
    ("pmuldq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x28, 0xc1, 0xf4]),
    ("pmuldq xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x28, 0x06, 0xf4]),
    ("pmulld xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x40, 0xc1, 0xf4]),
    ("pmulld xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x40, 0x06, 0xf4]),
    ("pminsb xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x38, 0xc1, 0xf4]),
    ("pminsb xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x38, 0x06, 0xf4]),
    ("pminsd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x39, 0xc1, 0xf4]),
    ("pminsd xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x39, 0x06, 0xf4]),
    ("pminuw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x3a, 0xc1, 0xf4]),
    ("pminuw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x3a, 0x06, 0xf4]),
    ("pminud xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x3b, 0xc1, 0xf4]),
    ("pminud xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x3b, 0x06, 0xf4]),
    ("pmaxsb xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x3c, 0xc1, 0xf4]),
    ("pmaxsb xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x3c, 0x06, 0xf4]),
    ("pmaxsd xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x3d, 0xc1, 0xf4]),
    ("pmaxsd xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x3d, 0x06, 0xf4]),
    ("pmaxuw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x3e, 0xc1, 0xf4]),
    ("pmaxuw xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x3e, 0x06, 0xf4]),
    ("pmaxud xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x3f, 0xc1, 0xf4]),
    ("pmaxud xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x3f, 0x06, 0xf4]),
    ("pcmpeqq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x29, 0xc1, 0xf4]),
    ("pcmpeqq xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x29, 0x06, 0xf4]),
    ("packusdw xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x2b, 0xc1, 0xf4]),
    (
        "packusdw xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x2b, 0x06, 0xf4],
    ),
    (
        "phminposuw xmm0, xmm1",
        &[0x66, 0x0f, 0x38, 0x41, 0xc1, 0xf4],
    ),
    (
        "phminposuw xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x41, 0x06, 0xf4],
    ),
    (
        "movntdqa xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0x2a, 0x06, 0xf4],
    ),
    (
        ".byte 0x66, 0x0f, 0x38, 0x2a, 0xc1",
        &[0x66, 0x0f, 0x38, 0x2a, 0xc1, 0xf4],
    ),
    (
        "pextrb eax, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x14, 0xc8, 0x00, 0xf4],
    ),
    (
        "pextrb [rsi], xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x14, 0x0e, 0x00, 0xf4],
    ),
    (
        "pextrb r9d, xmm9, 0",
        &[0x66, 0x45, 0x0f, 0x3a, 0x14, 0xc9, 0x00, 0xf4],
    ),
    (
        ".byte 0x66, 0x0f, 0x3a, 0x15, 0xc8, 0x00",
        &[0x66, 0x0f, 0x3a, 0x15, 0xc8, 0x00, 0xf4],
    ),
    (
        "pextrw [rsi], xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x15, 0x0e, 0x00, 0xf4],
    ),
    (
        "pextrd eax, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x16, 0xc8, 0x00, 0xf4],
    ),
    (
        "pextrd [rsi], xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x16, 0x0e, 0x00, 0xf4],
    ),
    (
        "pextrq rax, xmm1, 0",
        &[0x66, 0x48, 0x0f, 0x3a, 0x16, 0xc8, 0x00, 0xf4],
    ),
    (
        "pextrq [rsi], xmm1, 0",
        &[0x66, 0x48, 0x0f, 0x3a, 0x16, 0x0e, 0x00, 0xf4],
    ),
    (
        "pextrq r9, xmm9, 0",
        &[0x66, 0x4d, 0x0f, 0x3a, 0x16, 0xc9, 0x00, 0xf4],
    ),
    (
        "extractps eax, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x17, 0xc8, 0x00, 0xf4],
    ),
    (
        "extractps [rsi], xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x17, 0x0e, 0x00, 0xf4],
    ),
    (
        "pinsrb xmm0, eax, 0",
        &[0x66, 0x0f, 0x3a, 0x20, 0xc0, 0x00, 0xf4],
    ),
    (
        "pinsrb xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x20, 0x06, 0x00, 0xf4],
    ),
    (
        "pinsrd xmm0, eax, 0",
        &[0x66, 0x0f, 0x3a, 0x22, 0xc0, 0x00, 0xf4],
    ),
    (
        "pinsrd xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x22, 0x06, 0x00, 0xf4],
    ),
    (
        "pinsrq xmm0, rax, 0",
        &[0x66, 0x48, 0x0f, 0x3a, 0x22, 0xc0, 0x00, 0xf4],
    ),
    (
        "pinsrq xmm0, [rsi], 0",
        &[0x66, 0x48, 0x0f, 0x3a, 0x22, 0x06, 0x00, 0xf4],
    ),
    (
        "pinsrq xmm9, r8, 0",
        &[0x66, 0x4d, 0x0f, 0x3a, 0x22, 0xc8, 0x00, 0xf4],
    ),
    (
        "insertps xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x21, 0xc1, 0x00, 0xf4],
    ),
    (
        "insertps xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x21, 0x06, 0x00, 0xf4],
    ),
    (
        "movaps xmm0, xmm1; roundps xmm0, [rsi], 0",
        &[0x0f, 0x28, 0xc1, 0x66, 0x0f, 0x3a, 0x08, 0x06, 0x00, 0xf4],
    ),
    // SNIPPETS-END
];
