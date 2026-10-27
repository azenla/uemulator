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
        let bytes = code(asm);
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
    // SNIPPETS-END
];
