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
    // SNIPPETS-END
];
