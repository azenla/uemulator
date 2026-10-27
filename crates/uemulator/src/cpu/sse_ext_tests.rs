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

mod sse42 {
    use super::*;

    fn exec_imm(rig: &mut Rig, asm: &str, imm: u8) -> Stop {
        let mut bytes = code(asm).to_vec();
        let last = bytes.len() - 2;
        bytes[last] = imm;
        rig.exec_bytes(&bytes)
    }

    #[test]
    fn pcmpgtq_compares_signed_quadwords() {
        let mut rig = Rig::new();
        check_binary(&mut rig, "pcmpgtq", |a, b| {
            let (x, y) = (qwords_of(a), qwords_of(b));
            lanes64(std::array::from_fn(|k| {
                if (x[k] as i64) > (y[k] as i64) {
                    u64::MAX
                } else {
                    0
                }
            }))
        });
        assert_eq!(
            rig.op(
                "pcmpgtq xmm0, xmm1",
                lanes64([1, u64::MAX]),
                lanes64([u64::MAX, 1])
            ),
            lanes64([u64::MAX, 0]),
            "-1 < 1 as signed"
        );
        assert_eq!(rig.run("pcmpgtq xmm0, [rsi]"), Stop::Halted);
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 8;
        assert_eq!(rig.exec("pcmpgtq xmm0, [rsi]"), gp0());
    }

    /// CRC-32C as software computes it around the instruction: all-ones initial value and a final
    /// inversion.
    const CRC_INITIAL: u64 = 0xffff_ffff;

    fn crc32c_of(rig: &mut Rig, data: &[u8]) -> u32 {
        let mut crc = CRC_INITIAL;
        for &byte in data {
            rig.h.cpu.gpr[RAX] = crc;
            rig.h.cpu.gpr[RBX] = byte as u64;
            rig.run_ok("crc32 eax, bl");
            crc = rig.gpr(RAX);
        }
        !(crc as u32)
    }

    #[test]
    fn crc32_bytes_produce_the_standard_crc32c_check_values() {
        let mut rig = Rig::new();
        // The CRC-32C catalogue check value, and the RFC 3720 (iSCSI) appendix B.4 patterns.
        assert_eq!(crc32c_of(&mut rig, b"123456789"), 0xe306_9283);
        assert_eq!(crc32c_of(&mut rig, &[0u8; 32]), 0x8a91_36aa);
        assert_eq!(crc32c_of(&mut rig, &[0xffu8; 32]), 0x62a8_ab43);
        let ascending: Vec<u8> = (0..32).collect();
        assert_eq!(crc32c_of(&mut rig, &ascending), 0x46dd_794e);
        let descending: Vec<u8> = (0..32).rev().collect();
        assert_eq!(crc32c_of(&mut rig, &descending), 0x113f_db5c);
    }

    #[test]
    fn wider_crc32_forms_equal_repeated_byte_steps() {
        let mut rig = Rig::new();
        let data: Vec<u8> = (1..=8).map(|k| k * 0x1f).collect();
        let want = crc32c_of(&mut rig, &data);
        let run = |rig: &mut Rig, asm: &str, chunks: &[&[u8]]| -> u32 {
            let mut crc = CRC_INITIAL;
            for chunk in chunks {
                let mut value = [0u8; 8];
                value[..chunk.len()].copy_from_slice(chunk);
                rig.h.cpu.gpr[RAX] = crc;
                rig.h.cpu.gpr[RBX] = u64::from_le_bytes(value);
                rig.h.write(DATA, &value);
                rig.run_ok(asm);
                crc = rig.gpr(RAX);
            }
            !(crc as u32)
        };
        let words: Vec<&[u8]> = data.chunks(2).collect();
        let dwords: Vec<&[u8]> = data.chunks(4).collect();
        let qwords: Vec<&[u8]> = data.chunks(8).collect();
        for (asm, chunks) in [
            ("crc32 eax, bx", &words),
            ("crc32 eax, word ptr [rsi]", &words),
            ("crc32 eax, ebx", &dwords),
            ("crc32 eax, dword ptr [rsi]", &dwords),
            ("crc32 rax, rbx", &qwords),
            ("crc32 rax, qword ptr [rsi]", &qwords),
        ] {
            assert_eq!(run(&mut rig, asm, chunks), want, "{asm}");
        }
    }

    #[test]
    fn crc32_destination_width_and_register_selection() {
        let mut rig = Rig::new();
        // A 32-bit destination zero-extends; a REX.W one with a byte source gives a 64-bit write
        // whose upper half is zero too.
        rig.h.cpu.gpr[RAX] = 0xdead_beef_0000_0000;
        rig.h.cpu.gpr[RBX] = 0x55;
        rig.run_ok("crc32 eax, bl");
        assert_eq!(rig.gpr(RAX), crypto::crc32c(0, 0x55, 1) as u64);
        rig.h.cpu.gpr[RAX] = 0xdead_beef_0000_0001;
        rig.run_ok("crc32 rax, bl");
        assert_eq!(rig.gpr(RAX), crypto::crc32c(1, 0x55, 1) as u64);
        // Without a REX prefix byte register 7 is bh, the second byte of rbx.
        rig.h.cpu.gpr[RAX] = 0;
        rig.h.cpu.gpr[RBX] = 0x1234;
        rig.run_ok("crc32 eax, bh");
        assert_eq!(rig.gpr(RAX), crypto::crc32c(0, 0x12, 1) as u64);
        // With a REX prefix the byte registers are the low bytes (r10b here).
        rig.h.cpu.gpr[R9] = 0xffff_ffff_0000_0003;
        rig.h.cpu.gpr[R10] = 0x77;
        rig.run_ok("crc32 r9d, r10b");
        assert_eq!(rig.gpr(R9), crypto::crc32c(3, 0x77, 1) as u64);
        rig.h.cpu.gpr[R9] = 7;
        rig.h.cpu.gpr[R10] = 0x0102_0304_0506_0708;
        rig.run_ok("crc32 r9, r10");
        assert_eq!(
            rig.gpr(R9),
            crypto::crc32c(7, 0x0102_0304_0506_0708, 8) as u64
        );
        // Memory forms.
        rig.h.write(DATA, &[0x9a, 0xbc]);
        rig.h.cpu.gpr[RAX] = 5;
        rig.run_ok("crc32 eax, byte ptr [rsi]");
        assert_eq!(rig.gpr(RAX), crypto::crc32c(5, 0x9a, 1) as u64);
    }

    #[test]
    fn crc32_needs_the_f2_prefix_and_ignores_the_sse_enable_rules() {
        let mut rig = Rig::new();
        assert_eq!(rig.run(".byte 0xf3, 0x0f, 0x38, 0xf0, 0xc3"), ud());
        rig.reset();
        rig.h.cpu.cr4 &= !CR4_OSFXSR;
        rig.h.cpu.cr0 |= CR0_TS | CR0_EM;
        assert_eq!(rig.exec("crc32 eax, bl"), Stop::Halted);
    }

    fn element(register: u128, index: i32, control: u8) -> i32 {
        let k = index as usize;
        match (control & 1 != 0, control & 2 != 0) {
            (false, false) => bytes_of(register)[k] as i32,
            (false, true) => bytes_of(register)[k] as i8 as i32,
            (true, false) => words_of(register)[k] as i32,
            (true, true) => words_of(register)[k] as i16 as i32,
        }
    }

    /// A second formulation of the string comparison, structured after the loops QEMU uses (bit
    /// positions built by shifting, lengths held as index-of-last-valid). `first_length` and
    /// `second_length` are element counts. Returns IntRes2 and (carry, zero, sign, overflow).
    fn reference_compare(
        control: u8,
        first: u128,
        first_length: i32,
        second: u128,
        second_length: i32,
    ) -> (u32, [bool; 4]) {
        let upper: i32 = if control & 1 != 0 { 7 } else { 15 };
        let valid_first = first_length - 1;
        let valid_second = second_length - 1;
        let mut result: i32 = 0;
        match (control >> 2) & 3 {
            0 => {
                for j in (0..=valid_second).rev() {
                    result <<= 1;
                    let v = element(second, j, control);
                    for i in (0..=valid_first).rev() {
                        result |= (v == element(first, i, control)) as i32;
                    }
                }
            }
            1 => {
                for j in (0..=valid_second).rev() {
                    result <<= 1;
                    let v = element(second, j, control);
                    let mut i = (valid_first - 1) | 1;
                    while i >= 0 {
                        result |= (element(first, i, control) >= v
                            && element(first, i - 1, control) <= v)
                            as i32;
                        i -= 2;
                    }
                }
            }
            2 => {
                let larger = valid_first.max(valid_second);
                let smaller = valid_first.min(valid_second);
                result = (1 << (upper - larger)) - 1;
                result <<= larger - smaller;
                for i in (0..=smaller).rev() {
                    result <<= 1;
                    result |= (element(second, i, control) == element(first, i, control)) as i32;
                }
            }
            _ => {
                if valid_first == -1 {
                    result = (2 << upper) - 1;
                } else {
                    let start = if valid_second == upper {
                        valid_second
                    } else {
                        valid_second - valid_first
                    };
                    for j in (0..=start).rev() {
                        result <<= 1;
                        let mut all = 1;
                        for i in (0..=(upper - j).min(valid_first)).rev() {
                            all &= (element(second, i + j, control) == element(first, i, control))
                                as i32;
                        }
                        result |= all;
                    }
                }
            }
        }
        match (control >> 4) & 3 {
            1 => result ^= (2 << upper) - 1,
            3 => result ^= (1 << (valid_second + 1)) - 1,
            _ => {}
        }
        let result = result as u32;
        (
            result,
            [
                result != 0,
                valid_second < upper,
                valid_first < upper,
                result & 1 != 0,
            ],
        )
    }

    fn string_alphabet(control: u8) -> Vec<u16> {
        if control & 1 != 0 {
            vec![0, 0, 1, 0x4142, 0x4143, 0x7fff, 0x8000, 0xffff, 0x0100]
        } else {
            vec![0, 0, 1, 0x41, 0x42, 0x43, 0x61, 0x7f, 0x80, 0xff]
        }
    }

    fn random_string(rng: &mut XorShift, control: u8) -> u128 {
        let alphabet = string_alphabet(control);
        let words = control & 1 != 0;
        let mut register = 0u128;
        for k in 0..element_count(words) {
            let pick = alphabet[(rng.next() % alphabet.len() as u64) as usize] as u128;
            register |= pick << (k * if words { 16 } else { 8 });
        }
        register
    }

    fn element_count(words: bool) -> usize {
        if words { 8 } else { 16 }
    }

    /// Index of the first zero element, or the element count.
    fn implicit_length(register: u128, control: u8) -> i32 {
        let count = element_count(control & 1 != 0) as i32;
        (0..count)
            .find(|&k| element(register, k, control) == 0)
            .unwrap_or(count)
    }

    fn flags_of(flags: [bool; 4]) -> u64 {
        [CF, ZF, SF, OF]
            .into_iter()
            .zip(flags)
            .filter(|(_, set)| *set)
            .fold(0, |acc, (flag, _)| acc | flag)
    }

    #[test]
    fn implicit_length_comparisons_match_the_reference_for_every_control_byte() {
        let mut rig = Rig::new();
        let mut rng = XorShift(0x1357_9bdf_2468_ace0);
        for control in 0..0x80u8 {
            for _ in 0..24 {
                let first = random_string(&mut rng, control);
                let second = random_string(&mut rng, control);
                let (la, lb) = (
                    implicit_length(first, control),
                    implicit_length(second, control),
                );
                let (result, flags) = reference_compare(control, first, la, second, lb);
                let words = control & 1 != 0;
                let count = element_count(words) as u32;
                let want_index = if result == 0 {
                    count
                } else if control & 0x40 != 0 {
                    31 - result.leading_zeros()
                } else {
                    result.trailing_zeros()
                };
                let width = if words { 16 } else { 8 };
                let want_mask = if control & 0x40 == 0 {
                    result as u128
                } else {
                    (0..count).fold(0u128, |acc, k| {
                        if result >> k & 1 == 1 {
                            acc | (((1u128 << width) - 1) << (k * width))
                        } else {
                            acc
                        }
                    })
                };
                for memory in [false, true] {
                    rig.h.cpu.xmm[0] = first;
                    rig.h.cpu.xmm[1] = second;
                    rig.h.write(DATA, &second.to_le_bytes());
                    rig.h.cpu.gpr[RCX] = u64::MAX;
                    rig.reset();
                    rig.h.cpu.set_eflags_arith(AF | PF);
                    let asm = if memory {
                        "pcmpistri xmm0, [rsi], 0"
                    } else {
                        "pcmpistri xmm0, xmm1, 0"
                    };
                    assert_eq!(exec_imm(&mut rig, asm, control), Stop::Halted);
                    let context =
                        format!("control {control:#x} first {first:#x} second {second:#x}");
                    assert_eq!(rig.gpr(RCX), want_index as u64, "pcmpistri ecx {context}");
                    assert_eq!(rig.flags(), flags_of(flags), "pcmpistri flags {context}");
                    assert_eq!(rig.h.cpu.xmm[0], first, "pcmpistri leaves xmm0");

                    rig.h.cpu.xmm[0] = first;
                    rig.h.cpu.xmm[1] = second;
                    rig.reset();
                    let asm = if memory {
                        "pcmpistrm xmm0, [rsi], 0"
                    } else {
                        "pcmpistrm xmm0, xmm1, 0"
                    };
                    assert_eq!(exec_imm(&mut rig, asm, control), Stop::Halted);
                    assert_eq!(rig.h.cpu.xmm[0], want_mask, "pcmpistrm xmm0 {context}");
                    assert_eq!(rig.flags(), flags_of(flags), "pcmpistrm flags {context}");
                }
            }
        }
    }

    #[test]
    fn explicit_length_comparisons_use_eax_and_edx_saturated_and_signed() {
        let mut rig = Rig::new();
        let mut rng = XorShift(0x0f1e_2d3c_4b5a_6978);
        for control in 0..0x80u8 {
            let count = element_count(control & 1 != 0) as i64;
            for round in 0..16 {
                let first = random_string(&mut rng, control);
                let second = random_string(&mut rng, control);
                // Lengths from empty to past the register size, and negative ones.
                let pick = |rng: &mut XorShift| -> i64 {
                    let magnitude = (rng.next() % (count as u64 + 3)) as i64;
                    if rng.next().is_multiple_of(4) {
                        -magnitude
                    } else {
                        magnitude
                    }
                };
                let (eax, edx) = (pick(&mut rng), pick(&mut rng));
                let (la, lb) = (
                    eax.unsigned_abs().min(count as u64) as i32,
                    edx.unsigned_abs().min(count as u64) as i32,
                );
                let (result, flags) = reference_compare(control, first, la, second, lb);
                let want_index = if result == 0 {
                    count as u32
                } else if control & 0x40 != 0 {
                    31 - result.leading_zeros()
                } else {
                    result.trailing_zeros()
                };
                rig.h.cpu.xmm[0] = first;
                rig.h.cpu.xmm[1] = second;
                rig.h.cpu.gpr[RAX] = eax as u32 as u64 | 0xdead_0000_0000;
                rig.h.cpu.gpr[RDX] = edx as u32 as u64 | 0xbeef_0000_0000;
                rig.h.cpu.gpr[RCX] = u64::MAX;
                rig.reset();
                assert_eq!(
                    exec_imm(&mut rig, "pcmpestri xmm0, xmm1, 0", control),
                    Stop::Halted
                );
                let context = format!("control {control:#x} eax {eax} edx {edx} round {round}");
                assert_eq!(rig.gpr(RCX), want_index as u64, "pcmpestri {context}");
                assert_eq!(rig.flags(), flags_of(flags), "pcmpestri flags {context}");
                // The mask form must agree with the index form on the same operands.
                rig.h.cpu.xmm[0] = first;
                rig.h.cpu.gpr[RAX] = eax as u32 as u64;
                rig.h.cpu.gpr[RDX] = edx as u32 as u64;
                rig.reset();
                assert_eq!(
                    exec_imm(&mut rig, "pcmpestrm xmm0, xmm1, 0", control),
                    Stop::Halted
                );
                if control & 0x40 == 0 {
                    assert_eq!(rig.h.cpu.xmm[0], result as u128, "pcmpestrm {context}");
                } else {
                    let nonzero_elements = (0..count)
                        .filter(|&k| {
                            let width = if control & 1 != 0 { 16 } else { 8 };
                            (rig.h.cpu.xmm[0] >> (k * width)) & ((1u128 << width) - 1) != 0
                        })
                        .count();
                    assert_eq!(nonzero_elements as u32, result.count_ones(), "{context}");
                }
            }
        }
    }

    #[test]
    fn rex_w_widens_the_explicit_lengths_to_rax_and_rdx() {
        let mut rig = Rig::new();
        // With 32-bit lengths 0x1_0000_0003 is 3, so the differing last byte is never reached;
        // with REX.W it is huge and saturates to 16.
        let first = lanes8(*b"abcdefghijklmnop");
        let second = lanes8(*b"abcdefghijklmnoX");
        rig.h.cpu.xmm[0] = first;
        rig.h.cpu.xmm[1] = second;
        rig.h.cpu.gpr[RAX] = 0x1_0000_0003;
        rig.h.cpu.gpr[RDX] = 0x1_0000_0003;
        rig.reset();
        // equal each, negative polarity: the index of the first difference
        assert_eq!(
            exec_imm(&mut rig, "pcmpestri xmm0, xmm1, 0", 0x18),
            Stop::Halted
        );
        assert_eq!(
            rig.gpr(RCX),
            16,
            "three valid equal elements, the rest compare as ended"
        );
        rig.reset();
        assert_eq!(
            exec_imm(
                &mut rig,
                ".byte 0x66, 0x48, 0x0f, 0x3a, 0x61, 0xc1, 0x00",
                0x18
            ),
            Stop::Halted
        );
        assert_eq!(rig.gpr(RCX), 15);
        // A negative 64-bit length is its absolute value; as 32 bits the same pattern is huge.
        rig.h.cpu.gpr[RAX] = (-4i64) as u64;
        rig.h.cpu.gpr[RDX] = (-4i64) as u64;
        rig.reset();
        assert_eq!(
            exec_imm(
                &mut rig,
                ".byte 0x66, 0x48, 0x0f, 0x3a, 0x61, 0xc1, 0x00",
                0x18
            ),
            Stop::Halted
        );
        assert_eq!(rig.gpr(RCX), 16, "four equal elements, then both ended");
    }

    fn pad(text: &[u8]) -> u128 {
        let mut bytes = [0u8; 16];
        bytes[..text.len()].copy_from_slice(text);
        lanes8(bytes)
    }

    fn index_form(rig: &mut Rig, control: u8, first: &[u8], second: &[u8]) -> (u64, u64) {
        rig.h.cpu.xmm[0] = pad(first);
        rig.h.cpu.xmm[1] = pad(second);
        rig.reset();
        assert_eq!(
            exec_imm(rig, "pcmpistri xmm0, xmm1, 0", control),
            Stop::Halted
        );
        (rig.gpr(RCX), rig.flags())
    }

    #[test]
    fn equal_any_finds_the_first_character_of_a_set_like_strpbrk() {
        let mut rig = Rig::new();
        // unsigned bytes, equal any, positive polarity, least significant index
        let (index, flags) = index_form(&mut rig, 0x00, b"aeiou", b"rhythm and blues");
        assert_eq!(index, 7, "the 'a' of 'and'");
        assert_eq!(
            flags,
            CF | SF,
            "found; the set is shorter than a register, the text is not"
        );
        let (index, flags) = index_form(&mut rig, 0x00, b"aeiou", b"rhythm");
        assert_eq!(index, 16, "no match gives the element count");
        assert_eq!(
            flags,
            ZF | SF,
            "both strings ended inside the register, nothing found"
        );
        let (index, flags) = index_form(&mut rig, 0x00, b"xyz", b"zebra");
        assert_eq!(index, 0);
        assert_eq!(flags, CF | ZF | SF | OF);
        // most significant index
        let (index, _) = index_form(&mut rig, 0x40, b"aeiou", b"education");
        assert_eq!(index, 7, "the last vowel of 'education' is the 'o' at 7");
    }

    #[test]
    fn equal_each_compares_two_strings_like_strcmp() {
        let mut rig = Rig::new();
        // unsigned bytes, equal each, negative polarity: the index of the first difference
        let (index, flags) = index_form(&mut rig, 0x18, b"kernel", b"kernal");
        assert_eq!(index, 4);
        assert_ne!(flags & CF, 0);
        let (index, flags) = index_form(&mut rig, 0x18, b"kernel", b"kernel");
        assert_eq!(index, 16, "equal strings differ nowhere");
        assert_eq!(flags, ZF | SF, "both ended");
        let (index, _) = index_form(&mut rig, 0x18, b"kernel", b"kern");
        assert_eq!(
            index, 4,
            "a shorter second string differs at its terminator"
        );
        let (index, _) = index_form(&mut rig, 0x08, b"abc", b"xbz");
        assert_eq!(index, 1, "positive polarity: the first equal element");
    }

    #[test]
    fn ranges_match_characters_between_bounds_pairs() {
        let mut rig = Rig::new();
        // unsigned bytes, ranges, positive: first character within 'a'..='z' or '0'..='9'
        let (index, _) = index_form(&mut rig, 0x04, b"az09", b"__-9x");
        assert_eq!(index, 3);
        // negative polarity: first character outside the ranges
        let (index, _) = index_form(&mut rig, 0x14, b"azAZ", b"abc4ef");
        assert_eq!(index, 3);
        // an odd trailing bound has no partner and never matches, even for a value above it
        let (index, flags) = index_form(&mut rig, 0x04, b"az5", b"9");
        assert_eq!(index, 16);
        assert_eq!(flags & CF, 0);
        // Signedness matters: the pair 0xf0..=0x10 is empty as unsigned bytes but -16..=16 as
        // signed ones, and contains 5.
        let (index, _) = index_form(&mut rig, 0x06, &[0xf0, 0x10], &[0x05]);
        assert_eq!(index, 0);
        let (index, _) = index_form(&mut rig, 0x04, &[0xf0, 0x10], &[0x05]);
        assert_eq!(index, 16);
    }

    #[test]
    fn equal_ordered_finds_a_substring_like_strstr() {
        let mut rig = Rig::new();
        // unsigned bytes, equal ordered, positive, least significant index
        let (index, flags) = index_form(&mut rig, 0x0c, b"lo", b"hello world");
        assert_eq!(index, 3);
        assert_ne!(flags & CF, 0);
        let (index, flags) = index_form(&mut rig, 0x0c, b"xyz", b"hello world");
        assert_eq!(index, 16);
        assert_eq!(flags & CF, 0);
        // A needle that starts to match at the very end of a full register still reports there.
        let (index, _) = index_form(&mut rig, 0x0c, b"pq", b"0123456789abcdep");
        assert_eq!(index, 15);
        // ...but not when the haystack ended inside the register.
        let (index, _) = index_form(&mut rig, 0x0c, b"pq", b"0123456789p");
        assert_eq!(index, 16, "the terminator after 'p' cannot be part of 'pq'");
        // An empty needle matches at once.
        let (index, _) = index_form(&mut rig, 0x0c, b"", b"anything");
        assert_eq!(index, 0);
    }

    #[test]
    fn word_and_signed_element_formats() {
        let mut rig = Rig::new();
        // unsigned words, equal any: 0x4142 (two ASCII letters) is found at word 2
        rig.h.cpu.xmm[0] = lanes16([0x4142, 0x9999, 0, 0, 0, 0, 0, 0]);
        rig.h.cpu.xmm[1] = lanes16([0x1111, 0x2222, 0x4142, 0x3333, 0, 0, 0, 0]);
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistri xmm0, xmm1, 0", 0x01),
            Stop::Halted
        );
        assert_eq!(rig.gpr(RCX), 2);
        // Seen as bytes the set is 0x42 0x41 0x99 0x99, and the first member shows up at byte 4.
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistri xmm0, xmm1, 0", 0x00),
            Stop::Halted
        );
        assert_eq!(rig.gpr(RCX), 4);
        // signed words: 0xffff is -1, below 1: ranges [-5, 5] contain it, unsigned ranges do not
        rig.h.cpu.xmm[0] = lanes16([0xfffb, 5, 0, 0, 0, 0, 0, 0]);
        rig.h.cpu.xmm[1] = lanes16([0xffff, 0, 0, 0, 0, 0, 0, 0]);
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistri xmm0, xmm1, 0", 0b0000_0111),
            Stop::Halted
        );
        assert_eq!(rig.gpr(RCX), 0, "-1 lies within -5..=5");
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistri xmm0, xmm1, 0", 0b0000_0101),
            Stop::Halted
        );
        assert_eq!(rig.gpr(RCX), 8, "0xffff is not within 0xfffb..=5 unsigned");
    }

    #[test]
    fn masked_polarity_only_inverts_inside_the_second_string() {
        let mut rig = Rig::new();
        // equal any with polarity 2 (masked positive) and 3 (masked negative)
        rig.h.cpu.xmm[0] = pad(b"ab");
        rig.h.cpu.xmm[1] = pad(b"axbx");
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistrm xmm0, xmm1, 0", 0x20),
            Stop::Halted
        );
        assert_eq!(
            rig.h.cpu.xmm[0], 0b0101,
            "masked positive leaves the result as is"
        );
        rig.h.cpu.xmm[0] = pad(b"ab");
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistrm xmm0, xmm1, 0", 0x30),
            Stop::Halted
        );
        assert_eq!(
            rig.h.cpu.xmm[0], 0b1010,
            "masked negative inverts only the four valid bits"
        );
        rig.h.cpu.xmm[0] = pad(b"ab");
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistrm xmm0, xmm1, 0", 0x10),
            Stop::Halted
        );
        assert_eq!(
            rig.h.cpu.xmm[0],
            0xffff & !0b0101,
            "negative inverts every position"
        );
        // the element-mask output form
        rig.h.cpu.xmm[0] = pad(b"ab");
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pcmpistrm xmm0, xmm1, 0", 0x40),
            Stop::Halted
        );
        assert_eq!(
            rig.h.cpu.xmm[0],
            lanes8([0xff, 0, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        );
    }

    #[test]
    fn string_compare_memory_operand_must_be_aligned() {
        let mut rig = Rig::new();
        for asm in ["pcmpistri xmm0, [rsi], 0", "pcmpestrm xmm0, [rsi], 0"] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 1;
            assert_eq!(exec_imm(&mut rig, asm, 0), gp0(), "{asm}");
        }
    }

    #[test]
    fn enable_rules_apply() {
        let mut rig = Rig::new();
        for asm in [
            "pcmpgtq xmm0, xmm1",
            "pcmpistri xmm0, xmm1, 0",
            "pcmpestrm xmm0, xmm1, 0",
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
            "pcmpgtq xmm0, xmm1",
            "pcmpistri xmm0, xmm1, 0",
            "pcmpistrm xmm0, [rsi], 0",
            "pcmpestri xmm0, xmm1, 0",
            "pcmpestrm xmm0, xmm1, 0",
            "crc32 eax, bl",
            "crc32 rax, qword ptr [rsi]",
            "crc32 eax, bx",
        ] {
            assert_jit_matches_interpreter(code(asm), |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                h.cpu.gpr[RSI] = DATA;
                h.cpu.gpr[RAX] = 5;
                h.cpu.gpr[RDX] = 9;
                h.cpu.gpr[RBX] = 0x0123_4567_89ab_cdef;
                h.cpu.xmm[0] = pad(b"needle");
                h.cpu.xmm[1] = pad(b"a needle in hay");
                h.write(DATA, &h.cpu.xmm[1].to_le_bytes());
            });
        }
    }
}

mod crypto_instructions {
    use super::*;

    /// A block written the way FIPS-197 prints it: byte 0 first.
    fn block(hex: &str) -> u128 {
        let bytes: Vec<u8> = (0..hex.len() / 2)
            .map(|k| u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap())
            .collect();
        u128::from_le_bytes(bytes.try_into().unwrap())
    }

    fn exec_imm(rig: &mut Rig, asm: &str, imm: u8) -> Stop {
        let mut bytes = code(asm).to_vec();
        let last = bytes.len() - 2;
        bytes[last] = imm;
        rig.exec_bytes(&bytes)
    }

    /// AES-128 key expansion built from `aeskeygenassist`, `pshufd`, `pslldq` and `pxor` executed
    /// by the emulator, the way the Linux and OpenSSL AES-NI code does it.
    fn expand_key(rig: &mut Rig, key: u128) -> [u128; 11] {
        const ROUND_CONSTANTS: [u8; 10] =
            [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];
        let mut keys = [0u128; 11];
        keys[0] = key;
        rig.h.cpu.xmm[1] = key;
        rig.reset();
        for (round, &constant) in ROUND_CONSTANTS.iter().enumerate() {
            assert_eq!(
                exec_imm(rig, "aeskeygenassist xmm2, xmm1, 0", constant),
                Stop::Halted
            );
            assert_eq!(rig.exec("pshufd xmm2, xmm2, 255"), Stop::Halted);
            assert_eq!(
                rig.exec(
                    "movdqa xmm3, xmm1; pslldq xmm3, 4; pxor xmm1, xmm3; pslldq xmm3, 4; pxor xmm1, xmm3; pslldq xmm3, 4; pxor xmm1, xmm3; pxor xmm1, xmm2"
                ),
                Stop::Halted
            );
            keys[round + 1] = rig.h.cpu.xmm[1];
        }
        keys
    }

    fn encrypt(rig: &mut Rig, keys: &[u128; 11], plaintext: u128) -> u128 {
        rig.h.cpu.xmm[0] = plaintext;
        rig.h.cpu.xmm[4] = keys[0];
        rig.reset();
        assert_eq!(rig.exec("pxor xmm0, xmm4"), Stop::Halted);
        for key in &keys[1..10] {
            rig.h.cpu.xmm[4] = *key;
            assert_eq!(rig.exec("aesenc xmm0, xmm4"), Stop::Halted);
        }
        rig.h.cpu.xmm[4] = keys[10];
        assert_eq!(rig.exec("aesenclast xmm0, xmm4"), Stop::Halted);
        rig.h.cpu.xmm[0]
    }

    /// The equivalent inverse cipher: the middle round keys go through aesimc.
    fn decrypt(rig: &mut Rig, keys: &[u128; 11], ciphertext: u128) -> u128 {
        rig.h.cpu.xmm[0] = ciphertext;
        rig.h.cpu.xmm[4] = keys[10];
        rig.reset();
        assert_eq!(rig.exec("pxor xmm0, xmm4"), Stop::Halted);
        for key in keys[1..10].iter().rev() {
            rig.h.cpu.xmm[4] = *key;
            assert_eq!(rig.exec("aesimc xmm5, xmm4"), Stop::Halted);
            rig.h.cpu.xmm[4] = rig.h.cpu.xmm[5];
            assert_eq!(rig.exec("aesdec xmm0, xmm4"), Stop::Halted);
        }
        rig.h.cpu.xmm[4] = keys[0];
        assert_eq!(rig.exec("aesdeclast xmm0, xmm4"), Stop::Halted);
        rig.h.cpu.xmm[0]
    }

    #[test]
    fn aeskeygenassist_matches_the_fips_197_key_expansion_step() {
        let mut rig = Rig::new();
        // Appendix A.1: w3 = 09cf4f3c gives temp = SubWord(RotWord(w3)) ^ Rcon[1] = 8b84eb01.
        let key = block("2b7e151628aed2a6abf7158809cf4f3c");
        rig.h.cpu.xmm[1] = key;
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "aeskeygenassist xmm0, xmm1, 0", 0x01),
            Stop::Halted
        );
        let result = rig.h.cpu.xmm[0].to_le_bytes();
        assert_eq!(
            result[12..16],
            [0x8b, 0x84, 0xeb, 0x01],
            "RotWord(SubWord(X3)) ^ rcon"
        );
        assert_eq!(result[8..12], [0x01, 0x8a, 0x84, 0xeb], "SubWord(X3)");
        // X1 = 28aed2a6 under the S-box is 34e4b524, and RotWord moves the first byte last; the
        // round constant is XORed into the first byte of the result.
        assert_eq!(result[0..4], [0x34, 0xe4, 0xb5, 0x24], "SubWord(X1)");
        assert_eq!(
            result[4..8],
            [0xe4 ^ 0x01, 0xb5, 0x24, 0x34],
            "RotWord(SubWord(X1)) ^ rcon"
        );
        // The memory form reads an aligned m128 and ignores the old destination.
        rig.h.cpu.xmm[0] = u128::MAX;
        rig.h.write(DATA, &key.to_le_bytes());
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "aeskeygenassist xmm0, [rsi], 0", 0x01),
            Stop::Halted
        );
        assert_eq!(rig.h.cpu.xmm[0].to_le_bytes(), result);
    }

    #[test]
    fn one_aesenc_round_matches_the_fips_197_appendix_b_walkthrough() {
        let mut rig = Rig::new();
        // Round 1: start of round, then SubBytes, ShiftRows, MixColumns and the round key.
        let state = block("193de3bea0f4e22b9ac68d2ae9f84808");
        let round_key = block("a0fafe1788542cb123a339392a6c7605");
        assert_eq!(
            rig.op("aesenc xmm0, xmm1", state, round_key),
            block("a49c7ff2689f352b6b5bea43026a5049")
        );
        assert_eq!(
            rig.op_mem("aesenc xmm0, [rsi]", state, round_key),
            block("a49c7ff2689f352b6b5bea43026a5049")
        );
        // With a zero key the result is exactly MixColumns(ShiftRows(SubBytes(state))).
        assert_eq!(
            rig.op("aesenc xmm0, xmm1", state, 0),
            block("046681e5e0cb199a48f8d37a2806264c")
        );
        // aesenclast skips MixColumns: ShiftRows(SubBytes(state)).
        assert_eq!(
            rig.op("aesenclast xmm0, xmm1", state, 0),
            block("d4bf5d30e0b452aeb84111f11e2798e5")
        );
        assert_eq!(
            rig.op_mem("aesenclast xmm0, [rsi]", state, 0),
            block("d4bf5d30e0b452aeb84111f11e2798e5")
        );
    }

    #[test]
    fn aesdeclast_and_aesimc_invert_the_fips_197_round_steps() {
        let mut rig = Rig::new();
        // The same Appendix B round: InvShiftRows then InvSubBytes undoes ShiftRows/SubBytes, and
        // InvMixColumns undoes MixColumns.
        let shifted = block("d4bf5d30e0b452aeb84111f11e2798e5");
        let mixed = block("046681e5e0cb199a48f8d37a2806264c");
        let round_input = block("193de3bea0f4e22b9ac68d2ae9f84808");
        assert_eq!(rig.op("aesdeclast xmm0, xmm1", shifted, 0), round_input);
        assert_eq!(
            rig.op_mem("aesdeclast xmm0, [rsi]", shifted, 0),
            round_input
        );
        assert_eq!(rig.op("aesimc xmm0, xmm1", !0, mixed), shifted);
        assert_eq!(rig.op_mem("aesimc xmm0, [rsi]", !0, mixed), shifted);
    }

    #[test]
    fn aesimc_undoes_the_mixcolumns_that_aesenc_applies() {
        let mut rig = Rig::new();
        // aesdeclast(y, 0) = InvSub(InvShift(y)), so aesenc(that, 0) = MixColumns(y); aesimc must
        // bring back y. This holds for arbitrary y, which pins InvMixColumns against MixColumns
        // and the inverse S-box against the S-box.
        for &y in &vectors() {
            let pre_image = rig.op("aesdeclast xmm0, xmm1", y, 0);
            let mixed = rig.op("aesenc xmm0, xmm1", pre_image, 0);
            assert_eq!(rig.op("aesimc xmm0, xmm1", 0, mixed), y, "{y:#x}");
        }
    }

    #[test]
    fn aes_128_matches_the_fips_197_vectors() {
        let mut rig = Rig::new();
        // Appendix B.
        let keys = expand_key(&mut rig, block("2b7e151628aed2a6abf7158809cf4f3c"));
        assert_eq!(keys[1], block("a0fafe1788542cb123a339392a6c7605"));
        assert_eq!(keys[10], block("d014f9a8c9ee2589e13f0cc8b6630ca6"));
        let plaintext = block("3243f6a8885a308d313198a2e0370734");
        let ciphertext = block("3925841d02dc09fbdc118597196a0b32");
        assert_eq!(encrypt(&mut rig, &keys, plaintext), ciphertext);
        assert_eq!(decrypt(&mut rig, &keys, ciphertext), plaintext);
        // Appendix C.1.
        let keys = expand_key(&mut rig, block("000102030405060708090a0b0c0d0e0f"));
        let plaintext = block("00112233445566778899aabbccddeeff");
        let ciphertext = block("69c4e0d86a7b0430d8cdb78070b4c55a");
        assert_eq!(encrypt(&mut rig, &keys, plaintext), ciphertext);
        assert_eq!(decrypt(&mut rig, &keys, ciphertext), plaintext);
    }

    #[test]
    fn aes_memory_operands_must_be_aligned() {
        let mut rig = Rig::new();
        for asm in [
            "aesenc xmm0, [rsi]",
            "aesenclast xmm0, [rsi]",
            "aesdec xmm0, [rsi]",
            "aesdeclast xmm0, [rsi]",
            "aesimc xmm0, [rsi]",
            "pclmulqdq xmm0, [rsi], 0",
            "aeskeygenassist xmm0, [rsi], 0",
        ] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 8;
            assert_eq!(rig.exec(asm), gp0(), "{asm}");
        }
    }

    /// Carry-less product by polynomial convolution over bit arrays: a formulation that shares no
    /// code with the shift-and-xor loop in the implementation.
    fn reference_product(a: u64, b: u64) -> u128 {
        let mut coefficients = [false; 128];
        for i in 0..64 {
            for j in 0..64 {
                if (a >> i) & 1 != 0 && (b >> j) & 1 != 0 {
                    coefficients[i + j] ^= true;
                }
            }
        }
        coefficients
            .iter()
            .enumerate()
            .fold(0u128, |acc, (k, &set)| acc | ((set as u128) << k))
    }

    #[test]
    fn pclmulqdq_is_a_carry_less_multiply_of_the_selected_quadwords() {
        let mut rig = Rig::new();
        let all = vectors();
        for imm in [0x00u8, 0x01, 0x10, 0x11] {
            for (index, &dst) in all.iter().enumerate() {
                let src = all[(index * 3 + 1) % all.len()];
                let a = qwords_of(dst)[(imm & 1) as usize];
                let b = qwords_of(src)[((imm >> 4) & 1) as usize];
                rig.h.cpu.xmm[0] = dst;
                rig.h.cpu.xmm[1] = src;
                rig.reset();
                assert_eq!(
                    exec_imm(&mut rig, "pclmulqdq xmm0, xmm1, 0", imm),
                    Stop::Halted
                );
                assert_eq!(
                    rig.h.cpu.xmm[0],
                    reference_product(a, b),
                    "imm {imm:#x} {dst:#x} {src:#x}"
                );
            }
        }
    }

    #[test]
    fn pclmulqdq_known_products_and_properties() {
        let mut rig = Rig::new();
        let product = |rig: &mut Rig, a: u64, b: u64| -> u128 {
            rig.h.cpu.xmm[0] = a as u128;
            rig.h.cpu.xmm[1] = b as u128;
            rig.reset();
            assert_eq!(exec_imm(rig, "pclmulqdq xmm0, xmm1, 0", 0), Stop::Halted);
            rig.h.cpu.xmm[0]
        };
        assert_eq!(product(&mut rig, 0x1234, 1), 0x1234);
        assert_eq!(product(&mut rig, 0x1234, 2), 0x2468);
        assert_eq!(product(&mut rig, 3, 3), 5, "(x+1)^2 = x^2 + 1");
        assert_eq!(product(&mut rig, 0, u64::MAX), 0);
        // Squaring spreads the bits: no carries, so all-ones squares to 0x5555...
        assert_eq!(
            product(&mut rig, u64::MAX, u64::MAX),
            0x5555_5555_5555_5555_5555_5555_5555_5555
        );
        assert_eq!(product(&mut rig, 1 << 63, 1 << 63), 1 << 126);
        // (x^4 + x^3 + x + 1) * x^63 = x^67 + x^66 + x^64 + x^63
        assert_eq!(
            product(&mut rig, 0x1b, 1 << 63),
            (1u128 << 67) ^ (1 << 66) ^ (1 << 64) ^ (1 << 63)
        );
        // Immediate bits other than 0 and 4 are ignored.
        rig.h.cpu.xmm[0] = lanes64([5, 9]);
        rig.h.cpu.xmm[1] = lanes64([7, 11]);
        rig.reset();
        assert_eq!(
            exec_imm(&mut rig, "pclmulqdq xmm0, xmm1, 0", 0xee),
            Stop::Halted
        );
        // 0b101 * 0b111 = 0b111 ^ (0b111 << 2) = 0b11011
        assert_eq!(rig.h.cpu.xmm[0], 0b11011, "low quadword of both operands");
    }

    #[test]
    fn enable_rules_apply() {
        let mut rig = Rig::new();
        for asm in [
            "aesenc xmm0, xmm1",
            "aesimc xmm0, xmm1",
            "aeskeygenassist xmm0, xmm1, 0",
            "pclmulqdq xmm0, xmm1, 0",
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
            "aesenc xmm0, xmm1",
            "aesenclast xmm0, [rsi]",
            "aesdec xmm0, xmm1",
            "aesdeclast xmm0, xmm1",
            "aesimc xmm0, xmm1",
            "aeskeygenassist xmm0, xmm1, 0",
            "pclmulqdq xmm0, xmm1, 0",
        ] {
            assert_jit_matches_interpreter(code(asm), |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                h.cpu.gpr[RSI] = DATA;
                h.cpu.xmm[0] = block("00112233445566778899aabbccddeeff");
                h.cpu.xmm[1] = block("0f1571c947d9e8590cb7add6af7f6798");
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
    ("pcmpgtq xmm0, xmm1", &[0x66, 0x0f, 0x38, 0x37, 0xc1, 0xf4]),
    ("pcmpgtq xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0x37, 0x06, 0xf4]),
    (
        "pcmpistri xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x63, 0xc1, 0x00, 0xf4],
    ),
    (
        "pcmpistri xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x63, 0x06, 0x00, 0xf4],
    ),
    (
        "pcmpistrm xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x62, 0xc1, 0x00, 0xf4],
    ),
    (
        "pcmpistrm xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x62, 0x06, 0x00, 0xf4],
    ),
    (
        "pcmpestri xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x61, 0xc1, 0x00, 0xf4],
    ),
    (
        "pcmpestri xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x61, 0x06, 0x00, 0xf4],
    ),
    (
        "pcmpestrm xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x60, 0xc1, 0x00, 0xf4],
    ),
    (
        "pcmpestrm xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x60, 0x06, 0x00, 0xf4],
    ),
    (
        ".byte 0x66, 0x48, 0x0f, 0x3a, 0x61, 0xc1, 0x00",
        &[0x66, 0x48, 0x0f, 0x3a, 0x61, 0xc1, 0x00, 0xf4],
    ),
    (
        ".byte 0x66, 0x48, 0x0f, 0x3a, 0x60, 0xc1, 0x00",
        &[0x66, 0x48, 0x0f, 0x3a, 0x60, 0xc1, 0x00, 0xf4],
    ),
    ("crc32 eax, bl", &[0xf2, 0x0f, 0x38, 0xf0, 0xc3, 0xf4]),
    ("crc32 eax, bh", &[0xf2, 0x0f, 0x38, 0xf0, 0xc7, 0xf4]),
    ("crc32 eax, bx", &[0x66, 0xf2, 0x0f, 0x38, 0xf1, 0xc3, 0xf4]),
    ("crc32 eax, ebx", &[0xf2, 0x0f, 0x38, 0xf1, 0xc3, 0xf4]),
    (
        "crc32 rax, rbx",
        &[0xf2, 0x48, 0x0f, 0x38, 0xf1, 0xc3, 0xf4],
    ),
    ("crc32 rax, bl", &[0xf2, 0x48, 0x0f, 0x38, 0xf0, 0xc3, 0xf4]),
    (
        "crc32 r9d, r10b",
        &[0xf2, 0x45, 0x0f, 0x38, 0xf0, 0xca, 0xf4],
    ),
    ("crc32 r9, r10", &[0xf2, 0x4d, 0x0f, 0x38, 0xf1, 0xca, 0xf4]),
    (
        "crc32 eax, byte ptr [rsi]",
        &[0xf2, 0x0f, 0x38, 0xf0, 0x06, 0xf4],
    ),
    (
        "crc32 eax, word ptr [rsi]",
        &[0x66, 0xf2, 0x0f, 0x38, 0xf1, 0x06, 0xf4],
    ),
    (
        "crc32 eax, dword ptr [rsi]",
        &[0xf2, 0x0f, 0x38, 0xf1, 0x06, 0xf4],
    ),
    (
        "crc32 rax, qword ptr [rsi]",
        &[0xf2, 0x48, 0x0f, 0x38, 0xf1, 0x06, 0xf4],
    ),
    (
        "crc32 rax, byte ptr [rsi]",
        &[0xf2, 0x48, 0x0f, 0x38, 0xf0, 0x06, 0xf4],
    ),
    (
        ".byte 0xf3, 0x0f, 0x38, 0xf0, 0xc3",
        &[0xf3, 0x0f, 0x38, 0xf0, 0xc3, 0xf4],
    ),
    (
        ".byte 0x66, 0x0f, 0x38, 0xf0, 0xc3",
        &[0x66, 0x0f, 0x38, 0xf0, 0xc3, 0xf4],
    ),
    ("aesenc xmm0, xmm1", &[0x66, 0x0f, 0x38, 0xdc, 0xc1, 0xf4]),
    ("aesenc xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0xdc, 0x06, 0xf4]),
    (
        "aesenclast xmm0, xmm1",
        &[0x66, 0x0f, 0x38, 0xdd, 0xc1, 0xf4],
    ),
    (
        "aesenclast xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0xdd, 0x06, 0xf4],
    ),
    ("aesdec xmm0, xmm1", &[0x66, 0x0f, 0x38, 0xde, 0xc1, 0xf4]),
    ("aesdec xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0xde, 0x06, 0xf4]),
    (
        "aesdeclast xmm0, xmm1",
        &[0x66, 0x0f, 0x38, 0xdf, 0xc1, 0xf4],
    ),
    (
        "aesdeclast xmm0, [rsi]",
        &[0x66, 0x0f, 0x38, 0xdf, 0x06, 0xf4],
    ),
    ("aesimc xmm0, xmm1", &[0x66, 0x0f, 0x38, 0xdb, 0xc1, 0xf4]),
    ("aesimc xmm0, [rsi]", &[0x66, 0x0f, 0x38, 0xdb, 0x06, 0xf4]),
    (
        "aeskeygenassist xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0xdf, 0xc1, 0x00, 0xf4],
    ),
    (
        "aeskeygenassist xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0xdf, 0x06, 0x00, 0xf4],
    ),
    (
        "pclmulqdq xmm0, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0x44, 0xc1, 0x00, 0xf4],
    ),
    (
        "pclmulqdq xmm0, [rsi], 0",
        &[0x66, 0x0f, 0x3a, 0x44, 0x06, 0x00, 0xf4],
    ),
    (
        "aeskeygenassist xmm2, xmm1, 0",
        &[0x66, 0x0f, 0x3a, 0xdf, 0xd1, 0x00, 0xf4],
    ),
    (
        "pshufd xmm2, xmm2, 255",
        &[0x66, 0x0f, 0x70, 0xd2, 0xff, 0xf4],
    ),
    (
        "movdqa xmm3, xmm1; pslldq xmm3, 4; pxor xmm1, xmm3; pslldq xmm3, 4; pxor xmm1, xmm3; pslldq xmm3, 4; pxor xmm1, xmm3; pxor xmm1, xmm2",
        &[
            0x66, 0x0f, 0x6f, 0xd9, 0x66, 0x0f, 0x73, 0xfb, 0x04, 0x66, 0x0f, 0xef, 0xcb, 0x66,
            0x0f, 0x73, 0xfb, 0x04, 0x66, 0x0f, 0xef, 0xcb, 0x66, 0x0f, 0x73, 0xfb, 0x04, 0x66,
            0x0f, 0xef, 0xcb, 0x66, 0x0f, 0xef, 0xca, 0xf4,
        ],
    ),
    ("pxor xmm0, xmm4", &[0x66, 0x0f, 0xef, 0xc4, 0xf4]),
    ("aesenc xmm0, xmm4", &[0x66, 0x0f, 0x38, 0xdc, 0xc4, 0xf4]),
    (
        "aesenclast xmm0, xmm4",
        &[0x66, 0x0f, 0x38, 0xdd, 0xc4, 0xf4],
    ),
    ("aesimc xmm5, xmm4", &[0x66, 0x0f, 0x38, 0xdb, 0xec, 0xf4]),
    ("aesdec xmm0, xmm4", &[0x66, 0x0f, 0x38, 0xde, 0xc4, 0xf4]),
    (
        "aesdeclast xmm0, xmm4",
        &[0x66, 0x0f, 0x38, 0xdf, 0xc4, 0xf4],
    ),
    ("aesimc xmm2, xmm1", &[0x66, 0x0f, 0x38, 0xdb, 0xd1, 0xf4]),
    // SNIPPETS-END
];
