pub(crate) use super::emit_x64::Addrs;
use super::emit_x64::{FRAME, Gen};
use super::helpers;
use super::x64::*;
use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

pub(crate) struct Prologue {
    pub code: Vec<u8>,
    pub exit_ok: u64,
    pub exit_fail: u64,
}

// Saved in this order; RSI/RDI are callee-saved on Win64 and used as scratch by blocks.
const SAVED: [u8; 8] = [RBX, RBP, RSI, RDI, R12, R13, R14, R15];

pub(crate) fn prologue(base: u64) -> Prologue {
    let mut a = Asm::new(base);
    for r in SAVED {
        a.push(r);
    }
    a.alu_ri(SUB, RSP, FRAME, true);
    a.mov_rr(RBX, ARGS[0], true);
    a.ld(RBP, mem(RBX, std::mem::offset_of!(Cpu, tlb_cur) as i32));
    a.jmp_reg(ARGS[1]);
    // rax = helper status, rcx = index of the failing instruction
    let exit_fail = a.here();
    a.alu_rr(XOR, RDX, RDX, false);
    a.alu_ri(CMP, RAX, 2, true);
    a.sete(RDX);
    a.alu_rr(ADD, RCX, RDX, true);
    a.shift_ri(SHL, RCX, 8, true);
    a.alu_rr(OR, RAX, RCX, true);
    let exit_ok = a.here();
    a.alu_ri(ADD, RSP, FRAME, true);
    for r in SAVED.iter().rev() {
        a.pop(*r);
    }
    a.ret();
    a.finish();
    Prologue {
        code: a.code,
        exit_ok,
        exit_fail,
    }
}

pub(crate) fn addrs(exit_ok: u64, exit_fail: u64) -> Addrs {
    Addrs {
        exec: helpers::h_exec as *const () as usize as u64,
        ld: helpers::h_ld_x as *const () as usize as u64,
        st: helpers::h_st_x as *const () as usize as u64,
        cond: helpers::h_cond as *const () as usize as u64,
        exit_ok,
        exit_fail,
    }
}

pub(crate) fn compile_block(
    addr: u64,
    ad: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
) -> (Vec<u8>, u32) {
    let mut g = Gen::new(addr, ad, insns);
    g.compile(rips, end_rip);
    (g.a.code, g.inlined)
}
