use super::a64::{self, Asm};
use super::emit::Gen;
use super::helpers;
use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

pub(crate) use super::emit::Addrs;

pub(crate) struct Prologue {
    pub code: Vec<u8>,
    pub exit_ok: u64,
    pub exit_fail: u64,
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub(crate) fn prologue(base: u64) -> Prologue {
    let mut a = Asm::new(base);
    a.stp_pre(29, 30, a64::SP, -80);
    a.stp_off(19, 20, a64::SP, 16);
    a.stp_off(21, 22, a64::SP, 32);
    a.stp_off(23, 24, a64::SP, 48);
    a.mov(19, 0);
    a.ldr(20, 19, std::mem::offset_of!(Cpu, tlb_cur) as u32);
    a.br(1);
    let exit_fail = a.here();
    a.cmp_imm(0, 2, true);
    a.cinc_eq(1, 1);
    a.lsl_imm(1, 1, 8);
    a.orr(0, 0, 1, true);
    let exit_ok = a.here();
    a.ldp_off(23, 24, a64::SP, 48);
    a.ldp_off(21, 22, a64::SP, 32);
    a.ldp_off(19, 20, a64::SP, 16);
    a.ldp_post(29, 30, a64::SP, 80);
    a.ret();
    a.finish();
    Prologue {
        code: bytes(&a.code),
        exit_ok,
        exit_fail,
    }
}

pub(crate) fn addrs(exit_ok: u64, exit_fail: u64) -> Addrs {
    Addrs {
        exec: helpers::h_exec as *const () as usize as u64,
        ld: helpers::h_ld as *const () as usize as u64,
        st: helpers::h_st as *const () as usize as u64,
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
    (bytes(&g.a.code), g.inlined)
}
