use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

#[cfg(target_arch = "aarch64")]
#[repr(C)]
pub(crate) struct Pair {
    pub val: u64,
    pub status: u64,
}

pub(crate) extern "C" fn h_exec(cpu: &mut Cpu, insn: &Insn, rip: u64) -> u64 {
    cpu.rip = rip;
    match cpu.exec(insn) {
        Ok(()) => {
            if cpu.exit_block {
                2
            } else {
                0
            }
        }
        Err(e) => {
            cpu.pending = Some(e);
            1
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn h_ld(cpu: &mut Cpu, la: u64, size: u64, rip: u64) -> Pair {
    cpu.rip = rip;
    match cpu.read(la, size as u8) {
        Ok(v) => Pair { val: v, status: 0 },
        Err(e) => {
            cpu.pending = Some(e);
            Pair { val: 0, status: 1 }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn h_st(cpu: &mut Cpu, la: u64, size: u64, val: u64, rip: u64) -> u64 {
    cpu.rip = rip;
    match cpu.write(la, size as u8, val) {
        Ok(()) => 0,
        Err(e) => {
            cpu.pending = Some(e);
            1
        }
    }
}

// x86-64 variants: at most four arguments (Win64 passes more on the stack) and no 16-byte
// struct return (Win64 returns it through a hidden pointer). The JIT stores `cpu.rip` itself.
#[cfg(target_arch = "x86_64")]
pub(crate) extern "C" fn h_ld_x(cpu: &mut Cpu, la: u64, size: u64, out: *mut u64) -> u64 {
    match cpu.read(la, size as u8) {
        Ok(v) => {
            unsafe { *out = v };
            0
        }
        Err(e) => {
            cpu.pending = Some(e);
            1
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) extern "C" fn h_st_x(cpu: &mut Cpu, la: u64, size: u64, val: u64) -> u64 {
    match cpu.write(la, size as u8, val) {
        Ok(()) => 0,
        Err(e) => {
            cpu.pending = Some(e);
            1
        }
    }
}

pub(crate) extern "C" fn h_cond(cpu: &mut Cpu, cc: u64) -> u64 {
    cpu.cond(cc as u8) as u64
}

pub(crate) fn is_term(i: &Insn) -> bool {
    match i.op {
        0x70..=0x7f
        | 0xe0..=0xe3
        | 0xe8
        | 0xe9
        | 0xeb
        | 0xc2
        | 0xc3
        | 0xca
        | 0xcb
        | 0xcc
        | 0xcd
        | 0xcf
        | 0xf4 => true,
        0x6c..=0x6f
        | 0xa4..=0xa7
        | 0xaa..=0xaf
        | 0xe4..=0xe7
        | 0xec..=0xef
        | 0xfa
        | 0xfb
        | 0x9d
        | 0x8e
        | 0xc8 => true,
        0xff => matches!(i.reg & 7, 2..=5),
        0x105..=0x107
        | 0x10b
        | 0x180..=0x18f
        | 0x1a1
        | 0x1a9
        | 0x120..=0x123
        | 0x130
        | 0x100
        | 0x101
        | 0xffff => true,
        _ => false,
    }
}
