use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

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

pub(crate) extern "C" fn h_cond(cpu: &mut Cpu, cc: u64) -> u64 {
    cpu.cond(cc as u8) as u64
}
