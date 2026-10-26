use std::collections::HashMap;

use super::arch;
use super::helpers;
use crate::cpu::Cpu;
use crate::cpu::decode::{self, DecodeErr, Insn};
use crate::cpu::exc::Exc;
use crate::cpu::mem::Acc;
use crate::sys::ExecMem;

const CODE_SIZE: usize = 128 << 20;
const MAX_INSNS: usize = 48;
const JC_BITS: usize = 16;

pub(crate) struct Block {
    rip: u64,
    pa: u64,
    code: u64,
    valid: bool,
    _insns: Box<[Insn]>,
}

pub(crate) struct Jit {
    mem: ExecMem,
    used: usize,
    pub enter: u64,
    addrs: arch::Addrs,
    blocks: Vec<Block>,
    jc: Vec<u32>,
    page_blocks: HashMap<u64, Vec<u32>>,
    header: usize,
    pub translated: u64,
    pub inlined: u64,
    pub total: u64,
    pub resets: u64,
}

fn jc_hash(rip: u64) -> usize {
    ((rip ^ (rip >> JC_BITS)) as usize) & ((1 << JC_BITS) - 1)
}

impl Jit {
    pub fn new() -> Option<Jit> {
        if !crate::jit::probe() {
            return None;
        }
        let mem = ExecMem::new(CODE_SIZE)?;
        let base = mem.base();
        let pro = arch::prologue(base);
        mem.write_bytes(0, &pro.code);
        let header = (pro.code.len() + 63) & !63;
        Some(Jit {
            mem,
            used: header,
            enter: base,
            addrs: arch::addrs(pro.exit_ok, pro.exit_fail),
            blocks: Vec::new(),
            jc: vec![0; 1 << JC_BITS],
            page_blocks: HashMap::new(),
            header,
            translated: 0,
            inlined: 0,
            total: 0,
            resets: 0,
        })
    }

    pub fn invalidate_page(&mut self, ppn: u64) {
        if let Some(list) = self.page_blocks.remove(&ppn) {
            for idx in list {
                self.blocks[idx as usize].valid = false;
            }
        }
    }

    fn reset(&mut self) {
        self.blocks.clear();
        self.page_blocks.clear();
        self.jc.iter_mut().for_each(|e| *e = 0);
        self.used = self.header;
        self.resets += 1;
    }

    pub fn lookup(&self, rip: u64, pa: u64) -> Option<u64> {
        let idx = self.jc[jc_hash(rip)];
        if idx == 0 {
            return None;
        }
        let b = &self.blocks[idx as usize - 1];
        (b.valid && b.rip == rip && b.pa == pa).then_some(b.code)
    }

    pub fn translate(&mut self, cpu: &mut Cpu, rip: u64, pa: u64) -> Result<u64, Exc> {
        if self.used + (1 << 20) > self.mem.len() {
            self.reset();
        }
        let ram_len = cpu.ram.len() as u64;
        if pa >= ram_len {
            return Err(cpu.page_fault(rip, Acc::Exec, cpu.cpl == 3, false, false));
        }
        let avail = 4096 - (pa & 0xfff) as usize;
        let mut insns: Vec<Insn> = Vec::new();
        let mut rips: Vec<u64> = Vec::new();
        let mut off = 0usize;
        let mut cross = false;
        let mut end_rip = rip;
        while insns.len() < MAX_INSNS {
            let n = (avail - off).min(15);
            if n == 0 {
                break;
            }
            let a = pa as usize + off;
            match decode::decode(&cpu.ram[a..a + n]) {
                Ok(i) => {
                    rips.push(rip + off as u64);
                    off += i.len as usize;
                    end_rip = rip + off as u64;
                    insns.push(i);
                    if helpers::is_term(insns.last().unwrap()) {
                        break;
                    }
                }
                Err(DecodeErr::Truncated) => {
                    if insns.is_empty() {
                        let i = cpu.fetch_decode()?;
                        rips.push(rip);
                        end_rip = rip + i.len as u64;
                        insns.push(i);
                        cross = true;
                    }
                    break;
                }
                Err(DecodeErr::Invalid) => {
                    let bad = Insn {
                        op: 0xffff,
                        len: 1,
                        ..Default::default()
                    };
                    rips.push(rip + off as u64);
                    end_rip = rip + off as u64 + 1;
                    insns.push(bad);
                    break;
                }
            }
        }
        if insns.is_empty() {
            let i = cpu.fetch_decode()?;
            rips.push(rip);
            end_rip = rip + i.len as u64;
            insns.push(i);
            cross = true;
        }
        cpu.mark_code(pa);
        let mut pages = vec![pa >> 12];
        if cross {
            let pa2 = cpu.translate((rip & !0xfff).wrapping_add(4096), Acc::Exec, false)?;
            pages.push(pa2 >> 12);
        }
        let insns: Box<[Insn]> = insns.into_boxed_slice();
        self.used = (self.used + 15) & !15;
        let code_addr = self.mem.base() + self.used as u64;
        let (code, inlined) = arch::compile_block(code_addr, &self.addrs, &insns, &rips, end_rip);
        self.inlined += inlined as u64;
        self.total += insns.len() as u64;
        self.mem.write_bytes(self.used, &code);
        self.used += code.len();
        let idx = self.blocks.len() as u32;
        self.blocks.push(Block {
            rip,
            pa,
            code: code_addr,
            valid: true,
            _insns: insns,
        });
        for p in pages {
            self.page_blocks.entry(p).or_default().push(idx);
        }
        self.jc[jc_hash(rip)] = idx + 1;
        self.translated += 1;
        Ok(code_addr)
    }
}

impl Cpu {
    pub fn step_block(&mut self) -> Result<(), Exc> {
        let rip = self.rip;
        let pa = self.translate(rip, Acc::Exec, false)?;
        let mut jit = self.jit.take().unwrap();
        let code = match jit.lookup(rip, pa) {
            Some(c) => Ok(c),
            None => jit.translate(self, rip, pa),
        };
        let enter = jit.enter;
        self.jit = Some(jit);
        let code = code?;
        let f: extern "C" fn(*mut Cpu, u64) -> u64 = unsafe { std::mem::transmute(enter as usize) };
        let r = f(self as *mut Cpu, code);
        let done = r >> 8;
        self.last_done = done;
        let prepaid = std::mem::take(&mut self.prepaid);
        self.exit_block = false;
        match r & 0xff {
            1 => {
                self.icount += done - prepaid;
                Err(self.pending.take().expect("pending exception"))
            }
            _ => {
                self.icount += done.saturating_sub(1) - prepaid;
                Ok(())
            }
        }
    }
}
