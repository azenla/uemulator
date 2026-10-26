use super::flags::*;
use super::{Cpu, R, Seg};
use crate::machine::ExitReason;

#[derive(Debug)]
pub(crate) enum Exc {
    Fault {
        vec: u8,
        err: Option<u32>,
        soft: bool,
    },
    Exit(ExitReason),
}

impl Exc {
    pub fn fault(vec: u8, err: Option<u32>) -> Exc {
        Exc::Fault {
            vec,
            err,
            soft: false,
        }
    }
    pub fn gp(code: u32) -> Exc {
        Exc::fault(13, Some(code))
    }
    pub fn ud() -> Exc {
        Exc::fault(6, None)
    }
    pub fn de() -> Exc {
        Exc::fault(0, None)
    }
    pub fn nm() -> Exc {
        Exc::fault(7, None)
    }
    pub fn ss(code: u32) -> Exc {
        Exc::fault(12, Some(code))
    }
}

fn contributory(v: u8) -> bool {
    matches!(v, 0 | 10 | 11 | 12 | 13)
}

impl Cpu {
    pub fn deliver_event(&mut self, vec: u8, err: Option<u32>, soft: bool) -> Option<ExitReason> {
        let (mut v, mut e, mut s) = (vec, err, soft);
        for _ in 0..4 {
            match self.deliver(v, e, s) {
                Ok(()) => return None,
                Err(Exc::Exit(r)) => return Some(r),
                Err(Exc::Fault {
                    vec: v2, err: e2, ..
                }) => {
                    if v == 8 {
                        return Some(ExitReason::TripleFault);
                    }
                    s = false;
                    if (contributory(v) && (contributory(v2) || v2 == 14))
                        || (v == 14 && (contributory(v2) || v2 == 14))
                    {
                        v = 8;
                        e = Some(0);
                    } else {
                        v = v2;
                        e = e2;
                    }
                }
            }
        }
        Some(ExitReason::TripleFault)
    }

    fn deliver(&mut self, vec: u8, err: Option<u32>, soft: bool) -> R<()> {
        let off = vec as u64 * 16;
        if off + 15 > self.idtr.limit as u64 {
            return Err(Exc::gp(vec as u32 * 8 + 2));
        }
        let a = self.idtr.base + off;
        let lo = self.read_sys(a, 8)?;
        let hi = self.read_sys(a + 8, 8)?;
        let sel = (lo >> 16) as u16;
        let ist = ((lo >> 32) & 7) as usize;
        let attr = (lo >> 40) as u8;
        let gate_dpl = (attr >> 5) & 3;
        let typ = attr & 0xf;
        if attr & 0x80 == 0 {
            return Err(Exc::fault(11, Some(vec as u32 * 8 + 2)));
        }
        if typ != 0xe && typ != 0xf {
            return Err(Exc::gp(vec as u32 * 8 + 2));
        }
        if soft && gate_dpl < self.cpl {
            return Err(Exc::gp(vec as u32 * 8 + 2));
        }
        let target = (lo & 0xffff) | ((lo >> 32) & 0xffff_0000) | (hi << 32);
        let cs = self.load_desc_seg(sel, true)?;
        if !cs.is_code() || !cs.long() {
            return Err(Exc::gp(sel as u32 & 0xfffc));
        }
        let new_cpl = cs.dpl();
        let old_ss = self.seg[2].sel as u64;
        let old_rsp = self.gpr[4];
        let rflags = self.eflags_full();
        let mut rsp = if ist != 0 {
            self.tss_read(36 + (ist as u64 - 1) * 8)?
        } else if new_cpl < self.cpl {
            self.tss_read(4 + new_cpl as u64 * 8)?
        } else {
            old_rsp
        };
        rsp &= !0xf;
        let old_cs = self.seg[1].sel as u64;
        let mut push = |cpu: &mut Cpu, v: u64| -> R<()> {
            rsp = rsp.wrapping_sub(8);
            cpu.write_sys(rsp, 8, v)
        };
        push(self, old_ss)?;
        push(self, old_rsp)?;
        push(self, rflags)?;
        push(self, old_cs)?;
        let rip = self.rip;
        push(self, rip)?;
        if let Some(e) = err {
            push(self, e as u64)?;
        }
        self.gpr[4] = rsp;
        if new_cpl != self.cpl {
            self.seg[2] = Seg {
                sel: new_cpl as u16,
                base: 0,
                limit: 0xffff_ffff,
                flags: 0x00cf_9200 | ((new_cpl as u32) << 13),
            };
        }
        self.seg[1] = cs;
        self.cpl = new_cpl;
        self.rip = target;
        self.set_arith_flags_from_cc();
        self.eflags &= !(TF | NT | RF | VM | AC);
        if typ == 0xe {
            self.eflags &= !IF;
        }
        Ok(())
    }

    fn tss_read(&mut self, off: u64) -> R<u64> {
        if off + 7 > self.tr.limit as u64 {
            return Err(Exc::fault(10, Some(self.tr.sel as u32 & 0xfffc)));
        }
        self.read_sys(self.tr.base + off, 8)
    }

    pub fn read_desc(&mut self, sel: u16) -> R<(u64, u64)> {
        let idx = sel as u64 & 0xfff8;
        let table = if sel & 4 != 0 {
            (self.ldtr.base, self.ldtr.limit as u64)
        } else {
            (self.gdtr.base, self.gdtr.limit as u64)
        };
        if idx + 7 > table.1 {
            return Err(Exc::gp(sel as u32 & 0xfffc));
        }
        let lo = self.read_sys(table.0 + idx, 8)?;
        let hi = if lo & (1 << 44) == 0 && idx + 15 <= table.1 {
            self.read_sys(table.0 + idx + 8, 8)?
        } else {
            0
        };
        Ok((lo, hi))
    }

    pub fn load_desc_seg(&mut self, sel: u16, require_present: bool) -> R<Seg> {
        let (lo, _) = self.read_desc(sel)?;
        let dw0 = lo as u32;
        let dw1 = (lo >> 32) as u32;
        let base = ((dw0 >> 16) | ((dw1 & 0xff) << 16) | (dw1 & 0xff00_0000)) as u64;
        let mut limit = (dw0 & 0xffff) | (dw1 & 0xf_0000);
        if dw1 & (1 << 23) != 0 {
            limit = (limit << 12) | 0xfff;
        }
        let seg = Seg {
            sel,
            base,
            limit,
            flags: dw1 & 0x00f0_ff00,
        };
        if require_present && !seg.present() {
            return Err(Exc::fault(11, Some(sel as u32 & 0xfffc)));
        }
        Ok(seg)
    }
}
