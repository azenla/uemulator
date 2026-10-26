use super::decode::*;
use super::exc::Exc;
use super::exec::Loc;
use super::flags::*;
use super::*;

pub(crate) const MSR_TSC: u32 = 0x10;
pub(crate) const MSR_APIC_BASE: u32 = 0x1b;
pub(crate) const MSR_PAT: u32 = 0x277;
pub(crate) const MSR_EFER: u32 = 0xc000_0080;
pub(crate) const MSR_STAR: u32 = 0xc000_0081;
pub(crate) const MSR_LSTAR: u32 = 0xc000_0082;
pub(crate) const MSR_CSTAR: u32 = 0xc000_0083;
pub(crate) const MSR_SFMASK: u32 = 0xc000_0084;
pub(crate) const MSR_FS_BASE: u32 = 0xc000_0100;
pub(crate) const MSR_GS_BASE: u32 = 0xc000_0101;
pub(crate) const MSR_KGS_BASE: u32 = 0xc000_0102;
pub(crate) const MSR_TSC_AUX: u32 = 0xc000_0103;

const CPUID_EAX1: u32 = 0x0060_0f01;
const FEAT_EDX: u32 = (1 << 0)
    | (1 << 2)
    | (1 << 3)
    | (1 << 4)
    | (1 << 5)
    | (1 << 6)
    | (1 << 8)
    | (1 << 13)
    | (1 << 15)
    | (1 << 16)
    | (1 << 19)
    | (1 << 23)
    | (1 << 24)
    | (1 << 25)
    | (1 << 26);

impl Cpu {
    pub fn need_cpl0(&self) -> R<()> {
        if self.cpl != 0 {
            Err(Exc::gp(0))
        } else {
            Ok(())
        }
    }

    pub fn cpuid(&mut self) {
        let leaf = self.gpr[0] as u32;
        let (a, b, c, d): (u32, u32, u32, u32) = match leaf {
            0 => (
                7,
                u32::from_le_bytes(*b"Auth"),
                u32::from_le_bytes(*b"cAMD"),
                u32::from_le_bytes(*b"enti"),
            ),
            1 => (CPUID_EAX1, 0x0001_0800, 0, FEAT_EDX),
            0x8000_0000 => (
                0x8000_0008,
                u32::from_le_bytes(*b"Auth"),
                u32::from_le_bytes(*b"cAMD"),
                u32::from_le_bytes(*b"enti"),
            ),
            0x8000_0001 => (
                CPUID_EAX1,
                0,
                1,
                (1 << 11) | (1 << 20) | (1 << 29) | (FEAT_EDX & 0x0181_a171),
            ),
            0x8000_0002..=0x8000_0004 => {
                let s = b"uemulator virtual x86-64 cpu\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
                let o = (leaf - 0x8000_0002) as usize * 16;
                let w =
                    |k: usize| u32::from_le_bytes(s[o + k * 4..o + k * 4 + 4].try_into().unwrap());
                (w(0), w(1), w(2), w(3))
            }
            0x8000_0005 => (0xff40_ff40, 0xff40_ff40, 0x2008_0140, 0x4002_0140),
            0x8000_0006 => (0x6400_6400, 0x6400_4200, 0x0200_6140, 0),
            0x8000_0007 => (0, 0, 0, 1 << 8),
            0x8000_0008 => (0x3028, 0, 0, 0),
            _ => (0, 0, 0, 0),
        };
        self.gpr[0] = a as u64;
        self.gpr[3] = b as u64;
        self.gpr[1] = c as u64;
        self.gpr[2] = d as u64;
    }

    pub fn rdmsr(&mut self, msr: u32) -> R<u64> {
        Ok(match msr {
            MSR_TSC => self.tsc(),
            MSR_APIC_BASE => return Err(Exc::gp(0)),
            MSR_PAT => self.pat,
            MSR_EFER => self.efer,
            MSR_STAR => self.star,
            MSR_LSTAR => self.lstar,
            MSR_CSTAR => self.cstar,
            MSR_SFMASK => self.sfmask,
            MSR_FS_BASE => self.seg[4].base,
            MSR_GS_BASE => self.seg[5].base,
            MSR_KGS_BASE => self.kgsbase,
            MSR_TSC_AUX => self.tsc_aux,
            0x174..=0x176 => self.sysenter[(msr - 0x174) as usize],
            _ => match self.msr_misc.get(&msr) {
                Some(v) => *v,
                None => {
                    let rip = self.rip;
                    self.log_once(0x5000_0000 + msr as u64, || {
                        format!("rdmsr {msr:#x} -> #GP at rip {rip:#x}")
                    });
                    return Err(Exc::gp(0));
                }
            },
        })
    }

    pub fn wrmsr(&mut self, msr: u32, v: u64) -> R<()> {
        match msr {
            MSR_PAT => self.pat = v,
            MSR_EFER => {
                if v & !0xd01 != 0 {
                    return Err(Exc::gp(0));
                }
                self.efer = (self.efer & !0x901) | (v & 0x901);
            }
            MSR_STAR => self.star = v,
            MSR_LSTAR => self.lstar = v,
            MSR_CSTAR => self.cstar = v,
            MSR_SFMASK => self.sfmask = v,
            MSR_FS_BASE => self.seg[4].base = v,
            MSR_GS_BASE => self.seg[5].base = v,
            MSR_KGS_BASE => self.kgsbase = v,
            MSR_TSC_AUX => self.tsc_aux = v,
            MSR_TSC => {}
            0x174..=0x176 => self.sysenter[(msr - 0x174) as usize] = v,
            _ if self.msr_misc.contains_key(&msr) => {
                self.msr_misc.insert(msr, v);
            }
            _ => {
                let rip = self.rip;
                self.log_once(0x6000_0000 + msr as u64, || {
                    format!("wrmsr {msr:#x} <- {v:#x} -> #GP at rip {rip:#x}")
                });
                return Err(Exc::gp(0));
            }
        }
        Ok(())
    }

    pub fn mov_cr_dr(&mut self, i: &Insn) -> R<()> {
        self.need_cpl0()?;
        let n = i.reg as usize;
        let g = i.rm as usize;
        match i.op {
            0x120 => {
                self.gpr[g] = match n {
                    0 => self.cr0,
                    2 => self.cr2,
                    3 => self.cr3,
                    4 => self.cr4,
                    8 => self.cr8,
                    _ => return Err(Exc::ud()),
                }
            }
            0x122 => {
                let v = self.gpr[g];
                match n {
                    0 => {
                        self.cr0 = v | 0x10;
                        if self.cr0 & (1 << 31) != 0 && self.efer & 0x100 != 0 {
                            self.efer |= 0x400;
                        } else {
                            self.efer &= !0x400;
                        }
                        self.flush_tlb();
                    }
                    2 => self.cr2 = v,
                    3 => {
                        self.cr3 = v & 0x000f_ffff_ffff_f018;
                        self.flush_tlb();
                    }
                    4 => {
                        if v & !0x7bc != 0 {
                            return Err(Exc::gp(0));
                        }
                        self.cr4 = v;
                        self.flush_tlb();
                    }
                    8 => self.cr8 = v & 0xf,
                    _ => return Err(Exc::ud()),
                }
            }
            0x121 => self.gpr[g] = self.dr[n & 7],
            _ => self.dr[n & 7] = self.gpr[g],
        }
        Ok(())
    }

    pub fn do_syscall(&mut self, next: u64) -> R<()> {
        if self.efer & 1 == 0 {
            return Err(Exc::ud());
        }
        let f = self.eflags_full();
        self.gpr[1] = next;
        self.gpr[11] = f & !RF;
        let base = ((self.star >> 32) & 0xfffc) as u16;
        self.seg[1] = Seg {
            sel: base,
            base: 0,
            limit: 0xffff_ffff,
            flags: 0x0020_9a00,
        };
        self.seg[2] = Seg {
            sel: base + 8,
            base: 0,
            limit: 0xffff_ffff,
            flags: 0x00cf_9200,
        };
        self.eflags = (f & !self.sfmask & !RF) | 2;
        self.cc = Cc::Eflags;
        self.set_cpl(0);
        self.rip = self.lstar;
        Ok(())
    }

    pub fn do_sysret(&mut self, i: &Insn) -> R<()> {
        if self.efer & 1 == 0 {
            return Err(Exc::ud());
        }
        self.need_cpl0()?;
        if i.opsize != 8 {
            return Err(self.unimpl(i));
        }
        let base = ((self.star >> 48) & 0xfffc) as u16;
        self.seg[1] = Seg {
            sel: (base + 16) | 3,
            base: 0,
            limit: 0xffff_ffff,
            flags: 0x0020_fa00,
        };
        self.seg[2] = Seg {
            sel: (base + 8) | 3,
            base: 0,
            limit: 0xffff_ffff,
            flags: 0x00cf_f200,
        };
        let r11 = self.gpr[11];
        let m = CF | PF | AF | ZF | SF | TF | IF | DF | OF | NT | AC | (1 << 21);
        self.eflags = (r11 & m) | 2;
        self.cc = Cc::Eflags;
        self.set_cpl(3);
        self.rip = self.gpr[1];
        Ok(())
    }

    fn sys_desc_base(lo: u64, hi: u64) -> (u64, u32) {
        let dw0 = lo as u32;
        let dw1 = (lo >> 32) as u32;
        let base = ((dw0 >> 16) | ((dw1 & 0xff) << 16) | (dw1 & 0xff00_0000)) as u64
            | ((hi & 0xffff_ffff) << 32);
        let mut limit = (dw0 & 0xffff) | (dw1 & 0xf_0000);
        if dw1 & (1 << 23) != 0 {
            limit = (limit << 12) | 0xfff;
        }
        (base, limit)
    }

    pub fn sys_group(&mut self, i: &Insn, next: u64) -> R<()> {
        let r = i.reg & 7;
        if i.op == 0x100 {
            let loc = self.rm_loc(i, next);
            let size = if i.md == 3 { i.opsize } else { 2 };
            match r {
                0 => {
                    let v = self.ldtr.sel as u64;
                    self.st(loc, size, i, v)?;
                }
                1 => {
                    let v = self.tr.sel as u64;
                    self.st(loc, size, i, v)?;
                }
                2 | 3 => {
                    self.need_cpl0()?;
                    let sel = self.ld(loc, 2, i)? as u16;
                    if sel & 0xfffc == 0 {
                        if r == 2 {
                            self.ldtr = Seg::default();
                            return {
                                let _: () = self.rip = next;
                                Ok(())
                            };
                        }
                        return Err(Exc::gp(0));
                    }
                    let (lo, hi) = self.read_desc(sel)?;
                    let (base, limit) = Self::sys_desc_base(lo, hi);
                    let s = Seg {
                        sel,
                        base,
                        limit,
                        flags: ((lo >> 32) as u32) & 0x00f0_ff00,
                    };
                    if !s.present() {
                        return Err(Exc::fault(11, Some(sel as u32 & 0xfffc)));
                    }
                    if r == 2 {
                        self.ldtr = s;
                    } else {
                        self.tr = s;
                    }
                }
                4 | 5 => {
                    let f = self.eflags_full();
                    self.set_eflags_arith(f | ZF);
                }
                _ => return Err(Exc::ud()),
            }
            self.rip = next;
            return Ok(());
        }
        if i.md == 3 && r != 4 && r != 6 {
            match i.modrm {
                0xf8 => {
                    self.need_cpl0()?;
                    std::mem::swap(&mut self.seg[5].base, &mut self.kgsbase);
                }
                _ => return Err(self.unimpl(i)),
            }
            self.rip = next;
            return Ok(());
        }
        let loc = self.rm_loc(i, next);
        match r {
            0 | 1 => {
                let a = match loc {
                    Loc::Mem(a) => a,
                    _ => return Err(Exc::ud()),
                };
                let t = if r == 0 { self.gdtr } else { self.idtr };
                self.write(a, 2, t.limit as u64)?;
                self.write(a + 2, 8, t.base)?;
            }
            2 | 3 => {
                self.need_cpl0()?;
                let a = match loc {
                    Loc::Mem(a) => a,
                    _ => return Err(Exc::ud()),
                };
                let limit = self.read(a, 2)? as u16;
                let base = self.read(a + 2, 8)?;
                let t = Dt { base, limit };
                if r == 2 {
                    self.gdtr = t;
                } else {
                    self.idtr = t;
                }
            }
            4 => {
                let size = if i.md == 3 { i.opsize } else { 2 };
                let v = self.cr0 & 0xffff;
                self.st(loc, size, i, v)?;
            }
            6 => {
                self.need_cpl0()?;
                let v = self.ld(loc, 2, i)?;
                self.cr0 = (self.cr0 & !0xe) | (v & 0xe) | (self.cr0 & 1) | (v & 1);
            }
            7 => {
                self.need_cpl0()?;
                if let Loc::Mem(a) = loc {
                    self.invlpg(a);
                } else {
                    return Err(Exc::ud());
                }
            }
            _ => return Err(Exc::ud()),
        }
        self.rip = next;
        Ok(())
    }

    pub fn group_ae(&mut self, i: &Insn, next: u64) -> R<()> {
        let r = i.reg & 7;
        if i.md == 3 {
            if i.rep != 0 || !(5..=7).contains(&r) {
                return Err(Exc::ud());
            }
        } else {
            let a = self.lin(i, self.ea(i, next));
            match r {
                0 => self.fxsave(a)?,
                1 => self.fxrstor(a)?,
                2 => {
                    let v = self.read(a, 4)? as u32;
                    if v & !0xffff != 0 {
                        return Err(Exc::gp(0));
                    }
                    self.mxcsr = v;
                }
                3 => {
                    let v = self.mxcsr as u64;
                    self.write(a, 4, v)?;
                }
                7 => {
                    self.read(a, 1)?;
                }
                _ => return Err(Exc::ud()),
            }
        }
        self.rip = next;
        Ok(())
    }
}
