use super::decode::*;
use super::exc::Exc;
use super::exec::Loc;
use super::*;

impl Cpu {
    fn fpu_check(&self) -> R<()> {
        if self.cr0 & 0xc != 0 {
            return Err(Exc::nm());
        }
        Ok(())
    }

    pub fn x87(&mut self, i: &Insn, next: u64) -> R<()> {
        self.fpu_check()?;
        let op = (i.op & 0xff) as u8;
        let r = i.reg & 7;
        if i.md == 3 {
            match (op, i.modrm) {
                (0xdb, 0xe3) => {
                    self.fcw = 0x37f;
                    self.fsw = 0;
                    self.ftw = 0;
                }
                (0xdb, 0xe2) => self.fsw &= 0x7f00,
                (0xdf, 0xe0) => {
                    let v = self.fsw as u64;
                    self.wr_reg(0, 2, true, v);
                }
                _ => return Err(self.unimpl(i)),
            }
        } else {
            let a = match self.rm_loc(i, next) {
                Loc::Mem(a) => a,
                _ => unreachable!(),
            };
            match (op, r) {
                (0xd9, 5) => self.fcw = self.read(a, 2)? as u16,
                (0xd9, 7) => self.write(a, 2, self.fcw as u64)?,
                (0xdd, 7) => self.write(a, 2, self.fsw as u64)?,
                _ => return Err(self.unimpl(i)),
            }
        }
        self.rip = next;
        Ok(())
    }

    pub fn sse(&mut self, _i: &Insn, _next: u64) -> R<bool> {
        Ok(false)
    }

    pub fn fxsave(&mut self, a: u64) -> R<()> {
        if self.cr0 & 4 != 0 {
            return Err(Exc::ud());
        }
        self.fpu_check()?;
        self.translate(a, mem::Acc::Write, false)?;
        self.translate(a + 511, mem::Acc::Write, false)?;
        self.write(a, 2, self.fcw as u64)?;
        self.write(a + 2, 2, self.fsw as u64)?;
        self.write(a + 4, 1, self.ftw as u64 & 0xff)?;
        for k in 0..8 {
            let b = self.st[k];
            for (j, byte) in b.iter().enumerate() {
                self.write(a + 32 + k as u64 * 16 + j as u64, 1, *byte as u64)?;
            }
        }
        self.write(a + 24, 4, self.mxcsr as u64)?;
        self.write(a + 28, 4, 0xffff)?;
        for k in 0..16 {
            self.write128(a + 160 + k as u64 * 16, self.xmm[k])?;
        }
        Ok(())
    }

    pub fn fxrstor(&mut self, a: u64) -> R<()> {
        if self.cr0 & 4 != 0 {
            return Err(Exc::ud());
        }
        self.fpu_check()?;
        let fcw = self.read(a, 2)? as u16;
        let fsw = self.read(a + 2, 2)? as u16;
        let ftw = self.read(a + 4, 1)? as u16;
        let mxcsr = self.read(a + 24, 4)? as u32;
        if mxcsr & !0xffff != 0 {
            return Err(Exc::gp(0));
        }
        for k in 0..8 {
            for j in 0..10 {
                self.st[k][j] = self.read(a + 32 + k as u64 * 16 + j as u64, 1)? as u8;
            }
        }
        for k in 0..16 {
            self.xmm[k] = self.read128(a + 160 + k as u64 * 16)?;
        }
        self.fcw = fcw;
        self.fsw = fsw;
        self.ftw = ftw;
        self.mxcsr = mxcsr;
        Ok(())
    }
}
