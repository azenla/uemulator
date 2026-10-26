use super::decode::{self, DecodeErr, Insn};
use super::exc::Exc;
use super::flags::*;
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Acc {
    Read,
    Write,
    Exec,
}

const PTE_P: u64 = 1;
const PTE_W: u64 = 2;
const PTE_U: u64 = 4;
const PTE_A: u64 = 0x20;
const PTE_D: u64 = 0x40;
const PTE_PS: u64 = 0x80;
const PTE_NX: u64 = 1 << 63;
const PTE_RSVD: u64 = 0x000f_ff00_0000_0000;

impl Cpu {
    pub fn eflags_full(&self) -> u64 {
        let arith = flags::compute(
            self.cc,
            self.cc_size,
            self.cc_dst,
            self.cc_src,
            self.cc_src2,
            self.eflags,
        );
        (self.eflags & !ARITH) | arith | 2
    }

    pub fn set_arith_flags_from_cc(&mut self) {
        self.eflags = self.eflags_full();
        self.cc = Cc::Eflags;
    }

    pub fn set_eflags_arith(&mut self, arith: u64) {
        self.eflags = (self.eflags & !ARITH) | (arith & ARITH) | 2;
        self.cc = Cc::Eflags;
    }

    pub fn set_cc(&mut self, cc: Cc, size: u8, dst: u64, src: u64, src2: u64) {
        self.cc = cc;
        self.cc_size = size;
        self.cc_dst = dst;
        self.cc_src = src;
        self.cc_src2 = src2;
    }

    pub fn cf(&self) -> bool {
        self.eflags_full() & CF != 0
    }

    pub fn page_fault(&mut self, la: u64, acc: Acc, user: bool, present: bool, rsvd: bool) -> Exc {
        self.cr2 = la;
        let mut e = 0u32;
        if present {
            e |= 1;
        }
        if acc == Acc::Write {
            e |= 2;
        }
        if user {
            e |= 4;
        }
        if rsvd {
            e |= 8;
        }
        if acc == Acc::Exec && self.efer & (1 << 11) != 0 {
            e |= 16;
        }
        Exc::fault(14, Some(e))
    }

    fn rd64_phys(&self, pa: u64) -> Option<u64> {
        let a = pa as usize;
        self.ram
            .get(a..a + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn walk(&mut self, la: u64, acc: Acc, user: bool) -> R<(u64, u8)> {
        let write = acc == Acc::Write;
        let nxe = self.efer & (1 << 11) != 0;
        let mut table = self.cr3 & 0x000f_ffff_ffff_f000;
        let mut rw = true;
        let mut us = true;
        let mut nx = false;
        let mut level = 3i32;
        loop {
            let idx = (la >> (12 + 9 * level)) & 0x1ff;
            let pa = table + idx * 8;
            let Some(mut e) = self.rd64_phys(pa) else {
                return Err(self.page_fault(la, acc, user, false, false));
            };
            if e & PTE_P == 0 {
                return Err(self.page_fault(la, acc, user, false, false));
            }
            if e & PTE_RSVD != 0 || (!nxe && e & PTE_NX != 0) {
                return Err(self.page_fault(la, acc, user, true, true));
            }
            let leaf = level == 0 || (e & PTE_PS != 0 && level <= 2);
            rw &= e & PTE_W != 0;
            us &= e & PTE_U != 0;
            nx |= e & PTE_NX != 0;
            if leaf {
                let wp = self.cr0 & (1 << 16) != 0;
                let denied = (user && !us)
                    || (write && !rw && (user || wp))
                    || (acc == Acc::Exec && nx && nxe);
                if denied {
                    return Err(self.page_fault(la, acc, user, true, false));
                }
                let mut upd = e | PTE_A;
                if write {
                    upd |= PTE_D;
                }
                if upd != e {
                    e = upd;
                    self.ram[pa as usize..pa as usize + 8].copy_from_slice(&e.to_le_bytes());
                }
                let size_bits = 12 + 9 * level as u32;
                let base = e & 0x000f_ffff_ffff_f000 & !((1u64 << size_bits) - 1);
                let pa_page = base | (la & ((1u64 << size_bits) - 1) & !0xfff);
                let mut f = 0;
                if rw {
                    f |= TLB_W;
                }
                if us {
                    f |= TLB_U;
                }
                if nx && nxe {
                    f |= TLB_NX;
                }
                if e & PTE_D != 0 {
                    f |= TLB_D;
                }
                return Ok((pa_page, f));
            }
            if e & PTE_A == 0 {
                e |= PTE_A;
                self.ram[pa as usize..pa as usize + 8].copy_from_slice(&e.to_le_bytes());
            }
            table = e & 0x000f_ffff_ffff_f000;
            level -= 1;
        }
    }

    pub fn translate(&mut self, la: u64, acc: Acc, sys: bool) -> R<u64> {
        if (la as i64) >> 47 != 0 && (la as i64) >> 47 != -1 {
            return Err(Exc::gp(0));
        }
        let user = self.cpl == 3 && !sys;
        let page = la & !0xfff;
        let slot = ((user as usize) << TLB_BITS) | ((la >> 12) as usize & (TLB_N - 1));
        let e = self.tlb[slot];
        let tag = match acc {
            Acc::Read => e.tag_r,
            Acc::Write => e.tag_w,
            Acc::Exec => e.tag_x,
        };
        if tag == page {
            return Ok(la.wrapping_add(e.addend).wrapping_sub(self.ram_base));
        }
        let (pbase, flags) = self.walk(la, acc, user)?;
        if pbase < self.ram.len() as u64 {
            let wp = self.cr0 & (1 << 16) != 0;
            let mut ent = if e.tag_r == page || e.tag_w == page || e.tag_x == page {
                e
            } else {
                TLB_EMPTY
            };
            ent.addend = (self.ram_base + pbase).wrapping_sub(page);
            if !user || flags & TLB_U != 0 {
                ent.tag_r = page;
                if flags & TLB_NX == 0 {
                    ent.tag_x = page;
                }
            }
            let w_ok = flags & TLB_D != 0
                && if user {
                    flags & (TLB_U | TLB_W) == (TLB_U | TLB_W)
                } else {
                    flags & TLB_W != 0 || !wp
                };
            if w_ok && self.code_page[(pbase >> 12) as usize] == 0 {
                ent.tag_w = page;
            }
            self.tlb[slot] = ent;
        }
        Ok(pbase | (la & 0xfff))
    }

    pub fn flush_tlb(&mut self) {
        for e in self.tlb.iter_mut() {
            *e = TLB_EMPTY;
        }
    }

    pub fn invlpg(&mut self, la: u64) {
        let i = (la >> 12) as usize & (TLB_N - 1);
        self.tlb[i] = TLB_EMPTY;
        self.tlb[TLB_N + i] = TLB_EMPTY;
    }

    fn phys_read(&mut self, pa: u64, size: u8) -> u64 {
        let a = pa as usize;
        let n = size as usize;
        if let Some(b) = self.ram.get(a..a + n) {
            let mut v = [0u8; 8];
            v[..n].copy_from_slice(b);
            u64::from_le_bytes(v)
        } else {
            let now = self.now_ns();
            let _ = now;
            self.log_once(0x1000 + (pa >> 12), || {
                format!("read of unmapped physical address {pa:#x}")
            });
            u64::MAX >> (64 - 8 * size as u32)
        }
    }

    fn phys_write(&mut self, pa: u64, size: u8, val: u64) {
        let a = pa as usize;
        let n = size as usize;
        if a + n <= self.ram.len() {
            if self.code_page[a >> 12] != 0 {
                self.code_written(a >> 12);
            }
            self.ram[a..a + n].copy_from_slice(&val.to_le_bytes()[..n]);
        } else {
            self.log_once(0x2000 + (pa >> 12), || {
                format!("write to unmapped physical address {pa:#x}")
            });
        }
    }

    fn code_written(&mut self, ppn: usize) {
        if let Some(mut j) = self.jit.take() {
            j.invalidate_page(ppn as u64);
            self.jit = Some(j);
            self.code_page[ppn] = 0;
            self.dc_gen = self.dc_gen.wrapping_add(1);
            self.exit_block = true;
        } else {
            self.flush_decode();
        }
    }

    pub fn flush_decode(&mut self) {
        self.flushes += 1;
        self.dc_gen = self.dc_gen.wrapping_add(1);
        for p in self.code_list.drain(..) {
            self.code_page[p as usize] = 0;
        }
    }

    pub fn mark_code(&mut self, pa: u64) {
        let p = (pa >> 12) as usize;
        if p < self.code_page.len() && self.code_page[p] == 0 {
            self.code_page[p] = 1;
            self.code_list.push(p as u32);
            let base = self.ram_base;
            for e in self.tlb.iter_mut() {
                if e.tag_w != TLB_INV
                    && (e.tag_w.wrapping_add(e.addend).wrapping_sub(base) >> 12) as usize == p
                {
                    e.tag_w = TLB_INV;
                }
            }
        }
    }

    fn mem_read(&mut self, la: u64, size: u8, sys: bool) -> R<u64> {
        let user = self.cpl == 3 && !sys;
        let e = &self.tlb[((user as usize) << TLB_BITS) | ((la >> 12) as usize & (TLB_N - 1))];
        if e.tag_r == la & !0xfff && (la & 0xfff) + size as u64 <= 4096 {
            let p = la.wrapping_add(e.addend);
            unsafe {
                return Ok(match size {
                    1 => *(p as *const u8) as u64,
                    2 => (p as *const u16).read_unaligned() as u64,
                    4 => (p as *const u32).read_unaligned() as u64,
                    _ => (p as *const u64).read_unaligned(),
                });
            }
        }
        let off = (la & 0xfff) as usize;
        if off + size as usize <= 4096 {
            let pa = self.translate(la, Acc::Read, sys)?;
            return Ok(self.phys_read(pa, size));
        }
        let first = 4096 - off;
        let pa1 = self.translate(la, Acc::Read, sys)?;
        let pa2 = self.translate((la & !0xfff).wrapping_add(4096), Acc::Read, sys)?;
        let lo = self.phys_read(pa1, first as u8);
        let hi = self.phys_read(pa2, size - first as u8);
        Ok(lo | (hi << (8 * first)))
    }

    fn mem_write(&mut self, la: u64, size: u8, val: u64, sys: bool) -> R<()> {
        let user = self.cpl == 3 && !sys;
        let e = &self.tlb[((user as usize) << TLB_BITS) | ((la >> 12) as usize & (TLB_N - 1))];
        if e.tag_w == la & !0xfff && (la & 0xfff) + size as u64 <= 4096 {
            let p = la.wrapping_add(e.addend);
            unsafe {
                match size {
                    1 => *(p as *mut u8) = val as u8,
                    2 => (p as *mut u16).write_unaligned(val as u16),
                    4 => (p as *mut u32).write_unaligned(val as u32),
                    _ => (p as *mut u64).write_unaligned(val),
                }
            }
            return Ok(());
        }
        let off = (la & 0xfff) as usize;
        if off + size as usize <= 4096 {
            let pa = self.translate(la, Acc::Write, sys)?;
            self.phys_write(pa, size, val);
            return Ok(());
        }
        let first = 4096 - off;
        let pa1 = self.translate(la, Acc::Write, sys)?;
        let pa2 = self.translate((la & !0xfff).wrapping_add(4096), Acc::Write, sys)?;
        self.phys_write(pa1, first as u8, val);
        self.phys_write(pa2, size - first as u8, val >> (8 * first));
        Ok(())
    }

    pub fn read(&mut self, la: u64, size: u8) -> R<u64> {
        self.mem_read(la, size, false)
    }

    pub fn write(&mut self, la: u64, size: u8, val: u64) -> R<()> {
        self.mem_write(la, size, val, false)
    }

    pub fn read_sys(&mut self, la: u64, size: u8) -> R<u64> {
        self.mem_read(la, size, true)
    }

    pub fn write_sys(&mut self, la: u64, size: u8, val: u64) -> R<()> {
        self.mem_write(la, size, val, true)
    }

    pub fn read128(&mut self, la: u64) -> R<u128> {
        let lo = self.read(la, 8)?;
        let hi = self.read(la.wrapping_add(8), 8)?;
        Ok(lo as u128 | ((hi as u128) << 64))
    }

    pub fn write128(&mut self, la: u64, v: u128) -> R<()> {
        self.translate(la, Acc::Write, false)?;
        self.translate(la.wrapping_add(15), Acc::Write, false)?;
        self.write(la, 8, v as u64)?;
        self.write(la.wrapping_add(8), 8, (v >> 64) as u64)
    }

    pub fn fetch_decode(&mut self) -> R<Insn> {
        let rip = self.rip;
        let pa = self.translate(rip, Acc::Exec, false)?;
        let h = (pa ^ (pa >> 16)) as usize & (DC_SIZE - 1);
        let e = &self.dcache[h];
        if e.gpa == pa && e.gen_ == self.dc_gen {
            return Ok(e.insn);
        }
        let page_left = 4096 - (pa & 0xfff) as usize;
        let n = page_left.min(15);
        let a = pa as usize;
        let mut buf = [0u8; 16];
        let Some(src) = self.ram.get(a..a + n) else {
            return Err(self.page_fault(rip, Acc::Exec, self.cpl == 3, false, false));
        };
        buf[..n].copy_from_slice(src);
        match decode::decode(&buf[..n]) {
            Ok(i) => {
                self.dcache[h] = DEntry {
                    gpa: pa,
                    gen_: self.dc_gen,
                    insn: i,
                };
                self.mark_code(pa);
                Ok(i)
            }
            Err(DecodeErr::Invalid) => Err(Exc::ud()),
            Err(DecodeErr::Truncated) if n == 15 => Err(Exc::ud()),
            Err(DecodeErr::Truncated) => {
                let pa2 = self.translate((rip & !0xfff).wrapping_add(4096), Acc::Exec, false)?;
                let m = 15 - n;
                let b = pa2 as usize;
                let Some(src2) = self.ram.get(b..b + m) else {
                    return Err(Exc::ud());
                };
                buf[n..15].copy_from_slice(src2);
                match decode::decode(&buf[..15]) {
                    Ok(i) => {
                        self.mark_code(pa);
                        self.mark_code(pa2);
                        Ok(i)
                    }
                    Err(_) => Err(Exc::ud()),
                }
            }
        }
    }
}
