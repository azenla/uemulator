//! Minimal x86-64 assembler for the JIT backend.

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;
pub const R12: u8 = 12;
pub const R13: u8 = 13;
pub const R14: u8 = 14;
pub const R15: u8 = 15;

#[cfg(windows)]
pub const ARGS: [u8; 4] = [RCX, RDX, R8, R9];
#[cfg(not(windows))]
pub const ARGS: [u8; 4] = [RDI, RSI, RDX, RCX];

pub const ADD: u8 = 0;
pub const OR: u8 = 1;
pub const AND: u8 = 4;
pub const SUB: u8 = 5;
pub const XOR: u8 = 6;
pub const CMP: u8 = 7;

pub const SHL: u8 = 4;
pub const SHR: u8 = 5;

pub const NE: u8 = 5;

#[derive(Clone, Copy)]
pub struct Mem {
    base: u8,
    index: Option<(u8, u8)>,
    disp: i32,
}

pub fn mem(base: u8, disp: i32) -> Mem {
    Mem {
        base,
        index: None,
        disp,
    }
}

/// `[base + index << scale + disp]`; `index` must not be RSP.
pub fn mem_idx(base: u8, index: u8, scale: u8, disp: i32) -> Mem {
    Mem {
        base,
        index: Some((index, scale)),
        disp,
    }
}

#[derive(Clone, Copy)]
pub struct Label(usize);

pub struct Asm {
    base: u64,
    pub code: Vec<u8>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, usize)>,
}

impl Asm {
    pub fn new(base: u64) -> Asm {
        Asm {
            base,
            code: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
        }
    }

    pub fn here(&self) -> u64 {
        self.base + self.code.len() as u64
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, l: Label) {
        self.labels[l.0] = Some(self.code.len());
    }

    pub fn finish(&mut self) {
        for &(pos, l) in &self.fixups {
            let target = self.labels[l].expect("unbound label");
            let rel = target as i64 - (pos as i64 + 4);
            self.code[pos..pos + 4].copy_from_slice(&(rel as i32).to_le_bytes());
        }
        self.fixups.clear();
    }

    fn rex(&mut self, w: bool, reg: u8, index: u8, base: u8, force: bool) {
        let r =
            0x40 | (w as u8) << 3 | (reg >> 3 & 1) << 2 | (index >> 3 & 1) << 1 | (base >> 3 & 1);
        if r != 0x40 || force {
            self.code.push(r);
        }
    }

    fn modrm_mem(&mut self, reg: u8, m: Mem) {
        let base = m.base & 7;
        let sib = m.index.is_some() || base == 4;
        let md = if m.disp == 0 && base != 5 {
            0
        } else if i8::try_from(m.disp).is_ok() {
            1
        } else {
            2
        };
        self.code
            .push(md << 6 | (reg & 7) << 3 | if sib { 4 } else { base });
        if sib {
            let (ix, sc) = m.index.map_or((4, 0), |(i, s)| (i & 7, s));
            self.code.push(sc << 6 | ix << 3 | base);
        }
        match md {
            1 => self.code.push(m.disp as i8 as u8),
            2 => self.code.extend_from_slice(&m.disp.to_le_bytes()),
            _ => {}
        }
    }

    fn op_rm(&mut self, pfx: bool, w: bool, force: bool, opc: &[u8], reg: u8, m: Mem) {
        if pfx {
            self.code.push(0x66);
        }
        self.rex(w, reg, m.index.map_or(0, |i| i.0), m.base, force);
        self.code.extend_from_slice(opc);
        self.modrm_mem(reg, m);
    }

    fn op_rr(&mut self, pfx: bool, w: bool, force: bool, opc: &[u8], reg: u8, rm: u8) {
        if pfx {
            self.code.push(0x66);
        }
        self.rex(w, reg, 0, rm, force);
        self.code.extend_from_slice(opc);
        self.code.push(0xc0 | (reg & 7) << 3 | (rm & 7));
    }

    pub fn mov_rr(&mut self, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x89], s, d);
    }

    pub fn mov_imm(&mut self, r: u8, v: u64) {
        if v <= u32::MAX as u64 {
            self.rex(false, 0, 0, r, false);
            self.code.push(0xb8 | (r & 7));
            self.code.extend_from_slice(&(v as u32).to_le_bytes());
        } else if i32::try_from(v as i64).is_ok() {
            self.op_rr(false, true, false, &[0xc7], 0, r);
            self.code.extend_from_slice(&(v as u32).to_le_bytes());
        } else {
            self.rex(true, 0, 0, r, false);
            self.code.push(0xb8 | (r & 7));
            self.code.extend_from_slice(&v.to_le_bytes());
        }
    }

    pub fn ld(&mut self, d: u8, m: Mem) {
        self.op_rm(false, true, false, &[0x8b], d, m);
    }

    /// Load `size` bytes, zero- or sign-extended to 64 bits.
    pub fn ld_ext(&mut self, d: u8, m: Mem, size: u8, signed: bool) {
        match (size, signed) {
            (1, false) => self.op_rm(false, false, false, &[0x0f, 0xb6], d, m),
            (1, true) => self.op_rm(false, true, false, &[0x0f, 0xbe], d, m),
            (2, false) => self.op_rm(false, false, false, &[0x0f, 0xb7], d, m),
            (2, true) => self.op_rm(false, true, false, &[0x0f, 0xbf], d, m),
            (4, false) => self.op_rm(false, false, false, &[0x8b], d, m),
            (4, true) => self.op_rm(false, true, false, &[0x63], d, m),
            _ => self.op_rm(false, true, false, &[0x8b], d, m),
        }
    }

    pub fn ext_rr(&mut self, d: u8, s: u8, size: u8, signed: bool) {
        match (size, signed) {
            (1, false) => self.op_rr(false, false, true, &[0x0f, 0xb6], d, s),
            (1, true) => self.op_rr(false, true, true, &[0x0f, 0xbe], d, s),
            (2, false) => self.op_rr(false, false, false, &[0x0f, 0xb7], d, s),
            (2, true) => self.op_rr(false, true, false, &[0x0f, 0xbf], d, s),
            (4, false) => self.mov_rr(d, s, false),
            (4, true) => self.op_rr(false, true, false, &[0x63], d, s),
            _ => self.mov_rr(d, s, true),
        }
    }

    pub fn st(&mut self, m: Mem, s: u8, size: u8) {
        match size {
            1 => self.op_rm(false, false, true, &[0x88], s, m),
            2 => self.op_rm(true, false, false, &[0x89], s, m),
            4 => self.op_rm(false, false, false, &[0x89], s, m),
            _ => self.op_rm(false, true, false, &[0x89], s, m),
        }
    }

    pub fn st_imm16(&mut self, m: Mem, v: u16) {
        self.op_rm(true, false, false, &[0xc7], 0, m);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    pub fn lea(&mut self, d: u8, m: Mem) {
        self.op_rm(false, true, false, &[0x8d], d, m);
    }

    /// `op d, s` for ADD/OR/AND/SUB/XOR/CMP (the `op r/m, r` forms).
    pub fn alu_rr(&mut self, op: u8, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[op << 3 | 1], s, d);
    }

    pub fn test_rr(&mut self, a: u8, b: u8, w: bool) {
        self.op_rr(false, w, false, &[0x85], b, a);
    }

    pub fn alu_ri(&mut self, op: u8, r: u8, imm: i32, w: bool) {
        if let Ok(b) = i8::try_from(imm) {
            self.op_rr(false, w, false, &[0x83], op, r);
            self.code.push(b as u8);
        } else {
            self.op_rr(false, w, false, &[0x81], op, r);
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
    }

    pub fn shift_ri(&mut self, op: u8, r: u8, n: u8, w: bool) {
        self.op_rr(false, w, false, &[0xc1], op, r);
        self.code.push(n);
    }

    pub fn cmp_mr(&mut self, m: Mem, r: u8) {
        self.op_rm(false, true, false, &[0x39], r, m);
    }

    pub fn cmp_m8_imm(&mut self, m: Mem, v: u8) {
        self.op_rm(false, false, false, &[0x80], 7, m);
        self.code.push(v);
    }

    pub fn sete(&mut self, r: u8) {
        self.op_rr(false, false, true, &[0x0f, 0x94], 0, r);
    }

    pub fn jcc(&mut self, cc: u8, l: Label) {
        self.code.extend_from_slice(&[0x0f, 0x80 | cc]);
        self.fixups.push((self.code.len(), l.0));
        self.code.extend_from_slice(&[0; 4]);
    }

    pub fn jmp(&mut self, l: Label) {
        self.code.push(0xe9);
        self.fixups.push((self.code.len(), l.0));
        self.code.extend_from_slice(&[0; 4]);
    }

    pub fn jmp_abs(&mut self, target: u64) {
        let rel = target as i64 - (self.here() as i64 + 5);
        self.code.push(0xe9);
        self.code
            .extend_from_slice(&i32::try_from(rel).expect("jump out of range").to_le_bytes());
    }

    pub fn call_reg(&mut self, r: u8) {
        self.op_rr(false, false, false, &[0xff], 2, r);
    }

    pub fn jmp_reg(&mut self, r: u8) {
        self.op_rr(false, false, false, &[0xff], 4, r);
    }

    pub fn push(&mut self, r: u8) {
        self.rex(false, 0, 0, r, false);
        self.code.push(0x50 | (r & 7));
    }

    pub fn pop(&mut self, r: u8) {
        self.rex(false, 0, 0, r, false);
        self.code.push(0x58 | (r & 7));
    }

    pub fn ret(&mut self) {
        self.code.push(0xc3);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::sys::ExecMem;

    fn run(f: impl FnOnce(&mut Asm), a0: u64, a1: u64) -> u64 {
        let m = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(m.base());
        a.mov_rr(RAX, ARGS[0], true);
        a.mov_rr(RCX, ARGS[1], true);
        f(&mut a);
        a.ret();
        a.finish();
        m.write_bytes(0, &a.code);
        let fun: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(m.base() as usize) };
        fun(a0, a1)
    }

    #[test]
    fn arithmetic() {
        assert_eq!(run(|a| a.alu_rr(ADD, RAX, RCX, true), 40, 2), 42);
        assert_eq!(run(|a| a.alu_rr(SUB, RAX, RCX, true), 40, 2), 38);
        assert_eq!(run(|a| a.alu_rr(AND, RAX, RCX, true), 0xff0, 0x0ff), 0x0f0);
        assert_eq!(run(|a| a.alu_rr(OR, RAX, RCX, true), 0xf0, 0x0f), 0xff);
        assert_eq!(run(|a| a.alu_rr(XOR, RAX, RCX, true), 0xff, 0x0f), 0xf0);
        assert_eq!(
            run(|a| a.alu_rr(ADD, RAX, RCX, false), 0xffff_ffff, 2),
            1,
            "32-bit ops zero-extend"
        );
        assert_eq!(run(|a| a.alu_ri(ADD, RAX, 5, true), 1, 0), 6);
        assert_eq!(run(|a| a.alu_ri(SUB, RAX, 0x1000, true), 0x1001, 0), 1);
        assert_eq!(run(|a| a.shift_ri(SHL, RAX, 4, true), 3, 0), 48);
        assert_eq!(run(|a| a.shift_ri(SHR, RAX, 4, true), 0xf0, 0), 0xf);
        assert_eq!(run(|a| a.lea(RAX, mem_idx(RAX, RCX, 3, -2)), 100, 5), 138);
    }

    #[test]
    fn immediates_and_high_registers() {
        for &v in &[
            0u64,
            1,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0xffff_ffff_ffff_fff0,
            0x1234_5678_9abc_def0,
        ] {
            assert_eq!(run(|a| a.mov_imm(RAX, v), 0, 0), v, "{v:#x}");
            assert_eq!(
                run(
                    |a| {
                        a.mov_imm(R9, v);
                        a.mov_rr(RAX, R9, true);
                    },
                    0,
                    0
                ),
                v,
                "{v:#x} via r9"
            );
            assert_eq!(
                run(
                    |a| {
                        a.push(R13);
                        a.mov_imm(R13, v);
                        a.mov_rr(RAX, R13, true);
                        a.pop(R13);
                    },
                    0,
                    0
                ),
                v,
                "{v:#x} via r13"
            );
        }
    }

    #[test]
    fn loads_and_stores() {
        let mut buf = [0u64; 4];
        let p = buf.as_mut_ptr() as u64;
        run(
            |a| {
                a.mov_imm(RDX, 0xdead_beef_1234_5678);
                a.st(mem(RAX, 8), RDX, 8);
            },
            p,
            0,
        );
        assert_eq!(buf[1], 0xdead_beef_1234_5678);
        let ld = |size, signed, off: i32| {
            run(
                move |a| {
                    a.mov_imm(RDX, off as u64);
                    a.ld_ext(RAX, mem_idx(RAX, RDX, 0, 0), size, signed);
                },
                p,
                0,
            )
        };
        assert_eq!(ld(8, false, 8), 0xdead_beef_1234_5678);
        assert_eq!(ld(4, false, 8), 0x1234_5678);
        assert_eq!(ld(4, true, 12), 0xffff_ffff_dead_beef);
        assert_eq!(ld(2, false, 8), 0x5678);
        assert_eq!(ld(2, true, 12), 0xffff_ffff_ffff_beef);
        assert_eq!(ld(1, false, 8), 0x78);
        assert_eq!(ld(1, true, 15), 0xffff_ffff_ffff_ffde);
        for (size, mask) in [(1u8, 0xffu64), (2, 0xffff), (4, 0xffff_ffff)] {
            buf[2] = !0;
            run(
                |a| {
                    a.mov_imm(R9, 0x1111_2222_3333_4444);
                    a.mov_imm(RDI, 16);
                    a.st(mem_idx(RAX, RDI, 0, 0), R9, size);
                },
                p,
                0,
            );
            assert_eq!(
                buf[2],
                !mask | (0x1111_2222_3333_4444 & mask),
                "size {size}"
            );
        }
    }

    #[test]
    fn base_register_encodings() {
        let mut buf = [0u64; 4];
        buf[1] = 0x42;
        let p = buf.as_mut_ptr() as u64;
        for base in [RBX, RBP, RSI, R8, R12, R13, R15] {
            let got = run(
                |a| {
                    a.push(base);
                    a.mov_rr(base, RAX, true);
                    a.ld(RAX, mem(base, 8));
                    a.pop(base);
                },
                p,
                0,
            );
            assert_eq!(got, 0x42, "base {base}");
        }
        assert_eq!(
            run(
                |a| {
                    a.alu_ri(SUB, RSP, 16, true);
                    a.st(mem(RSP, 8), RAX, 8);
                    a.ld(RAX, mem(RSP, 8));
                    a.alu_ri(ADD, RSP, 16, true);
                },
                7,
                0
            ),
            7
        );
    }

    #[test]
    fn extensions() {
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 1, true), 0, 0x80), !0x7f);
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 1, false), !0, 0x1ff), 0xff);
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 2, true), 0, 0x8000), !0x7fff);
        assert_eq!(
            run(|a| a.ext_rr(RAX, RCX, 4, true), 0, 0x8000_0000),
            !0x7fff_ffff
        );
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 4, false), 0, !0), 0xffff_ffff);
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(RSI, 0x1ff);
                    a.ext_rr(RAX, RSI, 1, false);
                },
                0,
                0
            ),
            0xff
        );
    }

    #[test]
    fn branches_and_flags() {
        let r = run(
            |a| {
                let l = a.label();
                a.alu_rr(CMP, RAX, RCX, true);
                a.mov_imm(RAX, 1);
                a.jcc(0xc, l);
                a.mov_imm(RAX, 2);
                a.bind(l);
            },
            1,
            5,
        );
        assert_eq!(r, 1);
        let r = run(
            |a| {
                a.alu_rr(ADD, RAX, RCX, true);
                a.sete(RDX);
                a.mov_imm(RAX, 0);
                a.ext_rr(RAX, RDX, 1, false);
            },
            u64::MAX,
            1,
        );
        assert_eq!(r, 1);
    }
}
