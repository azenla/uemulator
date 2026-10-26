#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Insn {
    pub len: u8,
    pub op: u16,
    pub rep: u8,
    pub lock: bool,
    pub p66: bool,
    pub rex: u8,
    pub rex_present: bool,
    pub opsize: u8,
    pub addr32: bool,
    pub seg: u8,
    pub has_modrm: bool,
    pub modrm: u8,
    pub md: u8,
    pub reg: u8,
    pub rm: u8,
    pub base: u8,
    pub index: u8,
    pub scale: u8,
    pub rip_rel: bool,
    pub disp: i64,
    pub imm: i64,
    pub imm2: u16,
}

pub(crate) const NO_REG: u8 = 0xff;
pub(crate) const NO_SEG: u8 = 0xff;
pub(crate) const MAP_0F: u16 = 0x100;
pub(crate) const MAP_0F38: u16 = 0x200;
pub(crate) const MAP_0F3A: u16 = 0x300;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DecodeErr {
    Truncated,
    Invalid,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Imm {
    None,
    I8,
    I16,
    Iz,
    Iv,
    Moffs,
    Enter,
    Rel8,
    Rel32,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn u8(&mut self) -> Result<u8, DecodeErr> {
        let v = *self.b.get(self.p).ok_or(DecodeErr::Truncated)?;
        self.p += 1;
        if self.p > 15 {
            return Err(DecodeErr::Invalid);
        }
        Ok(v)
    }

    fn int(&mut self, n: usize) -> Result<i64, DecodeErr> {
        let mut v: u64 = 0;
        for i in 0..n {
            v |= (self.u8()? as u64) << (8 * i);
        }
        Ok(match n {
            1 => v as u8 as i8 as i64,
            2 => v as u16 as i16 as i64,
            4 => v as u32 as i32 as i64,
            _ => v as i64,
        })
    }
}

fn imm_kind(map: u16, op: u8) -> Imm {
    match map {
        0 => match op {
            0x00..=0x3f => match op & 7 {
                4 => Imm::I8,
                5 => Imm::Iz,
                _ => Imm::None,
            },
            0x68 | 0x69 | 0x81 | 0xc7 | 0xa9 => Imm::Iz,
            0x6a | 0x6b | 0x80 | 0x83 | 0xc0 | 0xc1 | 0xc6 | 0xa8 | 0xcd => Imm::I8,
            0x70..=0x7f | 0xe0..=0xe3 | 0xeb => Imm::Rel8,
            0xe8 | 0xe9 => Imm::Rel32,
            0xa0..=0xa3 => Imm::Moffs,
            0xb0..=0xb7 | 0xe4..=0xe7 => Imm::I8,
            0xb8..=0xbf => Imm::Iv,
            0xc2 | 0xca => Imm::I16,
            0xc8 => Imm::Enter,
            _ => Imm::None,
        },
        1 => match op {
            0x80..=0x8f => Imm::Rel32,
            0x70..=0x73 | 0xa4 | 0xac | 0xba | 0xc2 | 0xc4..=0xc6 => Imm::I8,
            _ => Imm::None,
        },
        3 => Imm::I8,
        _ => Imm::None,
    }
}

fn has_modrm(map: u16, op: u8) -> bool {
    match map {
        0 => match op {
            0x00..=0x3f => op & 7 < 4,
            0x62 | 0x63 | 0x69 | 0x6b | 0x80..=0x8f | 0xc0 | 0xc1 | 0xc4 | 0xc5 | 0xc6 | 0xc7 => {
                true
            }
            0xd0..=0xd3 | 0xd8..=0xdf | 0xf6 | 0xf7 | 0xfe | 0xff => true,
            _ => false,
        },
        1 => !matches!(
            op,
            0x05..=0x09 | 0x0b | 0x0e | 0x30..=0x37 | 0x77 | 0x80..=0x8f | 0xa0..=0xa2
                | 0xa8..=0xaa | 0xc8..=0xcf
        ),
        _ => true,
    }
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Insn, DecodeErr> {
    let mut r = Reader { b: bytes, p: 0 };
    let mut i = Insn {
        seg: NO_SEG,
        base: NO_REG,
        index: NO_REG,
        ..Default::default()
    };
    let mut rexw = false;
    let mut op = loop {
        let b = r.u8()?;
        match b {
            0x66 => i.p66 = true,
            0x67 => i.addr32 = true,
            0xf0 => i.lock = true,
            0xf2 => i.rep = 2,
            0xf3 => i.rep = 1,
            0x26 => i.seg = 0,
            0x2e => i.seg = 1,
            0x36 => i.seg = 2,
            0x3e => i.seg = 3,
            0x64 => i.seg = 4,
            0x65 => i.seg = 5,
            _ => break b,
        }
    };
    if (0x40..=0x4f).contains(&op) {
        i.rex = op;
        i.rex_present = true;
        rexw = op & 8 != 0;
        op = r.u8()?;
    }
    let mut map = 0u16;
    let mut opb = op;
    if op == 0x0f {
        opb = r.u8()?;
        map = 1;
        if opb == 0x38 || opb == 0x3a {
            map = if opb == 0x38 { 2 } else { 3 };
            opb = r.u8()?;
        }
    }
    i.op = (map << 8) | opb as u16;
    i.opsize = if rexw {
        8
    } else if i.p66 {
        2
    } else {
        4
    };

    if map == 0
        && matches!(
            opb,
            0x06 | 0x07
                | 0x0e
                | 0x16
                | 0x17
                | 0x1e
                | 0x1f
                | 0x27
                | 0x2f
                | 0x37
                | 0x3f
                | 0x60
                | 0x61
                | 0x62
                | 0x82
                | 0x9a
                | 0xc4
                | 0xc5
                | 0xce
                | 0xd4
                | 0xd5
                | 0xd6
                | 0xea
        )
    {
        return Err(DecodeErr::Invalid);
    }

    if has_modrm(map, opb) {
        let m = r.u8()?;
        i.has_modrm = true;
        i.modrm = m;
        i.md = m >> 6;
        i.reg = ((m >> 3) & 7) | ((i.rex & 4) << 1);
        i.rm = (m & 7) | ((i.rex & 1) << 3);
        if i.md != 3 {
            let mut base_field = m & 7;
            if base_field == 4 {
                let sib = r.u8()?;
                i.scale = sib >> 6;
                let idx = ((sib >> 3) & 7) | ((i.rex & 2) << 2);
                if idx != 4 {
                    i.index = idx;
                }
                base_field = sib & 7;
                if base_field == 5 && i.md == 0 {
                    i.disp = r.int(4)?;
                } else {
                    i.base = base_field | ((i.rex & 1) << 3);
                }
            } else if base_field == 5 && i.md == 0 {
                i.rip_rel = true;
                i.disp = r.int(4)?;
            } else {
                i.base = base_field | ((i.rex & 1) << 3);
            }
            match i.md {
                1 => i.disp = r.int(1)?,
                2 => i.disp = r.int(4)?,
                _ => {}
            }
        }
    }

    let mut kind = imm_kind(map, opb);
    if map == 0 && (opb == 0xf6 || opb == 0xf7) && ((i.modrm >> 3) & 7) < 2 {
        kind = if opb == 0xf6 { Imm::I8 } else { Imm::Iz };
    }
    match kind {
        Imm::None => {}
        Imm::I8 | Imm::Rel8 => i.imm = r.int(1)?,
        Imm::I16 => i.imm = r.int(2)? & 0xffff,
        Imm::Iz => i.imm = r.int(if i.opsize == 2 { 2 } else { 4 })?,
        Imm::Iv => i.imm = r.int(i.opsize as usize)?,
        Imm::Rel32 => i.imm = r.int(4)?,
        Imm::Moffs => i.imm = r.int(if i.addr32 { 4 } else { 8 })?,
        Imm::Enter => {
            i.imm = r.int(2)? & 0xffff;
            i.imm2 = r.int(1)? as u8 as u16;
        }
    }
    i.len = r.p as u8;
    Ok(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        let cases: &[(&[u8], u8)] = &[
            (&[0x90], 1),
            (&[0x48, 0x89, 0xe5], 3),
            (&[0x48, 0xc7, 0xc0, 1, 0, 0, 0], 7),
            (&[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8], 10),
            (&[0x8b, 0x04, 0x25, 0, 0, 0, 0], 7),
            (&[0x48, 0x8d, 0x05, 0, 0, 0, 0], 7),
            (&[0xf3, 0x0f, 0x1e, 0xfa], 4),
            (&[0x0f, 0x1f, 0x44, 0x00, 0x00], 5),
            (&[0x66, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0], 9),
            (&[0xf7, 0x45, 0xfc, 1, 0, 0, 0], 7),
            (&[0xf7, 0xd0], 2),
            (&[0xe8, 0, 0, 0, 0], 5),
            (&[0x0f, 0x84, 0, 0, 0, 0], 6),
            (&[0x66, 0x83, 0xc0, 1], 4),
            (&[0xc8, 0x10, 0, 0], 4),
        ];
        for (b, n) in cases {
            assert_eq!(decode(b).unwrap().len, *n, "{b:02x?}");
        }
    }

    #[test]
    fn truncated() {
        assert_eq!(decode(&[0x48]).unwrap_err(), DecodeErr::Truncated);
        assert_eq!(decode(&[0x8b, 0x04]).unwrap_err(), DecodeErr::Truncated);
    }
}
