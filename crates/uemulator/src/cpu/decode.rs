use super::{SEG_CS, SEG_DS, SEG_ES, SEG_FS, SEG_GS, SEG_SS};

/// One decoded instruction. Cached per guest physical address in `Cpu::decode_cache` and
/// consumed by the interpreter (`exec.rs`) and the JIT front ends (`jit/`), so field meanings
/// are shared contracts.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Insn {
    /// Total encoded length in bytes, prefixes included (at most `MAX_INSN_LEN`).
    pub len: u8,
    /// `(map << 8) | opcode`, where map is 0 for one-byte opcodes, 1 for `0F xx`, 2 for
    /// `0F 38 xx` and 3 for `0F 3A xx`. The JIT uses `0xffff` for an undecodable instruction.
    pub op: u16,
    /// Last repeat prefix seen: 0 none, 1 = `F3` (rep/repe), 2 = `F2` (repne). For SSE opcodes
    /// the same prefixes select the scalar/other variant instead.
    pub rep: u8,
    pub lock: bool,
    /// `66` prefix seen. Kept even when REX.W overrides it in `opsize`, because push/pop and
    /// SSE opcodes look at the prefix itself.
    pub p66: bool,
    /// Raw REX byte (`0x40..=0x4f`), or 0 when absent.
    pub rex: u8,
    /// Distinguishes a bare `0x40` REX from none: it changes which byte register `spl/bpl/sil/dil`
    /// vs `ah/ch/dh/bh` is addressed by register numbers 4-7.
    pub rex_present: bool,
    /// Operand size in bytes from REX.W and `66` only (8, 2 or 4). Byte-sized opcodes and opcodes
    /// with a fixed or default-64-bit size (push/pop, near branches) must not use this blindly.
    pub opsize: u8,
    /// `67` prefix: 32-bit effective addresses (and a 4-byte `Moffs`).
    pub addr32: bool,
    /// Segment-override prefix as an index into `Cpu::seg` (`SEG_ES`..`SEG_GS`), or `NO_SEG`.
    /// The last override wins. The JIT also indexes `Cpu::seg` with it directly.
    pub seg: u8,
    pub has_modrm: bool,
    pub modrm: u8,
    /// ModRM.mod: 3 means register operand, otherwise memory.
    pub md: u8,
    /// ModRM.reg extended with REX.R (bit 3). For group opcodes the opcode extension is
    /// `reg & 7`.
    pub reg: u8,
    /// ModRM.rm extended with REX.B. Only a register number when `md == 3`; memory operands use
    /// `base`/`index`.
    pub rm: u8,
    /// Memory base register (REX.B applied), or `NO_REG` for RIP-relative and disp32-only forms.
    pub base: u8,
    /// SIB index register (REX.X applied), or `NO_REG` when there is no index.
    pub index: u8,
    /// SIB scale as log2 (0..=3).
    pub scale: u8,
    /// ModRM.mod == 0 with rm == 5 and no SIB: `disp` is relative to the next instruction.
    pub rip_rel: bool,
    /// Sign-extended displacement.
    pub disp: i64,
    /// Sign-extended immediate, except `ret imm16` and `enter` sizes, which are zero-extended.
    /// Also holds relative branch offsets (relative to the next instruction) and the absolute
    /// address of `Moffs` forms.
    pub imm: i64,
    /// Nesting level of `enter` (second immediate).
    pub imm2: u16,
}

/// Marks an absent `base`/`index` register.
pub(crate) const NO_REG: u8 = 0xff;
/// No segment-override prefix.
pub(crate) const NO_SEG: u8 = 0xff;

/// Architectural instruction length limit; a longer encoding is #UD.
const MAX_INSN_LEN: usize = 15;

const PREFIX_OPERAND_SIZE: u8 = 0x66;
const PREFIX_ADDRESS_SIZE: u8 = 0x67;
const PREFIX_LOCK: u8 = 0xf0;
const PREFIX_REPNE: u8 = 0xf2;
const PREFIX_REP: u8 = 0xf3;
const PREFIX_SEG_ES: u8 = 0x26;
const PREFIX_SEG_CS: u8 = 0x2e;
const PREFIX_SEG_SS: u8 = 0x36;
const PREFIX_SEG_DS: u8 = 0x3e;
const PREFIX_SEG_FS: u8 = 0x64;
const PREFIX_SEG_GS: u8 = 0x65;

const REX_FIRST: u8 = 0x40;
const REX_LAST: u8 = 0x4f;
const REX_W: u8 = 8;
const REX_R: u8 = 4;
const REX_X: u8 = 2;
const REX_B: u8 = 1;

const OPCODE_ESCAPE_0F: u8 = 0x0f;
const OPCODE_ESCAPE_0F38: u8 = 0x38;
const OPCODE_ESCAPE_0F3A: u8 = 0x3a;

/// Opcode map numbers, stored in the high byte of `Insn::op`.
const MAP_ONE_BYTE: u16 = 0;
const MAP_0F: u16 = 1;
const MAP_0F38: u16 = 2;
const MAP_0F3A: u16 = 3;

/// ModRM.rm value that selects a SIB byte (mod != 3), and the rm/base value that means
/// disp32 (mod == 0) instead of a register.
const RM_SIB_FOLLOWS: u8 = 4;
const RM_DISP32_OR_RIP: u8 = 5;
const SIB_BASE_DISP32: u8 = 5;
/// SIB.index value meaning "no index" (before REX.X is applied, so r12 remains usable).
const SIB_INDEX_NONE: u8 = 4;

const MODRM_MOD_DISP8: u8 = 1;
const MODRM_MOD_DISP32: u8 = 2;
const MODRM_MOD_REGISTER: u8 = 3;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DecodeErr {
    /// The byte slice ended in the middle of the instruction. The caller supplies more bytes
    /// (an instruction may straddle a page boundary) and retries; at the full 15 bytes it is #UD.
    Truncated,
    /// Not decodable as an instruction in 64-bit mode (opcode removed in long mode, or longer
    /// than 15 bytes). Becomes #UD.
    Invalid,
}

/// Immediate operand encodings, named after the SDM operand-type letters.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Imm {
    None,
    I8,
    I16,
    /// 16 bits with a 66 prefix, else 32 bits (sign-extended under REX.W).
    Iz,
    /// Full operand size, up to 64 bits (`mov r64, imm64`).
    Iv,
    /// Absolute address operand of `mov al/eax, [moffs]`: 8 bytes, or 4 with a 67 prefix.
    Moffs,
    Enter,
    Rel8,
    Rel32,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u8(&mut self) -> Result<u8, DecodeErr> {
        let v = *self.bytes.get(self.pos).ok_or(DecodeErr::Truncated)?;
        self.pos += 1;
        if self.pos > MAX_INSN_LEN {
            return Err(DecodeErr::Invalid);
        }
        Ok(v)
    }

    /// Reads `n` little-endian bytes and sign-extends from that width (8 bytes is returned as is).
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

fn imm_kind(map: u16, opcode: u8) -> Imm {
    match map {
        MAP_ONE_BYTE => match opcode {
            // ALU group opcodes 00-3f: low three bits 4 and 5 are `op al, imm8` / `op eax, imm32`.
            0x00..=0x3f => match opcode & 7 {
                4 => Imm::I8,
                5 => Imm::Iz,
                _ => Imm::None,
            },
            // 68/69 push/imul imm32, 81 group 1, c7 mov, a9 test eax
            0x68 | 0x69 | 0x81 | 0xc7 | 0xa9 => Imm::Iz,
            // 6a/6b push/imul imm8, 80/83 group 1, c0/c1 shift groups, c6 mov, a8 test al, cd int
            0x6a | 0x6b | 0x80 | 0x83 | 0xc0 | 0xc1 | 0xc6 | 0xa8 | 0xcd => Imm::I8,
            // jcc, loopne/loope/loop/jrcxz, jmp short
            0x70..=0x7f | 0xe0..=0xe3 | 0xeb => Imm::Rel8,
            0xe8 | 0xe9 => Imm::Rel32,
            0xa0..=0xa3 => Imm::Moffs,
            // mov r8, imm8 and in/out with imm8 port
            0xb0..=0xb7 | 0xe4..=0xe7 => Imm::I8,
            0xb8..=0xbf => Imm::Iv,
            // ret/retf imm16
            0xc2 | 0xca => Imm::I16,
            0xc8 => Imm::Enter,
            // f6/f7 /0 and /1 (test) take an immediate; `decode` handles that using ModRM.reg.
            _ => Imm::None,
        },
        MAP_0F => match opcode {
            // jcc rel32
            0x80..=0x8f => Imm::Rel32,
            // pshuf* (70), psrl/psra/psll groups (71-73), shld/shrd imm8, bt group,
            // cmpps, pinsrw, pextrw, shufps
            0x70..=0x73 | 0xa4 | 0xac | 0xba | 0xc2 | 0xc4..=0xc6 => Imm::I8,
            _ => Imm::None,
        },
        MAP_0F3A => Imm::I8,
        _ => Imm::None,
    }
}

fn has_modrm(map: u16, opcode: u8) -> bool {
    match map {
        MAP_ONE_BYTE => match opcode {
            // ALU group: only the four `op r/m, r` / `op r, r/m` forms of each row take ModRM
            0x00..=0x3f => opcode & 7 < 4,
            // 62/c4/c5 are rejected earlier as EVEX/VEX; 63 movsxd, 69/6b imul
            0x62 | 0x63 | 0x69 | 0x6b | 0x80..=0x8f | 0xc0 | 0xc1 | 0xc4 | 0xc5 | 0xc6 | 0xc7 => {
                true
            }
            0xd0..=0xd3 | 0xd8..=0xdf | 0xf6 | 0xf7 | 0xfe | 0xff => true,
            _ => false,
        },
        // 0F opcodes without ModRM: syscall/clts/sysret/invd/wbinvd (05-09), ud2, femms,
        // wrmsr/rdtsc/rdmsr/rdpmc/sysenter/sysexit/getsec (30-37), emms, jcc rel32, push/pop fs/gs,
        // cpuid, rsm, bswap.
        MAP_0F => !matches!(
            opcode,
            0x05..=0x09 | 0x0b | 0x0e | 0x30..=0x37 | 0x77 | 0x80..=0x8f | 0xa0..=0xa2
                | 0xa8..=0xaa | 0xc8..=0xcf
        ),
        _ => true,
    }
}

/// One-byte opcodes that are invalid in 64-bit mode, plus VEX/EVEX (`c4`, `c5`, `62`), which are
/// not supported because AVX is never advertised.
fn invalid_in_long_mode(opcode: u8) -> bool {
    matches!(
        opcode,
        0x06 | 0x07 // push/pop es
            | 0x0e // push cs
            | 0x16 | 0x17 // push/pop ss
            | 0x1e | 0x1f // push/pop ds
            | 0x27 | 0x2f | 0x37 | 0x3f // daa das aaa aas
            | 0x60 | 0x61 // pusha popa
            | 0x62 // bound (EVEX)
            | 0x82 // group 1 alias
            | 0x9a // far call
            | 0xc4 | 0xc5 // les/lds (VEX)
            | 0xce // into
            | 0xd4 | 0xd5 | 0xd6 // aam aad salc
            | 0xea // far jmp
    )
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Insn, DecodeErr> {
    let mut r = Reader { bytes, pos: 0 };
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
            PREFIX_OPERAND_SIZE => i.p66 = true,
            PREFIX_ADDRESS_SIZE => i.addr32 = true,
            PREFIX_LOCK => i.lock = true,
            PREFIX_REPNE => i.rep = 2,
            PREFIX_REP => i.rep = 1,
            PREFIX_SEG_ES => i.seg = SEG_ES as u8,
            PREFIX_SEG_CS => i.seg = SEG_CS as u8,
            PREFIX_SEG_SS => i.seg = SEG_SS as u8,
            PREFIX_SEG_DS => i.seg = SEG_DS as u8,
            PREFIX_SEG_FS => i.seg = SEG_FS as u8,
            PREFIX_SEG_GS => i.seg = SEG_GS as u8,
            _ => break b,
        }
    };
    // 40-4f are always REX here (never inc/dec). Only a REX directly before the opcode is
    // honored; a REX followed by a legacy prefix is not re-scanned for prefixes.
    if (REX_FIRST..=REX_LAST).contains(&op) {
        i.rex = op;
        i.rex_present = true;
        rexw = op & REX_W != 0;
        op = r.u8()?;
    }
    let mut map = MAP_ONE_BYTE;
    let mut opb = op;
    if op == OPCODE_ESCAPE_0F {
        opb = r.u8()?;
        map = MAP_0F;
        if opb == OPCODE_ESCAPE_0F38 || opb == OPCODE_ESCAPE_0F3A {
            map = if opb == OPCODE_ESCAPE_0F38 {
                MAP_0F38
            } else {
                MAP_0F3A
            };
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

    if map == MAP_ONE_BYTE && invalid_in_long_mode(opb) {
        return Err(DecodeErr::Invalid);
    }

    if has_modrm(map, opb) {
        let m = r.u8()?;
        i.has_modrm = true;
        i.modrm = m;
        i.md = m >> 6;
        // REX.R/X/B extend the 3-bit fields to 4 bits; shift moves the REX bit to bit 3.
        i.reg = ((m >> 3) & 7) | ((i.rex & REX_R) << 1);
        i.rm = (m & 7) | ((i.rex & REX_B) << 3);
        if i.md != MODRM_MOD_REGISTER {
            let mut base_field = m & 7;
            if base_field == RM_SIB_FOLLOWS {
                let sib = r.u8()?;
                i.scale = sib >> 6;
                let index_field = ((sib >> 3) & 7) | ((i.rex & REX_X) << 2);
                if index_field != SIB_INDEX_NONE {
                    i.index = index_field;
                }
                base_field = sib & 7;
                // mod 0 with SIB base 5 means no base register, disp32 follows (index may remain)
                if base_field == SIB_BASE_DISP32 && i.md == 0 {
                    i.disp = r.int(4)?;
                } else {
                    i.base = base_field | ((i.rex & REX_B) << 3);
                }
            } else if base_field == RM_DISP32_OR_RIP && i.md == 0 {
                i.rip_rel = true;
                i.disp = r.int(4)?;
            } else {
                i.base = base_field | ((i.rex & REX_B) << 3);
            }
            match i.md {
                MODRM_MOD_DISP8 => i.disp = r.int(1)?,
                MODRM_MOD_DISP32 => i.disp = r.int(4)?,
                _ => {}
            }
        }
    }

    let mut kind = imm_kind(map, opb);
    // group 3: only /0 and /1 (test) carry an immediate; not/neg/mul/div do not
    if map == MAP_ONE_BYTE && (opb == 0xf6 || opb == 0xf7) && ((i.modrm >> 3) & 7) < 2 {
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
    i.len = r.pos as u8;
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
