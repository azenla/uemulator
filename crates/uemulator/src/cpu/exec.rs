use super::decode::*;
use super::exception::{Exception, VECTOR_SEGMENT_NOT_PRESENT, VECTOR_STACK_FAULT};
use super::flags::*;
use super::mem::AccessKind;
use super::sys::{CR0_TS, CR4_TSD};
use super::*;

// Indices into `Cpu::gpr` (architectural register numbers); GPR_RSP and GPR_RSI come from mod.rs.
const GPR_RAX: usize = 0;
const GPR_RCX: usize = 1;
const GPR_RDX: usize = 2;
const GPR_RBX: usize = 3;
const GPR_RBP: usize = 5;
const GPR_RDI: usize = 7;

/// Register numbers are 4 bits wide once REX extends the ModRM/opcode field.
const GPR_INDEX_MASK: usize = 15;

/// Without a REX prefix, byte-register numbers 4..=7 mean AH, CH, DH, BH (bits 15:8 of
/// registers 0..=3) instead of SPL, BPL, SIL, DIL.
const BYTE_REG_AH: u8 = 4;
const BYTE_REG_BH: u8 = 7;

const BYTE_MASK: u64 = 0xff;
const WORD_MASK: u64 = 0xffff;
const DWORD_MASK: u64 = 0xffff_ffff;

const REX_B: u8 = 1;

/// `Insn::rep` values: F3 is rep/repe, F2 is repne.
const REP_REPE: u8 = 1;
const REP_REPNE: u8 = 2;

/// A rep string instruction returns to the run loop after this many iterations (RIP unchanged,
/// so it resumes where it left off) so pending interrupts and device events are still serviced.
const REP_ITERATIONS_PER_SLICE: u64 = 4096;

const VECTOR_BREAKPOINT: u8 = 3;

/// The low two bits of a selector are the RPL; the rest (index and TI) identify the descriptor.
const SELECTOR_RPL_MASK: u16 = 3;

const EFLAGS_ID: u64 = 1 << 21;
const IOPL_SHIFT: u32 = 12;

/// EFLAGS bits popf/iretq load at any CPL. IF is added when CPL <= IOPL and IOPL when CPL is 0
/// (see `Cpu::loadable_eflags`). RF and VM are never loaded by either.
const EFLAGS_ALWAYS_LOADABLE: u64 = CF | PF | AF | ZF | SF | TF | DF | OF | NT | AC | EFLAGS_ID;

/// sahf/lahf transfer these flags (SF, ZF, AF, PF, CF) to and from AH.
const LAHF_SAHF_FLAGS: u64 = SF | ZF | AF | PF | CF;

/// Shift/rotate counts are masked to 5 bits, or 6 bits for 64-bit operands.
const SHIFT_COUNT_MASK: u64 = 0x1f;
const SHIFT_COUNT_MASK_64BIT: u64 = 0x3f;

/// Offset of the 16-bit I/O permission bitmap base in a 64-bit TSS.
const TSS64_IO_MAP_BASE_OFFSET: u64 = 102;

/// ENTER takes its nesting level modulo 32.
const ENTER_LEVEL_MASK: u16 = 31;

/// `log_once` key space for unimplemented opcodes; other callers (mem.rs, sys.rs) use other bases.
const UNIMPLEMENTED_OPCODE_LOG_BASE: u64 = 0x4000_0000;

// ALU operation numbers: bits 5:3 of opcodes 0x00..=0x3f and ModRM.reg of group 1 (0x80/81/83).
// 6 is xor and is the fallthrough arm of `alu`.
const ALU_ADD: u8 = 0;
const ALU_OR: u8 = 1;
const ALU_ADC: u8 = 2;
const ALU_SBB: u8 = 3;
const ALU_AND: u8 = 4;
const ALU_SUB: u8 = 5;
const ALU_CMP: u8 = 7;

// ModRM.reg of the shift/rotate group (0xc0/c1/d0..d3). /6 is an undocumented alias of SHL and
// /7 (sar) is the fallthrough arm of `shift_rotate`.
const SHIFT_ROL: u8 = 0;
const SHIFT_ROR: u8 = 1;
const SHIFT_RCL: u8 = 2;
const SHIFT_RCR: u8 = 3;
const SHIFT_SHL: u8 = 4;
const SHIFT_SHR: u8 = 5;
const SHIFT_SAL_ALIAS: u8 = 6;

// ModRM.reg of group 3 (0xf6/f7); /1 is an alias of /0 (test), /7 is idiv.
const GROUP3_TEST: u8 = 0;
const GROUP3_TEST_ALIAS: u8 = 1;
const GROUP3_NOT: u8 = 2;
const GROUP3_NEG: u8 = 3;
const GROUP3_MUL: u8 = 4;
const GROUP3_IMUL: u8 = 5;
const GROUP3_DIV: u8 = 6;

// ModRM.reg of groups 4/5 (0xfe/0xff).
const GROUP5_INC: u8 = 0;
const GROUP5_DEC: u8 = 1;
const GROUP5_CALL_NEAR: u8 = 2;
const GROUP5_JMP_NEAR: u8 = 4;
const GROUP5_PUSH: u8 = 6;

// Bit-test family: bt, bts, btr, btc.
const BIT_TEST: u8 = 0;
const BIT_SET: u8 = 1;
const BIT_RESET: u8 = 2;
const BIT_COMPLEMENT: u8 = 3;
/// Group 8 (0f ba): ModRM.reg 4..=7 are bt/bts/btr/btc with an imm8 bit offset.
const GROUP8_FIRST_BIT_OP: u8 = 4;

/// Operand location after ModRM decoding. Memory addresses are linear (segment base applied).
#[derive(Clone, Copy)]
pub(crate) enum Loc {
    Reg(u8),
    Mem(u64),
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// The flag state `Cpu::save_flags` captures: `eflags` plus the deferred-arithmetic fields.
#[derive(Clone, Copy)]
struct FlagsSnapshot {
    eflags: u64,
    lazy_op: LazyFlagsOp,
    lazy_size: u8,
    lazy_dst: u64,
    lazy_src: u64,
    lazy_src2: u64,
}

/// A null selector has index 0 and TI 0 whatever its RPL. A selector with TI set and index 0
/// refers to LDT entry 0 and is not null, so TI stays part of the comparison.
fn selector_is_null(sel: u16) -> bool {
    sel & !SELECTOR_RPL_MASK == 0
}

/// Descriptor type bit 10 of a code segment: conforming (callable from less privileged code and
/// exempt from the DPL checks).
const SEG_TYPE_CONFORMING: u32 = 1 << 10;

fn is_conforming_code(seg: &Seg) -> bool {
    seg.is_code() && seg.flags & SEG_TYPE_CONFORMING != 0
}

/// Error code for a selector-related fault: the selector with RPL cleared (EXT bit 0).
fn selector_error_code(sel: u16) -> u32 {
    (sel & !SELECTOR_RPL_MASK) as u32
}

/// Operand size of push/pop/pushf/popf in 64-bit mode: 8 bytes, or 2 with a 66 prefix. REX.W
/// is irrelevant and 4 is not encodable, so `Insn::opsize` must not be used for these.
fn stack_operand_size(i: &Insn) -> u8 {
    if i.p66 { 2 } else { 8 }
}

/// Register number embedded in the low opcode bits (push/pop/mov imm/xchg/bswap), extended by REX.B.
fn opcode_register(i: &Insn) -> u8 {
    (i.op as u8 & 7) | ((i.rex & REX_B) << 3)
}

fn address_mask(i: &Insn) -> u64 {
    if i.addr32 { DWORD_MASK } else { u64::MAX }
}

fn shift_count_mask(size: u8) -> u64 {
    if size == 8 {
        SHIFT_COUNT_MASK_64BIT
    } else {
        SHIFT_COUNT_MASK
    }
}

/// Result and carry of shld (`left`) or shrd for a nonzero `count` already masked to the
/// operand's range, with `a` the destination and `b` the source.
///
/// A 16-bit operand with a count of 17..=31 is undefined in the SDM. Intel hardware shifts the
/// 48-bit value A:B:A, which is what this does (QEMU's `gen_shiftd_rm_T1` implements the same); it
/// also gives the ordinary result for counts up to 16, and keeps `bits - count` from underflowing.
fn double_shift(left: bool, a: u64, b: u64, count: u32, size: u8) -> (u64, u64) {
    let m = mask(size);
    if size == 2 {
        // Bits of the shifted value above bit 63 are lost, but only bits 47:32 (left) or 15:0
        // (right) of the result are used.
        let wide = (a << 32) | (b << 16) | a;
        return if left {
            (
                ((wide << count) >> 32) & m,
                ((wide << (count - 1)) >> 47) & 1,
            )
        } else {
            ((wide >> count) & m, (wide >> (count - 1)) & 1)
        };
    }
    let bits = size as u32 * 8;
    if left {
        (
            ((a << count) | (b >> (bits - count))) & m,
            (a >> (bits - count)) & 1,
        )
    } else {
        (
            ((a >> count) | (b << (bits - count))) & m,
            (a >> (count - 1)) & 1,
        )
    }
}

/// Joins EDX:EAX style halves (wrmsr, cmpxchg8b); the upper 32 bits of both inputs are ignored.
fn combine_dwords(low: u64, high: u64) -> u64 {
    (low & DWORD_MASK) | (high << 32)
}

impl Cpu {
    /// Reads a general register at the given operand size. With `rex` false, byte registers 4..=7
    /// are AH/CH/DH/BH.
    pub fn read_gpr(&self, r: u8, size: u8, rex: bool) -> u64 {
        let g = self.gpr[r as usize & GPR_INDEX_MASK];
        match size {
            1 => {
                if !rex && (BYTE_REG_AH..=BYTE_REG_BH).contains(&r) {
                    (self.gpr[(r - BYTE_REG_AH) as usize] >> 8) & BYTE_MASK
                } else {
                    g & BYTE_MASK
                }
            }
            2 => g & WORD_MASK,
            4 => g & DWORD_MASK,
            _ => g,
        }
    }

    /// Writes a general register. 8- and 16-bit writes merge into the old value; a 32-bit write
    /// zero-extends into the full register (64-bit mode rule). `rex` as in `read_gpr`.
    pub fn write_gpr(&mut self, r: u8, size: u8, rex: bool, v: u64) {
        match size {
            1 => {
                if !rex && (BYTE_REG_AH..=BYTE_REG_BH).contains(&r) {
                    let g = &mut self.gpr[(r - BYTE_REG_AH) as usize];
                    *g = (*g & !(BYTE_MASK << 8)) | ((v & BYTE_MASK) << 8);
                } else {
                    let g = &mut self.gpr[r as usize];
                    *g = (*g & !BYTE_MASK) | (v & BYTE_MASK);
                }
            }
            2 => {
                let g = &mut self.gpr[r as usize];
                *g = (*g & !WORD_MASK) | (v & WORD_MASK);
            }
            4 => self.gpr[r as usize] = v & DWORD_MASK,
            _ => self.gpr[r as usize] = v,
        }
    }

    fn set_edx_eax(&mut self, v: u64) {
        self.gpr[GPR_RAX] = v & DWORD_MASK;
        self.gpr[GPR_RDX] = v >> 32;
    }

    fn iopl(&self) -> u8 {
        ((self.eflags & IOPL) >> IOPL_SHIFT) as u8
    }

    /// EFLAGS bits that popf/iretq may change when executed at `cpl`.
    fn loadable_eflags(&self, cpl: u8) -> u64 {
        let mut loadable = EFLAGS_ALWAYS_LOADABLE;
        if cpl <= self.iopl() {
            loadable |= IF;
        }
        if cpl == 0 {
            loadable |= IOPL;
        }
        loadable
    }

    /// Effective address (offset within the segment), truncated for the 67 prefix. RIP-relative
    /// displacements are relative to the end of the whole instruction, immediate included, which
    /// is what `next` is.
    pub fn effective_address(&self, i: &Insn, next: u64) -> u64 {
        let mut a = i.disp as u64;
        if i.rip_rel {
            a = a.wrapping_add(next);
        } else {
            if i.base != NO_REG {
                a = a.wrapping_add(self.gpr[i.base as usize]);
            }
            if i.index != NO_REG {
                a = a.wrapping_add(self.gpr[i.index as usize] << i.scale);
            }
        }
        if i.addr32 {
            a &= DWORD_MASK;
        }
        a
    }

    /// Applies the segment base. In 64-bit mode only FS and GS have a base; ES/CS/SS/DS
    /// overrides are ignored.
    pub fn linear_address(&self, i: &Insn, a: u64) -> u64 {
        let seg = i.seg as usize;
        match seg {
            SEG_FS | SEG_GS => a.wrapping_add(self.seg[seg].base),
            _ => a,
        }
    }

    pub fn modrm_location(&self, i: &Insn, next: u64) -> Loc {
        if i.md == 3 {
            Loc::Reg(i.rm)
        } else {
            Loc::Mem(self.linear_address(i, self.effective_address(i, next)))
        }
    }

    pub fn load_operand(&mut self, loc: Loc, size: u8, i: &Insn) -> ExecResult<u64> {
        match loc {
            Loc::Reg(r) => Ok(self.read_gpr(r, size, i.rex_present)),
            Loc::Mem(a) => self.read(a, size),
        }
    }

    pub fn store_operand(&mut self, loc: Loc, size: u8, i: &Insn, v: u64) -> ExecResult<()> {
        match loc {
            Loc::Reg(r) => {
                self.write_gpr(r, size, i.rex_present, v);
                Ok(())
            }
            Loc::Mem(a) => self.write(a, size, v),
        }
    }

    fn save_flags(&self) -> FlagsSnapshot {
        FlagsSnapshot {
            eflags: self.eflags,
            lazy_op: self.lazy_op,
            lazy_size: self.lazy_size,
            lazy_dst: self.lazy_dst,
            lazy_src: self.lazy_src,
            lazy_src2: self.lazy_src2,
        }
    }

    /// Stores the result of a read-modify-write instruction that has already updated the flags.
    /// If the store faults the flags are put back: the instruction is restarted after the fault
    /// and adc/sbb/rcl/rcr would otherwise read the carry the failed attempt left behind.
    fn store_result(
        &mut self,
        loc: Loc,
        size: u8,
        i: &Insn,
        v: u64,
        saved: FlagsSnapshot,
    ) -> ExecResult<()> {
        let result = self.store_operand(loc, size, i, v);
        if result.is_err() {
            self.eflags = saved.eflags;
            self.lazy_op = saved.lazy_op;
            self.lazy_size = saved.lazy_size;
            self.lazy_dst = saved.lazy_dst;
            self.lazy_src = saved.lazy_src;
            self.lazy_src2 = saved.lazy_src2;
        }
        result
    }

    /// GPR_RSP is only updated after the store succeeds so a fault leaves it unchanged.
    pub fn push(&mut self, size: u8, v: u64) -> ExecResult<()> {
        let sp = self.gpr[GPR_RSP].wrapping_sub(size as u64);
        self.write_stack(sp, size, v)?;
        self.gpr[GPR_RSP] = sp;
        Ok(())
    }

    pub fn pop(&mut self, size: u8) -> ExecResult<u64> {
        let v = self.read_stack(self.gpr[GPR_RSP], size)?;
        self.gpr[GPR_RSP] = self.gpr[GPR_RSP].wrapping_add(size as u64);
        Ok(v)
    }

    /// Evaluates a 4-bit condition code (jcc/setcc/cmovcc low opcode nibble). Bits 3:1 pick the
    /// condition and bit 0 negates it.
    pub fn condition_holds(&self, cc: u8) -> bool {
        let f = self.eflags_full();
        let sf = f & SF != 0;
        let of = f & OF != 0;
        let r = match cc >> 1 {
            0 => of,                      // o
            1 => f & CF != 0,             // b
            2 => f & ZF != 0,             // e
            3 => f & (CF | ZF) != 0,      // be
            4 => sf,                      // s
            5 => f & PF != 0,             // p
            6 => sf != of,                // l
            _ => f & ZF != 0 || sf != of, // le
        };
        r ^ (cc & 1 == 1)
    }

    /// Two-operand ALU op. CMP computes flags like SUB; the caller decides not to store it.
    fn alu(&mut self, operation: u8, a: u64, b: u64, size: u8) -> u64 {
        let m = mask(size);
        let (a, b) = (a & m, b & m);
        match operation {
            ALU_ADD => {
                let r = a.wrapping_add(b) & m;
                self.set_lazy_flags(LazyFlagsOp::Add, size, r, b, 0);
                r
            }
            ALU_OR => {
                let r = a | b;
                self.set_lazy_flags(LazyFlagsOp::Logic, size, r, 0, 0);
                r
            }
            ALU_ADC => {
                let carry = self.cf() as u64;
                let r = a.wrapping_add(b).wrapping_add(carry) & m;
                self.set_lazy_flags(LazyFlagsOp::Adc, size, r, b, carry);
                r
            }
            ALU_SBB => {
                let carry = self.cf() as u64;
                let r = a.wrapping_sub(b).wrapping_sub(carry) & m;
                self.set_lazy_flags(LazyFlagsOp::Sbb, size, r, b, carry);
                r
            }
            ALU_AND => {
                let r = a & b;
                self.set_lazy_flags(LazyFlagsOp::Logic, size, r, 0, 0);
                r
            }
            ALU_SUB | ALU_CMP => {
                let r = a.wrapping_sub(b) & m;
                self.set_lazy_flags(LazyFlagsOp::Sub, size, r, b, 0);
                r
            }
            _ => {
                let r = a ^ b;
                self.set_lazy_flags(LazyFlagsOp::Logic, size, r, 0, 0);
                r
            }
        }
    }

    /// Shifts and rotates. A count that is zero after masking changes neither the value nor any
    /// flag. OF is computed for every count, not only count 1 where the SDM defines it.
    fn shift_rotate(&mut self, operation: u8, value: u64, count: u64, size: u8) -> u64 {
        let bits = size as u32 * 8;
        let m = mask(size);
        let count = (count & shift_count_mask(size)) as u32;
        if count == 0 {
            return value & m;
        }
        let a = value & m;
        let fl = self.eflags_full();
        let msb = |v: u64| (v >> (bits - 1)) & 1;
        let second_msb = |v: u64| (v >> (bits - 2)) & 1;
        // (result, CF, OF, whether SF/ZF/PF come from the result)
        let (r, cf, of, szp) = match operation {
            SHIFT_ROL => {
                // Counts above the width wrap, but CF/OF are still updated when count % bits == 0.
                let rotate = count % bits;
                let r = if rotate == 0 {
                    a
                } else {
                    ((a << rotate) | (a >> (bits - rotate))) & m
                };
                let cf = r & 1;
                (r, cf, msb(r) ^ cf, false)
            }
            SHIFT_ROR => {
                let rotate = count % bits;
                let r = if rotate == 0 {
                    a
                } else {
                    ((a >> rotate) | (a << (bits - rotate))) & m
                };
                (r, msb(r), msb(r) ^ second_msb(r), false)
            }
            SHIFT_RCL => {
                // Rotating through CF is a cycle of bits + 1 positions.
                let mut r = a;
                let mut cf = fl & CF;
                for _ in 0..count % (bits + 1) {
                    let out = msb(r);
                    r = ((r << 1) | cf) & m;
                    cf = out;
                }
                (r, cf, msb(r) ^ cf, false)
            }
            SHIFT_RCR => {
                let mut r = a;
                let mut cf = fl & CF;
                for _ in 0..count % (bits + 1) {
                    let out = r & 1;
                    r = (r >> 1) | (cf << (bits - 1));
                    cf = out;
                }
                (r, cf, msb(r) ^ second_msb(r), false)
            }
            SHIFT_SHL | SHIFT_SAL_ALIAS => {
                let r = if count >= bits { 0 } else { (a << count) & m };
                let cf = if count <= bits {
                    (a >> (bits - count)) & 1
                } else {
                    0
                };
                (r, cf, msb(r) ^ cf, true)
            }
            SHIFT_SHR => {
                let r = if count >= bits { 0 } else { a >> count };
                let cf = if count <= bits {
                    (a >> (count - 1)) & 1
                } else {
                    0
                };
                (r, cf, msb(a), true)
            }
            _ => {
                // SAR: shift the sign-extended value so counts above the width fill with the sign.
                let signed = sign_extend(a, bits);
                let r = (signed >> count.min(63)) as u64 & m;
                let cf = ((signed >> (count - 1).min(63)) & 1) as u64;
                (r, cf, 0, true)
            }
        };
        let mut f = if szp {
            szp_flags(r, size)
        } else {
            fl & (PF | AF | ZF | SF)
        };
        if cf != 0 {
            f |= CF;
        }
        if of != 0 {
            f |= OF;
        }
        self.set_eflags_arith(f);
        r
    }

    /// Group 3 mul/imul/div/idiv (one-operand forms). Byte forms use AX as the wide register;
    /// the others use rDX:rAX. Division raises #DE for a zero divisor and for a quotient that does
    /// not fit the destination.
    fn mul_div(&mut self, i: &Insn, next: u64, size: u8) -> ExecResult<()> {
        let loc = self.modrm_location(i, next);
        let kind = i.reg & 7;
        let src = self.load_operand(loc, size, i)?;
        let bits = size as u32 * 8;
        let m = mask(size);
        let rax = self.gpr[GPR_RAX] & m;
        let rdx = self.gpr[GPR_RDX] & m;
        match kind {
            GROUP3_MUL | GROUP3_IMUL => {
                let (lo, hi, overflow) = if kind == GROUP3_MUL {
                    let p = rax as u128 * src as u128;
                    let lo = p as u64 & m;
                    let hi = (p >> bits) as u64 & m;
                    (lo, hi, hi != 0)
                } else {
                    let p = sign_extend(rax, bits) as i128 * sign_extend(src, bits) as i128;
                    let lo = p as u64 & m;
                    let hi = (p >> bits) as u64 & m;
                    // The high half is just the sign of the low half iff the product fits.
                    (lo, hi, p != sign_extend(lo, bits) as i128)
                };
                if size == 1 {
                    self.write_gpr(0, 2, true, lo | (hi << 8));
                } else {
                    self.write_gpr(0, size, true, lo);
                    self.write_gpr(2, size, true, hi);
                }
                let mut f = szp_flags(lo, size);
                if overflow {
                    f |= CF | OF;
                }
                self.set_eflags_arith(f);
            }
            _ => {
                if src == 0 {
                    return Err(Exception::divide_error());
                }
                let (quotient, remainder);
                if kind == GROUP3_DIV {
                    let dividend = if size == 1 {
                        self.gpr[GPR_RAX] as u128 & WORD_MASK as u128
                    } else {
                        (rdx as u128) << bits | rax as u128
                    };
                    let (q, r) = (dividend / src as u128, dividend % src as u128);
                    if q > m as u128 {
                        return Err(Exception::divide_error());
                    }
                    quotient = q as u64;
                    remainder = r as u64;
                } else {
                    let dividend: i128 = if size == 1 {
                        (self.gpr[GPR_RAX] as u16 as i16) as i128
                    } else {
                        // Sign-extend the 2 * bits wide rDX:rAX value to 128 bits.
                        (((rdx as u128) << bits | rax as u128) << (128 - 2 * bits)) as i128
                            >> (128 - 2 * bits)
                    };
                    let divisor = sign_extend(src, bits) as i128;
                    // i128::MIN / -1 overflows the host division itself (a panic, not a wrap). It
                    // is the one quotient that cannot fit any operand size, so it is #DE too.
                    let (Some(q), Some(r)) =
                        (dividend.checked_div(divisor), dividend.checked_rem(divisor))
                    else {
                        return Err(Exception::divide_error());
                    };
                    let limit = 1i128 << (bits - 1);
                    if q >= limit || q < -limit {
                        return Err(Exception::divide_error());
                    }
                    quotient = q as u64 & m;
                    remainder = r as u64 & m;
                }
                if size == 1 {
                    self.write_gpr(0, 2, true, quotient | (remainder << 8));
                } else {
                    self.write_gpr(0, size, true, quotient);
                    self.write_gpr(2, size, true, remainder);
                }
            }
        }
        Ok(())
    }

    /// Signed multiply truncated to `size` (imul r, r/m and the three-operand forms). CF and OF
    /// are set when the full product does not fit.
    fn imul_truncated(&mut self, a: u64, b: u64, size: u8) -> u64 {
        let bits = size as u32 * 8;
        let p = sign_extend(a, bits) as i128 * sign_extend(b, bits) as i128;
        let r = p as u64 & mask(size);
        let mut f = szp_flags(r, size);
        if p != sign_extend(r, bits) as i128 {
            f |= CF | OF;
        }
        self.set_eflags_arith(f);
        r
    }

    /// bt/bts/btr/btc. With a register bit offset and a memory operand the offset is signed and
    /// may reach outside the addressed operand: the address moves by whole operands and the bit
    /// index is the remainder. An immediate offset is taken modulo the operand width instead.
    fn bit_op(&mut self, i: &Insn, next: u64, operation: u8, imm_offset: bool) -> ExecResult<()> {
        let size = i.opsize;
        let bits = size as i64 * 8;
        let offset = if imm_offset {
            i.imm as u64
        } else {
            self.read_gpr(i.reg, size, true)
        };
        let loc = if i.md == 3 {
            Loc::Reg(i.rm)
        } else {
            let mut address = self.effective_address(i, next);
            if !imm_offset {
                let signed_offset = sign_extend(offset, bits as u32);
                address =
                    address.wrapping_add((signed_offset.div_euclid(bits) * size as i64) as u64);
            }
            Loc::Mem(self.linear_address(i, address))
        };
        let bit = if imm_offset || i.md == 3 {
            (offset as i64).rem_euclid(bits) as u32
        } else {
            sign_extend(offset, bits as u32).rem_euclid(bits) as u32
        };
        let v = self.load_operand(loc, size, i)?;
        let old = (v >> bit) & 1;
        let new_value = match operation {
            BIT_SET => v | (1 << bit),
            BIT_RESET => v & !(1 << bit),
            BIT_COMPLEMENT => v ^ (1 << bit),
            _ => v,
        };
        if operation != BIT_TEST {
            self.store_operand(loc, size, i, new_value)?;
        }
        let f = self.eflags_full();
        self.set_eflags_arith((f & !CF) | old);
        Ok(())
    }

    /// Bulk path of a forward `rep movs`/`rep stos` with 64-bit addressing: moves as many whole
    /// elements as fit in the current source and destination pages (and `budget`) with one page
    /// translation and one host copy or fill, then returns the count. Returns 0, having touched
    /// nothing, when the plain per-element loop must do it: an element straddling a page, a
    /// destination or source outside RAM (MMIO), a destination page holding translated code
    /// (`phys_write` invalidates it), or a forward overlap where the copy has to re-read what it
    /// just wrote (`rep movsb` as a memset idiom). A fault is raised exactly as the first
    /// element would raise it: the source is translated before the destination, and a fault on a
    /// later page comes from the next call with the registers advanced by what was done.
    fn rep_bulk(&mut self, i: &Insn, op: u8, size: u8, budget: u64) -> ExecResult<u64> {
        let element = size as u64;
        let copies = matches!(op, 0xa4 | 0xa5);
        let dst_linear = self.gpr[GPR_RDI];
        let src_linear = self.linear_address(i, self.gpr[GPR_RSI]);
        let room = |linear: u64| (PAGE_SIZE - (linear & PAGE_OFFSET_MASK)) / element;
        let mut elements = self.gpr[GPR_RCX].min(budget).min(room(dst_linear));
        if copies {
            elements = elements.min(room(src_linear));
        }
        if elements == 0 {
            return Ok(0);
        }
        let bytes = elements * element;
        let src_pa = if copies {
            Some(self.translate(src_linear, AccessKind::Read, false)?)
        } else {
            None
        };
        let dst_pa = self.translate(dst_linear, AccessKind::Write, false)?;
        let ram_bytes = self.ram.len() as u64;
        if dst_pa + bytes > ram_bytes || self.page_has_code[(dst_pa >> PAGE_SHIFT) as usize] != 0 {
            return Ok(0);
        }
        let ram = self.ram.as_mut_ptr();
        if let Some(src_pa) = src_pa {
            if src_pa + bytes > ram_bytes || (dst_pa > src_pa && dst_pa < src_pa + bytes) {
                return Ok(0);
            }
            // SAFETY: both ranges lie inside `ram` (checked above); a copy that moves memory
            // down (dst below src) may overlap, which `copy` allows.
            unsafe {
                core::ptr::copy(
                    ram.add(src_pa as usize),
                    ram.add(dst_pa as usize),
                    bytes as usize,
                );
            }
            self.gpr[GPR_RSI] = self.gpr[GPR_RSI].wrapping_add(bytes);
        } else {
            let value = self.read_gpr(0, size, true);
            // SAFETY: the destination range lies inside `ram`.
            unsafe {
                let dst = ram.add(dst_pa as usize);
                match size {
                    1 => core::ptr::write_bytes(dst, value as u8, bytes as usize),
                    _ => {
                        for k in 0..elements as usize {
                            let at = dst.add(k * size as usize);
                            match size {
                                2 => (at as *mut u16).write_unaligned(value as u16),
                                4 => (at as *mut u32).write_unaligned(value as u32),
                                _ => (at as *mut u64).write_unaligned(value),
                            }
                        }
                    }
                }
            }
        }
        self.gpr[GPR_RDI] = self.gpr[GPR_RDI].wrapping_add(bytes);
        self.gpr[GPR_RCX] -= elements;
        Ok(elements)
    }

    /// movs/stos/lods/cmps/scas with optional rep. State is committed per iteration, so a fault
    /// mid-rep leaves GPR_RSI/GPR_RDI/GPR_RCX consistent with the iterations already done.
    fn string_op(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op as u8;
        let size = if op & 1 == 0 { 1 } else { i.opsize };
        let amask = address_mask(i);
        let step: u64 = if self.eflags & DF != 0 {
            (size as u64).wrapping_neg()
        } else {
            size as u64
        };
        // Only the index registers an instruction uses are written back; with the 67 prefix that
        // is a 32-bit write, which zeroes the upper half. An unused one keeps its upper half.
        let advance = |index: u64| {
            let moved = index.wrapping_add(step);
            if i.addr32 { moved & DWORD_MASK } else { moved }
        };
        let rep = i.rep != 0;
        let compares = matches!(op, 0xa6 | 0xa7 | 0xae | 0xaf);
        let mut iterations = 0;
        loop {
            if rep && self.gpr[GPR_RCX] & amask == 0 {
                break;
            }
            if rep && matches!(op, 0xa4 | 0xa5 | 0xaa | 0xab) && !i.addr32 && self.eflags & DF == 0
            {
                let done = self.rep_bulk(i, op, size, REP_ITERATIONS_PER_SLICE - iterations)?;
                if done > 0 {
                    iterations += done;
                    if self.gpr[GPR_RCX] == 0 {
                        break;
                    }
                    if iterations >= REP_ITERATIONS_PER_SLICE {
                        self.icount += iterations;
                        return Ok(());
                    }
                    continue;
                }
            }
            // Only the source honours a segment override; the destination is always ES, which has
            // no base in 64-bit mode.
            let src = self.linear_address(i, self.gpr[GPR_RSI] & amask);
            let dst = self.gpr[GPR_RDI] & amask;
            match op {
                // movs
                0xa4 | 0xa5 => {
                    let v = self.read(src, size)?;
                    self.write(dst, size, v)?;
                    self.gpr[GPR_RSI] = advance(self.gpr[GPR_RSI]);
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
                // stos
                0xaa | 0xab => {
                    let v = self.read_gpr(0, size, true);
                    self.write(dst, size, v)?;
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
                // lods
                0xac | 0xad => {
                    let v = self.read(src, size)?;
                    self.write_gpr(0, size, true, v);
                    self.gpr[GPR_RSI] = advance(self.gpr[GPR_RSI]);
                }
                // cmps: flags of [rsi] - [rdi]
                0xa6 | 0xa7 => {
                    let a = self.read(src, size)?;
                    let b = self.read(dst, size)?;
                    self.set_lazy_flags(
                        LazyFlagsOp::Sub,
                        size,
                        a.wrapping_sub(b) & mask(size),
                        b,
                        0,
                    );
                    self.gpr[GPR_RSI] = advance(self.gpr[GPR_RSI]);
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
                // scas: flags of rAX - [rdi]
                _ => {
                    let a = self.read_gpr(0, size, true);
                    let b = self.read(dst, size)?;
                    self.set_lazy_flags(
                        LazyFlagsOp::Sub,
                        size,
                        a.wrapping_sub(b) & mask(size),
                        b,
                        0,
                    );
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
            }
            if !rep {
                break;
            }
            let count = self.gpr[GPR_RCX].wrapping_sub(1);
            self.gpr[GPR_RCX] = if i.addr32 { count & DWORD_MASK } else { count };
            iterations += 1;
            if compares {
                let zf = self.eflags_full() & ZF != 0;
                if (i.rep == REP_REPE && !zf) || (i.rep == REP_REPNE && zf) {
                    break;
                }
            }
            if iterations >= REP_ITERATIONS_PER_SLICE && self.gpr[GPR_RCX] & amask != 0 {
                // RIP stays on the instruction; the run loop re-executes it after servicing events.
                self.icount += iterations;
                return Ok(());
            }
        }
        self.icount += iterations;
        self.rip = next;
        Ok(())
    }

    /// Loads DS/ES/FS/GS/SS (`idx` is a `SEG_*` index) from a selector in 64-bit mode, with the
    /// privilege checks of `mov sreg` made against the current CPL. Nothing is loaded on a fault.
    pub fn load_data_seg(&mut self, idx: usize, sel: u16) -> ExecResult<()> {
        // A null SS is only usable when it names the current privilege level.
        if idx == SEG_SS && selector_is_null(sel) && (sel & SELECTOR_RPL_MASK) as u8 != self.cpl {
            return Err(Exception::general_protection(0));
        }
        let seg = self.checked_data_seg(idx, sel, self.cpl)?;
        self.seg[idx] = seg;
        Ok(())
    }

    /// Validates `sel` for DS/ES/FS/GS/SS against privilege level `cpl` and returns the segment
    /// to load, leaving the CPU untouched (iretq and far return validate before they commit).
    /// `cpl` is the current CPL for `mov sreg` and the one being returned to for iretq/retf.
    ///
    /// A null selector is allowed everywhere except SS at CPL 3. Otherwise (SDM 3A 5.4):
    /// system descriptors and execute-only code are #GP for a data register; data and
    /// non-conforming code need DPL >= max(CPL, RPL), conforming code is exempt; SS must be
    /// writable data with RPL = DPL = CPL; a missing segment is #SS for SS and #NP otherwise.
    fn checked_data_seg(&mut self, idx: usize, sel: u16, cpl: u8) -> ExecResult<Seg> {
        if selector_is_null(sel) {
            if idx == SEG_SS && cpl == USER_CPL {
                return Err(Exception::general_protection(0));
            }
            return Ok(Seg {
                sel,
                ..Default::default()
            });
        }
        let s = self.load_desc_seg(sel, false)?;
        let rpl = (sel & SELECTOR_RPL_MASK) as u8;
        let privilege_violation = Exception::general_protection(selector_error_code(sel));
        if s.flags & SEG_NON_SYSTEM == 0 {
            return Err(privilege_violation);
        }
        if idx == SEG_SS {
            if s.is_code() || s.flags & SEG_TYPE_READ_WRITE == 0 || rpl != cpl || s.dpl() != cpl {
                return Err(privilege_violation);
            }
        } else {
            if s.is_code() && s.flags & SEG_TYPE_READ_WRITE == 0 {
                return Err(privilege_violation);
            }
            if !is_conforming_code(&s) && (s.dpl() < cpl || s.dpl() < rpl) {
                return Err(privilege_violation);
            }
        }
        if !s.present() {
            return Err(Exception::fault(
                if idx == SEG_SS {
                    VECTOR_STACK_FAULT
                } else {
                    VECTOR_SEGMENT_NOT_PRESENT
                },
                Some(selector_error_code(sel)),
            ));
        }
        Ok(s)
    }

    /// Validates the code segment selector popped by iretq or a far return (SDM 3A IRET/RET):
    /// present 64-bit code, RPL not below the current CPL, and DPL equal to RPL for
    /// non-conforming code or at most RPL for conforming code. Nothing is loaded.
    fn checked_return_code_seg(&mut self, cs: u16) -> ExecResult<Seg> {
        if selector_is_null(cs) {
            return Err(Exception::general_protection(0));
        }
        let s = self.load_desc_seg(cs, false)?;
        let rpl = (cs & SELECTOR_RPL_MASK) as u8;
        let privilege_violation = Exception::general_protection(selector_error_code(cs));
        let dpl_matches = if is_conforming_code(&s) {
            s.dpl() <= rpl
        } else {
            s.dpl() == rpl
        };
        if !s.is_code() || rpl < self.cpl || !dpl_matches {
            return Err(privilege_violation);
        }
        if !s.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(selector_error_code(cs)),
            ));
        }
        // Compatibility mode is not implemented, so a code segment without CS.L cannot be entered.
        if !s.long() {
            return Err(privilege_violation);
        }
        Ok(s)
    }

    /// After a return to an outer ring, DS/ES/FS/GS that hold a data or non-conforming code
    /// segment with DPL below the new CPL are replaced by null selectors, so the less privileged
    /// code cannot keep using them. A register that is already null is left as it is, which keeps
    /// the FS/GS bases.
    fn null_inaccessible_data_segments(&mut self, new_cpl: u8) {
        for register in [SEG_ES, SEG_DS, SEG_FS, SEG_GS] {
            let s = self.seg[register];
            if !selector_is_null(s.sel) && s.dpl() < new_cpl && !is_conforming_code(&s) {
                self.seg[register] = Seg::default();
            }
        }
    }

    /// 64-bit iretq: pops RIP, CS, RFLAGS, RSP, SS (always all five in 64-bit mode). Only the
    /// 8-byte operand size is implemented. Every check is made before any state changes, so a
    /// fault leaves CS, SS, CPL, RSP, RIP and the flags as they were. The SS checks are made
    /// against the CPL being returned to.
    fn iretq(&mut self, i: &Insn) -> ExecResult<()> {
        const FRAME_CS: u64 = 8;
        const FRAME_RFLAGS: u64 = 16;
        const FRAME_RSP: u64 = 24;
        const FRAME_SS: u64 = 32;
        if i.opsize != 8 {
            return Err(Exception::invalid_opcode());
        }
        let sp = self.gpr[GPR_RSP];
        let rip = self.read_stack(sp, 8)?;
        let cs = self.read_stack(sp.wrapping_add(FRAME_CS), 8)? as u16;
        let new_rflags = self.read_stack(sp.wrapping_add(FRAME_RFLAGS), 8)?;
        let new_cs = self.checked_return_code_seg(cs)?;
        let new_rsp = self.read_stack(sp.wrapping_add(FRAME_RSP), 8)?;
        let ss = self.read_stack(sp.wrapping_add(FRAME_SS), 8)? as u16;
        let new_cpl = (cs & SELECTOR_RPL_MASK) as u8;
        let new_ss = self.checked_data_seg(SEG_SS, ss, new_cpl)?;
        let old_cpl = self.cpl;
        self.seg[SEG_CS] = new_cs;
        self.seg[SEG_SS] = new_ss;
        // set_cpl also switches to the user or supervisor half of the soft TLB.
        self.set_cpl(new_cpl);
        // IF and IOPL are gated by the CPL iretq started at, not the one it returns to.
        let loadable = self.loadable_eflags(old_cpl);
        let current = self.eflags_full();
        self.eflags = ((current & !loadable) | (new_rflags & loadable)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.gpr[GPR_RSP] = new_rsp;
        self.rip = rip;
        if new_cpl > old_cpl {
            self.null_inaccessible_data_segments(new_cpl);
        }
        Ok(())
    }

    /// Far return (0xca/0xcb) with `imm` extra bytes released after CS. Only the 8-byte operand
    /// size (REX.W) is implemented. For a return to an outer ring RSP:SS sit above the released
    /// bytes, and the bytes are released from the new stack as well. Validation comes before any
    /// state change, as for `iretq`.
    fn far_return(&mut self, i: &Insn, imm: u64) -> ExecResult<()> {
        // Frame: rip, cs, `imm` released bytes, then (outer-ring returns only) rsp and ss.
        // FRAME_RSP and FRAME_SS are offsets from RSP before `imm` is added.
        const FRAME_CS: u64 = 8;
        const FRAME_RSP: u64 = 16;
        const FRAME_SS: u64 = 24;
        if i.opsize != 8 {
            return Err(Exception::invalid_opcode());
        }
        let sp = self.gpr[GPR_RSP];
        let rip = self.read_stack(sp, 8)?;
        let cs = self.read_stack(sp.wrapping_add(FRAME_CS), 8)? as u16;
        let new_cs = self.checked_return_code_seg(cs)?;
        let new_cpl = (cs & SELECTOR_RPL_MASK) as u8;
        if new_cpl == self.cpl {
            self.seg[SEG_CS] = new_cs;
            self.gpr[GPR_RSP] = sp.wrapping_add(FRAME_RSP).wrapping_add(imm);
        } else {
            let new_rsp = self.read_stack(sp.wrapping_add(FRAME_RSP).wrapping_add(imm), 8)?;
            let ss = self.read_stack(sp.wrapping_add(FRAME_SS).wrapping_add(imm), 8)? as u16;
            let new_ss = self.checked_data_seg(SEG_SS, ss, new_cpl)?;
            self.seg[SEG_CS] = new_cs;
            self.seg[SEG_SS] = new_ss;
            self.set_cpl(new_cpl);
            self.gpr[GPR_RSP] = new_rsp.wrapping_add(imm);
            self.null_inaccessible_data_segments(new_cpl);
        }
        self.rip = rip;
        Ok(())
    }

    /// Executes one decoded instruction; also the slow path the JIT calls for instructions it does
    /// not compile (`jit/helpers.rs`), which makes it the oracle for `--verify`.
    ///
    /// On `Ok` RIP is already advanced (next instruction or branch target). On `Err` RIP is still
    /// the faulting instruction, except int/int3 which set it to the next instruction first so the
    /// pushed return address is right. Opcodes are `Insn::op`: one-byte opcodes as is, `0f xx` as
    /// `0x1xx`.
    pub fn exec(&mut self, i: &Insn) -> ExecResult<()> {
        let next = self.rip.wrapping_add(i.len as u64);
        let osz = i.opsize;
        let op = i.op;
        let rex = i.rex_present;
        match op {
            // The eight ALU ops (add/or/adc/sbb/and/sub/xor/cmp), forms 0..=5: r/m,r (byte, full),
            // r,r/m (byte, full), al,imm and eax,imm.
            0x00..=0x3f if op & 7 < 6 => {
                let operation = ((op >> 3) & 7) as u8;
                let form = op & 7;
                let size = if form & 1 == 0 { 1 } else { osz };
                match form {
                    0 | 1 => {
                        let loc = self.modrm_location(i, next);
                        let a = self.load_operand(loc, size, i)?;
                        let b = self.read_gpr(i.reg, size, rex);
                        let saved = self.save_flags();
                        let r = self.alu(operation, a, b, size);
                        if operation != ALU_CMP {
                            self.store_result(loc, size, i, r, saved)?;
                        }
                    }
                    2 | 3 => {
                        let loc = self.modrm_location(i, next);
                        let b = self.load_operand(loc, size, i)?;
                        let a = self.read_gpr(i.reg, size, rex);
                        let r = self.alu(operation, a, b, size);
                        if operation != ALU_CMP {
                            self.write_gpr(i.reg, size, rex, r);
                        }
                    }
                    _ => {
                        let a = self.read_gpr(0, size, true);
                        let r = self.alu(operation, a, i.imm as u64, size);
                        if operation != ALU_CMP {
                            self.write_gpr(0, size, true, r);
                        }
                    }
                }
            }
            // push r64
            0x50..=0x57 => {
                let r = opcode_register(i);
                let sz = stack_operand_size(i);
                let v = self.gpr[r as usize];
                self.push(sz, v)?;
            }
            // pop r64
            0x58..=0x5f => {
                let r = opcode_register(i);
                let sz = stack_operand_size(i);
                let v = self.pop(sz)?;
                self.write_gpr(r, sz, true, v);
            }
            // movsxd
            0x63 => {
                let loc = self.modrm_location(i, next);
                let v = self.load_operand(loc, 4, i)?;
                if osz == 8 {
                    self.gpr[i.reg as usize] = v as u32 as i32 as i64 as u64;
                } else {
                    self.write_gpr(i.reg, osz, true, v);
                }
            }
            // push imm32 / imm8
            0x68 | 0x6a => {
                let sz = stack_operand_size(i);
                self.push(sz, i.imm as u64)?;
            }
            // imul r, r/m, imm32 / imm8
            0x69 | 0x6b => {
                let loc = self.modrm_location(i, next);
                let a = self.load_operand(loc, osz, i)?;
                let r = self.imul_truncated(a, i.imm as u64, osz);
                self.write_gpr(i.reg, osz, true, r);
            }
            // jcc rel8
            0x70..=0x7f => {
                if self.condition_holds((op & 15) as u8) {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            // group 1: ALU r/m, imm
            0x80 | 0x81 | 0x83 => {
                let size = if op == 0x80 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let a = self.load_operand(loc, size, i)?;
                let operation = i.reg & 7;
                let saved = self.save_flags();
                let r = self.alu(operation, a, i.imm as u64, size);
                if operation != ALU_CMP {
                    self.store_result(loc, size, i, r, saved)?;
                }
            }
            // test r/m, r
            0x84 | 0x85 => {
                let size = if op == 0x84 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let a = self.load_operand(loc, size, i)?;
                let b = self.read_gpr(i.reg, size, rex);
                self.set_lazy_flags(LazyFlagsOp::Logic, size, a & b, 0, 0);
            }
            // xchg r/m, r
            0x86 | 0x87 => {
                let size = if op == 0x86 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let a = self.load_operand(loc, size, i)?;
                let b = self.read_gpr(i.reg, size, rex);
                self.store_operand(loc, size, i, b)?;
                self.write_gpr(i.reg, size, rex, a);
            }
            // mov r/m, r
            0x88 | 0x89 => {
                let size = if op == 0x88 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let v = self.read_gpr(i.reg, size, rex);
                self.store_operand(loc, size, i, v)?;
            }
            // mov r, r/m
            0x8a | 0x8b => {
                let size = if op == 0x8a { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let v = self.load_operand(loc, size, i)?;
                self.write_gpr(i.reg, size, rex, v);
            }
            // mov r/m, sreg: a memory destination is always 16 bits, a register one takes the
            // operand size (upper bits zeroed).
            0x8c => {
                // REX.R extends ModRM.reg, but the segment register field is only 3 bits.
                let idx = (i.reg & 7) as usize;
                if idx > SEG_GS {
                    return Err(Exception::invalid_opcode());
                }
                let v = self.seg[idx].sel as u64;
                let loc = self.modrm_location(i, next);
                let size = if i.md == 3 { osz } else { 2 };
                self.store_operand(loc, size, i, v)?;
            }
            // lea
            0x8d => {
                if i.md == 3 {
                    return Err(Exception::invalid_opcode());
                }
                let a = self.effective_address(i, next);
                self.write_gpr(i.reg, osz, true, a);
            }
            // mov sreg, r/m. CS cannot be loaded this way.
            0x8e => {
                let idx = (i.reg & 7) as usize;
                if idx == SEG_CS || idx > SEG_GS {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next);
                let sel = self.load_operand(loc, 2, i)? as u16;
                self.load_data_seg(idx, sel)?;
                // Interrupts are held off until after the instruction following mov ss, so that
                // a mov ss / mov rsp pair is atomic.
                if idx == SEG_SS {
                    self.interrupt_shadow = true;
                }
            }
            // pop r/m (only /0; the other encodings are XOP)
            0x8f => {
                if i.reg & 7 != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let sz = stack_operand_size(i);
                let loc = self.modrm_location(i, next);
                let sp = self.gpr[GPR_RSP];
                let v = self.read_stack(sp, sz)?;
                self.gpr[GPR_RSP] = sp.wrapping_add(sz as u64);
                // A memory destination addressed through GPR_RSP uses the value after the pop.
                let loc = match loc {
                    Loc::Mem(a) if i.base == GPR_RSP as u8 => Loc::Mem(a.wrapping_add(sz as u64)),
                    l => l,
                };
                if let Err(e) = self.store_operand(loc, sz, i, v) {
                    self.gpr[GPR_RSP] = sp;
                    return Err(e);
                }
            }
            // nop, or xchg r8, rax with REX.B. pause (f3 90) is a plain nop.
            0x90 => {
                if i.rex & REX_B != 0 {
                    let r = 8;
                    let a = self.read_gpr(0, osz, true);
                    let b = self.read_gpr(r, osz, true);
                    self.write_gpr(0, osz, true, b);
                    self.write_gpr(r, osz, true, a);
                }
            }
            // xchg r, rax
            0x91..=0x97 => {
                let r = opcode_register(i);
                let a = self.read_gpr(0, osz, true);
                let b = self.read_gpr(r, osz, true);
                self.write_gpr(0, osz, true, b);
                self.write_gpr(r, osz, true, a);
            }
            // cdqe / cwde / cbw
            0x98 => match osz {
                8 => self.gpr[GPR_RAX] = self.gpr[GPR_RAX] as u32 as i32 as i64 as u64,
                4 => self.gpr[GPR_RAX] = self.gpr[GPR_RAX] as u16 as i16 as i32 as u32 as u64,
                _ => {
                    let v = self.gpr[GPR_RAX] as u8 as i8 as i16 as u16 as u64;
                    self.write_gpr(0, 2, true, v)
                }
            },
            // cqo / cdq / cwd: fill rDX with the sign of rAX
            0x99 => match osz {
                8 => self.gpr[GPR_RDX] = ((self.gpr[GPR_RAX] as i64) >> 63) as u64,
                4 => self.gpr[GPR_RDX] = (((self.gpr[GPR_RAX] as u32 as i32) >> 31) as u32) as u64,
                _ => {
                    let v = (((self.gpr[GPR_RAX] as u16 as i16) >> 15) as u16) as u64;
                    self.write_gpr(2, 2, true, v)
                }
            },
            // fwait: #NM when both CR0.MP and CR0.TS are set
            0x9b => {
                if self.cr0 & (CR0_MP | CR0_TS) == (CR0_MP | CR0_TS) {
                    return Err(Exception::device_not_available());
                }
            }
            // pushf: the pushed image has RF and VM clear
            0x9c => {
                let sz = stack_operand_size(i);
                let f = self.eflags_full() & !(RF | VM);
                self.push(sz, f)?;
            }
            // popf
            0x9d => {
                let sz = stack_operand_size(i);
                let v = self.pop(sz)?;
                let mut loadable = self.loadable_eflags(self.cpl);
                if sz == 2 {
                    loadable &= WORD_MASK;
                }
                let current = self.eflags_full();
                self.eflags = ((current & !loadable) | (v & loadable)) | RESERVED_ONE;
                self.lazy_op = LazyFlagsOp::Eflags;
            }
            // sahf
            0x9e => {
                let ah = (self.gpr[GPR_RAX] >> 8) & LAHF_SAHF_FLAGS;
                let f = self.eflags_full();
                self.set_eflags_arith((f & !LAHF_SAHF_FLAGS) | ah);
            }
            // lahf: bit 1 of AH always reads 1
            0x9f => {
                let f = (self.eflags_full() & LAHF_SAHF_FLAGS) | RESERVED_ONE;
                self.write_gpr(BYTE_REG_AH, 1, false, f);
            }
            // mov al/eax, moffs and mov moffs, al/eax
            0xa0..=0xa3 => {
                let size = if op & 1 == 0 { 1 } else { osz };
                let a = self.linear_address(
                    i,
                    if i.addr32 {
                        i.imm as u64 & DWORD_MASK
                    } else {
                        i.imm as u64
                    },
                );
                if op < 0xa2 {
                    let v = self.read(a, size)?;
                    self.write_gpr(0, size, true, v);
                } else {
                    let v = self.read_gpr(0, size, true);
                    self.write(a, size, v)?;
                }
            }
            // movs, cmps, stos, lods, scas
            0xa4..=0xa7 | 0xaa..=0xaf => return self.string_op(i, next),
            // test al/eax, imm
            0xa8 | 0xa9 => {
                let size = if op == 0xa8 { 1 } else { osz };
                let a = self.read_gpr(0, size, true);
                self.set_lazy_flags(
                    LazyFlagsOp::Logic,
                    size,
                    a & i.imm as u64 & mask(size),
                    0,
                    0,
                );
            }
            // mov r8, imm8
            0xb0..=0xb7 => {
                let r = opcode_register(i);
                self.write_gpr(r, 1, rex, i.imm as u64);
            }
            // mov r, imm (imm64 with REX.W)
            0xb8..=0xbf => {
                let r = opcode_register(i);
                self.write_gpr(r, osz, true, i.imm as u64);
            }
            // shift/rotate group: imm8, 1 or CL count
            0xc0 | 0xc1 | 0xd0..=0xd3 => {
                let size = if op & 1 == 0 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let count = match op {
                    0xc0 | 0xc1 => i.imm as u64,
                    0xd0 | 0xd1 => 1,
                    _ => self.gpr[GPR_RCX] & BYTE_MASK,
                };
                let a = self.load_operand(loc, size, i)?;
                let saved = self.save_flags();
                let r = self.shift_rotate(i.reg & 7, a, count, size);
                self.store_result(loc, size, i, r, saved)?;
            }
            // ret / ret imm16
            0xc2 | 0xc3 => {
                let target = self.read_stack(self.gpr[GPR_RSP], 8)?;
                let extra = if op == 0xc2 { i.imm as u64 } else { 0 };
                self.gpr[GPR_RSP] = self.gpr[GPR_RSP].wrapping_add(8 + extra);
                self.rip = target;
                return Ok(());
            }
            // mov r/m, imm. Only /0 is defined (c6 /7 and c7 /7 are xabort/xbegin, which need RTM).
            0xc6 | 0xc7 => {
                if i.reg & 7 != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let size = if op == 0xc6 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                self.store_operand(loc, size, i, i.imm as u64)?;
            }
            // enter: the nesting level is taken modulo 32. Registers are committed only after the
            // last push, so a fault leaves RSP and RBP as they were (the pushes already made
            // stay in memory, as on hardware).
            0xc8 => {
                let size = stack_operand_size(i);
                let width = size as u64;
                let level = i.imm2 & ENTER_LEVEL_MASK;
                let mut sp = self.gpr[GPR_RSP].wrapping_sub(width);
                self.write_stack(sp, size, self.gpr[GPR_RBP])?;
                let frame_pointer = sp;
                if level > 0 {
                    // Copy the enclosing frames' pointers down the chain, then push our own.
                    let mut outer = self.gpr[GPR_RBP];
                    for _ in 1..level {
                        outer = outer.wrapping_sub(width);
                        let enclosing = self.read_stack(outer, size)?;
                        sp = sp.wrapping_sub(width);
                        self.write_stack(sp, size, enclosing)?;
                    }
                    sp = sp.wrapping_sub(width);
                    self.write_stack(sp, size, frame_pointer)?;
                }
                self.gpr[GPR_RBP] = frame_pointer;
                self.gpr[GPR_RSP] = sp.wrapping_sub(i.imm as u64);
            }
            // leave: GPR_RSP is only committed after the load of GPR_RBP succeeded. With a 66
            // prefix it pops a word into BP and leaves the rest of RBP alone.
            0xc9 => {
                let size = stack_operand_size(i);
                let bp = self.gpr[GPR_RBP];
                let v = self.read_stack(bp, size)?;
                self.gpr[GPR_RSP] = bp.wrapping_add(size as u64);
                self.write_gpr(GPR_RBP as u8, size, true, v);
            }
            // far ret imm16 / far ret
            0xca | 0xcb => {
                let imm = if op == 0xca { i.imm as u64 } else { 0 };
                return self.far_return(i, imm);
            }
            // int3
            0xcc => {
                self.rip = next;
                return Err(Exception::Fault {
                    vector: VECTOR_BREAKPOINT,
                    error_code: None,
                    software: true,
                });
            }
            // int imm8
            0xcd => {
                self.rip = next;
                return Err(Exception::Fault {
                    vector: i.imm as u8,
                    error_code: None,
                    software: true,
                });
            }
            0xcf => return self.iretq(i),
            // xlat: al = [rbx + al]; with the 67 prefix the sum is formed and wrapped at 32 bits
            0xd7 => {
                let a =
                    self.gpr[GPR_RBX].wrapping_add(self.gpr[GPR_RAX] & BYTE_MASK) & address_mask(i);
                let v = self.read(self.linear_address(i, a), 1)?;
                self.write_gpr(0, 1, true, v);
            }
            0xd8..=0xdf => return self.x87(i, next),
            // loopne, loope, loop, jrcxz (jecxz with the 67 prefix)
            0xe0..=0xe3 => {
                let amask = address_mask(i);
                let take = if op == 0xe3 {
                    self.gpr[GPR_RCX] & amask == 0
                } else {
                    let count = self.gpr[GPR_RCX].wrapping_sub(1) & amask;
                    self.gpr[GPR_RCX] = count;
                    let zf = self.eflags_full() & ZF != 0;
                    count != 0
                        && match op {
                            0xe0 => !zf,
                            0xe1 => zf,
                            _ => true,
                        }
                };
                if take {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            // in/out with imm8 port (e4..e7) or DX (ec..ef)
            0xe4..=0xe7 | 0xec..=0xef => return self.port_io(i, next),
            // call rel32
            0xe8 => {
                self.push(8, next)?;
                self.rip = next.wrapping_add(i.imm as u64);
                return Ok(());
            }
            // jmp rel32 / rel8
            0xe9 | 0xeb => {
                self.rip = next.wrapping_add(i.imm as u64);
                return Ok(());
            }
            // hlt
            0xf4 => {
                if self.cpl != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.halted = true;
            }
            // cmc
            0xf5 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f ^ CF);
            }
            // group 3: test/not/neg/mul/imul/div/idiv
            0xf6 | 0xf7 => {
                let size = if op == 0xf6 { 1 } else { osz };
                match i.reg & 7 {
                    GROUP3_TEST | GROUP3_TEST_ALIAS => {
                        let loc = self.modrm_location(i, next);
                        let a = self.load_operand(loc, size, i)?;
                        self.set_lazy_flags(
                            LazyFlagsOp::Logic,
                            size,
                            a & i.imm as u64 & mask(size),
                            0,
                            0,
                        );
                    }
                    GROUP3_NOT => {
                        let loc = self.modrm_location(i, next);
                        let a = self.load_operand(loc, size, i)?;
                        self.store_operand(loc, size, i, !a)?;
                    }
                    GROUP3_NEG => {
                        let loc = self.modrm_location(i, next);
                        let a = self.load_operand(loc, size, i)?;
                        let r = 0u64.wrapping_sub(a) & mask(size);
                        let saved = self.save_flags();
                        self.set_lazy_flags(LazyFlagsOp::Sub, size, r, a & mask(size), 0);
                        self.store_result(loc, size, i, r, saved)?;
                    }
                    _ => self.mul_div(i, next, size)?,
                }
            }
            // clc
            0xf8 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f & !CF);
            }
            // stc
            0xf9 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f | CF);
            }
            // cli / sti: need CPL <= IOPL (VME/PVI are not emulated)
            0xfa | 0xfb => {
                if self.cpl > self.iopl() {
                    return Err(Exception::general_protection(0));
                }
                if op == 0xfa {
                    self.eflags &= !IF;
                } else {
                    // Only the transition from IF=0 opens the one-instruction interrupt shadow.
                    if self.eflags & IF == 0 {
                        self.interrupt_shadow = true;
                    }
                    self.eflags |= IF;
                }
            }
            // cld / std
            0xfc => self.eflags &= !DF,
            0xfd => self.eflags |= DF,
            // group 4/5: inc, dec, call, jmp, push (far forms are not implemented)
            0xfe | 0xff => {
                let size = if op == 0xfe { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                match i.reg & 7 {
                    GROUP5_INC | GROUP5_DEC => {
                        let a = self.load_operand(loc, size, i)?;
                        // inc/dec leave CF alone: the lazy op takes the current CF as its source.
                        let cf = self.cf() as u64;
                        let (r, lazy_op) = if i.reg & 7 == GROUP5_INC {
                            (a.wrapping_add(1) & mask(size), LazyFlagsOp::Inc)
                        } else {
                            (a.wrapping_sub(1) & mask(size), LazyFlagsOp::Dec)
                        };
                        let saved = self.save_flags();
                        self.set_lazy_flags(lazy_op, size, r, cf, 0);
                        self.store_result(loc, size, i, r, saved)?;
                    }
                    GROUP5_CALL_NEAR if op == 0xff => {
                        // The target is read before the push so `call [rsp+x]` sees the old GPR_RSP.
                        let target = self.load_operand(loc, 8, i)?;
                        self.push(8, next)?;
                        self.rip = target;
                        return Ok(());
                    }
                    GROUP5_JMP_NEAR if op == 0xff => {
                        self.rip = self.load_operand(loc, 8, i)?;
                        return Ok(());
                    }
                    GROUP5_PUSH if op == 0xff => {
                        let sz = stack_operand_size(i);
                        let v = self.load_operand(loc, sz, i)?;
                        self.push(sz, v)?;
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                }
            }
            _ => return self.exec_two_byte(i, next),
        }
        self.rip = next;
        Ok(())
    }

    /// #GP(0) unless an I/O access of `size` bytes at `port` is permitted. At CPL <= IOPL it
    /// always is; otherwise every bit the access covers in the TSS I/O permission bitmap must be
    /// clear, and the bitmap must lie inside the TSS limit (SDM 3A 20.5; QEMU `check_io`).
    fn check_io_permission(&mut self, port: u16, size: u8) -> ExecResult<()> {
        if self.cpl <= self.iopl() {
            return Ok(());
        }
        let tss_limit = self.tr.limit as u64;
        // The 16-bit map base is the last field of the fixed part of the TSS.
        if tss_limit < TSS64_IO_MAP_BASE_OFFSET + 1 {
            return Err(Exception::general_protection(0));
        }
        let map_base = self.read_sys(self.tr.base.wrapping_add(TSS64_IO_MAP_BASE_OFFSET), 2)?;
        let byte_offset = map_base + (port as u64 >> 3);
        // The bits are read as a 16-bit word because an access can straddle a bitmap byte.
        if byte_offset + 1 > tss_limit {
            return Err(Exception::general_protection(0));
        }
        let word = self.read_sys(self.tr.base.wrapping_add(byte_offset), 2)?;
        if (word >> (port & 7)) & ((1 << size) - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    /// in/out with an imm8 or DX port.
    fn port_io(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op;
        let size = if op & 1 == 0 {
            1
        } else if i.p66 {
            2
        } else {
            4
        };
        // e4..e7 take an imm8 port, ec..ef use DX.
        let port = if op < 0xe8 {
            i.imm as u8 as u16
        } else {
            self.gpr[GPR_RDX] as u16
        };
        self.check_io_permission(port, size)?;
        let now = self.now_ns();
        if op & 2 == 0 {
            let v = self.dev.io_in(port, size, now);
            self.write_gpr(0, size, true, v as u64);
        } else {
            let v = self.read_gpr(0, size, true) as u32;
            self.dev.io_out(port, size, v, now);
            // A device asked for a machine reset; RIP is advanced first so the exit state is
            // after the out.
            if self.dev.reset_requested {
                self.rip = next;
                return Err(Exception::Exit(ExitReason::Reset));
            }
        }
        self.rip = next;
        Ok(())
    }

    /// Raises #UD for an opcode that is valid but not implemented, logging each distinct
    /// (opcode, ModRM.reg) once.
    pub fn unimplemented_opcode(&mut self, i: &Insn) -> Exception {
        // ModRM.reg has 8 values, so this keeps each opcode's /0../7 variants on distinct keys.
        const REG_VALUES_PER_OPCODE: u64 = 8;
        let rip = self.rip;
        self.log_once(
            UNIMPLEMENTED_OPCODE_LOG_BASE
                + i.op as u64 * REG_VALUES_PER_OPCODE
                + (i.reg as u64 & 7),
            || {
                format!(
                    "unimplemented opcode {:#05x} /{} at rip {:#x} (modrm {:#04x})",
                    i.op,
                    i.reg & 7,
                    rip,
                    i.modrm
                )
            },
        );
        Exception::invalid_opcode()
    }

    /// `0f xx` opcodes (`Insn::op` is `0x100 | xx`); anything not handled here goes to SSE.
    fn exec_two_byte(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op;
        let osz = i.opsize;
        let rex = i.rex_present;
        match op {
            // group 6 / group 7 (sldt, ltr, sgdt, lidt, ...)
            0x100 | 0x101 => return self.sys_group(i, next),
            0x105 => return self.do_syscall(next),
            0x107 => return self.do_sysret(i),
            // clts
            0x106 => {
                self.need_cpl0()?;
                self.cr0 &= !CR0_TS;
            }
            // invd / wbinvd: privileged no-ops
            0x108 | 0x109 => self.need_cpl0()?,
            // ud2
            0x10b => return Err(Exception::invalid_opcode()),
            // prefetch and the reserved hint-nop space (includes multi-byte nop, endbr64)
            0x10d | 0x118..=0x11f => {}
            // mov to/from cr and dr
            0x120..=0x123 => self.mov_cr_dr(i)?,
            // wrmsr: MSR in ECX, value in EDX:EAX
            0x130 => {
                self.need_cpl0()?;
                let v = combine_dwords(self.gpr[GPR_RAX], self.gpr[GPR_RDX]);
                self.wrmsr(self.gpr[GPR_RCX] as u32, v)?;
            }
            // rdtsc
            0x131 => {
                if self.cr4 & CR4_TSD != 0 && self.cpl != 0 {
                    return Err(Exception::general_protection(0));
                }
                let t = self.tsc();
                self.set_edx_eax(t);
            }
            // rdmsr
            0x132 => {
                self.need_cpl0()?;
                let v = self.rdmsr(self.gpr[GPR_RCX] as u32)?;
                self.set_edx_eax(v);
            }
            // cmovcc: the source is always read (it can fault); a 32-bit cmov that does not move
            // still zero-extends the destination.
            0x140..=0x14f => {
                let loc = self.modrm_location(i, next);
                let v = self.load_operand(loc, osz, i)?;
                if self.condition_holds((i.op & 15) as u8) {
                    self.write_gpr(i.reg, osz, true, v);
                } else if osz == 4 {
                    self.gpr[i.reg as usize] &= DWORD_MASK;
                }
            }
            // jcc rel32
            0x180..=0x18f => {
                if self.condition_holds((i.op & 15) as u8) {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            // setcc
            0x190..=0x19f => {
                let loc = self.modrm_location(i, next);
                let v = self.condition_holds((i.op & 15) as u8) as u64;
                self.store_operand(loc, 1, i, v)?;
            }
            // push fs / push gs
            0x1a0 | 0x1a8 => {
                let reg = if op == 0x1a0 { SEG_FS } else { SEG_GS };
                let sel = self.seg[reg].sel as u64;
                let sz = stack_operand_size(i);
                self.push(sz, sel)?;
            }
            // pop fs / pop gs: GPR_RSP is rolled back if the segment load faults
            0x1a1 | 0x1a9 => {
                let sz = stack_operand_size(i);
                let v = self.pop(sz)? as u16;
                let reg = if op == 0x1a1 { SEG_FS } else { SEG_GS };
                if let Err(e) = self.load_data_seg(reg, v) {
                    self.gpr[GPR_RSP] = self.gpr[GPR_RSP].wrapping_sub(sz as u64);
                    return Err(e);
                }
            }
            0x1a2 => self.cpuid(),
            // bt / bts / btr / btc r/m, r
            0x1a3 => self.bit_op(i, next, BIT_TEST, false)?,
            0x1ab => self.bit_op(i, next, BIT_SET, false)?,
            0x1b3 => self.bit_op(i, next, BIT_RESET, false)?,
            0x1bb => self.bit_op(i, next, BIT_COMPLEMENT, false)?,
            // group 8: bt/bts/btr/btc r/m, imm8
            0x1ba => {
                let group_op = i.reg & 7;
                if group_op < GROUP8_FIRST_BIT_OP {
                    return Err(Exception::invalid_opcode());
                }
                self.bit_op(i, next, group_op - GROUP8_FIRST_BIT_OP, true)?
            }
            // shld imm8 / cl, shrd imm8 / cl
            0x1a4 | 0x1a5 | 0x1ac | 0x1ad => {
                let loc = self.modrm_location(i, next);
                let a = self.load_operand(loc, osz, i)?;
                let b = self.read_gpr(i.reg, osz, rex);
                let count = if op & 1 == 0 {
                    i.imm as u64
                } else {
                    self.gpr[GPR_RCX] & BYTE_MASK
                };
                let bits = osz as u32 * 8;
                let c = (count & shift_count_mask(osz)) as u32;
                if c != 0 {
                    let (r, cf) = double_shift(op < 0x1a8, a, b, c, osz);
                    // OF is set when the sign bit changed.
                    let of = ((r ^ a) >> (bits - 1)) & 1;
                    let mut f = szp_flags(r, osz);
                    if cf != 0 {
                        f |= CF;
                    }
                    if of != 0 {
                        f |= OF;
                    }
                    let saved = self.save_flags();
                    self.set_eflags_arith(f);
                    self.store_result(loc, osz, i, r, saved)?;
                }
            }
            // group 15 (fences, fxsave, ldmxcsr, ...)
            0x1ae => return self.group_ae(i, next),
            // imul r, r/m
            0x1af => {
                let loc = self.modrm_location(i, next);
                let b = self.load_operand(loc, osz, i)?;
                let a = self.read_gpr(i.reg, osz, true);
                let r = self.imul_truncated(a, b, osz);
                self.write_gpr(i.reg, osz, true, r);
            }
            // cmpxchg r/m, r. The destination is stored even when the compare fails, as on
            // hardware, so a write fault is taken either way.
            0x1b0 | 0x1b1 => {
                let size = if op == 0x1b0 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let dest = self.load_operand(loc, size, i)?;
                let acc = self.read_gpr(0, size, true);
                let m = mask(size);
                let saved = self.save_flags();
                self.set_lazy_flags(LazyFlagsOp::Sub, size, acc.wrapping_sub(dest) & m, dest, 0);
                if acc == dest {
                    let src = self.read_gpr(i.reg, size, rex);
                    self.store_result(loc, size, i, src, saved)?;
                } else {
                    self.store_result(loc, size, i, dest, saved)?;
                    self.write_gpr(0, size, true, dest);
                }
            }
            // xadd r/m, r
            0x1c0 | 0x1c1 => {
                let size = if op == 0x1c0 { 1 } else { osz };
                let loc = self.modrm_location(i, next);
                let dest = self.load_operand(loc, size, i)?;
                let src = self.read_gpr(i.reg, size, rex);
                let saved = self.save_flags();
                let sum = self.alu(ALU_ADD, dest, src, size);
                self.store_result(loc, size, i, sum, saved)?;
                self.write_gpr(i.reg, size, rex, dest);
            }
            // movnti
            0x1c3 => {
                if i.md == 3 {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next);
                let v = self.read_gpr(i.reg, osz, true);
                self.store_operand(loc, osz, i, v)?;
            }
            // cmpxchg8b m64 (group 9 /1); cmpxchg16b (REX.W) is not implemented
            0x1c7 => {
                if i.md == 3 || i.reg & 7 != 1 || osz == 8 {
                    return Err(self.unimplemented_opcode(i));
                }
                let a = self.linear_address(i, self.effective_address(i, next));
                let current = self.read(a, 8)?;
                let expected = combine_dwords(self.gpr[GPR_RAX], self.gpr[GPR_RDX]);
                let f = self.eflags_full();
                if current == expected {
                    let replacement = combine_dwords(self.gpr[GPR_RBX], self.gpr[GPR_RCX]);
                    self.write(a, 8, replacement)?;
                    self.set_eflags_arith(f | ZF);
                } else {
                    self.set_edx_eax(current);
                    self.set_eflags_arith(f & !ZF);
                }
            }
            // bswap: a 16-bit bswap is undefined; it is treated as the 32-bit form
            0x1c8..=0x1cf => {
                let r = opcode_register(i) as usize;
                if osz == 8 {
                    self.gpr[r] = self.gpr[r].swap_bytes();
                } else {
                    self.gpr[r] = (self.gpr[r] as u32).swap_bytes() as u64;
                }
            }
            // movzx / movsx from byte or word
            0x1b6 | 0x1b7 | 0x1be | 0x1bf => {
                let source_size = if op & 1 == 0 { 1 } else { 2 };
                let loc = self.modrm_location(i, next);
                let v = self.load_operand(loc, source_size, i)?;
                let v = if op < 0x1b8 {
                    v
                } else if source_size == 1 {
                    v as u8 as i8 as i64 as u64
                } else {
                    v as u16 as i16 as i64 as u64
                };
                self.write_gpr(i.reg, osz, true, v);
            }
            // bsf / bsr: a zero source sets ZF and leaves the destination unchanged
            0x1bc | 0x1bd => {
                let loc = self.modrm_location(i, next);
                let v = self.load_operand(loc, osz, i)? & mask(osz);
                let f = self.eflags_full();
                if v == 0 {
                    self.set_eflags_arith(f | ZF);
                } else {
                    let bit_index = if op == 0x1bc {
                        v.trailing_zeros()
                    } else {
                        63 - v.leading_zeros()
                    };
                    self.write_gpr(i.reg, osz, true, bit_index as u64);
                    self.set_eflags_arith(f & !ZF);
                }
            }
            _ => {
                if self.sse(i, next)? {
                    self.rip = next;
                    return Ok(());
                }
                return Err(self.unimplemented_opcode(i));
            }
        }
        self.rip = next;
        Ok(())
    }
}

#[cfg(test)]
#[path = "exec_tests.rs"]
mod exec_tests;
