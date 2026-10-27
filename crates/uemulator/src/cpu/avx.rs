//! AVX/AVX2 vector instructions. Not advertised until the whole set is done (see
//! `plans/14-avx-and-virtualization.md`); until then XCR0 cannot enable the AVX component, so
//! these paths only run in tests that set `Cpu::xcr0` directly.
//!
//! Most VEX instructions are an SSE instruction with a non-destructive first source (VEX.vvvv),
//! a zeroed upper half, no alignment rule, and for VEX.256 the same operation on the upper 128
//! bits (`Cpu::ymm_hi`). They therefore run through the existing SSE kernels instead of
//! duplicating them: `exec_vex_vector` stages the operands into the registers the legacy code
//! reads, runs it (once per 128-bit lane), and fixes up the destination. `vex_form` is the table
//! that says, per opcode, how. Everything that is not a plain lane-wise extension (cross-lane
//! shuffles, widening/narrowing converts, mask moves, gathers, FMA...) is in `avx_special.rs`.
//!
//! Hazard: the legacy code reads and writes `xmm[reg]`/`xmm[rm]` directly, so the staging must
//! never clobber a register the instruction still has to read. `free_register` hands out a
//! scratch register when the destination collides with another operand.

use super::decode::{Insn, VEX_OP_FLAG};
use super::exception::Exception;
use super::sse::Prefix;
use super::sse::ext::{Op, decode_op};
use super::sys::CR0_TS;
use super::sys::CR4_OSXSAVE;
use super::xsave::{XCR0_AVX, XCR0_SSE};
use super::{Cpu, ExecResult};

const MODRM_MOD_REGISTER: u8 = 3;
const OPCODE_MAP_0F: u16 = 1;
const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;
/// `0F 77`: vzeroupper (VEX.L = 0) and vzeroall (VEX.L = 1).
const OP_VZERO: u16 = 0x177;
const REGISTER_COUNT: usize = 16;

/// Where the legacy kernel leaves its result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Dest {
    /// ModRM.reg.
    Reg,
    /// ModRM.rm (stores; with a register operand it is a vector register).
    Rm,
    /// A general register, memory or the flags: no vector destination to zero-extend.
    None,
}

/// How VEX.vvvv takes part.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Merge {
    /// vvvv is unused and must encode 1111.
    No,
    /// The destination starts as a copy of vvvv: the legacy kernel's "old destination" operand.
    Always,
    /// Only the register form merges (vmovss/vmovsd: the memory form loads and zeroes).
    IfRegister,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Width {
    /// 128 bits, or 256 with VEX.L (the upper lane repeats the operation).
    Pair,
    /// Scalar or otherwise L-insensitive: one pass, VEX.L ignored.
    Scalar,
    /// 128-bit only: VEX.L = 1 is #UD.
    Xmm,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Form {
    pub merge: Merge,
    pub dest: Dest,
    pub width: Width,
}

const fn form(merge: Merge, dest: Dest, width: Width) -> Option<Form> {
    Some(Form { merge, dest, width })
}

/// `dst = op(vvvv, rm)`, packed.
const BINARY: Option<Form> = form(Merge::Always, Dest::Reg, Width::Pair);
/// `dst = op(vvvv, rm)`, scalar.
const BINARY_SCALAR: Option<Form> = form(Merge::Always, Dest::Reg, Width::Scalar);
/// `dst = op(rm)`, packed.
const UNARY: Option<Form> = form(Merge::No, Dest::Reg, Width::Pair);
/// `rm = reg`, packed (memory or register destination).
const STORE: Option<Form> = form(Merge::No, Dest::Rm, Width::Pair);
/// Reads vector operands, writes a general register, memory or flags.
const NO_VECTOR_RESULT: Option<Form> = form(Merge::No, Dest::None, Width::Scalar);

/// Classifies a VEX instruction. `None` means "not a plain extension of a legacy instruction":
/// either it is handled in `avx_special.rs` or it does not exist.
pub(super) fn vex_form(map: u16, opcode: u8, prefix: Prefix) -> Option<Form> {
    use Prefix::*;
    if map == OPCODE_MAP_0F {
        let legacy = match (opcode, prefix) {
            (0x10, Np | P66) => UNARY,
            (0x10, F3 | F2) => form(Merge::IfRegister, Dest::Reg, Width::Scalar),
            (0x11, Np | P66) => STORE,
            (0x11, F3 | F2) => form(Merge::IfRegister, Dest::Rm, Width::Scalar),
            // movlps/movlpd/movhps/movhpd loads merge into vvvv; the register forms are
            // movhlps/movlhps. All are 128-bit only.
            (0x12 | 0x16, Np | P66) => form(Merge::Always, Dest::Reg, Width::Xmm),
            (0x13 | 0x17, Np | P66) => form(Merge::No, Dest::None, Width::Xmm),
            (0x14 | 0x15, Np | P66) => BINARY,
            (0x28, Np | P66) => UNARY,
            (0x29, Np | P66) => STORE,
            (0x2a, F3 | F2) => BINARY_SCALAR,
            (0x2b, Np | P66) => STORE,
            (0x2c | 0x2d, F3 | F2) => NO_VECTOR_RESULT,
            (0x2e | 0x2f, Np | P66) => NO_VECTOR_RESULT,
            (0x51, Np | P66) => UNARY,
            (0x51, F3 | F2) => BINARY_SCALAR,
            (0x52 | 0x53, Np) => UNARY,
            (0x52 | 0x53, F3) => BINARY_SCALAR,
            (0x54..=0x57, Np | P66) => BINARY,
            (0x58 | 0x59 | 0x5c..=0x5f, Np | P66) => BINARY,
            (0x58 | 0x59 | 0x5c..=0x5f, F3 | F2) => BINARY_SCALAR,
            (0x5a, F3 | F2) => BINARY_SCALAR,
            (0x5b, Np | P66 | F3) => UNARY,
            (0x60..=0x6d, P66) => BINARY,
            (0x6e, P66) => form(Merge::No, Dest::Reg, Width::Xmm),
            (0x6f, P66 | F3) => UNARY,
            (0x70, P66 | F3 | F2) => UNARY,
            (0x74..=0x76, P66) => BINARY,
            (0x7c | 0x7d, P66 | F2) => BINARY,
            (0x7e, P66) => form(Merge::No, Dest::None, Width::Xmm),
            (0x7e, F3) => form(Merge::No, Dest::Reg, Width::Xmm),
            (0x7f, P66 | F3) => STORE,
            (0xc2, Np | P66) => BINARY,
            (0xc2, F3 | F2) => BINARY_SCALAR,
            (0xc4, P66) => form(Merge::Always, Dest::Reg, Width::Xmm),
            (0xc5, P66) => form(Merge::No, Dest::None, Width::Xmm),
            (0xc6, Np | P66) => BINARY,
            (0xd0, P66 | F2) => BINARY,
            (0xd4 | 0xd5, P66) => BINARY,
            (0xd6, P66) => form(Merge::No, Dest::Rm, Width::Xmm),
            (0xd8..=0xdf, P66) => BINARY,
            (0xe0 | 0xe3 | 0xe4 | 0xe5, P66) => BINARY,
            (0xe7, P66) => STORE,
            (0xe8..=0xef, P66) => BINARY,
            (0xf0, F2) => UNARY,
            (0xf4..=0xf6, P66) => BINARY,
            (0xf7, P66) => form(Merge::No, Dest::None, Width::Xmm),
            (0xf8..=0xfe, P66) => BINARY,
            _ => None,
        };
        if legacy.is_some() {
            return legacy;
        }
    }
    // The SSE3..SSE4.2/AES/PCLMUL extension maps share one decoder with the legacy code.
    form_for_extension(decode_op(map, opcode, prefix)?, map, opcode)
}

fn form_for_extension(op: Op, map: u16, opcode: u8) -> Option<Form> {
    Some(match op {
        Op::Movddup | Op::Movsldup | Op::Movshdup | Op::Lddqu | Op::LoadAligned | Op::Unary(_) => {
            // pabs*, phminposuw (128-bit only), pmovsx/zx are widening and handled elsewhere
            if map == OPCODE_MAP_0F38 && opcode == 0x41 {
                return form(Merge::No, Dest::Reg, Width::Xmm);
            }
            Form {
                merge: Merge::No,
                dest: Dest::Reg,
                width: Width::Pair,
            }
        }
        Op::AddSub { .. } | Op::Horizontal { .. } | Op::Binary(_) | Op::WithImm(_) => {
            // AES and PCLMULQDQ have 256-bit forms only with VAES/VPCLMULQDQ, which are not here.
            let crypto = (map == OPCODE_MAP_0F38 && matches!(opcode, 0xdb..=0xdf))
                || (map == OPCODE_MAP_0F3A && matches!(opcode, 0x44 | 0xdf));
            Form {
                merge: Merge::Always,
                dest: Dest::Reg,
                width: if crypto { Width::Xmm } else { Width::Pair },
            }
        }
        Op::Round { scalar: false, .. } => Form {
            merge: Merge::No,
            dest: Dest::Reg,
            width: Width::Pair,
        },
        Op::Round { scalar: true, .. } => Form {
            merge: Merge::Always,
            dest: Dest::Reg,
            width: Width::Scalar,
        },
        Op::DotProduct { double } => Form {
            merge: Merge::Always,
            dest: Dest::Reg,
            width: if double { Width::Xmm } else { Width::Pair },
        },
        Op::Extract(_) => Form {
            merge: Merge::No,
            dest: Dest::None,
            width: Width::Xmm,
        },
        Op::Insert(_) | Op::Insertps => Form {
            merge: Merge::Always,
            dest: Dest::Reg,
            width: Width::Xmm,
        },
        Op::StringCompare { .. } => Form {
            merge: Merge::No,
            dest: Dest::None,
            width: Width::Xmm,
        },
        // ptest, pmovsx/zx and the variable blends need operand handling of their own
        Op::Ptest | Op::Widen { .. } | Op::WithMask(_) => return None,
    })
}

impl Cpu {
    /// #UD unless the OS enabled AVX state (CR4.OSXSAVE and XCR0 bits 2:1), then #NM under CR0.TS.
    pub(super) fn vex_gate(&self) -> ExecResult<()> {
        const NEEDED: u64 = XCR0_SSE | XCR0_AVX;
        if self.cr4 & CR4_OSXSAVE == 0 || self.xcr0 & NEEDED != NEEDED {
            return Err(Exception::invalid_opcode());
        }
        if self.cr0 & CR0_TS != 0 {
            return Err(Exception::device_not_available());
        }
        Ok(())
    }

    /// A register that is neither of `avoid`, usable as scratch.
    fn free_register(avoid: [usize; 3]) -> usize {
        (0..REGISTER_COUNT).find(|r| !avoid.contains(r)).unwrap()
    }

    /// Entry for every VEX-encoded instruction that is not a BMI one.
    pub(super) fn exec_vex_vector(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op & !VEX_OP_FLAG;
        let map = op >> 8;
        let opcode = (op & 0xff) as u8;
        let prefix = match (i.rep, i.p66) {
            (1, _) => Prefix::F3,
            (2, _) => Prefix::F2,
            (_, true) => Prefix::P66,
            _ => Prefix::Np,
        };
        if op == OP_VZERO && prefix == Prefix::Np {
            self.vex_gate()?;
            if i.vex_v != 0 {
                return Err(Exception::invalid_opcode());
            }
            // vzeroupper clears bits 255:128 of every register; vzeroall also clears the low halves
            self.ymm_hi = [0; REGISTER_COUNT];
            if i.vex_l {
                self.xmm = [0; REGISTER_COUNT];
            }
            self.rip = next;
            return Ok(());
        }
        let Some(form) = vex_form(map, opcode, prefix) else {
            if self.exec_vex_special(i, next, prefix)? {
                self.rip = next;
                return Ok(());
            }
            return Err(self.unimplemented_opcode(i));
        };
        self.vex_gate()?;
        if (form.merge == Merge::No && i.vex_v != 0)
            || (form.width == Width::Xmm && i.vex_l)
            || (form.merge == Merge::IfRegister && i.md != MODRM_MOD_REGISTER && i.vex_v != 0)
        {
            return Err(Exception::invalid_opcode());
        }
        let snapshot = (self.xmm, self.ymm_hi);
        let result = self.run_vex_passes(i, next, form);
        self.vex_active = false;
        if let Err(error) = result {
            // A fault in the second lane must not leave the first one written.
            (self.xmm, self.ymm_hi) = snapshot;
            return Err(error);
        }
        self.rip = next;
        Ok(())
    }

    fn run_vex_passes(&mut self, i: &Insn, next: u64, form: Form) -> ExecResult<()> {
        let two_lanes = i.vex_l && form.width == Width::Pair;
        let mut legacy = *i;
        legacy.vex = false;
        legacy.op = i.op & !VEX_OP_FLAG;
        let vvvv = i.vex_v as usize;
        let destination = match form.dest {
            Dest::Reg => Some(i.reg as usize),
            Dest::Rm if i.md == MODRM_MOD_REGISTER => Some(i.rm as usize),
            _ => None,
        };
        let merges = match form.merge {
            Merge::No => false,
            Merge::Always => true,
            Merge::IfRegister => i.md == MODRM_MOD_REGISTER,
        };
        self.vex_active = true;
        for lane in 0..if two_lanes { 2 } else { 1 } {
            if lane == 1 {
                // The upper lane is processed as if it were the whole register file, and its
                // memory operand is the next 16 bytes.
                core::mem::swap(&mut self.xmm, &mut self.ymm_hi);
                legacy.disp = legacy.disp.wrapping_add(16);
            }
            let outcome = self.run_vex_lane(&mut legacy, next, destination, merges, vvvv);
            if lane == 1 {
                core::mem::swap(&mut self.xmm, &mut self.ymm_hi);
            }
            if !outcome? {
                return Err(self.unimplemented_opcode(i));
            }
        }
        if let Some(register) = destination
            && !two_lanes
        {
            self.ymm_hi[register] = 0;
        }
        Ok(())
    }

    /// One 128-bit pass through the legacy kernel. Returns whether the kernel knew the opcode.
    fn run_vex_lane(
        &mut self,
        legacy: &mut Insn,
        next: u64,
        destination: Option<usize>,
        merges: bool,
        vvvv: usize,
    ) -> ExecResult<bool> {
        let mut scratch: Option<(usize, u128)> = None;
        let original = *legacy;
        if let (true, Some(destination)) = (merges, destination)
            && vvvv != destination
        {
            let is_reg_source = legacy.md == MODRM_MOD_REGISTER;
            // The legacy kernel reads the old destination as its first source, so seed it from
            // vvvv. If the destination is also the register source it still has to be read:
            // point the kernel at a scratch copy.
            let source_field_is_destination = if destination == legacy.rm as usize {
                // stores: the destination is rm and the kernel's source is reg
                legacy.reg as usize == destination
            } else {
                is_reg_source && legacy.rm as usize == destination
            };
            if source_field_is_destination {
                let k = Self::free_register([legacy.reg as usize, legacy.rm as usize, vvvv]);
                scratch = Some((k, self.xmm[k]));
                self.xmm[k] = self.xmm[destination];
                if destination == legacy.rm as usize {
                    legacy.reg = k as u8;
                } else {
                    legacy.rm = k as u8;
                }
            }
            self.xmm[destination] = self.xmm[vvvv];
        }
        let outcome = self.sse(legacy, next);
        if let Some((k, saved)) = scratch {
            self.xmm[k] = saved;
        }
        *legacy = Insn {
            reg: original.reg,
            rm: original.rm,
            ..*legacy
        };
        outcome
    }
}

#[cfg(test)]
#[path = "avx_tests.rs"]
mod avx_tests;
