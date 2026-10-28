//! EVEX conversions between integer and floating-point elements and between float widths, packed
//! and scalar.

use super::*;

const OPCODE_MAP_0F: u16 = 1;

impl Cpu {
    pub(super) fn exec_evex_convert(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        if map != OPCODE_MAP_0F {
            return Ok(false);
        }
        let (single_or_double, dword_or_qword) = (format_of(wide), if wide { 8 } else { 4 });
        // (source bytes, destination bytes, rounding, per-element function)
        match (opcode, prefix) {
            (0x5b, Np) if wide => {
                // vcvtqq2ps
                self.convert_packed(i, next, 8, 4, Rounding::Embedded, |env, bits| {
                    int_to_float(env, SINGLE, bits as i64)
                })?;
            }
            (0xe6, F3) => {
                // vcvtdq2pd (W0) and vcvtqq2pd (W1)
                let rounding = if wide {
                    Rounding::Embedded
                } else {
                    Rounding::None
                };
                self.convert_packed(i, next, dword_or_qword, 8, rounding, move |env, bits| {
                    int_to_float(env, DOUBLE, signed(bits, dword_or_qword))
                })?;
            }
            (0x7a, F3) => {
                let rounding = if wide {
                    Rounding::Embedded
                } else {
                    Rounding::None
                };
                self.convert_packed(i, next, dword_or_qword, 8, rounding, move |env, bits| {
                    uint_to_float(env, DOUBLE, bits & size_mask(dword_or_qword))
                })?;
            }
            (0x7a, F2) => {
                // vcvtudq2ps (W0) and vcvtuqq2ps (W1)
                self.convert_packed(
                    i,
                    next,
                    dword_or_qword,
                    4,
                    Rounding::Embedded,
                    move |env, bits| uint_to_float(env, SINGLE, bits & size_mask(dword_or_qword)),
                )?;
            }
            (0xe6, F2) if wide => {
                self.convert_packed(i, next, 8, 4, Rounding::Embedded, |env, bits| {
                    float_to_int(env, DOUBLE, bits, 32, false)
                })?;
            }
            (0xe6, P66) if wide => {
                self.convert_packed(i, next, 8, 4, Rounding::Sae, |env, bits| {
                    float_to_int(env, DOUBLE, bits, 32, true)
                })?;
            }
            (0x5a, Np) if !wide => {
                self.convert_packed(i, next, 4, 8, Rounding::Sae, |env, bits| {
                    convert_float(env, SINGLE, DOUBLE, bits)
                })?;
            }
            (0x5a, P66) if wide => {
                self.convert_packed(i, next, 8, 4, Rounding::Embedded, |env, bits| {
                    convert_float(env, DOUBLE, SINGLE, bits)
                })?;
            }
            // float to unsigned 32/64 and float to signed 64: the destination width follows W for
            // the unsigned dword forms (no prefix) and the quadword forms (66)
            (0x78 | 0x79, Np) => {
                let truncate = opcode == 0x78;
                let rounding = if truncate {
                    Rounding::Sae
                } else {
                    Rounding::Embedded
                };
                self.convert_packed(i, next, dword_or_qword, 4, rounding, move |env, bits| {
                    float_to_uint(env, single_or_double, bits, 32, truncate)
                })?;
            }
            (0x78..=0x7b, P66) => {
                let truncate = matches!(opcode, 0x78 | 0x7a);
                let unsigned = opcode <= 0x79;
                let rounding = if truncate {
                    Rounding::Sae
                } else {
                    Rounding::Embedded
                };
                self.convert_packed(i, next, dword_or_qword, 8, rounding, move |env, bits| {
                    if unsigned {
                        float_to_uint(env, single_or_double, bits, 64, truncate)
                    } else {
                        float_to_int(env, single_or_double, bits, 64, truncate)
                    }
                })?;
            }
            (0x2a | 0x7b, F3 | F2) => {
                self.convert_integer_to_scalar(i, next, opcode == 0x7b, prefix == F3, wide)?
            }
            (0x2c | 0x2d | 0x78 | 0x79, F3 | F2) => {
                self.convert_scalar_to_integer(i, next, opcode, prefix == F3, wide)?;
            }
            (0x5a, F3) => {
                self.fp_scalar(i, next, SINGLE, Rounding::Sae, false, |env, _, b, _| {
                    convert_float(env, SINGLE, DOUBLE, b)
                })?;
            }
            (0x5a, F2) => {
                self.fp_scalar(
                    i,
                    next,
                    DOUBLE,
                    Rounding::Embedded,
                    false,
                    |env, _, b, _| convert_float(env, DOUBLE, SINGLE, b),
                )?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Packed conversion of `count` elements from `source_bytes`-wide to `destination_bytes`-wide
    /// elements, where `count` is the vector length over the wider of the two.
    pub(super) fn convert_packed(
        &mut self,
        i: &Insn,
        next: u64,
        source_bytes: usize,
        destination_bytes: usize,
        rounding: Rounding,
        mut f: impl FnMut(&mut FpEnv, u64) -> u64,
    ) -> ExecResult<()> {
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let length = self.vector_bytes(i, rounding)?;
        let count = length / source_bytes.max(destination_bytes);
        let mask = self.write_mask(i);
        let shape = RmShape {
            bytes: count * source_bytes,
            element: source_bytes,
            broadcast: true,
            suppress: (i.evex_mask != 0).then_some(mask),
        };
        let source = self.fetch_rm(i, next, shape)?;
        let result = self.with_static_rounding(i, rounding, |cpu| {
            let mut env = FpEnv::new(cpu.mxcsr);
            let out = build(count * destination_bytes, destination_bytes, |k| {
                if mask >> k & 1 == 1 {
                    f(&mut env, element(&source, source_bytes, k))
                } else {
                    0
                }
            });
            cpu.fp_commit(&env)?;
            Ok(out)
        })?;
        self.commit_masked(
            i,
            i.reg as usize,
            destination_bytes,
            count * destination_bytes,
            &result,
        );
        Ok(())
    }

    /// vcvtsi2ss/sd and vcvtusi2ss/sd: a general register or memory integer into element 0, the
    /// rest of the low lane from vvvv.
    fn convert_integer_to_scalar(
        &mut self,
        i: &Insn,
        next: u64,
        unsigned: bool,
        single: bool,
        wide: bool,
    ) -> ExecResult<()> {
        let format = if single { SINGLE } else { DOUBLE };
        let size = if wide { 8 } else { 4 };
        let source = if i.md == 3 {
            self.read_gpr(i.rm & 0xf, size, true)
        } else {
            let address = self.evex_address(i, next, size as usize)?;
            self.read(address, size)?
        };
        let element = format.bytes() as usize;
        let enabled = i.evex_mask == 0 || self.opmask[i.evex_mask as usize] & 1 == 1;
        let value = if enabled {
            self.with_static_rounding(i, Rounding::Embedded, |cpu| {
                let mut env = FpEnv::new(cpu.mxcsr);
                let bits = if unsigned {
                    uint_to_float(&mut env, format, source)
                } else {
                    int_to_float(&mut env, format, signed(source, size as usize))
                };
                cpu.fp_commit(&env)?;
                Ok(bits)
            })?
        } else {
            0
        };
        let first = self.zmm(i.vex_v as usize)[0];
        let lane = first & !(size_mask(element) as u128) | value as u128;
        self.commit_scalar(i, i.reg as usize, element, lane);
        Ok(())
    }

    /// vcvt(t)ss2si, vcvt(t)sd2si and the unsigned vcvt(t)ss2usi/sd2usi: element 0 to a general
    /// register.
    fn convert_scalar_to_integer(
        &mut self,
        i: &Insn,
        next: u64,
        opcode: u8,
        single: bool,
        wide: bool,
    ) -> ExecResult<()> {
        if i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let format = if single { SINGLE } else { DOUBLE };
        let element = format.bytes() as usize;
        let shape = RmShape {
            bytes: element,
            element,
            broadcast: false,
            suppress: None,
        };
        let source = self.fetch_rm(i, next, shape)?[0] as u64 & size_mask(element);
        let truncate = matches!(opcode, 0x2c | 0x78);
        let unsigned = opcode >= 0x78;
        let width = if wide { 64 } else { 32 };
        let rounding = if truncate {
            Rounding::Sae
        } else {
            Rounding::Embedded
        };
        let result = self.with_static_rounding(i, rounding, |cpu| {
            let mut env = FpEnv::new(cpu.mxcsr);
            let bits = if unsigned {
                float_to_uint(&mut env, format, source, width, truncate)
            } else {
                float_to_int(&mut env, format, source, width, truncate)
            };
            cpu.fp_commit(&env)?;
            Ok(bits)
        })?;
        self.write_gpr(i.reg & 0xf, (width / 8) as u8, true, result);
        Ok(())
    }
}
