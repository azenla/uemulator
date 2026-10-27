//! XSAVE feature set: xgetbv/xsetbv, xsave/xrstor/xsaveopt (standard, non-compacted format), and
//! the rdrand/rdseed generator. x87 and SSE are the only state components; the legacy 512 bytes
//! are the fxsave image (`fpu.rs`), followed by the 64-byte header.

use super::exception::Exception;
use super::flags::CF;
use super::fpu::{FXSAVE_AREA_SIZE, LegacyRestore};
use super::sys::{CR0_TS, CR4_OSXSAVE, RAX, RCX, RDX};
use super::{Cpu, ExecResult, mem};

pub(super) const XCR0_X87: u64 = 1 << 0;
pub(super) const XCR0_SSE: u64 = 1 << 1;
/// YMM_Hi128: the upper halves of YMM0-15 (`Cpu::ymm_hi`).
pub(super) const XCR0_AVX: u64 = 1 << 2;
/// Components xsetbv accepts. Extending this also means extending CPUID leaf 0xD and the save
/// area layout below, or the kernel will size its buffers for state the instructions skip.
pub(super) const XCR0_SUPPORTED: u64 = XCR0_X87 | XCR0_SSE | XCR0_AVX;

/// The XSAVE area is the legacy image plus a 64-byte header: XSTATE_BV (8 bytes), XCOMP_BV (8)
/// and 48 reserved bytes, all of which xrstor requires to be zero except XSTATE_BV.
pub(super) const XSAVE_HEADER_SIZE: u64 = 64;
pub(super) const XSAVE_AREA_SIZE: u64 = FXSAVE_AREA_SIZE + XSAVE_HEADER_SIZE;
/// The AVX component sits right after the header (CPUID.0xD.2 reports this offset) and holds
/// YMM0_H..YMM15_H, 16 bytes each.
pub(super) const AVX_STATE_OFFSET: u64 = XSAVE_AREA_SIZE;
pub(super) const AVX_STATE_SIZE: u64 = 256;
pub(super) const XSAVE_AREA_SIZE_WITH_AVX: u64 = AVX_STATE_OFFSET + AVX_STATE_SIZE;
const YMM_HIGH_STRIDE: u64 = 16;
const XSAVE_ALIGNMENT: u64 = 64;
const XSTATE_BV_OFFSET: u64 = FXSAVE_AREA_SIZE;
const XCOMP_BV_OFFSET: u64 = XSTATE_BV_OFFSET + 8;
const XSAVE_HEADER_RESERVED_OFFSET: u64 = XCOMP_BV_OFFSET + 8;
const XSAVE_HEADER_RESERVED_QWORDS: u64 = 6;

/// splitmix64 constants (Steele, Lea, Flood; also what Java's SplittableRandom uses).
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MULTIPLY_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MULTIPLY_2: u64 = 0x94d0_49bb_1331_11eb;

impl Cpu {
    /// Next value of the rdrand/rdseed generator. Deterministic given `MachineConfig::random_seed`.
    pub(super) fn next_random(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(SPLITMIX_GAMMA);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MULTIPLY_1);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MULTIPLY_2);
        z ^ (z >> 31)
    }

    /// 0F C7 /6 (rdrand) and /7 (rdseed), register forms only. Always succeeds: CF=1 and the other
    /// arithmetic flags cleared, as the SDM describes for a valid result.
    pub(super) fn rdrand(&mut self, register: u8, size: u8) {
        let value = self.next_random();
        self.write_gpr(register, size, true, value);
        self.set_eflags_arith(CF);
    }

    /// 0F 01 D0. #UD unless the OS enabled XSAVE (CR4.OSXSAVE); any privilege level may read.
    /// Only XCR0 exists; XGETBV with ECX=1 (XINUSE) needs its own CPUID bit.
    pub(super) fn xgetbv(&mut self) -> ExecResult<()> {
        if self.cr4 & CR4_OSXSAVE == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.gpr[RCX] as u32 != 0 {
            return Err(Exception::general_protection(0));
        }
        let value = self.xcr0;
        self.gpr[RAX] = value & 0xffff_ffff;
        self.gpr[RDX] = value >> 32;
        Ok(())
    }

    /// 0F 01 D1. CPL 0 only; x87 must stay enabled, unknown components and the AVX-without-SSE
    /// combination are #GP.
    pub(super) fn xsetbv(&mut self) -> ExecResult<()> {
        if self.cr4 & CR4_OSXSAVE == 0 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()?;
        if self.gpr[RCX] as u32 != 0 {
            return Err(Exception::general_protection(0));
        }
        let value = (self.gpr[RDX] << 32) | (self.gpr[RAX] & 0xffff_ffff);
        if value & XCR0_X87 == 0
            || value & !XCR0_SUPPORTED != 0
            || (value & XCR0_AVX != 0 && value & XCR0_SSE == 0)
        {
            return Err(Exception::general_protection(0));
        }
        self.xcr0 = value;
        Ok(())
    }

    /// The checks xsave/xsaveopt/xrstor share, in SDM order: #UD without OSXSAVE, #NM under
    /// CR0.TS, #GP(0) for a misaligned area. Returns the requested-feature bitmap, EDX:EAX
    /// masked by XCR0.
    fn xsave_prologue(&self, address: u64) -> ExecResult<u64> {
        if self.cr4 & CR4_OSXSAVE == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.cr0 & CR0_TS != 0 {
            return Err(Exception::device_not_available());
        }
        if address & (XSAVE_ALIGNMENT - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        let mask = (self.gpr[RDX] << 32) | (self.gpr[RAX] & 0xffff_ffff);
        Ok(mask & self.xcr0)
    }

    /// xsave and xsaveopt (REX.W forms are identical here: the x87 pointers are not stored).
    /// Components in the bitmap are always written and always marked in use in XSTATE_BV; xsaveopt's
    /// "skip components in their initial state" is an optimization software cannot observe except
    /// through XSTATE_BV, which saying "in use" keeps correct.
    pub(super) fn xsave(&mut self, address: u64) -> ExecResult<()> {
        let requested = self.xsave_prologue(address)?;
        let size = enabled_area_size(self.xcr0) as u64;
        self.translate(address + size - 1, mem::AccessKind::Write, false)?;
        self.save_legacy_region(
            address,
            requested & XCR0_X87 != 0,
            requested & XCR0_SSE != 0,
        )?;
        if requested & XCR0_AVX != 0 {
            for k in 0..self.ymm_hi.len() {
                let at = address + AVX_STATE_OFFSET + k as u64 * YMM_HIGH_STRIDE;
                self.write128(at, self.ymm_hi[k])?;
            }
        }
        let previous = self.read(address + XSTATE_BV_OFFSET, 8)?;
        self.write(
            address + XSTATE_BV_OFFSET,
            8,
            (previous & !requested) | requested,
        )
    }

    /// xrstor in the standard format. The whole header is validated and every byte of the
    /// components read before any register changes; a component the bitmap names but XSTATE_BV
    /// does not is reset to its initial state instead of loaded.
    pub(super) fn xrstor(&mut self, address: u64) -> ExecResult<()> {
        let requested = self.xsave_prologue(address)?;
        let state_bv = self.read(address + XSTATE_BV_OFFSET, 8)?;
        let compaction = self.read(address + XCOMP_BV_OFFSET, 8)?;
        if compaction != 0 || state_bv & !self.xcr0 != 0 {
            return Err(Exception::general_protection(0));
        }
        for k in 0..XSAVE_HEADER_RESERVED_QWORDS {
            if self.read(address + XSAVE_HEADER_RESERVED_OFFSET + 8 * k, 8)? != 0 {
                return Err(Exception::general_protection(0));
            }
        }
        let component = |bit: u64| {
            if requested & bit == 0 {
                LegacyRestore::Skip
            } else if state_bv & bit != 0 {
                LegacyRestore::Load
            } else {
                LegacyRestore::Init
            }
        };
        // Read the AVX component before anything is committed, like the legacy region.
        let mut ymm_hi = self.ymm_hi;
        match component(XCR0_AVX) {
            LegacyRestore::Skip => {}
            LegacyRestore::Init => ymm_hi = [0; 16],
            LegacyRestore::Load => {
                for (k, half) in ymm_hi.iter_mut().enumerate() {
                    *half =
                        self.read128(address + AVX_STATE_OFFSET + k as u64 * YMM_HIGH_STRIDE)?;
                }
            }
        }
        self.restore_legacy_region(address, component(XCR0_X87), component(XCR0_SSE))?;
        self.ymm_hi = ymm_hi;
        Ok(())
    }
}

/// CPUID.0xD.0:EBX: bytes the XSAVE area needs for the components `xcr0` enables.
pub(super) fn enabled_area_size(xcr0: u64) -> u32 {
    // The header is always part of the area; each further component extends it.
    if xcr0 & XCR0_AVX != 0 {
        XSAVE_AREA_SIZE_WITH_AVX as u32
    } else {
        XSAVE_AREA_SIZE as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{CODE_BASE, Harness, RDI, SCRATCH, Stop};

    const AREA: u64 = SCRATCH + 0x1000;
    const HLT: u8 = 0xf4;
    const XGETBV: &[u8] = &[0x0f, 0x01, 0xd0, HLT];
    const XSETBV: &[u8] = &[0x0f, 0x01, 0xd1, HLT];
    // xsave/xrstor/xsaveopt [rdi]
    const XSAVE_RDI: &[u8] = &[0x0f, 0xae, 0x27, HLT];
    const XRSTOR_RDI: &[u8] = &[0x0f, 0xae, 0x2f, HLT];
    const XSAVEOPT_RDI: &[u8] = &[0x0f, 0xae, 0x37, HLT];
    const RDRAND_RAX: &[u8] = &[0x48, 0x0f, 0xc7, 0xf0, HLT];
    const RDRAND_EAX: &[u8] = &[0x0f, 0xc7, 0xf0, HLT];
    const RDRAND_AX: &[u8] = &[0x66, 0x0f, 0xc7, 0xf0, HLT];
    const RDSEED_RAX: &[u8] = &[0x48, 0x0f, 0xc7, 0xf8, HLT];
    const CR4_FXSR_XSAVE: u64 = super::super::sys::CR4_OSFXSR | CR4_OSXSAVE;

    fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> (Harness, Stop) {
        let mut h = Harness::new(code);
        h.cpu.cr4 |= CR4_FXSR_XSAVE;
        h.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
        h.cpu.gpr[RDI] = AREA;
        setup(&mut h);
        let stop = h.run(100);
        (h, stop)
    }

    fn fault(vector: u8, error_code: Option<u32>) -> Stop {
        Stop::Fault {
            vector,
            error_code,
            software: false,
        }
    }

    fn pattern(k: usize) -> u128 {
        0x0101_0101_0101_0101_0101_0101_0101_0101u128.wrapping_mul(k as u128 + 0x11)
    }

    #[test]
    fn xgetbv_reads_xcr0_into_edx_eax() {
        let (h, stop) = run(XGETBV, |h| {
            h.cpu.gpr[RCX] = 0;
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!((h.cpu.gpr[RAX], h.cpu.gpr[RDX]), (3, 0));
    }

    #[test]
    fn xgetbv_is_undefined_without_osxsave_and_faults_on_other_registers() {
        let (_, stop) = run(XGETBV, |h| h.cpu.cr4 &= !CR4_OSXSAVE);
        assert_eq!(stop, fault(6, None));
        let (_, stop) = run(XGETBV, |h| h.cpu.gpr[RCX] = 1);
        assert_eq!(stop, fault(13, Some(0)));
    }

    #[test]
    fn xsetbv_writes_xcr0_and_validates_it() {
        let write = |value: u64| {
            run(XSETBV, move |h| {
                h.cpu.gpr[RCX] = 0;
                h.cpu.gpr[RAX] = value & 0xffff_ffff;
                h.cpu.gpr[RDX] = value >> 32;
            })
        };
        let (h, stop) = write(XCR0_X87);
        assert_eq!((stop, h.cpu.xcr0), (Stop::Halted, XCR0_X87));
        let (h, stop) = write(XCR0_X87 | XCR0_SSE | XCR0_AVX);
        assert_eq!((stop, h.cpu.xcr0), (Stop::Halted, 7));
        // x87 cannot be disabled, unsupported components and high garbage are #GP
        for bad in [XCR0_SSE, 0, 0b101, 0b1000 | 1, 1 << 63 | 1, 1 << 32 | 1] {
            let (h, stop) = write(bad);
            assert_eq!(stop, fault(13, Some(0)), "value {bad:#x}");
            assert_eq!(h.cpu.xcr0, 3);
        }
        let (_, stop) = run(XSETBV, |h| {
            h.cpu.gpr[RCX] = 1;
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, fault(13, Some(0)));
        let (_, stop) = run(XSETBV, |h| h.cpu.cr4 &= !CR4_OSXSAVE);
        assert_eq!(stop, fault(6, None));
    }

    #[test]
    fn xsave_writes_the_fxsave_image_and_marks_components_in_use() {
        let (h, stop) = run(XSAVE_RDI, |h| {
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
            h.cpu.fcw = 0x027f;
            for k in 0..16 {
                h.cpu.xmm[k] = pattern(k);
            }
        });
        assert_eq!(stop, Stop::Halted);
        let image = h.read(AREA, XSAVE_AREA_SIZE as usize);
        assert_eq!(u16::from_le_bytes([image[0], image[1]]), 0x027f);
        for k in 0..16 {
            let at = 160 + 16 * k;
            assert_eq!(
                u128::from_le_bytes(image[at..at + 16].try_into().unwrap()),
                pattern(k)
            );
        }
        assert_eq!(h.read_u64(AREA + XSTATE_BV_OFFSET), 3);
        assert_eq!(h.read_u64(AREA + XCOMP_BV_OFFSET), 0);
    }

    #[test]
    fn xsave_with_a_partial_mask_leaves_the_other_component_alone() {
        let (h, _) = run(XSAVE_RDI, |h| {
            h.write(AREA, &[0xaa; XSAVE_AREA_SIZE as usize]);
            h.write_u64(AREA + XSTATE_BV_OFFSET, 0b10);
            h.cpu.gpr[RAX] = XCR0_X87;
            h.cpu.gpr[RDX] = 0;
            h.cpu.xmm[0] = 7;
        });
        // x87 saved (FCW is the reset value), SSE bytes untouched, XSTATE_BV keeps bit 1
        assert_eq!(h.read(AREA, 2), 0x037fu16.to_le_bytes());
        assert_eq!(h.read(AREA + 160, 16), [0xaa; 16]);
        assert_eq!(h.read_u64(AREA + XSTATE_BV_OFFSET), 0b11);
    }

    #[test]
    fn xsaveopt_matches_xsave() {
        let setup = |h: &mut Harness| {
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
            h.cpu.xmm[3] = pattern(3);
        };
        let (a, _) = run(XSAVE_RDI, setup);
        let (b, _) = run(XSAVEOPT_RDI, setup);
        assert_eq!(
            a.read(AREA, XSAVE_AREA_SIZE as usize),
            b.read(AREA, XSAVE_AREA_SIZE as usize)
        );
    }

    #[test]
    fn xrstor_round_trips_x87_and_sse_state() {
        let (saved, _) = run(XSAVE_RDI, |h| {
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
            h.cpu.fcw = 0x0c7f;
            h.cpu.mxcsr = 0x1fa0;
            for k in 0..16 {
                h.cpu.xmm[k] = pattern(k);
            }
        });
        let image = saved.read(AREA, XSAVE_AREA_SIZE as usize);
        let (h, stop) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &image);
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!((h.cpu.fcw, h.cpu.mxcsr), (0x0c7f, 0x1fa0));
        for k in 0..16 {
            assert_eq!(h.cpu.xmm[k], pattern(k));
        }
    }

    #[test]
    fn xrstor_initializes_components_whose_state_bit_is_clear() {
        let (h, stop) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
            h.write(AREA + 24, &0x1f80u32.to_le_bytes());
            h.cpu.fcw = 0x0c7f;
            h.cpu.xmm[5] = 9;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!((h.cpu.fcw, h.cpu.mxcsr, h.cpu.xmm[5]), (0x037f, 0x1f80, 0));
    }

    #[test]
    fn xrstor_skips_components_outside_the_mask() {
        let (h, _) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
            h.cpu.gpr[RAX] = XCR0_X87;
            h.cpu.gpr[RDX] = 0;
            h.cpu.xmm[5] = 9;
            h.cpu.fcw = 0x0c7f;
        });
        assert_eq!((h.cpu.fcw, h.cpu.xmm[5]), (0x037f, 9));
    }

    #[test]
    fn xrstor_rejects_bad_headers_without_touching_state() {
        for (offset, value) in [
            (XSTATE_BV_OFFSET, 4u64),
            (XCOMP_BV_OFFSET, 1 << 63),
            (XSAVE_HEADER_RESERVED_OFFSET, 1),
            (XSAVE_HEADER_RESERVED_OFFSET + 40, 1),
        ] {
            let (h, stop) = run(XRSTOR_RDI, |h| {
                h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
                h.write_u64(AREA + offset, value);
                h.cpu.gpr[RAX] = 3;
                h.cpu.gpr[RDX] = 0;
                h.cpu.xmm[1] = 5;
            });
            assert_eq!(stop, fault(13, Some(0)), "offset {offset:#x}");
            assert_eq!(h.cpu.xmm[1], 5);
        }
    }

    #[test]
    fn xrstor_rejects_a_reserved_mxcsr_bit() {
        let (h, stop) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
            h.write(AREA + 24, &0x0001_0000u32.to_le_bytes());
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, fault(13, Some(0)));
        assert_eq!(h.cpu.mxcsr, 0x1f80);
    }

    #[test]
    fn xsave_family_faults_follow_the_sdm_order() {
        for code in [XSAVE_RDI, XRSTOR_RDI, XSAVEOPT_RDI] {
            let (_, stop) = run(code, |h| h.cpu.cr4 &= !CR4_OSXSAVE);
            assert_eq!(stop, fault(6, None), "no OSXSAVE");
            let (_, stop) = run(code, |h| h.cpu.cr0 |= CR0_TS);
            assert_eq!(stop, fault(7, None), "CR0.TS");
            let (_, stop) = run(code, |h| h.cpu.gpr[RDI] = AREA + 16);
            assert_eq!(stop, fault(13, Some(0)), "16-byte aligned is not enough");
        }
    }

    #[test]
    fn xsave_mask_is_limited_to_xcr0() {
        let (h, _) = run(XSAVE_RDI, |h| {
            h.cpu.xcr0 = XCR0_X87;
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
        });
        assert_eq!(h.read_u64(AREA + XSTATE_BV_OFFSET), 1);
    }

    #[test]
    fn rdrand_sets_cf_and_clears_the_other_arithmetic_flags() {
        let (h, stop) = run(RDRAND_RAX, |h| {
            h.cpu.set_eflags_arith(super::super::flags::ARITH & !CF)
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.eflags_full() & super::super::flags::ARITH, CF);
        assert_ne!(h.cpu.gpr[RAX], 0);
    }

    #[test]
    fn rdrand_and_rdseed_follow_the_seed_and_the_operand_size() {
        let (a, _) = run(RDRAND_RAX, |h| h.cpu.rng_state = 1);
        let (b, _) = run(RDSEED_RAX, |h| h.cpu.rng_state = 1);
        let (c, _) = run(RDRAND_RAX, |h| h.cpu.rng_state = 2);
        assert_eq!(a.cpu.gpr[RAX], b.cpu.gpr[RAX]);
        assert_ne!(a.cpu.gpr[RAX], c.cpu.gpr[RAX]);
        // 32-bit writes zero the upper half, 16-bit writes keep it
        let (d, _) = run(RDRAND_EAX, |h| {
            h.cpu.rng_state = 1;
            h.cpu.gpr[RAX] = u64::MAX;
        });
        assert_eq!(d.cpu.gpr[RAX], a.cpu.gpr[RAX] & 0xffff_ffff);
        let (e, _) = run(RDRAND_AX, |h| {
            h.cpu.rng_state = 1;
            h.cpu.gpr[RAX] = u64::MAX;
        });
        assert_eq!(e.cpu.gpr[RAX], (u64::MAX << 16) | (a.cpu.gpr[RAX] & 0xffff));
    }

    #[test]
    fn the_generator_does_not_repeat_over_a_short_run() {
        let mut h = Harness::new(&[HLT]);
        let values: Vec<u64> = (0..1000).map(|_| h.cpu.next_random()).collect();
        let mut sorted = values.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), values.len());
        // splitmix64 reference output for seed 0 (first value)
        let mut reference = Harness::new(&[HLT]);
        reference.cpu.rng_state = 0;
        assert_eq!(reference.cpu.next_random(), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn rdrand_with_a_register_form_prefix_or_memory_operand_is_undefined() {
        // rdrand [rdi]: memory form of /6
        let (_, stop) = run(&[0x0f, 0xc7, 0x37, HLT], |_| {});
        assert_eq!(stop, fault(6, None));
        let _ = CODE_BASE;
    }
}
