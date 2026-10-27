//! Register layouts and architectural constants shared by both backends. Nothing here touches
//! the hardware, so it also builds (and is tested) on a hosted target.

/// Guest general-purpose registers, in x86 encoding order. The layout is read by offset from the
/// world-switch stubs in `arch`. The VMCB holds the guest's rax and rsp on SVM, and the VMCS holds
/// rsp on VMX, so those slots are not touched by the SVM stub (rax, rsp) or the VMX stub (rsp).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Gprs {
    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rbx: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

pub const MSR_TSC: u32 = 0x10;
pub const MSR_FEATURE_CONTROL: u32 = 0x3a;
pub const MSR_PAT: u32 = 0x277;
pub const MSR_SYSENTER_CS: u32 = 0x174;
pub const MSR_SYSENTER_ESP: u32 = 0x175;
pub const MSR_SYSENTER_EIP: u32 = 0x176;
pub const MSR_EFER: u32 = 0xc000_0080;
pub const MSR_STAR: u32 = 0xc000_0081;
pub const MSR_LSTAR: u32 = 0xc000_0082;
pub const MSR_CSTAR: u32 = 0xc000_0083;
pub const MSR_SFMASK: u32 = 0xc000_0084;
pub const MSR_FS_BASE: u32 = 0xc000_0100;
pub const MSR_GS_BASE: u32 = 0xc000_0101;
pub const MSR_KERNEL_GS_BASE: u32 = 0xc000_0102;

/// MSRs the guest reads and writes without a VM exit (cleared in the MSR permission map on SVM and
/// the MSR bitmap on VMX). Every one is either switched by hardware (VMCB/VMCS fields, VMLOAD and
/// VMSAVE, or the VMX MSR lists) or has no host state to protect; the TSC is not offset, so the
/// guest sees the host's.
pub const PASS_THROUGH_MSRS: [u32; 12] = [
    MSR_TSC,
    MSR_STAR,
    MSR_LSTAR,
    MSR_CSTAR,
    MSR_SFMASK,
    MSR_FS_BASE,
    MSR_GS_BASE,
    MSR_KERNEL_GS_BASE,
    MSR_SYSENTER_CS,
    MSR_SYSENTER_ESP,
    MSR_SYSENTER_EIP,
    MSR_PAT,
];

/// The two ranges of MSR numbers that the SVM MSR permission map and the VMX MSR bitmap cover:
/// `0..=MSR_LOW_RANGE_END` and the same length starting at `MSR_HIGH_RANGE_START`. Other MSRs
/// always exit.
pub const MSR_LOW_RANGE_END: u32 = 0x1fff;
pub const MSR_HIGH_RANGE_START: u32 = 0xc000_0000;
pub const MSR_HIGH_RANGE_END: u32 = 0xc000_1fff;

pub const EFER_SCE: u64 = 1;
pub const EFER_LME: u64 = 1 << 8;
pub const EFER_LMA: u64 = 1 << 10;
pub const EFER_NXE: u64 = 1 << 11;
pub const EFER_SVME: u64 = 1 << 12;

pub const RFLAGS_IF: u64 = 1 << 9;
/// RFLAGS bit 1 reads as one and must be one in guest state.
pub const RFLAGS_RESERVED_ONE: u64 = 1 << 1;

pub const CR0_PE: u64 = 1 << 0;
pub const CR0_MP: u64 = 1 << 1;
pub const CR0_ET: u64 = 1 << 4;
pub const CR0_NE: u64 = 1 << 5;
pub const CR0_WP: u64 = 1 << 16;
pub const CR0_PG: u64 = 1 << 31;
pub const CR4_PAE: u64 = 1 << 5;

/// CR0 and CR4 as the 64-bit Linux boot protocol enters the kernel: paging on, long mode, PAE.
pub const GUEST_ENTRY_CR0: u64 = CR0_PG | CR0_WP | CR0_NE | CR0_ET | CR0_MP | CR0_PE;
pub const GUEST_ENTRY_CR4: u64 = CR4_PAE;

/// DR7 and DR6 reset values.
pub const DR7_RESET: u64 = 0x400;
pub const DR6_RESET: u64 = 0xffff_0ff0;

/// IA32_PAT reset value: WB, WT, UC-, UC repeated for entries 0-3 and 4-7.
pub const PAT_RESET: u64 = 0x0007_0406_0007_0406;

/// Segment limit of a flat 4 GiB segment.
pub const FLAT_SEGMENT_LIMIT: u32 = 0xffff_ffff;
/// Limit of the LDTR and TR the guest starts with.
pub const SYSTEM_SEGMENT_LIMIT: u32 = 0xffff;

/// Interrupt and exception injection encoding shared by VMX (VM-entry interruption-information)
/// and SVM (EVENTINJ): vector in bits 7:0, type in 10:8, "has error code" in bit 11, valid in 31.
pub const EVENT_TYPE_SHIFT: u32 = 8;
pub const EVENT_TYPE_HARDWARE_EXCEPTION: u32 = 3;
pub const EVENT_ERROR_CODE_VALID: u32 = 1 << 11;
pub const EVENT_VALID: u32 = 1 << 31;

pub const VECTOR_INVALID_OPCODE: u32 = 6;
pub const VECTOR_GENERAL_PROTECTION: u32 = 13;

pub const CPUID_LEAF_VENDOR: u32 = 0;
pub const CPUID_LEAF_FEATURES: u32 = 1;
pub const CPUID_LEAF_EXT_MAX: u32 = 0x8000_0000;
pub const CPUID_LEAF_EXT_FEATURES: u32 = 0x8000_0001;
pub const CPUID_LEAF_BRAND_STRING_FIRST: u32 = 0x8000_0002;
pub const CPUID_LEAF_BRAND_STRING_LAST: u32 = 0x8000_0004;
pub const CPUID_LEAF_EXT_POWER_MANAGEMENT: u32 = 0x8000_0007;
pub const CPUID_LEAF_EXT_ADDRESS_SIZES: u32 = 0x8000_0008;
pub const CPUID_LEAF_SVM: u32 = 0x8000_000a;

/// CPUID.1:ECX bit 5.
pub const CPUID_1_ECX_VMX: u32 = 1 << 5;
/// CPUID.80000001h:ECX bit 2.
pub const CPUID_EXT1_ECX_SVM: u32 = 1 << 2;

/// Interruption-information / EVENTINJ value (without the valid bit) for a hardware exception.
pub fn exception_event(vector: u32, has_error_code: bool) -> u32 {
    let error_code = if has_error_code {
        EVENT_ERROR_CODE_VALID
    } else {
        0
    };
    vector | (EVENT_TYPE_HARDWARE_EXCEPTION << EVENT_TYPE_SHIFT) | error_code
}

/// The 10-byte memory operand of SGDT, SIDT, LGDT and LIDT.
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct DescriptorTableRegister {
    pub limit: u16,
    pub base: u64,
}

const FXSAVE_AREA_SIZE: usize = 512;
const FXSAVE_FCW_OFFSET: usize = 0;
const FXSAVE_MXCSR_OFFSET: usize = 24;
/// x87 control word after FINIT and MXCSR after reset.
const FCW_RESET: u16 = 0x37f;
const MXCSR_RESET: u32 = 0x1f80;

/// The guest's x87/SSE state as an FXSAVE image (16-byte aligned, as FXSAVE requires). Firmware
/// code that runs between VM exits (interrupt handlers, `Stall`) clobbers XMM, so the event loop
/// restores the guest's image before every entry and saves it after every exit.
#[repr(C, align(16))]
pub struct FpuState(pub [u8; FXSAVE_AREA_SIZE]);

impl FpuState {
    pub fn reset() -> Self {
        let mut s = FpuState([0; FXSAVE_AREA_SIZE]);
        s.0[FXSAVE_FCW_OFFSET..FXSAVE_FCW_OFFSET + 2].copy_from_slice(&FCW_RESET.to_le_bytes());
        s.0[FXSAVE_MXCSR_OFFSET..FXSAVE_MXCSR_OFFSET + 4]
            .copy_from_slice(&MXCSR_RESET.to_le_bytes());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn gprs_follow_the_x86_register_encoding_order() {
        let offsets = [
            offset_of!(Gprs, rax),
            offset_of!(Gprs, rcx),
            offset_of!(Gprs, rdx),
            offset_of!(Gprs, rbx),
            offset_of!(Gprs, rsp),
            offset_of!(Gprs, rbp),
            offset_of!(Gprs, rsi),
            offset_of!(Gprs, rdi),
            offset_of!(Gprs, r8),
            offset_of!(Gprs, r9),
            offset_of!(Gprs, r10),
            offset_of!(Gprs, r11),
            offset_of!(Gprs, r12),
            offset_of!(Gprs, r13),
            offset_of!(Gprs, r14),
            offset_of!(Gprs, r15),
        ];
        for (index, offset) in offsets.into_iter().enumerate() {
            assert_eq!(offset, index * 8, "register {index}");
        }
        assert_eq!(size_of::<Gprs>(), 16 * 8);
    }

    #[test]
    fn gprs_default_is_all_zero() {
        let g = Gprs::default();
        assert_eq!((g.rax, g.rsp, g.r15), (0, 0, 0));
    }

    #[test]
    fn descriptor_table_register_is_the_ten_byte_sgdt_operand() {
        assert_eq!(size_of::<DescriptorTableRegister>(), 10);
        assert_eq!(offset_of!(DescriptorTableRegister, base), 2);
    }

    #[test]
    fn efer_bits_are_at_their_architectural_positions() {
        assert_eq!(EFER_SCE, 1 << 0);
        assert_eq!(EFER_LME, 1 << 8);
        assert_eq!(EFER_LMA, 1 << 10);
        assert_eq!(EFER_NXE, 1 << 11);
        assert_eq!(EFER_SVME, 1 << 12);
    }

    #[test]
    fn msr_numbers_match_the_manuals() {
        assert_eq!(MSR_TSC, 0x10);
        assert_eq!(MSR_FEATURE_CONTROL, 0x3a);
        assert_eq!(MSR_SYSENTER_CS, 0x174);
        assert_eq!(MSR_SYSENTER_ESP, 0x175);
        assert_eq!(MSR_SYSENTER_EIP, 0x176);
        assert_eq!(MSR_PAT, 0x277);
        assert_eq!(MSR_EFER, 0xc000_0080);
        assert_eq!(MSR_STAR, 0xc000_0081);
        assert_eq!(MSR_LSTAR, 0xc000_0082);
        assert_eq!(MSR_CSTAR, 0xc000_0083);
        assert_eq!(MSR_SFMASK, 0xc000_0084);
        assert_eq!(MSR_FS_BASE, 0xc000_0100);
        assert_eq!(MSR_GS_BASE, 0xc000_0101);
        assert_eq!(MSR_KERNEL_GS_BASE, 0xc000_0102);
    }

    #[test]
    fn the_two_covered_msr_ranges_have_the_same_length() {
        assert_eq!(MSR_LOW_RANGE_END, 0x1fff);
        assert_eq!(MSR_HIGH_RANGE_START, 0xc000_0000);
        assert_eq!(MSR_HIGH_RANGE_END - MSR_HIGH_RANGE_START, MSR_LOW_RANGE_END);
    }

    #[test]
    fn pass_through_msrs_are_distinct() {
        for (i, a) in PASS_THROUGH_MSRS.iter().enumerate() {
            for b in &PASS_THROUGH_MSRS[i + 1..] {
                assert_ne!(a, b, "{a:#x} is listed twice");
            }
        }
    }

    #[test]
    fn every_pass_through_msr_lies_in_a_range_the_permission_maps_cover() {
        for msr in PASS_THROUGH_MSRS {
            let covered = msr <= MSR_LOW_RANGE_END
                || (MSR_HIGH_RANGE_START..=MSR_HIGH_RANGE_END).contains(&msr);
            assert!(
                covered,
                "{msr:#x} would still exit, being outside both ranges"
            );
        }
    }

    #[test]
    fn pass_through_msrs_include_the_syscall_state_a_64_bit_kernel_uses() {
        for msr in [
            MSR_STAR,
            MSR_LSTAR,
            MSR_CSTAR,
            MSR_SFMASK,
            MSR_KERNEL_GS_BASE,
            MSR_FS_BASE,
            MSR_GS_BASE,
            MSR_TSC,
        ] {
            assert!(PASS_THROUGH_MSRS.contains(&msr), "{msr:#x}");
        }
    }

    #[test]
    fn msrs_the_host_must_mediate_are_not_passed_through() {
        // EFER carries the hidden SVME bit on SVM and the mode bits on VMX; the feature-control
        // MSR would let the guest see the firmware's lock.
        assert!(!PASS_THROUGH_MSRS.contains(&MSR_EFER));
        assert!(!PASS_THROUGH_MSRS.contains(&MSR_FEATURE_CONTROL));
    }

    #[test]
    fn guest_enters_with_paging_write_protect_and_native_fpu_errors() {
        // PG | WP | NE | ET | MP | PE.
        assert_eq!(GUEST_ENTRY_CR0, 0x8001_0033);
        assert_eq!(GUEST_ENTRY_CR0 & CR0_PG, CR0_PG);
    }

    #[test]
    fn guest_enters_with_only_pae_in_cr4() {
        assert_eq!(GUEST_ENTRY_CR4, 0x20);
    }

    #[test]
    fn debug_registers_reset_to_their_manual_values() {
        assert_eq!(DR7_RESET, 0x400);
        assert_eq!(DR6_RESET, 0xffff_0ff0);
    }

    #[test]
    fn rflags_starts_with_only_the_reserved_bit_one() {
        assert_eq!(RFLAGS_RESERVED_ONE, 0x2);
        assert_eq!(RFLAGS_IF, 0x200);
    }

    #[test]
    fn pat_resets_to_wb_wt_uc_minus_uc_twice() {
        const WB: u8 = 0x06;
        const WT: u8 = 0x04;
        const UC_MINUS: u8 = 0x07;
        const UC: u8 = 0x00;
        assert_eq!(
            PAT_RESET.to_le_bytes(),
            [WB, WT, UC_MINUS, UC, WB, WT, UC_MINUS, UC]
        );
    }

    #[test]
    fn segment_limits_are_a_flat_4_gib_and_a_64_kib_system_segment() {
        assert_eq!(FLAT_SEGMENT_LIMIT, 0xffff_ffff);
        assert_eq!(SYSTEM_SEGMENT_LIMIT, 0xffff);
    }

    #[test]
    fn cpuid_leaf_and_bit_numbers_match_the_manuals() {
        assert_eq!(CPUID_LEAF_VENDOR, 0);
        assert_eq!(CPUID_LEAF_FEATURES, 1);
        assert_eq!(CPUID_LEAF_EXT_MAX, 0x8000_0000);
        assert_eq!(CPUID_LEAF_EXT_FEATURES, 0x8000_0001);
        assert_eq!(CPUID_LEAF_BRAND_STRING_FIRST, 0x8000_0002);
        assert_eq!(CPUID_LEAF_BRAND_STRING_LAST, 0x8000_0004);
        assert_eq!(CPUID_LEAF_EXT_POWER_MANAGEMENT, 0x8000_0007);
        assert_eq!(CPUID_LEAF_EXT_ADDRESS_SIZES, 0x8000_0008);
        assert_eq!(CPUID_LEAF_SVM, 0x8000_000a);
        assert_eq!(CPUID_1_ECX_VMX, 1 << 5);
        assert_eq!(CPUID_EXT1_ECX_SVM, 1 << 2);
    }

    #[test]
    fn general_protection_with_error_code_encodes_as_the_manuals_say() {
        // Vector 13, type 3 (hardware exception), error code valid: SVM EVENTINJ without V.
        assert_eq!(exception_event(VECTOR_GENERAL_PROTECTION, true), 0x0b0d);
    }

    #[test]
    fn invalid_opcode_has_no_error_code() {
        assert_eq!(exception_event(VECTOR_INVALID_OPCODE, false), 0x0306);
    }

    #[test]
    fn the_valid_bit_is_bit_31_and_is_not_part_of_the_encoding() {
        assert_eq!(EVENT_VALID, 0x8000_0000);
        assert_eq!(
            exception_event(VECTOR_GENERAL_PROTECTION, true) & EVENT_VALID,
            0
        );
        assert_eq!(
            exception_event(VECTOR_GENERAL_PROTECTION, true) | EVENT_VALID,
            0x8000_0b0d
        );
    }

    #[test]
    fn exception_vector_occupies_the_low_byte_only() {
        let event = exception_event(0xff, false);
        assert_eq!(event & 0xff, 0xff);
        assert_eq!(event >> EVENT_TYPE_SHIFT, EVENT_TYPE_HARDWARE_EXCEPTION);
    }

    #[test]
    fn fpu_state_is_an_aligned_fxsave_image() {
        assert_eq!(size_of::<FpuState>(), 512);
        assert_eq!(align_of::<FpuState>(), 16);
    }

    #[test]
    fn fpu_state_resets_to_finit_and_mxcsr_defaults() {
        let state = FpuState::reset();
        // FCW at offset 0, MXCSR at offset 24 of the FXSAVE area.
        assert_eq!(u16::from_le_bytes([state.0[0], state.0[1]]), 0x037f);
        assert_eq!(
            u32::from_le_bytes([state.0[24], state.0[25], state.0[26], state.0[27]]),
            0x1f80
        );
    }

    #[test]
    fn fpu_state_reset_leaves_every_other_byte_zero() {
        let state = FpuState::reset();
        let nonzero: alloc::vec::Vec<usize> = state
            .0
            .iter()
            .enumerate()
            .filter(|(_, b)| **b != 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(nonzero, [0, 1, 24, 25]);
    }
}
