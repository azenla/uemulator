use super::arch::{
    CPUID_LEAF_BRAND_STRING_FIRST, CPUID_LEAF_BRAND_STRING_LAST, CPUID_LEAF_EXT_ADDRESS_SIZES,
    CPUID_LEAF_EXT_FEATURES, CPUID_LEAF_EXT_MAX, CPUID_LEAF_EXT_POWER_MANAGEMENT,
    CPUID_LEAF_FEATURES, CPUID_LEAF_VENDOR,
};

/// Highest basic leaf the guest is told about; the ones above 1 read as zero.
const GUEST_MAX_BASIC_LEAF: u32 = 7;
/// Highest extended leaf the guest is told about.
const GUEST_MAX_EXT_LEAF: u32 = 0x8000_0008;

// CPUID.1:ECX features passed through.
const ECX1_SSE3: u32 = 1 << 0;
const ECX1_PCLMULQDQ: u32 = 1 << 1;
const ECX1_SSSE3: u32 = 1 << 9;
const ECX1_CMPXCHG16B: u32 = 1 << 13;
const ECX1_SSE4_1: u32 = 1 << 19;
const ECX1_SSE4_2: u32 = 1 << 20;
const ECX1_MOVBE: u32 = 1 << 22;
const ECX1_POPCNT: u32 = 1 << 23;
const ECX1_AES: u32 = 1 << 25;
const ECX1: u32 = ECX1_SSE3
    | ECX1_PCLMULQDQ
    | ECX1_SSSE3
    | ECX1_CMPXCHG16B
    | ECX1_SSE4_1
    | ECX1_SSE4_2
    | ECX1_MOVBE
    | ECX1_POPCNT
    | ECX1_AES;

// CPUID.1:EDX features passed through. No APIC, MTRR, MCE or MCA bits, as none are emulated.
const EDX1_FPU: u32 = 1 << 0;
const EDX1_VME: u32 = 1 << 1;
const EDX1_DE: u32 = 1 << 2;
const EDX1_PSE: u32 = 1 << 3;
const EDX1_TSC: u32 = 1 << 4;
const EDX1_MSR: u32 = 1 << 5;
const EDX1_PAE: u32 = 1 << 6;
const EDX1_CX8: u32 = 1 << 8;
const EDX1_SEP: u32 = 1 << 11;
const EDX1_PGE: u32 = 1 << 13;
const EDX1_CMOV: u32 = 1 << 15;
const EDX1_PAT: u32 = 1 << 16;
const EDX1_CLFSH: u32 = 1 << 19;
const EDX1_MMX: u32 = 1 << 23;
const EDX1_FXSR: u32 = 1 << 24;
const EDX1_SSE: u32 = 1 << 25;
const EDX1_SSE2: u32 = 1 << 26;
const EDX1: u32 = EDX1_FPU
    | EDX1_VME
    | EDX1_DE
    | EDX1_PSE
    | EDX1_TSC
    | EDX1_MSR
    | EDX1_PAE
    | EDX1_CX8
    | EDX1_SEP
    | EDX1_PGE
    | EDX1_CMOV
    | EDX1_PAT
    | EDX1_CLFSH
    | EDX1_MMX
    | EDX1_FXSR
    | EDX1_SSE
    | EDX1_SSE2;

/// CPUID.1:EBX keeps the brand index and CLFLUSH line size (bits 15:0); the initial APIC ID
/// (31:24) is zeroed, and the logical processor count (23:16) is set to 1.
const EBX1_KEPT_MASK: u32 = 0xffff;
const EBX1_LOGICAL_PROCESSORS_SHIFT: u32 = 16;

// CPUID.80000001h features passed through.
const EXT1_ECX_LAHF_SAHF: u32 = 1 << 0;
const EXT1_EDX_SYSCALL: u32 = 1 << 11;
const EXT1_EDX_NX: u32 = 1 << 20;
const EXT1_EDX_PAGE_1GB: u32 = 1 << 26;
const EXT1_EDX_LONG_MODE: u32 = 1 << 29;
const EXT1_EDX: u32 = EXT1_EDX_SYSCALL | EXT1_EDX_NX | EXT1_EDX_PAGE_1GB | EXT1_EDX_LONG_MODE;

/// CPUID.80000007h:EDX bit 8. The guest is also given the TSC frequency on its command line.
const EXT7_EDX_INVARIANT_TSC: u32 = 1 << 8;

/// CPUID.80000008h:EAX low 16 bits: physical (7:0) and linear (15:8) address sizes.
const EXT8_EAX_ADDRESS_SIZES_MASK: u32 = 0xffff;

/// What the guest sees: the host's identity with the features the guest can actually use. There is
/// no APIC, MTRR, XSAVE or machine-check hardware, and no virtualization, as none of it is
/// emulated.
#[cfg(target_os = "uefi")]
pub fn guest(leaf: u32, sub: u32) -> [u32; 4] {
    guest_from(leaf, super::arch::cpuid(leaf, sub))
}

/// `guest` for the host's answer `h` (eax, ebx, ecx, edx) to the same query.
fn guest_from(leaf: u32, h: [u32; 4]) -> [u32; 4] {
    match leaf {
        CPUID_LEAF_VENDOR => [GUEST_MAX_BASIC_LEAF, h[1], h[2], h[3]],
        CPUID_LEAF_FEATURES => [
            h[0],
            (h[1] & EBX1_KEPT_MASK) | (1 << EBX1_LOGICAL_PROCESSORS_SHIFT),
            h[2] & ECX1,
            h[3] & EDX1,
        ],
        CPUID_LEAF_EXT_MAX => [GUEST_MAX_EXT_LEAF, h[1], h[2], h[3]],
        CPUID_LEAF_EXT_FEATURES => [h[0], 0, h[2] & EXT1_ECX_LAHF_SAHF, h[3] & EXT1_EDX],
        // Processor brand string.
        CPUID_LEAF_BRAND_STRING_FIRST..=CPUID_LEAF_BRAND_STRING_LAST => h,
        CPUID_LEAF_EXT_POWER_MANAGEMENT => [0, 0, 0, EXT7_EDX_INVARIANT_TSC],
        CPUID_LEAF_EXT_ADDRESS_SIZES => [h[0] & EXT8_EAX_ADDRESS_SIZES_MASK, 0, 0, 0],
        _ => [0; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::super::arch::CPUID_LEAF_SVM;
    use super::*;

    const ALL_ONES: [u32; 4] = [u32::MAX; 4];

    fn bits(positions: &[u32]) -> u32 {
        positions.iter().fold(0, |acc, p| acc | 1 << p)
    }

    #[test]
    fn vendor_leaf_caps_the_basic_leaves_at_seven_and_keeps_the_vendor_string() {
        let host = [0x20, 0x756e_6547, 0x6c65_746e, 0x4965_6e69];
        assert_eq!(
            guest_from(CPUID_LEAF_VENDOR, host),
            [7, 0x756e_6547, 0x6c65_746e, 0x4965_6e69]
        );
    }

    #[test]
    fn feature_leaf_keeps_the_signature() {
        let host = [0x000a_0655, 0, 0, 0];
        assert_eq!(guest_from(CPUID_LEAF_FEATURES, host)[0], 0x000a_0655);
    }

    #[test]
    fn feature_ebx_keeps_brand_and_clflush_size_and_reports_one_logical_processor() {
        let host = [0, 0x0a0f_1234, 0, 0];
        let ebx = guest_from(CPUID_LEAF_FEATURES, host)[1];
        assert_eq!(ebx & 0xffff, 0x1234);
        assert_eq!((ebx >> 16) & 0xff, 1, "logical processor count");
        assert_eq!(ebx >> 24, 0, "initial APIC id");
    }

    #[test]
    fn feature_ebx_with_every_host_bit_set_still_has_a_zero_apic_id() {
        assert_eq!(guest_from(CPUID_LEAF_FEATURES, ALL_ONES)[1], 0x0001_ffff);
    }

    #[test]
    fn feature_ecx_is_exactly_the_passed_through_set() {
        // SSE3, PCLMULQDQ, SSSE3, CMPXCHG16B, SSE4.1, SSE4.2, MOVBE, POPCNT, AES.
        let expected = bits(&[0, 1, 9, 13, 19, 20, 22, 23, 25]);
        assert_eq!(guest_from(CPUID_LEAF_FEATURES, ALL_ONES)[2], expected);
    }

    #[test]
    fn feature_edx_is_exactly_the_passed_through_set() {
        // FPU, VME, DE, PSE, TSC, MSR, PAE, CX8, SEP, PGE, CMOV, PAT, CLFSH, MMX, FXSR, SSE, SSE2.
        let expected = bits(&[0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 15, 16, 19, 23, 24, 25, 26]);
        assert_eq!(guest_from(CPUID_LEAF_FEATURES, ALL_ONES)[3], expected);
    }

    #[test]
    fn feature_ecx_hides_what_the_guest_cannot_use() {
        let ecx = guest_from(CPUID_LEAF_FEATURES, ALL_ONES)[2];
        let hidden = [
            (3, "MONITOR"),
            (5, "VMX"),
            (6, "SMX"),
            (7, "EIST"),
            (12, "FMA"),
            (17, "PCID"),
            (21, "x2APIC"),
            (24, "TSC-deadline"),
            (26, "XSAVE"),
            (27, "OSXSAVE"),
            (28, "AVX"),
            (29, "F16C"),
            (30, "RDRAND"),
            (31, "hypervisor"),
        ];
        for (bit, name) in hidden {
            assert_eq!(ecx & (1 << bit), 0, "{name} must not be advertised");
        }
    }

    #[test]
    fn feature_edx_hides_the_hardware_that_is_not_emulated() {
        let edx = guest_from(CPUID_LEAF_FEATURES, ALL_ONES)[3];
        let hidden = [
            (7, "MCE"),
            (9, "APIC"),
            (12, "MTRR"),
            (14, "MCA"),
            (17, "PSE-36"),
            (18, "PSN"),
            (21, "DS"),
            (22, "ACPI"),
            (27, "SS"),
            (28, "HTT"),
            (29, "TM"),
            (31, "PBE"),
        ];
        for (bit, name) in hidden {
            assert_eq!(edx & (1 << bit), 0, "{name} must not be advertised");
        }
    }

    #[test]
    fn features_the_host_lacks_stay_absent() {
        let r = guest_from(CPUID_LEAF_FEATURES, [0; 4]);
        assert_eq!(r[2], 0);
        assert_eq!(r[3], 0);
        let r = guest_from(CPUID_LEAF_EXT_FEATURES, [0; 4]);
        assert_eq!((r[2], r[3]), (0, 0));
    }

    #[test]
    fn each_passed_through_feature_follows_the_host() {
        let sse2 = 1 << 26;
        assert_eq!(guest_from(CPUID_LEAF_FEATURES, [0, 0, 0, sse2])[3], sse2);
        let popcnt = 1 << 23;
        assert_eq!(
            guest_from(CPUID_LEAF_FEATURES, [0, 0, popcnt, 0])[2],
            popcnt
        );
    }

    #[test]
    fn extended_max_leaf_is_capped_at_the_address_size_leaf() {
        let r = guest_from(CPUID_LEAF_EXT_MAX, [0x8000_0021, 1, 2, 3]);
        assert_eq!(r, [0x8000_0008, 1, 2, 3]);
    }

    #[test]
    fn extended_features_expose_long_mode_syscall_nx_and_gigabyte_pages_only() {
        let r = guest_from(CPUID_LEAF_EXT_FEATURES, ALL_ONES);
        assert_eq!(r[0], u32::MAX, "signature passes through");
        assert_eq!(r[1], 0);
        assert_eq!(r[2], bits(&[0]), "LAHF/SAHF");
        assert_eq!(r[3], bits(&[11, 20, 26, 29]), "SYSCALL, NX, 1GB pages, LM");
    }

    #[test]
    fn extended_features_hide_svm_and_rdtscp() {
        let r = guest_from(CPUID_LEAF_EXT_FEATURES, ALL_ONES);
        assert_eq!(r[2] & (1 << 2), 0, "SVM");
        assert_eq!(r[3] & (1 << 27), 0, "RDTSCP");
    }

    #[test]
    fn brand_string_leaves_pass_through_untouched() {
        let host = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        for leaf in CPUID_LEAF_BRAND_STRING_FIRST..=CPUID_LEAF_BRAND_STRING_LAST {
            assert_eq!(guest_from(leaf, host), host, "leaf {leaf:#x}");
        }
    }

    #[test]
    fn invariant_tsc_is_the_only_thing_the_power_management_leaf_reports() {
        assert_eq!(
            guest_from(CPUID_LEAF_EXT_POWER_MANAGEMENT, ALL_ONES),
            [0, 0, 0, 1 << 8]
        );
    }

    #[test]
    fn address_size_leaf_keeps_only_the_two_size_fields() {
        let r = guest_from(CPUID_LEAF_EXT_ADDRESS_SIZES, [0x00ab_3930, 7, 8, 9]);
        assert_eq!(r, [0x3930, 0, 0, 0]);
    }

    #[test]
    fn svm_leaf_reads_as_zero_so_the_guest_never_sees_nested_paging_or_nrips() {
        assert_eq!(guest_from(CPUID_LEAF_SVM, ALL_ONES), [0; 4]);
    }

    #[test]
    fn leaves_without_an_answer_read_as_zero() {
        for leaf in [
            2,
            3,
            4,
            5,
            6,
            7,
            0xb,
            0xd,
            0x1f,
            0x4000_0000,
            0x4000_0001,
            0x8000_0005,
            0x8000_0006,
            0x8000_0009,
            0x8000_001d,
            0x8000_0021,
            u32::MAX,
        ] {
            assert_eq!(guest_from(leaf, ALL_ONES), [0; 4], "leaf {leaf:#x}");
        }
    }
}
