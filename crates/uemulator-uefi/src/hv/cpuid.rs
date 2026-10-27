use super::arch::{
    CPUID_LEAF_BRAND_STRING_FIRST, CPUID_LEAF_BRAND_STRING_LAST, CPUID_LEAF_EXT_ADDRESS_SIZES,
    CPUID_LEAF_EXT_FEATURES, CPUID_LEAF_EXT_MAX, CPUID_LEAF_EXT_POWER_MANAGEMENT,
    CPUID_LEAF_FEATURES, CPUID_LEAF_VENDOR, cpuid,
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
pub fn guest(leaf: u32, sub: u32) -> [u32; 4] {
    let h = cpuid(leaf, sub);
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
