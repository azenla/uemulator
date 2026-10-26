use super::arch::cpuid;

const ECX1: u32 =
    1 << 0 | 1 << 1 | 1 << 9 | 1 << 13 | 1 << 19 | 1 << 20 | 1 << 22 | 1 << 23 | 1 << 25;
// FPU VME DE PSE TSC MSR PAE CX8 SEP PGE CMOV PAT CLFSH MMX FXSR SSE SSE2
const EDX1: u32 = 1 << 0
    | 1 << 1
    | 1 << 2
    | 1 << 3
    | 1 << 4
    | 1 << 5
    | 1 << 6
    | 1 << 8
    | 1 << 11
    | 1 << 13
    | 1 << 15
    | 1 << 16
    | 1 << 19
    | 1 << 23
    | 1 << 24
    | 1 << 25
    | 1 << 26;
const EDX_EXT1: u32 = 1 << 11 | 1 << 20 | 1 << 26 | 1 << 29;

/// What the guest sees: the host's identity with the features the guest can actually use. There is
/// no APIC, MTRR, XSAVE or machine-check hardware, and no virtualization, as none of it is
/// emulated.
pub fn guest(leaf: u32, sub: u32) -> [u32; 4] {
    let h = cpuid(leaf, sub);
    match leaf {
        0 => [7, h[1], h[2], h[3]],
        1 => [h[0], (h[1] & 0xffff) | (1 << 16), h[2] & ECX1, h[3] & EDX1],
        0x8000_0000 => [0x8000_0008, h[1], h[2], h[3]],
        0x8000_0001 => [h[0], 0, h[2] & 1, h[3] & EDX_EXT1],
        0x8000_0002..=0x8000_0004 => h,
        0x8000_0007 => [0, 0, 0, 1 << 8],
        0x8000_0008 => [h[0] & 0xffff, 0, 0, 0],
        _ => [0; 4],
    }
}
