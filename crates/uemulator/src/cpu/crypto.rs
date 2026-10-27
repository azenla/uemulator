//! Pure helpers behind the SSE4.2 CRC32 instruction, AES-NI and PCLMULQDQ.

/// The CRC-32C (Castagnoli) polynomial in reflected bit order, which is what the CRC32
/// instruction implements. Unlike zlib's CRC-32 there is no initial or final inversion: those are
/// the caller's business (the kernel's crc32c does them in software around the instruction).
const CRC32C_POLYNOMIAL_REFLECTED: u32 = 0x82f6_3b78;

const fn crc32c_byte_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut crc = byte as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32C_POLYNOMIAL_REFLECTED
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[byte] = crc;
        byte += 1;
    }
    table
}

static CRC32C_TABLE: [u32; 256] = crc32c_byte_table();

/// Accumulates the low `bytes` bytes of `data` (least significant byte first) into `crc`.
pub(super) fn crc32c(mut crc: u32, data: u64, bytes: u8) -> u32 {
    for index in 0..bytes {
        let byte = (data >> (8 * index)) as u8;
        crc = (crc >> 8) ^ CRC32C_TABLE[((crc as u8) ^ byte) as usize];
    }
    crc
}
