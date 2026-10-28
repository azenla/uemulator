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

/// Multiplication in GF(2^8) with the AES reduction polynomial x^8 + x^4 + x^3 + x + 1.
pub(in crate::cpu::sse) const fn gf_multiply(mut a: u8, mut b: u8) -> u8 {
    const REDUCTION: u8 = 0x1b;
    let mut product = 0;
    while b != 0 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= REDUCTION;
        }
        b >>= 1;
    }
    product
}

/// The AES S-box, derived from its definition (multiplicative inverse in GF(2^8), then the affine
/// transform with constant 0x63) rather than typed in, so a typo cannot hide in a table. The
/// FIPS-197 test vectors in the tests are what pins it.
const fn aes_sbox_table() -> [u8; 256] {
    const AFFINE_CONSTANT: u8 = 0x63;
    let mut table = [0u8; 256];
    let mut value = 0;
    while value < 256 {
        let x = value as u8;
        // x^254 is the inverse of x for nonzero x, and gives 0 for x = 0, which the S-box wants.
        let mut inverse = 1u8;
        let mut base = x;
        let mut exponent = 254;
        while exponent > 0 {
            if exponent & 1 != 0 {
                inverse = gf_multiply(inverse, base);
            }
            base = gf_multiply(base, base);
            exponent >>= 1;
        }
        table[value] = inverse
            ^ inverse.rotate_left(1)
            ^ inverse.rotate_left(2)
            ^ inverse.rotate_left(3)
            ^ inverse.rotate_left(4)
            ^ AFFINE_CONSTANT;
        value += 1;
    }
    table
}

const fn aes_inverse_sbox_table(sbox: &[u8; 256]) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut value = 0;
    while value < 256 {
        table[sbox[value] as usize] = value as u8;
        value += 1;
    }
    table
}

static AES_SBOX: [u8; 256] = aes_sbox_table();
static AES_INVERSE_SBOX: [u8; 256] = aes_inverse_sbox_table(&AES_SBOX);

/// AES state as a vector register: byte `row + 4 * column` of the little-endian value is the
/// state matrix entry in that row and column, which is how the instructions lay out the block.
const AES_COLUMNS: usize = 4;
const AES_ROWS: usize = 4;

fn sub_bytes(state: [u8; 16], table: &[u8; 256]) -> [u8; 16] {
    state.map(|byte| table[byte as usize])
}

/// ShiftRows rotates row `r` left by `r` columns; the inverse rotates right.
fn shift_rows(state: [u8; 16], inverse: bool) -> [u8; 16] {
    let mut result = [0u8; 16];
    for column in 0..AES_COLUMNS {
        for row in 0..AES_ROWS {
            let source_column = if inverse {
                (column + AES_COLUMNS - row) % AES_COLUMNS
            } else {
                (column + row) % AES_COLUMNS
            };
            result[row + AES_ROWS * column] = state[row + AES_ROWS * source_column];
        }
    }
    result
}

/// MixColumns, or InvMixColumns: each column is multiplied by the circulant matrix whose first
/// row is (2 3 1 1), or (14 11 13 9) for the inverse.
fn mix_columns(state: [u8; 16], inverse: bool) -> [u8; 16] {
    const FORWARD: [u8; 4] = [2, 3, 1, 1];
    const INVERSE: [u8; 4] = [14, 11, 13, 9];
    let coefficients = if inverse { INVERSE } else { FORWARD };
    let mut result = [0u8; 16];
    for column in 0..AES_COLUMNS {
        for row in 0..AES_ROWS {
            let mut sum = 0;
            for k in 0..AES_ROWS {
                // Entry (row, k) of the circulant matrix is the first row rotated right by `row`.
                let coefficient = coefficients[(k + AES_ROWS - row) % AES_ROWS];
                sum ^= gf_multiply(coefficient, state[k + AES_ROWS * column]);
            }
            result[row + AES_ROWS * column] = sum;
        }
    }
    result
}

fn xor_blocks(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    core::array::from_fn(|k| a[k] ^ b[k])
}

/// aesenc / aesenclast: one encryption round and the AddRoundKey with `round_key`; the last
/// round has no MixColumns.
pub(super) fn aes_encrypt_round(state: u128, round_key: u128, last: bool) -> u128 {
    let mut block = shift_rows(sub_bytes(state.to_le_bytes(), &AES_SBOX), false);
    if !last {
        block = mix_columns(block, false);
    }
    u128::from_le_bytes(xor_blocks(block, round_key.to_le_bytes()))
}

/// aesdec / aesdeclast: one round of the equivalent inverse cipher (InvShiftRows, InvSubBytes,
/// then InvMixColumns unless last, then AddRoundKey).
pub(super) fn aes_decrypt_round(state: u128, round_key: u128, last: bool) -> u128 {
    let mut block = sub_bytes(shift_rows(state.to_le_bytes(), true), &AES_INVERSE_SBOX);
    if !last {
        block = mix_columns(block, true);
    }
    u128::from_le_bytes(xor_blocks(block, round_key.to_le_bytes()))
}

/// aesimc: InvMixColumns of a round key, to prepare it for the equivalent inverse cipher.
pub(super) fn aes_inverse_mix_columns(state: u128) -> u128 {
    u128::from_le_bytes(mix_columns(state.to_le_bytes(), true))
}

/// aeskeygenassist: SubWord of dwords 1 and 3 of the source, the odd results also rotated by a
/// byte and XORed with the round constant.
pub(super) fn aes_key_generation_assist(source: u128, round_constant: u8) -> u128 {
    let substitute =
        |word: u32| u32::from_le_bytes(word.to_le_bytes().map(|b| AES_SBOX[b as usize]));
    let word = |index: u32| (source >> (32 * index)) as u32;
    let (x1, x3) = (substitute(word(1)), substitute(word(3)));
    let rotated = |value: u32| value.rotate_right(8) ^ round_constant as u32;
    (x1 as u128)
        | ((rotated(x1) as u128) << 32)
        | ((x3 as u128) << 64)
        | ((rotated(x3) as u128) << 96)
}

/// pclmulqdq's multiplication: the 128-bit carry-less (GF(2)[x]) product of two 64-bit values.
pub(super) fn carry_less_multiply(a: u64, b: u64) -> u128 {
    let mut product = 0u128;
    for bit in 0..u64::BITS {
        if (b >> bit) & 1 != 0 {
            product ^= (a as u128) << bit;
        }
    }
    product
}

/// Multiplicative inverse in GF(2^8) (a^254); 0 has no inverse and maps to 0, as gf2p8affineinvqb
/// defines it.
pub(in crate::cpu::sse) fn gf_inverse(a: u8) -> u8 {
    /// The multiplicative group has order 255, so a^254 = a^-1.
    const INVERSE_EXPONENT: u32 = 254;
    if a == 0 {
        return 0;
    }
    let (mut result, mut power, mut exponent) = (1u8, a, INVERSE_EXPONENT);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = gf_multiply(result, power);
        }
        power = gf_multiply(power, power);
        exponent >>= 1;
    }
    result
}

/// One byte of gf2p8affineqb: bit `n` of the result is the parity of (matrix byte 7-n AND x) XOR
/// bit `n` of `immediate`, where `matrix` is the 8x8 bit matrix held in one quadword.
pub(in crate::cpu::sse) fn gf_affine_byte(matrix: u64, x: u8, immediate: u8) -> u8 {
    let mut result = 0u8;
    for bit in 0..8 {
        let row = (matrix >> (8 * (7 - bit))) as u8;
        let parity = (row & x).count_ones() as u8 & 1;
        result |= (parity ^ (immediate >> bit & 1)) << bit;
    }
    result
}

/// gf2p8mulb on one 128-bit lane: bytewise GF(2^8) products.
pub(super) fn gf_multiply_bytes(a: u128, b: u128) -> u128 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    u128::from_le_bytes(core::array::from_fn(|k| gf_multiply(a[k], b[k])))
}

/// gf2p8affineqb/gf2p8affineinvqb on one 128-bit lane: `data` bytes transformed (after inversion
/// for the `inverse` form) by the matrix in the corresponding quadword of `matrix`.
pub(super) fn gf_affine(data: u128, matrix: u128, immediate: u8, inverse: bool) -> u128 {
    let (data, matrices) = (data.to_le_bytes(), matrix.to_le_bytes());
    u128::from_le_bytes(core::array::from_fn(|k| {
        let quadword = u64::from_le_bytes(matrices[k / 8 * 8..k / 8 * 8 + 8].try_into().unwrap());
        let x = if inverse {
            gf_inverse(data[k])
        } else {
            data[k]
        };
        gf_affine_byte(quadword, x, immediate)
    }))
}
