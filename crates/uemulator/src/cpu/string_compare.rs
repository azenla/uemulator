//! The comparison core of pcmpistri/pcmpistrm/pcmpestri/pcmpestrm (SSE4.2, SDM vol. 2 "Intel SSE4
//! string and text processing instructions"). Pure: lengths come in already resolved, so the
//! implicit-length (stop at the first zero element) and explicit-length (EAX/EDX) forms share it.
//!
//! The imm8 control byte: bit 0 selects words over bytes, bit 1 signed over unsigned elements,
//! bits 3:2 the aggregation (equal any, ranges, equal each, equal ordered), bits 5:4 the polarity
//! of the result, and bit 6 the output form (least/most significant index, or bit/element mask).

const CONTROL_WORDS: u8 = 1 << 0;
const CONTROL_SIGNED: u8 = 1 << 1;
const CONTROL_AGGREGATION_SHIFT: u32 = 2;
const CONTROL_AGGREGATION_MASK: u8 = 3;
const CONTROL_POLARITY_SHIFT: u32 = 4;
const CONTROL_POLARITY_MASK: u8 = 3;
pub(super) const CONTROL_MOST_SIGNIFICANT: u8 = 1 << 6;

#[derive(Clone, Copy, PartialEq)]
enum Aggregation {
    EqualAny,
    Ranges,
    EqualEach,
    EqualOrdered,
}

/// Polarity values 1 (negative) and 3 (masked negative) invert the result; the masked forms only
/// invert the positions that lie inside the second string.
const POLARITY_NEGATIVE: u8 = 1;
const POLARITY_MASKED_NEGATIVE: u8 = 3;

/// IntRes2 and the four flags the instructions set (AF and PF are cleared by the caller).
pub(super) struct Comparison {
    /// One bit per element of the second string (bit `j` for element `j`).
    pub result: u32,
    pub carry: bool,
    pub zero: bool,
    pub sign: bool,
    pub overflow: bool,
}

/// Number of elements in a register for this control byte: 16 bytes or 8 words.
pub(super) fn element_count(control: u8) -> usize {
    if control & CONTROL_WORDS != 0 { 8 } else { 16 }
}

fn element(register: u128, index: usize, control: u8) -> i32 {
    let words = control & CONTROL_WORDS != 0;
    let signed = control & CONTROL_SIGNED != 0;
    match (words, signed) {
        (false, false) => (register >> (8 * index)) as u8 as i32,
        (false, true) => (register >> (8 * index)) as u8 as i8 as i32,
        (true, false) => (register >> (16 * index)) as u16 as i32,
        (true, true) => (register >> (16 * index)) as u16 as i16 as i32,
    }
}

/// Length of the string in `register` for the implicit-length forms: the index of the first zero
/// element, or the full element count.
pub(super) fn implicit_length(register: u128, control: u8) -> usize {
    let count = element_count(control);
    (0..count)
        .find(|&index| element(register, index, control) == 0)
        .unwrap_or(count)
}

/// Compares string `first` (the ModRM.reg operand, `first_length` elements) with `second` (the
/// r/m operand). Both lengths are already clamped to the element count.
pub(super) fn compare(
    control: u8,
    first: u128,
    first_length: usize,
    second: u128,
    second_length: usize,
) -> Comparison {
    let count = element_count(control);
    let aggregation = match (control >> CONTROL_AGGREGATION_SHIFT) & CONTROL_AGGREGATION_MASK {
        0 => Aggregation::EqualAny,
        1 => Aggregation::Ranges,
        2 => Aggregation::EqualEach,
        _ => Aggregation::EqualOrdered,
    };
    // BoolRes[j][i]: element `second[j]` against `first[i]`. SDM table 4-4 fixes the value for
    // elements past the end of either string, which is what makes "equal each" compare lengths
    // and "equal ordered" accept a needle that runs off the end of the haystack.
    let matches = |j: usize, i: usize| -> bool {
        match (i < first_length, j < second_length) {
            (true, true) => {
                let (a, b) = (element(first, i, control), element(second, j, control));
                if aggregation == Aggregation::Ranges {
                    // Even entries of the first string are lower bounds, odd ones upper bounds.
                    if i.is_multiple_of(2) { b >= a } else { b <= a }
                } else {
                    a == b
                }
            }
            (false, false) => matches!(
                aggregation,
                Aggregation::EqualEach | Aggregation::EqualOrdered
            ),
            (false, true) => aggregation == Aggregation::EqualOrdered,
            (true, false) => false,
        }
    };
    let mut result = 0u32;
    for j in 0..count {
        let bit = match aggregation {
            Aggregation::EqualAny => (0..count).any(|i| matches(j, i)),
            Aggregation::Ranges => (0..count)
                .step_by(2)
                .any(|i| matches(j, i) && matches(j, i + 1)),
            Aggregation::EqualEach => matches(j, j),
            Aggregation::EqualOrdered => (0..count - j).all(|i| matches(j + i, i)),
        };
        result |= (bit as u32) << j;
    }
    let all_elements = (1u32 << count) - 1;
    result = match (control >> CONTROL_POLARITY_SHIFT) & CONTROL_POLARITY_MASK {
        POLARITY_NEGATIVE => result ^ all_elements,
        POLARITY_MASKED_NEGATIVE => result ^ (((1u32 << second_length) - 1) & all_elements),
        _ => result,
    };
    Comparison {
        result,
        carry: result != 0,
        zero: second_length < count,
        sign: first_length < count,
        overflow: result & 1 != 0,
    }
}

/// The index pcmpistri/pcmpestri put in ECX: of the lowest or highest set bit of the result, or
/// the element count when no bit is set.
pub(super) fn result_index(control: u8, result: u32) -> u32 {
    if result == 0 {
        element_count(control) as u32
    } else if control & CONTROL_MOST_SIGNIFICANT != 0 {
        u32::BITS - 1 - result.leading_zeros()
    } else {
        result.trailing_zeros()
    }
}

/// The value pcmpistrm/pcmpestrm put in XMM0: the result bits themselves, or each set bit
/// expanded to an all-ones element.
pub(super) fn result_mask(control: u8, result: u32) -> u128 {
    if control & CONTROL_MOST_SIGNIFICANT == 0 {
        return result as u128;
    }
    let element_bits = if control & CONTROL_WORDS != 0 { 16 } else { 8 };
    let mut mask = 0u128;
    for j in 0..element_count(control) {
        if (result >> j) & 1 != 0 {
            mask |= ((1u128 << element_bits) - 1) << (j * element_bits);
        }
    }
    mask
}
