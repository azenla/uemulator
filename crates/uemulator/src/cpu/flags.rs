pub(crate) const CF: u64 = 1;
pub(crate) const PF: u64 = 1 << 2;
pub(crate) const AF: u64 = 1 << 4;
pub(crate) const ZF: u64 = 1 << 6;
pub(crate) const SF: u64 = 1 << 7;
pub(crate) const TF: u64 = 1 << 8;
pub(crate) const IF: u64 = 1 << 9;
pub(crate) const DF: u64 = 1 << 10;
pub(crate) const OF: u64 = 1 << 11;
/// Two-bit field (bits 12-13), not a single flag.
pub(crate) const IOPL: u64 = 3 << 12;
pub(crate) const NT: u64 = 1 << 14;
pub(crate) const RF: u64 = 1 << 16;
pub(crate) const VM: u64 = 1 << 17;
pub(crate) const AC: u64 = 1 << 18;
/// Bit 1 reads as 1 in every EFLAGS value.
pub(crate) const RESERVED_ONE: u64 = 1 << 1;
/// The six status flags that `LazyFlagsOp` can defer; everything else in EFLAGS is kept eagerly.
pub(crate) const ARITH: u64 = CF | PF | AF | ZF | SF | OF;

/// How to derive the `ARITH` flags from `Cpu::lazy_dst/lazy_src/lazy_src2` when they are needed.
///
/// `lazy_dst` holds the operation's *result* (not the old destination operand) and `lazy_src`
/// the second operand. `Adc`/`Sbb` take the carry-in from bit 0 of `lazy_src2`; `Inc`/`Dec`
/// keep the untouched CF in bit 0 of `lazy_src`. `Eflags` means the flags are already
/// materialized in `Cpu::eflags`.
///
/// The JIT writes the discriminant with a 16-bit store together with `Cpu::lazy_size`, so it
/// must stay `repr(u8)`, and the variants it emits (Add, Sub, Logic) must not be renumbered
/// without checking `jit/emit*.rs`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub(crate) enum LazyFlagsOp {
    #[default]
    Eflags,
    Add,
    Adc,
    Sub,
    Sbb,
    Logic,
    Inc,
    Dec,
}

/// Value mask for an operand `size` in bytes; anything other than 1, 2 or 4 means 8.
pub(crate) fn mask(size: u8) -> u64 {
    match size {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    }
}

/// Sign bit of an operand `size` bytes wide.
pub(crate) fn sign(size: u8) -> u64 {
    1u64 << (size as u32 * 8 - 1)
}

/// `res` must already be truncated to `size`. PF looks at the low byte only, whatever the width.
fn szp(res: u64, size: u8) -> u64 {
    let mut f = 0;
    if res == 0 {
        f |= ZF;
    }
    if res & sign(size) != 0 {
        f |= SF;
    }
    if (res as u8).count_ones() & 1 == 0 {
        f |= PF;
    }
    f
}

/// AF is the carry out of bit 3, so INC/DEC derive it from the low nibble of the result.
const NIBBLE_MASK: u64 = 0xf;

/// Materializes the `ARITH` flags for a deferred operation; the single source of truth that
/// the interpreter and the JIT (via helper calls) share. `eflags` is only read for `Eflags`.
pub(crate) fn compute(
    op: LazyFlagsOp,
    size: u8,
    dst: u64,
    src: u64,
    src2: u64,
    eflags: u64,
) -> u64 {
    let width_mask = mask(size);
    let result = dst & width_mask;
    let b = src & width_mask;
    let sign_bit = sign(size);
    match op {
        LazyFlagsOp::Eflags => eflags & ARITH,
        LazyFlagsOp::Add | LazyFlagsOp::Adc => {
            let carry_in = if op == LazyFlagsOp::Adc { src2 & 1 } else { 0 };
            // first operand, recovered from the result
            let a = result.wrapping_sub(b).wrapping_sub(carry_in) & width_mask;
            // with a carry-in, a + b + 1 wraps to exactly `b` when a == !0, hence <= and not <
            let carry = if carry_in != 0 {
                result <= b
            } else {
                result < b
            };
            let mut f = szp(result, size);
            f |= carry as u64;
            f |= (a ^ b ^ result) & AF;
            if (!(a ^ b) & (a ^ result)) & sign_bit != 0 {
                f |= OF;
            }
            f
        }
        LazyFlagsOp::Sub | LazyFlagsOp::Sbb => {
            let carry_in = if op == LazyFlagsOp::Sbb { src2 & 1 } else { 0 };
            let a = result.wrapping_add(b).wrapping_add(carry_in) & width_mask;
            let borrow = if carry_in != 0 { a <= b } else { a < b };
            let mut f = szp(result, size);
            f |= borrow as u64;
            f |= (a ^ b ^ result) & AF;
            if ((a ^ b) & (a ^ result)) & sign_bit != 0 {
                f |= OF;
            }
            f
        }
        // CF and OF cleared; AF is architecturally undefined and left clear.
        LazyFlagsOp::Logic => szp(result, size),
        // INC/DEC leave CF alone: it rides along in bit 0 of `src`.
        LazyFlagsOp::Inc => {
            let mut f = szp(result, size) | (src & 1);
            if result & NIBBLE_MASK == 0 {
                f |= AF;
            }
            if result == sign_bit {
                f |= OF;
            }
            f
        }
        LazyFlagsOp::Dec => {
            let mut f = szp(result, size) | (src & 1);
            if result & NIBBLE_MASK == NIBBLE_MASK {
                f |= AF;
            }
            if result == sign_bit.wrapping_sub(1) & width_mask {
                f |= OF;
            }
            f
        }
    }
}

/// SF/ZF/PF of `res` truncated to `size`, for instructions that set only those.
pub(crate) fn szp_flags(res: u64, size: u8) -> u64 {
    szp(res & mask(size), size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_flags() {
        // 0 - 1 (8-bit): borrow
        let f = compute(LazyFlagsOp::Sub, 1, 0xff, 1, 0, 0);
        assert_ne!(f & (CF | SF), 0);
        // 5 - 1: no borrow
        let f = compute(LazyFlagsOp::Sub, 4, 4, 1, 0, 0);
        assert_eq!(f & (CF | ZF | OF), 0);
        // 0x80 - 1 (8-bit): signed overflow
        let f = compute(LazyFlagsOp::Sub, 1, 0x7f, 1, 0, 0);
        assert_ne!(f & OF, 0);
        // 1 - 1
        let f = compute(LazyFlagsOp::Sub, 4, 0, 1, 0, 0);
        assert_eq!(f & (CF | ZF), ZF);
    }

    #[test]
    fn add_flags() {
        let f = compute(LazyFlagsOp::Add, 1, 0, 1, 0, 0);
        assert_eq!(f & (CF | ZF), CF | ZF);
        let f = compute(LazyFlagsOp::Add, 1, 0x80, 0x7f, 0, 0);
        assert_ne!(f & OF, 0);
        let f = compute(LazyFlagsOp::Adc, 1, 0, 0xff, 1, 0);
        assert_ne!(f & CF, 0);
    }
}
