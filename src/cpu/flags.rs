pub(crate) const CF: u64 = 1;
pub(crate) const PF: u64 = 1 << 2;
pub(crate) const AF: u64 = 1 << 4;
pub(crate) const ZF: u64 = 1 << 6;
pub(crate) const SF: u64 = 1 << 7;
pub(crate) const TF: u64 = 1 << 8;
pub(crate) const IF: u64 = 1 << 9;
pub(crate) const DF: u64 = 1 << 10;
pub(crate) const OF: u64 = 1 << 11;
pub(crate) const IOPL: u64 = 3 << 12;
pub(crate) const NT: u64 = 1 << 14;
pub(crate) const RF: u64 = 1 << 16;
pub(crate) const VM: u64 = 1 << 17;
pub(crate) const AC: u64 = 1 << 18;
pub(crate) const ARITH: u64 = CF | PF | AF | ZF | SF | OF;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub(crate) enum Cc {
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

pub(crate) fn mask(size: u8) -> u64 {
    match size {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    }
}

pub(crate) fn sign(size: u8) -> u64 {
    1u64 << (size as u32 * 8 - 1)
}

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

pub(crate) fn compute(op: Cc, size: u8, dst: u64, src: u64, src2: u64, eflags: u64) -> u64 {
    let m = mask(size);
    let res = dst & m;
    let b = src & m;
    let s = sign(size);
    match op {
        Cc::Eflags => eflags & ARITH,
        Cc::Add | Cc::Adc => {
            let cin = if op == Cc::Adc { src2 & 1 } else { 0 };
            let a = res.wrapping_sub(b).wrapping_sub(cin) & m;
            let cf = if cin != 0 { res <= b } else { res < b };
            let mut f = szp(res, size);
            f |= cf as u64;
            f |= (a ^ b ^ res) & AF;
            if (!(a ^ b) & (a ^ res)) & s != 0 {
                f |= OF;
            }
            f
        }
        Cc::Sub | Cc::Sbb => {
            let cin = if op == Cc::Sbb { src2 & 1 } else { 0 };
            let a = res.wrapping_add(b).wrapping_add(cin) & m;
            let cf = if cin != 0 { a <= b } else { a < b };
            let mut f = szp(res, size);
            f |= cf as u64;
            f |= (a ^ b ^ res) & AF;
            if ((a ^ b) & (a ^ res)) & s != 0 {
                f |= OF;
            }
            f
        }
        Cc::Logic => szp(res, size),
        Cc::Inc => {
            let mut f = szp(res, size) | (src & 1);
            if res & 0xf == 0 {
                f |= AF;
            }
            if res == s {
                f |= OF;
            }
            f
        }
        Cc::Dec => {
            let mut f = szp(res, size) | (src & 1);
            if res & 0xf == 0xf {
                f |= AF;
            }
            if res == s.wrapping_sub(1) & m {
                f |= OF;
            }
            f
        }
    }
}

pub(crate) fn szp_flags(res: u64, size: u8) -> u64 {
    szp(res & mask(size), size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_flags() {
        // 0 - 1 (8-bit): borrow
        let f = compute(Cc::Sub, 1, 0xff, 1, 0, 0);
        assert_ne!(f & (CF | SF), 0);
        // 5 - 1: no borrow
        let f = compute(Cc::Sub, 4, 4, 1, 0, 0);
        assert_eq!(f & (CF | ZF | OF), 0);
        // 0x80 - 1 (8-bit): signed overflow
        let f = compute(Cc::Sub, 1, 0x7f, 1, 0, 0);
        assert_ne!(f & OF, 0);
        // 1 - 1
        let f = compute(Cc::Sub, 4, 0, 1, 0, 0);
        assert_eq!(f & (CF | ZF), ZF);
    }

    #[test]
    fn add_flags() {
        let f = compute(Cc::Add, 1, 0, 1, 0, 0);
        assert_eq!(f & (CF | ZF), CF | ZF);
        let f = compute(Cc::Add, 1, 0x80, 0x7f, 0, 0);
        assert_ne!(f & OF, 0);
        let f = compute(Cc::Adc, 1, 0, 0xff, 1, 0);
        assert_ne!(f & CF, 0);
    }
}
