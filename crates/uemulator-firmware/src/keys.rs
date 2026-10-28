//! Turns the serial port's byte stream (a terminal's keystrokes) into UEFI key events.

use crate::uefi::InputKey;

pub const SCAN_NONE: u16 = 0;
pub const SCAN_UP: u16 = 0x01;
pub const SCAN_DOWN: u16 = 0x02;
pub const SCAN_RIGHT: u16 = 0x03;
pub const SCAN_LEFT: u16 = 0x04;
pub const SCAN_HOME: u16 = 0x05;
pub const SCAN_END: u16 = 0x06;
pub const SCAN_INSERT: u16 = 0x07;
pub const SCAN_DELETE: u16 = 0x08;
pub const SCAN_PAGE_UP: u16 = 0x09;
pub const SCAN_PAGE_DOWN: u16 = 0x0a;
pub const SCAN_F1: u16 = 0x0b;
pub const SCAN_ESCAPE: u16 = 0x17;

const ESCAPE: u8 = 0x1b;
const DELETE_BYTE: u8 = 0x7f;
const BACKSPACE: u16 = 8;

fn scan(code: u16) -> InputKey {
    InputKey {
        scan_code: code,
        unicode_char: 0,
    }
}

/// Maps the number in `ESC [ n ~` to a key.
fn tilde_key(number: u32) -> Option<InputKey> {
    Some(scan(match number {
        1 | 7 => SCAN_HOME,
        2 => SCAN_INSERT,
        3 => SCAN_DELETE,
        4 | 8 => SCAN_END,
        5 => SCAN_PAGE_UP,
        6 => SCAN_PAGE_DOWN,
        // The function keys skip numbers in xterm's encoding.
        11..=15 => SCAN_F1 + (number - 11) as u16,
        17..=21 => SCAN_F1 + (number - 12) as u16,
        _ => return None,
    }))
}

/// Reads one key. `next` returns the next byte if one is waiting, without blocking: a lone
/// `ESC` (the Escape key) is told from the start of a sequence by whether more bytes follow at
/// once, as terminals send a whole sequence in one burst.
pub fn decode(mut next: impl FnMut() -> Option<u8>) -> Option<InputKey> {
    let first = next()?;
    let character = |unit: u16| {
        Some(InputKey {
            scan_code: SCAN_NONE,
            unicode_char: unit,
        })
    };
    match first {
        ESCAPE => {}
        DELETE_BYTE => return character(BACKSPACE),
        0x80..=0xff => return decode_utf8(first, &mut next),
        byte => return character(byte as u16),
    }
    let Some(introducer) = next() else {
        return Some(scan(SCAN_ESCAPE));
    };
    match introducer {
        b'[' | b'O' => {}
        _ => return Some(scan(SCAN_ESCAPE)),
    }
    let mut number = 0u32;
    loop {
        let Some(byte) = next() else {
            return Some(scan(SCAN_ESCAPE));
        };
        match byte {
            b'0'..=b'9' => number = number * 10 + (byte - b'0') as u32,
            b';' => number = 0,
            b'A' => return Some(scan(SCAN_UP)),
            b'B' => return Some(scan(SCAN_DOWN)),
            b'C' => return Some(scan(SCAN_RIGHT)),
            b'D' => return Some(scan(SCAN_LEFT)),
            b'H' => return Some(scan(SCAN_HOME)),
            b'F' => return Some(scan(SCAN_END)),
            b'P'..=b'S' => return Some(scan(SCAN_F1 + (byte - b'P') as u16)),
            b'~' => return tilde_key(number),
            _ => return None,
        }
    }
}

fn decode_utf8(first: u8, next: &mut impl FnMut() -> Option<u8>) -> Option<InputKey> {
    let (length, mut unit) = match first {
        0xc0..=0xdf => (1, (first & 0x1f) as u32),
        0xe0..=0xef => (2, (first & 0x0f) as u32),
        _ => return None,
    };
    for _ in 0..length {
        unit = unit << 6 | (next()? & 0x3f) as u32;
    }
    Some(InputKey {
        scan_code: SCAN_NONE,
        unicode_char: u16::try_from(unit).ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(bytes: &[u8]) -> alloc::vec::Vec<(u16, u16)> {
        let position = core::cell::Cell::new(0);
        let mut result = alloc::vec::Vec::new();
        while position.get() < bytes.len() {
            let key = decode(|| {
                let byte = bytes.get(position.get()).copied();
                position.set(position.get() + 1);
                byte
            });
            if let Some(key) = key {
                result.push((key.scan_code, key.unicode_char));
            }
        }
        result
    }

    #[test]
    fn plain_characters_and_control_keys() {
        assert_eq!(keys(b"a\r\x7f"), [(0, 'a' as u16), (0, 13), (0, 8)]);
    }

    #[test]
    fn arrows_and_navigation() {
        assert_eq!(
            keys(b"\x1b[A\x1b[B\x1b[C\x1b[D\x1b[H\x1b[F\x1b[3~\x1b[5~\x1b[6~"),
            [
                (SCAN_UP, 0),
                (SCAN_DOWN, 0),
                (SCAN_RIGHT, 0),
                (SCAN_LEFT, 0),
                (SCAN_HOME, 0),
                (SCAN_END, 0),
                (SCAN_DELETE, 0),
                (SCAN_PAGE_UP, 0),
                (SCAN_PAGE_DOWN, 0)
            ]
        );
    }

    #[test]
    fn function_keys() {
        assert_eq!(
            keys(b"\x1bOP\x1b[15~\x1b[17~\x1b[21~"),
            [(SCAN_F1, 0), (SCAN_F1 + 4, 0), (SCAN_F1 + 5, 0), (SCAN_F1 + 9, 0)]
        );
    }

    #[test]
    fn a_lone_escape_is_the_escape_key() {
        assert_eq!(keys(b"\x1b"), [(SCAN_ESCAPE, 0)]);
        assert_eq!(keys(b"\x1bx"), [(SCAN_ESCAPE, 0)]);
    }

    #[test]
    fn utf8_becomes_one_character() {
        assert_eq!(keys("é".as_bytes()), [(0, 0xe9)]);
    }
}
