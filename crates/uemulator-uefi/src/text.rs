//! The conversion of guest console bytes to what the firmware's text console takes.

use alloc::string::String;
use alloc::vec::Vec;

/// UTF-16 code units are collected in a buffer of this capacity and handed over as soon as
/// `OUTPUT_FLUSH_THRESHOLD` is reached. The gap leaves room for the largest step (CR plus a
/// surrogate pair) and the terminating NUL, so the buffer never grows.
pub const OUTPUT_BUFFER_UNITS: usize = 512;
pub const OUTPUT_FLUSH_THRESHOLD: usize = 500;

/// Appends `bytes` to `tail` (the bytes of a UTF-8 sequence cut off by the previous call) and
/// returns the text that is complete, leaving an unfinished sequence in `tail`.
pub fn take_text(tail: &mut Vec<u8>, bytes: &[u8]) -> String {
    tail.extend_from_slice(bytes);
    let valid = match core::str::from_utf8(tail) {
        Ok(s) => s.len(),
        // `error_len() == None` means the input ends inside a sequence: keep it for the next call.
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        // Not UTF-8 at all: pass the bytes through as Latin-1 rather than stall.
        Err(_) => tail.len(),
    };
    let text: String = match core::str::from_utf8(&tail[..valid]) {
        Ok(s) => s.into(),
        Err(_) => tail[..valid].iter().map(|&b| b as char).collect(),
    };
    tail.drain(..valid);
    text
}

/// Converts `text` to UTF-16 with LF turned into CR LF, as the firmware's console needs, and
/// calls `emit` with each full buffer of units and once more with the remainder.
pub fn to_firmware_units(text: &str, mut emit: impl FnMut(&mut Vec<u16>)) {
    let mut units: Vec<u16> = Vec::with_capacity(OUTPUT_BUFFER_UNITS);
    for c in text.chars() {
        if c == '\n' {
            units.push(b'\r' as u16);
        }
        let mut buf = [0u16; 2];
        units.extend_from_slice(c.encode_utf16(&mut buf));
        if units.len() >= OUTPUT_FLUSH_THRESHOLD {
            emit(&mut units);
        }
    }
    emit(&mut units);
}

/// Turns `units` into a NUL-terminated string: a NUL inside would end it early, so those go.
pub fn terminate(units: &mut Vec<u16>) {
    units.retain(|&u| u != 0);
    units.push(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn units_of(text: &str) -> Vec<u16> {
        let mut out = Vec::new();
        to_firmware_units(text, |units| {
            out.extend_from_slice(units);
            units.clear();
        });
        out
    }

    fn chunks_of(text: &str) -> Vec<Vec<u16>> {
        let mut out = Vec::new();
        to_firmware_units(text, |units| {
            out.push(units.clone());
            units.clear();
        });
        out
    }

    #[test]
    fn ascii_passes_through_and_leaves_no_tail() {
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, b"hello\n"), "hello\n");
        assert!(tail.is_empty());
    }

    #[test]
    fn empty_input_gives_empty_text() {
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, b""), "");
        assert!(tail.is_empty());
    }

    #[test]
    fn two_byte_sequence_split_across_calls_is_joined() {
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, &[b'a', 0xc3]), "a");
        assert_eq!(tail, vec![0xc3]);
        assert_eq!(take_text(&mut tail, &[0xa9, b'b']), "\u{e9}b");
        assert!(tail.is_empty());
    }

    #[test]
    fn three_byte_sequence_split_one_byte_at_a_time_is_joined() {
        // U+20AC EURO SIGN is E2 82 AC.
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, &[0xe2]), "");
        assert_eq!(take_text(&mut tail, &[0x82]), "");
        assert_eq!(tail, vec![0xe2, 0x82]);
        assert_eq!(take_text(&mut tail, &[0xac]), "\u{20ac}");
        assert!(tail.is_empty());
    }

    #[test]
    fn four_byte_sequence_split_in_the_middle_is_joined() {
        // U+1F600 is F0 9F 98 80.
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, &[0xf0, 0x9f]), "");
        assert_eq!(take_text(&mut tail, &[0x98, 0x80]), "\u{1f600}");
    }

    #[test]
    fn complete_text_before_an_unfinished_sequence_is_returned_at_once() {
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, b"ab\xe2\x82"), "ab");
        assert_eq!(tail, vec![0xe2, 0x82]);
    }

    #[test]
    fn bytes_that_are_not_utf8_pass_through_as_latin_1() {
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, &[0xff, b'x']), "\u{ff}x");
        assert!(tail.is_empty());
    }

    #[test]
    fn lone_continuation_byte_is_not_held_back() {
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, &[0x80]), "\u{80}");
        assert!(tail.is_empty());
    }

    #[test]
    fn lf_becomes_cr_lf() {
        assert_eq!(units_of("a\nb"), vec![0x61, 0x0d, 0x0a, 0x62]);
    }

    #[test]
    fn an_existing_cr_is_kept_and_the_lf_after_it_gets_its_own() {
        assert_eq!(units_of("\r\n"), vec![0x0d, 0x0d, 0x0a]);
    }

    #[test]
    fn characters_above_the_bmp_become_surrogate_pairs() {
        assert_eq!(units_of("\u{1f600}"), vec![0xd83d, 0xde00]);
    }

    #[test]
    fn bmp_characters_are_one_unit() {
        assert_eq!(units_of("\u{e9}\u{20ac}"), vec![0xe9, 0x20ac]);
    }

    #[test]
    fn empty_text_still_calls_emit_once_with_nothing() {
        assert_eq!(chunks_of(""), vec![Vec::<u16>::new()]);
    }

    #[test]
    fn buffer_is_handed_over_when_the_threshold_is_reached() {
        let text = "a".repeat(OUTPUT_FLUSH_THRESHOLD + 100);
        let chunks = chunks_of(&text);
        assert_eq!(
            chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![OUTPUT_FLUSH_THRESHOLD, 100]
        );
    }

    #[test]
    fn exactly_the_threshold_leaves_an_empty_remainder() {
        let chunks = chunks_of(&"a".repeat(OUTPUT_FLUSH_THRESHOLD));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), OUTPUT_FLUSH_THRESHOLD);
        assert!(chunks[1].is_empty());
    }

    #[test]
    fn one_unit_short_of_the_threshold_is_a_single_chunk() {
        let chunks = chunks_of(&"a".repeat(OUTPUT_FLUSH_THRESHOLD - 1));
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunks_concatenate_to_the_whole_conversion() {
        let text = "x\n\u{1f600}".repeat(400);
        let joined: Vec<u16> = chunks_of(&text).concat();
        let expected: Vec<u16> = text
            .chars()
            .flat_map(|c| {
                let mut buf = [0u16; 2];
                let mut units = Vec::new();
                if c == '\n' {
                    units.push(0x0d);
                }
                units.extend_from_slice(c.encode_utf16(&mut buf));
                units
            })
            .collect();
        assert_eq!(joined, expected);
    }

    #[test]
    fn buffer_never_outgrows_its_capacity_so_a_nul_always_fits() {
        // Worst case: the buffer sits one unit under the threshold and the next step is the
        // largest, a surrogate pair or a CR LF.
        for filler in 0..4 {
            for tail_char in ["\n", "\u{1f600}"] {
                let text = format!(
                    "{}{tail_char}",
                    "a".repeat(OUTPUT_FLUSH_THRESHOLD - 2 + filler)
                );
                to_firmware_units(&text, |units| {
                    assert!(units.len() < OUTPUT_BUFFER_UNITS);
                    assert_eq!(units.capacity(), OUTPUT_BUFFER_UNITS);
                    units.clear();
                });
            }
        }
    }

    #[test]
    fn terminate_appends_one_nul() {
        let mut units = vec![0x61, 0x62];
        terminate(&mut units);
        assert_eq!(units, vec![0x61, 0x62, 0]);
    }

    #[test]
    fn terminate_removes_nuls_inside_the_text() {
        let mut units = vec![0, 0x61, 0, 0, 0x62, 0];
        terminate(&mut units);
        assert_eq!(units, vec![0x61, 0x62, 0]);
    }

    #[test]
    fn terminate_of_nothing_is_just_the_nul() {
        let mut units = Vec::new();
        terminate(&mut units);
        assert_eq!(units, vec![0]);
    }

    #[test]
    fn a_nul_byte_from_the_guest_survives_the_text_stage() {
        // It is `terminate` that removes it, so the text stage must not choke on it.
        let mut tail = Vec::new();
        assert_eq!(take_text(&mut tail, b"a\0b"), "a\0b".to_string());
    }
}
