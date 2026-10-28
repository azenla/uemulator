//! UCS-2/UTF-16 helpers: UEFI strings are NUL-terminated `u16` arrays handed over as raw
//! pointers.

use alloc::string::String;
use alloc::vec::Vec;

/// Length in code units of the NUL-terminated string at `string`.
///
/// # Safety
/// `string` must point at a readable, NUL-terminated array.
pub unsafe fn length(string: *const u16) -> usize {
    let mut length = 0;
    while unsafe { string.add(length).read_unaligned() } != 0 {
        length += 1;
    }
    length
}

/// # Safety
/// As `length`.
pub unsafe fn to_units<'a>(string: *const u16) -> Vec<u16> {
    let length = unsafe { length(string) };
    (0..length)
        .map(|index| unsafe { string.add(index).read_unaligned() })
        .collect()
}

pub fn to_string(units: &[u16]) -> String {
    char::decode_utf16(units.iter().copied())
        .map(|unit| unit.unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect()
}

/// The string as UTF-16 with its terminating NUL.
pub fn to_utf16_nul(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(core::iter::once(0)).collect()
}

/// ASCII case folding only, which is what FAT's short names and UEFI's file names need here.
pub fn eq_ignore_case(left: &str, right: &str) -> bool {
    left.len() == right.len()
        && left
            .bytes()
            .zip(right.bytes())
            .all(|(a, b)| a.eq_ignore_ascii_case(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_utf16() {
        let units = to_utf16_nul("EFI\\BOOT é");
        assert_eq!(*units.last().unwrap(), 0);
        assert_eq!(to_string(&units[..units.len() - 1]), "EFI\\BOOT é");
        assert_eq!(unsafe { length(units.as_ptr()) }, units.len() - 1);
        assert_eq!(
            unsafe { to_units(units.as_ptr()) },
            &units[..units.len() - 1]
        );
    }

    #[test]
    fn unpaired_surrogates_become_the_replacement_character() {
        assert_eq!(to_string(&[0x41, 0xd800, 0x42]), "A\u{fffd}B");
    }

    #[test]
    fn case_folding_is_ascii_only() {
        assert!(eq_ignore_case("BootX64.EFI", "BOOTX64.efi"));
        assert!(!eq_ignore_case("a", "ab"));
    }
}
