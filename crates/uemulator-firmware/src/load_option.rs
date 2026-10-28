//! `EFI_LOAD_OPTION`: a `Boot####` variable's description, device path and optional data.

use alloc::string::String;
use alloc::vec::Vec;

use crate::devicepath::DevicePath;

const ACTIVE: u32 = 1;
/// attributes u32, file path list length u16, then the description.
const FIXED_BYTES: usize = 6;

pub struct Candidate {
    pub description: String,
    pub path: DevicePath,
    pub options: Vec<u8>,
    /// The `Boot####` number, for `BootCurrent`.
    pub number: Option<u16>,
}

pub fn u16_list(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]])).collect()
}

/// Parses an `EFI_LOAD_OPTION`; `None` for inactive or malformed ones.
pub fn parse(bytes: &[u8], number: u16) -> Option<Candidate> {
    let attributes = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    let path_length = u16::from_le_bytes(bytes.get(4..6)?.try_into().ok()?) as usize;
    if attributes & ACTIVE == 0 {
        return None;
    }
    let rest = bytes.get(FIXED_BYTES..)?;
    let units = u16_list(rest);
    let name_units = units.iter().position(|&unit| unit == 0)?;
    let description = crate::text::to_string(&units[..name_units]);
    let path_start = (name_units + 1) * 2;
    let path_bytes = rest.get(path_start..path_start + path_length)?;
    let path = unsafe { DevicePath::from_raw(path_bytes.as_ptr()) }?;
    Some(Candidate {
        description,
        path,
        options: rest[path_start + path_length..].to_vec(),
        number: Some(number),
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    fn load_option(active: bool, description: &str, path: &DevicePath, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(active as u32).to_le_bytes());
        bytes.extend_from_slice(&(path.as_bytes().len() as u16).to_le_bytes());
        for unit in crate::text::to_utf16_nul(description) {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        bytes.extend_from_slice(path.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn a_load_option_gives_its_path_and_options() {
        let path = DevicePath::new().file_path("\\EFI\\fedora\\shimx64.efi");
        let bytes = load_option(true, "Fedora", &path, b"o\0p\0");
        let candidate = parse(&bytes, 3).unwrap();
        assert_eq!(candidate.description, "Fedora");
        assert_eq!(candidate.path, path);
        assert_eq!(candidate.options, b"o\0p\0");
        assert_eq!(candidate.number, Some(3));
    }

    #[test]
    fn inactive_and_truncated_options_are_skipped() {
        let path = DevicePath::new().file_path("\\x");
        assert!(parse(&load_option(false, "Off", &path, b""), 0).is_none());
        let mut truncated = load_option(true, "Short", &path, b"");
        truncated.truncate(truncated.len() - 6);
        assert!(parse(&truncated, 0).is_none());
        assert!(parse(&[1, 0], 0).is_none());
    }
}
