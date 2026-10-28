//! The UEFI firmware image `build.rs` compiled from `crates/uemulator-firmware`, and the
//! variables the CLI seeds into a machine's nvram so it boots something specific.

use uemulator::{KeyValueError, KeyValueStore};

/// The flat image `MachineConfig::uefi_firmware` takes.
pub const IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/uemulator-firmware.bin"));

/// `8be4df61-93ca-11d2-aa0d-00e098032b8c`, the vendor of the variables the boot manager reads.
const GLOBAL_VARIABLE_GUID: &str = "8be4df61-93ca-11d2-aa0d-00e098032b8c";
const ATTRIBUTES_NV_BS_RT: u32 = 0x07;
const LOAD_OPTION_ACTIVE: u32 = 1;
const DESCRIPTION: &str = "uemulator";
const MEDIA_DEVICE_PATH: u8 = 4;
const FILE_PATH_SUBTYPE: u8 = 4;
const END_DEVICE_PATH: [u8; 4] = [0x7f, 0xff, 4, 0];
const NODE_HEADER_BYTES: usize = 4;

/// The firmware's key for a variable (`var/<guid>/<name>`, `plans/19-uefi-firmware.md`).
fn variable_key(name: &str) -> Vec<u8> {
    format!("var/{GLOBAL_VARIABLE_GUID}/{name}").into_bytes()
}

fn variable_value(data: &[u8]) -> Vec<u8> {
    let mut value = ATTRIBUTES_NV_BS_RT.to_le_bytes().to_vec();
    value.extend_from_slice(data);
    value
}

fn utf16_nul(text: &str) -> Vec<u8> {
    text.encode_utf16()
        .chain(std::iter::once(0))
        .flat_map(|unit| unit.to_le_bytes())
        .collect()
}

/// An `EFI_LOAD_OPTION` that runs `loader` (a path on the first file system that has it, such as
/// `\EFI\BOOT\BOOTX64.EFI`) with `options` as its load options.
pub fn load_option(loader: &str, options: &str) -> Vec<u8> {
    let name = utf16_nul(loader);
    let mut path = vec![MEDIA_DEVICE_PATH, FILE_PATH_SUBTYPE];
    path.extend_from_slice(&((NODE_HEADER_BYTES + name.len()) as u16).to_le_bytes());
    path.extend_from_slice(&name);
    path.extend_from_slice(&END_DEVICE_PATH);
    let mut option = LOAD_OPTION_ACTIVE.to_le_bytes().to_vec();
    option.extend_from_slice(&(path.len() as u16).to_le_bytes());
    option.extend_from_slice(&utf16_nul(DESCRIPTION));
    option.extend_from_slice(&path);
    if !options.is_empty() {
        option.extend_from_slice(&utf16_nul(options));
    }
    option
}

/// Makes `Boot0000` run `loader` with `options` and puts it first in `BootOrder`, keeping the
/// other entries.
pub fn set_boot_entry(
    store: &mut dyn KeyValueStore,
    loader: &str,
    options: &str,
) -> Result<(), KeyValueError> {
    store.set(&variable_key("Boot0000"), &variable_value(&load_option(loader, options)))?;
    let order_key = variable_key("BootOrder");
    let mut order = vec![0u8, 0];
    if let Some(existing) = store.get(&order_key) {
        for pair in existing.get(4..).unwrap_or(&[]).chunks_exact(2) {
            if pair != [0, 0] {
                order.extend_from_slice(pair);
            }
        }
    }
    store.set(&order_key, &variable_value(&order))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uemulator::MemoryKeyValueStore;

    #[test]
    fn the_image_has_the_header_the_library_checks() {
        assert_eq!(&IMAGE[..8], b"UEMUFW64");
        assert!(IMAGE.len() > 4096);
    }

    #[test]
    fn a_load_option_lays_out_as_the_specification_says() {
        let option = load_option("\\EFI\\BOOT\\BOOTX64.EFI", "a b");
        assert_eq!(u32::from_le_bytes(option[..4].try_into().unwrap()), 1);
        let path_length = u16::from_le_bytes(option[4..6].try_into().unwrap()) as usize;
        let description = utf16_nul("uemulator");
        assert_eq!(&option[6..6 + description.len()], description.as_slice());
        let path = &option[6 + description.len()..][..path_length];
        assert_eq!(&path[..2], &[4, 4]);
        assert_eq!(&path[path.len() - 4..], &END_DEVICE_PATH);
        assert_eq!(&option[6 + description.len() + path_length..], utf16_nul("a b").as_slice());
    }

    #[test]
    fn the_boot_entry_goes_first_and_the_others_stay() {
        let mut store = MemoryKeyValueStore::new();
        store
            .set(&variable_key("BootOrder"), &variable_value(&[1, 0, 2, 0]))
            .unwrap();
        set_boot_entry(&mut store, "\\x.efi", "").unwrap();
        let order = store.get(&variable_key("BootOrder")).unwrap();
        assert_eq!(&order[4..], &[0, 0, 1, 0, 2, 0]);
        // Setting it again does not duplicate it.
        set_boot_entry(&mut store, "\\y.efi", "").unwrap();
        assert_eq!(&store.get(&variable_key("BootOrder")).unwrap()[4..], &[0, 0, 1, 0, 2, 0]);
        assert!(store.get(&variable_key("Boot0000")).is_some());
    }
}
