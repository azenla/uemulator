//! The boot manager: `BootNext` and `BootOrder`/`Boot####` from the variable store, then the
//! removable-media path `\EFI\BOOT\BOOTX64.EFI` on every file system.

use alloc::vec::Vec;
use core::ptr;

use crate::devicepath::DevicePath;
use crate::load_option::{Candidate, parse, u16_list};
use crate::serial::println;
use crate::tables::BOOT_SERVICES;
use crate::uefi::variable_attribute::{BOOTSERVICE_ACCESS, RUNTIME_ACCESS};
use crate::uefi::*;
use crate::{cpu, images, protocols, variables};

const REMOVABLE_MEDIA_PATH: &str = "\\EFI\\BOOT\\BOOTX64.EFI";

fn boot_option(number: u16) -> Option<Candidate> {
    let name = alloc::format!("Boot{number:04X}");
    parse(&variables::read(&name, &guids::GLOBAL_VARIABLE)?, number)
}

/// `BootNext`, then `BootOrder`, then whatever the media offers.
fn candidates() -> Vec<Candidate> {
    let mut found = Vec::new();
    if let Some(next) =
        variables::read("BootNext", &guids::GLOBAL_VARIABLE).filter(|data| data.len() == 2)
    {
        // One-shot: gone before it is tried, so a bad entry cannot trap the machine.
        variables::write("BootNext", &guids::GLOBAL_VARIABLE, 0, &[]);
        found.extend(boot_option(u16::from_le_bytes([next[0], next[1]])));
    }
    if let Some(order) = variables::read("BootOrder", &guids::GLOBAL_VARIABLE) {
        for number in u16_list(&order) {
            found.extend(boot_option(number));
        }
    }
    for handle in protocols::handles_with(&guids::SIMPLE_FILE_SYSTEM) {
        let Some(device) = protocols::interface_of(handle, &guids::DEVICE_PATH)
            .and_then(|pointer| unsafe { DevicePath::from_raw(pointer.cast()) })
        else {
            continue;
        };
        let Some(volume) = protocols::interface_of(handle, &guids::SIMPLE_FILE_SYSTEM) else {
            continue;
        };
        if crate::files::exists(volume.cast(), REMOVABLE_MEDIA_PATH) {
            found.push(Candidate {
                description: alloc::format!("removable media on {}", device.text()),
                path: device.append(&DevicePath::new().file_path(REMOVABLE_MEDIA_PATH)),
                options: Vec::new(),
                number: None,
            });
        }
    }
    found
}

fn try_boot(candidate: &Candidate) -> Status {
    let boot = unsafe { BOOT_SERVICES.get().assume_init_ref() };
    let mut path = candidate.path.as_bytes().to_vec();
    let mut image: Handle = ptr::null_mut();
    let status = (boot.load_image)(
        1,
        ptr::null_mut(),
        path.as_mut_ptr().cast(),
        ptr::null_mut(),
        0,
        &mut image,
    );
    if status != SUCCESS {
        return status;
    }
    if !candidate.options.is_empty() {
        images::set_load_options(image, &candidate.options);
    }
    if let Some(number) = candidate.number {
        variables::write(
            "BootCurrent",
            &guids::GLOBAL_VARIABLE,
            BOOTSERVICE_ACCESS | RUNTIME_ACCESS,
            &number.to_le_bytes(),
        );
    }
    (boot.start_image)(image, ptr::null_mut(), ptr::null_mut())
}

pub fn run() -> ! {
    println!("uemulator UEFI firmware");
    for candidate in candidates() {
        println!(
            "Booting {} ({})",
            candidate.description,
            candidate.path.text()
        );
        let status = try_boot(&candidate);
        println!("{} returned with status {status:#x}", candidate.description);
    }
    println!("No bootable device.");
    cpu::halt_forever()
}
