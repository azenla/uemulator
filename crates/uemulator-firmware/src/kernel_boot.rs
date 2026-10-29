//! Boots the kernel the library placed in RAM, the way OVMF boots QEMU's `-kernel`: the image is
//! loaded from its buffer, the command line is its load options, and the initramfs is served by
//! the `LoadFile2` protocol the Linux EFI stub looks for under `LINUX_EFI_INITRD_MEDIA_GUID`.

use core::ffi::c_void;
use core::ptr;

use crate::devicepath::DevicePath;
use crate::handoff::Payload;
use crate::sync::Global;
use crate::tables::BOOT_SERVICES;
use crate::uefi::*;
use crate::{images, protocols, text};

/// The name the loaded image's device path carries; nothing resolves it.
const KERNEL_FILE_NAME: &str = "kernel";

#[repr(C)]
struct LoadFile2 {
    load_file:
        extern "efiapi" fn(*mut LoadFile2, *mut c_void, u8, *mut usize, *mut c_void) -> Status,
}

static PROTOCOL: Global<LoadFile2> = Global::new(LoadFile2 { load_file });
static INITRD: Global<(u64, u64)> = Global::new((0, 0));
/// Handed to the protocol database, which keeps the pointer: it must outlive every user.
static INITRD_PATH: Global<Option<DevicePath>> = Global::new(None);

extern "efiapi" fn load_file(
    _: *mut LoadFile2,
    _file_path: *mut c_void,
    boot_policy: u8,
    size: *mut usize,
    buffer: *mut c_void,
) -> Status {
    if size.is_null() || boot_policy != 0 {
        return if size.is_null() {
            INVALID_PARAMETER
        } else {
            UNSUPPORTED
        };
    }
    let (base, length) = *INITRD.get();
    let length = length as usize;
    unsafe {
        let capacity = size.read_unaligned();
        size.write_unaligned(length);
        if buffer.is_null() || capacity < length {
            return BUFFER_TOO_SMALL;
        }
        ptr::copy_nonoverlapping(base as *const u8, buffer.cast::<u8>(), length);
    }
    SUCCESS
}

fn publish_initrd(range: (u64, u64)) -> Result<(), Status> {
    *INITRD.get() = range;
    let path = INITRD_PATH
        .get()
        .insert(DevicePath::new().vendor_media(guids::LINUX_EFI_INITRD_MEDIA));
    let handle = protocols::install(
        ptr::null_mut(),
        guids::DEVICE_PATH,
        path.as_bytes().as_ptr().cast_mut().cast(),
    )?;
    protocols::install(
        handle,
        guids::LOAD_FILE2,
        PROTOCOL.get() as *mut LoadFile2 as *mut c_void,
    )?;
    Ok(())
}

/// Starts the kernel. Returns only when it could not be loaded or when it returned.
pub fn run(payload: &Payload) -> Status {
    let slice = |(base, size): (u64, u64)| unsafe {
        core::slice::from_raw_parts(base as *const u8, size as usize)
    };
    let kernel = slice(payload.kernel);
    let cmdline = core::str::from_utf8(slice(payload.cmdline)).unwrap_or("");
    if let Some(range) = payload.initrd
        && let Err(status) = publish_initrd(range)
    {
        return status;
    }
    let path = DevicePath::new().file_path(KERNEL_FILE_NAME);
    let image = match images::load(kernel, ptr::null_mut(), ptr::null_mut(), Some(&path)) {
        Ok(image) => image,
        Err(status) => {
            match status {
                UNSUPPORTED | LOAD_ERROR => crate::serial::println!(
                    "the kernel is not an EFI-stub image (status {status:#x}); boot it with --firmware none"
                ),
                _ => crate::serial::println!("could not load the kernel (status {status:#x})"),
            }
            return status;
        }
    };
    let options: alloc::vec::Vec<u8> = text::to_utf16_nul(cmdline)
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    images::set_load_options(image, &options);
    let boot = unsafe { BOOT_SERVICES.get().assume_init_ref() };
    (boot.start_image)(image, ptr::null_mut(), ptr::null_mut())
}
