//! `EFI_SIMPLE_FILE_SYSTEM_PROTOCOL` and `EFI_FILE_PROTOCOL` over the read-only FAT reader.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;

use crate::disks::BlockStorage;
use crate::fat::{self, Cursor, DirEntry};
use crate::text;
use crate::uefi::*;

const FILE_REVISION: u64 = 0x0001_0000;
const SIMPLE_FILE_SYSTEM_REVISION: u64 = 0x0001_0000;
const MODE_READ: u64 = 1;
const WARN_DELETE_FAILURE: Status = 4;
const FAT_ATTRIBUTE_VOLUME_MASK: u8 = 0x3f;
/// `EFI_FILE_INFO` before the name: size, file size, physical size, three times, attribute.
const FILE_INFO_FIXED_BYTES: usize = 8 * 3 + 16 * 3 + 8;
/// `EFI_FILE_SYSTEM_INFO` before the label: size, read-only (padded), volume size, free space,
/// block size (padded).
const FILE_SYSTEM_INFO_FIXED_BYTES: usize = 8 + 8 + 8 + 8 + 8;

#[repr(C)]
pub struct SimpleFileSystem {
    revision: u64,
    open_volume: extern "efiapi" fn(*mut SimpleFileSystem, *mut *mut FileNode) -> Status,
}

/// What `SimpleFileSystem`'s interface pointer really points at.
#[repr(C)]
struct Volume {
    protocol: SimpleFileSystem,
    fat: fat::Volume<BlockStorage>,
}

#[repr(C)]
pub struct FileNode {
    revision: u64,
    open: extern "efiapi" fn(*mut FileNode, *mut *mut FileNode, *const u16, u64, u64) -> Status,
    close: extern "efiapi" fn(*mut FileNode) -> Status,
    delete: extern "efiapi" fn(*mut FileNode) -> Status,
    read: extern "efiapi" fn(*mut FileNode, *mut usize, *mut c_void) -> Status,
    write: extern "efiapi" fn(*mut FileNode, *mut usize, *const c_void) -> Status,
    get_position: extern "efiapi" fn(*mut FileNode, *mut u64) -> Status,
    set_position: extern "efiapi" fn(*mut FileNode, u64) -> Status,
    get_info: extern "efiapi" fn(*mut FileNode, *const Guid, *mut usize, *mut c_void) -> Status,
    set_info: extern "efiapi" fn(*mut FileNode, *const Guid, usize, *const c_void) -> Status,
    flush: extern "efiapi" fn(*mut FileNode) -> Status,
    // Not part of the protocol: the node's own state, after the function table.
    volume: *mut Volume,
    entry: DirEntry,
    is_root: bool,
    cursor: Cursor,
    position: u64,
    /// Directory contents, read on the first `Read`.
    listing: Option<Vec<DirEntry>>,
}

pub fn create_volume(storage: BlockStorage) -> Option<*mut SimpleFileSystem> {
    let fat = fat::Volume::open(storage).ok()?;
    let volume = Box::new(Volume {
        protocol: SimpleFileSystem {
            revision: SIMPLE_FILE_SYSTEM_REVISION,
            open_volume,
        },
        fat,
    });
    Some(Box::into_raw(volume).cast())
}

/// The volume outlives every node (volumes are never destroyed), and one thread uses it.
fn volume_of(node: &FileNode) -> &'static mut fat::Volume<BlockStorage> {
    unsafe { &mut (*node.volume).fat }
}

fn new_node(volume: *mut Volume, entry: DirEntry, is_root: bool) -> *mut FileNode {
    let cursor = Cursor::new(&entry);
    Box::into_raw(Box::new(FileNode {
        revision: FILE_REVISION,
        open: file_open,
        close: file_close,
        delete: file_delete,
        read: file_read,
        write: file_write,
        get_position,
        set_position,
        get_info,
        set_info,
        flush,
        volume,
        entry,
        is_root,
        cursor,
        position: 0,
        listing: None,
    }))
}

fn root_entry() -> DirEntry {
    DirEntry {
        name: String::new(),
        attributes: 0x10,
        first_cluster: 0,
        size: 0,
        created: (0, 0),
        accessed: 0,
        modified: (0, 0),
    }
}

extern "efiapi" fn open_volume(this: *mut SimpleFileSystem, root: *mut *mut FileNode) -> Status {
    if root.is_null() {
        return INVALID_PARAMETER;
    }
    unsafe { root.write(new_node(this.cast(), root_entry(), true)) };
    SUCCESS
}

/// Looks a path up from `start` (or the root for an absolute one). Returns the entry and
/// whether it is the root itself.
fn resolve(node: &FileNode, path: &str) -> Result<(DirEntry, bool), Status> {
    let volume = volume_of(node);
    let absolute = path.starts_with(['\\', '/']);
    let (mut current, mut is_root) = if absolute {
        (root_entry(), true)
    } else {
        (node.entry.clone(), node.is_root)
    };
    for part in path.split(['\\', '/']).filter(|part| !part.is_empty() && *part != ".") {
        if !current.is_directory() {
            return Err(NOT_FOUND);
        }
        let directory = if is_root { volume.root() } else { current.first_cluster };
        if part == ".." {
            if is_root {
                continue;
            }
            let parent = volume.lookup(directory, "..").map_err(|_| NOT_FOUND)?;
            is_root = parent.first_cluster == 0;
            current = if is_root { root_entry() } else { parent };
            continue;
        }
        current = volume.lookup(directory, part).map_err(|_| NOT_FOUND)?;
        is_root = false;
    }
    Ok((current, is_root))
}

extern "efiapi" fn file_open(
    this: *mut FileNode,
    new_handle: *mut *mut FileNode,
    name: *const u16,
    mode: u64,
    _attributes: u64,
) -> Status {
    if this.is_null() || new_handle.is_null() || name.is_null() {
        return INVALID_PARAMETER;
    }
    let node = unsafe { &*this };
    let path = text::to_string(&unsafe { text::to_units(name) });
    let (entry, is_root) = match resolve(node, &path) {
        Ok(found) => found,
        Err(NOT_FOUND) if mode != MODE_READ => return WRITE_PROTECTED,
        Err(status) => return status,
    };
    if mode != MODE_READ {
        return WRITE_PROTECTED;
    }
    unsafe { new_handle.write(new_node(node.volume, entry, is_root)) };
    SUCCESS
}

extern "efiapi" fn file_close(this: *mut FileNode) -> Status {
    if this.is_null() {
        return INVALID_PARAMETER;
    }
    drop(unsafe { Box::from_raw(this) });
    SUCCESS
}

/// Nothing can be deleted; the handle is closed regardless, as the specification says.
extern "efiapi" fn file_delete(this: *mut FileNode) -> Status {
    file_close(this);
    WARN_DELETE_FAILURE
}

fn fat_time(date_time: (u16, u16)) -> Time {
    let (date, time) = date_time;
    Time {
        year: 1980 + (date >> 9),
        month: (((date >> 5) & 0xf) as u8).clamp(1, 12),
        day: ((date & 0x1f) as u8).max(1),
        hour: ((time >> 11) as u8).min(23),
        minute: (((time >> 5) & 0x3f) as u8).min(59),
        second: (((time & 0x1f) * 2) as u8).min(59),
        pad1: 0,
        nanosecond: 0,
        time_zone: UNSPECIFIED_TIMEZONE,
        daylight: 0,
        pad2: 0,
    }
}

/// An `EFI_FILE_INFO` for `entry`.
fn file_info(entry: &DirEntry) -> Vec<u8> {
    let name = text::to_utf16_nul(&entry.name);
    let size = FILE_INFO_FIXED_BYTES + name.len() * 2;
    let mut bytes = Vec::with_capacity(size);
    bytes.extend_from_slice(&(size as u64).to_le_bytes());
    bytes.extend_from_slice(&(entry.size as u64).to_le_bytes());
    bytes.extend_from_slice(&(entry.size as u64).to_le_bytes());
    let times = [fat_time(entry.created), fat_time((entry.accessed, 0)), fat_time(entry.modified)];
    for time in &times {
        bytes.extend_from_slice(unsafe {
            core::slice::from_raw_parts((time as *const Time).cast::<u8>(), core::mem::size_of::<Time>())
        });
    }
    bytes.extend_from_slice(&((entry.attributes & FAT_ATTRIBUTE_VOLUME_MASK) as u64).to_le_bytes());
    for unit in name {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    bytes
}

/// Hands `bytes` to the caller's buffer with the `BUFFER_TOO_SMALL` protocol.
unsafe fn deliver(bytes: &[u8], size: *mut usize, buffer: *mut c_void) -> Status {
    unsafe {
        let capacity = size.read_unaligned();
        size.write_unaligned(bytes.len());
        if capacity < bytes.len() {
            return BUFFER_TOO_SMALL;
        }
        ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), bytes.len());
    }
    SUCCESS
}

extern "efiapi" fn file_read(this: *mut FileNode, size: *mut usize, buffer: *mut c_void) -> Status {
    if this.is_null() || size.is_null() || (buffer.is_null() && unsafe { size.read_unaligned() } != 0) {
        return INVALID_PARAMETER;
    }
    let node = unsafe { &mut *this };
    let requested = unsafe { size.read_unaligned() };
    if !node.entry.is_directory() {
        let destination = unsafe { core::slice::from_raw_parts_mut(buffer.cast::<u8>(), requested) };
        let volume = volume_of(node);
        return match volume.read(&mut node.cursor, node.position, destination) {
            Ok(count) => {
                node.position += count as u64;
                unsafe { size.write_unaligned(count) };
                SUCCESS
            }
            Err(_) => DEVICE_ERROR,
        };
    }
    if node.listing.is_none() {
        let directory = if node.is_root { 0 } else { node.entry.first_cluster };
        match volume_of(node).list(directory) {
            Ok(entries) => node.listing = Some(entries),
            Err(_) => return DEVICE_ERROR,
        }
    }
    let Some(entry) = node
        .listing
        .as_ref()
        .and_then(|entries| entries.get(node.position as usize))
    else {
        unsafe { size.write_unaligned(0) };
        return SUCCESS;
    };
    let info = file_info(entry);
    let status = unsafe { deliver(&info, size, buffer) };
    if status == SUCCESS {
        node.position += 1;
    }
    status
}

extern "efiapi" fn file_write(_: *mut FileNode, _: *mut usize, _: *const c_void) -> Status {
    WRITE_PROTECTED
}

extern "efiapi" fn get_position(this: *mut FileNode, position: *mut u64) -> Status {
    if this.is_null() || position.is_null() {
        return INVALID_PARAMETER;
    }
    let node = unsafe { &*this };
    if node.entry.is_directory() {
        return UNSUPPORTED;
    }
    unsafe { position.write_unaligned(node.position) };
    SUCCESS
}

extern "efiapi" fn set_position(this: *mut FileNode, position: u64) -> Status {
    if this.is_null() {
        return INVALID_PARAMETER;
    }
    let node = unsafe { &mut *this };
    if node.entry.is_directory() {
        if position != 0 {
            return UNSUPPORTED;
        }
        node.position = 0;
        return SUCCESS;
    }
    node.position = if position == u64::MAX { node.entry.size as u64 } else { position };
    SUCCESS
}

extern "efiapi" fn get_info(
    this: *mut FileNode,
    kind: *const Guid,
    size: *mut usize,
    buffer: *mut c_void,
) -> Status {
    if this.is_null() || kind.is_null() || size.is_null() {
        return INVALID_PARAMETER;
    }
    let node = unsafe { &*this };
    let kind = unsafe { kind.read_unaligned() };
    let volume = volume_of(node);
    let bytes = if kind == guids::FILE_INFO {
        file_info(&node.entry)
    } else if kind == guids::FILE_SYSTEM_INFO {
        let label = text::to_utf16_nul(&volume.label);
        let total = FILE_SYSTEM_INFO_FIXED_BYTES + label.len() * 2;
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(&(total as u64).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&volume.total_bytes().to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(volume.cluster_bytes() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        for unit in label {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        bytes
    } else if kind == guids::FILE_SYSTEM_VOLUME_LABEL {
        text::to_utf16_nul(&volume.label).iter().flat_map(|unit| unit.to_le_bytes()).collect()
    } else {
        return UNSUPPORTED;
    };
    if buffer.is_null() && unsafe { size.read_unaligned() } >= bytes.len() {
        return INVALID_PARAMETER;
    }
    unsafe { deliver(&bytes, size, buffer) }
}

extern "efiapi" fn set_info(_: *mut FileNode, _: *const Guid, _: usize, _: *const c_void) -> Status {
    WRITE_PROTECTED
}

extern "efiapi" fn flush(_: *mut FileNode) -> Status {
    SUCCESS
}

/// Reads a whole file by path (`\EFI\BOOT\BOOTX64.EFI`) from a volume.
pub fn read_file(volume: *mut SimpleFileSystem, path: &str) -> Result<Vec<u8>, Status> {
    let mut root: *mut FileNode = ptr::null_mut();
    let status = open_volume(volume, &mut root);
    if status != SUCCESS {
        return Err(status);
    }
    let result = (|| {
        let (entry, _) = resolve(unsafe { &*root }, path)?;
        if entry.is_directory() {
            return Err(NOT_FOUND);
        }
        let mut file = unsafe { &mut *root }.clone_for(&entry);
        let mut data = alloc::vec![0u8; entry.size as usize];
        let count = volume_of(&file).read(&mut file.cursor, 0, &mut data).map_err(|_| DEVICE_ERROR)?;
        data.truncate(count);
        Ok(data)
    })();
    file_close(root);
    result
}

impl FileNode {
    /// A scratch node for `entry`, used for reads that do not go through the protocol.
    fn clone_for(&mut self, entry: &DirEntry) -> Box<FileNode> {
        unsafe { Box::from_raw(new_node(self.volume, entry.clone(), false)) }
    }
}

/// Whether the path names a file on the volume.
pub fn exists(volume: *mut SimpleFileSystem, path: &str) -> bool {
    let mut root: *mut FileNode = ptr::null_mut();
    if open_volume(volume, &mut root) != SUCCESS {
        return false;
    }
    let found = resolve(unsafe { &*root }, path).is_ok();
    file_close(root);
    found
}
