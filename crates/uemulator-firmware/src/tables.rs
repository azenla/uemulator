//! The system table and the two service tables, kept in the firmware image so they survive
//! `ExitBootServices`, and the configuration table array.

use core::ffi::c_void;
use core::mem::MaybeUninit;
use core::ptr;

use crate::crc32::crc32;
use crate::sync::Global;
use crate::uefi::*;

const MAX_CONFIGURATION_TABLES: usize = 16;

pub static SYSTEM_TABLE: Global<SystemTable> = Global::new(SystemTable {
    header: TableHeader {
        signature: SYSTEM_TABLE_SIGNATURE,
        revision: SPECIFICATION_REVISION,
        header_size: core::mem::size_of::<SystemTable>() as u32,
        crc32: 0,
        reserved: 0,
    },
    firmware_vendor: ptr::null(),
    firmware_revision: 0,
    console_in_handle: ptr::null_mut(),
    con_in: ptr::null_mut(),
    console_out_handle: ptr::null_mut(),
    con_out: ptr::null_mut(),
    standard_error_handle: ptr::null_mut(),
    std_err: ptr::null_mut(),
    runtime_services: ptr::null_mut(),
    boot_services: ptr::null_mut(),
    number_of_table_entries: 0,
    configuration_table: ptr::null_mut(),
});

/// Filled in field by field by the modules that implement the services (their `install`
/// functions) before anything reads them; `services::init` is the only place that does that.
pub static BOOT_SERVICES: Global<MaybeUninit<BootServices>> = Global::new(MaybeUninit::zeroed());
pub static RUNTIME_SERVICES: Global<MaybeUninit<RuntimeServices>> =
    Global::new(MaybeUninit::zeroed());

static CONFIGURATION: Global<[ConfigurationTable; MAX_CONFIGURATION_TABLES]> = Global::new(
    [ConfigurationTable {
        guid: Guid::new(0, 0, 0, [0; 8]),
        table: ptr::null_mut(),
    }; MAX_CONFIGURATION_TABLES],
);
static CONFIGURATION_COUNT: Global<usize> = Global::new(0);

pub fn system_table() -> *mut SystemTable {
    SYSTEM_TABLE.get()
}

/// Recomputes a table's CRC-32, which covers `size` bytes with the CRC field zeroed.
///
/// # Safety
/// `header` must start a table of at least `size` bytes.
pub unsafe fn seal(header: *mut TableHeader, size: usize) {
    unsafe {
        (*header).header_size = size as u32;
        (*header).crc32 = 0;
        let bytes = core::slice::from_raw_parts(header.cast::<u8>(), size);
        (*header).crc32 = crc32(bytes);
    }
}

pub fn seal_all() {
    unsafe {
        seal(
            ptr::addr_of_mut!((*system_table()).header),
            core::mem::size_of::<SystemTable>(),
        );
        if !SYSTEM_TABLE.get().boot_services.is_null() {
            let boot = BOOT_SERVICES.get().as_mut_ptr();
            seal(ptr::addr_of_mut!((*boot).header), core::mem::size_of::<BootServices>());
        }
        let runtime = RUNTIME_SERVICES.get().as_mut_ptr();
        seal(ptr::addr_of_mut!((*runtime).header), core::mem::size_of::<RuntimeServices>());
    }
}

/// Publishes (or, with a null `table`, removes) a configuration table.
pub fn install_configuration(guid: Guid, table: *mut c_void) -> Status {
    let count = CONFIGURATION_COUNT.get();
    let entries = CONFIGURATION.get();
    match entries[..*count].iter().position(|entry| entry.guid == guid) {
        Some(position) if table.is_null() => {
            entries.copy_within(position + 1..*count, position);
            *count -= 1;
        }
        Some(position) => entries[position].table = table,
        None if table.is_null() => return NOT_FOUND,
        None => {
            if *count == MAX_CONFIGURATION_TABLES {
                return OUT_OF_RESOURCES;
            }
            entries[*count] = ConfigurationTable { guid, table };
            *count += 1;
        }
    }
    let system = SYSTEM_TABLE.get();
    system.number_of_table_entries = *count;
    system.configuration_table = entries.as_mut_ptr();
    seal_all();
    SUCCESS
}

pub extern "efiapi" fn install_configuration_table(guid: *const Guid, table: *mut c_void) -> Status {
    if guid.is_null() {
        return INVALID_PARAMETER;
    }
    install_configuration(unsafe { guid.read_unaligned() }, table)
}
