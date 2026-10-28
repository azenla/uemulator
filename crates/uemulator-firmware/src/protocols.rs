//! The handle database: handles, the protocol interfaces installed on them, who has them open,
//! and the boot services that query it.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;

use crate::devicepath::DevicePath;
use crate::events;
use crate::memory;
use crate::sync::Global;
use crate::uefi::*;

struct OpenRecord {
    agent: Handle,
    controller: Handle,
    attributes: u32,
    count: u32,
}

struct Protocol {
    guid: Guid,
    interface: *mut c_void,
    opens: Vec<OpenRecord>,
}

struct HandleData {
    protocols: Vec<Protocol>,
}

struct Registration {
    guid: Guid,
    event: Event,
    /// Handles that got the protocol since the last `LocateHandle(ByRegisterNotify)`.
    newly_installed: Vec<Handle>,
}

struct Database {
    handles: Vec<*mut HandleData>,
    // Boxed: the registration token handed to applications is the entry's address.
    #[allow(clippy::vec_box)]
    registrations: Vec<Box<Registration>>,
}

static DATABASE: Global<Database> = Global::new(Database {
    handles: Vec::new(),
    registrations: Vec::new(),
});

fn known(handle: Handle) -> bool {
    DATABASE.get().handles.contains(&handle.cast())
}

fn data(handle: Handle) -> &'static mut HandleData {
    unsafe { &mut *handle.cast::<HandleData>() }
}

/// The interface of `guid` on `handle`, if it has one.
pub fn interface_of(handle: Handle, guid: &Guid) -> Option<*mut c_void> {
    if !known(handle) {
        return None;
    }
    data(handle)
        .protocols
        .iter()
        .find(|protocol| protocol.guid == *guid)
        .map(|protocol| protocol.interface)
}

/// Every handle that carries `guid`, in the order they were created.
pub fn handles_with(guid: &Guid) -> Vec<Handle> {
    DATABASE
        .get()
        .handles
        .iter()
        .copied()
        .filter(|&handle| unsafe {
            (*handle)
                .protocols
                .iter()
                .any(|protocol| protocol.guid == *guid)
        })
        .map(|handle| handle.cast())
        .collect()
}

pub fn all_handles() -> Vec<Handle> {
    DATABASE
        .get()
        .handles
        .iter()
        .map(|&handle| handle.cast())
        .collect()
}

/// Installs `interface` on `handle`, or on a new handle when it is null.
pub fn install(handle: Handle, guid: Guid, interface: *mut c_void) -> Result<Handle, Status> {
    let database = DATABASE.get();
    let handle = if handle.is_null() {
        let created = Box::into_raw(Box::new(HandleData {
            protocols: Vec::new(),
        }));
        database.handles.push(created);
        created.cast()
    } else if known(handle) {
        handle
    } else {
        return Err(INVALID_PARAMETER);
    };
    if data(handle)
        .protocols
        .iter()
        .any(|protocol| protocol.guid == guid)
    {
        return Err(INVALID_PARAMETER);
    }
    data(handle).protocols.push(Protocol {
        guid,
        interface,
        opens: Vec::new(),
    });
    for registration in database
        .registrations
        .iter_mut()
        .filter(|registration| registration.guid == guid)
    {
        registration.newly_installed.push(handle);
        events::signal(registration.event);
    }
    Ok(handle)
}

fn uninstall(handle: Handle, guid: &Guid, interface: *mut c_void) -> Status {
    if !known(handle) {
        return INVALID_PARAMETER;
    }
    let protocols = &mut data(handle).protocols;
    let Some(position) = protocols
        .iter()
        .position(|protocol| protocol.guid == *guid && protocol.interface == interface)
    else {
        return NOT_FOUND;
    };
    // An interface somebody opened by driver cannot be pulled out from under them.
    if protocols[position]
        .opens
        .iter()
        .any(|open| open.attributes & OPEN_PROTOCOL_BY_DRIVER != 0)
    {
        return ACCESS_DENIED;
    }
    protocols.remove(position);
    SUCCESS
}

unsafe fn guid_arg(guid: *const Guid) -> Option<Guid> {
    (!guid.is_null()).then(|| unsafe { guid.read_unaligned() })
}

extern "efiapi" fn install_protocol_interface(
    handle: *mut Handle,
    guid: *const Guid,
    interface_type: u32,
    interface: *mut c_void,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if handle.is_null() || interface_type != NATIVE_INTERFACE {
        return INVALID_PARAMETER;
    }
    match install(unsafe { handle.read() }, guid, interface) {
        Ok(installed) => {
            unsafe { handle.write(installed) };
            SUCCESS
        }
        Err(status) => status,
    }
}

extern "efiapi" fn reinstall_protocol_interface(
    handle: Handle,
    guid: *const Guid,
    old: *mut c_void,
    new: *mut c_void,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if !known(handle) {
        return INVALID_PARAMETER;
    }
    match data(handle)
        .protocols
        .iter_mut()
        .find(|protocol| protocol.guid == guid && protocol.interface == old)
    {
        Some(protocol) => {
            protocol.interface = new;
            SUCCESS
        }
        None => NOT_FOUND,
    }
}

extern "efiapi" fn uninstall_protocol_interface(
    handle: Handle,
    guid: *const Guid,
    interface: *mut c_void,
) -> Status {
    match unsafe { guid_arg(guid) } {
        Some(guid) => uninstall(handle, &guid, interface),
        None => INVALID_PARAMETER,
    }
}

extern "efiapi" fn handle_protocol(
    handle: Handle,
    guid: *const Guid,
    interface: *mut *mut c_void,
) -> Status {
    open_protocol(
        handle,
        guid,
        interface,
        ptr::null_mut(),
        ptr::null_mut(),
        OPEN_PROTOCOL_BY_HANDLE_PROTOCOL,
    )
}

extern "efiapi" fn open_protocol(
    handle: Handle,
    guid: *const Guid,
    interface: *mut *mut c_void,
    agent: Handle,
    controller: Handle,
    attributes: u32,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if !known(handle) {
        return INVALID_PARAMETER;
    }
    let test = attributes == OPEN_PROTOCOL_TEST_PROTOCOL;
    if interface.is_null() && !test {
        return INVALID_PARAMETER;
    }
    let Some(protocol) = data(handle)
        .protocols
        .iter_mut()
        .find(|protocol| protocol.guid == guid)
    else {
        if !interface.is_null() {
            unsafe { interface.write(ptr::null_mut()) };
        }
        return UNSUPPORTED;
    };
    if !test && !interface.is_null() {
        unsafe { interface.write(protocol.interface) };
    }
    // The usage records matter to drivers deciding whom to bind; applications only need the
    // interface, so GET/BY_HANDLE are not recorded.
    if attributes
        & (OPEN_PROTOCOL_BY_DRIVER | OPEN_PROTOCOL_EXCLUSIVE | OPEN_PROTOCOL_BY_CHILD_CONTROLLER)
        != 0
    {
        match protocol.opens.iter_mut().find(|open| {
            open.agent == agent && open.controller == controller && open.attributes == attributes
        }) {
            Some(open) => open.count += 1,
            None => protocol.opens.push(OpenRecord {
                agent,
                controller,
                attributes,
                count: 1,
            }),
        }
    }
    SUCCESS
}

extern "efiapi" fn close_protocol(
    handle: Handle,
    guid: *const Guid,
    agent: Handle,
    controller: Handle,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if !known(handle) {
        return INVALID_PARAMETER;
    }
    let Some(protocol) = data(handle)
        .protocols
        .iter_mut()
        .find(|protocol| protocol.guid == guid)
    else {
        return NOT_FOUND;
    };
    // Handles opened without a record (plain GET/BY_HANDLE) close successfully too.
    protocol
        .opens
        .retain(|open| !(open.agent == agent && open.controller == controller));
    SUCCESS
}

extern "efiapi" fn open_protocol_information(
    handle: Handle,
    guid: *const Guid,
    buffer: *mut *mut OpenProtocolInformation,
    count: *mut usize,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if !known(handle) || buffer.is_null() || count.is_null() {
        return INVALID_PARAMETER;
    }
    let Some(protocol) = data(handle)
        .protocols
        .iter()
        .find(|protocol| protocol.guid == guid)
    else {
        return NOT_FOUND;
    };
    let size = protocol.opens.len() * core::mem::size_of::<OpenProtocolInformation>();
    let array = memory::pool_allocate(memory_type::BOOT_SERVICES_DATA, size.max(1));
    if array.is_null() {
        return OUT_OF_RESOURCES;
    }
    for (index, open) in protocol.opens.iter().enumerate() {
        unsafe {
            array
                .cast::<OpenProtocolInformation>()
                .add(index)
                .write(OpenProtocolInformation {
                    agent_handle: open.agent,
                    controller_handle: open.controller,
                    attributes: open.attributes,
                    open_count: open.count,
                });
        }
    }
    unsafe {
        buffer.write(array.cast());
        count.write(protocol.opens.len());
    }
    SUCCESS
}

extern "efiapi" fn protocols_per_handle(
    handle: Handle,
    buffer: *mut *mut *mut Guid,
    count: *mut usize,
) -> Status {
    if !known(handle) || buffer.is_null() || count.is_null() {
        return INVALID_PARAMETER;
    }
    let protocols = &mut data(handle).protocols;
    let array = memory::pool_allocate(
        memory_type::BOOT_SERVICES_DATA,
        (protocols.len() * core::mem::size_of::<*mut Guid>()).max(1),
    )
    .cast::<*mut Guid>();
    if array.is_null() {
        return OUT_OF_RESOURCES;
    }
    for (index, protocol) in protocols.iter_mut().enumerate() {
        unsafe { array.add(index).write(&mut protocol.guid) };
    }
    unsafe {
        buffer.write(array);
        count.write(protocols.len());
    }
    SUCCESS
}

/// The handles a `LocateHandle` search selects.
fn search(search_type: u32, guid: *const Guid, key: *mut c_void) -> Result<Vec<Handle>, Status> {
    match search_type {
        ALL_HANDLES => Ok(all_handles()),
        BY_REGISTER_NOTIFY => {
            let registration = DATABASE
                .get()
                .registrations
                .iter_mut()
                .find(|registration| {
                    ptr::eq(
                        &***registration as *const Registration as *const c_void,
                        key,
                    )
                })
                .ok_or(INVALID_PARAMETER)?;
            Ok(core::mem::take(&mut registration.newly_installed))
        }
        BY_PROTOCOL => unsafe { guid_arg(guid) }
            .map(|guid| handles_with(&guid))
            .ok_or(INVALID_PARAMETER),
        _ => Err(INVALID_PARAMETER),
    }
}

extern "efiapi" fn locate_handle(
    search_type: u32,
    guid: *const Guid,
    key: *mut c_void,
    size: *mut usize,
    buffer: *mut Handle,
) -> Status {
    if size.is_null() {
        return INVALID_PARAMETER;
    }
    let found = match search(search_type, guid, key) {
        Ok(found) => found,
        Err(status) => return status,
    };
    if found.is_empty() {
        return NOT_FOUND;
    }
    let needed = found.len() * core::mem::size_of::<Handle>();
    let available = unsafe { size.read() };
    unsafe { size.write(needed) };
    if available < needed {
        return BUFFER_TOO_SMALL;
    }
    if buffer.is_null() {
        return INVALID_PARAMETER;
    }
    for (index, handle) in found.iter().enumerate() {
        unsafe { buffer.add(index).write(*handle) };
    }
    SUCCESS
}

extern "efiapi" fn locate_handle_buffer(
    search_type: u32,
    guid: *const Guid,
    key: *mut c_void,
    count: *mut usize,
    buffer: *mut *mut Handle,
) -> Status {
    if count.is_null() || buffer.is_null() {
        return INVALID_PARAMETER;
    }
    let found = match search(search_type, guid, key) {
        Ok(found) => found,
        Err(status) => return status,
    };
    if found.is_empty() {
        return NOT_FOUND;
    }
    let array = memory::pool_allocate(
        memory_type::BOOT_SERVICES_DATA,
        found.len() * core::mem::size_of::<Handle>(),
    )
    .cast::<Handle>();
    if array.is_null() {
        return OUT_OF_RESOURCES;
    }
    for (index, handle) in found.iter().enumerate() {
        unsafe { array.add(index).write(*handle) };
    }
    unsafe {
        buffer.write(array);
        count.write(found.len());
    }
    SUCCESS
}

extern "efiapi" fn locate_protocol(
    guid: *const Guid,
    registration: *mut c_void,
    interface: *mut *mut c_void,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if interface.is_null() {
        return INVALID_PARAMETER;
    }
    let mut handles = if registration.is_null() {
        handles_with(&guid)
    } else {
        match search(BY_REGISTER_NOTIFY, &guid, registration) {
            Ok(found) => found,
            Err(status) => return status,
        }
    };
    handles.retain(|&handle| interface_of(handle, &guid).is_some());
    match handles
        .first()
        .and_then(|&handle| interface_of(handle, &guid))
    {
        Some(found) => {
            unsafe { interface.write(found) };
            SUCCESS
        }
        None => NOT_FOUND,
    }
}

extern "efiapi" fn locate_device_path(
    guid: *const Guid,
    path: *mut *mut DevicePathNode,
    device: *mut Handle,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if path.is_null() || device.is_null() || unsafe { path.read() }.is_null() {
        return INVALID_PARAMETER;
    }
    let Some(wanted) = (unsafe { DevicePath::from_raw(path.read().cast()) }) else {
        return INVALID_PARAMETER;
    };
    // The handle whose device path is the longest prefix of `wanted`.
    let mut best: Option<(Handle, usize)> = None;
    for handle in handles_with(&guid) {
        let Some(candidate) = interface_of(handle, &guids::DEVICE_PATH)
            .and_then(|pointer| unsafe { DevicePath::from_raw(pointer.cast()) })
        else {
            continue;
        };
        let length = candidate.as_bytes().len() - 4;
        if wanted.starts_with(&candidate) && best.is_none_or(|(_, longest)| length > longest) {
            best = Some((handle, length));
        }
    }
    let Some((handle, length)) = best else {
        return NOT_FOUND;
    };
    unsafe {
        device.write(handle);
        path.write(path.read().cast::<u8>().add(length).cast());
    }
    SUCCESS
}

extern "efiapi" fn register_protocol_notify(
    guid: *const Guid,
    event: Event,
    registration: *mut *mut c_void,
) -> Status {
    let Some(guid) = (unsafe { guid_arg(guid) }) else {
        return INVALID_PARAMETER;
    };
    if event.is_null() || registration.is_null() {
        return INVALID_PARAMETER;
    }
    let entry = Box::new(Registration {
        guid,
        event,
        newly_installed: Vec::new(),
    });
    unsafe { registration.write((&*entry as *const Registration).cast_mut().cast()) };
    DATABASE.get().registrations.push(entry);
    SUCCESS
}

/// The most interface/GUID pairs `InstallMultipleProtocolInterfaces` is read for. The function is
/// variadic in the specification and Rust cannot define one, so it is declared with this many
/// fixed parameters: the extra ones read stack slots the caller did not fill, which is harmless
/// because the list ends at the first null GUID.
const MULTIPLE_PAIRS: usize = 8;

#[allow(clippy::too_many_arguments)]
extern "efiapi" fn install_multiple(
    handle: *mut Handle,
    g0: *const Guid,
    i0: *mut c_void,
    g1: *const Guid,
    i1: *mut c_void,
    g2: *const Guid,
    i2: *mut c_void,
    g3: *const Guid,
    i3: *mut c_void,
    g4: *const Guid,
    i4: *mut c_void,
    g5: *const Guid,
    i5: *mut c_void,
    g6: *const Guid,
    i6: *mut c_void,
    g7: *const Guid,
    i7: *mut c_void,
) -> Status {
    let pairs = [
        (g0, i0),
        (g1, i1),
        (g2, i2),
        (g3, i3),
        (g4, i4),
        (g5, i5),
        (g6, i6),
        (g7, i7),
    ];
    debug_assert_eq!(pairs.len(), MULTIPLE_PAIRS);
    if handle.is_null() {
        return INVALID_PARAMETER;
    }
    let mut done: Vec<(Guid, *mut c_void)> = Vec::new();
    for (guid, interface) in pairs {
        let Some(guid) = (unsafe { guid_arg(guid) }) else {
            break;
        };
        match install(unsafe { handle.read() }, guid, interface) {
            Ok(installed) => {
                unsafe { handle.write(installed) };
                done.push((guid, interface));
            }
            Err(status) => {
                for (guid, interface) in done {
                    uninstall(unsafe { handle.read() }, &guid, interface);
                }
                return status;
            }
        }
    }
    SUCCESS
}

#[allow(clippy::too_many_arguments)]
extern "efiapi" fn uninstall_multiple(
    handle: Handle,
    g0: *const Guid,
    i0: *mut c_void,
    g1: *const Guid,
    i1: *mut c_void,
    g2: *const Guid,
    i2: *mut c_void,
    g3: *const Guid,
    i3: *mut c_void,
    g4: *const Guid,
    i4: *mut c_void,
    g5: *const Guid,
    i5: *mut c_void,
    g6: *const Guid,
    i6: *mut c_void,
    g7: *const Guid,
    i7: *mut c_void,
) -> Status {
    let pairs = [
        (g0, i0),
        (g1, i1),
        (g2, i2),
        (g3, i3),
        (g4, i4),
        (g5, i5),
        (g6, i6),
        (g7, i7),
    ];
    for (guid, interface) in pairs {
        let Some(guid) = (unsafe { guid_arg(guid) }) else {
            break;
        };
        let status = uninstall(handle, &guid, interface);
        if status != SUCCESS {
            return status;
        }
    }
    SUCCESS
}

/// No drivers exist to connect, which the specification reports as `NOT_FOUND`; loaders call it
/// on every handle and ignore the answer.
extern "efiapi" fn connect_controller(
    _: Handle,
    _: *mut Handle,
    _: *mut DevicePathNode,
    _: u8,
) -> Status {
    NOT_FOUND
}

extern "efiapi" fn disconnect_controller(_: Handle, _: Handle, _: Handle) -> Status {
    SUCCESS
}

pub fn install_services(services: &mut BootServices) {
    services.install_protocol_interface = install_protocol_interface;
    services.reinstall_protocol_interface = reinstall_protocol_interface;
    services.uninstall_protocol_interface = uninstall_protocol_interface;
    services.handle_protocol = handle_protocol;
    services.register_protocol_notify = register_protocol_notify;
    services.locate_handle = locate_handle;
    services.locate_device_path = locate_device_path;
    services.connect_controller = connect_controller;
    services.disconnect_controller = disconnect_controller;
    services.open_protocol = open_protocol;
    services.close_protocol = close_protocol;
    services.open_protocol_information = open_protocol_information;
    services.protocols_per_handle = protocols_per_handle;
    services.locate_handle_buffer = locate_handle_buffer;
    services.locate_protocol = locate_protocol;
    services.install_multiple_protocol_interfaces = install_multiple as *const () as usize;
    services.uninstall_multiple_protocol_interfaces = uninstall_multiple as *const () as usize;
}
