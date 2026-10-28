//! UEFI variable semantics over two byte-string stores: a fixed RAM arena for volatile
//! variables (it must live in the firmware image, which the OS keeps, so no heap) and the
//! persistent store the library provides. A key is `var/<guid>/<name>` in UTF-8 and a value is
//! the attributes (`u32`, little-endian) followed by the data.

use alloc::string::String;
use alloc::vec::Vec;

use crate::uefi::variable_attribute::*;
use crate::uefi::{Guid, guids};

pub const KEY_PREFIX: &[u8] = b"var/";
/// What `QueryVariableInfo` reports as the largest variable and the whole store.
pub const MAX_VARIABLE_BYTES: usize = 64 * 1024;
pub const STORAGE_BYTES: u64 = 1 << 20;
const ARENA_BYTES: usize = 192 * 1024;
const SLOTS: usize = 96;

#[derive(Debug, PartialEq, Eq)]
pub enum VariableError {
    NotFound,
    /// The buffer holds fewer bytes than the answer needs.
    BufferTooSmall(usize),
    InvalidParameter,
    WriteProtected,
    OutOfResources,
    SecurityViolation,
    Unsupported,
    DeviceError,
}

/// A byte-string store. `next_key` is the smallest key greater than `after` (any key when
/// `after` is empty) that starts with `KEY_PREFIX`.
pub trait Store {
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>>;
    fn set(&mut self, key: &[u8], value: &[u8]) -> bool;
    fn delete(&mut self, key: &[u8]) -> bool;
    fn next_key(&mut self, after: &[u8]) -> Option<Vec<u8>>;
}

#[derive(Clone, Copy)]
struct Slot {
    used: bool,
    key_offset: u32,
    key_length: u16,
    value_offset: u32,
    value_length: u32,
}

/// Volatile storage in static memory.
pub struct RamStore {
    arena: [u8; ARENA_BYTES],
    used: usize,
    slots: [Slot; SLOTS],
}

impl RamStore {
    pub const fn new() -> Self {
        Self {
            arena: [0; ARENA_BYTES],
            used: 0,
            slots: [Slot {
                used: false,
                key_offset: 0,
                key_length: 0,
                value_offset: 0,
                value_length: 0,
            }; SLOTS],
        }
    }

    fn key_of(&self, slot: &Slot) -> &[u8] {
        &self.arena[slot.key_offset as usize..][..slot.key_length as usize]
    }

    fn find(&self, key: &[u8]) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| slot.used && self.key_of(slot) == key)
    }

    /// Slides every live entry down over the holes deleted ones left.
    fn compact(&mut self) {
        let mut order: Vec<usize> = (0..SLOTS).filter(|&index| self.slots[index].used).collect();
        order.sort_by_key(|&index| self.slots[index].key_offset);
        let mut at = 0usize;
        for index in order {
            let slot = self.slots[index];
            let length = slot.key_length as usize + slot.value_length as usize;
            // Keys and values are stored back to back, key first.
            self.arena
                .copy_within(slot.key_offset as usize..slot.key_offset as usize + length, at);
            self.slots[index].key_offset = at as u32;
            self.slots[index].value_offset = (at + slot.key_length as usize) as u32;
            at += length;
        }
        self.used = at;
    }
}

impl Store for RamStore {
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let slot = self.slots[self.find(key)?];
        Some(self.arena[slot.value_offset as usize..][..slot.value_length as usize].to_vec())
    }

    fn set(&mut self, key: &[u8], value: &[u8]) -> bool {
        let existing = self.find(key);
        let needed = key.len() + value.len();
        // Release the old entry first so replacing a big value does not need room for both.
        if let Some(index) = existing {
            self.slots[index].used = false;
        }
        let free_slot = self.slots.iter().position(|slot| !slot.used);
        if self.used + needed > ARENA_BYTES {
            self.compact();
        }
        let Some(index) = free_slot.filter(|_| self.used + needed <= ARENA_BYTES) else {
            // Out of room: the old value stays, as if the write never happened.
            if let Some(index) = existing {
                self.slots[index].used = true;
            }
            return false;
        };
        let at = self.used;
        self.arena[at..at + key.len()].copy_from_slice(key);
        self.arena[at + key.len()..at + needed].copy_from_slice(value);
        self.slots[index] = Slot {
            used: true,
            key_offset: at as u32,
            key_length: key.len() as u16,
            value_offset: (at + key.len()) as u32,
            value_length: value.len() as u32,
        };
        self.used += needed;
        true
    }

    fn delete(&mut self, key: &[u8]) -> bool {
        match self.find(key) {
            Some(index) => {
                self.slots[index].used = false;
                true
            }
            None => false,
        }
    }

    fn next_key(&mut self, after: &[u8]) -> Option<Vec<u8>> {
        self.slots
            .iter()
            .filter(|slot| slot.used)
            .map(|slot| self.key_of(slot))
            .filter(|key| *key > after && key.starts_with(KEY_PREFIX))
            .min()
            .map(<[u8]>::to_vec)
    }
}

/// Variables the firmware publishes and applications may not change.
const READ_ONLY: [&str; 8] = [
    "SecureBoot",
    "SetupMode",
    "AuditMode",
    "DeployedMode",
    "OsIndicationsSupported",
    "PlatformLangCodes",
    "BootOptionSupport",
    "SignatureSupport",
];

pub struct Variables<P: Store> {
    pub ram: RamStore,
    pub persistent: Option<P>,
}

pub fn key_for(name: &[u16], guid: &Guid) -> Vec<u8> {
    let mut key = KEY_PREFIX.to_vec();
    key.extend_from_slice(&guid.text());
    key.push(b'/');
    key.extend_from_slice(crate::text::to_string(name).as_bytes());
    key
}

/// Inverse of `key_for`.
pub fn split_key(key: &[u8]) -> Option<(Vec<u16>, Guid)> {
    let rest = key.strip_prefix(KEY_PREFIX)?;
    let guid = Guid::parse(rest.get(..36)?)?;
    let name = core::str::from_utf8(rest.get(37..)?).ok()?;
    Some((name.encode_utf16().collect(), guid))
}

fn split_value(value: &[u8]) -> Option<(u32, &[u8])> {
    Some((u32::from_le_bytes(value.get(..4)?.try_into().ok()?), &value[4..]))
}

impl<P: Store> Variables<P> {
    pub const fn new() -> Self {
        Self {
            ram: RamStore::new(),
            persistent: None,
        }
    }

    fn load(&mut self, key: &[u8]) -> Option<(bool, Vec<u8>)> {
        if let Some(value) = self.ram.get(key) {
            return Some((false, value));
        }
        self.persistent
            .as_mut()
            .and_then(|store| store.get(key))
            .map(|value| (true, value))
    }

    /// At runtime only variables marked `RUNTIME_ACCESS` exist for the caller.
    fn visible(attributes: u32, runtime: bool) -> bool {
        !runtime || attributes & RUNTIME_ACCESS != 0
    }

    pub fn get(
        &mut self,
        name: &[u16],
        guid: &Guid,
        runtime: bool,
    ) -> Result<(u32, Vec<u8>), VariableError> {
        if name.is_empty() {
            return Err(VariableError::InvalidParameter);
        }
        let (_, value) = self.load(&key_for(name, guid)).ok_or(VariableError::NotFound)?;
        let (attributes, data) = split_value(&value).ok_or(VariableError::DeviceError)?;
        if !Self::visible(attributes, runtime) {
            return Err(VariableError::NotFound);
        }
        Ok((attributes, data.to_vec()))
    }

    /// Stores a variable the way the firmware itself would, past every check an application
    /// call goes through.
    pub fn set_internal(&mut self, name: &str, guid: &Guid, attributes: u32, data: &[u8]) {
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut value = attributes.to_le_bytes().to_vec();
        value.extend_from_slice(data);
        self.ram.set(&key_for(&units, guid), &value);
    }

    pub fn set(
        &mut self,
        name: &[u16],
        guid: &Guid,
        attributes: u32,
        data: &[u8],
        runtime: bool,
    ) -> Result<(), VariableError> {
        if name.is_empty() || name[0] == 0 {
            return Err(VariableError::InvalidParameter);
        }
        if attributes & (AUTHENTICATED_WRITE_ACCESS | TIME_BASED_AUTHENTICATED_WRITE_ACCESS) != 0 {
            return Err(VariableError::SecurityViolation);
        }
        if attributes & HARDWARE_ERROR_RECORD != 0 {
            return Err(VariableError::Unsupported);
        }
        let append = attributes & APPEND_WRITE != 0;
        let attributes = attributes & !APPEND_WRITE;
        if data.len() > MAX_VARIABLE_BYTES {
            return Err(VariableError::OutOfResources);
        }
        let key = key_for(name, guid);
        if *guid == guids::GLOBAL_VARIABLE && READ_ONLY.contains(&describe(name).as_str()) {
            return Err(VariableError::WriteProtected);
        }
        let existing = self.load(&key);
        let existing_parts = existing
            .as_ref()
            .and_then(|(_, value)| split_value(value))
            .map(|(old_attributes, old_data)| (old_attributes, old_data.to_vec()));
        // A variable hidden from runtime callers does not exist for them.
        let existing_parts = existing_parts.filter(|(old, _)| Self::visible(*old, runtime));
        // Zero attributes or no data deletes; appending nothing does nothing.
        let access = attributes & (BOOTSERVICE_ACCESS | RUNTIME_ACCESS);
        if !append && (data.is_empty() || access == 0) {
            return self.remove(&key, existing_parts.is_some());
        }
        if append && data.is_empty() {
            return Ok(());
        }
        if access == 0 {
            return Err(VariableError::InvalidParameter);
        }
        // Runtime access needs boot-service access too (UEFI 8.2).
        if attributes & RUNTIME_ACCESS != 0 && attributes & BOOTSERVICE_ACCESS == 0 {
            return Err(VariableError::InvalidParameter);
        }
        if runtime && attributes & RUNTIME_ACCESS == 0 {
            return Err(VariableError::InvalidParameter);
        }
        let mut combined = Vec::new();
        if let Some((old_attributes, old_data)) = &existing_parts {
            if *old_attributes != attributes {
                return Err(VariableError::InvalidParameter);
            }
            if append {
                combined = old_data.clone();
            }
        }
        combined.extend_from_slice(data);
        if combined.len() > MAX_VARIABLE_BYTES {
            return Err(VariableError::OutOfResources);
        }
        let mut value = attributes.to_le_bytes().to_vec();
        value.extend_from_slice(&combined);
        let persistent = attributes & NON_VOLATILE != 0 && self.persistent.is_some();
        let stored = if persistent {
            self.ram.delete(&key);
            self.persistent.as_mut().unwrap().set(&key, &value)
        } else {
            if let Some(store) = self.persistent.as_mut() {
                store.delete(&key);
            }
            self.ram.set(&key, &value)
        };
        stored.then_some(()).ok_or(VariableError::OutOfResources)
    }

    fn remove(&mut self, key: &[u8], existed: bool) -> Result<(), VariableError> {
        if !existed {
            return Err(VariableError::NotFound);
        }
        self.ram.delete(key);
        if let Some(store) = self.persistent.as_mut() {
            store.delete(key);
        }
        Ok(())
    }

    /// The variable after `(name, guid)` in enumeration order: RAM keys, then persistent ones,
    /// each ascending. An empty name starts the walk; `NotFound` ends it.
    pub fn next(
        &mut self,
        name: &[u16],
        guid: &Guid,
        runtime: bool,
    ) -> Result<(Vec<u16>, Guid), VariableError> {
        let mut after = if name.is_empty() { Vec::new() } else { key_for(name, guid) };
        let mut in_ram = true;
        if !name.is_empty() {
            match self.load(&after) {
                Some((persistent, _)) => in_ram = !persistent,
                None => return Err(VariableError::InvalidParameter),
            }
        }
        loop {
            let candidate = if in_ram {
                match self.ram.next_key(&after) {
                    Some(key) => Some(key),
                    None => {
                        in_ram = false;
                        after = Vec::new();
                        continue;
                    }
                }
            } else {
                self.persistent.as_mut().and_then(|store| store.next_key(&after))
            };
            let key = candidate.ok_or(VariableError::NotFound)?;
            let visible = self
                .load(&key)
                .and_then(|(_, value)| split_value(&value).map(|(attributes, _)| attributes))
                .is_some_and(|attributes| Self::visible(attributes, runtime));
            match split_key(&key) {
                Some((units, guid)) if visible => return Ok((units, guid)),
                _ => after = key,
            }
        }
    }

    /// (maximum storage, remaining storage, maximum variable size) for `QueryVariableInfo`.
    pub fn info(&mut self) -> (u64, u64, u64) {
        let mut used = 0u64;
        for in_ram in [true, false] {
            let mut after = Vec::new();
            loop {
                let next = if in_ram {
                    self.ram.next_key(&after)
                } else {
                    self.persistent.as_mut().and_then(|store| store.next_key(&after))
                };
                let Some(key) = next else { break };
                used += self.load(&key).map_or(0, |(_, value)| (key.len() + value.len()) as u64);
                after = key;
            }
        }
        (STORAGE_BYTES, STORAGE_BYTES.saturating_sub(used), MAX_VARIABLE_BYTES as u64)
    }
}

/// `"Boot0001"`-style names for the boot manager and tests.
pub fn name_units(name: &str) -> Vec<u16> {
    name.encode_utf16().collect()
}

pub fn describe(name: &[u16]) -> String {
    crate::text::to_string(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::collections::BTreeMap;

    #[derive(Default)]
    struct Memory(BTreeMap<Vec<u8>, Vec<u8>>);

    impl Store for Memory {
        fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
            self.0.get(key).cloned()
        }
        fn set(&mut self, key: &[u8], value: &[u8]) -> bool {
            self.0.insert(key.to_vec(), value.to_vec());
            true
        }
        fn delete(&mut self, key: &[u8]) -> bool {
            self.0.remove(key).is_some()
        }
        fn next_key(&mut self, after: &[u8]) -> Option<Vec<u8>> {
            self.0
                .keys()
                .find(|key| key.as_slice() > after && key.starts_with(KEY_PREFIX))
                .cloned()
        }
    }

    const NV_BS_RT: u32 = NON_VOLATILE | BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
    const CUSTOM: Guid = Guid::new(1, 2, 3, [4; 8]);

    fn variables() -> Box<Variables<Memory>> {
        let mut variables = Box::new(Variables::new());
        variables.persistent = Some(Memory::default());
        variables
    }

    fn name(text: &str) -> Vec<u16> {
        name_units(text)
    }

    #[test]
    fn nonvolatile_variables_go_to_the_persistent_store() {
        let mut variables = variables();
        variables.set(&name("BootOrder"), &guids::GLOBAL_VARIABLE, NV_BS_RT, &[1, 0], false).unwrap();
        let (attributes, data) = variables.get(&name("BootOrder"), &guids::GLOBAL_VARIABLE, false).unwrap();
        assert_eq!((attributes, data), (NV_BS_RT, vec![1, 0]));
        assert_eq!(variables.persistent.as_ref().unwrap().0.len(), 1);
        assert!(variables.ram.next_key(&[]).is_none());
    }

    #[test]
    fn volatile_variables_stay_in_ram() {
        let mut variables = variables();
        let attributes = BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
        variables.set(&name("Temp"), &CUSTOM, attributes, b"x", false).unwrap();
        assert!(variables.persistent.as_ref().unwrap().0.is_empty());
        assert_eq!(variables.get(&name("Temp"), &CUSTOM, true).unwrap().1, b"x");
    }

    #[test]
    fn nonvolatile_without_a_store_falls_back_to_ram() {
        let mut variables: Box<Variables<Memory>> = Box::new(Variables::new());
        variables.set(&name("Kept"), &CUSTOM, NV_BS_RT, b"1", false).unwrap();
        assert_eq!(variables.get(&name("Kept"), &CUSTOM, false).unwrap().1, b"1");
    }

    #[test]
    fn boot_service_only_variables_vanish_at_runtime() {
        let mut variables = variables();
        variables.set(&name("Early"), &CUSTOM, BOOTSERVICE_ACCESS, b"1", false).unwrap();
        assert!(variables.get(&name("Early"), &CUSTOM, false).is_ok());
        assert_eq!(variables.get(&name("Early"), &CUSTOM, true), Err(VariableError::NotFound));
        assert_eq!(
            variables.next(&[], &Guid::default(), true),
            Err(VariableError::NotFound)
        );
    }

    #[test]
    fn writes_check_attributes() {
        let mut variables = variables();
        let set = |variables: &mut Variables<Memory>, attributes, data: &[u8], runtime| {
            variables.set(&name("V"), &CUSTOM, attributes, data, runtime)
        };
        assert_eq!(set(&mut variables, RUNTIME_ACCESS, b"x", false), Err(VariableError::InvalidParameter));
        assert_eq!(set(&mut variables, NV_BS_RT, b"x", false), Ok(()));
        assert_eq!(set(&mut variables, BOOTSERVICE_ACCESS, b"x", false), Err(VariableError::InvalidParameter), "attributes cannot change");
        assert_eq!(set(&mut variables, NON_VOLATILE | BOOTSERVICE_ACCESS, b"x", true), Err(VariableError::InvalidParameter), "no runtime access at runtime");
        assert_eq!(
            set(&mut variables, NV_BS_RT | TIME_BASED_AUTHENTICATED_WRITE_ACCESS, b"x", false),
            Err(VariableError::SecurityViolation)
        );
        assert_eq!(
            variables.set(&[], &CUSTOM, NV_BS_RT, b"x", false),
            Err(VariableError::InvalidParameter)
        );
    }

    #[test]
    fn empty_data_deletes() {
        let mut variables = variables();
        variables.set(&name("V"), &CUSTOM, NV_BS_RT, b"x", false).unwrap();
        variables.set(&name("V"), &CUSTOM, NV_BS_RT, b"", false).unwrap();
        assert_eq!(variables.get(&name("V"), &CUSTOM, false), Err(VariableError::NotFound));
        assert_eq!(variables.set(&name("V"), &CUSTOM, NV_BS_RT, b"", false), Err(VariableError::NotFound));
        variables.set(&name("W"), &CUSTOM, NV_BS_RT, b"x", false).unwrap();
        variables.set(&name("W"), &CUSTOM, 0, b"x", false).unwrap();
        assert_eq!(variables.get(&name("W"), &CUSTOM, false), Err(VariableError::NotFound));
    }

    #[test]
    fn append_extends_the_data() {
        let mut variables = variables();
        variables.set(&name("V"), &CUSTOM, NV_BS_RT, b"ab", false).unwrap();
        variables.set(&name("V"), &CUSTOM, NV_BS_RT | APPEND_WRITE, b"cd", false).unwrap();
        assert_eq!(variables.get(&name("V"), &CUSTOM, false).unwrap().1, b"abcd");
        variables.set(&name("New"), &CUSTOM, NV_BS_RT | APPEND_WRITE, b"x", false).unwrap();
        assert_eq!(variables.get(&name("New"), &CUSTOM, false).unwrap().1, b"x");
    }

    #[test]
    fn the_firmware_owned_variables_are_read_only() {
        let mut variables = variables();
        variables.set_internal("SecureBoot", &guids::GLOBAL_VARIABLE, BOOTSERVICE_ACCESS | RUNTIME_ACCESS, &[0]);
        assert_eq!(variables.get(&name("SecureBoot"), &guids::GLOBAL_VARIABLE, false).unwrap().1, [0]);
        assert_eq!(
            variables.set(&name("SecureBoot"), &guids::GLOBAL_VARIABLE, BOOTSERVICE_ACCESS | RUNTIME_ACCESS, &[1], false),
            Err(VariableError::WriteProtected)
        );
        // The same name under another vendor is an ordinary variable.
        assert!(variables.set(&name("SecureBoot"), &CUSTOM, NV_BS_RT, &[1], false).is_ok());
    }

    #[test]
    fn enumeration_visits_every_variable_once_ram_first() {
        let mut variables = variables();
        variables.set_internal("B", &CUSTOM, BOOTSERVICE_ACCESS | RUNTIME_ACCESS, b"1");
        variables.set_internal("A", &CUSTOM, BOOTSERVICE_ACCESS | RUNTIME_ACCESS, b"1");
        variables.set(&name("Z"), &CUSTOM, NV_BS_RT, b"1", false).unwrap();
        variables.set(&name("Y"), &CUSTOM, NV_BS_RT, b"1", false).unwrap();
        let mut seen = Vec::new();
        let (mut current, mut guid) = (Vec::new(), Guid::default());
        while let Ok((next, next_guid)) = variables.next(&current, &guid, false) {
            seen.push(describe(&next));
            (current, guid) = (next, next_guid);
        }
        assert_eq!(seen, ["A", "B", "Y", "Z"]);
        assert_eq!(
            variables.next(&name("Nope"), &CUSTOM, false),
            Err(VariableError::InvalidParameter)
        );
    }

    #[test]
    fn info_counts_what_is_stored() {
        let mut variables = variables();
        let (total, remaining_empty, maximum) = variables.info();
        variables.set(&name("V"), &CUSTOM, NV_BS_RT, &[0; 1000], false).unwrap();
        let (_, remaining, _) = variables.info();
        assert_eq!(total, STORAGE_BYTES);
        assert!(remaining < remaining_empty - 1000);
        assert_eq!(maximum, MAX_VARIABLE_BYTES as u64);
    }

    #[test]
    fn ram_store_compacts_when_full_and_refuses_when_really_full() {
        let mut store = Box::new(RamStore::new());
        let value = vec![7u8; 60_000];
        for round in 0..20u8 {
            let key = [KEY_PREFIX, &[b'a' + (round % 2)]].concat();
            assert!(store.set(&key, &value), "round {round}");
        }
        assert_eq!(store.get(b"var/a").map(|v| v.len()), Some(60_000));
        assert!(store.set(b"var/c", &value));
        assert!(!store.set(b"var/d", &vec![0u8; ARENA_BYTES]), "cannot fit");
        assert_eq!(store.get(b"var/c").map(|v| v.len()), Some(60_000), "the others survive a refused write");
    }

    #[test]
    fn keys_split_back_into_name_and_guid() {
        let key = key_for(&name("Boot0001"), &CUSTOM);
        assert_eq!(split_key(&key), Some((name("Boot0001"), CUSTOM)));
        assert_eq!(split_key(b"var/short"), None);
        assert_eq!(split_key(b"other/x"), None);
    }
}
