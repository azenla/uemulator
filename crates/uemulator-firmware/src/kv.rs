//! Driver for the library's key-value device (`crates/uemulator/src/dev/kv.rs`): four 32-bit
//! registers at the port the handoff names.

use alloc::vec::Vec;

use crate::cpu::{inl, outl};
use crate::handoff::Handoff;
use crate::variable_store::{KEY_PREFIX, Store};

const REGISTER_COMMAND_STATUS: u16 = 0x0;
const REGISTER_DATA: u16 = 0x4;
const REGISTER_LENGTH: u16 = 0x8;
const REGISTER_IDENT: u16 = 0xc;
/// "UKV1".
const IDENT: u32 = 0x3156_4b55;

const COMMAND_SELECT_KEY: u32 = 1;
const COMMAND_SELECT_AUX: u32 = 2;
const COMMAND_GET: u32 = 3;
const COMMAND_SET: u32 = 4;
const COMMAND_DELETE: u32 = 5;
const COMMAND_NEXT: u32 = 6;
const STATUS_OK: u32 = 0;

#[derive(Clone, Copy)]
pub struct KvPort {
    base: u16,
}

/// The device, if the machine has one.
pub fn init(handoff: &Handoff) {
    if handoff.kv_port == 0 {
        return;
    }
    let port = KvPort {
        base: handoff.kv_port,
    };
    if inl(port.base + REGISTER_IDENT) == IDENT {
        crate::variables::attach_store(port);
    }
}

impl KvPort {
    fn select(&self, command: u32, bytes: &[u8]) {
        outl(self.base + REGISTER_COMMAND_STATUS, command);
        outl(self.base + REGISTER_LENGTH, bytes.len() as u32);
        for chunk in bytes.chunks(4) {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            outl(self.base + REGISTER_DATA, u32::from_le_bytes(word));
        }
    }

    fn run(&self, command: u32) -> bool {
        outl(self.base + REGISTER_COMMAND_STATUS, command);
        inl(self.base + REGISTER_COMMAND_STATUS) == STATUS_OK
    }

    fn output(&self) -> Vec<u8> {
        let length = inl(self.base + REGISTER_LENGTH) as usize;
        let mut bytes = Vec::with_capacity(length.div_ceil(4) * 4);
        while bytes.len() < length {
            bytes.extend_from_slice(&inl(self.base + REGISTER_DATA).to_le_bytes());
        }
        bytes.truncate(length);
        bytes
    }
}

impl Store for KvPort {
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.select(COMMAND_SELECT_KEY, key);
        self.run(COMMAND_GET).then(|| self.output())
    }

    fn set(&mut self, key: &[u8], value: &[u8]) -> bool {
        self.select(COMMAND_SELECT_KEY, key);
        self.select(COMMAND_SELECT_AUX, value);
        self.run(COMMAND_SET)
    }

    fn delete(&mut self, key: &[u8]) -> bool {
        self.select(COMMAND_SELECT_KEY, key);
        self.run(COMMAND_DELETE)
    }

    fn next_key(&mut self, after: &[u8]) -> Option<Vec<u8>> {
        self.select(COMMAND_SELECT_KEY, after);
        self.select(COMMAND_SELECT_AUX, KEY_PREFIX);
        self.run(COMMAND_NEXT).then(|| self.output())
    }
}
