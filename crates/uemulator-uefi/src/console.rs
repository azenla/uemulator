use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;
use uefi::CStr16;
use uefi::proto::console::text::{Key, ScanCode};

/// Guest output waits here until the next poll, so the firmware is called once per burst
/// rather than once per byte.
static PENDING: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static UTF8_TAIL: Mutex<Vec<u8>> = Mutex::new(Vec::new());

pub fn queue(bytes: &[u8]) {
    PENDING.lock().extend_from_slice(bytes);
}

pub fn flush() {
    let bytes = core::mem::take(&mut *PENDING.lock());
    if !bytes.is_empty() {
        write_now(&bytes);
    }
}

pub fn print(text: &str) {
    flush();
    write_now(text.as_bytes());
}

fn write_now(bytes: &[u8]) {
    let mut tail = UTF8_TAIL.lock();
    tail.extend_from_slice(bytes);
    let valid = match core::str::from_utf8(&tail) {
        Ok(s) => s.len(),
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        // Not UTF-8 at all: pass the bytes through as Latin-1 rather than stall.
        Err(_) => tail.len(),
    };
    let text: String = match core::str::from_utf8(&tail[..valid]) {
        Ok(s) => s.into(),
        Err(_) => tail[..valid].iter().map(|&b| b as char).collect(),
    };
    tail.drain(..valid);
    drop(tail);
    let mut units: Vec<u16> = Vec::with_capacity(512);
    for c in text.chars() {
        if c == '\n' {
            units.push(b'\r' as u16);
        }
        let mut buf = [0u16; 2];
        units.extend_from_slice(c.encode_utf16(&mut buf));
        if units.len() >= 500 {
            emit(&mut units);
        }
    }
    emit(&mut units);
}

fn emit(units: &mut Vec<u16>) {
    // A NUL would end the string early.
    units.retain(|&u| u != 0);
    units.push(0);
    if let Ok(s) = CStr16::from_u16_with_nul(units) {
        uefi::system::with_stdout(|out| {
            let _ = out.output_string_lossy(s);
        });
    }
    units.clear();
}

/// Reads every pending key and returns it as the bytes a terminal would send.
pub fn read_input() -> Vec<u8> {
    let mut out = Vec::new();
    uefi::system::with_stdin(|input| {
        while let Ok(Some(key)) = input.read_key() {
            match key {
                Key::Printable(c) => {
                    let c = char::from(c);
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                Key::Special(code) => out.extend_from_slice(match code {
                    ScanCode::UP => b"\x1b[A",
                    ScanCode::DOWN => b"\x1b[B",
                    ScanCode::RIGHT => b"\x1b[C",
                    ScanCode::LEFT => b"\x1b[D",
                    ScanCode::HOME => b"\x1b[H",
                    ScanCode::END => b"\x1b[F",
                    ScanCode::DELETE => b"\x1b[3~",
                    ScanCode::PAGE_UP => b"\x1b[5~",
                    ScanCode::PAGE_DOWN => b"\x1b[6~",
                    ScanCode::ESCAPE => b"\x1b",
                    _ => b"",
                }),
            }
        }
    });
    out
}
