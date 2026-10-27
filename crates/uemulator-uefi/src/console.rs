use alloc::vec::Vec;
use spin::Mutex;
use uefi::CStr16;
use uefi::proto::console::text::{Key, ScanCode};

use crate::text;

/// Guest output waits here until the next poll, so the firmware is called once per burst
/// rather than once per byte.
static PENDING: Mutex<Vec<u8>> = Mutex::new(Vec::new());
/// Bytes of a UTF-8 sequence cut off at the end of the last write, completed by the next one.
static UTF8_TAIL: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Output callback for the guest console; never calls the firmware.
pub fn queue(bytes: &[u8]) {
    PENDING.lock().extend_from_slice(bytes);
}

/// Writes everything queued to the firmware console.
pub fn flush() {
    let bytes = core::mem::take(&mut *PENDING.lock());
    if !bytes.is_empty() {
        write_now(&bytes);
    }
}

/// Host messages; queued guest output goes first so the two stay in order.
pub fn print(text: &str) {
    flush();
    write_now(text.as_bytes());
}

/// Writes `bytes` (UTF-8, possibly cut inside a sequence) to the firmware console.
fn write_now(bytes: &[u8]) {
    let text = {
        let mut tail = UTF8_TAIL.lock();
        text::take_text(&mut tail, bytes)
    };
    text::to_firmware_units(&text, emit);
}

fn emit(units: &mut Vec<u16>) {
    text::terminate(units);
    if let Ok(s) = CStr16::from_u16_with_nul(units) {
        uefi::system::with_stdout(|out| {
            let _ = out.output_string_lossy(s);
        });
    }
    units.clear();
}

/// Reads every pending key and returns it as the bytes a terminal would send (xterm escape
/// sequences for the special keys, which the guest's tty expects).
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
