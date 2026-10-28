//! The 16550 at COM1: the firmware's console and its debug output.

use core::fmt;
use core::sync::atomic::{AtomicU16, Ordering};

use crate::cpu::{inb, outb};

const REGISTER_DATA: u16 = 0;
const REGISTER_LINE_STATUS: u16 = 5;
const LINE_STATUS_DATA_READY: u8 = 1 << 0;
const LINE_STATUS_TRANSMIT_EMPTY: u8 = 1 << 5;
/// The emulated UART is always ready; a real one might not be, and a stuck line must not hang
/// a panic message.
const TRANSMIT_SPINS: u32 = 100_000;

static PORT: AtomicU16 = AtomicU16::new(0x3f8);

pub fn init(port: u16) {
    PORT.store(port, Ordering::Relaxed);
}

pub fn write_byte(byte: u8) {
    let port = PORT.load(Ordering::Relaxed);
    for _ in 0..TRANSMIT_SPINS {
        if inb(port + REGISTER_LINE_STATUS) & LINE_STATUS_TRANSMIT_EMPTY != 0 {
            break;
        }
    }
    outb(port + REGISTER_DATA, byte);
}

pub fn read_byte() -> Option<u8> {
    let port = PORT.load(Ordering::Relaxed);
    (inb(port + REGISTER_LINE_STATUS) & LINE_STATUS_DATA_READY != 0)
        .then(|| inb(port + REGISTER_DATA))
}

pub struct Serial;

impl fmt::Write for Serial {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for byte in text.bytes() {
            // Terminals in raw mode need the carriage return too.
            if byte == b'\n' {
                write_byte(b'\r');
            }
            write_byte(byte);
        }
        Ok(())
    }
}

macro_rules! print {
    ($($argument:tt)*) => {{
        use core::fmt::Write;
        let _ = write!($crate::serial::Serial, $($argument)*);
    }};
}

macro_rules! println {
    () => { $crate::serial::print!("\n") };
    ($($argument:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::serial::Serial, $($argument)*);
    }};
}

/// Start-up progress, compiled in with the `trace` feature.
macro_rules! trace {
    ($($argument:tt)*) => {
        if cfg!(feature = "trace") {
            $crate::serial::println!($($argument)*);
        }
    };
}

pub(crate) use {print, println, trace};
