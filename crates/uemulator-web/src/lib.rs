//! The emulator as a WebAssembly module with a plain C ABI (no wasm-bindgen, so building needs
//! only `cargo`). `www/worker.js` loads it in a Web Worker, because `uw_run` blocks until the
//! guest exits and would freeze a page.
//!
//! The library is single-threaded here (see `freestanding` in `uemulator/build.rs`): the vCPU
//! runs inside `uw_run` and reaches the outside world only through `MachineConfig::poll`, which
//! calls the `host_*` imports below. A worker cannot receive messages while a synchronous call is
//! running, so keyboard input arrives through a `SharedArrayBuffer` ring that `host_input` drains.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use uemulator::{ClockMode, EmulatedMachine, ExitReason, JitMode, MachineConfig, StopHandle};

const NS_PER_MS: f64 = 1_000_000.0;

/// The vCPU calls the poll hook every few thousand instructions; crossing into JavaScript on each
/// one costs more than it saves, so busy guests are asked for input only every this-many polls.
/// An idle guest is always asked, since it has nothing better to do.
const POLL_INPUT_EVERY: u32 = 16;
/// Largest burst of keyboard bytes handed to the guest per poll.
const INPUT_CHUNK: usize = 256;

pub const EXIT_SHUTDOWN: i32 = 0;
pub const EXIT_RESET: i32 = 1;
pub const EXIT_TRIPLE_FAULT: i32 = 2;
pub const EXIT_STOPPED: i32 = 3;
pub const EXIT_INTERNAL: i32 = 4;
/// `uw_run` before a successful `uw_create`.
pub const EXIT_NO_MACHINE: i32 = -1;

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    /// Guest console output; JavaScript batches it and posts it to the page.
    fn host_output(ptr: *const u8, len: usize);
    /// Copies up to `cap` pending keystroke bytes to `ptr` and returns how many; never blocks.
    fn host_input(ptr: *mut u8, cap: usize) -> usize;
    /// Writes the page's requested terminal size (`cols`, `rows` as two little-endian `u16`) to
    /// `ptr` and returns 1 if it changed since the last call.
    fn host_resize(ptr: *mut u16) -> i32;
    /// Non-zero once the page asked to stop the machine.
    fn host_should_stop() -> i32;
    /// The guest is halted with nothing scheduled: JavaScript may sleep for a moment.
    fn host_idle();
    /// Milliseconds on a monotonic clock (`performance.now()`), fractional.
    fn host_now_ms() -> f64;
    /// Emulator diagnostics (the library's `log` hook).
    fn host_log(ptr: *const u8, len: usize);
}

thread_local! {
    static MACHINE: RefCell<Option<EmulatedMachine>> = const { RefCell::new(None) };
    /// A copy of the machine's stop handle. `uw_run` takes the machine out of `MACHINE` and the
    /// poll hook runs inside it, so the hook cannot reach the machine itself.
    static STOP: RefCell<Option<StopHandle>> = const { RefCell::new(None) };
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: String) {
    LAST_ERROR.with(|error| *error.borrow_mut() = message);
}

/// Reserves `len` bytes in the module's memory for the page to fill (kernel and initrd images,
/// the command line). Pair with `uw_create`, which takes ownership, or `uw_free`.
#[unsafe(no_mangle)]
pub extern "C" fn uw_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// # Safety
/// `ptr` and `len` must be exactly what a previous `uw_alloc(len)` returned and was given.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uw_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
}

/// Takes ownership of a buffer from `uw_alloc(len)` whose first `len` bytes the page filled.
///
/// # Safety
/// As for `uw_free`, and the buffer must not be used again.
unsafe fn take_buffer(ptr: *mut u8, len: usize) -> Vec<u8> {
    let mut buffer = unsafe { Vec::from_raw_parts(ptr, len, len.max(1)) };
    buffer.truncate(len);
    buffer
}

/// Builds the machine and loads the kernel. Returns 0, or 1 with the message in `uw_error_*`.
/// `initrd_len == 0` means no initramfs, `cmdline_len == 0` the library's default command line.
/// Takes ownership of all three buffers either way.
///
/// # Safety
/// Every pointer must come from `uw_alloc` with the matching length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uw_create(
    kernel: *mut u8,
    kernel_len: usize,
    initrd: *mut u8,
    initrd_len: usize,
    cmdline: *mut u8,
    cmdline_len: usize,
    memory_mib: u32,
    unix_time: f64,
    cols: u16,
    rows: u16,
) -> i32 {
    let kernel = unsafe { take_buffer(kernel, kernel_len) };
    let initrd = unsafe { take_buffer(initrd, initrd_len) };
    let cmdline = unsafe { take_buffer(cmdline, cmdline_len) };

    let mut config = MachineConfig::new()
        .kernel(kernel)
        .memory_mib(memory_mib as usize)
        // There is no JIT for WebAssembly.
        .jit(JitMode::Off)
        // Guest time follows the page's clock. A virtual clock jumps over every idle wait and
        // otherwise crawls at one nanosecond per interpreted instruction.
        .clock(ClockMode::Realtime)
        .host_clock(|| (unsafe { host_now_ms() } * NS_PER_MS) as u64)
        .unix_time(unix_time as u64)
        .log(|message| unsafe { host_log(message.as_ptr(), message.len()) })
        .poll(poll);
    if initrd_len != 0 {
        config = config.initrd(initrd);
    }
    if cmdline_len != 0 {
        config = config.cmdline(String::from_utf8_lossy(&cmdline).into_owned());
    }
    let mut machine = match EmulatedMachine::new(config) {
        Ok(machine) => machine,
        Err(error) => {
            set_error(error.to_string());
            return 1;
        }
    };
    let console = machine.console();
    console.resize(cols, rows);
    console.on_output(|bytes| unsafe { host_output(bytes.as_ptr(), bytes.len()) });
    if let Err(error) = machine.start() {
        set_error(error.to_string());
        return 1;
    }
    STOP.with(|stop| *stop.borrow_mut() = Some(machine.stop_handle()));
    MACHINE.with(|slot| *slot.borrow_mut() = Some(machine));
    0
}

fn poll(console: &uemulator::ConsoleHandle, idle: bool) {
    static POLLS: AtomicU32 = AtomicU32::new(0);
    if !idle
        && !POLLS
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(POLL_INPUT_EVERY)
    {
        return;
    }
    let mut chunk = [0u8; INPUT_CHUNK];
    loop {
        let count = unsafe { host_input(chunk.as_mut_ptr(), chunk.len()) };
        if count > 0 {
            console.write(&chunk[..count]);
        }
        if count < chunk.len() {
            break;
        }
    }
    let mut size = [0u16; 2];
    if unsafe { host_resize(size.as_mut_ptr()) } != 0 {
        console.resize(size[0], size[1]);
    }
    if unsafe { host_should_stop() } != 0 {
        STOP.with(|stop| {
            if let Some(stop) = &*stop.borrow() {
                stop.stop();
            }
        });
    }
    if idle {
        unsafe { host_idle() };
    }
}

/// Runs the guest on this thread until it exits and returns an `EXIT_*` code. For
/// `EXIT_INTERNAL` the message is in `uw_error_*`. The machine is gone afterwards; build a new one
/// to boot again.
#[unsafe(no_mangle)]
pub extern "C" fn uw_run() -> i32 {
    let Some(mut machine) = MACHINE.with(|slot| slot.borrow_mut().take()) else {
        return EXIT_NO_MACHINE;
    };
    let reason = machine.wait();
    STOP.with(|slot| *slot.borrow_mut() = None);
    match reason {
        ExitReason::Shutdown => EXIT_SHUTDOWN,
        ExitReason::Reset => EXIT_RESET,
        ExitReason::TripleFault => EXIT_TRIPLE_FAULT,
        ExitReason::Stopped => EXIT_STOPPED,
        ExitReason::Internal(message) => {
            set_error(message);
            EXIT_INTERNAL
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn uw_error_ptr() -> *const u8 {
    LAST_ERROR.with(|error| error.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn uw_error_len() -> usize {
    LAST_ERROR.with(|error| error.borrow().len())
}
