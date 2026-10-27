#[cfg(target_os = "uefi")]
use crate::prelude::*;
use alloc::collections::VecDeque;
use core::time::Duration;

use crate::sync::{Arc, Condvar, Mutex};

/// Cap on unread guest output (1 MiB). When no host thread reads, the oldest bytes are dropped
/// rather than blocking the guest. Input (`rx`) is not capped.
const OUTPUT_QUEUE_CAPACITY: usize = 1 << 20;

/// Window size reported until the host calls `resize`: the classic text-mode console.
const DEFAULT_COLS: u16 = 80;
const DEFAULT_ROWS: u16 = 25;

/// Push-style output callback, see `ConsoleHandle::on_output`.
type Sink = Box<dyn Fn(&[u8]) + Send>;

struct ConsoleState {
    /// Guest to host (serial TX / virtio-console transmit).
    tx: VecDeque<u8>,
    /// Host to guest.
    rx: VecDeque<u8>,
    /// While set, guest output goes here instead of `tx`.
    sink: Option<Sink>,
    cols: u16,
    rows: u16,
    /// Set by `resize`, cleared by `take_size_change`, so the virtio-console device reports each
    /// change to the guest once.
    size_dirty: bool,
}

impl Default for ConsoleState {
    fn default() -> Self {
        ConsoleState {
            tx: VecDeque::new(),
            rx: VecDeque::new(),
            sink: None,
            cols: DEFAULT_COLS,
            rows: DEFAULT_ROWS,
            size_dirty: false,
        }
    }
}

/// Byte stream between host and guest console, shared by clones. The vCPU thread (UART or
/// virtio-console device) and any number of host threads use the same single lock; the condvar
/// only wakes a host `read` blocked on empty guest output.
#[derive(Clone, Default)]
pub struct ConsoleHandle {
    inner: Arc<(Mutex<ConsoleState>, Condvar)>,
}

impl ConsoleHandle {
    /// Host to guest.
    pub fn write(&self, bytes: &[u8]) {
        let mut state = self.inner.0.lock().unwrap();
        state.rx.extend(bytes);
    }

    /// Guest to host. Without a timeout it never blocks. With one, waits once for output if none is queued, so it can return 0
    /// bytes after the timeout or a spurious wakeup. Output sent to an `on_output` callback never
    /// appears here.
    pub fn read(&self, buf: &mut [u8], timeout: Option<Duration>) -> usize {
        let (lock, output_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if state.tx.is_empty()
            && let Some(timeout) = timeout
        {
            state = output_ready.wait_timeout(state, timeout).unwrap().0;
        }
        let count = buf.len().min(state.tx.len());
        for (slot, byte) in buf.iter_mut().zip(state.tx.drain(..count)) {
            *slot = byte;
        }
        count
    }

    pub fn resize(&self, cols: u16, rows: u16) {
        let mut state = self.inner.0.lock().unwrap();
        if (state.cols, state.rows) != (cols, rows) {
            state.cols = cols;
            state.rows = rows;
            state.size_dirty = true;
        }
    }

    pub fn size(&self) -> (u16, u16) {
        let state = self.inner.0.lock().unwrap();
        (state.cols, state.rows)
    }

    /// Replaces polling with a push callback. It runs on the vCPU thread with the console lock
    /// held, so it must be quick and must not call back into this `ConsoleHandle`. Output queued
    /// before it was installed stays readable.
    pub fn on_output(&self, f: impl Fn(&[u8]) + Send + 'static) {
        self.inner.0.lock().unwrap().sink = Some(Box::new(f));
    }

    pub(crate) fn guest_tx(&self, byte: u8) {
        let (lock, output_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if let Some(sink) = &state.sink {
            sink(&[byte]);
            return;
        }
        if state.tx.len() >= OUTPUT_QUEUE_CAPACITY {
            state.tx.pop_front();
        }
        state.tx.push_back(byte);
        output_ready.notify_one();
    }

    pub(crate) fn guest_rx(&self) -> Option<u8> {
        self.inner.0.lock().unwrap().rx.pop_front()
    }

    pub(crate) fn guest_tx_bytes(&self, bytes: &[u8]) {
        let (lock, output_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if let Some(sink) = &state.sink {
            sink(bytes);
            return;
        }
        for &byte in bytes {
            if state.tx.len() >= OUTPUT_QUEUE_CAPACITY {
                state.tx.pop_front();
            }
            state.tx.push_back(byte);
        }
        output_ready.notify_one();
    }

    pub(crate) fn has_input(&self) -> bool {
        !self.inner.0.lock().unwrap().rx.is_empty()
    }

    pub(crate) fn guest_rx_bulk(&self, max: usize) -> Vec<u8> {
        let mut state = self.inner.0.lock().unwrap();
        let count = max.min(state.rx.len());
        state.rx.drain(..count).collect()
    }

    pub(crate) fn take_size_change(&self) -> Option<(u16, u16)> {
        let mut state = self.inner.0.lock().unwrap();
        if state.size_dirty {
            state.size_dirty = false;
            Some((state.cols, state.rows))
        } else {
            None
        }
    }
}
