use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const QUEUE_CAP: usize = 1 << 20;

type Sink = Box<dyn Fn(&[u8]) + Send>;

struct Inner {
    tx: VecDeque<u8>,
    rx: VecDeque<u8>,
    sink: Option<Sink>,
    cols: u16,
    rows: u16,
    size_dirty: bool,
}

impl Default for Inner {
    fn default() -> Self {
        Inner {
            tx: VecDeque::new(),
            rx: VecDeque::new(),
            sink: None,
            cols: 80,
            rows: 25,
            size_dirty: false,
        }
    }
}

#[derive(Clone, Default)]
pub struct ConsoleHandle {
    inner: Arc<(Mutex<Inner>, Condvar)>,
}

impl ConsoleHandle {
    pub fn write(&self, bytes: &[u8]) {
        let mut g = self.inner.0.lock().unwrap();
        g.rx.extend(bytes);
    }

    pub fn read(&self, buf: &mut [u8], timeout: Option<Duration>) -> usize {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        if g.tx.is_empty()
            && let Some(t) = timeout
        {
            g = cv.wait_timeout(g, t).unwrap().0;
        }
        let n = buf.len().min(g.tx.len());
        for (slot, b) in buf.iter_mut().zip(g.tx.drain(..n)) {
            *slot = b;
        }
        n
    }

    pub fn resize(&self, cols: u16, rows: u16) {
        let mut g = self.inner.0.lock().unwrap();
        if (g.cols, g.rows) != (cols, rows) {
            g.cols = cols;
            g.rows = rows;
            g.size_dirty = true;
        }
    }

    pub fn size(&self) -> (u16, u16) {
        let g = self.inner.0.lock().unwrap();
        (g.cols, g.rows)
    }

    pub fn on_output(&self, f: impl Fn(&[u8]) + Send + 'static) {
        self.inner.0.lock().unwrap().sink = Some(Box::new(f));
    }

    pub(crate) fn guest_tx(&self, byte: u8) {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        if let Some(sink) = &g.sink {
            sink(&[byte]);
            return;
        }
        if g.tx.len() >= QUEUE_CAP {
            g.tx.pop_front();
        }
        g.tx.push_back(byte);
        cv.notify_one();
    }

    pub(crate) fn guest_rx(&self) -> Option<u8> {
        self.inner.0.lock().unwrap().rx.pop_front()
    }

    pub(crate) fn guest_tx_bytes(&self, bytes: &[u8]) {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        if let Some(sink) = &g.sink {
            sink(bytes);
            return;
        }
        for &b in bytes {
            if g.tx.len() >= QUEUE_CAP {
                g.tx.pop_front();
            }
            g.tx.push_back(b);
        }
        cv.notify_one();
    }

    pub(crate) fn has_input(&self) -> bool {
        !self.inner.0.lock().unwrap().rx.is_empty()
    }

    pub(crate) fn guest_rx_bulk(&self, max: usize) -> Vec<u8> {
        let mut g = self.inner.0.lock().unwrap();
        let n = max.min(g.rx.len());
        g.rx.drain(..n).collect()
    }

    pub(crate) fn take_size_change(&self) -> Option<(u16, u16)> {
        let mut g = self.inner.0.lock().unwrap();
        if g.size_dirty {
            g.size_dirty = false;
            Some((g.cols, g.rows))
        } else {
            None
        }
    }
}
