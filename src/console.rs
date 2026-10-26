use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const QUEUE_CAP: usize = 1 << 20;

type Sink = Box<dyn Fn(&[u8]) + Send>;

#[derive(Default)]
struct Inner {
    tx: VecDeque<u8>,
    rx: VecDeque<u8>,
    sink: Option<Sink>,
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

    pub fn on_output(&self, f: impl Fn(&[u8]) + Send + 'static) {
        self.inner.0.lock().unwrap().sink = Some(Box::new(f));
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub(crate) fn guest_rx(&self) -> Option<u8> {
        self.inner.0.lock().unwrap().rx.pop_front()
    }
}
