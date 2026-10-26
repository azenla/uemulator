use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const QUEUE_FRAMES: usize = 1024;

#[derive(Default)]
struct Inner {
    to_guest: VecDeque<Vec<u8>>,
    from_guest: VecDeque<Vec<u8>>,
}

#[derive(Clone, Default)]
pub struct NetHandle {
    inner: Arc<(Mutex<Inner>, Condvar)>,
}

impl NetHandle {
    pub fn send_frame(&self, frame: &[u8]) -> bool {
        let mut g = self.inner.0.lock().unwrap();
        if g.to_guest.len() >= QUEUE_FRAMES {
            return false;
        }
        g.to_guest.push_back(frame.to_vec());
        true
    }

    pub fn try_recv_frame(&self) -> Option<Vec<u8>> {
        self.inner.0.lock().unwrap().from_guest.pop_front()
    }

    pub fn recv_frame(&self, timeout: Duration) -> Option<Vec<u8>> {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        if g.from_guest.is_empty() {
            g = cv.wait_timeout(g, timeout).unwrap().0;
        }
        g.from_guest.pop_front()
    }

    pub(crate) fn guest_send(&self, frame: Vec<u8>) {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        if g.from_guest.len() >= QUEUE_FRAMES {
            g.from_guest.pop_front();
        }
        g.from_guest.push_back(frame);
        cv.notify_one();
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.inner.0.lock().unwrap().to_guest.is_empty()
    }

    pub(crate) fn guest_recv(&self) -> Option<Vec<u8>> {
        self.inner.0.lock().unwrap().to_guest.pop_front()
    }
}
