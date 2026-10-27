#[cfg(target_os = "uefi")]
use crate::prelude::*;
use alloc::collections::VecDeque;
use core::time::Duration;

use crate::sync::{Arc, Condvar, Mutex};

/// Frames queued per direction. The two directions differ on overflow: a full host-to-guest queue
/// rejects the new frame (the sender can retry in order), a full guest-to-host queue drops the
/// oldest.
const QUEUE_FRAMES: usize = 1024;

#[derive(Default)]
struct NetState {
    to_guest: VecDeque<Vec<u8>>,
    from_guest: VecDeque<Vec<u8>>,
}

/// Ethernet frames between the host and the virtio-net device, shared by clones. One lock covers
/// both queues; the condvar only wakes a host `recv_frame` waiting for guest output.
#[derive(Clone, Default)]
pub struct NetHandle {
    inner: Arc<(Mutex<NetState>, Condvar)>,
}

impl NetHandle {
    /// Host to guest. Returns false when the queue is full; the frame is not queued.
    pub fn send_frame(&self, frame: &[u8]) -> bool {
        let mut state = self.inner.0.lock().unwrap();
        if state.to_guest.len() >= QUEUE_FRAMES {
            return false;
        }
        state.to_guest.push_back(frame.to_vec());
        true
    }

    pub fn try_recv_frame(&self) -> Option<Vec<u8>> {
        self.inner.0.lock().unwrap().from_guest.pop_front()
    }

    /// Waits once for a frame if none is queued, so a spurious wakeup can return `None` early.
    pub fn recv_frame(&self, timeout: Duration) -> Option<Vec<u8>> {
        let (lock, frame_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if state.from_guest.is_empty() {
            state = frame_ready.wait_timeout(state, timeout).unwrap().0;
        }
        state.from_guest.pop_front()
    }

    pub(crate) fn guest_send(&self, frame: Vec<u8>) {
        let (lock, frame_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if state.from_guest.len() >= QUEUE_FRAMES {
            state.from_guest.pop_front();
        }
        state.from_guest.push_back(frame);
        frame_ready.notify_one();
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.inner.0.lock().unwrap().to_guest.is_empty()
    }

    pub(crate) fn guest_recv(&self) -> Option<Vec<u8>> {
        self.inner.0.lock().unwrap().to_guest.pop_front()
    }
}
