//! The firmware is single threaded, so shared state is plain static storage.

use core::cell::UnsafeCell;

/// A static with interior mutability. There is one CPU and interrupts are off except while
/// idling in `time::idle` (whose handler touches nothing here), so no access races another.
/// What the type cannot prevent is re-entrancy: hold the reference for as short a time as
/// possible, and never across a call that calls back into the application.
pub struct Global<T>(UnsafeCell<T>);

unsafe impl<T> Sync for Global<T> {}

impl<T> Global<T> {
    pub const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    #[allow(clippy::mut_from_ref)]
    pub fn get(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }
}
