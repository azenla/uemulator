//! `std::sync` on hosted targets; on UEFI there is one thread, so these are minimal stand-ins.

#[cfg(not(target_os = "uefi"))]
pub(crate) use std::sync::{Arc, Condvar, Mutex};

#[cfg(target_os = "uefi")]
pub(crate) use alloc::sync::Arc;
#[cfg(target_os = "uefi")]
pub(crate) use uefi_stubs::{Condvar, Mutex};

#[cfg(target_os = "uefi")]
mod uefi_stubs {
    use core::cell::UnsafeCell;
    use core::convert::Infallible;
    use core::ops::{Deref, DerefMut};
    use core::sync::atomic::{AtomicBool, Ordering};
    use core::time::Duration;

    #[derive(Default)]
    pub(crate) struct Mutex<T> {
        locked: AtomicBool,
        value: UnsafeCell<T>,
    }

    unsafe impl<T: Send> Send for Mutex<T> {}
    unsafe impl<T: Send> Sync for Mutex<T> {}

    pub(crate) struct Guard<'a, T>(&'a Mutex<T>);

    impl<T> Mutex<T> {
        pub fn lock(&self) -> Result<Guard<'_, T>, Infallible> {
            while self.locked.swap(true, Ordering::Acquire) {
                core::hint::spin_loop();
            }
            Ok(Guard(self))
        }
    }

    impl<T> Deref for Guard<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            unsafe { &*self.0.value.get() }
        }
    }

    impl<T> DerefMut for Guard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            unsafe { &mut *self.0.value.get() }
        }
    }

    impl<T> Drop for Guard<'_, T> {
        fn drop(&mut self) {
            self.0.locked.store(false, Ordering::Release);
        }
    }

    #[derive(Default)]
    pub(crate) struct Condvar;

    impl Condvar {
        pub fn wait_timeout<'a, T>(
            &self,
            guard: Guard<'a, T>,
            _timeout: Duration,
        ) -> Result<(Guard<'a, T>, ()), Infallible> {
            Ok((guard, ()))
        }

        pub fn notify_one(&self) {}
    }
}
