//! Executable memory for the JIT. Only built with the `jit` feature.

pub(crate) struct ExecMem {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for ExecMem {}

impl ExecMem {
    pub fn base(&self) -> u64 {
        self.ptr as u64
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg(target_arch = "aarch64")]
    pub fn write(&self, off: usize, words: &[u32]) {
        let bytes =
            unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        self.write_bytes(off, bytes);
    }

    pub fn write_bytes(&self, off: usize, bytes: &[u8]) {
        assert!(off + bytes.len() <= self.len);
        unsafe {
            imp::write_enable(false);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(off), bytes.len());
            imp::write_enable(true);
            imp::flush(self.ptr.add(off), bytes.len());
        }
    }
}

#[cfg(unix)]
mod imp {
    use super::ExecMem;
    use std::ffi::c_void;

    #[cfg(target_os = "macos")]
    unsafe extern "C" {
        fn sys_icache_invalidate(start: *mut c_void, len: usize);
    }
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    unsafe extern "C" {
        fn __clear_cache(start: *mut c_void, end: *mut c_void);
    }

    #[cfg(target_os = "macos")]
    const EXTRA_FLAGS: i32 = libc::MAP_JIT;
    #[cfg(not(target_os = "macos"))]
    const EXTRA_FLAGS: i32 = 0;

    impl ExecMem {
        pub fn new(len: usize) -> Option<ExecMem> {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_PRIVATE | libc::MAP_ANON | EXTRA_FLAGS,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED || p.is_null() {
                return None;
            }
            Some(ExecMem {
                ptr: p as *mut u8,
                len,
            })
        }
    }

    impl Drop for ExecMem {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.ptr as *mut c_void, self.len);
            }
        }
    }

    // MAP_JIT pages are per-thread writable on Apple Silicon; Intel macs need nothing.
    pub unsafe fn write_enable(enabled: bool) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        unsafe {
            libc::pthread_jit_write_protect_np(enabled as i32);
        }
        let _ = enabled;
    }

    pub unsafe fn flush(start: *mut u8, len: usize) {
        #[cfg(target_os = "macos")]
        unsafe {
            sys_icache_invalidate(start as *mut c_void, len);
        }
        #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
        unsafe {
            __clear_cache(start as *mut c_void, start.add(len) as *mut c_void);
        }
        let _ = (start, len);
    }
}

#[cfg(windows)]
mod imp {
    use super::ExecMem;
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    impl ExecMem {
        pub fn new(len: usize) -> Option<ExecMem> {
            let p = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE,
                )
            };
            if p.is_null() {
                return None;
            }
            Some(ExecMem {
                ptr: p as *mut u8,
                len,
            })
        }
    }

    impl Drop for ExecMem {
        fn drop(&mut self) {
            unsafe {
                VirtualFree(self.ptr as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    pub unsafe fn write_enable(_enabled: bool) {}

    pub unsafe fn flush(start: *mut u8, len: usize) {
        unsafe {
            FlushInstructionCache(GetCurrentProcess(), start as *const _, len);
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn jit_probe() -> bool {
    let Some(m) = ExecMem::new(16384) else {
        return false;
    };
    m.write(0, &[0xd65f_03c0]);
    let f: extern "C" fn() = unsafe { std::mem::transmute(m.base() as usize) };
    f();
    true
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn jit_probe() -> bool {
    false
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe() {
        let _ = super::jit_probe();
    }
}
