//! Executable memory for the JIT. Only built with the `jit` feature.

/// One RWX mapping that generated code is written into and executed from. Writes go through
/// `write`/`write_bytes`, which also toggle write protection and flush the instruction cache;
/// callers use `base()` only to compute addresses.
pub(crate) struct ExecMem {
    ptr: *mut u8,
    len: usize,
}

// The mapping is owned exclusively by this value.
unsafe impl Send for ExecMem {}

impl ExecMem {
    pub fn base(&self) -> u64 {
        self.ptr as u64
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Writes aarch64 instruction words; they are little-endian in memory, as on the host.
    #[cfg(target_arch = "aarch64")]
    pub fn write(&self, offset: usize, words: &[u32]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words))
        };
        self.write_bytes(offset, bytes);
    }

    /// Makes the region writable for this thread only, copies, restores execute permission and
    /// flushes the instruction cache (see `imp`).
    pub fn write_bytes(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.len);
        unsafe {
            imp::set_write_protect(false);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
            imp::set_write_protect(true);
            imp::flush(self.ptr.add(offset), bytes.len());
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

    /// Apple Silicon only allows RWX anonymous memory mapped `MAP_JIT`; hardened-runtime
    /// binaries also need the allow-jit entitlement (`plans/06-jit.md`).
    #[cfg(target_os = "macos")]
    const PLATFORM_MMAP_FLAGS: i32 = libc::MAP_JIT;
    #[cfg(not(target_os = "macos"))]
    const PLATFORM_MMAP_FLAGS: i32 = 0;

    impl ExecMem {
        pub fn new(len: usize) -> Option<ExecMem> {
            let mapping = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_PRIVATE | libc::MAP_ANON | PLATFORM_MMAP_FLAGS,
                    -1,
                    0,
                )
            };
            if mapping == libc::MAP_FAILED || mapping.is_null() {
                return None;
            }
            Some(ExecMem {
                ptr: mapping as *mut u8,
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

    /// `protect == false` makes MAP_JIT pages writable (and non-executable) for the calling
    /// thread only; `true` flips back. Only Apple Silicon has this per-thread W^X switch;
    /// elsewhere the mapping is simply RWX.
    pub unsafe fn set_write_protect(protect: bool) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        unsafe {
            libc::pthread_jit_write_protect_np(protect as i32);
        }
        let _ = protect;
    }

    /// aarch64 has split, non-coherent instruction and data caches, so freshly written code must
    /// be flushed before it runs. x86 caches are coherent and need nothing.
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
            let mapping = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE,
                )
            };
            if mapping.is_null() {
                return None;
            }
            Some(ExecMem {
                ptr: mapping as *mut u8,
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

    pub unsafe fn set_write_protect(_protect: bool) {}

    pub unsafe fn flush(start: *mut u8, len: usize) {
        unsafe {
            FlushInstructionCache(GetCurrentProcess(), start as *const _, len);
        }
    }
}

/// One page on Apple Silicon (16 KiB); the probe only needs a small mapping.
const PROBE_MAPPING_SIZE: usize = 16384;
/// aarch64 `ret`.
#[cfg(target_arch = "aarch64")]
const AARCH64_RET: u32 = 0xd65f_03c0;
/// x86-64 `ret`.
#[cfg(target_arch = "x86_64")]
const X86_RET: u8 = 0xc3;

/// Actually maps memory, writes a `ret` and calls it. Checks the whole path (the OS allows RWX
/// or MAP_JIT mappings, the write switch and cache flush work), not just that mmap succeeded.
#[cfg(target_arch = "aarch64")]
pub(crate) fn jit_probe() -> bool {
    let Some(mem) = ExecMem::new(PROBE_MAPPING_SIZE) else {
        return false;
    };
    mem.write(0, &[AARCH64_RET]);
    let entry: extern "C" fn() = unsafe { std::mem::transmute(mem.base() as usize) };
    entry();
    true
}

/// See the aarch64 variant.
#[cfg(target_arch = "x86_64")]
pub(crate) fn jit_probe() -> bool {
    let Some(mem) = ExecMem::new(PROBE_MAPPING_SIZE) else {
        return false;
    };
    mem.write_bytes(0, &[X86_RET]);
    let entry: extern "C" fn() = unsafe { std::mem::transmute(mem.base() as usize) };
    entry();
    true
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
