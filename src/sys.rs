use std::ffi::c_void;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
    -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    #[cfg(target_os = "macos")]
    fn pthread_jit_write_protect_np(enabled: i32);
    #[cfg(target_os = "macos")]
    fn sys_icache_invalidate(start: *mut c_void, len: usize);
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    fn __clear_cache(start: *mut c_void, end: *mut c_void);
}

const PROT_RWX: i32 = 7;
#[cfg(target_os = "macos")]
const MAP_FLAGS: i32 = 0x0002 | 0x1000 | 0x0800;
#[cfg(not(target_os = "macos"))]
const MAP_FLAGS: i32 = 0x0002 | 0x0020;

pub(crate) struct ExecMem {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for ExecMem {}

impl ExecMem {
    pub fn new(len: usize) -> Option<ExecMem> {
        let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_RWX, MAP_FLAGS, -1, 0) };
        if p as isize == -1 || p.is_null() {
            return None;
        }
        Some(ExecMem {
            ptr: p as *mut u8,
            len,
        })
    }

    pub fn base(&self) -> u64 {
        self.ptr as u64
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn write(&self, off: usize, words: &[u32]) {
        assert!(off + words.len() * 4 <= self.len);
        unsafe {
            #[cfg(target_os = "macos")]
            pthread_jit_write_protect_np(0);
            std::ptr::copy_nonoverlapping(
                words.as_ptr() as *const u8,
                self.ptr.add(off),
                words.len() * 4,
            );
            #[cfg(target_os = "macos")]
            pthread_jit_write_protect_np(1);
            self.flush(off, words.len() * 4);
        }
    }

    unsafe fn flush(&self, off: usize, len: usize) {
        let start = unsafe { self.ptr.add(off) } as *mut c_void;
        #[cfg(target_os = "macos")]
        unsafe {
            sys_icache_invalidate(start, len);
        }
        #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
        unsafe {
            __clear_cache(start, (start as *mut u8).add(len) as *mut c_void);
        }
        let _ = (start, len);
    }
}

impl Drop for ExecMem {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
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
