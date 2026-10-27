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

    #[cfg_attr(not(test), allow(dead_code))]
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
        // `checked_add`: a wrapped sum would pass the check and write outside the mapping.
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.len),
            "write of {} bytes at offset {offset:#x} is outside the {}-byte mapping",
            bytes.len(),
            self.len
        );
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
    use super::*;

    const MAPPING_BYTES: usize = PROBE_MAPPING_SIZE;
    const PAGE_ALIGNMENT: u64 = 4096;
    const RETURNED_VALUE: u32 = 42;
    const PATCHED_RETURN_VALUE: u32 = 43;

    fn mapping() -> ExecMem {
        ExecMem::new(MAPPING_BYTES).expect("the host must allow an executable mapping")
    }

    fn contents(mem: &ExecMem, offset: usize, len: usize) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts((mem.ptr as *const u8).add(offset), len).to_vec() }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn probe_succeeds_on_a_host_with_a_jit_backend() {
        assert!(jit_probe());
    }

    #[test]
    fn a_new_mapping_is_page_aligned_zeroed_and_as_long_as_asked() {
        let mem = mapping();
        assert_eq!(mem.len(), MAPPING_BYTES);
        assert_ne!(mem.base(), 0);
        assert_eq!(mem.base() % PAGE_ALIGNMENT, 0);
        assert!(
            contents(&mem, 0, MAPPING_BYTES)
                .iter()
                .all(|&byte| byte == 0)
        );
    }

    #[test]
    fn write_bytes_lands_at_the_offset_and_nowhere_else() {
        let mem = mapping();
        mem.write_bytes(100, &[1, 2, 3, 4]);
        let all = contents(&mem, 0, MAPPING_BYTES);
        assert_eq!(&all[100..104], [1, 2, 3, 4]);
        assert!(all[..100].iter().all(|&byte| byte == 0));
        assert!(all[104..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn writes_can_be_repeated_and_overwrite_earlier_ones() {
        let mem = mapping();
        mem.write_bytes(0, &[1, 1, 1, 1]);
        mem.write_bytes(2, &[2, 2]);
        assert_eq!(contents(&mem, 0, 4), [1, 1, 2, 2]);
    }

    #[test]
    fn writing_the_last_byte_and_nothing_at_the_end_are_allowed() {
        let mem = mapping();
        mem.write_bytes(MAPPING_BYTES - 1, &[9]);
        mem.write_bytes(MAPPING_BYTES, &[]);
        assert_eq!(contents(&mem, MAPPING_BYTES - 1, 1), [9]);
    }

    #[test]
    #[should_panic(expected = "outside the")]
    fn writing_one_byte_past_the_end_is_refused() {
        mapping().write_bytes(MAPPING_BYTES - 1, &[1, 2]);
    }

    #[test]
    #[should_panic(expected = "outside the")]
    fn an_offset_past_the_end_is_refused() {
        mapping().write_bytes(MAPPING_BYTES + 1, &[]);
    }

    #[test]
    #[should_panic(expected = "outside the")]
    fn an_offset_that_wraps_the_address_space_is_refused_not_written() {
        // `usize::MAX + 1` wraps to 0, which a plain `+` would accept.
        mapping().write_bytes(usize::MAX, &[1]);
    }

    #[cfg(unix)]
    #[test]
    fn an_empty_or_absurd_mapping_is_refused_not_panicked_on() {
        assert!(ExecMem::new(0).is_none());
        assert!(ExecMem::new(usize::MAX / 2).is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn generated_code_runs_and_can_be_patched_and_rerun() {
        // mov eax, imm32; ret
        let code = |value: u32| {
            let mut bytes = vec![0xb8];
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes.push(X86_RET);
            bytes
        };
        let mem = mapping();
        mem.write_bytes(0, &code(RETURNED_VALUE));
        let entry: extern "C" fn() -> u32 = unsafe { std::mem::transmute(mem.base() as usize) };
        assert_eq!(entry(), RETURNED_VALUE);
        mem.write_bytes(0, &code(PATCHED_RETURN_VALUE));
        assert_eq!(entry(), PATCHED_RETURN_VALUE);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn generated_code_runs_and_can_be_patched_and_rerun() {
        // mov w0, #imm16; ret
        let code = |value: u32| [0x5280_0000 | (value << 5), AARCH64_RET];
        let mem = mapping();
        mem.write(0, &code(RETURNED_VALUE));
        let entry: extern "C" fn() -> u32 = unsafe { std::mem::transmute(mem.base() as usize) };
        assert_eq!(entry(), RETURNED_VALUE);
        mem.write(0, &code(PATCHED_RETURN_VALUE));
        assert_eq!(entry(), PATCHED_RETURN_VALUE);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn instruction_words_are_stored_little_endian_at_a_word_offset() {
        let mem = mapping();
        mem.write(8, &[0x0403_0201, 0x0807_0605]);
        assert_eq!(contents(&mem, 8, 8), [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn each_mapping_is_independent() {
        let (first, second) = (mapping(), mapping());
        assert_ne!(first.base(), second.base());
        first.write_bytes(0, &[0xaa]);
        assert_eq!(contents(&second, 0, 1), [0]);
    }

    #[test]
    fn mappings_can_be_used_from_another_thread() {
        let mem = mapping();
        let handle = std::thread::spawn(move || {
            mem.write_bytes(0, &[7]);
            contents(&mem, 0, 1)
        });
        assert_eq!(handle.join().unwrap(), [7]);
    }
}
