//! Host JIT. `native` (translate, block cache, `Cpu::step_block`) and `helpers` are shared by both
//! backends; `arch` is the per-host glue (prologue, helper addresses, block compiler) and
//! `emit*`/`a64`/`x64` are the code generators and assemblers. Hosts or builds without the backend
//! use `stub`, so `Cpu` can hold an `Option<Jit>` unconditionally and the interpreter runs alone.
//! See `plans/06-jit.md`.

#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod a64;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
#[path = "arch_a64.rs"]
mod arch;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
mod emit;

#[cfg(all(feature = "jit", target_arch = "x86_64"))]
#[path = "arch_x64.rs"]
mod arch;
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
mod emit_x64;
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
mod x64;

#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod helpers;
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod native;

#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) use native::Jit;
#[cfg(all(
    test,
    feature = "jit",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
pub(crate) use native::{PAUSE_BEFORE_INSTALL_MICROS, TEST_BLOCK_LIMIT};

/// Whether a writable+executable mapping can be created and run on this host.
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn probe() -> bool {
    crate::sys::jit_probe()
}

/// Held by tests while generated code runs. Rosetta (macOS x86_64 on Apple Silicon) keeps a
/// translation cache keyed by address; when one thread unmaps a code buffer and another thread
/// maps a new one at the same address while generated code runs, it can execute the stale
/// translation (wrong register values, SIGBUS). A standalone probe that maps, writes, runs and
/// unmaps a tiny function on 8 threads reproduces it without any emulator code, while rewriting
/// one long-lived buffer from 8 threads, or doing the same on a single thread, never fails. So it
/// is a Rosetta limitation, not an emulator bug: a real machine has one `Jit` per process and
/// native x86_64 hosts have coherent instruction caches. Tests there take turns; native hosts
/// need no lock and get an empty guard.
#[cfg(test)]
pub(crate) fn serialize_jit_tests() -> JitTestGuard {
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        use std::cell::Cell;
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        thread_local!(static HELD: Cell<bool> = const { Cell::new(false) });
        // Reentrant: a test thread that already has its turn (a harness holding the lock for its
        // lifetime, or a test that took it explicitly) must not deadlock on a second request.
        if HELD.with(Cell::get) {
            return JitTestGuard::default();
        }
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        HELD.with(|held| held.set(true));
        JitTestGuard {
            _lock: Some(lock),
            clear_held: Some(|| HELD.with(|held| held.set(false))),
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "macos")))]
    {
        JitTestGuard::default()
    }
}

/// Proof that this thread has its turn at running generated code; see `serialize_jit_tests`.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct JitTestGuard {
    _lock: Option<std::sync::MutexGuard<'static, ()>>,
    clear_held: Option<fn()>,
}

#[cfg(test)]
impl Drop for JitTestGuard {
    fn drop(&mut self) {
        if let Some(clear_held) = self.clear_held.take() {
            clear_held();
        }
    }
}

#[cfg(test)]
mod diff_alias;
#[cfg(test)]
mod diff_alu;
#[cfg(test)]
mod diff_bmi;
#[cfg(test)]
mod diff_flags;
#[cfg(test)]
mod diff_flow;
#[cfg(test)]
mod diff_hazards;
#[cfg(test)]
mod diff_memory;
#[cfg(test)]
mod diff_misc;
#[cfg(test)]
mod diff_mov;
#[cfg(test)]
mod diff_random;
#[cfg(test)]
mod diff_smc;
#[cfg(test)]
mod diff_tests;

#[cfg(not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64"))))]
mod stub {
    use crate::cpu::Cpu;
    use crate::cpu::exception::Exception;

    /// Never constructed: `Jit::new` returns `None`, so `Cpu::step_block` is unreachable.
    pub(crate) struct Jit;

    #[cfg(test)]
    #[derive(Default)]
    pub(crate) struct JitStats {
        pub translated: u64,
        pub stale_decodes: u64,
        pub resets: u64,
    }

    #[cfg(test)]
    pub(crate) static TEST_BLOCK_LIMIT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(usize::MAX);

    /// Only the real cache reads this; tests set it unconditionally.
    #[cfg(test)]
    pub(crate) static PAUSE_BEFORE_INSTALL_MICROS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    impl Jit {
        pub fn new(_chaining: bool) -> Option<Jit> {
            None
        }
        pub fn sibling(&self) -> Jit {
            Jit
        }
        pub fn invalidate_page(&self, _ppn: u64, _flags: &crate::ram::CodePages) {}
        pub fn set_ordered_memory(&mut self, _on: bool) {}
        #[cfg(test)]
        pub fn stats(&self) -> JitStats {
            JitStats::default()
        }
    }

    impl Cpu {
        pub fn step_block(&mut self) -> Result<(), Exception> {
            unreachable!()
        }
    }
}

#[cfg(not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64"))))]
pub(crate) use stub::Jit;
#[cfg(all(
    test,
    not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))
))]
pub(crate) use stub::{PAUSE_BEFORE_INSTALL_MICROS, TEST_BLOCK_LIMIT};

#[cfg(not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64"))))]
pub(crate) fn probe() -> bool {
    false
}
