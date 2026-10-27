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

/// Whether a writable+executable mapping can be created and run on this host.
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn probe() -> bool {
    crate::sys::jit_probe()
}

#[cfg(not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64"))))]
mod stub {
    use crate::cpu::Cpu;
    use crate::cpu::exception::Exception;

    /// Never constructed: `Jit::new` returns `None`, so `Cpu::step_block` is unreachable.
    pub(crate) struct Jit;

    impl Jit {
        pub fn new() -> Option<Jit> {
            None
        }
        pub fn invalidate_page(&mut self, _ppn: u64) {}
    }

    impl Cpu {
        pub fn step_block(&mut self) -> Result<(), Exception> {
            unreachable!()
        }
    }
}

#[cfg(not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64"))))]
pub(crate) use stub::Jit;

#[cfg(not(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64"))))]
pub(crate) fn probe() -> bool {
    false
}
