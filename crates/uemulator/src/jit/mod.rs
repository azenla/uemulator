#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod a64;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
mod emit;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
mod helpers;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
mod native;

#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) use native::Jit;

#[cfg(not(all(feature = "jit", target_arch = "aarch64")))]
mod stub {
    use crate::cpu::Cpu;
    use crate::cpu::exc::Exc;

    pub(crate) struct Jit;

    impl Jit {
        pub fn new() -> Option<Jit> {
            None
        }
        pub fn invalidate_page(&mut self, _ppn: u64) {}
    }

    impl Cpu {
        pub fn step_block(&mut self) -> Result<(), Exc> {
            unreachable!()
        }
    }
}

#[cfg(not(all(feature = "jit", target_arch = "aarch64")))]
pub(crate) use stub::Jit;

#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) fn probe() -> bool {
    crate::sys::jit_probe()
}

#[cfg(not(all(feature = "jit", target_arch = "aarch64")))]
pub(crate) fn probe() -> bool {
    false
}
