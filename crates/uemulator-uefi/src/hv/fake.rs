//! A `Vcpu` with plain fields, for tests of the code that drives one.

use alloc::string::String;
use alloc::vec::Vec;

use super::arch::RFLAGS_IF;
use super::{Exit, Gprs, Vcpu};

#[derive(Debug, PartialEq, Eq)]
pub enum Injected {
    Irq(u8),
    GeneralProtection,
    InvalidOpcode,
}

#[derive(Default)]
pub struct FakeVcpu {
    pub gprs: Gprs,
    pub rip: u64,
    pub rflags: u64,
    pub interrupt_shadow: bool,
    pub efer: u64,
    pub window: bool,
    pub injected: Vec<Injected>,
    pub skips: u32,
}

impl FakeVcpu {
    pub fn with_interrupts(enabled: bool) -> Self {
        Self {
            rflags: if enabled { RFLAGS_IF } else { 0 },
            ..Self::default()
        }
    }
}

impl Vcpu for FakeVcpu {
    fn gprs(&mut self) -> &mut Gprs {
        &mut self.gprs
    }

    fn run(&mut self) -> Result<Exit, String> {
        Err("the fake vCPU does not run".into())
    }

    fn skip(&mut self) {
        self.skips += 1;
        self.interrupt_shadow = false;
    }

    fn rip(&self) -> u64 {
        self.rip
    }

    fn irq_ok(&self) -> bool {
        self.rflags & RFLAGS_IF != 0 && !self.interrupt_shadow && self.injected.is_empty()
    }

    fn inject_irq(&mut self, vector: u8) {
        self.injected.push(Injected::Irq(vector));
    }

    fn window(&mut self, on: bool) {
        self.window = on;
    }

    fn inject_gp(&mut self) {
        self.injected.push(Injected::GeneralProtection);
    }

    fn inject_ud(&mut self) {
        self.injected.push(Injected::InvalidOpcode);
    }

    fn efer(&self) -> u64 {
        self.efer
    }

    fn set_efer(&mut self, efer: u64) {
        self.efer = efer;
    }
}
