use super::Vcpu;

const IA32_MISC_ENABLE: u32 = 0x1a0;
const FAST_STRINGS: u64 = 1;
const NO_BTS: u64 = 1 << 11;
const NO_PEBS: u64 = 1 << 12;

/// MSRs the guest reads through the intercept: the few that are virtualized, and the harmless ones
/// that are probed at boot and read as zero.
pub fn read(vcpu: &mut dyn Vcpu, msr: u32) -> Option<u64> {
    match msr {
        super::arch::MSR_EFER => Some(vcpu.efer()),
        IA32_MISC_ENABLE => Some(FAST_STRINGS | NO_BTS | NO_PEBS),
        // APIC base, platform id, microcode signature, MTRR capabilities and default type,
        // feature control, debug control, AMD SYSCFG.
        0x1b | 0x17 | 0x8b | 0xfe | 0x2ff | 0x3a | 0x1d9 | 0xc001_0010 => Some(0),
        _ => None,
    }
}

pub fn write(vcpu: &mut dyn Vcpu, msr: u32, value: u64) -> bool {
    match msr {
        super::arch::MSR_EFER => {
            // The mode bits follow from CR0 and cannot be changed from here; what the guest may
            // set is syscall and no-execute.
            let keep = super::arch::EFER_LME | super::arch::EFER_LMA;
            let allowed = super::arch::EFER_SCE | super::arch::EFER_NXE | keep;
            let efer = (value & allowed & !keep) | (vcpu.efer() & keep);
            vcpu.set_efer(efer);
            true
        }
        IA32_MISC_ENABLE | 0x1b | 0x1d9 | 0xc001_0010 => true,
        _ => false,
    }
}
