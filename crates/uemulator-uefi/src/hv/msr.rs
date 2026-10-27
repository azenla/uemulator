use super::Vcpu;
use super::arch::{EFER_LMA, EFER_LME, EFER_NXE, EFER_SCE, MSR_EFER, MSR_FEATURE_CONTROL};

const MSR_IA32_PLATFORM_ID: u32 = 0x17;
const MSR_IA32_APIC_BASE: u32 = 0x1b;
const MSR_IA32_BIOS_SIGN_ID: u32 = 0x8b;
const MSR_IA32_MTRRCAP: u32 = 0xfe;
const MSR_IA32_MISC_ENABLE: u32 = 0x1a0;
const MSR_IA32_DEBUGCTL: u32 = 0x1d9;
const MSR_IA32_MTRR_DEF_TYPE: u32 = 0x2ff;
const MSR_AMD_SYSCFG: u32 = 0xc001_0010;

const MISC_ENABLE_FAST_STRINGS: u64 = 1 << 0;
const MISC_ENABLE_BTS_UNAVAILABLE: u64 = 1 << 11;
const MISC_ENABLE_PEBS_UNAVAILABLE: u64 = 1 << 12;

/// MSRs the guest reads through the intercept: the few that are virtualized, and the harmless ones
/// that are probed at boot and read as zero.
pub fn read(vcpu: &mut dyn Vcpu, msr: u32) -> Option<u64> {
    match msr {
        MSR_EFER => Some(vcpu.efer()),
        MSR_IA32_MISC_ENABLE => Some(
            MISC_ENABLE_FAST_STRINGS | MISC_ENABLE_BTS_UNAVAILABLE | MISC_ENABLE_PEBS_UNAVAILABLE,
        ),
        MSR_IA32_APIC_BASE
        | MSR_IA32_PLATFORM_ID
        | MSR_IA32_BIOS_SIGN_ID
        | MSR_IA32_MTRRCAP
        | MSR_IA32_MTRR_DEF_TYPE
        | MSR_FEATURE_CONTROL
        | MSR_IA32_DEBUGCTL
        | MSR_AMD_SYSCFG => Some(0),
        _ => None,
    }
}

/// Returns false for an MSR the guest may not write, which the caller turns into `#GP`.
pub fn write(vcpu: &mut dyn Vcpu, msr: u32, value: u64) -> bool {
    match msr {
        MSR_EFER => {
            // The mode bits follow from CR0 and cannot be changed from here; what the guest may
            // set is syscall and no-execute.
            let keep = EFER_LME | EFER_LMA;
            let allowed = EFER_SCE | EFER_NXE | keep;
            let efer = (value & allowed & !keep) | (vcpu.efer() & keep);
            vcpu.set_efer(efer);
            true
        }
        // Writes are dropped; the matching reads return the fixed values above.
        MSR_IA32_MISC_ENABLE | MSR_IA32_APIC_BASE | MSR_IA32_DEBUGCTL | MSR_AMD_SYSCFG => true,
        _ => false,
    }
}
