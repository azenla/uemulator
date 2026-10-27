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

#[cfg(test)]
mod tests {
    use super::super::arch::{EFER_SVME, MSR_TSC, PASS_THROUGH_MSRS};
    use super::super::fake::FakeVcpu;
    use super::*;

    const SYSCALL_AND_NX: u64 = EFER_SCE | EFER_NXE;
    const MODE_BITS: u64 = EFER_LME | EFER_LMA;

    fn vcpu_with_efer(efer: u64) -> FakeVcpu {
        FakeVcpu {
            efer,
            ..FakeVcpu::default()
        }
    }

    #[test]
    fn efer_reads_back_what_the_vcpu_reports() {
        let mut vcpu = vcpu_with_efer(0xd01);
        assert_eq!(read(&mut vcpu, MSR_EFER), Some(0xd01));
    }

    #[test]
    fn misc_enable_reports_fast_strings_and_no_bts_or_pebs() {
        let mut vcpu = FakeVcpu::default();
        assert_eq!(read(&mut vcpu, MSR_IA32_MISC_ENABLE), Some(0x1801));
    }

    #[test]
    fn probed_msrs_read_as_zero() {
        let mut vcpu = FakeVcpu::default();
        for msr in [
            MSR_IA32_APIC_BASE,
            MSR_IA32_PLATFORM_ID,
            MSR_IA32_BIOS_SIGN_ID,
            MSR_IA32_MTRRCAP,
            MSR_IA32_MTRR_DEF_TYPE,
            MSR_FEATURE_CONTROL,
            MSR_IA32_DEBUGCTL,
            MSR_AMD_SYSCFG,
        ] {
            assert_eq!(read(&mut vcpu, msr), Some(0), "msr {msr:#x}");
        }
    }

    #[test]
    fn unknown_msrs_cannot_be_read() {
        let mut vcpu = FakeVcpu::default();
        for msr in [0, 0x1, 0x1b1, 0x4b564d00, 0xc000_0086, 0xffff_ffff] {
            assert_eq!(read(&mut vcpu, msr), None, "msr {msr:#x}");
        }
    }

    #[test]
    fn reads_do_not_change_the_vcpu() {
        let mut vcpu = vcpu_with_efer(0x500);
        let _ = read(&mut vcpu, MSR_EFER);
        let _ = read(&mut vcpu, MSR_IA32_MISC_ENABLE);
        assert_eq!(vcpu.efer, 0x500);
        assert_eq!(vcpu.skips, 0);
    }

    #[test]
    fn efer_write_sets_syscall_and_no_execute() {
        let mut vcpu = vcpu_with_efer(MODE_BITS);
        assert!(write(&mut vcpu, MSR_EFER, SYSCALL_AND_NX));
        assert_eq!(vcpu.efer, MODE_BITS | SYSCALL_AND_NX);
    }

    #[test]
    fn efer_write_can_clear_syscall_and_no_execute() {
        let mut vcpu = vcpu_with_efer(MODE_BITS | SYSCALL_AND_NX);
        assert!(write(&mut vcpu, MSR_EFER, 0));
        assert_eq!(vcpu.efer, MODE_BITS);
    }

    #[test]
    fn efer_write_cannot_clear_the_mode_bits() {
        let mut vcpu = vcpu_with_efer(MODE_BITS);
        assert!(write(&mut vcpu, MSR_EFER, SYSCALL_AND_NX));
        assert_eq!(vcpu.efer & MODE_BITS, MODE_BITS);
    }

    #[test]
    fn efer_write_cannot_set_the_mode_bits() {
        let mut vcpu = vcpu_with_efer(0);
        assert!(write(&mut vcpu, MSR_EFER, MODE_BITS | EFER_SCE));
        assert_eq!(vcpu.efer, EFER_SCE);
    }

    #[test]
    fn efer_write_cannot_set_svme_or_other_bits() {
        let mut vcpu = vcpu_with_efer(MODE_BITS);
        let others = EFER_SVME | (1 << 14) | (1 << 15) | (1 << 13) | (1 << 9);
        assert!(write(&mut vcpu, MSR_EFER, others | EFER_SCE));
        assert_eq!(vcpu.efer, MODE_BITS | EFER_SCE);
    }

    #[test]
    fn efer_write_ignores_the_upper_half_of_the_value() {
        let mut vcpu = vcpu_with_efer(MODE_BITS);
        assert!(write(&mut vcpu, MSR_EFER, (1 << 32) | EFER_NXE));
        assert_eq!(vcpu.efer, MODE_BITS | EFER_NXE);
    }

    #[test]
    fn writes_to_dropped_msrs_succeed_without_effect() {
        for msr in [
            MSR_IA32_MISC_ENABLE,
            MSR_IA32_APIC_BASE,
            MSR_IA32_DEBUGCTL,
            MSR_AMD_SYSCFG,
        ] {
            let mut vcpu = vcpu_with_efer(0x123);
            assert!(write(&mut vcpu, msr, u64::MAX), "msr {msr:#x}");
            assert_eq!(vcpu.efer, 0x123);
        }
    }

    #[test]
    fn read_only_probed_msrs_fault_on_write() {
        let mut vcpu = FakeVcpu::default();
        for msr in [
            MSR_IA32_PLATFORM_ID,
            MSR_IA32_BIOS_SIGN_ID,
            MSR_IA32_MTRRCAP,
            MSR_IA32_MTRR_DEF_TYPE,
            MSR_FEATURE_CONTROL,
        ] {
            assert!(!write(&mut vcpu, msr, 0), "msr {msr:#x}");
        }
    }

    #[test]
    fn unknown_msrs_fault_on_write() {
        let mut vcpu = FakeVcpu::default();
        assert!(!write(&mut vcpu, 0x4b56_4d00, 1));
        assert!(!write(&mut vcpu, 0xc000_0086, 1));
    }

    #[test]
    fn msrs_that_never_exit_have_no_handler() {
        // An intercept for an MSR that the permission map lets through would be dead code, and a
        // handler that answered for one would contradict the hardware value.
        let mut vcpu = FakeVcpu::default();
        for msr in PASS_THROUGH_MSRS {
            assert_eq!(read(&mut vcpu, msr), None, "read {msr:#x}");
            assert!(!write(&mut vcpu, msr, 0), "write {msr:#x}");
        }
        assert!(PASS_THROUGH_MSRS.contains(&MSR_TSC));
    }
}
