//! Machine-check architecture state. Nothing here ever reports an error: the emulated machine
//! has no hardware to fail. The registers exist so an OS that finds MCE/MCA in CPUID can
//! initialize its banks (enable reporting, clear status) without a #GP, which is what Windows
//! and Linux do at boot (SDM vol. 3 sec. 15.3, AMD APM vol. 2 sec. 9.3).

pub(crate) const MSR_MCG_CAP: u32 = 0x179;
pub(crate) const MSR_MCG_STATUS: u32 = 0x17a;
pub(crate) const MSR_MCG_CTL: u32 = 0x17b;
const MSR_MC_BANK_FIRST: u32 = 0x400;
const REGISTERS_PER_BANK: u32 = 4;
const BANK_CTL: u32 = 0;
const BANK_STATUS: u32 = 1;
const BANK_ADDR: u32 = 2;
const BANK_MISC: u32 = 3;

pub(crate) const BANKS: u32 = 6;
const MSR_MC_BANK_LAST: u32 = MSR_MC_BANK_FIRST + BANKS * REGISTERS_PER_BANK - 1;
/// MCG_CAP: bank count in 7:0 and MCG_CTL_P (bit 8), the MCG_CTL register exists.
const MCG_CAP_VALUE: u64 = BANKS as u64 | (1 << 8);

#[derive(Clone)]
pub(crate) struct Mca {
    global_status: u64,
    global_control: u64,
    /// Control, address and misc registers of each bank, stored as written. Status is read-only
    /// zero: only software may clear it, and nothing ever sets it.
    bank: [[u64; REGISTERS_PER_BANK as usize]; BANKS as usize],
}

impl Mca {
    pub fn new() -> Self {
        Mca {
            global_status: 0,
            global_control: 0,
            bank: [[0; REGISTERS_PER_BANK as usize]; BANKS as usize],
        }
    }

    pub fn owns(msr: u32) -> bool {
        matches!(msr, MSR_MCG_CAP..=MSR_MCG_CTL | MSR_MC_BANK_FIRST..=MSR_MC_BANK_LAST)
    }

    pub fn read(&self, msr: u32) -> Option<u64> {
        match msr {
            MSR_MCG_CAP => Some(MCG_CAP_VALUE),
            MSR_MCG_STATUS => Some(self.global_status),
            MSR_MCG_CTL => Some(self.global_control),
            MSR_MC_BANK_FIRST..=MSR_MC_BANK_LAST => {
                let (bank, register) = Self::split(msr);
                Some(if register == BANK_STATUS {
                    0
                } else {
                    self.bank[bank][register as usize]
                })
            }
            _ => None,
        }
    }

    fn split(msr: u32) -> (usize, u32) {
        let offset = msr - MSR_MC_BANK_FIRST;
        (
            (offset / REGISTERS_PER_BANK) as usize,
            offset % REGISTERS_PER_BANK,
        )
    }

    /// `false` is #GP: MCG_CAP is read-only, and a status register accepts only zero (a nonzero
    /// write to a status register faults on real parts, SDM 15.3.2.2).
    pub fn write(&mut self, msr: u32, value: u64) -> bool {
        match msr {
            MSR_MCG_CAP => false,
            MSR_MCG_STATUS => {
                self.global_status = value;
                true
            }
            MSR_MCG_CTL => {
                self.global_control = value;
                true
            }
            MSR_MC_BANK_FIRST..=MSR_MC_BANK_LAST => {
                let (bank, register) = Self::split(msr);
                match register {
                    BANK_STATUS => value == 0,
                    BANK_CTL | BANK_ADDR | BANK_MISC => {
                        self.bank[bank][register as usize] = value;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_reports_banks_and_the_global_control_register() {
        assert_eq!(Mca::new().read(MSR_MCG_CAP), Some(0x106));
    }

    #[test]
    fn bank_control_is_stored_and_status_stays_zero() {
        let mut mca = Mca::new();
        assert!(mca.write(0x400, u64::MAX));
        assert_eq!(mca.read(0x400), Some(u64::MAX));
        assert!(mca.write(0x401, 0));
        assert!(!mca.write(0x401, 1));
        assert_eq!(mca.read(0x401), Some(0));
    }

    #[test]
    fn registers_stop_after_the_last_bank() {
        let last = MSR_MC_BANK_FIRST + BANKS * 4 - 1;
        assert!(Mca::owns(last));
        assert!(!Mca::owns(last + 1));
    }

    #[test]
    fn capability_is_read_only() {
        assert!(!Mca::new().write(MSR_MCG_CAP, 0));
    }
}
