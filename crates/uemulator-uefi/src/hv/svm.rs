use alloc::format;
use alloc::string::String;
use uemulator::BootState;

use super::arch::{self, Gprs};
use super::{Exit, Vcpu, alloc_pages};

const MSR_VM_CR: u32 = 0xc001_0114;
const MSR_VM_HSAVE_PA: u32 = 0xc001_0117;
const VM_CR_SVMDIS: u64 = 1 << 4;

// Control area.
const INTERCEPT_VEC3: usize = 0x0c;
const INTERCEPT_VEC4: usize = 0x10;
const IOPM_BASE: usize = 0x40;
const MSRPM_BASE: usize = 0x48;
const ASID: usize = 0x58;
const TLB_CTL: usize = 0x5c;
const V_INTR: usize = 0x60;
const INT_STATE: usize = 0x68;
const EXIT_CODE: usize = 0x70;
const EXIT_INFO1: usize = 0x78;
const EXIT_INFO2: usize = 0x80;
const NP_ENABLE: usize = 0x90;
const EVENT_INJ: usize = 0xa8;
const EVENT_INJ_ERR: usize = 0xac;
const N_CR3: usize = 0xb0;

// Save area, at 0x400.
const SEG_ES: usize = 0x400;
const SEG_CS: usize = 0x410;
const SEG_SS: usize = 0x420;
const SEG_DS: usize = 0x430;
const SEG_FS: usize = 0x440;
const SEG_GS: usize = 0x450;
const GDTR: usize = 0x460;
const LDTR: usize = 0x470;
const IDTR: usize = 0x480;
const TR: usize = 0x490;
const CPL: usize = 0x4cb;
const EFER: usize = 0x4d0;
const CR4: usize = 0x548;
const CR3: usize = 0x550;
const CR0: usize = 0x558;
const DR7: usize = 0x560;
const DR6: usize = 0x568;
const RFLAGS: usize = 0x570;
const RIP: usize = 0x578;
const RSP: usize = 0x5d8;
const RAX: usize = 0x5f8;
const G_PAT: usize = 0x668;

const INTR_CPUID: u32 = 1 << 18;
const INTR_INVD: u32 = 1 << 22;
const INTR_HLT: u32 = 1 << 24;
const INTR_IOIO: u32 = 1 << 27;
const INTR_MSR: u32 = 1 << 28;
const INTR_SHUTDOWN: u32 = 1 << 31;
const INTR_INTR: u32 = 1;
const INTR_VINTR: u32 = 1 << 4;
const VEC4_VMRUN_TO_SKINIT: u32 = 0x7f;

const V_IRQ: u64 = 1 << 8;
const V_IGN_TPR: u64 = 1 << 20;
const V_INTR_MASKING: u64 = 1 << 24;

const EXIT_INTR: u64 = 0x60;
const EXIT_VINTR: u64 = 0x64;
const EXIT_INVD: u64 = 0x76;
const EXIT_CPUID: u64 = 0x72;
const EXIT_HLT: u64 = 0x78;
const EXIT_IOIO: u64 = 0x7b;
const EXIT_MSR: u64 = 0x7c;
const EXIT_SHUTDOWN: u64 = 0x7f;
const EXIT_VMRUN: u64 = 0x80;
const EXIT_SKINIT: u64 = 0x86;
const EXIT_NPF: u64 = 0x400;
const EXIT_INVALID: u64 = u64::MAX;

pub fn usable() -> bool {
    arch::cpuid(0x8000_000a, 0)[3] & 1 != 0 && arch::rdmsr(MSR_VM_CR) & VM_CR_SVMDIS == 0
}

pub struct Svm {
    vmcb: *mut u8,
    host: *mut u8,
    gprs: Gprs,
    next_rip: u64,
    event: Option<u32>,
    window: bool,
    first: bool,
}

impl Svm {
    pub fn new(npt: u64, boot: &BootState) -> Result<Self, String> {
        let vmcb = alloc_pages(1)?;
        let host = alloc_pages(1)?;
        let hsave = alloc_pages(1)?;
        let iopm = alloc_pages(3)?;
        let msrpm = alloc_pages(2)?;
        unsafe {
            core::ptr::write_bytes(iopm, 0xff, 3 * 4096);
            core::ptr::write_bytes(msrpm, 0xff, 2 * 4096);
        }
        for msr in [
            0x10,
            arch::MSR_STAR,
            arch::MSR_LSTAR,
            arch::MSR_CSTAR,
            arch::MSR_SFMASK,
            arch::MSR_FS_BASE,
            arch::MSR_GS_BASE,
            arch::MSR_KERNEL_GS_BASE,
            arch::MSR_SYSENTER_CS,
            arch::MSR_SYSENTER_ESP,
            arch::MSR_SYSENTER_EIP,
            arch::MSR_PAT,
        ] {
            pass_through(msrpm, msr);
        }

        arch::wrmsr(
            arch::MSR_EFER,
            arch::rdmsr(arch::MSR_EFER) | arch::EFER_SVME,
        );
        arch::wrmsr(MSR_VM_HSAVE_PA, hsave as u64);

        let mut s = Self {
            vmcb,
            host,
            gprs: Gprs {
                rsi: boot.rsi,
                ..Gprs::default()
            },
            next_rip: 0,
            event: None,
            window: false,
            first: true,
        };
        s.w32(
            INTERCEPT_VEC3,
            INTR_INTR | INTR_CPUID | INTR_INVD | INTR_HLT | INTR_IOIO | INTR_MSR | INTR_SHUTDOWN,
        );
        s.w32(INTERCEPT_VEC4, VEC4_VMRUN_TO_SKINIT);
        s.w64(IOPM_BASE, iopm as u64);
        s.w64(MSRPM_BASE, msrpm as u64);
        s.w32(ASID, 1);
        s.w64(V_INTR, V_INTR_MASKING | V_IGN_TPR);
        s.w64(NP_ENABLE, 1);
        s.w64(N_CR3, npt);

        s.segment(SEG_CS, boot.cs, 0xa9b);
        for seg in [SEG_ES, SEG_SS, SEG_DS, SEG_FS, SEG_GS] {
            s.segment(seg, boot.ds, 0xc93);
        }
        s.w16(GDTR + 4, boot.gdt_limit);
        s.w64(GDTR + 8, boot.gdt_base);
        s.w32(IDTR + 4, 0);
        s.w64(IDTR + 8, 0);
        s.w16(LDTR + 2, 0x82);
        s.w32(LDTR + 4, 0xffff);
        s.w16(TR + 2, 0x8b);
        s.w32(TR + 4, 0xffff);
        s.w8(CPL, 0);
        s.w64(EFER, arch::EFER_LME | arch::EFER_LMA | arch::EFER_SVME);
        s.w64(CR0, 0x8001_0033);
        s.w64(CR3, boot.cr3);
        s.w64(CR4, 0x20);
        s.w64(DR6, 0xffff_0ff0);
        s.w64(DR7, 0x400);
        s.w64(RFLAGS, 2);
        s.w64(RIP, boot.rip);
        s.w64(G_PAT, 0x0007_0406_0007_0406);
        Ok(s)
    }

    fn segment(&mut self, at: usize, selector: u16, attrib: u16) {
        self.w16(at, selector);
        self.w16(at + 2, attrib);
        self.w32(at + 4, 0xffff_ffff);
        self.w64(at + 8, 0);
    }

    fn r64(&self, at: usize) -> u64 {
        unsafe { (self.vmcb.add(at) as *const u64).read_volatile() }
    }
    fn w64(&mut self, at: usize, v: u64) {
        unsafe { (self.vmcb.add(at) as *mut u64).write_volatile(v) }
    }
    fn w32(&mut self, at: usize, v: u32) {
        unsafe { (self.vmcb.add(at) as *mut u32).write_volatile(v) }
    }
    fn w16(&mut self, at: usize, v: u16) {
        unsafe { (self.vmcb.add(at) as *mut u16).write_volatile(v) }
    }
    fn w8(&mut self, at: usize, v: u8) {
        unsafe { self.vmcb.add(at).write_volatile(v) }
    }
    fn r32(&self, at: usize) -> u32 {
        unsafe { (self.vmcb.add(at) as *const u32).read_volatile() }
    }
}

fn pass_through(msrpm: *mut u8, msr: u32) {
    let (base, index) = match msr {
        0..=0x1fff => (0, msr),
        0xc000_0000..=0xc000_1fff => (0x800, msr - 0xc000_0000),
        _ => return,
    };
    let bit = index as usize * 2;
    unsafe {
        *msrpm.add(base + bit / 8) &= !(3 << (bit % 8));
    }
}

impl Vcpu for Svm {
    fn gprs(&mut self) -> &mut Gprs {
        &mut self.gprs
    }

    fn run(&mut self) -> Result<Exit, String> {
        self.w64(RAX, self.gprs.rax);
        self.w64(RSP, self.gprs.rsp);
        let event = self.event.take().map_or(0, |e| e | (1 << 31));
        self.w32(EVENT_INJ, event);
        let mut vec3 = self.r32(INTERCEPT_VEC3);
        let mut vint = self.r64(V_INTR) & !(V_IRQ);
        if self.window {
            vec3 |= INTR_VINTR;
            vint |= V_IRQ;
        } else {
            vec3 &= !INTR_VINTR;
        }
        self.w32(INTERCEPT_VEC3, vec3);
        self.w64(V_INTR, vint);
        self.w32(TLB_CTL, if self.first { 1 } else { 0 });
        self.first = false;

        unsafe { arch::uemu_svm_enter(self.vmcb as u64, self.host as u64, &mut self.gprs) };

        self.gprs.rax = self.r64(RAX);
        self.gprs.rsp = self.r64(RSP);
        let rip = self.r64(RIP);
        let code = self.r64(EXIT_CODE);
        let (info1, info2) = (self.r64(EXIT_INFO1), self.r64(EXIT_INFO2));
        self.next_rip = rip;
        Ok(match code {
            EXIT_INTR => Exit::HostInterrupt,
            EXIT_VINTR => Exit::IntWindow,
            EXIT_CPUID => {
                self.next_rip = rip + 2;
                Exit::Cpuid
            }
            EXIT_HLT => {
                self.next_rip = rip + 1;
                Exit::Hlt
            }
            EXIT_INVD => {
                self.next_rip = rip + 2;
                Exit::Skip
            }
            EXIT_IOIO => {
                self.next_rip = info2;
                if info1 & 4 != 0 {
                    return Err(format!(
                        "string port I/O (port {:#x}) at rip {rip:#x} is not supported",
                        info1 >> 16
                    ));
                }
                let size = if info1 & 0x10 != 0 {
                    1
                } else if info1 & 0x20 != 0 {
                    2
                } else {
                    4
                };
                Exit::Io {
                    port: (info1 >> 16) as u16,
                    size,
                    write: info1 & 1 == 0,
                }
            }
            EXIT_MSR => {
                self.next_rip = rip + 2;
                if info1 == 0 { Exit::Rdmsr } else { Exit::Wrmsr }
            }
            EXIT_SHUTDOWN => Exit::TripleFault,
            EXIT_VMRUN..=EXIT_SKINIT => Exit::Undefined,
            EXIT_NPF => Exit::Fault {
                gpa: info2,
                write: info1 & 2 != 0,
            },
            EXIT_INVALID => {
                return Err(format!(
                    "VMRUN rejected the guest state (cr0 {:#x} cr3 {:#x} cr4 {:#x} efer {:#x} rip {rip:#x})",
                    self.r64(CR0),
                    self.r64(CR3),
                    self.r64(CR4),
                    self.r64(EFER)
                ));
            }
            other => Exit::Other(other),
        })
    }

    fn skip(&mut self) {
        let n = self.next_rip;
        self.w64(RIP, n);
        self.w64(INT_STATE, 0);
    }

    fn rip(&self) -> u64 {
        self.r64(RIP)
    }

    fn irq_ok(&self) -> bool {
        self.r64(RFLAGS) & (1 << 9) != 0 && self.r64(INT_STATE) & 1 == 0 && self.event.is_none()
    }

    fn inject_irq(&mut self, vector: u8) {
        self.event = Some(vector as u32);
    }

    fn window(&mut self, on: bool) {
        self.window = on;
    }

    fn inject_gp(&mut self) {
        // Exception 13 with an error code of 0.
        self.w32(EVENT_INJ_ERR, 0);
        self.event = Some(13 | (3 << 8) | (1 << 11));
    }

    fn inject_ud(&mut self) {
        self.event = Some(6 | (3 << 8));
    }

    fn efer(&self) -> u64 {
        self.r64(EFER) & !arch::EFER_SVME
    }

    fn set_efer(&mut self, efer: u64) {
        self.w64(EFER, efer | arch::EFER_SVME);
    }
}
