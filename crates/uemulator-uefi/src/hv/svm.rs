//! AMD SVM backend. Offsets and bit positions are from the AMD64 APM Vol. 2 (VMCB layout, Appendix
//! B). Pointers are used as physical addresses throughout: UEFI runs with an identity map.

use alloc::format;
use alloc::string::String;
use uemulator::BootState;

use super::arch::{self, Gprs};
use super::{Exit, PAGE_SIZE, Vcpu, alloc_pages};

const MSR_VM_CR: u32 = 0xc001_0114;
/// Physical address of the page where VMRUN saves part of the host state.
const MSR_VM_HSAVE_PA: u32 = 0xc001_0117;
/// VM_CR.SVMDIS: firmware set it to forbid enabling EFER.SVME.
const VM_CR_SVMDIS: u64 = 1 << 4;
/// CPUID.8000000Ah:EDX bit 0, nested paging.
const SVM_FEATURE_NESTED_PAGING: u32 = 1 << 0;

// VMCB control area offsets.
const VMCB_INTERCEPT_VECTOR3: usize = 0x0c;
const VMCB_INTERCEPT_VECTOR4: usize = 0x10;
const VMCB_IOPM_BASE_PA: usize = 0x40;
const VMCB_MSRPM_BASE_PA: usize = 0x48;
const VMCB_ASID: usize = 0x58;
const VMCB_TLB_CONTROL: usize = 0x5c;
const VMCB_V_INTR: usize = 0x60;
const VMCB_INTERRUPT_SHADOW: usize = 0x68;
const VMCB_EXITCODE: usize = 0x70;
const VMCB_EXITINFO1: usize = 0x78;
const VMCB_EXITINFO2: usize = 0x80;
const VMCB_NP_ENABLE: usize = 0x90;
const VMCB_EVENTINJ: usize = 0xa8;
const VMCB_EVENTINJ_ERR: usize = 0xac;
const VMCB_N_CR3: usize = 0xb0;

// VMCB state save area offsets; the area starts at 0x400. Each segment register is 16 bytes
// (selector, attrib, limit, base at the `SEGMENT_*` offsets below); GDTR and IDTR use the same
// shape with only limit and base meaningful.
const VMCB_SAVE_ES: usize = 0x400;
const VMCB_SAVE_CS: usize = 0x410;
const VMCB_SAVE_SS: usize = 0x420;
const VMCB_SAVE_DS: usize = 0x430;
const VMCB_SAVE_FS: usize = 0x440;
const VMCB_SAVE_GS: usize = 0x450;
const VMCB_SAVE_GDTR: usize = 0x460;
const VMCB_SAVE_LDTR: usize = 0x470;
const VMCB_SAVE_IDTR: usize = 0x480;
const VMCB_SAVE_TR: usize = 0x490;
const VMCB_SAVE_CPL: usize = 0x4cb;
const VMCB_SAVE_EFER: usize = 0x4d0;
const VMCB_SAVE_CR4: usize = 0x548;
const VMCB_SAVE_CR3: usize = 0x550;
const VMCB_SAVE_CR0: usize = 0x558;
const VMCB_SAVE_DR7: usize = 0x560;
const VMCB_SAVE_DR6: usize = 0x568;
const VMCB_SAVE_RFLAGS: usize = 0x570;
const VMCB_SAVE_RIP: usize = 0x578;
const VMCB_SAVE_RSP: usize = 0x5d8;
const VMCB_SAVE_RAX: usize = 0x5f8;
const VMCB_SAVE_G_PAT: usize = 0x668;

const SEGMENT_SELECTOR: usize = 0;
const SEGMENT_ATTRIB: usize = 2;
const SEGMENT_LIMIT: usize = 4;
const SEGMENT_BASE: usize = 8;

// Segment attrib bits. 0-3 type, 4 S, 7 P, 9 L, 10 D/B, 11 G (a different packing from VMX's).
const ATTRIB_TYPE_CODE_EXEC_READ_ACCESSED: u16 = 0xb;
const ATTRIB_TYPE_DATA_READ_WRITE_ACCESSED: u16 = 0x3;
const ATTRIB_TYPE_LDT: u16 = 0x2;
const ATTRIB_TYPE_BUSY_TSS: u16 = 0xb;
const ATTRIB_S: u16 = 1 << 4;
const ATTRIB_P: u16 = 1 << 7;
const ATTRIB_L: u16 = 1 << 9;
const ATTRIB_DB: u16 = 1 << 10;
const ATTRIB_G: u16 = 1 << 11;
const ATTRIB_CODE_64BIT: u16 =
    ATTRIB_TYPE_CODE_EXEC_READ_ACCESSED | ATTRIB_S | ATTRIB_P | ATTRIB_L | ATTRIB_G;
const ATTRIB_DATA_FLAT: u16 =
    ATTRIB_TYPE_DATA_READ_WRITE_ACCESSED | ATTRIB_S | ATTRIB_P | ATTRIB_DB | ATTRIB_G;
const ATTRIB_LDT: u16 = ATTRIB_TYPE_LDT | ATTRIB_P;
const ATTRIB_TSS: u16 = ATTRIB_TYPE_BUSY_TSS | ATTRIB_P;

// Intercept vector 3 (offset 0x0c).
const INTERCEPT_INTR: u32 = 1 << 0;
const INTERCEPT_VINTR: u32 = 1 << 4;
const INTERCEPT_CPUID: u32 = 1 << 18;
const INTERCEPT_INVD: u32 = 1 << 22;
const INTERCEPT_HLT: u32 = 1 << 24;
const INTERCEPT_IOIO_PROT: u32 = 1 << 27;
const INTERCEPT_MSR_PROT: u32 = 1 << 28;
const INTERCEPT_SHUTDOWN: u32 = 1 << 31;
// Intercept vector 4 (offset 0x10). VMRUN's own intercept is mandatory; the rest keep the guest
// from using SVM itself, which exits as `Undefined`.
const INTERCEPT_VMRUN: u32 = 1 << 0;
const INTERCEPT_VMMCALL: u32 = 1 << 1;
const INTERCEPT_VMLOAD: u32 = 1 << 2;
const INTERCEPT_VMSAVE: u32 = 1 << 3;
const INTERCEPT_STGI: u32 = 1 << 4;
const INTERCEPT_CLGI: u32 = 1 << 5;
const INTERCEPT_SKINIT: u32 = 1 << 6;

// V_INTR (offset 0x60) bits.
const V_IRQ: u64 = 1 << 8;
const V_IGN_TPR: u64 = 1 << 20;
const V_INTR_MASKING: u64 = 1 << 24;

/// INTERRUPT_SHADOW bit 0: the instruction after STI or MOV SS is not interruptible.
const INTERRUPT_SHADOW: u64 = 1 << 0;

/// TLB_CONTROL value that flushes all of the guest's TLB entries on VMRUN.
const TLB_CONTROL_FLUSH_ALL: u32 = 1;
/// ASID 0 is the host's; the guest needs a nonzero one.
const GUEST_ASID: u32 = 1;

const NP_ENABLE: u64 = 1 << 0;

// Exit codes.
const VMEXIT_INTR: u64 = 0x60;
const VMEXIT_VINTR: u64 = 0x64;
const VMEXIT_CPUID: u64 = 0x72;
const VMEXIT_INVD: u64 = 0x76;
const VMEXIT_HLT: u64 = 0x78;
const VMEXIT_IOIO: u64 = 0x7b;
const VMEXIT_MSR: u64 = 0x7c;
const VMEXIT_SHUTDOWN: u64 = 0x7f;
const VMEXIT_VMRUN: u64 = 0x80;
const VMEXIT_SKINIT: u64 = 0x86;
/// Nested page fault.
const VMEXIT_NPF: u64 = 0x400;
/// VMRUN refused the guest state.
const VMEXIT_INVALID: u64 = u64::MAX;

// EXITINFO1 for a VMEXIT_IOIO: bit 0 is 1 for IN, bit 2 string, 4 and 5 operand size 8 and 16
// bits (neither is 32), bits 31:16 the port. EXITINFO2 then holds the next rip.
const IOIO_TYPE_IN: u64 = 1 << 0;
const IOIO_STRING: u64 = 1 << 2;
const IOIO_SIZE_8: u64 = 1 << 4;
const IOIO_SIZE_16: u64 = 1 << 5;
const IOIO_PORT_SHIFT: u32 = 16;
/// EXITINFO1 of a VMEXIT_MSR: 0 for RDMSR, 1 for WRMSR.
const MSR_EXIT_RDMSR: u64 = 0;
/// EXITINFO1 of a VMEXIT_NPF is a page-fault style error code; bit 1 is a write. EXITINFO2 has the
/// guest physical address.
const NPF_WRITE: u64 = 1 << 1;

// The VMCB has no instruction length for these exits (the next-rip save feature is not used), so
// the length of the plain encoding is assumed; a prefix would make the guest skip too little.
const CPUID_LEN: u64 = 2;
const HLT_LEN: u64 = 1;
const INVD_LEN: u64 = 2;
const RDMSR_WRMSR_LEN: u64 = 2;

/// Sizes of the I/O permission map (12 KiB) and the MSR permission map (8 KiB).
const IOPM_PAGES: usize = 3;
const MSRPM_PAGES: usize = 2;
/// An MSR permission map entry is two bits per MSR (read, then write); the second MSR range
/// starts 0x800 bytes in.
const MSRPM_HIGH_RANGE_OFFSET: usize = 0x800;
const MSRPM_BITS_PER_MSR: usize = 2;
const MSRPM_ENTRY_MASK: u8 = 0b11;

pub fn usable() -> bool {
    arch::cpuid(arch::CPUID_LEAF_SVM, 0)[3] & SVM_FEATURE_NESTED_PAGING != 0
        && arch::rdmsr(MSR_VM_CR) & VM_CR_SVMDIS == 0
}

pub struct Svm {
    vmcb: *mut u8,
    /// Where `VMSAVE` keeps the host's FS/GS/TR/syscall MSRs across a run.
    host: *mut u8,
    gprs: Gprs,
    /// rip after the instruction behind the last exit, for `skip`.
    next_rip: u64,
    /// Event to inject on the next entry, in EVENTINJ format without the valid bit.
    event: Option<u32>,
    /// Ask for an exit as soon as the guest can take an interrupt.
    window: bool,
    /// The first run flushes the TLB.
    first: bool,
}

impl Svm {
    pub fn new(npt: u64, boot: &BootState) -> Result<Self, String> {
        let vmcb = alloc_pages(1)?;
        let host = alloc_pages(1)?;
        let hsave = alloc_pages(1)?;
        let iopm = alloc_pages(IOPM_PAGES)?;
        let msrpm = alloc_pages(MSRPM_PAGES)?;
        // All ones: every port and MSR exits unless cleared below.
        unsafe {
            core::ptr::write_bytes(iopm, 0xff, IOPM_PAGES * PAGE_SIZE);
            core::ptr::write_bytes(msrpm, 0xff, MSRPM_PAGES * PAGE_SIZE);
        }
        for msr in arch::PASS_THROUGH_MSRS {
            pass_through(msrpm, msr);
        }

        // EFER.SVME must be set before any SVM instruction runs.
        arch::wrmsr(
            arch::MSR_EFER,
            arch::rdmsr(arch::MSR_EFER) | arch::EFER_SVME,
        );
        arch::wrmsr(MSR_VM_HSAVE_PA, hsave as u64);

        let mut svm = Self {
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
        svm.write_u32(
            VMCB_INTERCEPT_VECTOR3,
            INTERCEPT_INTR
                | INTERCEPT_CPUID
                | INTERCEPT_INVD
                | INTERCEPT_HLT
                | INTERCEPT_IOIO_PROT
                | INTERCEPT_MSR_PROT
                | INTERCEPT_SHUTDOWN,
        );
        svm.write_u32(
            VMCB_INTERCEPT_VECTOR4,
            INTERCEPT_VMRUN
                | INTERCEPT_VMMCALL
                | INTERCEPT_VMLOAD
                | INTERCEPT_VMSAVE
                | INTERCEPT_STGI
                | INTERCEPT_CLGI
                | INTERCEPT_SKINIT,
        );
        svm.write_u64(VMCB_IOPM_BASE_PA, iopm as u64);
        svm.write_u64(VMCB_MSRPM_BASE_PA, msrpm as u64);
        svm.write_u32(VMCB_ASID, GUEST_ASID);
        // With V_INTR_MASKING the guest's IF masks only virtual interrupts, so a guest with
        // interrupts disabled still exits on a physical one for the host to service.
        svm.write_u64(VMCB_V_INTR, V_INTR_MASKING | V_IGN_TPR);
        svm.write_u64(VMCB_NP_ENABLE, NP_ENABLE);
        svm.write_u64(VMCB_N_CR3, npt);

        // Guest state: the 64-bit Linux entry.
        svm.segment(VMCB_SAVE_CS, boot.cs, ATTRIB_CODE_64BIT);
        for segment in [
            VMCB_SAVE_ES,
            VMCB_SAVE_SS,
            VMCB_SAVE_DS,
            VMCB_SAVE_FS,
            VMCB_SAVE_GS,
        ] {
            svm.segment(segment, boot.ds, ATTRIB_DATA_FLAT);
        }
        svm.write_u16(VMCB_SAVE_GDTR + SEGMENT_LIMIT, boot.gdt_limit);
        svm.write_u64(VMCB_SAVE_GDTR + SEGMENT_BASE, boot.gdt_base);
        svm.write_u32(VMCB_SAVE_IDTR + SEGMENT_LIMIT, 0);
        svm.write_u64(VMCB_SAVE_IDTR + SEGMENT_BASE, 0);
        svm.write_u16(VMCB_SAVE_LDTR + SEGMENT_ATTRIB, ATTRIB_LDT);
        svm.write_u32(VMCB_SAVE_LDTR + SEGMENT_LIMIT, arch::SYSTEM_SEGMENT_LIMIT);
        svm.write_u16(VMCB_SAVE_TR + SEGMENT_ATTRIB, ATTRIB_TSS);
        svm.write_u32(VMCB_SAVE_TR + SEGMENT_LIMIT, arch::SYSTEM_SEGMENT_LIMIT);
        svm.write_u8(VMCB_SAVE_CPL, 0);
        // VMRUN fails its consistency checks unless the guest EFER has SVME set; `efer` and
        // `set_efer` hide the bit from the guest.
        svm.write_u64(
            VMCB_SAVE_EFER,
            arch::EFER_LME | arch::EFER_LMA | arch::EFER_SVME,
        );
        svm.write_u64(VMCB_SAVE_CR0, arch::GUEST_ENTRY_CR0);
        svm.write_u64(VMCB_SAVE_CR3, boot.cr3);
        svm.write_u64(VMCB_SAVE_CR4, arch::GUEST_ENTRY_CR4);
        svm.write_u64(VMCB_SAVE_DR6, arch::DR6_RESET);
        svm.write_u64(VMCB_SAVE_DR7, arch::DR7_RESET);
        svm.write_u64(VMCB_SAVE_RFLAGS, arch::RFLAGS_RESERVED_ONE);
        svm.write_u64(VMCB_SAVE_RIP, boot.rip);
        svm.write_u64(VMCB_SAVE_G_PAT, arch::PAT_RESET);
        Ok(svm)
    }

    /// Flat 4 GiB segment with base 0.
    fn segment(&mut self, at: usize, selector: u16, attrib: u16) {
        self.write_u16(at + SEGMENT_SELECTOR, selector);
        self.write_u16(at + SEGMENT_ATTRIB, attrib);
        self.write_u32(at + SEGMENT_LIMIT, arch::FLAT_SEGMENT_LIMIT);
        self.write_u64(at + SEGMENT_BASE, 0);
    }

    fn read_u64(&self, at: usize) -> u64 {
        unsafe { (self.vmcb.add(at) as *const u64).read_volatile() }
    }
    fn write_u64(&mut self, at: usize, v: u64) {
        unsafe { (self.vmcb.add(at) as *mut u64).write_volatile(v) }
    }
    fn write_u32(&mut self, at: usize, v: u32) {
        unsafe { (self.vmcb.add(at) as *mut u32).write_volatile(v) }
    }
    fn write_u16(&mut self, at: usize, v: u16) {
        unsafe { (self.vmcb.add(at) as *mut u16).write_volatile(v) }
    }
    fn write_u8(&mut self, at: usize, v: u8) {
        unsafe { self.vmcb.add(at).write_volatile(v) }
    }
    fn read_u32(&self, at: usize) -> u32 {
        unsafe { (self.vmcb.add(at) as *const u32).read_volatile() }
    }
}

/// Clears both intercept bits (read and write) of `msr` in the MSR permission map.
fn pass_through(msrpm: *mut u8, msr: u32) {
    let (base, index) = match msr {
        0..=arch::MSR_LOW_RANGE_END => (0, msr),
        arch::MSR_HIGH_RANGE_START..=arch::MSR_HIGH_RANGE_END => {
            (MSRPM_HIGH_RANGE_OFFSET, msr - arch::MSR_HIGH_RANGE_START)
        }
        _ => return,
    };
    let bit = index as usize * MSRPM_BITS_PER_MSR;
    unsafe {
        *msrpm.add(base + bit / 8) &= !(MSRPM_ENTRY_MASK << (bit % 8));
    }
}

impl Vcpu for Svm {
    fn gprs(&mut self) -> &mut Gprs {
        &mut self.gprs
    }

    fn run(&mut self) -> Result<Exit, String> {
        // rax and rsp live in the VMCB, not in `Gprs`, while the guest runs.
        self.write_u64(VMCB_SAVE_RAX, self.gprs.rax);
        self.write_u64(VMCB_SAVE_RSP, self.gprs.rsp);
        let event = self.event.take().map_or(0, |e| e | arch::EVENT_VALID);
        self.write_u32(VMCB_EVENTINJ, event);
        // An interrupt window is a virtual interrupt that is never delivered: with the VINTR
        // intercept the guest exits as soon as it could take it, and the next entry clears it.
        let mut intercepts = self.read_u32(VMCB_INTERCEPT_VECTOR3);
        let mut v_intr = self.read_u64(VMCB_V_INTR) & !V_IRQ;
        if self.window {
            intercepts |= INTERCEPT_VINTR;
            v_intr |= V_IRQ;
        } else {
            intercepts &= !INTERCEPT_VINTR;
        }
        self.write_u32(VMCB_INTERCEPT_VECTOR3, intercepts);
        self.write_u64(VMCB_V_INTR, v_intr);
        self.write_u32(
            VMCB_TLB_CONTROL,
            if self.first { TLB_CONTROL_FLUSH_ALL } else { 0 },
        );
        self.first = false;

        unsafe { arch::uemu_svm_enter(self.vmcb as u64, self.host as u64, &mut self.gprs) };

        self.gprs.rax = self.read_u64(VMCB_SAVE_RAX);
        self.gprs.rsp = self.read_u64(VMCB_SAVE_RSP);
        let rip = self.read_u64(VMCB_SAVE_RIP);
        let code = self.read_u64(VMCB_EXITCODE);
        let (info1, info2) = (self.read_u64(VMCB_EXITINFO1), self.read_u64(VMCB_EXITINFO2));
        self.next_rip = rip;
        Ok(match code {
            VMEXIT_INTR => Exit::HostInterrupt,
            VMEXIT_VINTR => Exit::IntWindow,
            VMEXIT_CPUID => {
                self.next_rip = rip + CPUID_LEN;
                Exit::Cpuid
            }
            VMEXIT_HLT => {
                self.next_rip = rip + HLT_LEN;
                Exit::Hlt
            }
            VMEXIT_INVD => {
                self.next_rip = rip + INVD_LEN;
                Exit::Skip
            }
            VMEXIT_IOIO => {
                self.next_rip = info2;
                if info1 & IOIO_STRING != 0 {
                    return Err(format!(
                        "string port I/O (port {:#x}) at rip {rip:#x} is not supported",
                        info1 >> IOIO_PORT_SHIFT
                    ));
                }
                let size = if info1 & IOIO_SIZE_8 != 0 {
                    1
                } else if info1 & IOIO_SIZE_16 != 0 {
                    2
                } else {
                    4
                };
                Exit::Io {
                    port: (info1 >> IOIO_PORT_SHIFT) as u16,
                    size,
                    write: info1 & IOIO_TYPE_IN == 0,
                }
            }
            VMEXIT_MSR => {
                self.next_rip = rip + RDMSR_WRMSR_LEN;
                if info1 == MSR_EXIT_RDMSR {
                    Exit::Rdmsr
                } else {
                    Exit::Wrmsr
                }
            }
            VMEXIT_SHUTDOWN => Exit::TripleFault,
            // VMRUN, VMMCALL, VMLOAD, VMSAVE, STGI, CLGI, SKINIT.
            VMEXIT_VMRUN..=VMEXIT_SKINIT => Exit::Undefined,
            VMEXIT_NPF => Exit::Fault {
                gpa: info2,
                write: info1 & NPF_WRITE != 0,
            },
            VMEXIT_INVALID => {
                return Err(format!(
                    "VMRUN rejected the guest state (cr0 {:#x} cr3 {:#x} cr4 {:#x} efer {:#x} rip {rip:#x})",
                    self.read_u64(VMCB_SAVE_CR0),
                    self.read_u64(VMCB_SAVE_CR3),
                    self.read_u64(VMCB_SAVE_CR4),
                    self.read_u64(VMCB_SAVE_EFER)
                ));
            }
            other => Exit::Other(other),
        })
    }

    fn skip(&mut self) {
        let next_rip = self.next_rip;
        self.write_u64(VMCB_SAVE_RIP, next_rip);
        self.write_u64(VMCB_INTERRUPT_SHADOW, 0);
    }

    fn rip(&self) -> u64 {
        self.read_u64(VMCB_SAVE_RIP)
    }

    fn irq_ok(&self) -> bool {
        self.read_u64(VMCB_SAVE_RFLAGS) & arch::RFLAGS_IF != 0
            && self.read_u64(VMCB_INTERRUPT_SHADOW) & INTERRUPT_SHADOW == 0
            && self.event.is_none()
    }

    fn inject_irq(&mut self, vector: u8) {
        self.event = Some(vector as u32);
    }

    fn window(&mut self, on: bool) {
        self.window = on;
    }

    fn inject_gp(&mut self) {
        // Exception 13 with an error code of 0.
        self.write_u32(VMCB_EVENTINJ_ERR, 0);
        self.event = Some(
            arch::VECTOR_GENERAL_PROTECTION
                | (arch::EVENT_TYPE_HARDWARE_EXCEPTION << arch::EVENT_TYPE_SHIFT)
                | arch::EVENT_ERROR_CODE_VALID,
        );
    }

    fn inject_ud(&mut self) {
        self.event = Some(
            arch::VECTOR_INVALID_OPCODE
                | (arch::EVENT_TYPE_HARDWARE_EXCEPTION << arch::EVENT_TYPE_SHIFT),
        );
    }

    fn efer(&self) -> u64 {
        self.read_u64(VMCB_SAVE_EFER) & !arch::EFER_SVME
    }

    fn set_efer(&mut self, efer: u64) {
        self.write_u64(VMCB_SAVE_EFER, efer | arch::EFER_SVME);
    }
}
