//! AMD SVM backend. Offsets and bit positions are from the AMD64 APM Vol. 2 (VMCB layout, Appendix
//! B). Pointers are used as physical addresses throughout: UEFI runs with an identity map.

#[cfg(target_os = "uefi")]
use alloc::format;
#[cfg(target_os = "uefi")]
use alloc::string::String;
#[cfg(target_os = "uefi")]
use uemulator::BootState;

use super::Exit;
use super::arch;
#[cfg(target_os = "uefi")]
use super::arch::Gprs;
#[cfg(target_os = "uefi")]
use super::{PAGE_SIZE, Vcpu, alloc_pages};

const MSR_VM_CR: u32 = 0xc001_0114;
/// Physical address of the page where VMRUN saves part of the host state.
const MSR_VM_HSAVE_PA: u32 = 0xc001_0117;
/// VM_CR.SVMDIS: firmware set it to forbid enabling EFER.SVME.
const VM_CR_SVMDIS: u64 = 1 << 4;
/// CPUID.8000000Ah:EDX bit 0, nested paging.
const SVM_FEATURE_NESTED_PAGING: u32 = 1 << 0;
/// CPUID.8000000Ah:EDX bit 3, next-RIP save: #VMEXIT stores the address of the instruction after
/// the intercepted one in the VMCB.
const SVM_FEATURE_NRIPS: u32 = 1 << 3;

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
/// Written by #VMEXIT only when `SVM_FEATURE_NRIPS` is present.
const VMCB_NEXT_RIP: usize = 0xc8;

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

// Without the next-RIP save feature the VMCB has no instruction length for these exits, so the
// length of the plain encoding is assumed; a prefix would make the guest skip too little.
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

#[cfg(target_os = "uefi")]
pub fn usable() -> bool {
    usable_with(
        arch::cpuid(arch::CPUID_LEAF_SVM, 0)[3],
        arch::rdmsr(MSR_VM_CR),
    )
}

/// Nested paging is offered and the firmware has not disabled SVM, given CPUID.8000000Ah:EDX and
/// VM_CR.
fn usable_with(svm_features: u32, vm_cr: u64) -> bool {
    svm_features & SVM_FEATURE_NESTED_PAGING != 0 && vm_cr & VM_CR_SVMDIS == 0
}

#[cfg(target_os = "uefi")]
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
    /// #VMEXIT stores the next rip (CPUID.8000000Ah:EDX bit 3).
    nrips: bool,
}

#[cfg(target_os = "uefi")]
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
        let msrpm_bytes =
            unsafe { core::slice::from_raw_parts_mut(msrpm, MSRPM_PAGES * PAGE_SIZE) };
        for msr in arch::PASS_THROUGH_MSRS {
            pass_through(msrpm_bytes, msr);
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
            nrips: arch::cpuid(arch::CPUID_LEAF_SVM, 0)[3] & SVM_FEATURE_NRIPS != 0,
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
fn pass_through(msrpm: &mut [u8], msr: u32) {
    let (base, index) = match msr {
        0..=arch::MSR_LOW_RANGE_END => (0, msr),
        arch::MSR_HIGH_RANGE_START..=arch::MSR_HIGH_RANGE_END => {
            (MSRPM_HIGH_RANGE_OFFSET, msr - arch::MSR_HIGH_RANGE_START)
        }
        _ => return,
    };
    let bit = index as usize * MSRPM_BITS_PER_MSR;
    msrpm[base + bit / 8] &= !(MSRPM_ENTRY_MASK << (bit % 8));
}

/// Why `decode_exit` has no `Exit` to report.
#[derive(Debug, PartialEq, Eq)]
enum DecodeError {
    StringIo { port: u16 },
    EntryRejected,
}

/// The rip after an intercepted instruction: the one the CPU saved, if it did, else the length of
/// the unprefixed encoding.
fn next_rip(rip: u64, saved: Option<u64>, unprefixed_len: u64) -> u64 {
    saved.unwrap_or(rip + unprefixed_len)
}

/// Maps a #VMEXIT to what the event loop handles, and the rip to resume at if it emulates it.
/// `saved_next_rip` is the VMCB's NRIP, `None` where the feature is missing or the CPU left it 0.
fn decode_exit(
    code: u64,
    info1: u64,
    info2: u64,
    rip: u64,
    saved_next_rip: Option<u64>,
) -> Result<(Exit, u64), DecodeError> {
    let after = |len| next_rip(rip, saved_next_rip, len);
    Ok(match code {
        VMEXIT_INTR => (Exit::HostInterrupt, rip),
        VMEXIT_VINTR => (Exit::IntWindow, rip),
        VMEXIT_CPUID => (Exit::Cpuid, after(CPUID_LEN)),
        VMEXIT_HLT => (Exit::Hlt, after(HLT_LEN)),
        VMEXIT_INVD => (Exit::Skip, after(INVD_LEN)),
        VMEXIT_IOIO => {
            if info1 & IOIO_STRING != 0 {
                return Err(DecodeError::StringIo {
                    port: (info1 >> IOIO_PORT_SHIFT) as u16,
                });
            }
            let size = if info1 & IOIO_SIZE_8 != 0 {
                1
            } else if info1 & IOIO_SIZE_16 != 0 {
                2
            } else {
                4
            };
            let exit = Exit::Io {
                port: (info1 >> IOIO_PORT_SHIFT) as u16,
                size,
                write: info1 & IOIO_TYPE_IN == 0,
            };
            (exit, info2)
        }
        VMEXIT_MSR => {
            let exit = if info1 == MSR_EXIT_RDMSR {
                Exit::Rdmsr
            } else {
                Exit::Wrmsr
            };
            (exit, after(RDMSR_WRMSR_LEN))
        }
        VMEXIT_SHUTDOWN => (Exit::TripleFault, rip),
        // VMRUN, VMMCALL, VMLOAD, VMSAVE, STGI, CLGI, SKINIT.
        VMEXIT_VMRUN..=VMEXIT_SKINIT => (Exit::Undefined, rip),
        VMEXIT_NPF => (
            Exit::Fault {
                gpa: info2,
                write: info1 & NPF_WRITE != 0,
            },
            rip,
        ),
        VMEXIT_INVALID => return Err(DecodeError::EntryRejected),
        other => (Exit::Other(other), rip),
    })
}

#[cfg(target_os = "uefi")]
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
        // A stale value must never be taken for this exit's.
        if self.nrips {
            self.write_u64(VMCB_NEXT_RIP, 0);
        }

        unsafe { arch::uemu_svm_enter(self.vmcb as u64, self.host as u64, &mut self.gprs) };

        self.gprs.rax = self.read_u64(VMCB_SAVE_RAX);
        self.gprs.rsp = self.read_u64(VMCB_SAVE_RSP);
        let rip = self.read_u64(VMCB_SAVE_RIP);
        let code = self.read_u64(VMCB_EXITCODE);
        let (info1, info2) = (self.read_u64(VMCB_EXITINFO1), self.read_u64(VMCB_EXITINFO2));
        let saved_next_rip = self
            .nrips
            .then(|| self.read_u64(VMCB_NEXT_RIP))
            .filter(|&next| next != 0);
        match decode_exit(code, info1, info2, rip, saved_next_rip) {
            Ok((exit, next_rip)) => {
                self.next_rip = next_rip;
                Ok(exit)
            }
            Err(DecodeError::StringIo { port }) => Err(format!(
                "string port I/O (port {port:#x}) at rip {rip:#x} is not supported"
            )),
            Err(DecodeError::EntryRejected) => Err(format!(
                "VMRUN rejected the guest state (cr0 {:#x} cr3 {:#x} cr4 {:#x} efer {:#x} rip {rip:#x})",
                self.read_u64(VMCB_SAVE_CR0),
                self.read_u64(VMCB_SAVE_CR3),
                self.read_u64(VMCB_SAVE_CR4),
                self.read_u64(VMCB_SAVE_EFER)
            )),
        }
    }

    fn skip(&mut self) {
        let next_rip = self.next_rip;
        self.write_u64(VMCB_SAVE_RIP, next_rip);
        self.write_u64(VMCB_INTERRUPT_SHADOW, 0);
    }

    fn rip(&self) -> u64 {
        self.read_u64(VMCB_SAVE_RIP)
    }

    fn interrupts_enabled(&self) -> bool {
        self.read_u64(VMCB_SAVE_RFLAGS) & arch::RFLAGS_IF != 0
    }

    fn irq_ok(&self) -> bool {
        self.interrupts_enabled()
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
        self.event = Some(arch::exception_event(arch::VECTOR_GENERAL_PROTECTION, true));
    }

    fn inject_ud(&mut self) {
        self.event = Some(arch::exception_event(arch::VECTOR_INVALID_OPCODE, false));
    }

    fn efer(&self) -> u64 {
        self.read_u64(VMCB_SAVE_EFER) & !arch::EFER_SVME
    }

    fn set_efer(&mut self, efer: u64) {
        self.write_u64(VMCB_SAVE_EFER, efer | arch::EFER_SVME);
    }
}

#[cfg(test)]
mod tests {
    use super::super::PAGE_SIZE;
    use super::*;
    use alloc::vec::Vec;

    const RIP: u64 = 0x10_0000;
    /// The next rip the CPU would save for a plain two-byte instruction.
    const SAVED_PLAIN: u64 = RIP + 2;

    fn decode(code: u64) -> Result<(Exit, u64), DecodeError> {
        decode_exit(code, 0, 0, RIP, None)
    }

    // VMCB layout and encodings (AMD APM Vol. 2, Appendix B).

    #[test]
    fn control_area_offsets_are_the_apm_ones() {
        let offsets = [
            (VMCB_INTERCEPT_VECTOR3, 0x0c),
            (VMCB_INTERCEPT_VECTOR4, 0x10),
            (VMCB_IOPM_BASE_PA, 0x40),
            (VMCB_MSRPM_BASE_PA, 0x48),
            (VMCB_ASID, 0x58),
            (VMCB_TLB_CONTROL, 0x5c),
            (VMCB_V_INTR, 0x60),
            (VMCB_INTERRUPT_SHADOW, 0x68),
            (VMCB_EXITCODE, 0x70),
            (VMCB_EXITINFO1, 0x78),
            (VMCB_EXITINFO2, 0x80),
            (VMCB_NP_ENABLE, 0x90),
            (VMCB_EVENTINJ, 0xa8),
            (VMCB_EVENTINJ_ERR, 0xac),
            (VMCB_N_CR3, 0xb0),
            (VMCB_NEXT_RIP, 0xc8),
        ];
        for (offset, expected) in offsets {
            assert_eq!(offset, expected);
        }
    }

    #[test]
    fn save_area_segments_are_sixteen_bytes_apart_from_0x400() {
        let segments = [
            VMCB_SAVE_ES,
            VMCB_SAVE_CS,
            VMCB_SAVE_SS,
            VMCB_SAVE_DS,
            VMCB_SAVE_FS,
            VMCB_SAVE_GS,
            VMCB_SAVE_GDTR,
            VMCB_SAVE_LDTR,
            VMCB_SAVE_IDTR,
            VMCB_SAVE_TR,
        ];
        for (index, offset) in segments.into_iter().enumerate() {
            assert_eq!(offset, 0x400 + 0x10 * index, "segment {index}");
        }
    }

    #[test]
    fn segment_register_fields_are_selector_attrib_limit_base() {
        assert_eq!(SEGMENT_SELECTOR, 0);
        assert_eq!(SEGMENT_ATTRIB, 2);
        assert_eq!(SEGMENT_LIMIT, 4);
        assert_eq!(SEGMENT_BASE, 8);
    }

    #[test]
    fn save_area_register_offsets_are_the_apm_ones() {
        let offsets = [
            (VMCB_SAVE_CPL, 0x4cb),
            (VMCB_SAVE_EFER, 0x4d0),
            (VMCB_SAVE_CR4, 0x548),
            (VMCB_SAVE_CR3, 0x550),
            (VMCB_SAVE_CR0, 0x558),
            (VMCB_SAVE_DR7, 0x560),
            (VMCB_SAVE_DR6, 0x568),
            (VMCB_SAVE_RFLAGS, 0x570),
            (VMCB_SAVE_RIP, 0x578),
            (VMCB_SAVE_RSP, 0x5d8),
            (VMCB_SAVE_RAX, 0x5f8),
            (VMCB_SAVE_G_PAT, 0x668),
        ];
        for (offset, expected) in offsets {
            assert_eq!(offset, expected);
        }
    }

    #[test]
    fn segment_attributes_use_the_svm_packing() {
        // Type 0xb, S, P, L, G for 64-bit code; type 3, S, P, D/B, G for flat data.
        assert_eq!(ATTRIB_CODE_64BIT, 0x0a9b);
        assert_eq!(ATTRIB_DATA_FLAT, 0x0c93);
        assert_eq!(ATTRIB_LDT, 0x0082);
        assert_eq!(ATTRIB_TSS, 0x008b);
        assert_eq!(
            ATTRIB_CODE_64BIT & ATTRIB_DB,
            0,
            "D must be clear when L is set"
        );
    }

    #[test]
    fn intercept_bits_are_the_apm_positions() {
        assert_eq!(INTERCEPT_INTR, 1 << 0);
        assert_eq!(INTERCEPT_VINTR, 1 << 4);
        assert_eq!(INTERCEPT_CPUID, 1 << 18);
        assert_eq!(INTERCEPT_INVD, 1 << 22);
        assert_eq!(INTERCEPT_HLT, 1 << 24);
        assert_eq!(INTERCEPT_IOIO_PROT, 1 << 27);
        assert_eq!(INTERCEPT_MSR_PROT, 1 << 28);
        assert_eq!(INTERCEPT_SHUTDOWN, 1 << 31);
        assert_eq!(INTERCEPT_VMRUN, 1 << 0);
        assert_eq!(INTERCEPT_VMMCALL, 1 << 1);
        assert_eq!(INTERCEPT_VMLOAD, 1 << 2);
        assert_eq!(INTERCEPT_VMSAVE, 1 << 3);
        assert_eq!(INTERCEPT_STGI, 1 << 4);
        assert_eq!(INTERCEPT_CLGI, 1 << 5);
        assert_eq!(INTERCEPT_SKINIT, 1 << 6);
    }

    #[test]
    fn exit_codes_are_the_apm_numbers() {
        assert_eq!(VMEXIT_INTR, 0x60);
        assert_eq!(VMEXIT_VINTR, 0x64);
        assert_eq!(VMEXIT_CPUID, 0x72);
        assert_eq!(VMEXIT_INVD, 0x76);
        assert_eq!(VMEXIT_HLT, 0x78);
        assert_eq!(VMEXIT_IOIO, 0x7b);
        assert_eq!(VMEXIT_MSR, 0x7c);
        assert_eq!(VMEXIT_SHUTDOWN, 0x7f);
        assert_eq!(VMEXIT_VMRUN, 0x80);
        assert_eq!(VMEXIT_SKINIT, 0x86);
        assert_eq!(VMEXIT_NPF, 0x400);
        assert_eq!(VMEXIT_INVALID, u64::MAX);
    }

    #[test]
    fn v_intr_bits_and_flags_are_the_apm_positions() {
        assert_eq!(V_IRQ, 1 << 8);
        assert_eq!(V_IGN_TPR, 1 << 20);
        assert_eq!(V_INTR_MASKING, 1 << 24);
        assert_eq!(INTERRUPT_SHADOW, 1);
        assert_eq!(NP_ENABLE, 1);
    }

    #[test]
    fn permission_maps_are_the_apm_sizes() {
        assert_eq!(IOPM_PAGES * PAGE_SIZE, 12 * 1024);
        assert_eq!(MSRPM_PAGES * PAGE_SIZE, 8 * 1024);
    }

    #[test]
    fn nested_paging_and_nrips_are_distinct_feature_bits() {
        assert_eq!(SVM_FEATURE_NESTED_PAGING, 1 << 0);
        assert_eq!(SVM_FEATURE_NRIPS, 1 << 3);
    }

    // Capability check.

    #[test]
    fn nested_paging_with_svm_enabled_is_usable() {
        assert!(usable_with(SVM_FEATURE_NESTED_PAGING, 0));
    }

    #[test]
    fn missing_nested_paging_makes_it_unusable() {
        assert!(!usable_with(0, 0));
        assert!(!usable_with(SVM_FEATURE_NRIPS, 0));
    }

    #[test]
    fn firmware_that_disabled_svm_makes_it_unusable() {
        assert!(!usable_with(SVM_FEATURE_NESTED_PAGING, VM_CR_SVMDIS));
    }

    #[test]
    fn other_vm_cr_bits_do_not_matter() {
        assert!(usable_with(SVM_FEATURE_NESTED_PAGING, !VM_CR_SVMDIS));
    }

    #[test]
    fn nrips_is_not_needed_to_be_usable() {
        assert!(usable_with(SVM_FEATURE_NESTED_PAGING, 0));
        assert!(usable_with(
            SVM_FEATURE_NESTED_PAGING | SVM_FEATURE_NRIPS,
            0
        ));
    }

    // Next rip.

    #[test]
    fn without_a_saved_next_rip_the_unprefixed_length_is_assumed() {
        assert_eq!(next_rip(RIP, None, 2), RIP + 2);
        assert_eq!(next_rip(RIP, None, 1), RIP + 1);
    }

    #[test]
    fn a_saved_next_rip_wins_over_the_assumed_length() {
        assert_eq!(next_rip(RIP, Some(RIP + 3), 2), RIP + 3);
    }

    #[test]
    fn prefixed_cpuid_resumes_after_the_whole_instruction() {
        // A `rex.W cpuid` or `data16 cpuid` is three bytes; assuming two would re-execute the
        // second one as a new instruction.
        let (exit, next) = decode_exit(VMEXIT_CPUID, 0, 0, RIP, Some(RIP + 3)).unwrap();
        assert_eq!(exit, Exit::Cpuid);
        assert_eq!(next, RIP + 3);
    }

    #[test]
    fn prefixed_hlt_invd_rdmsr_and_wrmsr_resume_at_the_saved_rip() {
        let saved = Some(RIP + 4);
        let cases = [
            (VMEXIT_HLT, 0, Exit::Hlt),
            (VMEXIT_INVD, 0, Exit::Skip),
            (VMEXIT_MSR, 0, Exit::Rdmsr),
            (VMEXIT_MSR, 1, Exit::Wrmsr),
        ];
        for (code, info1, expected) in cases {
            let (exit, next) = decode_exit(code, info1, 0, RIP, saved).unwrap();
            assert_eq!(exit, expected);
            assert_eq!(next, RIP + 4, "{expected:?}");
        }
    }

    #[test]
    fn without_the_feature_the_plain_encodings_are_assumed() {
        assert_eq!(decode(VMEXIT_CPUID), Ok((Exit::Cpuid, RIP + 2)));
        assert_eq!(decode(VMEXIT_HLT), Ok((Exit::Hlt, RIP + 1)));
        assert_eq!(decode(VMEXIT_INVD), Ok((Exit::Skip, RIP + 2)));
        assert_eq!(
            decode_exit(VMEXIT_MSR, 0, 0, RIP, None),
            Ok((Exit::Rdmsr, RIP + 2))
        );
        assert_eq!(
            decode_exit(VMEXIT_MSR, 1, 0, RIP, None),
            Ok((Exit::Wrmsr, RIP + 2))
        );
    }

    #[test]
    fn the_saved_rip_agrees_with_the_assumed_one_for_plain_encodings() {
        for code in [VMEXIT_CPUID, VMEXIT_INVD, VMEXIT_MSR] {
            assert_eq!(
                decode_exit(code, 0, 0, RIP, Some(SAVED_PLAIN)),
                decode_exit(code, 0, 0, RIP, None),
                "code {code:#x}"
            );
        }
    }

    // Exit decoding.

    #[test]
    fn interrupt_and_window_exits_leave_rip_alone() {
        assert_eq!(decode(VMEXIT_INTR), Ok((Exit::HostInterrupt, RIP)));
        assert_eq!(decode(VMEXIT_VINTR), Ok((Exit::IntWindow, RIP)));
    }

    #[test]
    fn a_saved_rip_is_ignored_for_exits_that_do_not_skip_an_instruction() {
        assert_eq!(
            decode_exit(VMEXIT_INTR, 0, 0, RIP, Some(RIP + 9)),
            Ok((Exit::HostInterrupt, RIP))
        );
        assert_eq!(
            decode_exit(VMEXIT_SHUTDOWN, 0, 0, RIP, Some(RIP + 9)),
            Ok((Exit::TripleFault, RIP))
        );
    }

    #[test]
    fn shutdown_is_a_triple_fault() {
        assert_eq!(decode(VMEXIT_SHUTDOWN), Ok((Exit::TripleFault, RIP)));
    }

    #[test]
    fn svm_instructions_raise_invalid_opcode() {
        // VMRUN, VMMCALL, VMLOAD, VMSAVE, STGI, CLGI, SKINIT.
        for code in 0x80..=0x86 {
            assert_eq!(decode(code), Ok((Exit::Undefined, RIP)), "code {code:#x}");
        }
    }

    #[test]
    fn codes_next_to_the_svm_instructions_are_unhandled() {
        for code in [0x7e, 0x87, 0x88, 0x61, 0x7d, 0x401] {
            assert_eq!(decode(code), Ok((Exit::Other(code), RIP)), "code {code:#x}");
        }
    }

    #[test]
    fn rejected_guest_state_is_an_error() {
        assert_eq!(decode(VMEXIT_INVALID), Err(DecodeError::EntryRejected));
    }

    #[test]
    fn nested_page_faults_report_address_and_direction() {
        assert_eq!(
            decode_exit(VMEXIT_NPF, 0b110, 0xfee0_0000, RIP, None),
            Ok((
                Exit::Fault {
                    gpa: 0xfee0_0000,
                    write: true
                },
                RIP
            ))
        );
        assert_eq!(
            decode_exit(VMEXIT_NPF, 0b101, 0x1000, RIP, None),
            Ok((
                Exit::Fault {
                    gpa: 0x1000,
                    write: false
                },
                RIP
            ))
        );
    }

    fn io(info1: u64, info2: u64) -> Result<(Exit, u64), DecodeError> {
        decode_exit(VMEXIT_IOIO, info1, info2, RIP, Some(RIP + 100))
    }

    #[test]
    fn byte_out_decodes_with_its_port_and_resumes_at_exitinfo2() {
        // OUT DX, AL to 0x3f8: size-8 bit, type OUT (bit 0 clear).
        assert_eq!(
            io((0x3f8 << 16) | IOIO_SIZE_8, RIP + 1),
            Ok((
                Exit::Io {
                    port: 0x3f8,
                    size: 1,
                    write: true
                },
                RIP + 1
            ))
        );
    }

    #[test]
    fn word_in_decodes_as_a_read_of_two_bytes() {
        assert_eq!(
            io((0x60 << 16) | IOIO_SIZE_16 | IOIO_TYPE_IN, RIP + 2),
            Ok((
                Exit::Io {
                    port: 0x60,
                    size: 2,
                    write: false
                },
                RIP + 2
            ))
        );
    }

    #[test]
    fn io_without_a_size_bit_is_four_bytes() {
        assert_eq!(
            io(0xcf8 << 16, RIP + 1),
            Ok((
                Exit::Io {
                    port: 0xcf8,
                    size: 4,
                    write: true
                },
                RIP + 1
            ))
        );
    }

    #[test]
    fn io_never_takes_the_next_rip_from_the_saved_field() {
        // EXITINFO2 is exact even with a prefix; the saved NRIP must not override it.
        let (_, next) = io(0x80 << 16, RIP + 3).unwrap();
        assert_eq!(next, RIP + 3);
    }

    #[test]
    fn io_port_uses_the_whole_upper_half_of_exitinfo1() {
        let Ok((Exit::Io { port, .. }, _)) = io(0xffff << 16, 0) else {
            panic!("not an I/O exit");
        };
        assert_eq!(port, 0xffff);
    }

    #[test]
    fn io_decoding_ignores_the_address_size_and_segment_bits() {
        // Bits 6:4 address size, 12:10 segment: only 5:4 here are operand sizes.
        let address_size_64 = 1 << 9;
        let segment = 0b011 << 10;
        assert_eq!(
            io((0x70 << 16) | IOIO_SIZE_8 | address_size_64 | segment, 0),
            Ok((
                Exit::Io {
                    port: 0x70,
                    size: 1,
                    write: true
                },
                0
            ))
        );
    }

    #[test]
    fn string_io_is_refused_with_its_port() {
        let info1 = (0x1f0 << 16) | IOIO_STRING | IOIO_TYPE_IN;
        assert_eq!(io(info1, 0), Err(DecodeError::StringIo { port: 0x1f0 }));
    }

    #[test]
    fn rdmsr_and_wrmsr_are_told_apart_by_exitinfo1() {
        assert_eq!(MSR_EXIT_RDMSR, 0);
        assert_eq!(
            decode_exit(VMEXIT_MSR, 0, 0, RIP, None).unwrap().0,
            Exit::Rdmsr
        );
        assert_eq!(
            decode_exit(VMEXIT_MSR, 1, 0, RIP, None).unwrap().0,
            Exit::Wrmsr
        );
    }

    // MSR permission map.

    const MSRPM_BYTES: usize = MSRPM_PAGES * PAGE_SIZE;

    fn full_map() -> Vec<u8> {
        vec![0xff; MSRPM_BYTES]
    }

    fn cleared_bits(map: &[u8]) -> Vec<(usize, u8)> {
        let mut out = Vec::new();
        for (byte, value) in map.iter().enumerate() {
            for bit in 0..8 {
                if value & (1 << bit) == 0 {
                    out.push((byte, bit));
                }
            }
        }
        out
    }

    #[test]
    fn a_low_msr_clears_its_two_bits() {
        let mut map = full_map();
        // MSR 0x10: bits 32 and 33, the low two bits of byte 4.
        pass_through(&mut map, arch::MSR_TSC);
        assert_eq!(cleared_bits(&map), [(4, 0), (4, 1)]);
    }

    #[test]
    fn two_bits_per_msr_put_four_msrs_in_a_byte() {
        let mut map = full_map();
        // MSR 0x277: bit 0x277 * 2 = 1262, byte 157, bits 6 and 7.
        pass_through(&mut map, arch::MSR_PAT);
        assert_eq!(cleared_bits(&map), [(157, 6), (157, 7)]);
    }

    #[test]
    fn a_high_msr_uses_the_range_that_starts_0x800_in() {
        let mut map = full_map();
        // MSR 0xc0000081 is index 0x81: bit 258, byte 32, bits 2 and 3.
        pass_through(&mut map, arch::MSR_STAR);
        assert_eq!(cleared_bits(&map), [(0x800 + 32, 2), (0x800 + 32, 3)]);
    }

    #[test]
    fn the_first_high_msr_is_the_first_bits_of_the_second_range() {
        let mut map = full_map();
        pass_through(&mut map, arch::MSR_HIGH_RANGE_START);
        assert_eq!(cleared_bits(&map), [(0x800, 0), (0x800, 1)]);
    }

    #[test]
    fn the_last_msr_of_each_range_is_at_the_end_of_its_two_kilobytes() {
        let mut map = full_map();
        pass_through(&mut map, arch::MSR_LOW_RANGE_END);
        pass_through(&mut map, arch::MSR_HIGH_RANGE_END);
        assert_eq!(
            cleared_bits(&map),
            [(0x7ff, 6), (0x7ff, 7), (0xfff, 6), (0xfff, 7)]
        );
    }

    #[test]
    fn msrs_outside_both_ranges_leave_the_map_alone() {
        for msr in [
            arch::MSR_LOW_RANGE_END + 1,
            arch::MSR_HIGH_RANGE_START - 1,
            arch::MSR_HIGH_RANGE_END + 1,
            0xc001_0000,
            0xc001_0114,
            u32::MAX,
        ] {
            let mut map = full_map();
            pass_through(&mut map, msr);
            assert_eq!(map, full_map(), "msr {msr:#x}");
        }
    }

    #[test]
    fn neighbouring_msrs_keep_their_own_bits() {
        let mut map = full_map();
        pass_through(&mut map, arch::MSR_LSTAR);
        // STAR (index 0x81) and CSTAR (0x83) share a byte with LSTAR (0x82) and still exit.
        let byte = map[0x800 + 32];
        assert_eq!(byte, 0b1100_1111);
    }

    #[test]
    fn all_pass_through_msrs_clear_two_bits_each_and_efer_still_exits() {
        let mut map = full_map();
        for msr in arch::PASS_THROUGH_MSRS {
            pass_through(&mut map, msr);
        }
        assert_eq!(cleared_bits(&map).len(), 2 * arch::PASS_THROUGH_MSRS.len());
        // EFER is index 0x80 of the high range: byte 32, bits 0 and 1.
        assert_eq!(map[0x800 + 32] & 0b11, 0b11);
    }

    #[test]
    fn the_third_msr_range_is_left_to_exit() {
        let mut map = full_map();
        for msr in arch::PASS_THROUGH_MSRS {
            pass_through(&mut map, msr);
        }
        assert!(map[0x1000..].iter().all(|&b| b == 0xff));
    }
}
