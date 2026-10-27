//! Intel VMX backend with EPT and unrestricted guest. Field encodings, capability MSRs and bit
//! positions are from the Intel SDM Vol. 3 (VMCS fields, Appendix A and B). Never run on real
//! hardware yet (plans/12-uefi.md). Pointers are used as physical addresses throughout: UEFI runs
//! with an identity map.

use alloc::format;
use alloc::string::String;
use core::arch::asm;
use uemulator::BootState;

use super::arch::{self, DescriptorTableRegister, Gprs};
use super::{Exit, PAGE_SIZE, Vcpu, alloc_pages};

const MSR_VMX_BASIC: u32 = 0x480;
const MSR_VMX_PINBASED: u32 = 0x481;
const MSR_VMX_PROCBASED: u32 = 0x482;
const MSR_VMX_EXIT: u32 = 0x483;
const MSR_VMX_ENTRY: u32 = 0x484;
const MSR_VMX_CR0_FIXED0: u32 = 0x486;
const MSR_VMX_CR0_FIXED1: u32 = 0x487;
const MSR_VMX_CR4_FIXED0: u32 = 0x488;
const MSR_VMX_CR4_FIXED1: u32 = 0x489;
const MSR_VMX_PROCBASED2: u32 = 0x48b;
const MSR_VMX_EPT_VPID_CAP: u32 = 0x48c;
const MSR_VMX_TRUE_PINBASED: u32 = 0x48d;
const MSR_VMX_TRUE_PROCBASED: u32 = 0x48e;
const MSR_VMX_TRUE_EXIT: u32 = 0x48f;
const MSR_VMX_TRUE_ENTRY: u32 = 0x490;

/// IA32_VMX_BASIC bits 30:0 are the VMCS revision identifier; bit 31 of the region header must
/// stay clear.
const VMX_BASIC_REVISION_MASK: u32 = 0x7fff_ffff;
/// IA32_VMX_BASIC bit 55: the IA32_VMX_TRUE_* control MSRs exist.
const VMX_BASIC_TRUE_CONTROLS: u64 = 1 << 55;

/// IA32_VMX_EPT_VPID_CAP bits: 4-level page walk, write-back EPT memory type, 2 MiB pages.
const EPT_CAP_PAGE_WALK_4: u64 = 1 << 6;
const EPT_CAP_WRITE_BACK: u64 = 1 << 14;
const EPT_CAP_2MIB_PAGES: u64 = 1 << 16;

/// IA32_FEATURE_CONTROL: lock bit, and "VMXON outside SMX" enable.
const FEATURE_LOCK: u64 = 1 << 0;
const FEATURE_VMXON_OUTSIDE_SMX: u64 = 1 << 2;
const CR4_VMXE: u64 = 1 << 13;

// VMCS field encodings, named as in the SDM. The `*_ES_*` ones are the first of a group of
// per-segment fields 2 apart, in the order of `SEGMENT_*` below.
const VMCS_MSR_BITMAP_ADDRESS: u64 = 0x2004;
const VMCS_VMEXIT_MSR_STORE_ADDRESS: u64 = 0x2006;
const VMCS_VMEXIT_MSR_LOAD_ADDRESS: u64 = 0x2008;
const VMCS_VMENTRY_MSR_LOAD_ADDRESS: u64 = 0x200a;
const VMCS_EPT_POINTER: u64 = 0x201a;
const VMCS_GUEST_PHYSICAL_ADDRESS: u64 = 0x2400;
const VMCS_LINK_POINTER: u64 = 0x2800;
const VMCS_GUEST_IA32_DEBUGCTL: u64 = 0x2802;
const VMCS_GUEST_IA32_PAT: u64 = 0x2804;
const VMCS_GUEST_IA32_EFER: u64 = 0x2806;
const VMCS_HOST_IA32_PAT: u64 = 0x2c00;
const VMCS_HOST_IA32_EFER: u64 = 0x2c02;
const VMCS_PIN_BASED_CONTROLS: u64 = 0x4000;
const VMCS_PRIMARY_PROC_BASED_CONTROLS: u64 = 0x4002;
const VMCS_EXCEPTION_BITMAP: u64 = 0x4004;
const VMCS_PAGE_FAULT_ERROR_CODE_MASK: u64 = 0x4006;
const VMCS_PAGE_FAULT_ERROR_CODE_MATCH: u64 = 0x4008;
const VMCS_CR3_TARGET_COUNT: u64 = 0x400a;
const VMCS_VMEXIT_CONTROLS: u64 = 0x400c;
const VMCS_VMEXIT_MSR_STORE_COUNT: u64 = 0x400e;
const VMCS_VMEXIT_MSR_LOAD_COUNT: u64 = 0x4010;
const VMCS_VMENTRY_CONTROLS: u64 = 0x4012;
const VMCS_VMENTRY_MSR_LOAD_COUNT: u64 = 0x4014;
const VMCS_VMENTRY_INTERRUPTION_INFO: u64 = 0x4016;
const VMCS_VMENTRY_EXCEPTION_ERROR_CODE: u64 = 0x4018;
const VMCS_VMENTRY_INSTRUCTION_LENGTH: u64 = 0x401a;
const VMCS_SECONDARY_PROC_BASED_CONTROLS: u64 = 0x401e;
const VMCS_VM_INSTRUCTION_ERROR: u64 = 0x4400;
const VMCS_EXIT_REASON: u64 = 0x4402;
const VMCS_IDT_VECTORING_INFO: u64 = 0x4408;
const VMCS_IDT_VECTORING_ERROR_CODE: u64 = 0x440a;
const VMCS_VMEXIT_INSTRUCTION_LENGTH: u64 = 0x440c;
const VMCS_CR0_GUEST_HOST_MASK: u64 = 0x6000;
const VMCS_CR4_GUEST_HOST_MASK: u64 = 0x6002;
const VMCS_CR0_READ_SHADOW: u64 = 0x6004;
const VMCS_CR4_READ_SHADOW: u64 = 0x6006;
const VMCS_EXIT_QUALIFICATION: u64 = 0x6400;

const VMCS_GUEST_ES_SELECTOR: u64 = 0x0800;
const VMCS_HOST_ES_SELECTOR: u64 = 0x0c00;
/// The limit group continues after the segment registers with GDTR and IDTR.
const VMCS_GUEST_ES_LIMIT: u64 = 0x4800;
const VMCS_GUEST_ES_ACCESS_RIGHTS: u64 = 0x4814;
/// The base group continues after the segment registers with GDTR and IDTR.
const VMCS_GUEST_ES_BASE: u64 = 0x6806;
const VMCS_GUEST_INTERRUPTIBILITY_STATE: u64 = 0x4824;
const VMCS_GUEST_ACTIVITY_STATE: u64 = 0x4826;
const VMCS_GUEST_IA32_SYSENTER_CS: u64 = 0x482a;
const VMCS_HOST_IA32_SYSENTER_CS: u64 = 0x4c00;
const VMCS_GUEST_CR0: u64 = 0x6800;
const VMCS_GUEST_CR3: u64 = 0x6802;
const VMCS_GUEST_CR4: u64 = 0x6804;
const VMCS_GUEST_DR7: u64 = 0x681a;
const VMCS_GUEST_RSP: u64 = 0x681c;
const VMCS_GUEST_RIP: u64 = 0x681e;
const VMCS_GUEST_RFLAGS: u64 = 0x6820;
const VMCS_GUEST_PENDING_DEBUG_EXCEPTIONS: u64 = 0x6822;
const VMCS_GUEST_IA32_SYSENTER_ESP: u64 = 0x6824;
const VMCS_GUEST_IA32_SYSENTER_EIP: u64 = 0x6826;
const VMCS_HOST_CR0: u64 = 0x6c00;
const VMCS_HOST_CR3: u64 = 0x6c02;
const VMCS_HOST_CR4: u64 = 0x6c04;
const VMCS_HOST_FS_BASE: u64 = 0x6c06;
const VMCS_HOST_GS_BASE: u64 = 0x6c08;
const VMCS_HOST_TR_BASE: u64 = 0x6c0a;
const VMCS_HOST_GDTR_BASE: u64 = 0x6c0c;
const VMCS_HOST_IDTR_BASE: u64 = 0x6c0e;
const VMCS_HOST_IA32_SYSENTER_ESP: u64 = 0x6c10;
const VMCS_HOST_IA32_SYSENTER_EIP: u64 = 0x6c12;
/// Read by the `uemu_vmx_enter` stub in `arch.rs`, which writes the host RSP itself.
pub(super) const VMCS_HOST_RSP: u64 = 0x6c14;
const VMCS_HOST_RIP: u64 = 0x6c16;

/// Index of each segment register within the per-segment VMCS field groups.
const SEGMENT_ES: u64 = 0;
const SEGMENT_CS: u64 = 1;
const SEGMENT_SS: u64 = 2;
const SEGMENT_DS: u64 = 3;
const SEGMENT_FS: u64 = 4;
const SEGMENT_GS: u64 = 5;
const SEGMENT_LDTR: u64 = 6;
const SEGMENT_TR: u64 = 7;
const SEGMENT_GDTR: u64 = 8;
const SEGMENT_IDTR: u64 = 9;
/// Fields of one group are 2 apart.
const VMCS_GROUP_STRIDE: u64 = 2;

// Guest segment access rights: 3:0 type, 4 S, 7 P, 13 L, 14 D/B, 15 G, 16 unusable (a different
// packing from SVM's).
const AR_TYPE_CODE_EXEC_READ_ACCESSED: u64 = 0xb;
const AR_TYPE_DATA_READ_WRITE_ACCESSED: u64 = 0x3;
const AR_TYPE_BUSY_TSS: u64 = 0xb;
const AR_S: u64 = 1 << 4;
const AR_P: u64 = 1 << 7;
const AR_L: u64 = 1 << 13;
const AR_DB: u64 = 1 << 14;
const AR_G: u64 = 1 << 15;
const AR_UNUSABLE: u64 = 1 << 16;
const AR_CODE_64BIT: u64 = AR_TYPE_CODE_EXEC_READ_ACCESSED | AR_S | AR_P | AR_L | AR_G;
const AR_DATA_FLAT: u64 = AR_TYPE_DATA_READ_WRITE_ACCESSED | AR_S | AR_P | AR_DB | AR_G;
const AR_TSS: u64 = AR_TYPE_BUSY_TSS | AR_P;

// Pin-based controls.
const PIN_EXTERNAL_INTERRUPT_EXITING: u32 = 1 << 0;
// Primary processor-based controls.
const PROC_INTERRUPT_WINDOW_EXITING: u32 = 1 << 2;
const PROC_HLT_EXITING: u32 = 1 << 7;
const PROC_UNCONDITIONAL_IO_EXITING: u32 = 1 << 24;
const PROC_USE_MSR_BITMAPS: u32 = 1 << 28;
const PROC_ACTIVATE_SECONDARY_CONTROLS: u32 = 1 << 31;
// Secondary processor-based controls.
const PROC2_ENABLE_EPT: u32 = 1 << 1;
const PROC2_UNRESTRICTED_GUEST: u32 = 1 << 7;
// VM-exit controls.
const EXIT_SAVE_DEBUG_CONTROLS: u32 = 1 << 2;
const EXIT_HOST_ADDRESS_SPACE_SIZE: u32 = 1 << 9;
const EXIT_SAVE_IA32_PAT: u32 = 1 << 18;
const EXIT_LOAD_IA32_PAT: u32 = 1 << 19;
const EXIT_SAVE_IA32_EFER: u32 = 1 << 20;
const EXIT_LOAD_IA32_EFER: u32 = 1 << 21;
// VM-entry controls.
const ENTRY_LOAD_DEBUG_CONTROLS: u32 = 1 << 2;
const ENTRY_IA32E_MODE_GUEST: u32 = 1 << 9;
const ENTRY_LOAD_IA32_PAT: u32 = 1 << 14;
const ENTRY_LOAD_IA32_EFER: u32 = 1 << 15;

/// EPT pointer: bits 2:0 memory type (6 = write-back), bits 5:3 page-walk length minus one.
const EPTP_MEMORY_TYPE_WRITE_BACK: u64 = 6;
const EPTP_PAGE_WALK_LENGTH_SHIFT: u32 = 3;
const EPTP_PAGE_WALK_LENGTH_4_MINUS_1: u64 = 3;

// Basic exit reasons.
const REASON_EXTERNAL_INTERRUPT: u64 = 1;
const REASON_TRIPLE_FAULT: u64 = 2;
const REASON_INTERRUPT_WINDOW: u64 = 7;
const REASON_CPUID: u64 = 10;
const REASON_HLT: u64 = 12;
const REASON_INVD: u64 = 13;
const REASON_VMCALL: u64 = 18;
const REASON_VMXON: u64 = 27;
const REASON_IO_INSTRUCTION: u64 = 30;
const REASON_RDMSR: u64 = 31;
const REASON_WRMSR: u64 = 32;
const REASON_EPT_VIOLATION: u64 = 48;
const REASON_XSETBV: u64 = 55;
/// Bit 31 of the exit reason: the exit is a failed VM entry. Bits 15:0 are the basic reason.
const EXIT_REASON_ENTRY_FAILURE: u64 = 1 << 31;
const EXIT_REASON_BASIC_MASK: u64 = 0xffff;

// Exit qualification for an I/O instruction: 2:0 size minus one, 3 direction (1 = IN), 4 string,
// 31:16 port.
const IO_QUAL_SIZE_MASK: u64 = 0b111;
const IO_QUAL_DIRECTION_IN: u64 = 1 << 3;
const IO_QUAL_STRING: u64 = 1 << 4;
const IO_QUAL_PORT_SHIFT: u32 = 16;
/// Exit qualification for an EPT violation: bit 1 is a data write.
const EPT_VIOLATION_QUAL_WRITE: u64 = 1 << 1;

/// Interruption-information types 4 (software interrupt), 5 (privileged software exception) and
/// 6 (software exception) are delivered with an instruction length; the range is bounded by the
/// first and last.
const EVENT_TYPE_MASK: u32 = 0b111;
const EVENT_TYPE_SOFTWARE_INTERRUPT: u32 = 4;
const EVENT_TYPE_SOFTWARE_EXCEPTION: u32 = 6;

/// Interruptibility state: blocking by STI (bit 0) and by MOV SS (bit 1).
const INTERRUPTIBILITY_BLOCKING_BY_STI_AND_MOV_SS: u64 = 0b11;

/// The MSR bitmap is 4 KiB: read bitmaps for the low and high MSR ranges, then the two write
/// bitmaps. Each range covers 0x2000 MSRs, so 1 KiB.
const MSR_BITMAP_HIGH_RANGE_OFFSET: usize = 1024;
const MSR_BITMAP_WRITE_OFFSET: usize = 2048;

/// One page of guest entries and one of host entries.
const MSR_LIST_PAGES: usize = 2;
/// Each MSR-list entry is 16 bytes: the MSR index (upper half reserved), then the value.
const MSR_LIST_ENTRY_U64S: usize = 2;

/// Limit of a minimal 64-bit TSS (104 bytes, minus one).
const TSS_LIMIT: u64 = 0x67;
/// Descriptor access byte: present, type 9 (available 64-bit TSS).
const TSS_ACCESS: u64 = 0x89;
const SELECTOR_RPL_AND_TABLE_MASK: u16 = 0b111;
// Bit positions inside the low qword of a 64-bit TSS descriptor.
const DESCRIPTOR_BASE_15_0_POS: u32 = 16;
const DESCRIPTOR_BASE_23_16_POS: u32 = 32;
const DESCRIPTOR_ACCESS_POS: u32 = 40;
const DESCRIPTOR_BASE_31_24_POS: u32 = 56;
// Where those pieces sit in the base address itself.
const BASE_23_16_SHIFT: u32 = 16;
const BASE_31_24_SHIFT: u32 = 24;
const BASE_63_32_SHIFT: u32 = 32;
/// A system descriptor takes two GDT slots in long mode.
const SYSTEM_DESCRIPTOR_SIZE: usize = 16;

/// Guest MSRs the CPU does not switch by itself; they live in the VM-entry load and VM-exit store
/// lists, with the host's values restored from the VM-exit load list.
const SWITCHED_MSRS: [u32; 5] = [
    arch::MSR_STAR,
    arch::MSR_LSTAR,
    arch::MSR_CSTAR,
    arch::MSR_SFMASK,
    arch::MSR_KERNEL_GS_BASE,
];

/// Field of segment `segment` in the per-segment group starting at `group`.
fn segment_field(group: u64, segment: u64) -> u64 {
    group + VMCS_GROUP_STRIDE * segment
}

// vmwrite and vmread report failure in CF/ZF, which is not checked: a failing field access here
// would be a wrong encoding, and the VM entry then fails visibly. They modify flags, so no
// `preserves_flags`.
#[inline]
fn vmwrite(field: u64, value: u64) {
    unsafe {
        asm!("vmwrite {f}, {v}", f = in(reg) field, v = in(reg) value, options(nostack));
    }
}

#[inline]
fn vmread(field: u64) -> u64 {
    let v: u64;
    unsafe {
        asm!("vmread {v}, {f}", f = in(reg) field, v = out(reg) v, options(nostack));
    }
    v
}

/// Runs VMXON (`"on"`), VMCLEAR (`"clear"`) or, for anything else, VMPTRLD on the region at
/// physical address `pa`. Each takes a pointer to a memory operand holding that address, hence
/// the local. `setna` turns VMfailInvalid (CF) or VMfailValid (ZF) into a nonzero byte.
fn vm_op(op: &str, pa: u64) -> bool {
    let failed: u8;
    unsafe {
        let addr = &pa as *const u64;
        match op {
            "on" => {
                asm!("vmxon [{a}]", "setna {failed}", a = in(reg) addr, failed = out(reg_byte) failed, options(nostack))
            }
            "clear" => {
                asm!("vmclear [{a}]", "setna {failed}", a = in(reg) addr, failed = out(reg_byte) failed, options(nostack))
            }
            _ => {
                asm!("vmptrld [{a}]", "setna {failed}", a = in(reg) addr, failed = out(reg_byte) failed, options(nostack))
            }
        }
    }
    failed == 0
}

pub fn usable() -> bool {
    if arch::cpuid(arch::CPUID_LEAF_FEATURES, 0)[2] & arch::CPUID_1_ECX_VMX == 0 {
        return false;
    }
    let feature_control = arch::rdmsr(arch::MSR_FEATURE_CONTROL);
    if feature_control & FEATURE_LOCK != 0 && feature_control & FEATURE_VMXON_OUTSIDE_SMX == 0 {
        return false;
    }
    // Without the TRUE control MSRs, CR3 accesses would be forced to exit.
    if arch::rdmsr(MSR_VMX_BASIC) & VMX_BASIC_TRUE_CONTROLS == 0 {
        return false;
    }
    let proc = arch::rdmsr(MSR_VMX_PROCBASED);
    let proc2 = arch::rdmsr(MSR_VMX_PROCBASED2);
    let ept_vpid_cap = arch::rdmsr(MSR_VMX_EPT_VPID_CAP);
    let ept_caps = EPT_CAP_PAGE_WALK_4 | EPT_CAP_WRITE_BACK | EPT_CAP_2MIB_PAGES;
    let proc2_wanted = PROC2_ENABLE_EPT | PROC2_UNRESTRICTED_GUEST;
    // Secondary controls with EPT and unrestricted guest, 4-level walks, 2 MiB pages, WB memory.
    // The high half of a control MSR says which bits may be set.
    (proc >> 32) as u32 & PROC_ACTIVATE_SECONDARY_CONTROLS != 0
        && (proc2 >> 32) as u32 & proc2_wanted == proc2_wanted
        && ept_vpid_cap & ept_caps == ept_caps
}

/// Applies the capability MSR `msr` to a control word: bits set in its low half must be one, bits
/// clear in its high half cannot be set.
fn adjust(msr: u32, wanted: u32) -> Result<u32, String> {
    let capability = arch::rdmsr(msr);
    let (must_be_one, may_be_one) = (capability as u32, (capability >> 32) as u32);
    if wanted & !may_be_one != 0 {
        return Err(format!(
            "VMX control MSR {msr:#x} cannot enable {:#x}",
            wanted & !may_be_one
        ));
    }
    Ok(wanted | must_be_one)
}

pub struct Vmx {
    gprs: Gprs,
    /// Primary processor-based controls without the interrupt-window bit, which `run` toggles.
    proc_controls: u32,
    /// VMLAUNCH has succeeded; later entries use VMRESUME.
    launched: bool,
    /// rip after the instruction behind the last exit, for `skip`.
    next_rip: u64,
    /// Length of that instruction.
    insn_len: u64,
    /// Interruption information (without the valid bit) and error code to inject on the next
    /// entry.
    event: Option<(u32, Option<u32>)>,
    /// Ask for an exit as soon as the guest can take an interrupt.
    window: bool,
    /// A VM exit leaves the GDTR and IDTR limits at 0xffff, so `run` reloads these.
    host_gdt: DescriptorTableRegister,
    host_idt: DescriptorTableRegister,
}

impl Vmx {
    pub fn new(ept_root: u64, boot: &BootState) -> Result<Self, String> {
        let basic = arch::rdmsr(MSR_VMX_BASIC);
        let feature_control = arch::rdmsr(arch::MSR_FEATURE_CONTROL);
        // Firmware that left the MSR unlocked: enable VMXON and lock it, as VMXON needs. A locked
        // MSR with VMX off was already refused by `usable`.
        if feature_control & FEATURE_LOCK == 0 {
            arch::wrmsr(
                arch::MSR_FEATURE_CONTROL,
                feature_control | FEATURE_LOCK | FEATURE_VMXON_OUTSIDE_SMX,
            );
        }
        // VMXON requires CR0 and CR4 to satisfy the fixed-bit MSRs: a bit set in FIXED0 must be
        // one, a bit clear in FIXED1 must be zero.
        let apply_fixed_bits = |value: u64, fixed0: u32, fixed1: u32| {
            (value | arch::rdmsr(fixed0)) & arch::rdmsr(fixed1)
        };
        arch::set_cr0(apply_fixed_bits(
            arch::cr0(),
            MSR_VMX_CR0_FIXED0,
            MSR_VMX_CR0_FIXED1,
        ));
        arch::set_cr4(apply_fixed_bits(
            arch::cr4() | CR4_VMXE,
            MSR_VMX_CR4_FIXED0,
            MSR_VMX_CR4_FIXED1,
        ));

        let vmxon_region = alloc_pages(1)?;
        let vmcs = alloc_pages(1)?;
        let msr_bitmap = alloc_pages(1)?;
        // Page 0: guest values (VM-entry load, VM-exit store). Page 1: host values (VM-exit load).
        let msr_list = alloc_pages(MSR_LIST_PAGES)?;
        // The first dword of the VMXON region and of the VMCS is the revision identifier.
        let revision = (basic as u32) & VMX_BASIC_REVISION_MASK;
        unsafe {
            (vmxon_region as *mut u32).write(revision);
            (vmcs as *mut u32).write(revision);
            // All ones: every MSR access exits unless cleared below.
            core::ptr::write_bytes(msr_bitmap, 0xff, PAGE_SIZE);
        }
        if !vm_op("on", vmxon_region as u64) {
            return Err("VMXON failed".into());
        }
        if !vm_op("clear", vmcs as u64) || !vm_op("ptrld", vmcs as u64) {
            return Err("VMCLEAR/VMPTRLD failed".into());
        }

        for msr in arch::PASS_THROUGH_MSRS {
            pass_through(msr_bitmap, msr);
        }

        // The TRUE capability MSRs report which "default1" control bits can really be cleared.
        let true_controls = basic & VMX_BASIC_TRUE_CONTROLS != 0;
        let pick = |plain: u32, true_msr: u32| if true_controls { true_msr } else { plain };
        let pin = adjust(
            pick(MSR_VMX_PINBASED, MSR_VMX_TRUE_PINBASED),
            PIN_EXTERNAL_INTERRUPT_EXITING,
        )?;
        let proc = adjust(
            pick(MSR_VMX_PROCBASED, MSR_VMX_TRUE_PROCBASED),
            PROC_HLT_EXITING
                | PROC_UNCONDITIONAL_IO_EXITING
                | PROC_USE_MSR_BITMAPS
                | PROC_ACTIVATE_SECONDARY_CONTROLS,
        )?;
        let proc2 = adjust(
            MSR_VMX_PROCBASED2,
            PROC2_ENABLE_EPT | PROC2_UNRESTRICTED_GUEST,
        )?;
        let exit = adjust(
            pick(MSR_VMX_EXIT, MSR_VMX_TRUE_EXIT),
            EXIT_SAVE_DEBUG_CONTROLS
                | EXIT_HOST_ADDRESS_SPACE_SIZE
                | EXIT_SAVE_IA32_PAT
                | EXIT_LOAD_IA32_PAT
                | EXIT_SAVE_IA32_EFER
                | EXIT_LOAD_IA32_EFER,
        )?;
        let entry = adjust(
            pick(MSR_VMX_ENTRY, MSR_VMX_TRUE_ENTRY),
            ENTRY_LOAD_DEBUG_CONTROLS
                | ENTRY_IA32E_MODE_GUEST
                | ENTRY_LOAD_IA32_PAT
                | ENTRY_LOAD_IA32_EFER,
        )?;

        vmwrite(VMCS_PIN_BASED_CONTROLS, pin as u64);
        vmwrite(VMCS_PRIMARY_PROC_BASED_CONTROLS, proc as u64);
        vmwrite(VMCS_SECONDARY_PROC_BASED_CONTROLS, proc2 as u64);
        vmwrite(VMCS_VMEXIT_CONTROLS, exit as u64);
        vmwrite(VMCS_VMENTRY_CONTROLS, entry as u64);
        vmwrite(VMCS_MSR_BITMAP_ADDRESS, msr_bitmap as u64);
        vmwrite(
            VMCS_EPT_POINTER,
            ept_root
                | EPTP_MEMORY_TYPE_WRITE_BACK
                | (EPTP_PAGE_WALK_LENGTH_4_MINUS_1 << EPTP_PAGE_WALK_LENGTH_SHIFT),
        );
        vmwrite(VMCS_EXCEPTION_BITMAP, 0);
        vmwrite(VMCS_PAGE_FAULT_ERROR_CODE_MASK, 0);
        vmwrite(VMCS_PAGE_FAULT_ERROR_CODE_MATCH, 0);
        vmwrite(VMCS_CR3_TARGET_COUNT, 0);
        // All ones: there is no shadow VMCS.
        vmwrite(VMCS_LINK_POINTER, u64::MAX);

        let host_list = unsafe { msr_list.add(PAGE_SIZE) };
        for (i, msr) in SWITCHED_MSRS.iter().enumerate() {
            unsafe {
                let guest_entry = (msr_list as *mut u64).add(i * MSR_LIST_ENTRY_U64S);
                guest_entry.write(*msr as u64);
                guest_entry.add(1).write(0);
                let host_entry = (host_list as *mut u64).add(i * MSR_LIST_ENTRY_U64S);
                host_entry.write(*msr as u64);
                host_entry.add(1).write(arch::rdmsr(*msr));
            }
        }
        vmwrite(VMCS_VMENTRY_MSR_LOAD_ADDRESS, msr_list as u64);
        vmwrite(VMCS_VMENTRY_MSR_LOAD_COUNT, SWITCHED_MSRS.len() as u64);
        vmwrite(VMCS_VMEXIT_MSR_STORE_ADDRESS, msr_list as u64);
        vmwrite(VMCS_VMEXIT_MSR_STORE_COUNT, SWITCHED_MSRS.len() as u64);
        vmwrite(VMCS_VMEXIT_MSR_LOAD_ADDRESS, host_list as u64);
        vmwrite(VMCS_VMEXIT_MSR_LOAD_COUNT, SWITCHED_MSRS.len() as u64);

        let host_gdt = arch::sgdt();
        let host_idt = arch::sidt();
        let task_register = host_task_register(&host_gdt)?;
        // Read again: `host_task_register` may have loaded a new GDT.
        let host_gdt = arch::sgdt();
        let tr_base = tss_base(&host_gdt, task_register);
        for (i, selector) in [
            arch::es(),
            arch::cs(),
            arch::ss(),
            arch::ds(),
            arch::fs(),
            arch::gs(),
            task_register,
        ]
        .into_iter()
        .enumerate()
        {
            // The host group is ES, CS, SS, DS, FS, GS, TR (no LDTR). Its selectors must have
            // RPL and TI clear.
            vmwrite(
                segment_field(VMCS_HOST_ES_SELECTOR, i as u64),
                (selector & !SELECTOR_RPL_AND_TABLE_MASK) as u64,
            );
        }
        vmwrite(VMCS_HOST_CR0, arch::cr0());
        vmwrite(VMCS_HOST_CR3, arch::cr3());
        vmwrite(VMCS_HOST_CR4, arch::cr4());
        vmwrite(VMCS_HOST_FS_BASE, arch::rdmsr(arch::MSR_FS_BASE));
        vmwrite(VMCS_HOST_GS_BASE, arch::rdmsr(arch::MSR_GS_BASE));
        vmwrite(VMCS_HOST_TR_BASE, tr_base);
        let (gdtr_base, idtr_base) = (host_gdt.base, host_idt.base);
        vmwrite(VMCS_HOST_GDTR_BASE, gdtr_base);
        vmwrite(VMCS_HOST_IDTR_BASE, idtr_base);
        vmwrite(
            VMCS_HOST_IA32_SYSENTER_CS,
            arch::rdmsr(arch::MSR_SYSENTER_CS),
        );
        vmwrite(
            VMCS_HOST_IA32_SYSENTER_ESP,
            arch::rdmsr(arch::MSR_SYSENTER_ESP),
        );
        vmwrite(
            VMCS_HOST_IA32_SYSENTER_EIP,
            arch::rdmsr(arch::MSR_SYSENTER_EIP),
        );
        vmwrite(VMCS_HOST_IA32_PAT, arch::rdmsr(arch::MSR_PAT));
        vmwrite(VMCS_HOST_IA32_EFER, arch::rdmsr(arch::MSR_EFER));
        // The host RSP is written by the entry stub on every entry.
        vmwrite(
            VMCS_HOST_RIP,
            arch::uemu_vmx_exit as *const () as usize as u64,
        );

        // Guest state: the 64-bit Linux entry.
        let cr0 = apply_fixed_bits(
            arch::GUEST_ENTRY_CR0,
            MSR_VMX_CR0_FIXED0,
            MSR_VMX_CR0_FIXED1,
        );
        let cr4_fixed0 = arch::rdmsr(MSR_VMX_CR4_FIXED0);
        vmwrite(VMCS_GUEST_CR0, cr0);
        vmwrite(VMCS_GUEST_CR3, boot.cr3);
        vmwrite(VMCS_GUEST_CR4, arch::GUEST_ENTRY_CR4 | cr4_fixed0);
        // VMXE is the host's: the guest reads it as clear and its writes of it would exit.
        vmwrite(VMCS_CR0_GUEST_HOST_MASK, 0);
        vmwrite(VMCS_CR0_READ_SHADOW, cr0);
        vmwrite(VMCS_CR4_GUEST_HOST_MASK, CR4_VMXE);
        vmwrite(
            VMCS_CR4_READ_SHADOW,
            arch::GUEST_ENTRY_CR4 | (cr4_fixed0 & !CR4_VMXE),
        );
        vmwrite(VMCS_GUEST_DR7, arch::DR7_RESET);
        vmwrite(VMCS_GUEST_RSP, 0);
        vmwrite(VMCS_GUEST_RIP, boot.rip);
        vmwrite(VMCS_GUEST_RFLAGS, arch::RFLAGS_RESERVED_ONE);
        vmwrite(VMCS_GUEST_IA32_EFER, arch::EFER_LME | arch::EFER_LMA);
        vmwrite(VMCS_GUEST_IA32_PAT, arch::PAT_RESET);
        vmwrite(VMCS_GUEST_IA32_DEBUGCTL, 0);
        vmwrite(VMCS_GUEST_IA32_SYSENTER_CS, 0);
        vmwrite(VMCS_GUEST_IA32_SYSENTER_ESP, 0);
        vmwrite(VMCS_GUEST_IA32_SYSENTER_EIP, 0);
        vmwrite(VMCS_GUEST_INTERRUPTIBILITY_STATE, 0);
        vmwrite(VMCS_GUEST_ACTIVITY_STATE, 0);
        vmwrite(VMCS_GUEST_PENDING_DEBUG_EXCEPTIONS, 0);
        let set_segment =
            |segment: u64, selector: u64, limit: u64, access_rights: u64, base: u64| {
                vmwrite(segment_field(VMCS_GUEST_ES_SELECTOR, segment), selector);
                vmwrite(segment_field(VMCS_GUEST_ES_LIMIT, segment), limit);
                vmwrite(
                    segment_field(VMCS_GUEST_ES_ACCESS_RIGHTS, segment),
                    access_rights,
                );
                vmwrite(segment_field(VMCS_GUEST_ES_BASE, segment), base);
            };
        let flat_limit = arch::FLAT_SEGMENT_LIMIT as u64;
        let ds = boot.ds as u64;
        set_segment(SEGMENT_ES, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_CS, boot.cs as u64, flat_limit, AR_CODE_64BIT, 0);
        set_segment(SEGMENT_SS, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_DS, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_FS, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_GS, ds, flat_limit, AR_DATA_FLAT, 0);
        let system_limit = arch::SYSTEM_SEGMENT_LIMIT as u64;
        set_segment(SEGMENT_LDTR, 0, system_limit, AR_UNUSABLE, 0);
        set_segment(SEGMENT_TR, 0, system_limit, AR_TSS, 0);
        // GDTR and IDTR have only a limit and a base field, no selector or access rights.
        vmwrite(
            segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_GDTR),
            boot.gdt_limit as u64,
        );
        vmwrite(
            segment_field(VMCS_GUEST_ES_BASE, SEGMENT_GDTR),
            boot.gdt_base,
        );
        vmwrite(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_IDTR), 0);
        vmwrite(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_IDTR), 0);

        Ok(Self {
            gprs: Gprs {
                rsi: boot.rsi,
                ..Gprs::default()
            },
            proc_controls: proc,
            launched: false,
            next_rip: 0,
            insn_len: 0,
            event: None,
            window: false,
            host_gdt,
            host_idt,
        })
    }
}

/// The host needs a task register with a valid TSS for VM exits to load. The firmware sets one up
/// on some platforms; where it did not, a copy of its GDT gets a TSS descriptor.
fn host_task_register(gdt: &DescriptorTableRegister) -> Result<u16, String> {
    let selector = arch::task_register_selector();
    if selector != 0 {
        return Ok(selector);
    }
    let old_len = gdt.limit as usize + 1;
    let descriptor_offset = old_len.next_multiple_of(SYSTEM_DESCRIPTOR_SIZE);
    // The copied GDT and its new descriptor go in the first page, the (zeroed) TSS in the second.
    let page = alloc_pages(2)?;
    let tss = unsafe { page.add(PAGE_SIZE) };
    let base = tss as u64;
    unsafe {
        core::ptr::copy_nonoverlapping(gdt.base as *const u8, page, old_len);
        let descriptor = page.add(descriptor_offset);
        // Low qword: limit 15:0, base 23:0 split around the access byte, base 31:24. High qword:
        // base 63:32.
        let low = TSS_LIMIT
            | (base & u16::MAX as u64) << DESCRIPTOR_BASE_15_0_POS
            | ((base >> BASE_23_16_SHIFT) & u8::MAX as u64) << DESCRIPTOR_BASE_23_16_POS
            | TSS_ACCESS << DESCRIPTOR_ACCESS_POS
            | ((base >> BASE_31_24_SHIFT) & u8::MAX as u64) << DESCRIPTOR_BASE_31_24_POS;
        (descriptor as *mut u64).write(low);
        (descriptor as *mut u64)
            .add(1)
            .write(base >> BASE_63_32_SHIFT);
    }
    let new_gdt = DescriptorTableRegister {
        limit: (descriptor_offset + SYSTEM_DESCRIPTOR_SIZE - 1) as u16,
        base: page as u64,
    };
    arch::lgdt(&new_gdt);
    // The selector is the byte offset of the descriptor: index in bits 15:3, TI and RPL zero.
    let selector = descriptor_offset as u16;
    // No `nomem`: LTR reads the GDT and marks the TSS descriptor busy.
    unsafe { asm!("ltr {:x}", in(reg) selector, options(nostack)) };
    Ok(selector)
}

/// Base address of the 64-bit TSS that `selector` names in `gdt`.
fn tss_base(gdt: &DescriptorTableRegister, selector: u16) -> u64 {
    let descriptor = (gdt.base + (selector & !SELECTOR_RPL_AND_TABLE_MASK) as u64) as *const u64;
    let (low, high) = unsafe {
        (
            descriptor.read_unaligned(),
            descriptor.add(1).read_unaligned(),
        )
    };
    // Base 23:0 is contiguous from bit 16 of the low qword.
    ((low >> DESCRIPTOR_BASE_15_0_POS) & ((1 << BASE_31_24_SHIFT) - 1))
        | ((low >> DESCRIPTOR_BASE_31_24_POS) << BASE_31_24_SHIFT)
        | (high << BASE_63_32_SHIFT)
}

/// Clears the read and write intercept bits of `msr` in the MSR bitmap.
fn pass_through(bitmap: *mut u8, msr: u32) {
    let (base, index) = match msr {
        0..=arch::MSR_LOW_RANGE_END => (0, msr),
        arch::MSR_HIGH_RANGE_START..=arch::MSR_HIGH_RANGE_END => (
            MSR_BITMAP_HIGH_RANGE_OFFSET,
            msr - arch::MSR_HIGH_RANGE_START,
        ),
        _ => return,
    };
    let (byte, bit) = (index as usize / 8, index % 8);
    unsafe {
        *bitmap.add(base + byte) &= !(1 << bit);
        *bitmap.add(MSR_BITMAP_WRITE_OFFSET + base + byte) &= !(1 << bit);
    }
}

impl Vcpu for Vmx {
    fn gprs(&mut self) -> &mut Gprs {
        &mut self.gprs
    }

    fn run(&mut self) -> Result<Exit, String> {
        match self.event.take() {
            Some((info, error_code)) => {
                if let Some(error_code) = error_code {
                    vmwrite(VMCS_VMENTRY_EXCEPTION_ERROR_CODE, error_code as u64);
                }
                // Software interrupts and exceptions are re-delivered with their instruction length.
                if matches!(
                    (info >> arch::EVENT_TYPE_SHIFT) & EVENT_TYPE_MASK,
                    EVENT_TYPE_SOFTWARE_INTERRUPT..=EVENT_TYPE_SOFTWARE_EXCEPTION
                ) {
                    vmwrite(VMCS_VMENTRY_INSTRUCTION_LENGTH, self.insn_len);
                }
                vmwrite(
                    VMCS_VMENTRY_INTERRUPTION_INFO,
                    (info | arch::EVENT_VALID) as u64,
                );
            }
            None => vmwrite(VMCS_VMENTRY_INTERRUPTION_INFO, 0),
        }
        let controls = if self.window {
            self.proc_controls | PROC_INTERRUPT_WINDOW_EXITING
        } else {
            self.proc_controls & !PROC_INTERRUPT_WINDOW_EXITING
        };
        vmwrite(VMCS_PRIMARY_PROC_BASED_CONTROLS, controls as u64);

        let failed = unsafe { arch::uemu_vmx_enter(&mut self.gprs, self.launched as u32) };
        arch::lgdt(&self.host_gdt);
        arch::lidt(&self.host_idt);
        if failed != 0 {
            return Err(format!(
                "VMLAUNCH/VMRESUME failed, instruction error {}",
                vmread(VMCS_VM_INSTRUCTION_ERROR)
            ));
        }
        self.launched = true;

        let exit_reason = vmread(VMCS_EXIT_REASON);
        let qualification = vmread(VMCS_EXIT_QUALIFICATION);
        let rip = vmread(VMCS_GUEST_RIP);
        self.insn_len = vmread(VMCS_VMEXIT_INSTRUCTION_LENGTH);
        self.next_rip = rip + self.insn_len;
        if exit_reason & EXIT_REASON_ENTRY_FAILURE != 0 {
            return Err(format!(
                "VM entry failed: reason {:#x} qualification {qualification:#x} (rip {rip:#x})",
                exit_reason & EXIT_REASON_BASIC_MASK
            ));
        }
        let reason = exit_reason & EXIT_REASON_BASIC_MASK;
        // An event that was being delivered when the exit happened is delivered again.
        let vectoring = vmread(VMCS_IDT_VECTORING_INFO) as u32;
        if vectoring & arch::EVENT_VALID != 0 && reason != REASON_EPT_VIOLATION {
            let error_code = (vectoring & arch::EVENT_ERROR_CODE_VALID != 0)
                .then(|| vmread(VMCS_IDT_VECTORING_ERROR_CODE) as u32);
            self.event = Some((vectoring & !arch::EVENT_VALID, error_code));
        }
        Ok(match reason {
            REASON_EXTERNAL_INTERRUPT => Exit::HostInterrupt,
            REASON_INTERRUPT_WINDOW => Exit::IntWindow,
            REASON_CPUID => Exit::Cpuid,
            REASON_HLT => Exit::Hlt,
            REASON_INVD => Exit::Skip,
            REASON_RDMSR => Exit::Rdmsr,
            REASON_WRMSR => Exit::Wrmsr,
            REASON_TRIPLE_FAULT => Exit::TripleFault,
            // VMX instructions (VMCALL through VMXON) and XSETBV.
            REASON_VMCALL..=REASON_VMXON | REASON_XSETBV => Exit::Undefined,
            REASON_IO_INSTRUCTION => {
                if qualification & IO_QUAL_STRING != 0 {
                    return Err(format!(
                        "string port I/O (port {:#x}) at rip {rip:#x} is not supported",
                        qualification >> IO_QUAL_PORT_SHIFT
                    ));
                }
                Exit::Io {
                    port: (qualification >> IO_QUAL_PORT_SHIFT) as u16,
                    size: (qualification & IO_QUAL_SIZE_MASK) as u8 + 1,
                    write: qualification & IO_QUAL_DIRECTION_IN == 0,
                }
            }
            REASON_EPT_VIOLATION => Exit::Fault {
                gpa: vmread(VMCS_GUEST_PHYSICAL_ADDRESS),
                write: qualification & EPT_VIOLATION_QUAL_WRITE != 0,
            },
            other => Exit::Other(other),
        })
    }

    fn skip(&mut self) {
        vmwrite(VMCS_GUEST_RIP, self.next_rip);
        let state = vmread(VMCS_GUEST_INTERRUPTIBILITY_STATE);
        vmwrite(
            VMCS_GUEST_INTERRUPTIBILITY_STATE,
            state & !INTERRUPTIBILITY_BLOCKING_BY_STI_AND_MOV_SS,
        );
    }

    fn rip(&self) -> u64 {
        vmread(VMCS_GUEST_RIP)
    }

    fn irq_ok(&self) -> bool {
        vmread(VMCS_GUEST_RFLAGS) & arch::RFLAGS_IF != 0
            && vmread(VMCS_GUEST_INTERRUPTIBILITY_STATE)
                & INTERRUPTIBILITY_BLOCKING_BY_STI_AND_MOV_SS
                == 0
            && self.event.is_none()
    }

    fn inject_irq(&mut self, vector: u8) {
        self.event = Some((vector as u32, None));
    }

    fn window(&mut self, on: bool) {
        self.window = on;
    }

    fn inject_gp(&mut self) {
        self.event = Some((
            arch::VECTOR_GENERAL_PROTECTION
                | (arch::EVENT_TYPE_HARDWARE_EXCEPTION << arch::EVENT_TYPE_SHIFT)
                | arch::EVENT_ERROR_CODE_VALID,
            Some(0),
        ));
    }

    fn inject_ud(&mut self) {
        self.event = Some((
            arch::VECTOR_INVALID_OPCODE
                | (arch::EVENT_TYPE_HARDWARE_EXCEPTION << arch::EVENT_TYPE_SHIFT),
            None,
        ));
    }

    fn efer(&self) -> u64 {
        vmread(VMCS_GUEST_IA32_EFER)
    }

    fn set_efer(&mut self, efer: u64) {
        vmwrite(VMCS_GUEST_IA32_EFER, efer);
    }
}
