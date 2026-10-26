use alloc::format;
use alloc::string::String;
use core::arch::asm;
use uemulator::BootState;

use super::arch::{self, DescTable, Gprs};
use super::{Exit, Vcpu, alloc_pages};

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

const FEATURE_LOCK: u64 = 1;
const FEATURE_VMXON_OUTSIDE_SMX: u64 = 1 << 2;
const CR4_VMXE: u64 = 1 << 13;

// VMCS field encodings.
const MSR_BITMAP: u64 = 0x2004;
const EXIT_MSR_STORE_ADDR: u64 = 0x2006;
const EXIT_MSR_LOAD_ADDR: u64 = 0x2008;
const ENTRY_MSR_LOAD_ADDR: u64 = 0x200a;
const EPT_POINTER: u64 = 0x201a;
const GUEST_PHYS_ADDR: u64 = 0x2400;
const VMCS_LINK: u64 = 0x2800;
const GUEST_DEBUGCTL: u64 = 0x2802;
const GUEST_PAT: u64 = 0x2804;
const GUEST_EFER: u64 = 0x2806;
const HOST_PAT: u64 = 0x2c00;
const HOST_EFER: u64 = 0x2c02;
const PIN_CONTROLS: u64 = 0x4000;
const PROC_CONTROLS: u64 = 0x4002;
const EXCEPTION_BITMAP: u64 = 0x4004;
const PF_ERR_MASK: u64 = 0x4006;
const PF_ERR_MATCH: u64 = 0x4008;
const CR3_TARGET_COUNT: u64 = 0x400a;
const EXIT_CONTROLS: u64 = 0x400c;
const EXIT_MSR_STORE_COUNT: u64 = 0x400e;
const EXIT_MSR_LOAD_COUNT: u64 = 0x4010;
const ENTRY_CONTROLS: u64 = 0x4012;
const ENTRY_MSR_LOAD_COUNT: u64 = 0x4014;
const ENTRY_INTR_INFO: u64 = 0x4016;
const ENTRY_EXC_ERR: u64 = 0x4018;
const ENTRY_INSN_LEN: u64 = 0x401a;
const PROC_CONTROLS2: u64 = 0x401e;
const VM_INSN_ERROR: u64 = 0x4400;
const EXIT_REASON: u64 = 0x4402;
const IDT_VECTORING_INFO: u64 = 0x4408;
const IDT_VECTORING_ERR: u64 = 0x440a;
const EXIT_INSN_LEN: u64 = 0x440c;
const CR0_MASK: u64 = 0x6000;
const CR4_MASK: u64 = 0x6002;
const CR0_SHADOW: u64 = 0x6004;
const CR4_SHADOW: u64 = 0x6006;
const EXIT_QUAL: u64 = 0x6400;

const GUEST_SEL: u64 = 0x0800; // + 2 * index: ES CS SS DS FS GS LDTR TR
const HOST_SEL: u64 = 0x0c00; // + 2 * index: ES CS SS DS FS GS TR
const GUEST_LIMIT: u64 = 0x4800; // + 2 * index, then GDTR and IDTR
const GUEST_AR: u64 = 0x4814; // + 2 * index
const GUEST_BASE: u64 = 0x6806; // + 2 * index, then GDTR and IDTR
const GUEST_INTERRUPTIBILITY: u64 = 0x4824;
const GUEST_ACTIVITY: u64 = 0x4826;
const GUEST_SYSENTER_CS: u64 = 0x482a;
const HOST_SYSENTER_CS: u64 = 0x4c00;
const GUEST_CR0: u64 = 0x6800;
const GUEST_CR3: u64 = 0x6802;
const GUEST_CR4: u64 = 0x6804;
const GUEST_DR7: u64 = 0x681a;
const GUEST_RSP: u64 = 0x681c;
const GUEST_RIP: u64 = 0x681e;
const GUEST_RFLAGS: u64 = 0x6820;
const GUEST_PENDING_DEBUG: u64 = 0x6822;
const GUEST_SYSENTER_ESP: u64 = 0x6824;
const GUEST_SYSENTER_EIP: u64 = 0x6826;
const HOST_CR0: u64 = 0x6c00;
const HOST_CR3: u64 = 0x6c02;
const HOST_CR4: u64 = 0x6c04;
const HOST_FS_BASE: u64 = 0x6c06;
const HOST_GS_BASE: u64 = 0x6c08;
const HOST_TR_BASE: u64 = 0x6c0a;
const HOST_GDTR_BASE: u64 = 0x6c0c;
const HOST_IDTR_BASE: u64 = 0x6c0e;
const HOST_SYSENTER_ESP: u64 = 0x6c10;
const HOST_SYSENTER_EIP: u64 = 0x6c12;
const HOST_RIP: u64 = 0x6c16;

const PIN_EXTERNAL_INTERRUPT: u32 = 1;
const PROC_INTERRUPT_WINDOW: u32 = 1 << 2;
const PROC_HLT: u32 = 1 << 7;
const PROC_UNCONDITIONAL_IO: u32 = 1 << 24;
const PROC_MSR_BITMAPS: u32 = 1 << 28;
const PROC_SECONDARY: u32 = 1 << 31;
const PROC2_EPT: u32 = 1 << 1;
const PROC2_UNRESTRICTED: u32 = 1 << 7;
const EXIT_SAVE_DEBUG: u32 = 1 << 2;
const EXIT_HOST_64: u32 = 1 << 9;
const EXIT_SAVE_PAT: u32 = 1 << 18;
const EXIT_LOAD_PAT: u32 = 1 << 19;
const EXIT_SAVE_EFER: u32 = 1 << 20;
const EXIT_LOAD_EFER: u32 = 1 << 21;
const ENTRY_LOAD_DEBUG: u32 = 1 << 2;
const ENTRY_IA32E: u32 = 1 << 9;
const ENTRY_LOAD_PAT: u32 = 1 << 14;
const ENTRY_LOAD_EFER: u32 = 1 << 15;

const REASON_INTERRUPT: u64 = 1;
const REASON_TRIPLE_FAULT: u64 = 2;
const REASON_INTERRUPT_WINDOW: u64 = 7;
const REASON_CPUID: u64 = 10;
const REASON_HLT: u64 = 12;
const REASON_INVD: u64 = 13;
const REASON_VMCALL: u64 = 18;
const REASON_VMXON: u64 = 27;
const REASON_IO: u64 = 30;
const REASON_RDMSR: u64 = 31;
const REASON_WRMSR: u64 = 32;
const REASON_EPT_VIOLATION: u64 = 48;
const REASON_XSETBV: u64 = 55;

/// Guest MSRs the CPU does not switch by itself; they live in the VM-entry load and VM-exit store
/// lists, with the host's values restored from the VM-exit load list.
const SWITCHED_MSRS: [u32; 5] = [
    arch::MSR_STAR,
    arch::MSR_LSTAR,
    arch::MSR_CSTAR,
    arch::MSR_SFMASK,
    arch::MSR_KERNEL_GS_BASE,
];

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

fn vm_op(op: &str, pa: u64) -> bool {
    let ok: u8;
    unsafe {
        let addr = &pa as *const u64;
        match op {
            "on" => {
                asm!("vmxon [{a}]", "setna {ok}", a = in(reg) addr, ok = out(reg_byte) ok, options(nostack))
            }
            "clear" => {
                asm!("vmclear [{a}]", "setna {ok}", a = in(reg) addr, ok = out(reg_byte) ok, options(nostack))
            }
            _ => {
                asm!("vmptrld [{a}]", "setna {ok}", a = in(reg) addr, ok = out(reg_byte) ok, options(nostack))
            }
        }
    }
    ok == 0
}

pub fn usable() -> bool {
    if arch::cpuid(1, 0)[2] & (1 << 5) == 0 {
        return false;
    }
    let fc = arch::rdmsr(arch::MSR_FEATURE_CONTROL);
    if fc & FEATURE_LOCK != 0 && fc & FEATURE_VMXON_OUTSIDE_SMX == 0 {
        return false;
    }
    // Without the TRUE control MSRs, CR3 accesses would be forced to exit.
    if arch::rdmsr(MSR_VMX_BASIC) & (1 << 55) == 0 {
        return false;
    }
    let proc = arch::rdmsr(MSR_VMX_PROCBASED);
    let proc2 = arch::rdmsr(MSR_VMX_PROCBASED2);
    let cap = arch::rdmsr(MSR_VMX_EPT_VPID_CAP);
    // Secondary controls with EPT and unrestricted guest, 4-level walks, 2 MiB pages, WB memory.
    (proc >> 32) as u32 & PROC_SECONDARY != 0
        && (proc2 >> 32) as u32 & (PROC2_EPT | PROC2_UNRESTRICTED)
            == (PROC2_EPT | PROC2_UNRESTRICTED)
        && cap & (1 << 6 | 1 << 14 | 1 << 16) == (1 << 6 | 1 << 14 | 1 << 16)
}

fn adjust(msr: u32, wanted: u32) -> Result<u32, String> {
    let m = arch::rdmsr(msr);
    let (must_be_one, may_be_one) = (m as u32, (m >> 32) as u32);
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
    proc_controls: u32,
    launched: bool,
    next_rip: u64,
    insn_len: u64,
    event: Option<(u32, Option<u32>)>,
    window: bool,
    host_gdt: DescTable,
    host_idt: DescTable,
}

impl Vmx {
    pub fn new(ept: u64, boot: &BootState) -> Result<Self, String> {
        let basic = arch::rdmsr(MSR_VMX_BASIC);
        let fc = arch::rdmsr(arch::MSR_FEATURE_CONTROL);
        if fc & FEATURE_LOCK == 0 {
            arch::wrmsr(
                arch::MSR_FEATURE_CONTROL,
                fc | FEATURE_LOCK | FEATURE_VMXON_OUTSIDE_SMX,
            );
        }
        let fix = |v: u64, f0: u32, f1: u32| (v | arch::rdmsr(f0)) & arch::rdmsr(f1);
        arch::set_cr0(fix(arch::cr0(), MSR_VMX_CR0_FIXED0, MSR_VMX_CR0_FIXED1));
        arch::set_cr4(fix(
            arch::cr4() | CR4_VMXE,
            MSR_VMX_CR4_FIXED0,
            MSR_VMX_CR4_FIXED1,
        ));

        let vmxon = alloc_pages(1)?;
        let vmcs = alloc_pages(1)?;
        let msr_bitmap = alloc_pages(1)?;
        let msr_list = alloc_pages(2)?;
        let revision = (basic as u32) & 0x7fff_ffff;
        unsafe {
            (vmxon as *mut u32).write(revision);
            (vmcs as *mut u32).write(revision);
            core::ptr::write_bytes(msr_bitmap, 0xff, 4096);
        }
        if !vm_op("on", vmxon as u64) {
            return Err("VMXON failed".into());
        }
        if !vm_op("clear", vmcs as u64) || !vm_op("ptrld", vmcs as u64) {
            return Err("VMCLEAR/VMPTRLD failed".into());
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
            pass_through(msr_bitmap, msr);
        }

        let true_ctls = basic & (1 << 55) != 0;
        let pick = |plain: u32, tru: u32| if true_ctls { tru } else { plain };
        let pin = adjust(
            pick(MSR_VMX_PINBASED, MSR_VMX_TRUE_PINBASED),
            PIN_EXTERNAL_INTERRUPT,
        )?;
        let proc = adjust(
            pick(MSR_VMX_PROCBASED, MSR_VMX_TRUE_PROCBASED),
            PROC_HLT | PROC_UNCONDITIONAL_IO | PROC_MSR_BITMAPS | PROC_SECONDARY,
        )?;
        let proc2 = adjust(MSR_VMX_PROCBASED2, PROC2_EPT | PROC2_UNRESTRICTED)?;
        let exit = adjust(
            pick(MSR_VMX_EXIT, MSR_VMX_TRUE_EXIT),
            EXIT_SAVE_DEBUG
                | EXIT_HOST_64
                | EXIT_SAVE_PAT
                | EXIT_LOAD_PAT
                | EXIT_SAVE_EFER
                | EXIT_LOAD_EFER,
        )?;
        let entry = adjust(
            pick(MSR_VMX_ENTRY, MSR_VMX_TRUE_ENTRY),
            ENTRY_LOAD_DEBUG | ENTRY_IA32E | ENTRY_LOAD_PAT | ENTRY_LOAD_EFER,
        )?;

        vmwrite(PIN_CONTROLS, pin as u64);
        vmwrite(PROC_CONTROLS, proc as u64);
        vmwrite(PROC_CONTROLS2, proc2 as u64);
        vmwrite(EXIT_CONTROLS, exit as u64);
        vmwrite(ENTRY_CONTROLS, entry as u64);
        vmwrite(MSR_BITMAP, msr_bitmap as u64);
        vmwrite(EPT_POINTER, ept | 6 | (3 << 3));
        vmwrite(EXCEPTION_BITMAP, 0);
        vmwrite(PF_ERR_MASK, 0);
        vmwrite(PF_ERR_MATCH, 0);
        vmwrite(CR3_TARGET_COUNT, 0);
        vmwrite(VMCS_LINK, u64::MAX);

        // MSR lists: guest values first, host values in the second page.
        let host_list = unsafe { msr_list.add(4096) };
        for (i, msr) in SWITCHED_MSRS.iter().enumerate() {
            unsafe {
                let g = (msr_list as *mut u64).add(i * 2);
                g.write(*msr as u64);
                g.add(1).write(0);
                let h = (host_list as *mut u64).add(i * 2);
                h.write(*msr as u64);
                h.add(1).write(arch::rdmsr(*msr));
            }
        }
        vmwrite(ENTRY_MSR_LOAD_ADDR, msr_list as u64);
        vmwrite(ENTRY_MSR_LOAD_COUNT, SWITCHED_MSRS.len() as u64);
        vmwrite(EXIT_MSR_STORE_ADDR, msr_list as u64);
        vmwrite(EXIT_MSR_STORE_COUNT, SWITCHED_MSRS.len() as u64);
        vmwrite(EXIT_MSR_LOAD_ADDR, host_list as u64);
        vmwrite(EXIT_MSR_LOAD_COUNT, SWITCHED_MSRS.len() as u64);

        let host_gdt = arch::sgdt();
        let host_idt = arch::sidt();
        let tr = host_task_register(&host_gdt)?;
        let host_gdt = arch::sgdt();
        let tr_base = tss_base(&host_gdt, tr);
        for (i, sel) in [
            arch::es(),
            arch::cs(),
            arch::ss(),
            arch::ds(),
            arch::fs(),
            arch::gs(),
            tr,
        ]
        .into_iter()
        .enumerate()
        {
            vmwrite(HOST_SEL + 2 * i as u64, (sel & !7) as u64);
        }
        vmwrite(HOST_CR0, arch::cr0());
        vmwrite(HOST_CR3, arch::cr3());
        vmwrite(HOST_CR4, arch::cr4());
        vmwrite(HOST_FS_BASE, arch::rdmsr(arch::MSR_FS_BASE));
        vmwrite(HOST_GS_BASE, arch::rdmsr(arch::MSR_GS_BASE));
        vmwrite(HOST_TR_BASE, tr_base);
        let (gb, ib) = (host_gdt.base, host_idt.base);
        vmwrite(HOST_GDTR_BASE, gb);
        vmwrite(HOST_IDTR_BASE, ib);
        vmwrite(HOST_SYSENTER_CS, arch::rdmsr(arch::MSR_SYSENTER_CS));
        vmwrite(HOST_SYSENTER_ESP, arch::rdmsr(arch::MSR_SYSENTER_ESP));
        vmwrite(HOST_SYSENTER_EIP, arch::rdmsr(arch::MSR_SYSENTER_EIP));
        vmwrite(HOST_PAT, arch::rdmsr(arch::MSR_PAT));
        vmwrite(HOST_EFER, arch::rdmsr(arch::MSR_EFER));
        vmwrite(HOST_RIP, arch::uemu_vmx_exit as *const () as usize as u64);

        // Guest state: the 64-bit Linux entry.
        let cr0 = fix(0x8001_0033, MSR_VMX_CR0_FIXED0, MSR_VMX_CR0_FIXED1);
        let cr4_fixed0 = arch::rdmsr(MSR_VMX_CR4_FIXED0);
        vmwrite(GUEST_CR0, cr0);
        vmwrite(GUEST_CR3, boot.cr3);
        vmwrite(GUEST_CR4, 0x20 | cr4_fixed0);
        // VMXE is the host's: the guest reads it as clear and its writes of it would exit.
        vmwrite(CR0_MASK, 0);
        vmwrite(CR0_SHADOW, cr0);
        vmwrite(CR4_MASK, CR4_VMXE);
        vmwrite(CR4_SHADOW, 0x20 | (cr4_fixed0 & !CR4_VMXE));
        vmwrite(GUEST_DR7, 0x400);
        vmwrite(GUEST_RSP, 0);
        vmwrite(GUEST_RIP, boot.rip);
        vmwrite(GUEST_RFLAGS, 2);
        vmwrite(GUEST_EFER, arch::EFER_LME | arch::EFER_LMA);
        vmwrite(GUEST_PAT, 0x0007_0406_0007_0406);
        vmwrite(GUEST_DEBUGCTL, 0);
        vmwrite(GUEST_SYSENTER_CS, 0);
        vmwrite(GUEST_SYSENTER_ESP, 0);
        vmwrite(GUEST_SYSENTER_EIP, 0);
        vmwrite(GUEST_INTERRUPTIBILITY, 0);
        vmwrite(GUEST_ACTIVITY, 0);
        vmwrite(GUEST_PENDING_DEBUG, 0);
        let seg = |i: u64, sel: u16, ar: u64| {
            vmwrite(GUEST_SEL + 2 * i, sel as u64);
            vmwrite(GUEST_LIMIT + 2 * i, 0xffff_ffff);
            vmwrite(GUEST_AR + 2 * i, ar);
            vmwrite(GUEST_BASE + 2 * i, 0);
        };
        seg(0, boot.ds, 0xc093);
        seg(1, boot.cs, 0xa09b);
        seg(2, boot.ds, 0xc093);
        seg(3, boot.ds, 0xc093);
        seg(4, boot.ds, 0xc093);
        seg(5, boot.ds, 0xc093);
        vmwrite(GUEST_SEL + 12, 0);
        vmwrite(GUEST_LIMIT + 12, 0xffff);
        vmwrite(GUEST_AR + 12, 0x1_0000);
        vmwrite(GUEST_BASE + 12, 0);
        vmwrite(GUEST_SEL + 14, 0);
        vmwrite(GUEST_LIMIT + 14, 0xffff);
        vmwrite(GUEST_AR + 14, 0x8b);
        vmwrite(GUEST_BASE + 14, 0);
        vmwrite(GUEST_LIMIT + 16, boot.gdt_limit as u64);
        vmwrite(GUEST_BASE + 16, boot.gdt_base);
        vmwrite(GUEST_LIMIT + 18, 0);
        vmwrite(GUEST_BASE + 18, 0);

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
fn host_task_register(gdt: &DescTable) -> Result<u16, String> {
    let tr = arch::str_sel();
    if tr != 0 {
        return Ok(tr);
    }
    let old_len = gdt.limit as usize + 1;
    let off = old_len.next_multiple_of(16);
    let page = alloc_pages(2)?;
    let tss = unsafe { page.add(4096) };
    let base = tss as u64;
    unsafe {
        core::ptr::copy_nonoverlapping(gdt.base as *const u8, page, old_len);
        let d = page.add(off);
        let low = 0x67u64
            | (base & 0xffff) << 16
            | ((base >> 16) & 0xff) << 32
            | 0x89 << 40
            | ((base >> 24) & 0xff) << 56;
        (d as *mut u64).write(low);
        (d as *mut u64).add(1).write(base >> 32);
    }
    let new = DescTable {
        limit: (off + 15) as u16,
        base: page as u64,
    };
    arch::lgdt(&new);
    let sel = off as u16;
    unsafe { asm!("ltr {:x}", in(reg) sel, options(nostack)) };
    Ok(sel)
}

fn tss_base(gdt: &DescTable, sel: u16) -> u64 {
    let d = (gdt.base + (sel & !7) as u64) as *const u64;
    let (lo, hi) = unsafe { (d.read_unaligned(), d.add(1).read_unaligned()) };
    ((lo >> 16) & 0xff_ffff) | ((lo >> 56) << 24) | (hi << 32)
}

fn pass_through(bitmap: *mut u8, msr: u32) {
    let (base, index) = match msr {
        0..=0x1fff => (0, msr),
        0xc000_0000..=0xc000_1fff => (1024, msr - 0xc000_0000),
        _ => return,
    };
    let (byte, bit) = (index as usize / 8, index % 8);
    unsafe {
        *bitmap.add(base + byte) &= !(1 << bit);
        *bitmap.add(2048 + base + byte) &= !(1 << bit);
    }
}

impl Vcpu for Vmx {
    fn gprs(&mut self) -> &mut Gprs {
        &mut self.gprs
    }

    fn run(&mut self) -> Result<Exit, String> {
        match self.event.take() {
            Some((info, err)) => {
                if let Some(e) = err {
                    vmwrite(ENTRY_EXC_ERR, e as u64);
                }
                // Software interrupts and exceptions are re-delivered with their instruction length.
                if matches!((info >> 8) & 7, 4..=6) {
                    vmwrite(ENTRY_INSN_LEN, self.insn_len);
                }
                vmwrite(ENTRY_INTR_INFO, (info | 1 << 31) as u64);
            }
            None => vmwrite(ENTRY_INTR_INFO, 0),
        }
        let want = if self.window {
            self.proc_controls | PROC_INTERRUPT_WINDOW
        } else {
            self.proc_controls & !PROC_INTERRUPT_WINDOW
        };
        vmwrite(PROC_CONTROLS, want as u64);

        let failed = unsafe { arch::uemu_vmx_enter(&mut self.gprs, self.launched as u32) };
        arch::lgdt(&self.host_gdt);
        arch::lidt(&self.host_idt);
        if failed != 0 {
            return Err(format!(
                "VMLAUNCH/VMRESUME failed, instruction error {}",
                vmread(VM_INSN_ERROR)
            ));
        }
        self.launched = true;

        let reason = vmread(EXIT_REASON);
        let qual = vmread(EXIT_QUAL);
        let rip = vmread(GUEST_RIP);
        self.insn_len = vmread(EXIT_INSN_LEN);
        self.next_rip = rip + self.insn_len;
        if reason & (1 << 31) != 0 {
            return Err(format!(
                "VM entry failed: reason {:#x} qualification {qual:#x} (rip {rip:#x})",
                reason & 0xffff
            ));
        }
        let reason = reason & 0xffff;
        // An event that was being delivered when the exit happened is delivered again.
        let vectoring = vmread(IDT_VECTORING_INFO) as u32;
        if vectoring & (1 << 31) != 0 && reason != REASON_EPT_VIOLATION {
            let err = (vectoring & (1 << 11) != 0).then(|| vmread(IDT_VECTORING_ERR) as u32);
            self.event = Some((vectoring & !(1 << 31), err));
        }
        Ok(match reason {
            REASON_INTERRUPT => Exit::HostInterrupt,
            REASON_INTERRUPT_WINDOW => Exit::IntWindow,
            REASON_CPUID => Exit::Cpuid,
            REASON_HLT => Exit::Hlt,
            REASON_INVD => Exit::Skip,
            REASON_RDMSR => Exit::Rdmsr,
            REASON_WRMSR => Exit::Wrmsr,
            REASON_TRIPLE_FAULT => Exit::TripleFault,
            REASON_VMCALL..=REASON_VMXON | REASON_XSETBV => Exit::Undefined,
            REASON_IO => {
                if qual & (1 << 4) != 0 {
                    return Err(format!(
                        "string port I/O (port {:#x}) at rip {rip:#x} is not supported",
                        qual >> 16
                    ));
                }
                Exit::Io {
                    port: (qual >> 16) as u16,
                    size: (qual & 7) as u8 + 1,
                    write: qual & (1 << 3) == 0,
                }
            }
            REASON_EPT_VIOLATION => Exit::Fault {
                gpa: vmread(GUEST_PHYS_ADDR),
                write: qual & 2 != 0,
            },
            other => Exit::Other(other),
        })
    }

    fn skip(&mut self) {
        vmwrite(GUEST_RIP, self.next_rip);
        let state = vmread(GUEST_INTERRUPTIBILITY);
        vmwrite(GUEST_INTERRUPTIBILITY, state & !3);
    }

    fn rip(&self) -> u64 {
        vmread(GUEST_RIP)
    }

    fn irq_ok(&self) -> bool {
        vmread(GUEST_RFLAGS) & (1 << 9) != 0
            && vmread(GUEST_INTERRUPTIBILITY) & 3 == 0
            && self.event.is_none()
    }

    fn inject_irq(&mut self, vector: u8) {
        self.event = Some((vector as u32, None));
    }

    fn window(&mut self, on: bool) {
        self.window = on;
    }

    fn inject_gp(&mut self) {
        self.event = Some((13 | (3 << 8) | (1 << 11), Some(0)));
    }

    fn inject_ud(&mut self) {
        self.event = Some((6 | (3 << 8), None));
    }

    fn efer(&self) -> u64 {
        vmread(GUEST_EFER)
    }

    fn set_efer(&mut self, efer: u64) {
        vmwrite(GUEST_EFER, efer);
    }
}
