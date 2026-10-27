//! Everything that touches the CPU: instructions the compiler has no intrinsic for, and the
//! world-switch stubs. The register layouts and constants live in `defs`.

use core::arch::asm;
use core::mem::offset_of;

pub use super::defs::*;

pub fn cpuid(leaf: u32, sub: u32) -> [u32; 4] {
    let r = core::arch::x86_64::__cpuid_count(leaf, sub);
    [r.eax, r.ebx, r.ecx, r.edx]
}

pub fn rdmsr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    unsafe {
        asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags));
    }
    (hi as u64) << 32 | lo as u64
}

pub fn wrmsr(msr: u32, v: u64) {
    unsafe {
        asm!("wrmsr", in("ecx") msr, in("eax") v as u32, in("edx") (v >> 32) as u32, options(nomem, nostack, preserves_flags));
    }
}

pub fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

macro_rules! cr {
    ($read:ident, $name:literal) => {
        pub fn $read() -> u64 {
            let v: u64;
            unsafe { asm!(concat!("mov {}, ", $name), out(reg) v, options(nomem, nostack, preserves_flags)) };
            v
        }
    };
    ($read:ident, $write:ident, $name:literal) => {
        cr!($read, $name);
        pub fn $write(v: u64) {
            unsafe { asm!(concat!("mov ", $name, ", {}"), in(reg) v, options(nomem, nostack, preserves_flags)) };
        }
    };
}
cr!(cr0, set_cr0, "cr0");
cr!(cr3, "cr3");
cr!(cr4, set_cr4, "cr4");

// SGDT and SIDT write through the pointer, so unlike the register reads they omit `nomem`.
pub fn sgdt() -> DescriptorTableRegister {
    let mut d = DescriptorTableRegister::default();
    unsafe { asm!("sgdt [{}]", in(reg) &mut d, options(nostack, preserves_flags)) };
    d
}

pub fn sidt() -> DescriptorTableRegister {
    let mut d = DescriptorTableRegister::default();
    unsafe { asm!("sidt [{}]", in(reg) &mut d, options(nostack, preserves_flags)) };
    d
}

pub fn lgdt(d: &DescriptorTableRegister) {
    unsafe { asm!("lgdt [{}]", in(reg) d, options(nostack, preserves_flags)) };
}

pub fn lidt(d: &DescriptorTableRegister) {
    unsafe { asm!("lidt [{}]", in(reg) d, options(nostack, preserves_flags)) };
}

/// Selector of the task register (the `STR` instruction).
pub fn task_register_selector() -> u16 {
    let v: u16;
    unsafe { asm!("str {:x}", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

macro_rules! seg {
    ($name:ident, $reg:literal) => {
        pub fn $name() -> u16 {
            let v: u16;
            unsafe { asm!(concat!("mov {:x}, ", $reg), out(reg) v, options(nomem, nostack, preserves_flags)) };
            v
        }
    };
}
seg!(cs, "cs");
seg!(ss, "ss");
seg!(ds, "ds");
seg!(es, "es");
seg!(fs, "fs");
seg!(gs, "gs");

/// Lets a pending host interrupt run its handler, leaving the interrupt flag as it was. VMX clears
/// RFLAGS.IF on every VM exit and the stubs do not restore it, so after a physical-interrupt exit
/// this opens the window. On SVM `STGI` already ran with IF set and the interrupt was taken.
pub fn host_interrupt_window() {
    if interrupts_enabled() {
        return;
    }
    // STI delays interrupt recognition by one instruction, so the pending interrupt is taken
    // inside the nops, before CLI.
    unsafe { asm!("sti", "nop", "nop", "cli", options(nomem, nostack)) };
}

pub fn interrupts_enabled() -> bool {
    let f: u64;
    // No `nostack`: pushfq uses the stack, below the red zone the compiler accounts for.
    unsafe { asm!("pushfq", "pop {}", out(reg) f, options(nomem, preserves_flags)) };
    f & RFLAGS_IF != 0
}

impl FpuState {
    pub fn save(&mut self) {
        unsafe {
            asm!("fxsave64 [{}]", in(reg) self.0.as_mut_ptr(), options(nostack, preserves_flags))
        };
    }

    pub fn restore(&self) {
        unsafe {
            asm!("fxrstor64 [{}]", in(reg) self.0.as_ptr(), options(nostack, preserves_flags))
        };
    }
}

// World-switch stubs. Both follow the Windows x64 calling convention (the UEFI one): arguments in
// rcx, rdx, r8, callee-saved rbx rbp rdi rsi r12-r15. The guest overwrites all of them, so they
// are pushed on entry and popped on the way out. Guest registers other than the ones the VMCB or
// VMCS holds travel through `Gprs`, addressed with `offset_of!`.
//
// uemu_svm_enter(vmcb = rcx, host_vmcb = rdx, gprs = r8). After a pad slot the arguments are
// pushed, so [rsp] = vmcb, [rsp + 8] = host_vmcb, [rsp + 16] = gprs. GIF is clear from CLGI to
// STGI so no interrupt is taken while hidden state is swapped. VMRUN switches only part of the
// host state; the rest (FS, GS, TR, LDTR, KernelGSbase, STAR, LSTAR, CSTAR, SFMASK, SYSENTER) is
// saved into `host_vmcb` by VMSAVE before and reloaded by VMLOAD after. The guest's copy of that
// state is loaded from `vmcb` before VMRUN and saved back after. Returns 0 in rax.
//
// uemu_vmx_enter(gprs = rcx, resume = edx). `gprs` stays on the stack at [rsp], and that rsp is
// written to the VMCS as the host RSP, so uemu_vmx_exit (the host RIP) finds it again at [rsp].
// `test edx, edx` comes before the register loads, which are all `mov` and leave ZF intact, to
// choose between VMLAUNCH and VMRESUME. Falling through either of them means VMfail: return 1.
// A VM exit lands in uemu_vmx_exit with the guest's registers and the host's rsp; it saves the
// registers, then joins the shared epilogue with eax = 0.
core::arch::global_asm!(
    r#"
    .global uemu_svm_enter
    uemu_svm_enter:
        push rbx
        push rbp
        push rdi
        push rsi
        push r12
        push r13
        push r14
        push r15
        sub rsp, 8
        push r8
        push rdx
        push rcx
        clgi
        mov rax, [rsp + 8]
        vmsave rax
        mov rax, [rsp + 16]
        mov rcx, [rax + {RCX}]
        mov rdx, [rax + {RDX}]
        mov rbx, [rax + {RBX}]
        mov rbp, [rax + {RBP}]
        mov rsi, [rax + {RSI}]
        mov rdi, [rax + {RDI}]
        mov r8, [rax + {R8}]
        mov r9, [rax + {R9}]
        mov r10, [rax + {R10}]
        mov r11, [rax + {R11}]
        mov r12, [rax + {R12}]
        mov r13, [rax + {R13}]
        mov r14, [rax + {R14}]
        mov r15, [rax + {R15}]
        mov rax, [rsp]
        vmload rax
        vmrun rax
        mov rax, [rsp]
        vmsave rax
        push rax
        mov rax, [rsp + 24]
        mov [rax + {RCX}], rcx
        mov [rax + {RDX}], rdx
        mov [rax + {RBX}], rbx
        mov [rax + {RBP}], rbp
        mov [rax + {RSI}], rsi
        mov [rax + {RDI}], rdi
        mov [rax + {R8}], r8
        mov [rax + {R9}], r9
        mov [rax + {R10}], r10
        mov [rax + {R11}], r11
        mov [rax + {R12}], r12
        mov [rax + {R13}], r13
        mov [rax + {R14}], r14
        mov [rax + {R15}], r15
        pop rax
        mov rax, [rsp + 8]
        vmload rax
        stgi
        add rsp, 24
        add rsp, 8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rsi
        pop rdi
        pop rbp
        pop rbx
        xor eax, eax
        ret

    .global uemu_vmx_enter
    uemu_vmx_enter:
        push rbx
        push rbp
        push rdi
        push rsi
        push r12
        push r13
        push r14
        push r15
        push rcx
        mov rax, {HOST_RSP}
        vmwrite rax, rsp
        mov rax, rcx
        test edx, edx
        mov rcx, [rax + {RCX}]
        mov rdx, [rax + {RDX}]
        mov rbx, [rax + {RBX}]
        mov rbp, [rax + {RBP}]
        mov rsi, [rax + {RSI}]
        mov rdi, [rax + {RDI}]
        mov r8, [rax + {R8}]
        mov r9, [rax + {R9}]
        mov r10, [rax + {R10}]
        mov r11, [rax + {R11}]
        mov r12, [rax + {R12}]
        mov r13, [rax + {R13}]
        mov r14, [rax + {R14}]
        mov r15, [rax + {R15}]
        mov rax, [rax]
        jnz 2f
        vmlaunch
        jmp 3f
    2:  vmresume
    3:  mov eax, 1
        jmp 4f
    .global uemu_vmx_exit
    uemu_vmx_exit:
        push rax
        mov rax, [rsp + 8]
        mov [rax + {RCX}], rcx
        mov [rax + {RDX}], rdx
        mov [rax + {RBX}], rbx
        mov [rax + {RBP}], rbp
        mov [rax + {RSI}], rsi
        mov [rax + {RDI}], rdi
        mov [rax + {R8}], r8
        mov [rax + {R9}], r9
        mov [rax + {R10}], r10
        mov [rax + {R11}], r11
        mov [rax + {R12}], r12
        mov [rax + {R13}], r13
        mov [rax + {R14}], r14
        mov [rax + {R15}], r15
        pop rcx
        mov [rax], rcx
        xor eax, eax
    4:  add rsp, 8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rsi
        pop rdi
        pop rbp
        pop rbx
        ret
    "#,
    RCX = const offset_of!(Gprs, rcx),
    RDX = const offset_of!(Gprs, rdx),
    RBX = const offset_of!(Gprs, rbx),
    RBP = const offset_of!(Gprs, rbp),
    RSI = const offset_of!(Gprs, rsi),
    RDI = const offset_of!(Gprs, rdi),
    R8 = const offset_of!(Gprs, r8),
    R9 = const offset_of!(Gprs, r9),
    R10 = const offset_of!(Gprs, r10),
    R11 = const offset_of!(Gprs, r11),
    R12 = const offset_of!(Gprs, r12),
    R13 = const offset_of!(Gprs, r13),
    R14 = const offset_of!(Gprs, r14),
    R15 = const offset_of!(Gprs, r15),
    HOST_RSP = const super::vmx::VMCS_HOST_RSP,
);

unsafe extern "C" {
    /// `rcx` = guest VMCB physical address, `rdx` = host save VMCB physical address,
    /// `r8` = `Gprs`. Returns 0.
    pub fn uemu_svm_enter(vmcb: u64, host: u64, gprs: *mut Gprs) -> u64;
    /// `rcx` = `Gprs`, `edx` = nonzero to VMRESUME. Returns 0 after a VM exit, 1 if VMLAUNCH
    /// or VMRESUME failed.
    pub fn uemu_vmx_enter(gprs: *mut Gprs, resume: u32) -> u64;
    /// VMCS host RIP; reached only by a VM exit, never called.
    pub fn uemu_vmx_exit();
}
