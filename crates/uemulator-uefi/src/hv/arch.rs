use core::arch::asm;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Gprs {
    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rbx: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

pub const MSR_FEATURE_CONTROL: u32 = 0x3a;
pub const MSR_EFER: u32 = 0xc000_0080;
pub const MSR_STAR: u32 = 0xc000_0081;
pub const MSR_LSTAR: u32 = 0xc000_0082;
pub const MSR_CSTAR: u32 = 0xc000_0083;
pub const MSR_SFMASK: u32 = 0xc000_0084;
pub const MSR_FS_BASE: u32 = 0xc000_0100;
pub const MSR_GS_BASE: u32 = 0xc000_0101;
pub const MSR_KERNEL_GS_BASE: u32 = 0xc000_0102;
pub const MSR_PAT: u32 = 0x277;
pub const MSR_SYSENTER_CS: u32 = 0x174;
pub const MSR_SYSENTER_ESP: u32 = 0x175;
pub const MSR_SYSENTER_EIP: u32 = 0x176;

pub const EFER_SCE: u64 = 1;
pub const EFER_LME: u64 = 1 << 8;
pub const EFER_LMA: u64 = 1 << 10;
pub const EFER_NXE: u64 = 1 << 11;
pub const EFER_SVME: u64 = 1 << 12;

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

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct DescTable {
    pub limit: u16,
    pub base: u64,
}

pub fn sgdt() -> DescTable {
    let mut d = DescTable::default();
    unsafe { asm!("sgdt [{}]", in(reg) &mut d, options(nostack, preserves_flags)) };
    d
}

pub fn sidt() -> DescTable {
    let mut d = DescTable::default();
    unsafe { asm!("sidt [{}]", in(reg) &mut d, options(nostack, preserves_flags)) };
    d
}

pub fn lgdt(d: &DescTable) {
    unsafe { asm!("lgdt [{}]", in(reg) d, options(nostack, preserves_flags)) };
}

pub fn lidt(d: &DescTable) {
    unsafe { asm!("lidt [{}]", in(reg) d, options(nostack, preserves_flags)) };
}

pub fn str_sel() -> u16 {
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

/// Lets a pending host interrupt run its handler, leaving the interrupt flag as it was.
pub fn host_interrupt_window() {
    if interrupts_enabled() {
        return;
    }
    unsafe { asm!("sti", "nop", "nop", "cli", options(nomem, nostack)) };
}

pub fn interrupts_enabled() -> bool {
    let f: u64;
    unsafe { asm!("pushfq", "pop {}", out(reg) f, options(nomem, preserves_flags)) };
    f & (1 << 9) != 0
}

/// 512-byte FXSAVE image, 16-byte aligned.
#[repr(C, align(16))]
pub struct FpuState(pub [u8; 512]);

impl FpuState {
    pub fn reset() -> Self {
        let mut s = FpuState([0; 512]);
        s.0[0..2].copy_from_slice(&0x37fu16.to_le_bytes());
        s.0[24..28].copy_from_slice(&0x1f80u32.to_le_bytes());
        s
    }

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

// The guest's general registers except what the VMCB or VMCS holds (SVM: rax, rsp; VMX: rsp)
// travel through `Gprs`. Both stubs follow the Windows x64 calling convention.
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
        mov rcx, [rax + 8]
        mov rdx, [rax + 16]
        mov rbx, [rax + 24]
        mov rbp, [rax + 40]
        mov rsi, [rax + 48]
        mov rdi, [rax + 56]
        mov r8, [rax + 64]
        mov r9, [rax + 72]
        mov r10, [rax + 80]
        mov r11, [rax + 88]
        mov r12, [rax + 96]
        mov r13, [rax + 104]
        mov r14, [rax + 112]
        mov r15, [rax + 120]
        mov rax, [rsp]
        vmload rax
        vmrun rax
        mov rax, [rsp]
        vmsave rax
        push rax
        mov rax, [rsp + 24]
        mov [rax + 8], rcx
        mov [rax + 16], rdx
        mov [rax + 24], rbx
        mov [rax + 40], rbp
        mov [rax + 48], rsi
        mov [rax + 56], rdi
        mov [rax + 64], r8
        mov [rax + 72], r9
        mov [rax + 80], r10
        mov [rax + 88], r11
        mov [rax + 96], r12
        mov [rax + 104], r13
        mov [rax + 112], r14
        mov [rax + 120], r15
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
        mov rax, 0x6c14
        vmwrite rax, rsp
        mov rax, rcx
        test edx, edx
        mov rcx, [rax + 8]
        mov rdx, [rax + 16]
        mov rbx, [rax + 24]
        mov rbp, [rax + 40]
        mov rsi, [rax + 48]
        mov rdi, [rax + 56]
        mov r8, [rax + 64]
        mov r9, [rax + 72]
        mov r10, [rax + 80]
        mov r11, [rax + 88]
        mov r12, [rax + 96]
        mov r13, [rax + 104]
        mov r14, [rax + 112]
        mov r15, [rax + 120]
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
        mov [rax + 8], rcx
        mov [rax + 16], rdx
        mov [rax + 24], rbx
        mov [rax + 40], rbp
        mov [rax + 48], rsi
        mov [rax + 56], rdi
        mov [rax + 64], r8
        mov [rax + 72], r9
        mov [rax + 80], r10
        mov [rax + 88], r11
        mov [rax + 96], r12
        mov [rax + 104], r13
        mov [rax + 112], r14
        mov [rax + 120], r15
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
    "#
);

unsafe extern "C" {
    /// `rcx` = guest VMCB physical address, `rdx` = host save VMCB physical address,
    /// `r8` = `Gprs`. Returns 0.
    pub fn uemu_svm_enter(vmcb: u64, host: u64, gprs: *mut Gprs) -> u64;
    /// `rcx` = `Gprs`, `edx` = nonzero to VMRESUME. Returns 0 after a VM exit, 1 if VMLAUNCH
    /// or VMRESUME failed.
    pub fn uemu_vmx_enter(gprs: *mut Gprs, resume: u32) -> u64;
    pub fn uemu_vmx_exit();
}
