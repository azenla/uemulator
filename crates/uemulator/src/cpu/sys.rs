use super::decode::*;
use super::exception::{Exception, VECTOR_SEGMENT_NOT_PRESENT};
use super::exec::Loc;
use super::flags::*;
use super::fpu::MXCSR_VALID_BITS;
use super::*;

pub(crate) const MSR_TSC: u32 = 0x10;
pub(crate) const MSR_APIC_BASE: u32 = 0x1b;
pub(crate) const MSR_SYSENTER_CS: u32 = 0x174;
pub(crate) const MSR_SYSENTER_ESP: u32 = 0x175;
pub(crate) const MSR_SYSENTER_EIP: u32 = 0x176;
pub(crate) const MSR_PAT: u32 = 0x277;
pub(crate) const MSR_EFER: u32 = 0xc000_0080;
pub(crate) const MSR_STAR: u32 = 0xc000_0081;
pub(crate) const MSR_LSTAR: u32 = 0xc000_0082;
pub(crate) const MSR_CSTAR: u32 = 0xc000_0083;
pub(crate) const MSR_SFMASK: u32 = 0xc000_0084;
pub(crate) const MSR_FS_BASE: u32 = 0xc000_0100;
pub(crate) const MSR_GS_BASE: u32 = 0xc000_0101;
pub(crate) const MSR_KERNEL_GS_BASE: u32 = 0xc000_0102;
pub(crate) const MSR_TSC_AUX: u32 = 0xc000_0103;

// CR0_PE/MP/ET/PG, CR4_PAE and EFER_LME/LMA/NXE live in cpu/mod.rs.
pub(crate) const CR0_EM: u64 = 1 << 2;
pub(crate) const CR0_TS: u64 = 1 << 3;
const CR0_NW: u64 = 1 << 29;
const CR0_CD: u64 = 1 << 30;

pub(crate) const CR4_TSD: u64 = 1 << 2;
pub(crate) const CR4_DE: u64 = 1 << 3;
pub(crate) const CR4_PSE: u64 = 1 << 4;
pub(crate) const CR4_PGE: u64 = 1 << 7;
pub(crate) const CR4_PCE: u64 = 1 << 8;
pub(crate) const CR4_OSFXSR: u64 = 1 << 9;
pub(crate) const CR4_OSXMMEXCPT: u64 = 1 << 10;
/// CR4 bits a guest may set. Everything else (VME, PVI, MCE, UMIP, LA57, FSGSBASE, PCIDE,
/// OSXSAVE, SMEP, SMAP...) is rejected with #GP because CPUID does not advertise it.
const CR4_SUPPORTED: u64 =
    CR4_TSD | CR4_DE | CR4_PSE | CR4_PAE | CR4_PGE | CR4_PCE | CR4_OSFXSR | CR4_OSXMMEXCPT;

/// Address widths reported by CPUID 0x8000_0008. CR3 and the canonical-address checks follow them.
const PHYSICAL_ADDRESS_BITS: u32 = 40;
const VIRTUAL_ADDRESS_BITS: u32 = 48;

/// CR3 bits kept on write: physical address bits 51:12 plus PWT (bit 3) and PCD (bit 4).
/// Everything else (bits 2:0 and 11:5, the PCID field) is ignored without CR4.PCIDE.
const CR3_MASK: u64 = 0x000f_ffff_ffff_f018;
/// Bits at or above the physical address width are reserved in CR3: writing one is #GP(0).
const CR3_RESERVED: u64 = !0u64 << PHYSICAL_ADDRESS_BITS;
/// CR8 holds the 4-bit task priority; it is stored but no APIC consumes it. Higher bits are
/// reserved.
const CR8_MASK: u64 = 0xf;

/// DR6 bits software may write: B0-B3 (3:0), BD (13), BS (14), BT (15). Bits 31:16 and 11:4
/// always read as one (no RTM, no bus-lock detect) and bit 12 as zero.
const DR6_WRITABLE: u64 = 0xe00f;
const DR6_FIXED_ONE: u64 = 0xffff_0ff0;
/// DR7 bits software may write: L0-G3, LE, GE (9:0), GD (13) and the RW/LEN fields (31:16).
/// Bit 10 always reads as one; bits 15:14 and 12:11 as zero.
const DR7_WRITABLE: u64 = 0xffff_23ff;
const DR7_FIXED_ONE: u64 = 0x400;
/// Debug register numbers. DR4 and DR5 alias DR6 and DR7 when CR4.DE is clear and are #UD
/// otherwise; DR8-DR15 do not exist.
const DR4: usize = 4;
const DR5: usize = 5;
const DR6: usize = 6;
const DR7: usize = 7;
const DEBUG_REGISTER_COUNT: usize = 8;

pub(crate) const EFER_SCE: u64 = 1 << 0;
/// wrmsr to EFER faults if any bit outside this set is 1. LMA is accepted in the value but
/// never stored: it is recomputed from CR0.PG and EFER.LME on CR0 writes.
const EFER_WRMSR_ALLOWED: u64 = EFER_SCE | EFER_LME | EFER_LMA | EFER_NXE;
const EFER_WRMSR_STORED: u64 = EFER_SCE | EFER_LME | EFER_NXE;

const EFLAGS_ID: u64 = 1 << 21;
/// Bit 1 of EFLAGS always reads as 1.
const EFLAGS_ALWAYS_ONE: u64 = 2;
/// Flags sysret restores from r11. IOPL, RF and VM are deliberately not restored.
const SYSRET_RESTORED_FLAGS: u64 = CF | PF | AF | ZF | SF | TF | IF | DF | OF | NT | AC | EFLAGS_ID;

/// Low two bits of a segment selector.
const SELECTOR_RPL_MASK: u16 = 3;
const SELECTOR_RPL_USER: u16 = 3;

/// STAR holds the sysret selector base in bits 63:48 and the syscall selector base in 47:32.
const STAR_SYSCALL_SELECTOR_SHIFT: u32 = 32;
const STAR_SYSRET_SELECTOR_SHIFT: u32 = 48;

const DESC_LIMIT_HIGH_MASK: u32 = 0xf_0000;
/// Descriptor type field (bits 11:8 of the upper dword) values of the system descriptors LDT
/// and 64-bit TSS accept; a busy TSS is the available type with bit 1 set.
const DESC_TYPE_SHIFT: u32 = 8;
const DESC_TYPE_MASK: u32 = 0xf;
const SYSTEM_TYPE_LDT: u32 = 2;
const SYSTEM_TYPE_TSS64_AVAILABLE: u32 = 9;
/// The "busy" bit of the TSS type, in its place within the descriptor's upper dword.
const SYSTEM_TYPE_BUSY_BIT: u32 = 2 << DESC_TYPE_SHIFT;
/// Conforming bit of a code segment type.
const SEG_TYPE_CONFORMING: u32 = 1 << 10;
/// Descriptors are 8 bytes, system descriptors (LDT, TSS) 16 in long mode.
const DESCRIPTOR_BYTES: u64 = 8;
const SYSTEM_DESCRIPTOR_BYTES: u64 = 16;
/// Byte offset of the upper dword of a descriptor, where type and attributes live.
const DESC_HIGH_DWORD_OFFSET: u64 = 4;
/// Selector fields: table indicator (set selects the LDT) and the descriptor byte offset.
const SELECTOR_TABLE_INDICATOR: u16 = 1 << 2;
const SELECTOR_OFFSET_MASK: u16 = 0xfff8;
const FLAT_LIMIT: u32 = 0xffff_ffff;
const DESC_DPL_USER: u32 = SEG_DPL_MASK << SEG_DPL_SHIFT;
/// Flat segments installed by syscall/sysret (base 0, 4 GiB limit).
const USER_CODE64_FLAGS: u32 = SEG_FLAGS_CODE64 | DESC_DPL_USER;
const USER_DATA_FLAGS: u32 = SEG_FLAGS_FLAT_DATA | DESC_DPL_USER;

/// `Cpu::log_once` keys for faulting MSR accesses, offset by the 32-bit MSR number so each base
/// owns a 2^32-wide range. The other users are `unimplemented_opcode` in exec.rs (small keys from
/// 0x4000_0000) and the unmapped-access keys in mem.rs (bits 62 and 61 plus a page number), so
/// these two bases sit below those in bits 60 and 59.
const LOG_KEY_RDMSR_FAULT: u64 = 1 << 60;
const LOG_KEY_WRMSR_FAULT: u64 = 1 << 59;

// CPUID identity: AuthenticAMD, family 0x15, model 0, stepping 1 (plans/03-cpu.md). Only
// features that have tests behind them are advertised; everything else reads as zero.
const CPUID_LEAF_VENDOR: u32 = 0;
const CPUID_LEAF_FEATURES: u32 = 1;
const CPUID_LEAF_EXT_MAX: u32 = 0x8000_0000;
const CPUID_LEAF_EXT_FEATURES: u32 = 0x8000_0001;
const CPUID_LEAF_BRAND_FIRST: u32 = 0x8000_0002;
const CPUID_LEAF_BRAND_LAST: u32 = 0x8000_0004;
const CPUID_LEAF_L1_CACHE_TLB: u32 = 0x8000_0005;
const CPUID_LEAF_L2_CACHE_TLB: u32 = 0x8000_0006;
const CPUID_LEAF_ADVANCED_POWER: u32 = 0x8000_0007;
const CPUID_LEAF_ADDRESS_SIZES: u32 = 0x8000_0008;

/// Highest standard leaf. Leaf 7 is within range but returns all zeros (no extended features).
const CPUID_MAX_STANDARD_LEAF: u32 = 7;
const CPUID_MAX_EXTENDED_LEAF: u32 = 0x8000_0008;

/// Vendor string "AuthenticAMD" is returned in EBX, EDX, ECX order.
const VENDOR_EBX: u32 = u32::from_le_bytes(*b"Auth");
const VENDOR_EDX: u32 = u32::from_le_bytes(*b"enti");
const VENDOR_ECX: u32 = u32::from_le_bytes(*b"cAMD");

/// Returned as EAX of leaf 1 and leaf 0x8000_0001: stepping 1, model 0, base family 0xf plus
/// extended family 6.
const CPUID_SIGNATURE: u32 = 0x0060_0f01;
/// Leaf 1 EBX: CLFLUSH line size in 8-byte units (bits 15:8) and logical processor count
/// (23:16). The initial APIC id (31:24) is 0.
const CPUID_LEAF1_EBX: u32 = (1 << 16) | (8 << 8);

const FEATURE_FPU: u32 = 1 << 0;
const FEATURE_DE: u32 = 1 << 2;
const FEATURE_PSE: u32 = 1 << 3;
const FEATURE_TSC: u32 = 1 << 4;
const FEATURE_MSR: u32 = 1 << 5;
const FEATURE_PAE: u32 = 1 << 6;
const FEATURE_CX8: u32 = 1 << 8;
const FEATURE_PGE: u32 = 1 << 13;
const FEATURE_CMOV: u32 = 1 << 15;
const FEATURE_PAT: u32 = 1 << 16;
const FEATURE_CLFSH: u32 = 1 << 19;
const FEATURE_MMX: u32 = 1 << 23;
const FEATURE_FXSR: u32 = 1 << 24;
const FEATURE_SSE: u32 = 1 << 25;
const FEATURE_SSE2: u32 = 1 << 26;
/// Leaf 1 EDX. MMX is advertised because glibc's x86-64 baseline ISA check (CMOV, CX8, FPU,
/// FXSR, MMX, SSE, SSE2) needs the bit; without it every Debian binary refuses to start ("CPU ISA
/// level is lower than required"). APIC, SEP and PSE36 are clear.
const CPUID_LEAF1_EDX: u32 = FEATURE_FPU
    | FEATURE_DE
    | FEATURE_PSE
    | FEATURE_TSC
    | FEATURE_MSR
    | FEATURE_PAE
    | FEATURE_CX8
    | FEATURE_PGE
    | FEATURE_CMOV
    | FEATURE_PAT
    | FEATURE_CLFSH
    | FEATURE_MMX
    | FEATURE_FXSR
    | FEATURE_SSE
    | FEATURE_SSE2;

// Leaf 1 ECX. Each bit is advertised together with the tests of the instructions behind it.
// MONITOR/MWAIT, VMX/SVM, XSAVE/OSXSAVE, AVX, RDRAND and the hypervisor bit stay clear.
/// SSE3: movddup, movshdup, movsldup, lddqu, addsub*, hadd*, hsub*, and x87 fisttp.
const FEATURE_SSE3: u32 = 1 << 0;
/// CMPXCHG16B.
const FEATURE_CX16: u32 = 1 << 13;
/// SSSE3, XMM forms only (the MMX forms are #UD, see `sse_ext.rs`).
const FEATURE_SSSE3: u32 = 1 << 9;
const FEATURE_SSE4_1: u32 = 1 << 19;
/// SSE4.2: pcmpgtq, CRC32 and the pcmp[ei]str[im] string instructions.
const FEATURE_SSE4_2: u32 = 1 << 20;
const FEATURE_POPCNT: u32 = 1 << 23;
const CPUID_LEAF1_ECX: u32 =
    FEATURE_SSE3 | FEATURE_SSSE3 | FEATURE_CX16 | FEATURE_SSE4_1 | FEATURE_SSE4_2 | FEATURE_POPCNT;

/// Leaf 0x8000_0001 EDX repeats this subset of the leaf 1 bits (the AMD aliases); DE and PSE
/// are not mirrored.
const CPUID_EXT_ALIASED_EDX: u32 = FEATURE_FPU
    | FEATURE_TSC
    | FEATURE_MSR
    | FEATURE_PAE
    | FEATURE_CX8
    | FEATURE_PGE
    | FEATURE_CMOV
    | FEATURE_PAT
    | FEATURE_MMX
    | FEATURE_FXSR;
const EXT_FEATURE_SYSCALL: u32 = 1 << 11;
const EXT_FEATURE_NX: u32 = 1 << 20;
const EXT_FEATURE_LONG_MODE: u32 = 1 << 29;
/// Leaf 0x8000_0001 ECX bit 0: LAHF/SAHF in 64-bit mode.
const EXT_FEATURE_LAHF_LM: u32 = 1 << 0;
/// Leaf 0x8000_0001 ECX bit 5: advanced bit manipulation, i.e. LZCNT (POPCNT has its own bit in
/// leaf 1).
const EXT_FEATURE_ABM: u32 = 1 << 5;
/// Leaf 0x8000_0007 EDX bit 8: invariant TSC, so the kernel trusts it as a clocksource.
const APM_INVARIANT_TSC: u32 = 1 << 8;

/// Leaf 0x8000_0008 EAX: physical address bits (7:0) and linear address bits (15:8).
const CPUID_ADDRESS_SIZES: u32 = (VIRTUAL_ADDRESS_BITS << 8) | PHYSICAL_ADDRESS_BITS;

/// 48 bytes returned over leaves 0x8000_0002..=0x8000_0004, NUL padded.
const BRAND_STRING: &[u8; 48] =
    b"uemulator virtual x86-64 cpu\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

/// Indices into `Cpu::gpr`: hardware register numbers, so RCX is 1 and RBX is 3.
const RAX: usize = 0;
const RCX: usize = 1;
const RDX: usize = 2;
const RBX: usize = 3;
const R11: usize = 11;

/// Whether bits 63:47 of `address` are all copies of bit 47.
fn is_canonical(address: u64) -> bool {
    let sign_extension = (address as i64) >> (VIRTUAL_ADDRESS_BITS - 1);
    sign_extension == 0 || sign_extension == -1
}

impl Cpu {
    /// #GP(0) unless CPL is 0.
    pub fn need_cpl0(&self) -> ExecResult<()> {
        if self.cpl != 0 {
            Err(Exception::general_protection(0))
        } else {
            Ok(())
        }
    }

    /// Subleaf (ECX) is ignored; unknown leaves return zeros.
    pub fn cpuid(&mut self) {
        let leaf = self.gpr[RAX] as u32;
        let (eax, ebx, ecx, edx): (u32, u32, u32, u32) = match leaf {
            CPUID_LEAF_VENDOR => (CPUID_MAX_STANDARD_LEAF, VENDOR_EBX, VENDOR_ECX, VENDOR_EDX),
            CPUID_LEAF_FEATURES => (
                CPUID_SIGNATURE,
                CPUID_LEAF1_EBX,
                CPUID_LEAF1_ECX,
                CPUID_LEAF1_EDX,
            ),
            CPUID_LEAF_EXT_MAX => (CPUID_MAX_EXTENDED_LEAF, VENDOR_EBX, VENDOR_ECX, VENDOR_EDX),
            CPUID_LEAF_EXT_FEATURES => (
                CPUID_SIGNATURE,
                0,
                EXT_FEATURE_LAHF_LM | EXT_FEATURE_ABM,
                EXT_FEATURE_SYSCALL
                    | EXT_FEATURE_NX
                    | EXT_FEATURE_LONG_MODE
                    | (CPUID_LEAF1_EDX & CPUID_EXT_ALIASED_EDX),
            ),
            CPUID_LEAF_BRAND_FIRST..=CPUID_LEAF_BRAND_LAST => {
                let offset = (leaf - CPUID_LEAF_BRAND_FIRST) as usize * 16;
                let word = |k: usize| {
                    u32::from_le_bytes(
                        BRAND_STRING[offset + k * 4..offset + k * 4 + 4]
                            .try_into()
                            .unwrap(),
                    )
                };
                (word(0), word(1), word(2), word(3))
            }
            // Fixed AMD-layout cache/TLB descriptors: L1D 32 KiB 8-way, L1I 64 KiB 2-way,
            // L2 512 KiB, all with 64-byte lines; no L3.
            CPUID_LEAF_L1_CACHE_TLB => (0xff40_ff40, 0xff40_ff40, 0x2008_0140, 0x4002_0140),
            CPUID_LEAF_L2_CACHE_TLB => (0x6400_6400, 0x6400_4200, 0x0200_6140, 0),
            CPUID_LEAF_ADVANCED_POWER => (0, 0, 0, APM_INVARIANT_TSC),
            CPUID_LEAF_ADDRESS_SIZES => (CPUID_ADDRESS_SIZES, 0, 0, 0),
            _ => (0, 0, 0, 0),
        };
        self.gpr[RAX] = eax as u64;
        self.gpr[RBX] = ebx as u64;
        self.gpr[RCX] = ecx as u64;
        self.gpr[RDX] = edx as u64;
    }

    /// Unknown MSRs raise #GP(0) (the kernel probes with `rdmsr_safe`); each is logged once.
    pub fn rdmsr(&mut self, msr: u32) -> ExecResult<u64> {
        Ok(match msr {
            MSR_TSC => self.tsc(),
            // no local APIC is advertised
            MSR_APIC_BASE => return Err(Exception::general_protection(0)),
            MSR_PAT => self.pat,
            MSR_EFER => self.efer,
            MSR_STAR => self.star,
            MSR_LSTAR => self.lstar,
            MSR_CSTAR => self.cstar,
            MSR_SFMASK => self.sfmask,
            // FS/GS bases live in the segment cache, not in separate fields
            MSR_FS_BASE => self.seg[SEG_FS].base,
            MSR_GS_BASE => self.seg[SEG_GS].base,
            MSR_KERNEL_GS_BASE => self.kgsbase,
            MSR_TSC_AUX => self.tsc_aux,
            MSR_SYSENTER_CS..=MSR_SYSENTER_EIP => self.sysenter[(msr - MSR_SYSENTER_CS) as usize],
            _ => match self.msr_misc.get(&msr) {
                Some(v) => *v,
                None => {
                    let rip = self.rip;
                    self.log_once(LOG_KEY_RDMSR_FAULT + msr as u64, || {
                        format!("rdmsr {msr:#x} -> #GP at rip {rip:#x}")
                    });
                    return Err(Exception::general_protection(0));
                }
            },
        })
    }

    /// Writes to MSRs in `msr_misc` are stored and read back but have no effect. Unknown MSRs
    /// raise #GP(0).
    pub fn wrmsr(&mut self, msr: u32, v: u64) -> ExecResult<()> {
        // Addresses that the CPU will later jump or index through must be canonical at write
        // time, or swapgs/syscall/sysenter would fault in the kernel entry path instead.
        let holds_address = matches!(
            msr,
            MSR_FS_BASE
                | MSR_GS_BASE
                | MSR_KERNEL_GS_BASE
                | MSR_LSTAR
                | MSR_CSTAR
                | MSR_SYSENTER_ESP
                | MSR_SYSENTER_EIP
        );
        if holds_address && !is_canonical(v) {
            return Err(Exception::general_protection(0));
        }
        match msr {
            MSR_PAT => self.pat = v,
            MSR_EFER => {
                if v & !EFER_WRMSR_ALLOWED != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.efer = (self.efer & !EFER_WRMSR_STORED) | (v & EFER_WRMSR_STORED);
            }
            MSR_STAR => self.star = v,
            MSR_LSTAR => self.lstar = v,
            MSR_CSTAR => self.cstar = v,
            MSR_SFMASK => self.sfmask = v,
            MSR_FS_BASE => self.seg[SEG_FS].base = v,
            MSR_GS_BASE => self.seg[SEG_GS].base = v,
            MSR_KERNEL_GS_BASE => self.kgsbase = v,
            MSR_TSC_AUX => self.tsc_aux = v,
            // the TSC follows the virtual clock and cannot be set
            MSR_TSC => {}
            MSR_SYSENTER_CS..=MSR_SYSENTER_EIP => {
                self.sysenter[(msr - MSR_SYSENTER_CS) as usize] = v
            }
            _ if self.msr_misc.contains_key(&msr) => {
                self.msr_misc.insert(msr, v);
            }
            _ => {
                let rip = self.rip;
                self.log_once(LOG_KEY_WRMSR_FAULT + msr as u64, || {
                    format!("wrmsr {msr:#x} <- {v:#x} -> #GP at rip {rip:#x}")
                });
                return Err(Exception::general_protection(0));
            }
        }
        Ok(())
    }

    /// 0F 20..23: mov to/from control and debug registers. ModRM.reg is the CR/DR number
    /// (REX.R makes CR8 or DR8-DR15) and ModRM.rm the GPR; mod is ignored. Nothing triggers
    /// hardware breakpoints, so the debug registers are validated storage.
    pub fn mov_cr_dr(&mut self, i: &Insn) -> ExecResult<()> {
        const OP_MOV_FROM_CR: u16 = 0x120;
        const OP_MOV_FROM_DR: u16 = 0x121;
        const OP_MOV_TO_CR: u16 = 0x122;
        self.need_cpl0()?;
        let register = i.reg as usize;
        let gpr = i.rm as usize;
        match i.op {
            OP_MOV_FROM_CR => {
                self.gpr[gpr] = match register {
                    0 => self.cr0,
                    2 => self.cr2,
                    3 => self.cr3,
                    4 => self.cr4,
                    8 => self.cr8,
                    _ => return Err(Exception::invalid_opcode()),
                }
            }
            OP_MOV_TO_CR => {
                let v = self.gpr[gpr];
                match register {
                    0 => self.write_cr0(v)?,
                    2 => self.cr2 = v,
                    3 => {
                        if v & CR3_RESERVED != 0 {
                            return Err(Exception::general_protection(0));
                        }
                        self.cr3 = v & CR3_MASK;
                        self.flush_tlb();
                    }
                    4 => {
                        // PAE cannot be turned off under IA-32e paging.
                        if v & !CR4_SUPPORTED != 0
                            || (self.efer & EFER_LMA != 0 && v & CR4_PAE == 0)
                        {
                            return Err(Exception::general_protection(0));
                        }
                        self.cr4 = v;
                        self.flush_tlb();
                    }
                    8 => {
                        if v & !CR8_MASK != 0 {
                            return Err(Exception::general_protection(0));
                        }
                        self.cr8 = v;
                    }
                    _ => return Err(Exception::invalid_opcode()),
                }
            }
            OP_MOV_FROM_DR => {
                let index = self.debug_register_index(register)?;
                self.gpr[gpr] = self.dr[index];
            }
            _ => {
                let index = self.debug_register_index(register)?;
                let v = self.gpr[gpr];
                self.write_debug_register(index, v)?;
            }
        }
        Ok(())
    }

    /// Validates a CR0 write (SDM Vol. 3A 2.5 and 10.8.5) before anything changes. LMA is
    /// derived here: long mode is active exactly when paging is on with EFER.LME set.
    fn write_cr0(&mut self, v: u64) -> ExecResult<()> {
        let paging = v & CR0_PG != 0;
        let upper_half_set = v >> 32 != 0;
        let not_write_through_without_cache_disable = v & CR0_NW != 0 && v & CR0_CD == 0;
        let paging_without_protection = paging && v & CR0_PE == 0;
        // Paging is switched off from compatibility mode (that is how software leaves IA-32e
        // mode); doing it from 64-bit code would leave that code running unpaged.
        let leaving_long_mode = self.efer & EFER_LMA != 0 && !paging && self.seg[SEG_CS].long();
        let entering_long_mode_without_pae =
            paging && self.efer & EFER_LME != 0 && self.cr4 & CR4_PAE == 0;
        if upper_half_set
            || not_write_through_without_cache_disable
            || paging_without_protection
            || leaving_long_mode
            || entering_long_mode_without_pae
        {
            return Err(Exception::general_protection(0));
        }
        self.cr0 = v | CR0_ET;
        if paging && self.efer & EFER_LME != 0 {
            self.efer |= EFER_LMA;
        } else {
            self.efer &= !EFER_LMA;
        }
        self.flush_tlb();
        Ok(())
    }

    /// Maps a MOV DR register number to its slot in `dr`: DR4/DR5 alias DR6/DR7 unless CR4.DE
    /// makes them #UD, and numbers 8-15 (REX.R) do not exist.
    fn debug_register_index(&self, register: usize) -> ExecResult<usize> {
        match register {
            DR4 | DR5 if self.cr4 & CR4_DE != 0 => Err(Exception::invalid_opcode()),
            DR4 | DR5 => Ok(register + (DR6 - DR4)),
            _ if register < DEBUG_REGISTER_COUNT => Ok(register),
            _ => Err(Exception::invalid_opcode()),
        }
    }

    fn write_debug_register(&mut self, index: usize, v: u64) -> ExecResult<()> {
        match index {
            DR6 | DR7 => {
                if v >> 32 != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.dr[index] = if index == DR6 {
                    (v & DR6_WRITABLE) | DR6_FIXED_ONE
                } else {
                    (v & DR7_WRITABLE) | DR7_FIXED_ONE
                };
            }
            _ => self.dr[index] = v,
        }
        Ok(())
    }

    /// `next` is the address of the following instruction. SYSCALL loads CS from STAR[47:32]
    /// and SS from that plus 8, always flat, and masks RFLAGS with SFMASK.
    pub fn do_syscall(&mut self, next: u64) -> ExecResult<()> {
        if self.efer & EFER_SCE == 0 {
            return Err(Exception::invalid_opcode());
        }
        let rflags = self.eflags_full();
        self.gpr[RCX] = next;
        self.gpr[R11] = rflags & !RF;
        let selector_base = (self.star >> STAR_SYSCALL_SELECTOR_SHIFT) as u16 & !SELECTOR_RPL_MASK;
        self.seg[SEG_CS] = Seg {
            sel: selector_base,
            base: 0,
            limit: FLAT_LIMIT,
            flags: SEG_FLAGS_CODE64,
        };
        self.seg[SEG_SS] = Seg {
            sel: selector_base + 8,
            base: 0,
            limit: FLAT_LIMIT,
            flags: SEG_FLAGS_FLAT_DATA,
        };
        self.eflags = (rflags & !self.sfmask & !RF) | EFLAGS_ALWAYS_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.set_cpl(0);
        self.rip = self.lstar;
        Ok(())
    }

    /// Only the 64-bit form (REX.W, `sysretq`) is implemented: CS = STAR[63:48] + 16 and
    /// SS = STAR[63:48] + 8, both RPL 3. Returning to compatibility mode is #UD (logged).
    pub fn do_sysret(&mut self, i: &Insn) -> ExecResult<()> {
        if self.efer & EFER_SCE == 0 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()?;
        if i.opsize != 8 {
            return Err(self.unimplemented_opcode(i));
        }
        let selector_base = (self.star >> STAR_SYSRET_SELECTOR_SHIFT) as u16 & !SELECTOR_RPL_MASK;
        self.seg[SEG_CS] = Seg {
            sel: (selector_base + 16) | SELECTOR_RPL_USER,
            base: 0,
            limit: FLAT_LIMIT,
            flags: USER_CODE64_FLAGS,
        };
        self.seg[SEG_SS] = Seg {
            sel: (selector_base + 8) | SELECTOR_RPL_USER,
            base: 0,
            limit: FLAT_LIMIT,
            flags: USER_DATA_FLAGS,
        };
        let r11 = self.gpr[R11];
        self.eflags = (r11 & SYSRET_RESTORED_FLAGS) | EFLAGS_ALWAYS_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.set_cpl(3);
        self.rip = self.gpr[RCX];
        Ok(())
    }

    /// Base and byte-granular limit of a 16-byte system descriptor (LDT/TSS) given as two
    /// quadwords. Base bits 63:32 come from the second quadword.
    fn system_descriptor_base_limit(lo: u64, hi: u64) -> (u64, u32) {
        let dword0 = lo as u32;
        let dword1 = (lo >> 32) as u32;
        let base = ((dword0 >> 16) | ((dword1 & 0xff) << 16) | (dword1 & 0xff00_0000)) as u64
            | ((hi & 0xffff_ffff) << 32);
        let mut limit = (dword0 & 0xffff) | (dword1 & DESC_LIMIT_HIGH_MASK);
        if dword1 & SEG_GRANULARITY != 0 {
            limit = (limit << 12) | 0xfff;
        }
        (base, limit)
    }

    /// LLDT: the descriptor is a 16-byte system descriptor in the GDT. A null selector marks the
    /// LDT invalid but still becomes the visible selector, which SLDT reads back.
    fn lldt(&mut self, sel: u16) -> ExecResult<()> {
        if sel & !SELECTOR_RPL_MASK == 0 {
            self.ldtr = Seg {
                sel,
                ..Seg::default()
            };
            return Ok(());
        }
        let seg = self.load_system_descriptor(sel, SYSTEM_TYPE_LDT)?;
        self.ldtr = seg;
        Ok(())
    }

    /// LTR: only an available 64-bit TSS loads, and doing so marks it busy in the GDT.
    fn ltr(&mut self, sel: u16) -> ExecResult<()> {
        if sel & !SELECTOR_RPL_MASK == 0 {
            return Err(Exception::general_protection(0));
        }
        let mut seg = self.load_system_descriptor(sel, SYSTEM_TYPE_TSS64_AVAILABLE)?;
        let high_dword_address =
            self.gdtr.base + (sel & SELECTOR_OFFSET_MASK) as u64 + DESC_HIGH_DWORD_OFFSET;
        // The write comes before the TR update so a faulting store leaves TR as it was.
        let high_dword = self.read_sys(high_dword_address, 4)?;
        self.write_sys(
            high_dword_address,
            4,
            high_dword | SYSTEM_TYPE_BUSY_BIT as u64,
        )?;
        seg.flags |= SYSTEM_TYPE_BUSY_BIT;
        self.tr = seg;
        Ok(())
    }

    /// Reads the 16-byte GDT descriptor for `sel` and checks it is a present system descriptor
    /// of `required_type`: #GP(sel) for a selector in the LDT, outside the GDT limit (the whole
    /// 16 bytes must fit) or of another type, then #NP(sel). Error codes drop the RPL bits.
    fn load_system_descriptor(&mut self, sel: u16, required_type: u32) -> ExecResult<Seg> {
        let error_code = (sel & !SELECTOR_RPL_MASK) as u32;
        let offset = (sel & SELECTOR_OFFSET_MASK) as u64;
        if sel & SELECTOR_TABLE_INDICATOR != 0
            || offset + SYSTEM_DESCRIPTOR_BYTES - 1 > self.gdtr.limit as u64
        {
            return Err(Exception::general_protection(error_code));
        }
        let (lo, hi) = self.read_desc(sel)?;
        let (base, limit) = Self::system_descriptor_base_limit(lo, hi);
        let seg = Seg {
            sel,
            base,
            limit,
            flags: ((lo >> 32) as u32) & SEG_DESCRIPTOR_FLAGS_MASK,
        };
        let descriptor_type = (seg.flags >> DESC_TYPE_SHIFT) & DESC_TYPE_MASK;
        if seg.flags & SEG_NON_SYSTEM != 0 || descriptor_type != required_type {
            return Err(Exception::general_protection(error_code));
        }
        if !seg.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(error_code),
            ));
        }
        Ok(seg)
    }

    /// VERR/VERW: whether the segment named by `sel` can be read (or written) at the current
    /// privilege. Never faults on a bad selector (null, outside the table, system descriptor);
    /// only the descriptor fetch itself can fault.
    fn segment_accessible(&mut self, sel: u16, for_write: bool) -> ExecResult<bool> {
        if sel & !SELECTOR_RPL_MASK == 0 {
            return Ok(false);
        }
        let table_limit = if sel & SELECTOR_TABLE_INDICATOR != 0 {
            self.ldtr.limit
        } else {
            self.gdtr.limit as u32
        };
        if (sel & SELECTOR_OFFSET_MASK) as u64 + DESCRIPTOR_BYTES - 1 > table_limit as u64 {
            return Ok(false);
        }
        let (lo, _) = self.read_desc(sel)?;
        let flags = ((lo >> 32) as u32) & SEG_DESCRIPTOR_FLAGS_MASK;
        if flags & SEG_NON_SYSTEM == 0 {
            return Ok(false);
        }
        let descriptor_dpl = ((flags >> SEG_DPL_SHIFT) & SEG_DPL_MASK) as u8;
        let requestor_rpl = (sel & SELECTOR_RPL_MASK) as u8;
        let privileged_enough = descriptor_dpl >= self.cpl && descriptor_dpl >= requestor_rpl;
        let executable = flags & SEG_TYPE_EXECUTABLE != 0;
        // Bit 9 is W for data and R for code.
        let read_write = flags & SEG_TYPE_READ_WRITE != 0;
        Ok(if for_write {
            !executable && read_write && privileged_enough
        } else if executable {
            // Conforming code is readable from any privilege level.
            read_write && (flags & SEG_TYPE_CONFORMING != 0 || privileged_enough)
        } else {
            privileged_enough
        })
    }

    /// 0F 00 (group 6: sldt, str, lldt, ltr, verr, verw) and 0F 01 (group 7: sgdt, sidt,
    /// lgdt, lidt, smsw, lmsw, invlpg, swapgs), selected by ModRM.reg.
    pub fn sys_group(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        const OP_GROUP6: u16 = 0x100;
        const MODRM_SWAPGS: u8 = 0xf8;
        let reg = i.reg & 7;
        if i.op == OP_GROUP6 {
            let loc = self.modrm_location(i, next);
            // a register destination takes the operand size, memory is always 16 bits
            let size = if i.md == 3 { i.opsize } else { 2 };
            match reg {
                // sldt
                0 => {
                    let v = self.ldtr.sel as u64;
                    self.store_operand(loc, size, i, v)?;
                }
                // str
                1 => {
                    let v = self.tr.sel as u64;
                    self.store_operand(loc, size, i, v)?;
                }
                2 | 3 => {
                    self.need_cpl0()?;
                    let sel = self.load_operand(loc, 2, i)? as u16;
                    if reg == 2 {
                        self.lldt(sel)?;
                    } else {
                        self.ltr(sel)?;
                    }
                }
                4 | 5 => {
                    let sel = self.load_operand(loc, 2, i)? as u16;
                    let accessible = self.segment_accessible(sel, reg == 5)?;
                    let rflags = self.eflags_full();
                    self.set_eflags_arith((rflags & !ZF) | if accessible { ZF } else { 0 });
                }
                _ => return Err(Exception::invalid_opcode()),
            }
            self.rip = next;
            return Ok(());
        }
        // register forms of group 7 are separate instructions, except smsw (/4) and lmsw (/6)
        if i.md == 3 && reg != 4 && reg != 6 {
            match i.modrm {
                MODRM_SWAPGS => {
                    self.need_cpl0()?;
                    core::mem::swap(&mut self.seg[SEG_GS].base, &mut self.kgsbase);
                }
                _ => return Err(self.unimplemented_opcode(i)),
            }
            self.rip = next;
            return Ok(());
        }
        let loc = self.modrm_location(i, next);
        match reg {
            // sgdt, sidt: 2-byte limit then 8-byte base
            0 | 1 => {
                let addr = match loc {
                    Loc::Mem(addr) => addr,
                    _ => return Err(Exception::invalid_opcode()),
                };
                let table = if reg == 0 { self.gdtr } else { self.idtr };
                self.write(addr, 2, table.limit as u64)?;
                self.write(addr + 2, 8, table.base)?;
            }
            // lgdt, lidt
            2 | 3 => {
                self.need_cpl0()?;
                let addr = match loc {
                    Loc::Mem(addr) => addr,
                    _ => return Err(Exception::invalid_opcode()),
                };
                let limit = self.read(addr, 2)? as u16;
                let base = self.read(addr + 2, 8)?;
                let table = DescriptorTable { base, limit };
                if reg == 2 {
                    self.gdtr = table;
                } else {
                    self.idtr = table;
                }
            }
            // smsw: the store truncates to the operand size, so a register destination of 32 or 64
            // bits receives CR0 itself
            4 => {
                let size = if i.md == 3 { i.opsize } else { 2 };
                let v = self.cr0;
                self.store_operand(loc, size, i, v)?;
            }
            // lmsw loads MP, EM, TS and can set PE but never clear it
            6 => {
                const LMSW_MASK: u64 = CR0_MP | CR0_EM | CR0_TS;
                self.need_cpl0()?;
                let v = self.load_operand(loc, 2, i)?;
                self.cr0 =
                    (self.cr0 & !LMSW_MASK) | (v & LMSW_MASK) | (self.cr0 & CR0_PE) | (v & CR0_PE);
            }
            // invlpg
            7 => {
                self.need_cpl0()?;
                if let Loc::Mem(addr) = loc {
                    self.invlpg(addr);
                } else {
                    return Err(Exception::invalid_opcode());
                }
            }
            _ => return Err(Exception::invalid_opcode()),
        }
        self.rip = next;
        Ok(())
    }

    /// 0F AE: fxsave, fxrstor, ldmxcsr, stmxcsr, clflush (memory forms) and lfence/mfence/sfence
    /// (register forms /5../7, no-ops). xsave/xrstor are not advertised and fault with #UD.
    pub fn group_ae(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let reg = i.reg & 7;
        if i.md == 3 {
            if i.rep != 0 || !(5..=7).contains(&reg) {
                return Err(Exception::invalid_opcode());
            }
        } else {
            // ldmxcsr/stmxcsr obey the SSE enable rules (#UD for CR0.EM or CR4.OSFXSR clear, then
            // #NM for CR0.TS) before touching memory; fxsave/fxrstor have their own checks.
            if matches!(reg, 2 | 3) {
                self.sse_gate()?;
            }
            let addr = self.linear_address(i, self.effective_address(i, next));
            match reg {
                0 => self.fxsave(addr)?,
                1 => self.fxrstor(addr)?,
                2 => {
                    let v = self.read(addr, 4)? as u32;
                    if v & !MXCSR_VALID_BITS != 0 {
                        return Err(Exception::general_protection(0));
                    }
                    self.mxcsr = v;
                }
                3 => {
                    let v = self.mxcsr as u64;
                    self.write(addr, 4, v)?;
                }
                // clflush: only the access check (and its fault) is performed
                7 => {
                    self.read(addr, 1)?;
                }
                _ => return Err(Exception::invalid_opcode()),
            }
        }
        self.rip = next;
        Ok(())
    }
}

#[cfg(test)]
#[path = "sys_tests.rs"]
mod sys_tests;
