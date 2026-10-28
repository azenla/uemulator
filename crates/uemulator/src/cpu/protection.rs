//! Smaller protection and power features that CPUID advertises together with their CR4 bits:
//! UMIP, SMEP/SMAP (clac/stac; the checks live in the page walk in `mem.rs`), PCID/INVPCID, and
//! MONITOR/MWAIT.

use super::decode::Insn;
use super::exception::Exception;
use super::exec::Loc;
use super::sys::{CR4_UMIP, RAX, RCX};
use super::{Cpu, ExecResult};

/// INVPCID descriptor: PCID in bits 11:0 (the rest of the first quadword is reserved) followed by
/// a linear address.
const INVPCID_DESCRIPTOR_RESERVED: u64 = !0xfff;
const INVPCID_TYPE_MAX: u64 = 3;
const INVPCID_TYPE_INDIVIDUAL_ADDRESS: u64 = 0;

/// MWAIT's ECX extensions: bit 0 = treat masked interrupts as break events. Other bits are
/// reserved and #GP.
const MWAIT_ECX_INTERRUPT_BREAK: u64 = 1;
/// MONITOR's ECX/EDX (extensions, hints) must be zero here: no extensions are enumerated.
const MONITOR_NO_EXTENSIONS: u64 = 0;

impl Cpu {
    /// With CR4.UMIP set, the descriptor-table and machine-status reads (sgdt, sidt, sldt, str,
    /// smsw) are #GP(0) outside CPL 0, so user code cannot learn kernel addresses from them.
    pub(super) fn umip_check(&self) -> ExecResult<()> {
        if self.cr4 & CR4_UMIP != 0 && self.cpl != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    /// 66 0F 38 82 (invpcid r64, m128). The soft TLB is not tagged with PCIDs, so every type
    /// flushes it entirely, which is always a legal over-approximation of the narrower ones;
    /// operand validation is as specified so a guest probing the instruction sees the faults.
    pub(super) fn invpcid(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        if i.md == 3 || !i.p66 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()?;
        let kind = self.gpr[i.reg as usize];
        let Loc::Mem(address) = self.modrm_location(i, next)? else {
            return Err(Exception::invalid_opcode());
        };
        if kind > INVPCID_TYPE_MAX {
            return Err(Exception::general_protection(0));
        }
        let low = self.read(address, 8)?;
        let linear = self.read(address + 8, 8)?;
        if low & INVPCID_DESCRIPTOR_RESERVED != 0 {
            return Err(Exception::general_protection(0));
        }
        if kind == INVPCID_TYPE_INDIVIDUAL_ADDRESS && !super::sys::is_canonical(linear) {
            return Err(Exception::general_protection(0));
        }
        self.flush_tlb();
        self.rip = next;
        Ok(())
    }

    /// 0F 01 C8. CPL 0 only. Nothing else can write guest RAM while the CPU waits except device
    /// DMA, and the armed range is never consulted, so MWAIT below only wakes on interrupts.
    pub(super) fn monitor(&mut self) -> ExecResult<()> {
        self.need_cpl0()?;
        if self.gpr[RCX] != MONITOR_NO_EXTENSIONS
            || self.gpr[super::sys::RDX] != MONITOR_NO_EXTENSIONS
        {
            return Err(Exception::general_protection(0));
        }
        self.monitor_address = self.gpr[RAX];
        self.monitor_armed = true;
        Ok(())
    }

    /// 0F 01 C9. CPL 0 only, #GP on reserved ECX bits. Waits like `hlt` while interrupts are
    /// enabled; with IF=0 and no break-on-masked-interrupt it returns immediately, because a
    /// `hlt`-style stop there would be the machine-shutdown case in the run loop and MWAIT may
    /// always return early.
    pub(super) fn mwait(&mut self) -> ExecResult<()> {
        self.need_cpl0()?;
        if self.gpr[RCX] & !MWAIT_ECX_INTERRUPT_BREAK != 0 {
            return Err(Exception::general_protection(0));
        }
        if !self.monitor_armed {
            // Without a preceding monitor, mwait is a no-op in hardware terms: it just returns.
            return Ok(());
        }
        self.monitor_armed = false;
        if self.eflags_full() & super::flags::IF != 0 {
            self.halted = true;
        }
        Ok(())
    }
}
