use super::flags::*;
use super::{
    Cpu, ExecResult, GPR_RSP, SEG_CS, SEG_DESCRIPTOR_FLAGS_MASK, SEG_DPL_SHIFT,
    SEG_FLAGS_FLAT_DATA, SEG_GRANULARITY, SEG_LIMIT_4GIB, SEG_SS, Seg,
};
use crate::machine::ExitReason;

pub(crate) const VECTOR_DIVIDE_ERROR: u8 = 0;
pub(crate) const VECTOR_INVALID_OPCODE: u8 = 6;
pub(crate) const VECTOR_DEVICE_NOT_AVAILABLE: u8 = 7;
pub(crate) const VECTOR_DOUBLE_FAULT: u8 = 8;
pub(crate) const VECTOR_INVALID_TSS: u8 = 10;
pub(crate) const VECTOR_SEGMENT_NOT_PRESENT: u8 = 11;
pub(crate) const VECTOR_STACK_FAULT: u8 = 12;
pub(crate) const VECTOR_GENERAL_PROTECTION: u8 = 13;
pub(crate) const VECTOR_PAGE_FAULT: u8 = 14;

/// Long-mode IDT entries are 16 bytes.
const IDT_ENTRY_BYTES: u64 = 16;
/// Error codes for faults caused by an IDT entry are `vector * 8` with this bit (IDT) set.
const IDT_ERROR_CODE_FLAG: u32 = 2;
/// Clears the RPL bits of a selector, leaving index and table indicator, which is the form that
/// segment-related error codes use.
const SELECTOR_ERROR_CODE_MASK: u16 = 0xfffc;
/// Selector bits 3-15 pre-multiplied by 8: the byte offset of the descriptor in its table.
const SELECTOR_OFFSET_MASK: u16 = 0xfff8;
/// Selector table indicator: set selects the LDT, clear the GDT.
const SELECTOR_TI_LDT: u16 = 1 << 2;

const GATE_PRESENT: u8 = 1 << 7;
const GATE_DPL_SHIFT: u32 = 5;
const GATE_TYPE_MASK: u8 = 0xf;
const GATE_TYPE_INTERRUPT: u8 = 0xe;
const GATE_TYPE_TRAP: u8 = 0xf;
const GATE_IST_MASK: u64 = 7;
/// The S bit of a descriptor (bit 12 of its upper dword) clear means a system descriptor,
/// which is 16 bytes wide in long mode.
const DESCRIPTOR_S_BIT: u64 = 1 << 44;

/// Offsets into the 64-bit TSS: RSP0 (then RSP1, RSP2) and IST1 (then IST2..7), 8 bytes each.
const TSS64_RSP0_OFFSET: u64 = 4;
const TSS64_IST1_OFFSET: u64 = 36;
const TSS64_SLOT_BYTES: u64 = 8;
/// Interrupt delivery in 64-bit mode aligns the new stack pointer down to 16 bytes.
const STACK_ALIGN_MASK: u64 = !0xf;
/// Upper bound on delivery attempts, so a cycle of benign faults raised by delivery itself
/// ends as a triple fault instead of looping.
const MAX_DELIVERY_ATTEMPTS: usize = 4;

/// Why an instruction or memory access stopped executing normally.
#[derive(Debug)]
pub(crate) enum Exception {
    /// A guest-visible exception, delivered through the IDT.
    Fault {
        vector: u8,
        /// Error code to push; `None` for vectors that have none.
        error_code: Option<u32>,
        /// Raised by `int n`/`int3` rather than by a fault, so delivery also checks the gate's
        /// DPL against CPL.
        software: bool,
    },
    /// Not a guest exception: ends the run loop with this reason.
    Exit(ExitReason),
}

impl Exception {
    pub fn fault(vector: u8, error_code: Option<u32>) -> Exception {
        Exception::Fault {
            vector,
            error_code,
            software: false,
        }
    }
    pub fn general_protection(code: u32) -> Exception {
        Exception::fault(VECTOR_GENERAL_PROTECTION, Some(code))
    }
    pub fn invalid_opcode() -> Exception {
        Exception::fault(VECTOR_INVALID_OPCODE, None)
    }
    pub fn divide_error() -> Exception {
        Exception::fault(VECTOR_DIVIDE_ERROR, None)
    }
    pub fn device_not_available() -> Exception {
        Exception::fault(VECTOR_DEVICE_NOT_AVAILABLE, None)
    }
}

/// The SDM's "contributory" class: #DE, #TS, #NP, #SS, #GP. #PF is classified separately.
fn contributory(vector: u8) -> bool {
    matches!(
        vector,
        VECTOR_DIVIDE_ERROR
            | VECTOR_INVALID_TSS
            | VECTOR_SEGMENT_NOT_PRESENT
            | VECTOR_STACK_FAULT
            | VECTOR_GENERAL_PROTECTION
    )
}

fn idt_error_code(vector: u8) -> u32 {
    vector as u32 * 8 + IDT_ERROR_CODE_FLAG
}

impl Cpu {
    /// Delivers an exception or interrupt through the IDT. If delivery itself faults, the new
    /// fault replaces the old one, or becomes #DF when the pair is contributory; a fault while
    /// delivering #DF is a triple fault. Returns `Some` only when the run must end.
    ///
    /// A #PF raised while delivering a contributory exception escalates to #DF here too,
    /// whereas the SDM's table (and QEMU's `check_exception`) would handle that pair serially.
    pub fn deliver_event(
        &mut self,
        vector: u8,
        error_code: Option<u32>,
        software: bool,
    ) -> Option<ExitReason> {
        let (mut vector, mut error_code, mut software) = (vector, error_code, software);
        for _ in 0..MAX_DELIVERY_ATTEMPTS {
            match self.deliver(vector, error_code, software) {
                Ok(()) => return None,
                Err(Exception::Exit(reason)) => return Some(reason),
                Err(Exception::Fault {
                    vector: nested_vector,
                    error_code: nested_error,
                    ..
                }) => {
                    if vector == VECTOR_DOUBLE_FAULT {
                        return Some(ExitReason::TripleFault);
                    }
                    software = false;
                    let nested_is_page_fault = nested_vector == VECTOR_PAGE_FAULT;
                    if (contributory(vector)
                        && (contributory(nested_vector) || nested_is_page_fault))
                        || (vector == VECTOR_PAGE_FAULT
                            && (contributory(nested_vector) || nested_is_page_fault))
                    {
                        vector = VECTOR_DOUBLE_FAULT;
                        error_code = Some(0);
                    } else {
                        vector = nested_vector;
                        error_code = nested_error;
                    }
                }
            }
        }
        Some(ExitReason::TripleFault)
    }

    /// 64-bit-mode interrupt/trap gate delivery. Nothing is committed (RSP, CS, SS, RIP, flags)
    /// until after the last push, so a fault while pushing leaves the CPU state as it was and
    /// the nested fault is delivered instead.
    fn deliver(&mut self, vector: u8, error_code: Option<u32>, software: bool) -> ExecResult<()> {
        let entry_offset = vector as u64 * IDT_ENTRY_BYTES;
        if entry_offset + (IDT_ENTRY_BYTES - 1) > self.idtr.limit as u64 {
            return Err(Exception::general_protection(idt_error_code(vector)));
        }
        let entry_address = self.idtr.base + entry_offset;
        let lo = self.read_sys(entry_address, 8)?;
        let hi = self.read_sys(entry_address + 8, 8)?;
        // lo: offset[15:0], selector, IST in bits 32-34, attributes byte, offset[31:16];
        // hi: offset[63:32].
        let selector = (lo >> 16) as u16;
        let ist = ((lo >> 32) & GATE_IST_MASK) as usize;
        let attributes = (lo >> 40) as u8;
        let gate_dpl = (attributes >> GATE_DPL_SHIFT) & 3;
        let gate_type = attributes & GATE_TYPE_MASK;
        if attributes & GATE_PRESENT == 0 {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(idt_error_code(vector)),
            ));
        }
        if gate_type != GATE_TYPE_INTERRUPT && gate_type != GATE_TYPE_TRAP {
            return Err(Exception::general_protection(idt_error_code(vector)));
        }
        if software && gate_dpl < self.cpl {
            return Err(Exception::general_protection(idt_error_code(vector)));
        }
        let target = (lo & 0xffff) | ((lo >> 32) & 0xffff_0000) | (hi << 32);
        let new_cs = self.load_desc_seg(selector, true)?;
        if !new_cs.is_code() || !new_cs.long() {
            return Err(Exception::general_protection(
                (selector & SELECTOR_ERROR_CODE_MASK) as u32,
            ));
        }
        let new_cpl = new_cs.dpl();
        let old_ss = self.seg[SEG_SS].sel as u64;
        let old_rsp = self.gpr[GPR_RSP];
        let rflags = self.eflags_full();
        let mut stack_pointer = if ist != 0 {
            self.tss_read(TSS64_IST1_OFFSET + (ist as u64 - 1) * TSS64_SLOT_BYTES)?
        } else if new_cpl < self.cpl {
            self.tss_read(TSS64_RSP0_OFFSET + new_cpl as u64 * TSS64_SLOT_BYTES)?
        } else {
            old_rsp
        };
        stack_pointer &= STACK_ALIGN_MASK;
        let old_cs = self.seg[SEG_CS].sel as u64;
        let mut push = |cpu: &mut Cpu, value: u64| -> ExecResult<()> {
            stack_pointer = stack_pointer.wrapping_sub(8);
            cpu.write_sys(stack_pointer, 8, value)
        };
        // 64-bit mode always pushes all five words, with or without a privilege change.
        push(self, old_ss)?;
        push(self, old_rsp)?;
        push(self, rflags)?;
        push(self, old_cs)?;
        let rip = self.rip;
        push(self, rip)?;
        if let Some(error_code) = error_code {
            push(self, error_code as u64)?;
        }
        self.gpr[GPR_RSP] = stack_pointer;
        if new_cpl != self.cpl {
            // Hardware loads a null SS with RPL = new CPL; modeled as a flat data segment of
            // that DPL.
            self.seg[SEG_SS] = Seg {
                sel: new_cpl as u16,
                base: 0,
                limit: SEG_LIMIT_4GIB,
                flags: SEG_FLAGS_FLAT_DATA | ((new_cpl as u32) << SEG_DPL_SHIFT),
            };
        }
        self.seg[SEG_CS] = new_cs;
        self.set_cpl(new_cpl);
        self.rip = target;
        // Resolve lazy flags first: the masking below edits `eflags` directly.
        self.materialize_lazy_flags();
        self.eflags &= !(TF | NT | RF | VM | AC);
        // Only interrupt gates mask interrupts; trap gates leave IF alone.
        if gate_type == GATE_TYPE_INTERRUPT {
            self.eflags &= !IF;
        }
        Ok(())
    }

    /// Reads a qword from the 64-bit TSS; an offset past the TR limit is #TS.
    fn tss_read(&mut self, offset: u64) -> ExecResult<u64> {
        if offset + 7 > self.tr.limit as u64 {
            return Err(Exception::fault(
                VECTOR_INVALID_TSS,
                Some((self.tr.sel & SELECTOR_ERROR_CODE_MASK) as u32),
            ));
        }
        self.read_sys(self.tr.base + offset, 8)
    }

    /// Returns the descriptor's low qword and, for system descriptors (16 bytes in long mode)
    /// whose second half is inside the table limit, the high qword (otherwise 0).
    pub fn read_desc(&mut self, selector: u16) -> ExecResult<(u64, u64)> {
        let offset = (selector & SELECTOR_OFFSET_MASK) as u64;
        let (table_base, table_limit) = if selector & SELECTOR_TI_LDT != 0 {
            (self.ldtr.base, self.ldtr.limit as u64)
        } else {
            (self.gdtr.base, self.gdtr.limit as u64)
        };
        if offset + 7 > table_limit {
            return Err(Exception::general_protection(
                (selector & SELECTOR_ERROR_CODE_MASK) as u32,
            ));
        }
        let lo = self.read_sys(table_base + offset, 8)?;
        let hi = if lo & DESCRIPTOR_S_BIT == 0 && offset + 15 <= table_limit {
            self.read_sys(table_base + offset + 8, 8)?
        } else {
            0
        };
        Ok((lo, hi))
    }

    /// Builds a `Seg` from an 8-byte descriptor. Does no type or privilege checks (callers
    /// do), only the present check when asked, which raises #NP.
    pub fn load_desc_seg(&mut self, selector: u16, require_present: bool) -> ExecResult<Seg> {
        let (lo, _) = self.read_desc(selector)?;
        let dword0 = lo as u32;
        let dword1 = (lo >> 32) as u32;
        let base = ((dword0 >> 16) | ((dword1 & 0xff) << 16) | (dword1 & 0xff00_0000)) as u64;
        let mut limit = (dword0 & 0xffff) | (dword1 & 0xf_0000);
        if dword1 & SEG_GRANULARITY != 0 {
            limit = (limit << 12) | 0xfff;
        }
        let seg = Seg {
            sel: selector,
            base,
            limit,
            flags: dword1 & SEG_DESCRIPTOR_FLAGS_MASK,
        };
        if require_present && !seg.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some((selector & SELECTOR_ERROR_CODE_MASK) as u32),
            ));
        }
        Ok(seg)
    }
}
