//! VEX instructions that are not a plain lane-wise extension of an SSE instruction.

use super::decode::Insn;
use super::sse::Prefix;
use super::{Cpu, ExecResult};

impl Cpu {
    /// `Ok(false)` means the opcode is not implemented (the caller raises the logged #UD).
    pub(super) fn exec_vex_special(
        &mut self,
        _i: &Insn,
        _next: u64,
        _prefix: Prefix,
    ) -> ExecResult<bool> {
        Ok(false)
    }
}
