//! Randomized differential tests. A fixed-seed generator assembles short straight-line programs
//! (with forward branches and small counted loops) from templates covering the integer
//! instructions the JIT inlines and the ones it hands to the interpreter, in every operand size,
//! with register, immediate and memory operands, and checks the JIT against the interpreter.
//!
//! Register roles keep every program well-defined: `DATA_REGISTERS` are free to be overwritten,
//! `BASE_REGISTERS` and `INDEX_REGISTERS` only ever hold addresses into seeded scratch memory
//! and are never written, rsp is only touched by push/pop pairs. The programs never fault and
//! never divide.

use super::diff_tests::*;
use crate::cpu::flags::ARITH;
use crate::testkit::*;

/// Registers a program may overwrite (rax, rcx, rdx, rbx, r8, r9, r10, r11, r15's role is the loop
/// counter and is handled separately).
const DATA_REGISTERS: [u8; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
/// Never written; hold addresses inside the seeded scratch area.
const BASE_REGISTERS: [u8; 4] = [5, 7, 12, 13];
/// Never written; hold small values so `base + index * scale` stays in the scratch area.
const INDEX_REGISTERS: [u8; 2] = [6, 14];
const LOOP_COUNTER: u8 = 15;
const RSP: u8 = 4;

const BASE_OFFSETS: [u64; 4] = [0x1000, 0x1800, 0x2000, 0x2800];
const MAX_INDEX_VALUE: u64 = 0x40;
const MAX_DISPLACEMENT: i32 = 0x300;

const GROUP1_NAMES: [&str; 8] = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"];
const SHIFT_NAMES: [&str; 8] = ["rol", "ror", "rcl", "rcr", "shl", "shr", "shl", "sar"];
const CONDITION_NAMES: [&str; 16] = [
    "o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g",
];

/// Values that tend to expose carry, overflow and width bugs.
const INTERESTING_VALUES: [u64; 14] = [
    0,
    1,
    2,
    0x7f,
    0x80,
    0xff,
    0x7fff,
    0x8000,
    0xffff,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
];

#[derive(Clone, Copy)]
struct MemoryOperand {
    base: u8,
    index: Option<(u8, u8)>,
    displacement: i32,
}

#[derive(Clone, Copy)]
enum Operand {
    Register(u8),
    Memory(MemoryOperand),
}

/// One generated instruction: its bytes, and Intel syntax for checking the encoder against an
/// assembler.
#[derive(Clone)]
pub(super) struct Generated {
    pub bytes: Vec<u8>,
    pub text: String,
}

/// Operand widths in bytes.
#[derive(Clone, Copy, PartialEq)]
enum Width {
    Byte,
    Word,
    Dword,
    Qword,
}

impl Width {
    fn pointer_name(self) -> &'static str {
        match self {
            Width::Byte => "byte ptr",
            Width::Word => "word ptr",
            Width::Dword => "dword ptr",
            Width::Qword => "qword ptr",
        }
    }
}

const WIDE_WIDTHS: [Width; 3] = [Width::Word, Width::Dword, Width::Qword];
const ALL_WIDTHS: [Width; 4] = [Width::Byte, Width::Word, Width::Dword, Width::Qword];

fn register_name(register: u8, width: Width, legacy_byte: bool) -> String {
    const QWORD: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    const DWORD: [&str; 16] = [
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d",
        "r12d", "r13d", "r14d", "r15d",
    ];
    const WORD: [&str; 16] = [
        "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w",
        "r13w", "r14w", "r15w",
    ];
    const BYTE_WITH_REX: [&str; 16] = [
        "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
        "r13b", "r14b", "r15b",
    ];
    const BYTE_LEGACY: [&str; 8] = ["al", "cl", "dl", "bl", "ah", "ch", "dh", "bh"];
    let index = register as usize;
    match width {
        Width::Qword => QWORD[index],
        Width::Dword => DWORD[index],
        Width::Word => WORD[index],
        Width::Byte if legacy_byte => BYTE_LEGACY[index],
        Width::Byte => BYTE_WITH_REX[index],
    }
    .to_string()
}

fn memory_text(memory: &MemoryOperand) -> String {
    let mut text = format!("[{}", register_name(memory.base, Width::Qword, false));
    if let Some((index, scale)) = memory.index {
        text += &format!(
            " + {}*{}",
            register_name(index, Width::Qword, false),
            1 << scale
        );
    }
    if memory.displacement != 0 {
        text += &format!(" + {}", memory.displacement);
    }
    text + "]"
}

/// ModRM, SIB and displacement bytes for `reg` and `operand`, written the way a compiler would.
fn modrm_bytes(reg: u8, operand: Operand) -> Vec<u8> {
    match operand {
        Operand::Register(register) => vec![0xc0 | (reg & 7) << 3 | (register & 7)],
        Operand::Memory(memory) => {
            let base = memory.base & 7;
            let needs_sib = memory.index.is_some() || base == 4;
            let (mode, displacement): (u8, Vec<u8>) = if memory.displacement == 0 && base != 5 {
                (0, vec![])
            } else if i8::try_from(memory.displacement).is_ok() {
                (1, vec![memory.displacement as i8 as u8])
            } else {
                (2, memory.displacement.to_le_bytes().to_vec())
            };
            let mut bytes = vec![mode << 6 | (reg & 7) << 3 | if needs_sib { 4 } else { base }];
            if needs_sib {
                let (index, scale) = memory
                    .index
                    .map_or((4, 0), |(index, scale)| (index & 7, scale));
                bytes.push(scale << 6 | index << 3 | base);
            }
            bytes.extend(displacement);
            bytes
        }
    }
}

struct Builder {
    rng: Rng,
    program: Vec<Generated>,
}

impl Builder {
    fn data_register(&mut self) -> u8 {
        *self.rng.pick(&DATA_REGISTERS)
    }

    /// Any register that may be read: data, base, index (never the loop counter's role though it
    /// is harmless to read).
    fn source_register(&mut self) -> u8 {
        match self.rng.below(6) {
            0 => *self.rng.pick(&BASE_REGISTERS),
            1 => *self.rng.pick(&INDEX_REGISTERS),
            _ => self.data_register(),
        }
    }

    fn memory_operand(&mut self) -> MemoryOperand {
        let base = *self.rng.pick(&BASE_REGISTERS);
        let index = (self.rng.below(2) == 0)
            .then(|| (*self.rng.pick(&INDEX_REGISTERS), self.rng.below(4) as u8));
        let displacement = match self.rng.below(4) {
            0 => 0,
            1 => self.rng.below(0x70) as i32 - 0x38,
            _ => self.rng.below(2 * MAX_DISPLACEMENT as u64) as i32 - MAX_DISPLACEMENT,
        };
        MemoryOperand {
            base,
            index,
            displacement,
        }
    }

    fn immediate(&mut self, width: Width) -> i64 {
        let value = if self.rng.below(2) == 0 {
            *self.rng.pick(&INTERESTING_VALUES)
        } else {
            self.rng.next()
        };
        match width {
            Width::Byte => value as i8 as i64,
            Width::Word => value as i16 as i64,
            _ => value as i32 as i64,
        }
    }

    /// Assembles one instruction. `legacy_byte` means a byte register field of 4..=7 names
    /// ah/ch/dh/bh, which requires that no REX prefix is emitted at all.
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        width: Width,
        opcode: &[u8],
        reg: u8,
        operand: Operand,
        immediate: &[u8],
        legacy_byte: bool,
        text: String,
    ) {
        let mut bytes = Vec::new();
        if width == Width::Word {
            bytes.push(0x66);
        }
        let (index_extension, base_extension) = match operand {
            Operand::Register(register) => (0, register >> 3),
            Operand::Memory(memory) => (
                memory.index.map_or(0, |(index, _)| index >> 3),
                memory.base >> 3,
            ),
        };
        let mut rex = 0x40
            | ((width == Width::Qword) as u8) << 3
            | (reg >> 3) << 2
            | index_extension << 1
            | base_extension;
        // Byte operations on registers 4..=7 mean spl/bpl/sil/dil only with a REX prefix.
        let byte_needs_rex = width == Width::Byte
            && !legacy_byte
            && (matches!(operand, Operand::Register(4..=7)) || (4..=7).contains(&reg));
        if byte_needs_rex {
            rex |= 0x40;
        }
        if rex != 0x40 || byte_needs_rex {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.extend(modrm_bytes(reg, operand));
        bytes.extend_from_slice(immediate);
        self.program.push(Generated { bytes, text });
    }

    fn operand_text(&self, operand: Operand, width: Width, legacy_byte: bool) -> String {
        match operand {
            Operand::Register(register) => register_name(register, width, legacy_byte),
            Operand::Memory(memory) => format!("{} {}", width.pointer_name(), memory_text(&memory)),
        }
    }

    /// Picks the register or memory operand of an instruction and tells whether byte registers
    /// are in legacy (no REX) mode. Memory in legacy mode must use bases and indexes below 8.
    fn choose_operands(
        &mut self,
        width: Width,
        destination: bool,
        allow_memory: bool,
    ) -> (u8, Operand, bool) {
        let use_memory = allow_memory && self.rng.below(2) == 0;
        let legacy = width == Width::Byte && self.rng.below(3) == 0;
        let register = if legacy {
            // ah/ch/dh/bh are data; al/cl/dl/bl too
            *self.rng.pick(&[0u8, 1, 2, 3, 4, 5, 6, 7])
        } else if destination {
            self.data_register()
        } else {
            self.source_register()
        };
        let operand = if use_memory {
            let mut memory = self.memory_operand();
            if legacy {
                // No REX allowed: restrict to rbp/rdi bases and the rsi index.
                memory.base = *self.rng.pick(&[5u8, 7]);
                memory.index = memory.index.map(|(_, scale)| (6, scale));
            }
            Operand::Memory(memory)
        } else {
            let other = if legacy {
                *self.rng.pick(&[0u8, 1, 2, 3, 4, 5, 6, 7])
            } else if destination {
                self.data_register()
            } else {
                self.source_register()
            };
            Operand::Register(other)
        };
        (register, operand, legacy)
    }
}

impl Builder {
    fn alu_register_register(&mut self) {
        let operation = self.rng.below(8) as u8;
        let width = *self.rng.pick(&ALL_WIDTHS);
        let to_register = self.rng.below(2) == 0;
        let (reg, operand, legacy) = self.choose_operands(width, true, false);
        let opcode = (operation << 3)
            | if width == Width::Byte { 0 } else { 1 }
            | ((to_register as u8) << 1);
        // `op r/m, r` writes r/m (a data register here), `op r, r/m` writes r.
        let (first, second) = if to_register {
            (
                register_name(reg, width, legacy),
                self.operand_text(operand, width, legacy),
            )
        } else {
            (
                self.operand_text(operand, width, legacy),
                register_name(reg, width, legacy),
            )
        };
        let text = format!("{} {first}, {second}", GROUP1_NAMES[operation as usize]);
        self.emit(width, &[opcode], reg, operand, &[], legacy, text);
    }

    fn alu_register_memory(&mut self) {
        let operation = self.rng.below(8) as u8;
        let width = *self.rng.pick(&ALL_WIDTHS);
        let to_register = self.rng.below(2) == 0;
        let (reg, operand, legacy) = self.choose_operands_with_memory(width, to_register);
        let opcode = (operation << 3)
            | if width == Width::Byte { 0 } else { 1 }
            | ((to_register as u8) << 1);
        let (first, second) = if to_register {
            (
                register_name(reg, width, legacy),
                self.operand_text(operand, width, legacy),
            )
        } else {
            (
                self.operand_text(operand, width, legacy),
                register_name(reg, width, legacy),
            )
        };
        let text = format!("{} {first}, {second}", GROUP1_NAMES[operation as usize]);
        self.emit(width, &[opcode], reg, operand, &[], legacy, text);
    }

    /// A register and a memory operand; `register_is_destination` says which side is written, so
    /// a written register is always a data register.
    fn choose_operands_with_memory(
        &mut self,
        width: Width,
        register_is_destination: bool,
    ) -> (u8, Operand, bool) {
        let legacy = width == Width::Byte && self.rng.below(3) == 0;
        let register = if legacy {
            *self.rng.pick(&[0u8, 1, 2, 3, 4, 5, 6, 7])
        } else if register_is_destination {
            self.data_register()
        } else {
            self.source_register()
        };
        let mut memory = self.memory_operand();
        if legacy {
            memory.base = *self.rng.pick(&[5u8, 7]);
            memory.index = memory.index.map(|(_, scale)| (6, scale));
        }
        (register, Operand::Memory(memory), legacy)
    }

    fn alu_immediate(&mut self) {
        let operation = self.rng.below(8) as u8;
        let width = *self.rng.pick(&ALL_WIDTHS);
        let (_, operand, legacy) = self.choose_operands(width, true, true);
        // Memory destinations of every operation are fine; registers must be data registers,
        // which `choose_operands` guarantees for the destination side.
        let value = self.immediate(width);
        let (opcode, immediate): (u8, Vec<u8>) = match width {
            Width::Byte => (0x80, vec![value as u8]),
            _ if (-128..=127).contains(&value) && self.rng.below(2) == 0 => {
                (0x83, vec![value as u8])
            }
            Width::Word => (0x81, (value as i16).to_le_bytes().to_vec()),
            _ => (0x81, (value as i32).to_le_bytes().to_vec()),
        };
        let text = format!(
            "{} {}, {}",
            GROUP1_NAMES[operation as usize],
            self.operand_text(operand, width, legacy),
            value
        );
        self.emit(
            width,
            &[opcode],
            operation,
            operand,
            &immediate,
            legacy,
            text,
        );
    }

    fn test_instruction(&mut self) {
        let width = *self.rng.pick(&ALL_WIDTHS);
        let (reg, operand, legacy) = self.choose_operands(width, false, true);
        if self.rng.below(2) == 0 {
            let opcode = if width == Width::Byte { 0x84 } else { 0x85 };
            let text = format!(
                "test {}, {}",
                self.operand_text(operand, width, legacy),
                register_name(reg, width, legacy)
            );
            self.emit(width, &[opcode], reg, operand, &[], legacy, text);
        } else {
            let value = self.immediate(width);
            let (opcode, immediate): (u8, Vec<u8>) = match width {
                Width::Byte => (0xf6, vec![value as u8]),
                Width::Word => (0xf7, (value as i16).to_le_bytes().to_vec()),
                _ => (0xf7, (value as i32).to_le_bytes().to_vec()),
            };
            let text = format!(
                "test {}, {}",
                self.operand_text(operand, width, legacy),
                value
            );
            self.emit(width, &[opcode], 0, operand, &immediate, legacy, text);
        }
    }

    fn mov(&mut self) {
        let width = *self.rng.pick(&ALL_WIDTHS);
        match self.rng.below(5) {
            0 => {
                // mov r/m, r
                let (reg, operand, legacy) = self.choose_operands(width, true, true);
                let opcode = if width == Width::Byte { 0x88 } else { 0x89 };
                let text = format!(
                    "mov {}, {}",
                    self.operand_text(operand, width, legacy),
                    register_name(reg, width, legacy)
                );
                self.emit(width, &[opcode], reg, operand, &[], legacy, text);
            }
            1 => {
                // mov r, r/m
                let (reg, operand, legacy) = self.choose_operands(width, true, true);
                let opcode = if width == Width::Byte { 0x8a } else { 0x8b };
                let text = format!(
                    "mov {}, {}",
                    register_name(reg, width, legacy),
                    self.operand_text(operand, width, legacy)
                );
                self.emit(width, &[opcode], reg, operand, &[], legacy, text);
            }
            2 => {
                // mov r/m, imm
                let (_, operand, legacy) = self.choose_operands(width, true, true);
                let value = self.immediate(width);
                let (opcode, immediate): (u8, Vec<u8>) = match width {
                    Width::Byte => (0xc6, vec![value as u8]),
                    Width::Word => (0xc7, (value as i16).to_le_bytes().to_vec()),
                    _ => (0xc7, (value as i32).to_le_bytes().to_vec()),
                };
                let text = format!(
                    "mov {}, {}",
                    self.operand_text(operand, width, legacy),
                    value
                );
                self.emit(width, &[opcode], 0, operand, &immediate, legacy, text);
            }
            3 => {
                // mov r, imm (short forms; movabs for 64-bit with a full immediate)
                let register = self.data_register();
                let mut bytes = Vec::new();
                let text = match width {
                    Width::Byte => {
                        let value = self.immediate(width);
                        if register >= 8 {
                            bytes.push(0x41);
                        }
                        bytes.push(0xb0 | (register & 7));
                        bytes.push(value as u8);
                        format!("mov {}, {}", register_name(register, width, false), value)
                    }
                    Width::Word => {
                        let value = self.immediate(width);
                        bytes.push(0x66);
                        if register >= 8 {
                            bytes.push(0x41);
                        }
                        bytes.push(0xb8 | (register & 7));
                        bytes.extend((value as i16).to_le_bytes());
                        format!("mov {}, {}", register_name(register, width, false), value)
                    }
                    Width::Dword => {
                        let value = self.immediate(width);
                        if register >= 8 {
                            bytes.push(0x41);
                        }
                        bytes.push(0xb8 | (register & 7));
                        bytes.extend((value as i32).to_le_bytes());
                        format!(
                            "mov {}, {}",
                            register_name(register, width, false),
                            value as u32
                        )
                    }
                    Width::Qword => {
                        let value = if self.rng.below(2) == 0 {
                            *self.rng.pick(&INTERESTING_VALUES)
                        } else {
                            self.rng.next()
                        };
                        bytes.push(0x48 | (register >> 3));
                        bytes.push(0xb8 | (register & 7));
                        bytes.extend(value.to_le_bytes());
                        format!(
                            "movabs {}, {}",
                            register_name(register, width, false),
                            value as i64
                        )
                    }
                };
                self.program.push(Generated { bytes, text });
            }
            _ => {
                // mov r, r (partial-register merging for narrow widths)
                let (reg, operand, legacy) = self.choose_operands(width, true, false);
                let opcode = if width == Width::Byte { 0x8a } else { 0x8b };
                let text = format!(
                    "mov {}, {}",
                    register_name(reg, width, legacy),
                    self.operand_text(operand, width, legacy)
                );
                self.emit(width, &[opcode], reg, operand, &[], legacy, text);
            }
        }
    }

    fn lea(&mut self) {
        let width = *self.rng.pick(&WIDE_WIDTHS);
        let register = self.data_register();
        let memory = self.memory_operand();
        let text = format!(
            "lea {}, {}",
            register_name(register, width, false),
            memory_text(&memory)
        );
        self.emit(
            width,
            &[0x8d],
            register,
            Operand::Memory(memory),
            &[],
            false,
            text,
        );
    }

    fn unary(&mut self) {
        let width = *self.rng.pick(&ALL_WIDTHS);
        let (_, operand, legacy) = self.choose_operands(width, true, true);
        let (name, opcode, extension): (&str, u8, u8) = match self.rng.below(4) {
            0 => ("inc", 0xfe, 0),
            1 => ("dec", 0xfe, 1),
            2 => ("neg", 0xf6, 3),
            _ => ("not", 0xf6, 2),
        };
        let opcode = if width == Width::Byte {
            opcode
        } else {
            opcode | 1
        };
        let text = format!("{name} {}", self.operand_text(operand, width, legacy));
        self.emit(width, &[opcode], extension, operand, &[], legacy, text);
    }

    fn shift(&mut self) {
        let width = *self.rng.pick(&ALL_WIDTHS);
        let (_, operand, legacy) = self.choose_operands(width, true, true);
        let extension = self.rng.below(8) as u8;
        // /6 is an undocumented alias of shl; keep it out
        let extension = if extension == 6 { 4 } else { extension };
        let name = SHIFT_NAMES[extension as usize];
        let byte = (width == Width::Byte) as u8;
        match self.rng.below(3) {
            0 => {
                let text = format!("{name} {}, cl", self.operand_text(operand, width, legacy));
                self.emit(width, &[0xd3 - byte], extension, operand, &[], legacy, text);
            }
            1 => {
                let text = format!("{name} {}, 1", self.operand_text(operand, width, legacy));
                self.emit(width, &[0xd1 - byte], extension, operand, &[], legacy, text);
            }
            _ => {
                let count = *self
                    .rng
                    .pick(&[0u8, 1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64]);
                let text = format!(
                    "{name} {}, {count}",
                    self.operand_text(operand, width, legacy)
                );
                self.emit(
                    width,
                    &[0xc1 - byte],
                    extension,
                    operand,
                    &[count],
                    legacy,
                    text,
                );
            }
        }
    }

    fn extension_move(&mut self) {
        let signed = self.rng.below(2) == 0;
        let from_word = self.rng.below(2) == 0;
        let width = *self.rng.pick(&WIDE_WIDTHS);
        if from_word && width == Width::Word {
            return self.mov();
        }
        let source_width = if from_word { Width::Word } else { Width::Byte };
        let legacy = !from_word && self.rng.below(4) == 0 && width != Width::Qword;
        let register = self.data_register();
        let register = if legacy { register & 3 } else { register };
        let (operand, legacy) = if self.rng.below(2) == 0 {
            let mut memory = self.memory_operand();
            if legacy {
                memory.base = *self.rng.pick(&[5u8, 7]);
                memory.index = memory.index.map(|(_, scale)| (6, scale));
            }
            (Operand::Memory(memory), legacy)
        } else if legacy {
            (
                Operand::Register(*self.rng.pick(&[0u8, 1, 2, 3, 4, 5, 6, 7])),
                true,
            )
        } else if from_word {
            (Operand::Register(self.source_register()), false)
        } else {
            // A byte source numbered 4..=7 needs a REX prefix to mean spl..dil, and `emit` only
            // adds one for byte-sized operations, so use registers that are fine without.
            let register = *self.rng.pick(&[0u8, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14]);
            (Operand::Register(register), false)
        };
        let opcode = match (signed, from_word) {
            (false, false) => 0xb6,
            (false, true) => 0xb7,
            (true, false) => 0xbe,
            (true, true) => 0xbf,
        };
        let name = if signed { "movsx" } else { "movzx" };
        let text = format!(
            "{name} {}, {}",
            register_name(register, width, false),
            self.operand_text(operand, source_width, legacy)
        );
        self.emit(width, &[0x0f, opcode], register, operand, &[], legacy, text);
    }

    fn movsxd(&mut self) {
        let register = self.data_register();
        let operand = if self.rng.below(2) == 0 {
            Operand::Memory(self.memory_operand())
        } else {
            Operand::Register(self.source_register())
        };
        let text = format!(
            "movsxd {}, {}",
            register_name(register, Width::Qword, false),
            self.operand_text(operand, Width::Dword, false)
        );
        self.emit(Width::Qword, &[0x63], register, operand, &[], false, text);
    }

    fn setcc(&mut self) {
        let condition = self.rng.below(16) as u8;
        let (_, operand, legacy) = self.choose_operands(Width::Byte, true, true);
        let text = format!(
            "set{} {}",
            CONDITION_NAMES[condition as usize],
            self.operand_text(operand, Width::Byte, legacy)
        );
        self.emit(
            Width::Byte,
            &[0x0f, 0x90 | condition],
            0,
            operand,
            &[],
            legacy,
            text,
        );
    }

    fn cmovcc(&mut self) {
        let condition = self.rng.below(16) as u8;
        let width = *self.rng.pick(&WIDE_WIDTHS);
        let register = self.data_register();
        let operand = if self.rng.below(2) == 0 {
            Operand::Memory(self.memory_operand())
        } else {
            Operand::Register(self.source_register())
        };
        let text = format!(
            "cmov{} {}, {}",
            CONDITION_NAMES[condition as usize],
            register_name(register, width, false),
            self.operand_text(operand, width, false)
        );
        self.emit(
            width,
            &[0x0f, 0x40 | condition],
            register,
            operand,
            &[],
            false,
            text,
        );
    }

    fn multiply(&mut self) {
        let width = *self.rng.pick(&WIDE_WIDTHS);
        let register = self.data_register();
        let operand = if self.rng.below(2) == 0 {
            Operand::Memory(self.memory_operand())
        } else {
            Operand::Register(self.source_register())
        };
        match self.rng.below(3) {
            0 => {
                let text = format!(
                    "imul {}, {}",
                    register_name(register, width, false),
                    self.operand_text(operand, width, false)
                );
                self.emit(width, &[0x0f, 0xaf], register, operand, &[], false, text);
            }
            1 => {
                let value = self.immediate(width);
                let (opcode, immediate): (u8, Vec<u8>) = if (-128..=127).contains(&value) {
                    (0x6b, vec![value as u8])
                } else if width == Width::Word {
                    (0x69, (value as i16).to_le_bytes().to_vec())
                } else {
                    (0x69, (value as i32).to_le_bytes().to_vec())
                };
                let text = format!(
                    "imul {}, {}, {}",
                    register_name(register, width, false),
                    self.operand_text(operand, width, false),
                    value
                );
                self.emit(width, &[opcode], register, operand, &immediate, false, text);
            }
            _ => {
                // one-operand mul / imul write rdx:rax (both data registers)
                let extension = 4 + self.rng.below(2) as u8;
                let name = if extension == 4 { "mul" } else { "imul" };
                let text = format!("{name} {}", self.operand_text(operand, width, false));
                self.emit(width, &[0xf7], extension, operand, &[], false, text);
            }
        }
    }

    fn bit_operations(&mut self) {
        let width = *self.rng.pick(&WIDE_WIDTHS);
        let operation = self.rng.below(4) as u8;
        let names = ["bt", "bts", "btr", "btc"];
        if self.rng.below(2) == 0 {
            // reg, reg
            let register = self.source_register();
            let target = self.data_register();
            let opcode = [0xa3, 0xab, 0xb3, 0xbb][operation as usize];
            let text = format!(
                "{} {}, {}",
                names[operation as usize],
                register_name(target, width, false),
                register_name(register, width, false)
            );
            self.emit(
                width,
                &[0x0f, opcode],
                register,
                Operand::Register(target),
                &[],
                false,
                text,
            );
        } else {
            let operand = if self.rng.below(2) == 0 {
                Operand::Memory(self.memory_operand())
            } else {
                Operand::Register(self.data_register())
            };
            let bit = self.rng.below(64) as u8;
            let text = format!(
                "{} {}, {bit}",
                names[operation as usize],
                self.operand_text(operand, width, false)
            );
            self.emit(
                width,
                &[0x0f, 0xba],
                4 + operation,
                operand,
                &[bit],
                false,
                text,
            );
        }
    }

    fn bit_scan(&mut self) {
        let width = *self.rng.pick(&WIDE_WIDTHS);
        let register = self.data_register();
        let operand = if self.rng.below(2) == 0 {
            Operand::Memory(self.memory_operand())
        } else {
            Operand::Register(self.source_register())
        };
        let (name, opcode) = if self.rng.below(2) == 0 {
            ("bsf", 0xbc)
        } else {
            ("bsr", 0xbd)
        };
        let text = format!(
            "{name} {}, {}",
            register_name(register, width, false),
            self.operand_text(operand, width, false)
        );
        self.emit(width, &[0x0f, opcode], register, operand, &[], false, text);
    }

    fn exchange(&mut self) {
        let width = *self.rng.pick(&ALL_WIDTHS);
        let (reg, operand, legacy) = self.choose_operands(width, true, true);
        let byte = (width == Width::Byte) as u8;
        // xchg and xadd write both operands, so the register side must be data and a register
        // operand too; memory is fine.
        match self.rng.below(3) {
            0 => {
                let text = format!(
                    "xchg {}, {}",
                    self.operand_text(operand, width, legacy),
                    register_name(reg, width, legacy)
                );
                self.emit(width, &[0x87 - byte], reg, operand, &[], legacy, text);
            }
            1 => {
                let text = format!(
                    "xadd {}, {}",
                    self.operand_text(operand, width, legacy),
                    register_name(reg, width, legacy)
                );
                self.emit(width, &[0x0f, 0xc1 - byte], reg, operand, &[], legacy, text);
            }
            _ => {
                let text = format!(
                    "cmpxchg {}, {}",
                    self.operand_text(operand, width, legacy),
                    register_name(reg, width, legacy)
                );
                self.emit(width, &[0x0f, 0xb1 - byte], reg, operand, &[], legacy, text);
            }
        }
    }

    fn byte_swap(&mut self) {
        let register = self.data_register();
        let wide = self.rng.below(2) == 0;
        let mut bytes = Vec::new();
        if wide || register >= 8 {
            bytes.push(0x40 | (wide as u8) << 3 | (register >> 3));
        }
        bytes.extend([0x0f, 0xc8 | (register & 7)]);
        let width = if wide { Width::Qword } else { Width::Dword };
        self.program.push(Generated {
            text: format!("bswap {}", register_name(register, width, false)),
            bytes,
        });
    }

    fn push_pop_pair(&mut self) {
        let source = self.source_register();
        let destination = self.data_register();
        let mut push = Vec::new();
        if source >= 8 {
            push.push(0x41);
        }
        push.push(0x50 | (source & 7));
        let mut pop = Vec::new();
        if destination >= 8 {
            pop.push(0x41);
        }
        pop.push(0x58 | (destination & 7));
        self.program.push(Generated {
            bytes: push,
            text: format!("push {}", register_name(source, Width::Qword, false)),
        });
        self.program.push(Generated {
            bytes: pop,
            text: format!("pop {}", register_name(destination, Width::Qword, false)),
        });
    }

    fn flag_instruction(&mut self) {
        let (bytes, text): (&[u8], &str) = match self.rng.below(6) {
            0 => (&[0xf8], "clc"),
            1 => (&[0xf9], "stc"),
            2 => (&[0xf5], "cmc"),
            3 => (&[0x9f], "lahf"),
            4 => (&[0x9e], "sahf"),
            _ => (&[0x90], "nop"),
        };
        self.program.push(Generated {
            bytes: bytes.to_vec(),
            text: text.to_string(),
        });
    }

    fn pushf_to_register(&mut self) {
        let destination = self.data_register();
        let mut pop = Vec::new();
        if destination >= 8 {
            pop.push(0x41);
        }
        pop.push(0x58 | (destination & 7));
        self.program.push(Generated {
            bytes: vec![0x9c],
            text: "pushfq".to_string(),
        });
        self.program.push(Generated {
            bytes: pop,
            text: format!("pop {}", register_name(destination, Width::Qword, false)),
        });
    }

    fn sign_extension(&mut self) {
        let (bytes, text): (&[u8], &str) = match self.rng.below(6) {
            0 => (&[0x99], "cdq"),
            1 => (&[0x48, 0x99], "cqo"),
            2 => (&[0x66, 0x99], "cwd"),
            3 => (&[0x98], "cwde"),
            4 => (&[0x48, 0x98], "cdqe"),
            _ => (&[0x66, 0x98], "cbw"),
        };
        self.program.push(Generated {
            bytes: bytes.to_vec(),
            text: text.to_string(),
        });
    }

    fn clock_or_cpuid(&mut self) {
        let (bytes, text): (&[u8], &str) = if self.rng.below(2) == 0 {
            (&[0x0f, 0x31], "rdtsc")
        } else {
            (&[0x0f, 0xa2], "cpuid")
        };
        self.program.push(Generated {
            bytes: bytes.to_vec(),
            text: text.to_string(),
        });
    }

    fn straight_line_instruction(&mut self, profile: Profile) {
        let choice = self.rng.below(100);
        match profile {
            Profile::Alu => match choice {
                0..=24 => self.alu_register_register(),
                25..=44 => self.alu_immediate(),
                45..=54 => self.test_instruction(),
                55..=64 => self.unary(),
                65..=74 => self.mov(),
                75..=84 => self.flag_instruction(),
                85..=92 => self.lea(),
                _ => self.setcc(),
            },
            Profile::Memory => match choice {
                0..=19 => self.mov(),
                20..=34 => self.alu_register_memory(),
                35..=44 => self.alu_immediate(),
                45..=54 => self.extension_move(),
                55..=62 => self.movsxd(),
                63..=72 => self.lea(),
                73..=80 => self.exchange(),
                81..=88 => self.push_pop_pair(),
                89..=94 => self.cmovcc(),
                _ => self.unary(),
            },
            Profile::Shifts => match choice {
                0..=39 => self.shift(),
                40..=54 => self.alu_register_register(),
                55..=64 => self.flag_instruction(),
                65..=74 => self.setcc(),
                75..=84 => self.pushf_to_register(),
                85..=92 => self.unary(),
                _ => self.bit_operations(),
            },
            Profile::Flags => match choice {
                0..=19 => self.alu_register_register(),
                20..=34 => self.alu_immediate(),
                35..=49 => self.setcc(),
                50..=59 => self.cmovcc(),
                60..=69 => self.flag_instruction(),
                70..=79 => self.pushf_to_register(),
                80..=89 => self.unary(),
                _ => self.shift(),
            },
            Profile::Wide => match choice {
                0..=11 => self.multiply(),
                12..=23 => self.bit_operations(),
                24..=33 => self.bit_scan(),
                34..=43 => self.byte_swap(),
                44..=53 => self.sign_extension(),
                54..=63 => self.exchange(),
                64..=73 => self.extension_move(),
                74..=83 => self.alu_register_register(),
                84..=91 => self.clock_or_cpuid(),
                _ => self.shift(),
            },
            Profile::Everything => match choice {
                0..=11 => self.alu_register_register(),
                12..=21 => self.alu_immediate(),
                22..=29 => self.alu_register_memory(),
                30..=37 => self.mov(),
                38..=43 => self.lea(),
                44..=49 => self.unary(),
                50..=57 => self.shift(),
                58..=62 => self.extension_move(),
                63..=65 => self.movsxd(),
                66..=70 => self.setcc(),
                71..=74 => self.cmovcc(),
                75..=77 => self.multiply(),
                78..=80 => self.bit_operations(),
                81..=83 => self.exchange(),
                84..=86 => self.push_pop_pair(),
                87..=89 => self.flag_instruction(),
                90..=92 => self.test_instruction(),
                93..=94 => self.bit_scan(),
                95 => self.byte_swap(),
                96 => self.sign_extension(),
                97 => self.pushf_to_register(),
                _ => self.clock_or_cpuid(),
            },
        }
    }

    /// A forward `jcc` over the next `skipped` generated instructions. The skipped instructions
    /// are generated here so the displacement is known.
    fn forward_branch(&mut self, profile: Profile) {
        let condition = self.rng.below(16) as u8;
        let skipped = 1 + self.rng.below(3) as usize;
        let start = self.program.len();
        self.program.push(Generated {
            bytes: Vec::new(),
            text: String::new(),
        });
        for _ in 0..skipped {
            self.straight_line_instruction(profile);
        }
        let distance: usize = self.program[start + 1..]
            .iter()
            .map(|g| g.bytes.len())
            .sum();
        // The skipped instructions are far shorter than the rel8 limit.
        assert!(distance < 128);
        self.program[start] = Generated {
            bytes: vec![0x70 | condition, distance as u8],
            text: format!(
                "j{} .+{}",
                CONDITION_NAMES[condition as usize],
                distance + 2
            ),
        };
    }

    /// `mov r15d, n; loop: <body>; dec r15d; jnz loop`. The body never writes r15.
    fn counted_loop(&mut self, profile: Profile) {
        let iterations = 2 + self.rng.below(6) as u32;
        let mut prologue = vec![0x41, 0xb8 | (LOOP_COUNTER & 7)];
        prologue.extend(iterations.to_le_bytes());
        self.program.push(Generated {
            bytes: prologue,
            text: format!("mov r15d, {iterations}"),
        });
        let start = self.program.len();
        for _ in 0..2 + self.rng.below(4) {
            self.straight_line_instruction(profile);
        }
        let body: usize = self.program[start..].iter().map(|g| g.bytes.len()).sum();
        self.program.push(Generated {
            bytes: vec![0x41, 0xff, 0xc8 | (LOOP_COUNTER & 7)],
            text: "dec r15d".to_string(),
        });
        let back = -((body + 3 + 2) as i64);
        assert!(back >= -128, "loop body too long for rel8");
        self.program.push(Generated {
            bytes: vec![0x75, back as i8 as u8],
            text: format!("jne .-{}", body + 3),
        });
    }
}

#[derive(Clone, Copy, Debug)]
enum Profile {
    Alu,
    Memory,
    Shifts,
    Flags,
    Wide,
    Everything,
}

fn generate(seed: u64, profile: Profile, count: usize) -> Vec<Generated> {
    let mut builder = Builder {
        rng: Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1),
        program: Vec::new(),
    };
    for _ in 0..8 {
        builder.rng.next();
    }
    while builder.program.len() < count {
        match builder.rng.below(10) {
            0 => builder.forward_branch(profile),
            1 if builder.program.len() + 12 < count => builder.counted_loop(profile),
            _ => builder.straight_line_instruction(profile),
        }
    }
    builder.program
}

fn program_bytes(program: &[Generated]) -> Vec<u8> {
    let mut code: Vec<u8> = program
        .iter()
        .flat_map(|g| g.bytes.iter().copied())
        .collect();
    code.push(HLT);
    code
}

/// Initial machine state for a seed: random data registers, base/index registers pointing into
/// the seeded scratch memory, random arithmetic flags.
fn random_setup(seed: u64) -> impl Fn(&mut Harness) {
    move |harness| {
        seed_registers_and_memory(harness);
        let mut rng = Rng(seed ^ 0xa5a5_5a5a_dead_beef | 1);
        for _ in 0..4 {
            rng.next();
        }
        for register in DATA_REGISTERS {
            harness.cpu.gpr[register as usize] = if rng.below(3) == 0 {
                *rng.pick(&INTERESTING_VALUES)
            } else {
                rng.next()
            };
        }
        for register in BASE_REGISTERS {
            harness.cpu.gpr[register as usize] =
                SCRATCH + BASE_OFFSETS[rng.below(BASE_OFFSETS.len() as u64) as usize];
        }
        for register in INDEX_REGISTERS {
            harness.cpu.gpr[register as usize] = rng.below(MAX_INDEX_VALUE);
        }
        harness.cpu.gpr[LOOP_COUNTER as usize] = 0;
        harness.cpu.gpr[RSP as usize] = STACK_TOP;
        harness.cpu.set_eflags_arith(rng.next() & ARITH);
    }
}

const PROGRAM_LENGTH: usize = 40;
const SEEDS_PER_TEST: u64 = 150;

fn run_seeds(profile: Profile, first_seed: u64) {
    for seed in first_seed..first_seed + SEEDS_PER_TEST {
        let program = generate(seed, profile, PROGRAM_LENGTH);
        let code = program_bytes(&program);
        let result = run_serialized(&code, random_setup(seed));
        let listing: Vec<&str> = program.iter().map(|g| g.text.as_str()).collect();
        assert_eq!(
            result.interpreter_stop,
            Stop::Halted,
            "{profile:?} seed {seed}: {listing:?}\n{}",
            result.interpreter.cpu.dump_regs()
        );
        if let Some(difference) = difference_in(&result, HOT_RAM) {
            panic!("{profile:?} seed {seed}: {difference}\n{listing:#?}");
        }
    }
}

macro_rules! random_tests {
    ($($name:ident => ($profile:ident, $first_seed:expr),)*) => {
        $(#[test]
        fn $name() {
            run_seeds(Profile::$profile, $first_seed);
        })*
    };
}

random_tests! {
    random_alu_programs_a => (Alu, 0),
    random_alu_programs_b => (Alu, 1_000),
    random_memory_programs_a => (Memory, 2_000),
    random_memory_programs_b => (Memory, 3_000),
    random_memory_programs_c => (Memory, 3_500),
    random_shift_programs_a => (Shifts, 4_000),
    random_shift_programs_b => (Shifts, 5_000),
    random_flag_programs_a => (Flags, 6_000),
    random_flag_programs_b => (Flags, 7_000),
    random_wide_instruction_programs_a => (Wide, 8_000),
    random_wide_instruction_programs_b => (Wide, 9_000),
    random_mixed_programs_a => (Everything, 10_000),
    random_mixed_programs_b => (Everything, 11_000),
    random_mixed_programs_c => (Everything, 12_000),
    random_mixed_programs_d => (Everything, 13_000),
}

#[test]
fn generated_programs_are_deterministic_per_seed() {
    let first = program_bytes(&generate(42, Profile::Everything, PROGRAM_LENGTH));
    let second = program_bytes(&generate(42, Profile::Everything, PROGRAM_LENGTH));
    let other = program_bytes(&generate(43, Profile::Everything, PROGRAM_LENGTH));
    assert_eq!(first, second);
    assert_ne!(first, other);
}

#[test]
fn generated_programs_end_in_hlt_and_stay_short_enough_for_one_snippet() {
    for seed in 0..50 {
        let code = program_bytes(&generate(seed, Profile::Everything, PROGRAM_LENGTH));
        assert_eq!(*code.last().unwrap(), HLT);
        assert!(code.len() < 1024);
    }
}

#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
#[test]
fn random_programs_exercise_the_inline_paths_not_only_the_helper() {
    let mut inlined = 0;
    let mut total = 0;
    let mut blocks = 0;
    for seed in 0..40 {
        let program = generate(seed, Profile::Everything, PROGRAM_LENGTH);
        let code = program_bytes(&program);
        let result = run_serialized(&code, random_setup(seed));
        let (jit, _) = result.jit.as_ref().unwrap();
        let stats = jit.cpu.jit.as_ref().unwrap().stats();
        inlined += stats.inlined;
        total += stats.total;
        blocks += stats.translated;
    }
    assert!(
        inlined * 4 >= total,
        "only {inlined} of {total} instructions were inlined"
    );
    assert!(
        blocks > 40,
        "programs with branches and loops span many blocks: {blocks}"
    );
}
