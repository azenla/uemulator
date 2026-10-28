//! Test scaffolding for 16- and 32-bit modes: a flat 32-bit GDT, 8-byte IDT gates and a 32-bit TSS.

use super::test_support::{GDT_BASE, descriptor};
use super::*;
use crate::testkit::Harness;

pub(crate) const LEGACY_CS: u16 = 0x08;
pub(crate) const LEGACY_DS: u16 = 0x10;
pub(crate) const LEGACY_USER_CS: u16 = 0x18 | 3;
pub(crate) const LEGACY_USER_DS: u16 = 0x20 | 3;
pub(crate) const LEGACY_TSS: u16 = 0x28;
pub(crate) const LEGACY_SECOND_TSS: u16 = 0x30;
pub(crate) const LEGACY_CS16: u16 = 0x38;
pub(crate) const LEGACY_CONFORMING_CS: u16 = 0x50;
pub(crate) const LEGACY_LDT: u16 = 0x58;
pub(crate) const LEGACY_LONG_CS: u16 = 0x60;
pub(crate) const LEGACY_COMPAT_CS: u16 = 0x68;
pub(crate) const LEGACY_LAST_SELECTOR: u16 = 0x68;
pub(crate) const LEGACY_IDT: u64 = 0x10_0000;
pub(crate) const LEGACY_TSS_BASE: u64 = 0x11_0000;
pub(crate) const LEGACY_SECOND_TSS_BASE: u64 = 0x12_0000;
pub(crate) const KERNEL_STACK: u64 = 0x70_0000;
pub(crate) const LEGACY_LDT_BASE: u64 = 0x13_0000;

const ACCESS_CODE: u8 = 0x9a;
const ACCESS_DATA: u8 = 0x92;
const ACCESS_CONFORMING_CODE: u8 = 0x9e;
const ACCESS_TSS32_AVAILABLE: u8 = 0x89;
const FLAGS_BIG_PAGE: u8 = 0xc;
const FLAGS_SMALL: u8 = 0x0;
const TSS_LIMIT: u32 = 0x67;

pub(crate) fn dpl(access: u8, level: u8) -> u8 {
    (access & !0x60) | (level << 5)
}

pub(crate) fn set_descriptor(h: &mut Harness, selector: u16, value: u64) {
    h.write(GDT_BASE + (selector & 0xfff8) as u64, &value.to_le_bytes());
}

pub(crate) fn gate_descriptor(selector: u16, offset: u32, access: u8, count: u8) -> u64 {
    (offset as u64 & 0xffff)
        | ((selector as u64) << 16)
        | ((count as u64) << 32)
        | ((access as u64) << 40)
        | ((offset as u64 >> 16) << 48)
}

/// Writes IDT entry `vector`: a gate with the given type/DPL byte.
pub(crate) fn set_idt_gate(h: &mut Harness, vector: u8, selector: u16, offset: u32, access: u8) {
    h.write(
        LEGACY_IDT + vector as u64 * 8,
        &gate_descriptor(selector, offset, access, 0).to_le_bytes(),
    );
}

/// Flat 32-bit protected mode: paging and long mode off, ring 0, `LEGACY_*` descriptors installed.
pub(crate) fn protected32(code: &[u8]) -> Harness {
    let mut h = Harness::new(code);
    install_legacy_gdt(&mut h);
    let cpu = &mut h.cpu;
    cpu.efer = 0;
    cpu.cr0 = CR0_PE | CR0_ET;
    cpu.cr4 = 0;
    cpu.cr3 = 0;
    cpu.idtr = DescriptorTable {
        base: LEGACY_IDT,
        limit: 0x7ff,
    };
    h.cpu.flush_tlb();
    h.cpu.flush_decode();
    h.cpu.seg[SEG_CS] = flat_segment(LEGACY_CS, ACCESS_CODE, FLAGS_BIG_PAGE);
    for index in [SEG_ES, SEG_SS, SEG_DS, SEG_FS, SEG_GS] {
        h.cpu.seg[index] = flat_segment(LEGACY_DS, ACCESS_DATA, FLAGS_BIG_PAGE);
    }
    h.cpu.set_cpl(0);
    h.cpu.gpr[4] = crate::testkit::STACK_TOP;
    h
}

/// A segment register value as loading the descriptor would give it: base 0, 4 GiB.
pub(crate) fn flat_segment(selector: u16, access: u8, flags: u8) -> Seg {
    let desc = descriptor(0, 0xfffff, access, flags);
    seg_from_descriptor(selector, desc)
}

pub(crate) fn seg_from_descriptor(selector: u16, desc: u64) -> Seg {
    let dword0 = desc as u32;
    let dword1 = (desc >> 32) as u32;
    let base = ((dword0 >> 16) | ((dword1 & 0xff) << 16) | (dword1 & 0xff00_0000)) as u64;
    let mut limit = (dword0 & 0xffff) | (dword1 & 0xf_0000);
    if dword1 & SEG_GRANULARITY != 0 {
        limit = (limit << 12) | 0xfff;
    }
    Seg {
        sel: selector,
        base,
        limit,
        flags: dword1 & SEG_DESCRIPTOR_FLAGS_MASK,
    }
}

pub(crate) fn install_legacy_gdt(h: &mut Harness) {
    set_descriptor(h, 0, 0);
    set_descriptor(
        h,
        LEGACY_CS,
        descriptor(0, 0xfffff, ACCESS_CODE, FLAGS_BIG_PAGE),
    );
    set_descriptor(
        h,
        LEGACY_DS,
        descriptor(0, 0xfffff, ACCESS_DATA, FLAGS_BIG_PAGE),
    );
    set_descriptor(
        h,
        LEGACY_USER_CS,
        descriptor(0, 0xfffff, dpl(ACCESS_CODE, 3), FLAGS_BIG_PAGE),
    );
    set_descriptor(
        h,
        LEGACY_USER_DS,
        descriptor(0, 0xfffff, dpl(ACCESS_DATA, 3), FLAGS_BIG_PAGE),
    );
    set_descriptor(
        h,
        LEGACY_TSS,
        descriptor(
            LEGACY_TSS_BASE as u32,
            TSS_LIMIT,
            ACCESS_TSS32_AVAILABLE,
            FLAGS_SMALL,
        ),
    );
    set_descriptor(
        h,
        LEGACY_SECOND_TSS,
        descriptor(
            LEGACY_SECOND_TSS_BASE as u32,
            TSS_LIMIT,
            ACCESS_TSS32_AVAILABLE,
            FLAGS_SMALL,
        ),
    );
    set_descriptor(
        h,
        LEGACY_CS16,
        descriptor(0, 0xffff, ACCESS_CODE, FLAGS_SMALL),
    );
    set_descriptor(
        h,
        LEGACY_CONFORMING_CS,
        descriptor(0, 0xfffff, ACCESS_CONFORMING_CODE, FLAGS_BIG_PAGE),
    );
    set_descriptor(
        h,
        LEGACY_LDT,
        descriptor(LEGACY_LDT_BASE as u32, 0xff, 0x82, FLAGS_SMALL),
    );
    set_descriptor(h, LEGACY_LONG_CS, descriptor(0, 0xfffff, ACCESS_CODE, 0xa));
    set_descriptor(
        h,
        LEGACY_COMPAT_CS,
        descriptor(0, 0xfffff, ACCESS_CODE, FLAGS_BIG_PAGE),
    );
    h.cpu.gdtr = DescriptorTable {
        base: GDT_BASE,
        limit: LEGACY_LAST_SELECTOR | 7,
    };
}

/// Makes TR point at the first TSS with ring 0 stack `ss0:esp0`.
pub(crate) fn install_legacy_tss(h: &mut Harness, esp0: u32, ss0: u16) {
    h.write(LEGACY_TSS_BASE + 4, &esp0.to_le_bytes());
    h.write(LEGACY_TSS_BASE + 8, &(ss0 as u32).to_le_bytes());
    let mut tr = seg_from_descriptor(
        LEGACY_TSS,
        descriptor(LEGACY_TSS_BASE as u32, TSS_LIMIT, 0x8b, FLAGS_SMALL),
    );
    tr.sel = LEGACY_TSS;
    h.cpu.tr = tr;
    // mark busy in the GDT, as ltr would
    set_descriptor(
        h,
        LEGACY_TSS,
        descriptor(LEGACY_TSS_BASE as u32, TSS_LIMIT, 0x8b, FLAGS_SMALL),
    );
}

/// Drops the CPU to ring 3 with flat user segments and `esp`.
pub(crate) fn enter_ring3(h: &mut Harness, esp: u64) {
    h.cpu.seg[SEG_CS] = flat_segment(LEGACY_USER_CS, dpl(ACCESS_CODE, 3), FLAGS_BIG_PAGE);
    for index in [SEG_ES, SEG_SS, SEG_DS, SEG_FS, SEG_GS] {
        h.cpu.seg[index] = flat_segment(LEGACY_USER_DS, dpl(ACCESS_DATA, 3), FLAGS_BIG_PAGE);
    }
    h.cpu.set_cpl(3);
    h.cpu.gpr[4] = esp;
}
