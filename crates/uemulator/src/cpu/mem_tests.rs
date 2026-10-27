use super::*;
use crate::cpu::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_PAGE_FAULT, VECTOR_STACK_FAULT};
use crate::cpu::test_support::*;
use std::sync::Mutex;

use crate::dev::DevConfig;
use crate::testkit::{CODE_BASE, HARNESS_RAM_BYTES, HLT, Harness, SCRATCH};

/// Canonical linear addresses used for the mappings below; all 4 KiB aligned.
const VA: u64 = 0x0000_7f12_3456_0000;
const VA_NEXT_PAGE: u64 = VA + PAGE;
/// Physical pages inside RAM that are not adjacent, so a straddling access must be stitched
/// from two separate translations.
const PA: u64 = SCRATCH;
const PA_NEXT: u64 = SCRATCH + 5 * PAGE;
/// First physical address past the harness RAM: unmapped unless a device claims it.
const FIRST_UNMAPPED_PA: u64 = HARNESS_RAM_BYTES as u64;
const SUPERVISOR_ONLY: u64 = PTE_P | PTE_W;
const READ_ONLY_USER: u64 = PTE_P | PTE_U;
const NOT_CANONICAL: u64 = 0x0000_8000_0000_0000;
const LAST_LOW_CANONICAL: u64 = 0x0000_7fff_ffff_ffff;
const FIRST_HIGH_CANONICAL: u64 = 0xffff_8000_0000_0000;
const LAST_NON_CANONICAL: u64 = 0xffff_7fff_ffff_ffff;

fn harness() -> Harness {
    Harness::new(&[HLT])
}

fn paged() -> (Harness, PageTables) {
    let mut h = harness();
    let tables = PageTables::new(&mut h.cpu);
    (h, tables)
}

fn set_nxe(cpu: &mut Cpu, on: bool) {
    cpu.efer = if on {
        cpu.efer | EFER_NXE
    } else {
        cpu.efer & !EFER_NXE
    };
}

fn set_wp(cpu: &mut Cpu, on: bool) {
    cpu.cr0 = if on {
        cpu.cr0 | CR0_WP
    } else {
        cpu.cr0 & !CR0_WP
    };
}

fn slot_of(cpu: &Cpu, linear: u64, user: bool) -> TlbEntry {
    cpu.tlb[tlb_slot(linear, user)]
}

fn logged(log: &Mutex<Vec<String>>) -> Vec<String> {
    log.lock().unwrap().clone()
}

fn host_address(cpu: &Cpu, physical: u64) -> u64 {
    cpu.ram_base + physical
}

mod page_fault_construction {
    use super::*;

    #[test]
    fn sets_cr2_to_the_faulting_address_and_vector_14() {
        let mut h = harness();
        let exception = h
            .cpu
            .page_fault(0xdead_b000, AccessKind::Read, false, false, false);
        assert_eq!(fault_of::<()>(Err(exception)), (VECTOR_PAGE_FAULT, Some(0)));
        assert_eq!(h.cpu.cr2, 0xdead_b000);
    }

    #[test]
    fn every_error_code_bit_combination_matches_the_architecture() {
        for nxe in [false, true] {
            for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                for user in [false, true] {
                    for present in [false, true] {
                        for reserved in [false, true] {
                            let mut h = harness();
                            set_nxe(&mut h.cpu, nxe);
                            let exception =
                                h.cpu.page_fault(0x1234, access, user, present, reserved);
                            let mut expected = 0;
                            if present {
                                expected |= PF_P;
                            }
                            if access == AccessKind::Write {
                                expected |= PF_W;
                            }
                            if user {
                                expected |= PF_U;
                            }
                            if reserved {
                                expected |= PF_RSVD;
                            }
                            if access == AccessKind::Exec && nxe {
                                expected |= PF_ID;
                            }
                            assert_eq!(
                                fault_of::<()>(Err(exception)),
                                (VECTOR_PAGE_FAULT, Some(expected)),
                                "nxe={nxe} access={:?} user={user} present={present} reserved={reserved}",
                                access as u8,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn instruction_fetch_bit_is_absent_without_nxe() {
        let mut h = harness();
        let exception = h
            .cpu
            .page_fault(0x1000, AccessKind::Exec, false, true, false);
        assert_eq!(fault_of::<()>(Err(exception)).1, Some(PF_P));
    }

    #[test]
    fn a_later_fault_overwrites_cr2() {
        let mut h = harness();
        let _ = h
            .cpu
            .page_fault(0x1000, AccessKind::Read, false, false, false);
        let _ = h
            .cpu
            .page_fault(0x2000, AccessKind::Read, false, false, false);
        assert_eq!(h.cpu.cr2, 0x2000);
    }
}

mod page_walk {
    use super::*;

    #[test]
    fn four_kib_mapping_translates_with_the_page_offset() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        for offset in [0, 1, 0x123, 0xabc, 0xfff] {
            assert_eq!(
                h.cpu
                    .translate(VA + offset, AccessKind::Read, false)
                    .unwrap(),
                PA + offset
            );
        }
    }

    #[test]
    fn read_sets_accessed_on_every_level_and_never_dirty() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        for level in 0..=3 {
            let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, level));
            assert_ne!(entry & PTE_A, 0, "accessed at level {level}");
            assert_eq!(entry & PTE_D, 0, "dirty at level {level}");
        }
    }

    #[test]
    fn exec_sets_accessed_and_not_dirty() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Exec, false).unwrap();
        assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), PTE_A);
    }

    #[test]
    fn write_sets_dirty_on_the_leaf_only() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        for level in 1..=3 {
            let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, level));
            assert_eq!(entry & PTE_D, 0, "upper level {level} must not get dirty");
            assert_ne!(entry & PTE_A, 0);
        }
        let leaf = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, 0));
        assert_eq!(leaf & (PTE_A | PTE_D), PTE_A | PTE_D);
    }

    #[test]
    fn walk_preserves_the_other_entry_bits() {
        let (mut h, mut tables) = paged();
        let flags = PTE_P | PTE_W | PTE_U | (1 << 3) | (1 << 4) | (1 << 8) | (1 << 9) | PTE_NX;
        set_nxe(&mut h.cpu, true);
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, flags);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        assert_eq!(read_pte(&h.cpu, leaf), PA | flags | PTE_A | PTE_D);
    }

    #[test]
    fn already_set_accessed_and_dirty_bits_are_left_alone() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(
            &mut h.cpu,
            VA,
            PA,
            PTE_PRESENT_WRITABLE_USER | PTE_A | PTE_D,
        );
        let before = read_pte(&h.cpu, leaf);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        assert_eq!(read_pte(&h.cpu, leaf), before);
    }

    #[test]
    fn two_mib_leaf_translates_every_offset_inside_it() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_5000_0020_0000;
        tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
        for offset in [0, 0x1000, 0x1_2345, 0x10_0fff, TWO_MIB - 1] {
            assert_eq!(
                h.cpu
                    .translate(va + offset, AccessKind::Read, false)
                    .unwrap(),
                0x40_0000 + offset,
                "offset {offset:#x}"
            );
        }
    }

    #[test]
    fn two_mib_leaf_gets_accessed_on_three_levels_and_dirty_on_the_pde() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_5000_0020_0000;
        let pde = tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
        h.cpu
            .translate(va + 0x5000, AccessKind::Read, false)
            .unwrap();
        assert_eq!(read_pte(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A);
        h.cpu
            .translate(va + 0x6000, AccessKind::Write, false)
            .unwrap();
        assert_eq!(read_pte(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A | PTE_D);
        for level in 2..=3 {
            let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, va, level));
            assert_ne!(entry & PTE_A, 0);
            assert_eq!(entry & PTE_D, 0);
        }
    }

    #[test]
    fn two_mib_leaf_maps_each_small_page_to_its_own_ram_bytes() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_5000_0020_0000;
        tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
        h.write(0x40_0000 + 3 * PAGE + 8, &[0xa1]);
        h.write(0x40_0000 + 9 * PAGE + 8, &[0xb2]);
        assert_eq!(h.cpu.read(va + 3 * PAGE + 8, 1).unwrap(), 0xa1);
        assert_eq!(h.cpu.read(va + 9 * PAGE + 8, 1).unwrap(), 0xb2);
    }

    #[test]
    fn two_mib_leaf_with_pat_bit_set_is_not_a_reserved_violation() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_5000_0020_0000;
        tables.map(
            &mut h.cpu,
            va,
            0x40_0000,
            &[
                PTE_PRESENT_WRITABLE_USER,
                PTE_PRESENT_WRITABLE_USER,
                PTE_PRESENT_WRITABLE_USER | (1 << 12),
            ],
        );
        assert_eq!(
            h.cpu.translate(va, AccessKind::Read, false).unwrap(),
            0x40_0000
        );
    }

    #[test]
    fn one_gib_leaf_translates_inside_ram_and_beyond_it() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_4000_0000_0000;
        let pdpte = tables.map(&mut h.cpu, va, 0, &[PTE_PRESENT_WRITABLE_USER; 2]);
        assert_eq!(
            h.cpu
                .translate(va + 0x40_0123, AccessKind::Read, false)
                .unwrap(),
            0x40_0123
        );
        assert_eq!(
            h.cpu
                .translate(va + 0x2345_6789, AccessKind::Read, false)
                .unwrap(),
            0x2345_6789
        );
        assert_eq!(
            h.cpu
                .translate(va + ONE_GIB - 1, AccessKind::Read, false)
                .unwrap(),
            ONE_GIB - 1
        );
        assert_eq!(read_pte(&h.cpu, pdpte) & (PTE_A | PTE_D), PTE_A);
        h.cpu
            .translate(va + 0x1000, AccessKind::Write, false)
            .unwrap();
        assert_eq!(read_pte(&h.cpu, pdpte) & (PTE_A | PTE_D), PTE_A | PTE_D);
    }

    #[test]
    fn one_gib_leaf_with_a_nonzero_physical_base() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_4000_0000_0000;
        tables.map(&mut h.cpu, va, 3 * ONE_GIB, &[PTE_PRESENT_WRITABLE_USER; 2]);
        assert_eq!(
            h.cpu
                .translate(va + 0x1234_5678, AccessKind::Read, false)
                .unwrap(),
            3 * ONE_GIB + 0x1234_5678
        );
    }

    #[test]
    fn one_gib_leaf_reads_ram_through_the_mapping() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_4000_0000_0000;
        tables.map(&mut h.cpu, va, 0, &[PTE_PRESENT_WRITABLE_USER; 2]);
        h.write_u64(SCRATCH + 0x40, 0x1122_3344_5566_7788);
        assert_eq!(
            h.cpu.read(va + SCRATCH + 0x40, 8).unwrap(),
            0x1122_3344_5566_7788
        );
    }

    #[test]
    fn missing_entry_at_each_level_is_a_not_present_fault() {
        for level in 0..=3 {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            let entry_address = tables.entry_address(&h.cpu, VA, level);
            let entry = read_pte(&h.cpu, entry_address);
            write_pte(&mut h.cpu, entry_address, entry & !PTE_P);
            let result = h.cpu.translate(VA + 0x10, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(0)),
                "level {level}"
            );
            assert_eq!(h.cpu.cr2, VA + 0x10);
        }
    }

    #[test]
    fn not_present_error_codes_carry_write_user_and_fetch_bits() {
        let (mut h, _tables) = paged();
        set_nxe(&mut h.cpu, true);
        let cases = [
            (AccessKind::Read, false, 0),
            (AccessKind::Write, false, PF_W),
            (AccessKind::Exec, false, PF_ID),
            (AccessKind::Read, true, PF_U),
            (AccessKind::Write, true, PF_W | PF_U),
            (AccessKind::Exec, true, PF_U | PF_ID),
        ];
        for (access, user, expected) in cases {
            h.cpu.set_cpl(if user { 3 } else { 0 });
            let result = h.cpu.translate(VA, access, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(expected)));
        }
    }

    #[test]
    fn upper_levels_get_accessed_even_when_the_leaf_is_not_present() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        write_pte(&mut h.cpu, leaf, 0);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap_err();
        for level in 1..=3 {
            let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, level));
            assert_ne!(entry & PTE_A, 0, "level {level}");
        }
    }

    #[test]
    fn denied_access_leaves_the_leaf_accessed_and_dirty_untouched() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, READ_ONLY_USER);
        h.cpu.set_cpl(3);
        let result = h.cpu.translate(VA, AccessKind::Write, false);
        assert_eq!(
            fault_of(result),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
        );
        assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), 0);
    }

    #[test]
    fn page_table_outside_ram_counts_as_not_present() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        let pml4e = tables.entry_address(&h.cpu, VA, 3);
        write_pte(&mut h.cpu, pml4e, 0x1000_0000 | PTE_PRESENT_WRITABLE_USER);
        let result = h.cpu.translate(VA, AccessKind::Read, false);
        assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(0)));
    }

    #[test]
    fn cr3_outside_ram_counts_as_not_present() {
        let mut h = harness();
        h.cpu.cr3 = 0x4000_0000;
        let result = h.cpu.translate(VA, AccessKind::Read, false);
        assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(0)));
    }

    #[test]
    fn cr3_low_bits_are_ignored() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.cr3 = tables.pml4 | 0x18;
        assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
    }

    #[test]
    fn higher_half_addresses_use_the_same_walk() {
        let (mut h, mut tables) = paged();
        let va = 0xffff_ffff_8000_0000;
        tables.map_4k(&mut h.cpu, va, PA, SUPERVISOR_ONLY);
        assert_eq!(
            h.cpu.translate(va + 0x88, AccessKind::Read, false).unwrap(),
            PA + 0x88
        );
    }

    #[test]
    fn nonpresent_entry_with_reserved_bits_is_just_not_present() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        write_pte(&mut h.cpu, leaf, PA | (1 << 51) | PTE_NX);
        let result = h.cpu.translate(VA, AccessKind::Read, false);
        assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(0)));
    }
}

mod reserved_bits {
    use super::*;

    fn rsvd_code(access: AccessKind, user: bool, nxe: bool) -> u32 {
        let mut code = PF_P | PF_RSVD;
        if access == AccessKind::Write {
            code |= PF_W;
        }
        if user {
            code |= PF_U;
        }
        if access == AccessKind::Exec && nxe {
            code |= PF_ID;
        }
        code
    }

    #[test]
    fn physical_address_bits_40_to_51_are_reserved_at_every_level() {
        for level in 0..=3 {
            for bit in [40, 45, 51] {
                let (mut h, mut tables) = paged();
                tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                let entry_address = tables.entry_address(&h.cpu, VA, level);
                let entry = read_pte(&h.cpu, entry_address);
                write_pte(&mut h.cpu, entry_address, entry | (1 << bit));
                let result = h.cpu.translate(VA, AccessKind::Read, false);
                assert_eq!(
                    fault_of(result),
                    (
                        VECTOR_PAGE_FAULT,
                        Some(rsvd_code(AccessKind::Read, false, false))
                    ),
                    "level {level} bit {bit}"
                );
                assert_eq!(h.cpu.cr2, VA);
            }
        }
    }

    #[test]
    fn bits_52_to_62_are_not_checked() {
        // Software-available and protection-key bits: the emulator does not reserve them.
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        write_pte(
            &mut h.cpu,
            leaf,
            PA | PTE_PRESENT_WRITABLE_USER | (0x7ff << 52),
        );
        assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
    }

    #[test]
    fn no_execute_bit_is_reserved_at_every_level_without_nxe() {
        for level in 0..=3 {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            let entry_address = tables.entry_address(&h.cpu, VA, level);
            let entry = read_pte(&h.cpu, entry_address);
            write_pte(&mut h.cpu, entry_address, entry | PTE_NX);
            for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                let result = h.cpu.translate(VA, access, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_PAGE_FAULT, Some(rsvd_code(access, false, false))),
                    "level {level}"
                );
            }
        }
    }

    #[test]
    fn no_execute_bit_is_valid_with_nxe_and_only_blocks_fetches() {
        let (mut h, mut tables) = paged();
        set_nxe(&mut h.cpu, true);
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_NX);
        assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
        assert_eq!(h.cpu.translate(VA, AccessKind::Write, false).unwrap(), PA);
        let result = h.cpu.translate(VA, AccessKind::Exec, false);
        assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID)));
    }

    #[test]
    fn reserved_fault_reports_user_and_write_bits() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        write_pte(&mut h.cpu, leaf, PA | PTE_PRESENT_WRITABLE_USER | (1 << 50));
        h.cpu.set_cpl(3);
        let result = h.cpu.translate(VA, AccessKind::Write, false);
        assert_eq!(
            fault_of(result),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U | PF_RSVD))
        );
    }

    #[test]
    fn reserved_violation_does_not_set_leaf_accessed_or_dirty() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        write_pte(&mut h.cpu, leaf, PA | PTE_PRESENT_WRITABLE_USER | (1 << 50));
        h.cpu.translate(VA, AccessKind::Write, false).unwrap_err();
        assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), 0);
    }

    #[test]
    fn page_size_bit_in_a_pml4_entry_is_a_reserved_violation() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        let pml4e = tables.entry_address(&h.cpu, VA, 3);
        let entry = read_pte(&h.cpu, pml4e);
        write_pte(&mut h.cpu, pml4e, entry | PTE_PS);
        let result = h.cpu.translate(VA, AccessKind::Read, false);
        assert_eq!(
            fault_of(result),
            (
                VECTOR_PAGE_FAULT,
                Some(rsvd_code(AccessKind::Read, false, false))
            )
        );
    }

    #[test]
    fn pat_bit_in_a_four_kib_entry_is_not_reserved() {
        let (mut h, mut tables) = paged();
        tables.map(
            &mut h.cpu,
            VA,
            PA,
            &[
                PTE_PRESENT_WRITABLE_USER,
                PTE_PRESENT_WRITABLE_USER,
                PTE_PRESENT_WRITABLE_USER,
                PTE_PRESENT_WRITABLE_USER | PTE_PS,
            ],
        );
        assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
    }

    #[test]
    fn two_mib_leaf_with_address_bits_20_to_13_set_is_reserved() {
        for bit in [13, 16, 20] {
            let (mut h, mut tables) = paged();
            let va = 0x0000_5000_0020_0000;
            let pde = tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
            let entry = read_pte(&h.cpu, pde);
            write_pte(&mut h.cpu, pde, entry | (1 << bit));
            let result = h.cpu.translate(va, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (
                    VECTOR_PAGE_FAULT,
                    Some(rsvd_code(AccessKind::Read, false, false))
                ),
                "bit {bit}"
            );
        }
    }

    #[test]
    fn one_gib_leaf_with_address_bits_29_to_13_set_is_reserved() {
        for bit in [13, 21, 29] {
            let (mut h, mut tables) = paged();
            let va = 0x0000_4000_0000_0000;
            let pdpte = tables.map(&mut h.cpu, va, 0, &[PTE_PRESENT_WRITABLE_USER; 2]);
            let entry = read_pte(&h.cpu, pdpte);
            write_pte(&mut h.cpu, pdpte, entry | (1 << bit));
            let result = h.cpu.translate(va, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (
                    VECTOR_PAGE_FAULT,
                    Some(rsvd_code(AccessKind::Read, false, false))
                ),
                "bit {bit}"
            );
        }
    }
}

mod permissions {
    use super::*;

    /// What the SDM says should happen, restated independently of `walk`.
    fn expected_error_code(
        access: AccessKind,
        user: bool,
        write_protect: bool,
        nxe: bool,
        pte_user: bool,
        pte_writable: bool,
        pte_no_execute: bool,
    ) -> Option<u32> {
        let mut code = PF_P;
        if access == AccessKind::Write {
            code |= PF_W;
        }
        if user {
            code |= PF_U;
        }
        if access == AccessKind::Exec && nxe {
            code |= PF_ID;
        }
        if pte_no_execute && !nxe {
            return Some(code | PF_RSVD);
        }
        if user && !pte_user {
            return Some(code);
        }
        if access == AccessKind::Write && !pte_writable && (user || write_protect) {
            return Some(code);
        }
        if access == AccessKind::Exec && pte_no_execute && nxe {
            return Some(code);
        }
        None
    }

    #[test]
    fn leaf_permission_matrix() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        for user in [false, true] {
            for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                for write_protect in [false, true] {
                    for nxe in [false, true] {
                        for pte_user in [false, true] {
                            for pte_writable in [false, true] {
                                for pte_no_execute in [false, true] {
                                    let mut entry = PA | PTE_P;
                                    if pte_user {
                                        entry |= PTE_U;
                                    }
                                    if pte_writable {
                                        entry |= PTE_W;
                                    }
                                    if pte_no_execute {
                                        entry |= PTE_NX;
                                    }
                                    write_pte(&mut h.cpu, leaf, entry);
                                    h.cpu.flush_tlb();
                                    set_wp(&mut h.cpu, write_protect);
                                    set_nxe(&mut h.cpu, nxe);
                                    h.cpu.set_cpl(if user { 3 } else { 0 });
                                    let context = format!(
                                        "user={user} access={} wp={write_protect} nxe={nxe} pte(u={pte_user} w={pte_writable} nx={pte_no_execute})",
                                        access as u8
                                    );
                                    let expected = expected_error_code(
                                        access,
                                        user,
                                        write_protect,
                                        nxe,
                                        pte_user,
                                        pte_writable,
                                        pte_no_execute,
                                    );
                                    let result = h.cpu.translate(VA + 0x44, access, false);
                                    match expected {
                                        None => {
                                            assert_eq!(result.unwrap(), PA + 0x44, "{context}");
                                            let after = read_pte(&h.cpu, leaf);
                                            assert_ne!(after & PTE_A, 0, "{context}");
                                            assert_eq!(
                                                after & PTE_D != 0,
                                                access == AccessKind::Write,
                                                "{context}"
                                            );
                                        }
                                        Some(code) => {
                                            assert_eq!(
                                                fault_of(result),
                                                (VECTOR_PAGE_FAULT, Some(code)),
                                                "{context}"
                                            );
                                            assert_eq!(h.cpu.cr2, VA + 0x44, "{context}");
                                            assert_eq!(
                                                read_pte(&h.cpu, leaf) & (PTE_A | PTE_D),
                                                0,
                                                "{context}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn weaken(cpu: &mut Cpu, tables: &PageTables, level: u32, clear: u64, set: u64) {
        let address = tables.entry_address(cpu, VA, level);
        let entry = read_pte(cpu, address);
        write_pte(cpu, address, (entry & !clear) | set);
        cpu.flush_tlb();
    }

    #[test]
    fn user_bit_is_the_and_of_every_level() {
        for level in 0..=3 {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            weaken(&mut h.cpu, &tables, level, PTE_U, 0);
            assert_eq!(
                h.cpu.translate(VA, AccessKind::Read, false).unwrap(),
                PA,
                "supervisor, level {level}"
            );
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(VA, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)),
                "user, level {level}"
            );
        }
    }

    #[test]
    fn writable_bit_is_the_and_of_every_level() {
        for level in 0..=3 {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            weaken(&mut h.cpu, &tables, level, PTE_W, 0);
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(VA, AccessKind::Write, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U)),
                "user write, level {level}"
            );
            assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
        }
    }

    #[test]
    fn supervisor_write_to_a_read_only_level_depends_on_cr0_wp() {
        for level in 0..=3 {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            weaken(&mut h.cpu, &tables, level, PTE_W, 0);
            set_wp(&mut h.cpu, false);
            assert_eq!(
                h.cpu.translate(VA, AccessKind::Write, false).unwrap(),
                PA,
                "WP clear, level {level}"
            );
            h.cpu.flush_tlb();
            set_wp(&mut h.cpu, true);
            let result = h.cpu.translate(VA, AccessKind::Write, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W)),
                "WP set, level {level}"
            );
        }
    }

    #[test]
    fn no_execute_is_the_or_of_every_level() {
        for level in 0..=3 {
            let (mut h, mut tables) = paged();
            set_nxe(&mut h.cpu, true);
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            weaken(&mut h.cpu, &tables, level, 0, PTE_NX);
            let result = h.cpu.translate(VA, AccessKind::Exec, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID)),
                "level {level}"
            );
            assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
        }
    }

    #[test]
    fn large_page_leaf_applies_its_own_permissions() {
        let (mut h, mut tables) = paged();
        let va = 0x0000_5000_0020_0000;
        tables.map(
            &mut h.cpu,
            va,
            0x40_0000,
            &[
                PTE_PRESENT_WRITABLE_USER,
                PTE_PRESENT_WRITABLE_USER,
                PTE_P | PTE_U,
            ],
        );
        h.cpu.set_cpl(3);
        let result = h.cpu.translate(va + 0x1000, AccessKind::Write, false);
        assert_eq!(
            fault_of(result),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
        );
        assert_eq!(
            h.cpu
                .translate(va + 0x1000, AccessKind::Read, false)
                .unwrap(),
            0x40_0000 + 0x1000
        );
    }

    #[test]
    fn user_access_to_a_supervisor_only_page_faults_but_implicit_access_succeeds() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
        h.cpu.set_cpl(3);
        let result = h.cpu.translate(VA, AccessKind::Read, false);
        assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)));
        assert_eq!(h.cpu.translate(VA, AccessKind::Read, true).unwrap(), PA);
    }
}

mod tlb {
    use super::*;

    #[test]
    fn read_miss_fills_read_and_exec_tags_but_withholds_write_while_clean() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA + 8, AccessKind::Read, false).unwrap();
        let entry = slot_of(&h.cpu, VA, false);
        assert_eq!(entry.tag_r, VA);
        assert_eq!(entry.tag_x, VA);
        assert_eq!(entry.tag_w, TLB_TAG_INVALID);
    }

    #[test]
    fn addend_turns_a_linear_address_into_the_host_address() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        let entry = slot_of(&h.cpu, VA, false);
        assert_eq!(
            (VA + 0x123).wrapping_add(entry.addend),
            host_address(&h.cpu, PA + 0x123)
        );
    }

    #[test]
    fn write_miss_walks_sets_dirty_and_grants_the_write_tag() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        let entry = slot_of(&h.cpu, VA, false);
        assert_eq!(entry.tag_w, VA);
        assert_eq!(entry.tag_r, VA);
    }

    #[test]
    fn already_dirty_page_gets_its_write_tag_on_a_read_miss() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        assert_eq!(slot_of(&h.cpu, VA, false).tag_w, VA);
    }

    #[test]
    fn first_store_to_a_clean_page_goes_through_the_walk_that_sets_dirty() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.read(VA, 1).unwrap();
        assert_eq!(read_pte(&h.cpu, leaf) & PTE_D, 0);
        h.cpu.write(VA, 1, 0x5a).unwrap();
        assert_ne!(read_pte(&h.cpu, leaf) & PTE_D, 0);
        assert_eq!(h.read(PA, 1), [0x5a]);
    }

    #[test]
    fn write_tag_is_withheld_for_pages_holding_code() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
        h.cpu.mark_code(PA);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        let entry = slot_of(&h.cpu, VA, false);
        assert_eq!(entry.tag_w, TLB_TAG_INVALID);
        assert_eq!(entry.tag_r, VA);
    }

    #[test]
    fn write_tag_returns_once_the_page_no_longer_holds_code() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
        h.cpu.mark_code(PA);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        h.cpu.flush_decode();
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        assert_eq!(slot_of(&h.cpu, VA, false).tag_w, VA);
    }

    #[test]
    fn no_execute_page_gets_no_exec_tag() {
        let (mut h, mut tables) = paged();
        set_nxe(&mut h.cpu, true);
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_NX);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        let entry = slot_of(&h.cpu, VA, false);
        assert_eq!(entry.tag_r, VA);
        assert_eq!(entry.tag_x, TLB_TAG_INVALID);
    }

    #[test]
    fn read_only_page_gets_a_supervisor_write_tag_only_without_wp() {
        for write_protect in [false, true] {
            let (mut h, mut tables) = paged();
            set_wp(&mut h.cpu, write_protect);
            tables.map_4k(&mut h.cpu, VA, PA, PTE_P | PTE_U | PTE_D);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            let granted = slot_of(&h.cpu, VA, false).tag_w == VA;
            assert_eq!(granted, !write_protect, "WP={write_protect}");
        }
    }

    #[test]
    fn user_never_gets_a_write_tag_on_a_read_only_page() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, READ_ONLY_USER | PTE_D);
        h.cpu.set_cpl(3);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        assert_eq!(slot_of(&h.cpu, VA, true).tag_w, TLB_TAG_INVALID);
        assert_eq!(slot_of(&h.cpu, VA, true).tag_r, VA);
    }

    #[test]
    fn user_and_supervisor_accesses_fill_separate_halves() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.set_cpl(3);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        assert_eq!(slot_of(&h.cpu, VA, true).tag_r, VA);
        assert_eq!(slot_of(&h.cpu, VA, false).tag_r, TLB_TAG_INVALID);
        // Implicit accesses at CPL 3 use the supervisor half.
        h.cpu.translate(VA, AccessKind::Read, true).unwrap();
        assert_eq!(slot_of(&h.cpu, VA, false).tag_r, VA);
    }

    #[test]
    fn supervisor_half_is_not_visible_to_user_accesses() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        h.cpu.set_cpl(3);
        // A stale supervisor entry must not let CPL 3 read a supervisor page.
        let result = h.cpu.translate(VA, AccessKind::Read, false);
        assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)));
    }

    #[test]
    fn hit_uses_the_cached_translation_without_walking_again() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        write_pte(&mut h.cpu, leaf, PA_NEXT | PTE_PRESENT_WRITABLE_USER);
        assert_eq!(
            h.cpu.translate(VA + 4, AccessKind::Read, false).unwrap(),
            PA + 4
        );
    }

    #[test]
    fn invlpg_drops_the_cached_translation() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        write_pte(&mut h.cpu, leaf, PA_NEXT | PTE_PRESENT_WRITABLE_USER);
        h.cpu.invlpg(VA + 0x7f0);
        assert_eq!(
            h.cpu.translate(VA + 4, AccessKind::Read, false).unwrap(),
            PA_NEXT + 4
        );
    }

    #[test]
    fn invlpg_clears_the_slot_in_both_halves() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Read, true).unwrap();
        h.cpu.set_cpl(3);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        h.cpu.invlpg(VA);
        let slot = tlb_slot(VA, false);
        assert_eq!(h.cpu.tlb[slot].tag_r, TLB_TAG_INVALID);
        assert_eq!(
            h.cpu.tlb[TLB_ENTRIES_PER_MODE + slot].tag_r,
            TLB_TAG_INVALID
        );
    }

    #[test]
    fn invlpg_leaves_other_slots_alone() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        h.cpu
            .translate(VA_NEXT_PAGE, AccessKind::Read, false)
            .unwrap();
        h.cpu.invlpg(VA);
        assert_eq!(slot_of(&h.cpu, VA_NEXT_PAGE, false).tag_r, VA_NEXT_PAGE);
    }

    #[test]
    fn invlpg_also_drops_another_page_that_shares_the_slot() {
        let (mut h, mut tables) = paged();
        let aliasing_va = VA + TLB_ENTRIES_PER_MODE as u64 * PAGE;
        tables.map_4k(&mut h.cpu, aliasing_va, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
        h.cpu
            .translate(aliasing_va, AccessKind::Read, false)
            .unwrap();
        assert_eq!(tlb_slot(VA, false), tlb_slot(aliasing_va, false));
        h.cpu.invlpg(VA);
        assert_eq!(slot_of(&h.cpu, aliasing_va, false).tag_r, TLB_TAG_INVALID);
    }

    #[test]
    fn pages_that_share_a_slot_evict_each_other() {
        let (mut h, mut tables) = paged();
        let aliasing_va = VA + TLB_ENTRIES_PER_MODE as u64 * PAGE;
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        tables.map_4k(&mut h.cpu, aliasing_va, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        h.cpu
            .translate(aliasing_va, AccessKind::Read, false)
            .unwrap();
        let entry = slot_of(&h.cpu, VA, false);
        assert_eq!(entry.tag_r, aliasing_va);
        assert_eq!(
            entry.tag_w, TLB_TAG_INVALID,
            "the other page's write tag must not leak"
        );
        assert_eq!(entry.tag_x, aliasing_va);
        assert_eq!(
            (aliasing_va + 8).wrapping_add(entry.addend),
            host_address(&h.cpu, PA_NEXT + 8)
        );
    }

    #[test]
    fn refill_for_the_same_page_keeps_the_other_tags() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        let entry = slot_of(&h.cpu, VA, false);
        assert_eq!((entry.tag_r, entry.tag_w, entry.tag_x), (VA, VA, VA));
    }

    #[test]
    fn flush_tlb_empties_every_entry_in_both_halves() {
        let (mut h, mut tables) = paged();
        for i in 0..8 {
            let va = VA + i * PAGE;
            tables.map_4k(&mut h.cpu, va, PA + i * PAGE, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(va, AccessKind::Write, true).unwrap();
        }
        h.cpu.set_cpl(3);
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        h.cpu.flush_tlb();
        assert_eq!(h.cpu.tlb.len(), 2 * TLB_ENTRIES_PER_MODE);
        for entry in &h.cpu.tlb {
            assert_eq!(
                (entry.tag_r, entry.tag_w, entry.tag_x, entry.addend),
                (TLB_TAG_INVALID, TLB_TAG_INVALID, TLB_TAG_INVALID, 0)
            );
        }
    }

    #[test]
    fn pages_outside_ram_are_never_cached() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, 0xc000_0000, PTE_PRESENT_WRITABLE_USER);
        assert_eq!(
            h.cpu.translate(VA + 4, AccessKind::Read, false).unwrap(),
            0xc000_0004
        );
        assert_eq!(slot_of(&h.cpu, VA, false).tag_r, TLB_TAG_INVALID);
    }

    #[test]
    fn last_ram_page_is_cached_and_the_first_page_past_ram_is_not() {
        let (mut h, mut tables) = paged();
        let last_ram_page = FIRST_UNMAPPED_PA - PAGE;
        tables.map_4k(&mut h.cpu, VA, last_ram_page, PTE_PRESENT_WRITABLE_USER);
        tables.map_4k(
            &mut h.cpu,
            VA_NEXT_PAGE,
            FIRST_UNMAPPED_PA,
            PTE_PRESENT_WRITABLE_USER,
        );
        h.cpu.translate(VA, AccessKind::Read, false).unwrap();
        h.cpu
            .translate(VA_NEXT_PAGE, AccessKind::Read, false)
            .unwrap();
        assert_eq!(slot_of(&h.cpu, VA, false).tag_r, VA);
        assert_eq!(slot_of(&h.cpu, VA_NEXT_PAGE, false).tag_r, TLB_TAG_INVALID);
    }

    #[test]
    fn faulting_translation_leaves_the_tlb_untouched() {
        let (mut h, _tables) = paged();
        h.cpu.translate(VA, AccessKind::Read, false).unwrap_err();
        assert_eq!(slot_of(&h.cpu, VA, false).tag_r, TLB_TAG_INVALID);
    }

    #[test]
    fn mark_code_revokes_write_tags_in_both_halves_but_keeps_read_tags() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
        h.cpu.translate(VA, AccessKind::Write, true).unwrap();
        h.cpu.set_cpl(3);
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        assert_eq!(slot_of(&h.cpu, VA, false).tag_w, VA);
        assert_eq!(slot_of(&h.cpu, VA, true).tag_w, VA);
        h.cpu.mark_code(PA + 0x10);
        for user in [false, true] {
            let entry = slot_of(&h.cpu, VA, user);
            assert_eq!(entry.tag_w, TLB_TAG_INVALID, "user={user}");
            assert_eq!(entry.tag_r, VA, "user={user}");
        }
    }

    #[test]
    fn mark_code_leaves_write_tags_of_other_pages() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
        tables.map_4k(
            &mut h.cpu,
            VA_NEXT_PAGE,
            PA_NEXT,
            PTE_PRESENT_WRITABLE_USER | PTE_D,
        );
        h.cpu.translate(VA, AccessKind::Write, false).unwrap();
        h.cpu
            .translate(VA_NEXT_PAGE, AccessKind::Write, false)
            .unwrap();
        h.cpu.mark_code(PA);
        assert_eq!(slot_of(&h.cpu, VA, false).tag_w, TLB_TAG_INVALID);
        assert_eq!(slot_of(&h.cpu, VA_NEXT_PAGE, false).tag_w, VA_NEXT_PAGE);
    }

    #[test]
    fn set_cpl_switches_the_active_tlb_half() {
        let mut h = harness();
        let base = h.cpu.tlb.as_ptr() as u64;
        let half = (TLB_ENTRIES_PER_MODE * core::mem::size_of::<TlbEntry>()) as u64;
        assert_eq!(h.cpu.tlb_active_base, base);
        h.cpu.set_cpl(3);
        assert_eq!(h.cpu.cpl, 3);
        assert_eq!(h.cpu.tlb_active_base, base + half);
        for ring in [0, 1, 2] {
            h.cpu.set_cpl(ring);
            assert_eq!(
                h.cpu.tlb_active_base, base,
                "ring {ring} uses the supervisor half"
            );
        }
    }
}

mod canonical_addresses {
    use super::*;

    #[test]
    fn first_non_canonical_addresses_raise_gp0() {
        let mut h = harness();
        for address in [
            NOT_CANONICAL,
            LAST_NON_CANONICAL,
            0x1234_5678_0000_0000,
            u64::MAX >> 1,
        ] {
            let result = h.cpu.translate(address, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_GENERAL_PROTECTION, Some(0)),
                "{address:#x}"
            );
        }
    }

    #[test]
    fn last_low_and_first_high_canonical_addresses_reach_the_walk() {
        let mut h = harness();
        // The boot identity map only covers low memory, so these fault as not present: a
        // page fault, not the #GP of a non-canonical address.
        for address in [LAST_LOW_CANONICAL, FIRST_HIGH_CANONICAL, u64::MAX] {
            let result = h.cpu.translate(address, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(0)),
                "{address:#x}"
            );
        }
    }

    #[test]
    fn non_canonical_check_applies_to_every_access_kind_and_privilege() {
        let mut h = harness();
        for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
            for supervisor in [false, true] {
                let result = h.cpu.translate(NOT_CANONICAL, access, supervisor);
                assert_eq!(fault_of(result), (VECTOR_GENERAL_PROTECTION, Some(0)));
            }
        }
        h.cpu.set_cpl(3);
        let result = h.cpu.translate(NOT_CANONICAL, AccessKind::Read, false);
        assert_eq!(fault_of(result), (VECTOR_GENERAL_PROTECTION, Some(0)));
    }

    #[test]
    fn non_canonical_fault_does_not_touch_cr2() {
        let mut h = harness();
        h.cpu.cr2 = 0x1111;
        h.cpu
            .translate(NOT_CANONICAL, AccessKind::Read, false)
            .unwrap_err();
        assert_eq!(h.cpu.cr2, 0x1111);
    }

    #[test]
    fn data_accesses_raise_gp0_and_stack_accesses_ss0() {
        let mut h = harness();
        assert_eq!(
            fault_of(h.cpu.read(NOT_CANONICAL, 8)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        assert_eq!(
            fault_of(h.cpu.write(NOT_CANONICAL, 8, 0)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        assert_eq!(
            fault_of(h.cpu.read_stack(NOT_CANONICAL, 8)),
            (VECTOR_STACK_FAULT, Some(0))
        );
        assert_eq!(
            fault_of(h.cpu.write_stack(NOT_CANONICAL, 8, 0)),
            (VECTOR_STACK_FAULT, Some(0))
        );
    }

    #[test]
    fn implicit_system_accesses_to_non_canonical_addresses_raise_gp0() {
        let mut h = harness();
        assert_eq!(
            fault_of(h.cpu.read_sys(NOT_CANONICAL, 8)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        assert_eq!(
            fault_of(h.cpu.write_sys(NOT_CANONICAL, 8, 0)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn stack_accesses_keep_page_faults_as_page_faults() {
        let (mut h, _tables) = paged();
        assert_eq!(
            fault_of(h.cpu.read_stack(VA, 8)),
            (VECTOR_PAGE_FAULT, Some(0))
        );
        assert_eq!(
            fault_of(h.cpu.write_stack(VA, 8, 1)),
            (VECTOR_PAGE_FAULT, Some(PF_W))
        );
    }

    #[test]
    fn stack_accesses_work_like_data_accesses_on_valid_memory() {
        let mut h = harness();
        h.cpu
            .write_stack(SCRATCH + 0x10, 8, 0xfeed_face_cafe_beef)
            .unwrap();
        assert_eq!(h.read_u64(SCRATCH + 0x10), 0xfeed_face_cafe_beef);
        assert_eq!(
            h.cpu.read_stack(SCRATCH + 0x10, 8).unwrap(),
            0xfeed_face_cafe_beef
        );
    }

    #[test]
    fn access_crossing_into_the_non_canonical_hole_faults() {
        let (mut h, mut tables) = paged();
        let last_page = LAST_LOW_CANONICAL & !PAGE_OFFSET_MASK;
        tables.map_4k(&mut h.cpu, last_page, PA, PTE_PRESENT_WRITABLE_USER);
        assert_eq!(
            fault_of(h.cpu.read(LAST_LOW_CANONICAL - 3, 8)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        assert_eq!(
            fault_of(h.cpu.read_stack(LAST_LOW_CANONICAL - 3, 8)),
            (VECTOR_STACK_FAULT, Some(0))
        );
        assert_eq!(
            fault_of(h.cpu.write(LAST_LOW_CANONICAL - 3, 8, 0)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        // Entirely inside the last canonical page is fine.
        h.cpu.read(LAST_LOW_CANONICAL - 7, 8).unwrap();
    }
}

mod data_accesses {
    use super::*;

    fn two_pages() -> Harness {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
        h
    }

    /// Where byte `index` of an access starting `offset` bytes into the first page lives.
    fn physical_of(offset: u64, index: u64) -> u64 {
        let position = offset + index;
        if position < PAGE {
            PA + position
        } else {
            PA_NEXT + (position - PAGE)
        }
    }

    const PATTERN: u64 = 0x8877_6655_4433_2211;

    #[test]
    fn write_then_read_roundtrips_at_every_offset_for_every_size() {
        for size in [1u8, 2, 4, 8] {
            for offset in 0..PAGE {
                // All straddling offsets plus a few interior ones.
                let interesting = offset + size as u64 > PAGE || offset < 3 || offset == PAGE / 2;
                if !interesting {
                    continue;
                }
                let mut h = two_pages();
                let value = PATTERN & mask(size);
                h.cpu.write(VA + offset, size, value).unwrap();
                for index in 0..size as u64 {
                    assert_eq!(
                        h.read(physical_of(offset, index), 1)[0],
                        (value >> (8 * index)) as u8,
                        "size {size} offset {offset:#x} byte {index}"
                    );
                }
                assert_eq!(
                    h.cpu.read(VA + offset, size).unwrap(),
                    value,
                    "size {size} offset {offset:#x} (warm TLB)"
                );
                h.cpu.flush_tlb();
                assert_eq!(
                    h.cpu.read(VA + offset, size).unwrap(),
                    value,
                    "size {size} offset {offset:#x} (cold TLB)"
                );
            }
        }
    }

    #[test]
    fn straddling_write_does_not_touch_bytes_outside_the_access() {
        let mut h = two_pages();
        h.write(PA + PAGE - 8, &[0xee; 8]);
        h.write(PA_NEXT, &[0xdd; 8]);
        h.cpu.write(VA + PAGE - 3, 4, 0x0403_0201).unwrap();
        assert_eq!(
            h.read(PA + PAGE - 8, 8),
            [0xee, 0xee, 0xee, 0xee, 0xee, 1, 2, 3]
        );
        assert_eq!(
            h.read(PA_NEXT, 8),
            [4, 0xdd, 0xdd, 0xdd, 0xdd, 0xdd, 0xdd, 0xdd]
        );
    }

    #[test]
    fn read_of_sizes_one_to_eight_returns_zero_extended_values() {
        let mut h = two_pages();
        h.write(PA, &[0xff; 16]);
        for size in [1u8, 2, 4, 8] {
            assert_eq!(h.cpu.read(VA, size).unwrap(), mask(size), "size {size}");
        }
    }

    #[test]
    fn narrow_write_leaves_neighbouring_bytes() {
        let mut h = two_pages();
        h.write(PA, &[0xaa; 8]);
        h.cpu.write(VA + 2, 2, 0x1234).unwrap();
        assert_eq!(
            h.read(PA, 8),
            [0xaa, 0xaa, 0x34, 0x12, 0xaa, 0xaa, 0xaa, 0xaa]
        );
    }

    #[test]
    fn value_wider_than_the_access_is_truncated_on_write() {
        let mut h = two_pages();
        h.cpu.write(VA, 2, 0xffff_ffff_ffff_1234).unwrap();
        assert_eq!(h.read(PA, 4), [0x34, 0x12, 0, 0]);
    }

    #[test]
    fn sixteen_byte_accesses_roundtrip_at_every_straddling_offset() {
        for offset in (PAGE - 16)..PAGE {
            let mut h = two_pages();
            let value: u128 = 0x1f1e_1d1c_1b1a_1918_1716_1514_1312_1110;
            h.cpu.write128(VA + offset, value).unwrap();
            for index in 0..16u64 {
                assert_eq!(
                    h.read(physical_of(offset, index), 1)[0],
                    0x10 + index as u8,
                    "offset {offset:#x} byte {index}"
                );
            }
            assert_eq!(
                h.cpu.read128(VA + offset).unwrap(),
                value,
                "offset {offset:#x}"
            );
            h.cpu.flush_tlb();
            assert_eq!(
                h.cpu.read128(VA + offset).unwrap(),
                value,
                "cold, offset {offset:#x}"
            );
        }
    }

    #[test]
    fn straddling_read_with_unmapped_second_page_reports_the_second_page_in_cr2() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        assert_eq!(
            fault_of(h.cpu.read(VA + PAGE - 2, 4)),
            (VECTOR_PAGE_FAULT, Some(0))
        );
        assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
    }

    #[test]
    fn straddling_write_with_unmapped_second_page_writes_nothing() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.write(PA + PAGE - 8, &[0x77; 8]);
        assert_eq!(
            fault_of(h.cpu.write(VA + PAGE - 4, 8, u64::MAX)),
            (VECTOR_PAGE_FAULT, Some(PF_W))
        );
        assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
        assert_eq!(h.read(PA + PAGE - 8, 8), [0x77; 8]);
    }

    #[test]
    fn straddling_write_into_a_read_only_second_page_for_user_writes_nothing() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, READ_ONLY_USER);
        h.cpu.set_cpl(3);
        assert_eq!(
            fault_of(h.cpu.write(VA + PAGE - 2, 4, 0xaabb_ccdd)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
        );
        assert_eq!(h.read(PA + PAGE - 2, 2), [0, 0]);
        assert_eq!(h.read(PA_NEXT, 2), [0, 0]);
    }

    #[test]
    fn write128_probes_both_ends_before_storing() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        assert_eq!(
            fault_of(h.cpu.write128(VA + PAGE - 8, u128::MAX)),
            (VECTOR_PAGE_FAULT, Some(PF_W))
        );
        assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
        assert_eq!(h.read(PA + PAGE - 8, 8), [0; 8]);
    }

    #[test]
    fn read128_with_unmapped_second_page_faults() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        assert_eq!(
            fault_of(h.cpu.read128(VA + PAGE - 8 + 1)),
            (VECTOR_PAGE_FAULT, Some(0))
        );
        assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
    }

    #[test]
    fn sys_access_reads_supervisor_pages_from_ring_three_but_data_access_does_not() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
        h.write_u64(PA, 0x0123_4567_89ab_cdef);
        h.cpu.set_cpl(3);
        assert_eq!(
            fault_of(h.cpu.read(VA, 8)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_U))
        );
        assert_eq!(h.cpu.read_sys(VA, 8).unwrap(), 0x0123_4567_89ab_cdef);
        h.cpu.write_sys(VA + 8, 8, 0x42).unwrap();
        assert_eq!(h.read_u64(PA + 8), 0x42);
        assert_eq!(
            fault_of(h.cpu.write(VA + 8, 8, 1)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
        );
    }

    #[test]
    fn user_read_of_a_user_page_succeeds_through_the_user_tlb_half() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.write_u64(PA, 0x55);
        h.cpu.set_cpl(3);
        assert_eq!(h.cpu.read(VA, 8).unwrap(), 0x55);
        assert_eq!(h.cpu.read(VA, 8).unwrap(), 0x55);
    }

    #[test]
    fn stores_set_the_dirty_bit_even_when_the_page_was_read_first() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.cpu.read(VA, 8).unwrap();
        h.cpu.write(VA + 8, 8, 1).unwrap();
        assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), PTE_A | PTE_D);
    }

    #[test]
    fn read_after_remap_and_invlpg_sees_the_new_page() {
        let (mut h, mut tables) = paged();
        let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.write_u64(PA, 1);
        h.write_u64(PA_NEXT, 2);
        assert_eq!(h.cpu.read(VA, 8).unwrap(), 1);
        write_pte(&mut h.cpu, leaf, PA_NEXT | PTE_PRESENT_WRITABLE_USER);
        h.cpu.invlpg(VA);
        assert_eq!(h.cpu.read(VA, 8).unwrap(), 2);
    }
}

mod physical_memory {
    use super::*;

    /// The boot identity map covers 4 GiB, so linear == physical for these.
    const UNMAPPED_A: u64 = 0x4000_0000;
    const UNMAPPED_B: u64 = 0x4000_1000;

    #[test]
    fn unmapped_physical_reads_return_all_ones_for_each_size() {
        let mut h = harness();
        for size in [1u8, 2, 4, 8] {
            assert_eq!(
                h.cpu.read(UNMAPPED_A, size).unwrap(),
                mask(size),
                "size {size}"
            );
        }
    }

    #[test]
    fn unmapped_physical_read_is_logged_once_per_page() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        h.cpu.read(UNMAPPED_A, 4).unwrap();
        h.cpu.read(UNMAPPED_A + 8, 4).unwrap();
        assert_eq!(
            logged(&log),
            ["read of unmapped physical address 0x40000000"]
        );
        h.cpu.read(UNMAPPED_B, 4).unwrap();
        assert_eq!(logged(&log).len(), 2);
    }

    #[test]
    fn unmapped_physical_write_is_dropped_and_logged_once() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        h.cpu.write(UNMAPPED_A, 8, 0x1234).unwrap();
        h.cpu.write(UNMAPPED_A + 8, 8, 0x5678).unwrap();
        assert_eq!(
            logged(&log),
            ["write to unmapped physical address 0x40000000"]
        );
        // Nothing was stored: the page still reads as unmapped.
        assert_eq!(h.cpu.read(UNMAPPED_A, 8).unwrap(), u64::MAX);
    }

    #[test]
    fn unmapped_read_and_write_of_the_same_page_are_logged_separately() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        h.cpu.read(UNMAPPED_A, 1).unwrap();
        h.cpu.write(UNMAPPED_A, 1, 0).unwrap();
        assert_eq!(logged(&log).len(), 2);
    }

    #[test]
    fn log_keys_for_unmapped_reads_and_writes_never_collide() {
        // Page 0x3000 read and page 0x2000 written used to map to the same key
        // (0x1000 + 0x3000 == 0x2000 + 0x2000).
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        h.cpu.read(0x3000 << PAGE_SHIFT, 1).unwrap();
        h.cpu.write(0x2000 << PAGE_SHIFT, 1, 0).unwrap();
        assert_eq!(logged(&log).len(), 2, "{:?}", logged(&log));
    }

    #[test]
    fn log_keys_for_unmapped_reads_do_not_collide_across_pages() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        for page in [0x2000u64, 0x2001, 0x3000, 0x3001, 0x4000] {
            h.cpu.read(page << PAGE_SHIFT, 1).unwrap();
        }
        assert_eq!(logged(&log).len(), 5);
    }

    #[test]
    fn no_log_callback_is_fine() {
        let mut h = harness();
        assert!(h.cpu.log.is_none());
        h.cpu.read(UNMAPPED_A, 8).unwrap();
        h.cpu.write(UNMAPPED_A, 8, 1).unwrap();
    }

    #[test]
    fn last_byte_of_ram_is_ram_and_the_next_byte_is_not() {
        let mut h = harness();
        let log = capture_log(&mut h.cpu);
        h.write(FIRST_UNMAPPED_PA - 1, &[0x5c]);
        assert_eq!(h.cpu.read(FIRST_UNMAPPED_PA - 1, 1).unwrap(), 0x5c);
        assert!(logged(&log).is_empty());
        assert_eq!(h.cpu.read(FIRST_UNMAPPED_PA, 1).unwrap(), 0xff);
        assert_eq!(logged(&log).len(), 1);
    }

    #[test]
    fn straddle_from_ram_into_unmapped_space_combines_both_parts() {
        let mut h = harness();
        h.write(FIRST_UNMAPPED_PA - 2, &[0x11, 0x22]);
        assert_eq!(h.cpu.read(FIRST_UNMAPPED_PA - 2, 4).unwrap(), 0xffff_2211);
    }

    #[test]
    fn straddling_write_from_ram_into_unmapped_space_stores_only_the_ram_part() {
        let mut h = harness();
        h.cpu.write(FIRST_UNMAPPED_PA - 2, 4, 0xaabb_ccdd).unwrap();
        assert_eq!(h.read(FIRST_UNMAPPED_PA - 2, 2), [0xdd, 0xcc]);
    }

    fn enable_virtio_console_bar(h: &mut Harness) {
        // Slot 1 function 0, PCI command register: set memory space enable.
        const CONFIG_ADDRESS: u32 = 0x8000_0000 | (1 << 11) | 0x04;
        h.cpu.dev.io_out(0xcf8, 4, CONFIG_ADDRESS, 0);
        h.cpu.dev.io_out(0xcfc, 2, 2, 0);
    }

    #[test]
    fn mmio_region_is_routed_to_the_device_not_to_ram() {
        let mut h = Harness::with_devices(
            &[HLT],
            DevConfig {
                virtio_console: true,
                ..Default::default()
            },
        );
        let log = capture_log(&mut h.cpu);
        enable_virtio_console_bar(&mut h);
        // device_feature_select is a plain read/write register at offset 0 of the BAR.
        h.cpu.write(0xc000_0000, 4, 1).unwrap();
        assert_eq!(h.cpu.read(0xc000_0000, 4).unwrap(), 1);
        h.cpu.write(0xc000_0000, 4, 0).unwrap();
        assert_eq!(h.cpu.read(0xc000_0000, 4).unwrap(), 0);
        assert!(logged(&log).is_empty(), "{:?}", logged(&log));
    }

    #[test]
    fn mmio_is_unmapped_while_the_device_decoding_is_disabled() {
        let mut h = Harness::with_devices(
            &[HLT],
            DevConfig {
                virtio_console: true,
                ..Default::default()
            },
        );
        let log = capture_log(&mut h.cpu);
        assert_eq!(h.cpu.read(0xc000_0000, 4).unwrap(), 0xffff_ffff);
        assert_eq!(logged(&log).len(), 1);
    }

    #[test]
    fn mmio_accesses_are_never_cached_in_the_tlb() {
        let mut h = Harness::with_devices(
            &[HLT],
            DevConfig {
                virtio_console: true,
                ..Default::default()
            },
        );
        enable_virtio_console_bar(&mut h);
        h.cpu.read(0xc000_0000, 4).unwrap();
        assert_eq!(slot_of(&h.cpu, 0xc000_0000, false).tag_r, TLB_TAG_INVALID);
    }
}

mod code_pages {
    use super::*;

    #[test]
    fn mark_code_flags_the_page_and_records_it_once() {
        let mut h = harness();
        let page = (CODE_BASE >> PAGE_SHIFT) as usize;
        assert_eq!(h.cpu.page_has_code[page], 0);
        h.cpu.mark_code(CODE_BASE + 0x10);
        h.cpu.mark_code(CODE_BASE + 0x20);
        assert_eq!(h.cpu.page_has_code[page], 1);
        assert_eq!(h.cpu.pages_with_code, [page as u32]);
    }

    #[test]
    fn repeatedly_retranslated_code_page_does_not_grow_the_page_list_without_bound() {
        let mut h = harness();
        let pages = h.cpu.page_has_code.len();
        let page = 0x300usize;
        // What `code_written` does under the JIT: clear the flag, keep the list entry.
        for _ in 0..3 * pages {
            h.cpu.mark_code((page as u64) << PAGE_SHIFT);
            h.cpu.page_has_code[page] = 0;
        }
        assert!(
            h.cpu.pages_with_code.len() <= pages,
            "{}",
            h.cpu.pages_with_code.len()
        );
        // Still correct: flushing clears exactly the pages that are marked.
        h.cpu.mark_code((page as u64) << PAGE_SHIFT);
        h.cpu.mark_code(0x400 << PAGE_SHIFT);
        h.cpu.flush_decode();
        assert_eq!(h.cpu.page_has_code[page], 0);
        assert_eq!(h.cpu.page_has_code[0x400], 0);
    }

    #[test]
    fn page_list_compaction_keeps_every_page_that_is_still_marked() {
        let mut h = harness();
        let pages = h.cpu.page_has_code.len();
        for page in 0..(pages as u64 - 1) {
            h.cpu.mark_code(page << PAGE_SHIFT);
        }
        // List is now one short of full; un-flag half of it, then mark more pages to compact.
        for page in (0..pages).step_by(2) {
            h.cpu.page_has_code[page] = 0;
        }
        h.cpu.mark_code(0);
        h.cpu.mark_code(2 << PAGE_SHIFT);
        h.cpu.mark_code((pages as u64 - 1) << PAGE_SHIFT);
        h.cpu.flush_decode();
        assert!(h.cpu.page_has_code.iter().all(|&flag| flag == 0));
    }

    #[test]
    fn mark_code_ignores_addresses_outside_ram() {
        let mut h = harness();
        h.cpu.mark_code(FIRST_UNMAPPED_PA);
        h.cpu.mark_code(u64::MAX);
        assert!(h.cpu.pages_with_code.is_empty());
    }

    #[test]
    fn flush_decode_clears_every_marked_page_and_bumps_the_generation() {
        let mut h = harness();
        h.cpu.mark_code(0x10_0000);
        h.cpu.mark_code(0x20_0000);
        let generation = h.cpu.decode_cache_generation;
        h.cpu.flush_decode();
        assert_eq!(h.cpu.decode_cache_generation, generation + 1);
        assert_eq!(h.cpu.flushes, 1);
        assert!(h.cpu.pages_with_code.is_empty());
        assert_eq!(h.cpu.page_has_code[0x100], 0);
        assert_eq!(h.cpu.page_has_code[0x200], 0);
    }

    #[test]
    fn store_to_a_code_page_flushes_the_decode_cache_without_a_jit() {
        let mut h = harness();
        h.cpu.fetch_decode().unwrap();
        let generation = h.cpu.decode_cache_generation;
        h.cpu.write(CODE_BASE + 0x100, 1, 0x90).unwrap();
        assert_eq!(h.cpu.decode_cache_generation, generation + 1);
        assert_eq!(h.cpu.flushes, 1);
        assert_eq!(h.cpu.page_has_code[(CODE_BASE >> PAGE_SHIFT) as usize], 0);
        assert_eq!(h.read(CODE_BASE + 0x100, 1), [0x90]);
    }

    #[test]
    fn store_to_a_code_page_through_a_warm_tlb_entry_is_still_caught() {
        let mut h = harness();
        // Warm a write tag first, then translate code from the same page.
        h.cpu.write(CODE_BASE + 0x200, 1, 1).unwrap();
        let code_page = CODE_BASE & !PAGE_OFFSET_MASK;
        assert_eq!(slot_of(&h.cpu, CODE_BASE, false).tag_w, code_page);
        h.cpu.fetch_decode().unwrap();
        assert_eq!(slot_of(&h.cpu, CODE_BASE, false).tag_w, TLB_TAG_INVALID);
        h.cpu.write(CODE_BASE + 0x200, 1, 2).unwrap();
        assert_eq!(h.cpu.flushes, 1);
    }

    #[test]
    fn store_to_a_non_code_page_does_not_flush() {
        let mut h = harness();
        h.cpu.fetch_decode().unwrap();
        h.cpu.write(SCRATCH, 8, 1).unwrap();
        assert_eq!(h.cpu.flushes, 0);
    }

    #[test]
    fn code_written_with_a_jit_invalidates_only_that_page_and_requests_a_block_exit() {
        let mut h = harness();
        if !h.enable_jit() {
            return;
        }
        let page = (CODE_BASE >> PAGE_SHIFT) as usize;
        h.cpu.mark_code(CODE_BASE);
        h.cpu.mark_code(SCRATCH);
        let generation = h.cpu.decode_cache_generation;
        h.cpu.write(CODE_BASE + 0x100, 1, 0x90).unwrap();
        assert_eq!(h.cpu.page_has_code[page], 0);
        assert_eq!(h.cpu.page_has_code[(SCRATCH >> PAGE_SHIFT) as usize], 1);
        assert_eq!(h.cpu.decode_cache_generation, generation + 1);
        assert!(h.cpu.block_exit_requested);
        assert_eq!(h.cpu.flushes, 0);
        assert!(h.cpu.jit.is_some(), "the JIT must be put back");
    }

    #[test]
    fn device_writes_to_code_pages_use_the_same_invalidation() {
        let mut h = harness();
        h.cpu.mark_code(CODE_BASE);
        h.cpu.code_written((CODE_BASE >> PAGE_SHIFT) as usize);
        assert_eq!(h.cpu.flushes, 1);
        assert_eq!(h.cpu.page_has_code[(CODE_BASE >> PAGE_SHIFT) as usize], 0);
    }
}

fn code_of_length(len: usize) -> Vec<u8> {
    let mut code = vec![0x66; len - 1];
    code.push(0x90);
    code
}

mod instruction_fetch {
    use super::*;

    // mov rax, 5; hlt
    const MOV_RAX_5: [u8; 8] = [0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0xf4];
    // nop
    const NOP: u8 = 0x90;
    const ONE_BYTE_OP_NOP: u16 = 0x90;
    const OP_MOV_RM_IMM: u16 = 0xc7;
    // 14 operand-size prefixes then nop: exactly 15 bytes.
    const MAX_LENGTH_INSTRUCTION_LEN: usize = 15;

    #[test]
    fn decodes_the_instruction_at_rip() {
        let mut h = harness();
        h.write(CODE_BASE, &MOV_RAX_5);
        let insn = h.cpu.fetch_decode().unwrap();
        assert_eq!(insn.op, OP_MOV_RM_IMM);
        assert_eq!(insn.len, 7);
        assert_eq!(insn.imm, 5);
    }

    #[test]
    fn decode_follows_rip() {
        let mut h = harness();
        h.write(CODE_BASE, &MOV_RAX_5);
        h.cpu.rip = CODE_BASE + 7;
        let insn = h.cpu.fetch_decode().unwrap();
        assert_eq!(insn.op, 0xf4);
        assert_eq!(insn.len, 1);
    }

    #[test]
    fn marks_the_code_page_and_caches_the_decode() {
        let mut h = harness();
        h.write(CODE_BASE, &MOV_RAX_5);
        h.cpu.fetch_decode().unwrap();
        assert_eq!(h.cpu.page_has_code[(CODE_BASE >> PAGE_SHIFT) as usize], 1);
        // Changing RAM behind the CPU's back: the cached decode must still be returned.
        h.write(CODE_BASE, &[NOP]);
        assert_eq!(h.cpu.fetch_decode().unwrap().op, OP_MOV_RM_IMM);
        h.cpu.flush_decode();
        assert_eq!(h.cpu.fetch_decode().unwrap().op, ONE_BYTE_OP_NOP);
    }

    #[test]
    fn cache_entries_that_share_a_slot_do_not_return_each_others_instruction() {
        let mut h = harness();
        // Two physical addresses with the same cache slot but different contents.
        let first_pa = 0x1000u64;
        let second_pa = 0x1_1001u64;
        let slot = |pa: u64| (pa ^ (pa >> 16)) as usize & (DECODE_CACHE_ENTRIES - 1);
        assert_eq!(slot(first_pa), slot(second_pa));
        h.write(first_pa, &MOV_RAX_5);
        h.write(second_pa, &[NOP]);
        h.cpu.rip = first_pa;
        assert_eq!(h.cpu.fetch_decode().unwrap().op, OP_MOV_RM_IMM);
        h.cpu.rip = second_pa;
        assert_eq!(h.cpu.fetch_decode().unwrap().op, ONE_BYTE_OP_NOP);
        h.cpu.rip = first_pa;
        assert_eq!(h.cpu.fetch_decode().unwrap().op, OP_MOV_RM_IMM);
    }

    #[test]
    fn generation_bump_invalidates_even_beyond_the_old_32_bit_range() {
        let mut h = harness();
        h.write(CODE_BASE, &MOV_RAX_5);
        h.cpu.decode_cache_generation = u32::MAX as u64;
        h.cpu.fetch_decode().unwrap();
        h.write(CODE_BASE, &[NOP]);
        h.cpu.flush_decode();
        // A u32 counter would have wrapped to 0 here and matched zeroed entries.
        assert_eq!(h.cpu.decode_cache_generation, u32::MAX as u64 + 1);
        assert_eq!(h.cpu.fetch_decode().unwrap().op, ONE_BYTE_OP_NOP);
    }

    #[test]
    fn fifteen_byte_instruction_is_accepted() {
        let mut h = harness();
        h.write(CODE_BASE, &code_of_length(MAX_LENGTH_INSTRUCTION_LEN));
        assert_eq!(
            h.cpu.fetch_decode().unwrap().len as usize,
            MAX_LENGTH_INSTRUCTION_LEN
        );
    }

    #[test]
    fn sixteen_byte_instruction_raises_gp0() {
        let mut h = harness();
        h.write(CODE_BASE, &code_of_length(MAX_LENGTH_INSTRUCTION_LEN + 1));
        assert_eq!(
            fault_of(h.cpu.fetch_decode()),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn long_instruction_that_decodes_by_itself_is_still_gp0_beyond_fifteen() {
        let mut h = harness();
        // 11 prefixes + mov rax, imm64 (10 bytes) = 21 bytes.
        let mut code = vec![0x66; 11];
        code.extend_from_slice(&[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8]);
        h.write(CODE_BASE, &code);
        assert_eq!(
            fault_of(h.cpu.fetch_decode()),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn undecodable_opcode_is_ud() {
        let mut h = harness();
        // push es is invalid in 64-bit mode.
        h.write(CODE_BASE, &[0x06]);
        assert_eq!(fault_of(h.cpu.fetch_decode()), (6, None));
    }

    #[test]
    fn invalid_opcode_after_many_prefixes_is_ud_not_gp() {
        let mut h = harness();
        let mut code = vec![0x66; 10];
        code.push(0x06);
        h.write(CODE_BASE, &code);
        assert_eq!(fault_of(h.cpu.fetch_decode()), (6, None));
    }

    #[test]
    fn non_canonical_rip_raises_gp0() {
        let mut h = harness();
        h.cpu.rip = NOT_CANONICAL;
        assert_eq!(
            fault_of(h.cpu.fetch_decode()),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn fetch_from_a_page_that_is_not_present_faults_with_cr2_at_rip() {
        let (mut h, _tables) = paged();
        h.cpu.rip = VA + 0x123;
        assert_eq!(fault_of(h.cpu.fetch_decode()), (VECTOR_PAGE_FAULT, Some(0)));
        assert_eq!(h.cpu.cr2, VA + 0x123);
    }

    #[test]
    fn fetch_fault_reports_the_instruction_fetch_bit_only_with_nxe() {
        for nxe in [false, true] {
            let (mut h, _tables) = paged();
            set_nxe(&mut h.cpu, nxe);
            h.cpu.rip = VA;
            let expected = if nxe { PF_ID } else { 0 };
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_PAGE_FAULT, Some(expected)),
                "nxe={nxe}"
            );
        }
    }

    #[test]
    fn fetch_from_a_no_execute_page_faults_with_the_fetch_bit() {
        let (mut h, mut tables) = paged();
        set_nxe(&mut h.cpu, true);
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_NX);
        h.write(PA, &[NOP]);
        h.cpu.rip = VA;
        assert_eq!(
            fault_of(h.cpu.fetch_decode()),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID))
        );
    }

    #[test]
    fn user_fetch_from_a_supervisor_page_faults() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
        h.write(PA, &[NOP]);
        h.cpu.rip = VA;
        h.cpu.set_cpl(3);
        assert_eq!(
            fault_of(h.cpu.fetch_decode()),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_U))
        );
    }

    #[test]
    fn fetch_from_unmapped_physical_space_is_a_not_present_fault() {
        let mut h = harness();
        h.cpu.rip = 0x4000_0000;
        assert_eq!(fault_of(h.cpu.fetch_decode()), (VECTOR_PAGE_FAULT, Some(0)));
        assert_eq!(h.cpu.cr2, 0x4000_0000);
    }

    fn code_split_across_pages(
        h: &mut Harness,
        tables: &mut PageTables,
        code: &[u8],
        first_len: usize,
    ) {
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
        h.write(PA + PAGE - first_len as u64, &code[..first_len]);
        h.write(PA_NEXT, &code[first_len..]);
        h.cpu.rip = VA + PAGE - first_len as u64;
    }

    #[test]
    fn instruction_straddling_two_pages_decodes_from_both() {
        for first_len in 1..7 {
            let (mut h, mut tables) = paged();
            code_split_across_pages(&mut h, &mut tables, &MOV_RAX_5[..7], first_len);
            let insn = h.cpu.fetch_decode().unwrap();
            assert_eq!(insn.len, 7, "first_len {first_len}");
            assert_eq!(insn.imm, 5, "first_len {first_len}");
        }
    }

    #[test]
    fn straddling_instruction_marks_both_pages_as_code_but_is_not_cached() {
        let (mut h, mut tables) = paged();
        code_split_across_pages(&mut h, &mut tables, &MOV_RAX_5[..7], 3);
        h.cpu.fetch_decode().unwrap();
        assert_eq!(h.cpu.page_has_code[(PA >> PAGE_SHIFT) as usize], 1);
        assert_eq!(h.cpu.page_has_code[(PA_NEXT >> PAGE_SHIFT) as usize], 1);
        // Patch the second page behind the CPU's back: no stale cache may be used.
        h.write(PA_NEXT, &[9]);
        assert_eq!(h.cpu.fetch_decode().unwrap().imm, 9);
    }

    #[test]
    fn straddling_instruction_with_unmapped_second_page_faults_on_that_page() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.write(PA + PAGE - 3, &MOV_RAX_5[..3]);
        h.cpu.rip = VA + PAGE - 3;
        assert_eq!(fault_of(h.cpu.fetch_decode()), (VECTOR_PAGE_FAULT, Some(0)));
        assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
    }

    #[test]
    fn straddling_instruction_whose_second_page_is_no_execute_faults() {
        let (mut h, mut tables) = paged();
        set_nxe(&mut h.cpu, true);
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        tables.map_4k(
            &mut h.cpu,
            VA_NEXT_PAGE,
            PA_NEXT,
            PTE_PRESENT_WRITABLE_USER | PTE_NX,
        );
        h.write(PA + PAGE - 3, &MOV_RAX_5[..3]);
        h.cpu.rip = VA + PAGE - 3;
        assert_eq!(
            fault_of(h.cpu.fetch_decode()),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID))
        );
        assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
    }

    #[test]
    fn complete_instruction_at_the_end_of_a_page_does_not_touch_the_next_page() {
        let (mut h, mut tables) = paged();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
        h.write(PA + PAGE - 1, &[NOP]);
        h.cpu.rip = VA + PAGE - 1;
        assert_eq!(h.cpu.fetch_decode().unwrap().len, 1);
        assert_eq!(h.cpu.page_has_code[(PA_NEXT >> PAGE_SHIFT) as usize], 0);
    }

    #[test]
    fn sixteen_byte_instruction_across_a_page_boundary_raises_gp0() {
        let (mut h, mut tables) = paged();
        let code = code_of_length(MAX_LENGTH_INSTRUCTION_LEN + 1);
        code_split_across_pages(&mut h, &mut tables, &code, 8);
        assert_eq!(
            fault_of(h.cpu.fetch_decode()),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn fifteen_byte_instruction_across_a_page_boundary_decodes() {
        let (mut h, mut tables) = paged();
        let code = code_of_length(MAX_LENGTH_INSTRUCTION_LEN);
        code_split_across_pages(&mut h, &mut tables, &code, 8);
        assert_eq!(
            h.cpu.fetch_decode().unwrap().len as usize,
            MAX_LENGTH_INSTRUCTION_LEN
        );
    }

    #[test]
    fn invalid_opcode_across_a_page_boundary_is_ud() {
        let (mut h, mut tables) = paged();
        // 66 66 66 then push es, with the invalid byte on the second page.
        code_split_across_pages(&mut h, &mut tables, &[0x66, 0x66, 0x66, 0x06], 3);
        assert_eq!(fault_of(h.cpu.fetch_decode()), (6, None));
    }
}

mod jit_agreement {
    use super::*;
    use crate::testkit::{Stop, assert_jit_matches_interpreter, run_both};

    const NOP: u8 = 0x90;

    fn with_halt(mut code: Vec<u8>) -> Vec<u8> {
        code.push(HLT);
        code
    }

    #[test]
    fn sixteen_byte_instruction_is_gp0_on_both_engines() {
        let code = with_halt(code_of_length(16));
        let outcome = run_both(&code, |_| {});
        assert_eq!(
            outcome.interpreter_stop,
            Stop::Fault {
                vector: VECTOR_GENERAL_PROTECTION,
                error_code: Some(0),
                software: false
            }
        );
        assert_eq!(outcome.interpreter.cpu.rip, CODE_BASE);
        assert_jit_matches_interpreter(&code, |_| {});
    }

    #[test]
    fn fifteen_byte_instruction_runs_on_both_engines() {
        let code = with_halt(code_of_length(15));
        let outcome = run_both(&code, |_| {});
        assert_eq!(outcome.interpreter_stop, Stop::Halted);
        assert_jit_matches_interpreter(&code, |_| {});
    }

    #[test]
    fn too_long_instruction_after_other_instructions_faults_only_when_reached() {
        let mut code = vec![NOP, NOP];
        code.extend(code_of_length(16));
        let outcome = run_both(&code, |_| {});
        assert_eq!(
            outcome.interpreter_stop,
            Stop::Fault {
                vector: VECTOR_GENERAL_PROTECTION,
                error_code: Some(0),
                software: false
            }
        );
        assert_eq!(outcome.interpreter.cpu.rip, CODE_BASE + 2);
        assert_jit_matches_interpreter(&code, |_| {});
    }

    #[test]
    fn undecodable_opcode_behind_prefixes_is_ud_on_both_engines() {
        // 66 66 66 push es
        let code = [0x66, 0x66, 0x66, 0x06];
        let outcome = run_both(&code, |_| {});
        assert_eq!(
            outcome.interpreter_stop,
            Stop::Fault {
                vector: 6,
                error_code: None,
                software: false
            }
        );
        assert_jit_matches_interpreter(&code, |_| {});
    }
}

mod eflags_helpers {
    use super::*;

    #[test]
    fn eflags_full_always_has_the_reserved_bit_set() {
        let mut h = harness();
        h.cpu.eflags = 0;
        assert_eq!(h.cpu.eflags_full(), RESERVED_ONE);
    }

    #[test]
    fn eflags_full_resolves_the_deferred_flags_over_stale_eflags() {
        let mut h = harness();
        h.cpu.eflags = RESERVED_ONE | CF | ZF | OF | IF;
        // 0x7f + 1 in 8 bits: result 0x80.
        h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0x80, 1, 0);
        assert_eq!(h.cpu.eflags_full(), RESERVED_ONE | IF | SF | OF | AF);
    }

    #[test]
    fn materialize_keeps_the_value_and_drops_the_deferred_state() {
        let mut h = harness();
        h.cpu.eflags = RESERVED_ONE | IF | DF;
        // 0 - 5 in 32 bits: the result is 0xfffffffb.
        h.cpu.set_lazy_flags(LazyFlagsOp::Sub, 4, 0xffff_fffb, 5, 0);
        let before = h.cpu.eflags_full();
        h.cpu.materialize_lazy_flags();
        assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
        assert_eq!(h.cpu.eflags, before);
        assert_eq!(h.cpu.eflags_full(), before);
        assert_eq!(before & (IF | DF), IF | DF);
        assert_eq!(before & ARITH, CF | AF | SF);
    }

    #[test]
    fn set_eflags_arith_replaces_only_the_arithmetic_flags() {
        let mut h = harness();
        h.cpu.eflags = RESERVED_ONE | IF | TF | CF | ZF;
        h.cpu.set_lazy_flags(LazyFlagsOp::Add, 8, 1, 1, 0);
        h.cpu.set_eflags_arith(SF | IF | DF | AC);
        assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
        assert_eq!(h.cpu.eflags_full(), RESERVED_ONE | IF | TF | SF);
    }

    #[test]
    fn cf_reads_through_deferred_flags() {
        let mut h = harness();
        h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
        assert!(h.cpu.cf());
        h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 2, 1, 0);
        assert!(!h.cpu.cf());
        h.cpu.eflags = RESERVED_ONE | CF;
        h.cpu.lazy_op = LazyFlagsOp::Eflags;
        assert!(h.cpu.cf());
    }
}
