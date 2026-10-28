//! Guest page-table builders for the non-long-mode formats (32-bit and PAE), re-exported by
//! `test_support`. Like the rest of it, the encodings are restated rather than borrowed from
//! `mem.rs`.

use super::*;

/// Architectural CR bits restated for the non-long-mode paging tests.
pub(crate) const CR0_PG_BIT: u64 = 1 << 31;
pub(crate) const CR4_PSE_BIT: u64 = 1 << 4;
pub(crate) const CR4_PAE_BIT: u64 = 1 << 5;
pub(crate) const EFER_NXE_BIT: u64 = 1 << 11;
pub(crate) const FOUR_MIB: u64 = 4 << 20;

/// 32-bit (two-level) guest page tables in harness RAM. `new` takes the CPU out of long mode:
/// EFER cleared, CR4 = PSE only, paging on, CR3 pointing at an empty page directory. The
/// entries it writes are 4 bytes wide.
pub(crate) struct LegacyTables {
    pub directory: u64,
    next_free: u64,
}

impl LegacyTables {
    pub fn new(cpu: &mut Cpu) -> Self {
        let mut tables = LegacyTables {
            directory: 0,
            next_free: TABLE_POOL,
        };
        tables.directory = tables.alloc(cpu);
        cpu.efer = 0;
        cpu.cr4 = CR4_PSE_BIT;
        cpu.cr0 |= CR0_PG_BIT;
        cpu.cr3 = tables.directory;
        cpu.flush_tlb();
        tables
    }

    pub fn alloc(&mut self, cpu: &mut Cpu) -> u64 {
        let page = self.next_free;
        assert!(page < TABLE_POOL_END, "table pool exhausted");
        self.next_free += PAGE;
        cpu.ram[page as usize..(page + PAGE) as usize].fill(0);
        page
    }

    pub fn directory_entry_address(&self, linear: u64) -> u64 {
        self.directory + ((linear >> 22) & 0x3ff) * 4
    }

    /// Maps one 4 KiB page; `directory_flags` and `table_flags` are the PDE and PTE flag bits.
    /// Returns the address of the PTE.
    pub fn map_4k(
        &mut self,
        cpu: &mut Cpu,
        linear: u64,
        phys: u64,
        directory_flags: u64,
        table_flags: u64,
    ) -> u64 {
        let pde_address = self.directory_entry_address(linear);
        let pde = read_pte32(cpu, pde_address);
        let table = if pde & PTE_P != 0 {
            pde & 0xffff_f000
        } else {
            let table = self.alloc(cpu);
            write_pte32(cpu, pde_address, table | directory_flags);
            table
        };
        let pte_address = table + ((linear >> 12) & 0x3ff) * 4;
        write_pte32(cpu, pte_address, phys | table_flags);
        pte_address
    }

    /// Maps one 4 MiB page (PS set). `phys` may have bits 39:32 set, which land in PDE bits
    /// 20:13 as PSE-36 defines. Returns the address of the PDE.
    pub fn map_4m(&mut self, cpu: &mut Cpu, linear: u64, phys: u64, flags: u64) -> u64 {
        let address = self.directory_entry_address(linear);
        let encoded = (phys & 0xffc0_0000) | (((phys >> 32) & 0xff) << 13);
        write_pte32(cpu, address, encoded | flags | PTE_PS);
        address
    }

    /// Identity-maps the first `megabytes` MiB (a multiple of 4) with 4 MiB pages.
    pub fn identity_map(&mut self, cpu: &mut Cpu, megabytes: u64) {
        for index in 0..megabytes / 4 {
            let linear = index * FOUR_MIB;
            self.map_4m(cpu, linear, linear, PTE_P | PTE_W | PTE_U);
        }
    }
}

pub(crate) fn read_pte32(cpu: &Cpu, address: u64) -> u64 {
    u32::from_le_bytes(cpu.ram[address as usize..][..4].try_into().unwrap()) as u64
}

pub(crate) fn write_pte32(cpu: &mut Cpu, address: u64, value: u64) {
    cpu.ram[address as usize..][..4].copy_from_slice(&(value as u32).to_le_bytes());
}

/// PAE guest page tables in harness RAM. `new` takes the CPU out of long mode (EFER cleared,
/// CR4 = PAE only, paging on); `latch` copies the PDPT into the CPU as a CR3 write would.
pub(crate) struct PaeTables {
    pub pdpt: u64,
    next_free: u64,
}

impl PaeTables {
    pub fn new(cpu: &mut Cpu) -> Self {
        let mut tables = PaeTables {
            pdpt: 0,
            next_free: TABLE_POOL,
        };
        tables.pdpt = tables.alloc(cpu);
        cpu.efer = 0;
        cpu.cr4 = CR4_PAE_BIT;
        cpu.cr0 |= CR0_PG_BIT;
        cpu.cr3 = tables.pdpt;
        tables.latch(cpu);
        tables
    }

    pub fn alloc(&mut self, cpu: &mut Cpu) -> u64 {
        let page = self.next_free;
        assert!(page < TABLE_POOL_END, "table pool exhausted");
        self.next_free += PAGE;
        cpu.ram[page as usize..(page + PAGE) as usize].fill(0);
        page
    }

    pub fn latch(&self, cpu: &mut Cpu) {
        cpu.pdpte = cpu.read_pdptes(cpu.cr3).expect("valid PDPT");
        cpu.flush_tlb();
    }

    pub fn pdpte_address(&self, linear: u64) -> u64 {
        self.pdpt + ((linear >> 30) & 3) * 8
    }

    /// Like `PageTables::map` one level down: three flag entries (PDPTE, PDE, PTE) make a 4 KiB
    /// page, two (PDPTE, PDE) a 2 MiB leaf. The PDPTE keeps only the present bit of
    /// `flags[0]`, because its other low bits are reserved. Re-latches the PDPT. Returns the
    /// leaf address.
    pub fn map(&mut self, cpu: &mut Cpu, linear: u64, phys: u64, flags: &[u64]) -> u64 {
        assert!((2..=3).contains(&flags.len()));
        let mut table = self.pdpt;
        for (i, &entry_flags) in flags.iter().enumerate() {
            let level = 2 - i as u32;
            let index = (linear >> (12 + 9 * level)) & if level == 2 { 3 } else { 0x1ff };
            let entry_address = table + index * 8;
            if i == flags.len() - 1 {
                let page_size = if level > 0 { PTE_PS } else { 0 };
                write_pte(cpu, entry_address, phys | entry_flags | page_size);
                self.latch(cpu);
                return entry_address;
            }
            let existing = read_pte(cpu, entry_address);
            table = if existing & PTE_P != 0 {
                existing & PTE_ADDRESS_BITS
            } else {
                let next = self.alloc(cpu);
                let pointer_flags = if level == 2 { PTE_P } else { entry_flags };
                write_pte(cpu, entry_address, next | pointer_flags);
                next
            };
        }
        unreachable!()
    }

    pub fn map_4k(&mut self, cpu: &mut Cpu, linear: u64, phys: u64, flags: u64) -> u64 {
        self.map(cpu, linear, phys, &[flags, flags, flags])
    }

    /// Identity-maps the first `megabytes` MiB (a multiple of 2) with 2 MiB pages.
    pub fn identity_map(&mut self, cpu: &mut Cpu, megabytes: u64) {
        for index in 0..megabytes / 2 {
            let linear = index * TWO_MIB;
            self.map(cpu, linear, linear, &[PTE_P, PTE_P | PTE_W | PTE_U]);
        }
    }
}
