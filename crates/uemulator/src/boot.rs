use crate::machine::Error;
#[cfg(target_os = "uefi")]
use crate::prelude::*;

/// Where a relocatable kernel's protected-mode part is loaded. 16 MiB is the Debian kernel's
/// `pref_address` and a multiple of its 2 MiB `kernel_alignment`; a non-relocatable kernel is
/// loaded at its own `pref_address` instead.
pub(crate) const KERNEL_LOAD: u64 = 0x100_0000;

// The low-memory layout. Everything here sits inside the first e820 usable range (below
// `LOW_RAM_END`) and the ranges must not overlap: GDT 0x6000..0x6020, boot_params
// 0x7000..0x8000, page tables 0x9000..0xf000, command line from 0x20000.
const GDT_ADDR: u64 = 0x6000;
const BOOT_PARAMS: u64 = 0x7000;
const PML4: u64 = 0x9000;
const CMDLINE_ADDR: u64 = 0x2_0000;

/// The conventional 639 KiB end of low RAM in a PC memory map; the e820 map reports the rest of
/// the first MiB as reserved.
const LOW_RAM_END: u64 = 0x9_fc00;
const ONE_MIB: u64 = 0x10_0000;

/// GDT selectors the 64-bit boot protocol requires (`__BOOT_CS` and `__BOOT_DS` in the kernel),
/// i.e. descriptor indices 2 and 3 of `GDT_ADDR`'s table.
const BOOT_CS: u16 = 0x10;
const BOOT_DS: u16 = 0x18;

/// Flat 64-bit code segment: limit 0xfffff with 4 KiB granularity, access 0x9a (present, DPL 0,
/// code, execute/read), flags G|L. `Cpu::new` loads the cached segment state for `cs` with the
/// same access byte; keep the two in sync.
const BOOT_CODE_DESCRIPTOR: u64 = 0x00af_9a00_0000_ffff;
/// Flat data segment: access 0x92 (present, DPL 0, data, read/write), flags G|D/B. Matches the
/// cached `ds`/`es`/`ss` state in `Cpu::new`.
const BOOT_DATA_DESCRIPTOR: u64 = 0x00cf_9200_0000_ffff;

const PAGE_SIZE: u64 = 4096;
const PAGE_MASK: u64 = PAGE_SIZE - 1;
const SECTOR_SIZE: usize = 512;

// Page-table entry bits (SDM Vol. 3A, IA-32e paging). Table pointers are present|writable; leaf entries
// are 2 MiB pages (PS set in a PDE).
const PTE_PRESENT: u64 = 1;
const PTE_WRITABLE: u64 = 2;
const PTE_PAGE_SIZE: u64 = 0x80;
const TABLE_ENTRY_SIZE: usize = 8;
const ENTRIES_PER_TABLE: u64 = 512;
const HUGE_PAGE_SHIFT: u32 = 21;
/// The identity map covers 4 GiB: one PDPT, four page directories of 512 2 MiB pages each. It
/// therefore spans RAM (at most 3 GiB) and the MMIO window at 0xc000_0000 above it.
const IDENTITY_MAP_GIB: u64 = 4;
/// PML4, PDPT, then one page directory per GiB, laid out back to back from `PML4`.
const PAGE_TABLE_PAGES: usize = 2 + IDENTITY_MAP_GIB as usize;

// Offsets into the bzImage's setup header, which is also where `struct boot_params` keeps it
// (Documentation/arch/x86/boot.rst).
const SETUP_SECTS_OFFSET: usize = 0x1f1;
/// Second byte of the header's `jump` field: the length of the header after the jump.
const JUMP_LENGTH_OFFSET: usize = 0x201;
const HEADER_MAGIC_OFFSET: usize = 0x202;
const VERSION_OFFSET: usize = 0x206;
const TYPE_OF_LOADER_OFFSET: usize = 0x210;
const LOADFLAGS_OFFSET: usize = 0x211;
const RAMDISK_IMAGE_OFFSET: usize = 0x218;
const RAMDISK_SIZE_OFFSET: usize = 0x21c;
const CMD_LINE_PTR_OFFSET: usize = 0x228;
const INITRD_ADDR_MAX_OFFSET: usize = 0x22c;
const RELOCATABLE_KERNEL_OFFSET: usize = 0x234;
const XLOADFLAGS_OFFSET: usize = 0x236;
const CMDLINE_SIZE_OFFSET: usize = 0x238;
const PREF_ADDRESS_OFFSET: usize = 0x258;
const INIT_SIZE_OFFSET: usize = 0x260;
/// Past `handover_offset` (0x264, 4 bytes): the shortest image that has every field we read.
const SETUP_HEADER_MIN_LEN: usize = 0x268;
/// `edd_mbr_sig_buffer`: the first `boot_params` field after the setup header.
const SETUP_HEADER_END_LIMIT: usize = 0x290;

const HEADER_MAGIC: &[u8; 4] = b"HdrS";
/// `setup_sects` of 0 means 4 (boot.rst).
const DEFAULT_SETUP_SECTS: u8 = 4;
/// Boot protocol 2.15, what the target kernel speaks (`plans/02-boot.md`); older ones are
/// rejected rather than checked field by field.
const MIN_PROTOCOL_VERSION: u16 = 0x020f;
const LOADFLAGS_LOADED_HIGH: u8 = 1 << 0;
const LOADFLAGS_CAN_USE_HEAP: u8 = 1 << 7;
const XLF_KERNEL_64: u16 = 1 << 0;
/// "Undefined bootloader" in `type_of_loader`.
const LOADER_TYPE_UNDEFINED: u8 = 0xff;
/// A 64-bit kernel enters at this offset into its protected-mode image (`startup_64`), not at
/// the start of it (which is the 32-bit entry).
const ENTRY_64_OFFSET: u64 = 0x200;

// `struct boot_params` (the "zero page") fields.
const BOOT_PARAMS_SIZE: usize = 4096;
const E820_ENTRIES_OFFSET: usize = 0x1e8;
const E820_TABLE_OFFSET: usize = 0x2d0;
/// `struct boot_e820_entry` is packed: addr u64, size u64, type u32.
const E820_ENTRY_SIZE: usize = 20;
const E820_RAM: u32 = 1;
const E820_RESERVED: u32 = 2;

#[derive(Debug, Clone)]
pub(crate) struct SetupHeader {
    pub setup_sects: u8,
    pub version: u16,
    pub loadflags: u8,
    pub xloadflags: u16,
    pub relocatable: bool,
    pub pref_address: u64,
    pub init_size: u32,
    pub cmdline_max: u32,
    pub initrd_addr_max: u32,
}

impl SetupHeader {
    pub fn parse(image: &[u8]) -> Result<Self, Error> {
        let bad = |message: &str| Error::BadKernel(message.to_string());
        if image.len() < SETUP_HEADER_MIN_LEN
            || &image[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 4] != HEADER_MAGIC
        {
            return Err(bad("missing HdrS magic"));
        }
        let u16_at = |offset| u16::from_le_bytes(image[offset..offset + 2].try_into().unwrap());
        let u32_at = |offset| u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap());
        let u64_at = |offset| u64::from_le_bytes(image[offset..offset + 8].try_into().unwrap());
        let header = SetupHeader {
            setup_sects: if image[SETUP_SECTS_OFFSET] == 0 {
                DEFAULT_SETUP_SECTS
            } else {
                image[SETUP_SECTS_OFFSET]
            },
            version: u16_at(VERSION_OFFSET),
            loadflags: image[LOADFLAGS_OFFSET],
            xloadflags: u16_at(XLOADFLAGS_OFFSET),
            relocatable: image[RELOCATABLE_KERNEL_OFFSET] != 0,
            pref_address: u64_at(PREF_ADDRESS_OFFSET),
            init_size: u32_at(INIT_SIZE_OFFSET),
            cmdline_max: u32_at(CMDLINE_SIZE_OFFSET),
            initrd_addr_max: u32_at(INITRD_ADDR_MAX_OFFSET),
        };
        if header.version < MIN_PROTOCOL_VERSION {
            return Err(bad("boot protocol older than 2.15"));
        }
        if header.loadflags & LOADFLAGS_LOADED_HIGH == 0 {
            return Err(bad("not a bzImage (LOADED_HIGH unset)"));
        }
        if header.xloadflags & XLF_KERNEL_64 == 0 {
            return Err(bad("kernel has no 64-bit entry (XLF_KERNEL_64 unset)"));
        }
        if header.protected_offset() >= image.len() {
            return Err(bad("truncated image"));
        }
        Ok(header)
    }

    /// File offset of the protected-mode kernel: the boot sector plus `setup_sects` sectors of
    /// real-mode setup code come first.
    pub fn protected_offset(&self) -> usize {
        (self.setup_sects as usize + 1) * SECTOR_SIZE
    }
}

/// Register state of a 64-bit Linux entry: paging on with `cr3`, the GDT at `gdt_base`, `cs` and
/// `ds` selecting its flat code and data descriptors.
#[derive(Debug, Clone, Copy, Default)]
pub struct BootState {
    pub rip: u64,
    /// Physical address of `boot_params`; the 64-bit boot protocol passes it in `rsi`.
    pub rsi: u64,
    pub cr3: u64,
    pub gdt_base: u64,
    pub gdt_limit: u16,
    pub cs: u16,
    pub ds: u16,
}

fn put(ram: &mut [u8], addr: u64, data: &[u8]) -> Result<(), Error> {
    let start = addr as usize;
    ram.get_mut(start..start + data.len())
        .ok_or_else(|| Error::BadConfig("guest memory too small".into()))?
        .copy_from_slice(data);
    Ok(())
}

/// Places the kernel, initrd, command line, `boot_params`, GDT and page tables in `ram` and
/// returns the register state for the 64-bit entry (Documentation/arch/x86/boot.rst, "64-bit
/// Boot Protocol"; `boot_params` layout follows QEMU's `x86_load_linux`). No firmware runs, so
/// everything the kernel expects a bootloader to have set up is done here.
pub(crate) fn load(
    ram: &mut [u8],
    kernel: &[u8],
    initrd: Option<&[u8]>,
    cmdline: &str,
) -> Result<BootState, Error> {
    let header = SetupHeader::parse(kernel)?;
    // +1 for the NUL; RAM is zeroed so the terminator is already there.
    if cmdline.len() + 1 > header.cmdline_max as usize {
        return Err(Error::BadConfig("kernel command line too long".into()));
    }
    let ram_top = ram.len() as u64;
    let payload = &kernel[header.protected_offset()..];
    let load_addr = if header.relocatable {
        KERNEL_LOAD
    } else {
        header.pref_address
    };
    // `init_size` is the footprint the kernel needs while it decompresses itself, which can
    // exceed the compressed payload; nothing else may live in it.
    let kernel_end = load_addr + (header.init_size as u64).max(payload.len() as u64);
    put(ram, load_addr, payload)?;
    put(ram, CMDLINE_ADDR, cmdline.as_bytes())?;

    let mut boot_params = vec![0u8; BOOT_PARAMS_SIZE];
    // The header's own length byte says where it ends; boot_params has other fields past
    // `SETUP_HEADER_END_LIMIT` that must stay zero.
    let copy_end =
        (HEADER_MAGIC_OFFSET + kernel[JUMP_LENGTH_OFFSET] as usize).min(SETUP_HEADER_END_LIMIT);
    boot_params[SETUP_SECTS_OFFSET..copy_end]
        .copy_from_slice(&kernel[SETUP_SECTS_OFFSET..copy_end]);
    boot_params[TYPE_OF_LOADER_OFFSET] = LOADER_TYPE_UNDEFINED;
    boot_params[LOADFLAGS_OFFSET] = header.loadflags | LOADFLAGS_CAN_USE_HEAP;
    boot_params[CMD_LINE_PTR_OFFSET..CMD_LINE_PTR_OFFSET + 4]
        .copy_from_slice(&(CMDLINE_ADDR as u32).to_le_bytes());

    if let Some(initrd) = initrd {
        // Highest page-aligned spot the kernel can still reach: it cannot read an initrd above
        // `initrd_addr_max`, and RAM may end below that.
        let limit = ram_top.min(header.initrd_addr_max as u64 + 1);
        let addr = (limit - initrd.len() as u64) & !PAGE_MASK;
        if limit < initrd.len() as u64 || addr < kernel_end {
            return Err(Error::BadConfig("guest memory too small for initrd".into()));
        }
        put(ram, addr, initrd)?;
        boot_params[RAMDISK_IMAGE_OFFSET..RAMDISK_IMAGE_OFFSET + 4]
            .copy_from_slice(&(addr as u32).to_le_bytes());
        boot_params[RAMDISK_SIZE_OFFSET..RAMDISK_SIZE_OFFSET + 4]
            .copy_from_slice(&(initrd.len() as u32).to_le_bytes());
    } else if kernel_end > ram_top {
        return Err(Error::BadConfig("guest memory too small for kernel".into()));
    }

    // (base, size, type). The kernel needs the low range usable for its real-mode trampoline.
    let e820: [(u64, u64, u32); 3] = [
        (0, LOW_RAM_END, E820_RAM),
        (LOW_RAM_END, ONE_MIB - LOW_RAM_END, E820_RESERVED),
        (ONE_MIB, ram_top - ONE_MIB, E820_RAM),
    ];
    boot_params[E820_ENTRIES_OFFSET] = e820.len() as u8;
    for (index, (base, size, kind)) in e820.iter().enumerate() {
        let entry = E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
        boot_params[entry..entry + 8].copy_from_slice(&base.to_le_bytes());
        boot_params[entry + 8..entry + 16].copy_from_slice(&size.to_le_bytes());
        boot_params[entry + 16..entry + 20].copy_from_slice(&kind.to_le_bytes());
    }
    put(ram, BOOT_PARAMS, &boot_params)?;

    // Entries 0 and 1 stay null so the code and data descriptors land on the selectors
    // `BOOT_CS` and `BOOT_DS`.
    let gdt: [u64; 4] = [0, 0, BOOT_CODE_DESCRIPTOR, BOOT_DATA_DESCRIPTOR];
    let gdt_bytes: Vec<u8> = gdt.iter().flat_map(|entry| entry.to_le_bytes()).collect();
    put(ram, GDT_ADDR, &gdt_bytes)?;

    let pdpt = PML4 + PAGE_SIZE;
    let first_page_directory = pdpt + PAGE_SIZE;
    let mut tables = vec![0u8; PAGE_TABLE_PAGES * PAGE_SIZE as usize];
    let entry_bytes = |value: u64| value.to_le_bytes();
    tables[0..TABLE_ENTRY_SIZE].copy_from_slice(&entry_bytes(pdpt | PTE_PRESENT | PTE_WRITABLE));
    for gib in 0..IDENTITY_MAP_GIB {
        let page_directory = first_page_directory + gib * PAGE_SIZE;
        let pdpt_entry = (pdpt - PML4) as usize + gib as usize * TABLE_ENTRY_SIZE;
        tables[pdpt_entry..pdpt_entry + TABLE_ENTRY_SIZE]
            .copy_from_slice(&entry_bytes(page_directory | PTE_PRESENT | PTE_WRITABLE));
        for slot in 0..ENTRIES_PER_TABLE {
            let entry = (page_directory - PML4) as usize + slot as usize * TABLE_ENTRY_SIZE;
            let physical = (gib * ENTRIES_PER_TABLE + slot) << HUGE_PAGE_SHIFT;
            tables[entry..entry + TABLE_ENTRY_SIZE].copy_from_slice(&entry_bytes(
                physical | PTE_PRESENT | PTE_WRITABLE | PTE_PAGE_SIZE,
            ));
        }
    }
    put(ram, PML4, &tables)?;

    Ok(BootState {
        rip: load_addr + ENTRY_64_OFFSET,
        rsi: BOOT_PARAMS,
        cr3: PML4,
        gdt_base: GDT_ADDR,
        gdt_limit: (gdt_bytes.len() - 1) as u16,
        cs: BOOT_CS,
        ds: BOOT_DS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads `~/.cache/uemulator/linux` (not the `quick/debian/artifacts` layout the integration
    /// tests use). Tests that need it pass without checking anything when it is missing.
    fn cached_kernel() -> Option<Vec<u8>> {
        let dir = std::env::var("UEMULATOR_CACHE")
            .map(std::path::PathBuf::from)
            .or_else(|_| {
                std::env::var("HOME").map(|h| std::path::Path::new(&h).join(".cache/uemulator"))
            })
            .ok()?;
        std::fs::read(dir.join("linux")).ok()
    }

    #[test]
    fn rejects_garbage() {
        assert!(SetupHeader::parse(&[0u8; BOOT_PARAMS_SIZE]).is_err());
    }

    #[test]
    fn parses_debian_kernel() {
        let Some(k) = cached_kernel() else { return };
        let h = SetupHeader::parse(&k).unwrap();
        assert_eq!(h.version, MIN_PROTOCOL_VERSION);
        assert!(h.relocatable);
        assert_eq!(h.pref_address, KERNEL_LOAD);
        assert_eq!(h.xloadflags & XLF_KERNEL_64, XLF_KERNEL_64);
    }

    #[test]
    fn loads_debian_kernel() {
        let Some(k) = cached_kernel() else { return };
        let mut ram = vec![0u8; 512 << 20];
        let cmdline = "console=ttyS0";
        let state = load(&mut ram, &k, None, cmdline).unwrap();
        assert_eq!(state.rip, KERNEL_LOAD + ENTRY_64_OFFSET);
        assert_eq!(
            &ram[CMDLINE_ADDR as usize..][..cmdline.len()],
            cmdline.as_bytes()
        );
    }
}
