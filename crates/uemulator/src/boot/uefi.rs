//! Loading `crates/uemulator-firmware` and the page it is handed. The byte layouts here are
//! the contract with the firmware (`plans/19-uefi-firmware.md`); the firmware crate has the same
//! offsets under tests of its own.

use super::{
    BOOT_CODE_DESCRIPTOR, BOOT_CS, BOOT_DS, BootMode, BootState, Error, GDT_ADDR, PML4, e820_map,
    put, write_gdt_and_page_tables,
};
use crate::dev::kv::{KV_PORT_BASE, KV_PORT_COUNT};
#[cfg(freestanding)]
use crate::prelude::*;

/// Where the image is placed: the first MiB belongs to legacy structures.
pub(crate) const FIRMWARE_BASE: u64 = 0x10_0000;
/// The page the firmware's `rdi` points at. `boot_params` has no use on this path, so its slot is
/// reused.
pub(crate) const HANDOFF_ADDR: u64 = super::BOOT_PARAMS;

const IMAGE_MAGIC: &[u8; 8] = b"UEMUFW64";
const IMAGE_VERSION: u32 = 1;
/// magic, version u32, entry offset u32, footprint u64: the image's first 24 bytes. The
/// footprint is the image plus its zero-initialised data, which the library reserves.
const IMAGE_HEADER_BYTES: usize = 24;
const IMAGE_ENTRY_OFFSET: usize = 12;
const IMAGE_FOOTPRINT_OFFSET: usize = 16;
const MAX_FOOTPRINT: u64 = 64 << 20;

const HANDOFF_MAGIC: &[u8; 8] = b"UEMUHAND";
const HANDOFF_VERSION: u32 = 1;
pub(crate) const HANDOFF_SIZE: usize = 0x308;

const HANDOFF_VERSION_OFFSET: usize = 0x08;
const HANDOFF_SIZE_OFFSET: usize = 0x0c;
const HANDOFF_FIRMWARE_BASE: usize = 0x10;
const HANDOFF_FIRMWARE_SIZE: usize = 0x18;
const HANDOFF_RAM_TOP: usize = 0x20;
const HANDOFF_RSDP: usize = 0x28;
const HANDOFF_ACPI_BASE: usize = 0x30;
const HANDOFF_ACPI_SIZE: usize = 0x38;
const HANDOFF_UNIX_TIME: usize = 0x40;
const HANDOFF_SERIAL_PORT: usize = 0x48;
const HANDOFF_KV_PORT: usize = 0x4a;
const HANDOFF_KV_PORT_COUNT: usize = 0x4c;
const HANDOFF_PCI_CONFIG_PORT: usize = 0x4e;
const HANDOFF_PCI_ECAM_BASE: usize = 0x50;
const HANDOFF_RAM_REGION_COUNT: usize = 0x58;
const HANDOFF_MMIO_REGION_COUNT: usize = 0x5c;
const HANDOFF_RANDOM_SEED: usize = 0x60;
const RANDOM_SEED_BYTES: usize = 32;
const HANDOFF_RAM_REGIONS: usize = 0x80;
/// base u64, size u64, e820 type u32, pad u32.
const RAM_REGION_BYTES: usize = 24;
const MAX_RAM_REGIONS: usize = 16;
const HANDOFF_MMIO_REGIONS: usize = 0x200;
/// base u64, size u64.
const MMIO_REGION_BYTES: usize = 16;
const MAX_MMIO_REGIONS: usize = 16;
const HANDOFF_TSC_HZ: usize = 0x300;
/// `rdtsc` ticks per second: the guest clock is in nanoseconds (`Cpu::tsc`).
const TSC_HZ: u64 = 1_000_000_000;

const SERIAL_PORT: u16 = 0x3f8;
const PCI_CONFIG_ADDRESS_PORT: u16 = 0xcf8;

/// The MMIO the machine decodes, as the firmware reports it in the memory map: the virtio BARs
/// (up to `MAX_VIRTIO_FUNCTIONS` of them), the IOAPIC and the local APIC.
const MMIO_REGIONS: [(u64, u64); 3] = [
    (crate::dev::PCI_MMIO_BASE, crate::dev::PCI_MMIO_BYTES),
    (0xfec0_0000, 0x1000),
    (0xfee0_0000, 0x1000),
];

/// What the machine tells the firmware besides its RAM size.
#[derive(Clone)]
pub(crate) struct FirmwareParams {
    pub key_value_store: bool,
    pub unix_time: u64,
    pub random_seed: u64,
    /// Physical address of the ACPI RSDP, 0 when the machine has no ACPI tables.
    pub rsdp: u64,
    /// The guest RAM the ACPI tables occupy (reported as ACPI reclaim memory), (0, 0) for none.
    pub acpi_region: (u64, u64),
}

struct Handoff(Vec<u8>);

impl Handoff {
    fn put_u16(&mut self, offset: usize, value: u16) {
        self.0[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    fn put_u32(&mut self, offset: usize, value: u32) {
        self.0[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn put_u64(&mut self, offset: usize, value: u64) {
        self.0[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Places the firmware image at `FIRMWARE_BASE`, the handoff page and the boot GDT and
/// identity-mapped page tables, and returns the 64-bit entry state (`rdi` = handoff).
pub(crate) fn load_uefi_firmware(
    ram: &mut [u8],
    image: &[u8],
    params: &FirmwareParams,
) -> Result<BootState, Error> {
    let bad = |message: &str| Error::BadConfig(format!("UEFI firmware image: {message}"));
    if image.len() < IMAGE_HEADER_BYTES || &image[..8] != IMAGE_MAGIC {
        return Err(bad("missing UEMUFW64 header"));
    }
    let word = |offset: usize| u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap());
    if word(8) != IMAGE_VERSION {
        return Err(bad("unsupported header version"));
    }
    let entry = word(IMAGE_ENTRY_OFFSET) as u64;
    let footprint = u64::from_le_bytes(
        image[IMAGE_FOOTPRINT_OFFSET..IMAGE_FOOTPRINT_OFFSET + 8]
            .try_into()
            .unwrap(),
    );
    if entry >= image.len() as u64 || footprint < image.len() as u64 || footprint > MAX_FOOTPRINT {
        return Err(bad("entry or size out of range"));
    }
    let ram_top = ram.len() as u64;
    if FIRMWARE_BASE + footprint > ram_top {
        return Err(Error::BadConfig(
            "guest memory too small for firmware".into(),
        ));
    }
    put(ram, FIRMWARE_BASE, image)?;

    let mut handoff = vec![0u8; HANDOFF_SIZE];
    handoff[..8].copy_from_slice(HANDOFF_MAGIC);
    let mut handoff = Handoff(handoff);
    handoff.put_u16(HANDOFF_SERIAL_PORT, SERIAL_PORT);
    handoff.put_u16(
        HANDOFF_KV_PORT,
        if params.key_value_store {
            KV_PORT_BASE
        } else {
            0
        },
    );
    handoff.put_u16(HANDOFF_KV_PORT_COUNT, KV_PORT_COUNT);
    handoff.put_u16(HANDOFF_PCI_CONFIG_PORT, PCI_CONFIG_ADDRESS_PORT);
    handoff.put_u32(HANDOFF_VERSION_OFFSET, HANDOFF_VERSION);
    handoff.put_u32(HANDOFF_SIZE_OFFSET, HANDOFF_SIZE as u32);
    let regions = e820_map(ram_top);
    debug_assert!(regions.len() <= MAX_RAM_REGIONS && MMIO_REGIONS.len() <= MAX_MMIO_REGIONS);
    handoff.put_u32(HANDOFF_RAM_REGION_COUNT, regions.len() as u32);
    handoff.put_u32(HANDOFF_MMIO_REGION_COUNT, MMIO_REGIONS.len() as u32);
    handoff.put_u64(HANDOFF_FIRMWARE_BASE, FIRMWARE_BASE);
    handoff.put_u64(HANDOFF_FIRMWARE_SIZE, footprint);
    handoff.put_u64(HANDOFF_RAM_TOP, ram_top);
    handoff.put_u64(HANDOFF_RSDP, params.rsdp);
    handoff.put_u64(HANDOFF_ACPI_BASE, params.acpi_region.0);
    handoff.put_u64(HANDOFF_ACPI_SIZE, params.acpi_region.1);
    handoff.put_u64(HANDOFF_UNIX_TIME, params.unix_time);
    handoff.put_u64(HANDOFF_PCI_ECAM_BASE, 0);
    handoff.put_u64(HANDOFF_TSC_HZ, TSC_HZ);
    let mut seed_state = params.random_seed;
    for chunk in 0..RANDOM_SEED_BYTES / 8 {
        handoff.put_u64(HANDOFF_RANDOM_SEED + chunk * 8, splitmix64(&mut seed_state));
    }
    for (index, (base, size, kind)) in regions.iter().enumerate() {
        let at = HANDOFF_RAM_REGIONS + index * RAM_REGION_BYTES;
        handoff.put_u64(at, *base);
        handoff.put_u64(at + 8, *size);
        handoff.put_u32(at + 16, *kind);
    }
    for (index, (base, size)) in MMIO_REGIONS.iter().enumerate() {
        let at = HANDOFF_MMIO_REGIONS + index * MMIO_REGION_BYTES;
        handoff.put_u64(at, *base);
        handoff.put_u64(at + 8, *size);
    }
    let handoff = handoff.0;
    put(ram, HANDOFF_ADDR, &handoff)?;

    let gdt_limit = write_gdt_and_page_tables(ram, BOOT_CODE_DESCRIPTOR)?;
    Ok(BootState {
        rip: FIRMWARE_BASE + entry,
        rdi: HANDOFF_ADDR,
        cr3: PML4,
        gdt_base: GDT_ADDR,
        gdt_limit,
        cs: BOOT_CS,
        ds: BOOT_DS,
        mode: BootMode::Long64,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAM_BYTES: usize = 64 << 20;
    const ENTRY: u32 = 0x40;

    fn image(entry: u32, footprint: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; 0x100];
        bytes[..8].copy_from_slice(IMAGE_MAGIC);
        bytes[8..12].copy_from_slice(&IMAGE_VERSION.to_le_bytes());
        bytes[IMAGE_ENTRY_OFFSET..IMAGE_ENTRY_OFFSET + 4].copy_from_slice(&entry.to_le_bytes());
        bytes[IMAGE_FOOTPRINT_OFFSET..IMAGE_FOOTPRINT_OFFSET + 8]
            .copy_from_slice(&footprint.to_le_bytes());
        bytes[ENTRY as usize] = 0xf4;
        bytes
    }

    fn params() -> FirmwareParams {
        FirmwareParams {
            key_value_store: true,
            unix_time: 1_700_000_000,
            random_seed: 7,
            rsdp: 0,
            acpi_region: (0, 0),
        }
    }

    fn u64_at(ram: &[u8], at: u64) -> u64 {
        u64::from_le_bytes(ram[at as usize..at as usize + 8].try_into().unwrap())
    }

    fn u16_at(ram: &[u8], at: u64) -> u16 {
        u16::from_le_bytes(ram[at as usize..at as usize + 2].try_into().unwrap())
    }

    fn load(image: &[u8], params: &FirmwareParams) -> Result<(BootState, Vec<u8>), Error> {
        let mut ram = vec![0u8; RAM_BYTES];
        load_uefi_firmware(&mut ram, image, params).map(|state| (state, ram))
    }

    #[test]
    fn starts_at_the_entry_in_long_mode_with_rdi_at_the_handoff() {
        let (state, ram) = load(&image(ENTRY, 0x2000), &params()).unwrap();
        assert_eq!(state.rip, FIRMWARE_BASE + ENTRY as u64);
        assert_eq!(state.rdi, HANDOFF_ADDR);
        assert_eq!(state.mode, BootMode::Long64);
        assert_eq!(ram[(FIRMWARE_BASE + ENTRY as u64) as usize], 0xf4);
        assert_eq!(state.cr3, PML4);
    }

    #[test]
    fn the_handoff_describes_the_machine() {
        let (_, ram) = load(&image(ENTRY, 0x2000), &params()).unwrap();
        let at = |offset: usize| HANDOFF_ADDR + offset as u64;
        assert_eq!(&ram[HANDOFF_ADDR as usize..][..8], HANDOFF_MAGIC);
        assert_eq!(u64_at(&ram, at(HANDOFF_FIRMWARE_BASE)), FIRMWARE_BASE);
        assert_eq!(u64_at(&ram, at(HANDOFF_FIRMWARE_SIZE)), 0x2000);
        assert_eq!(u64_at(&ram, at(HANDOFF_RAM_TOP)), RAM_BYTES as u64);
        assert_eq!(u64_at(&ram, at(HANDOFF_UNIX_TIME)), 1_700_000_000);
        assert_eq!(u64_at(&ram, at(HANDOFF_RSDP)), 0);
        assert_eq!(u16_at(&ram, at(HANDOFF_SERIAL_PORT)), 0x3f8);
        assert_eq!(u16_at(&ram, at(HANDOFF_KV_PORT)), KV_PORT_BASE);
        assert_eq!(u16_at(&ram, at(HANDOFF_PCI_CONFIG_PORT)), 0xcf8);
        assert_eq!(u64_at(&ram, at(HANDOFF_RAM_REGIONS)), 0);
        let high = at(HANDOFF_RAM_REGIONS + 2 * RAM_REGION_BYTES);
        assert_eq!(u64_at(&ram, high), 0x10_0000);
        assert_eq!(u64_at(&ram, high + 8), RAM_BYTES as u64 - 0x10_0000);
        assert_eq!(
            u64_at(&ram, at(HANDOFF_MMIO_REGIONS)),
            crate::dev::PCI_MMIO_BASE
        );
    }

    #[test]
    fn no_store_means_no_kv_port() {
        let mut without = params();
        without.key_value_store = false;
        let (_, ram) = load(&image(ENTRY, 0x2000), &without).unwrap();
        assert_eq!(u16_at(&ram, HANDOFF_ADDR + HANDOFF_KV_PORT as u64), 0);
    }

    #[test]
    fn the_random_seed_depends_on_the_configured_seed() {
        let seed_of = |seed| {
            let mut with = params();
            with.random_seed = seed;
            let (_, ram) = load(&image(ENTRY, 0x2000), &with).unwrap();
            ram[(HANDOFF_ADDR as usize + HANDOFF_RANDOM_SEED)..][..RANDOM_SEED_BYTES].to_vec()
        };
        assert_ne!(seed_of(1), seed_of(2));
        assert_eq!(seed_of(1), seed_of(1));
    }

    #[test]
    fn acpi_details_are_passed_through() {
        let mut with = params();
        with.rsdp = 0xe0000;
        with.acpi_region = (0xe0000, 0x4000);
        let (_, ram) = load(&image(ENTRY, 0x2000), &with).unwrap();
        assert_eq!(u64_at(&ram, HANDOFF_ADDR + HANDOFF_RSDP as u64), 0xe0000);
        assert_eq!(
            u64_at(&ram, HANDOFF_ADDR + HANDOFF_ACPI_SIZE as u64),
            0x4000
        );
    }

    #[test]
    fn malformed_images_are_refused() {
        let mut wrong_magic = image(ENTRY, 0x2000);
        wrong_magic[0] = b'X';
        assert!(load(&wrong_magic, &params()).is_err());
        assert!(
            load(&image(0x100, 0x2000), &params()).is_err(),
            "entry past the image"
        );
        assert!(
            load(&image(ENTRY, 0x10), &params()).is_err(),
            "footprint below the image"
        );
        assert!(load(&image(ENTRY, u64::MAX), &params()).is_err());
        assert!(
            load(&image(ENTRY, RAM_BYTES as u64), &params()).is_err(),
            "does not fit"
        );
        assert!(load(&[0; 4], &params()).is_err());
    }
}
