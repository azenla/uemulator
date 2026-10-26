use crate::machine::Error;

pub(crate) const KERNEL_LOAD: u64 = 0x100_0000;
const BOOT_PARAMS: u64 = 0x7000;
const GDT_ADDR: u64 = 0x6000;
const CMDLINE_ADDR: u64 = 0x2_0000;
const PML4: u64 = 0x9000;
const LOW_RAM_END: u64 = 0x9_fc00;
const BOOT_CS: u16 = 0x10;
const BOOT_DS: u16 = 0x18;

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
        let bad = |m: &str| Error::BadKernel(m.to_string());
        if image.len() < 0x268 || &image[0x202..0x206] != b"HdrS" {
            return Err(bad("missing HdrS magic"));
        }
        let u16_at = |o| u16::from_le_bytes(image[o..o + 2].try_into().unwrap());
        let u32_at = |o| u32::from_le_bytes(image[o..o + 4].try_into().unwrap());
        let u64_at = |o| u64::from_le_bytes(image[o..o + 8].try_into().unwrap());
        let h = SetupHeader {
            setup_sects: if image[0x1f1] == 0 { 4 } else { image[0x1f1] },
            version: u16_at(0x206),
            loadflags: image[0x211],
            xloadflags: u16_at(0x236),
            relocatable: image[0x234] != 0,
            pref_address: u64_at(0x258),
            init_size: u32_at(0x260),
            cmdline_max: u32_at(0x238),
            initrd_addr_max: u32_at(0x22c),
        };
        if h.version < 0x020f {
            return Err(bad("boot protocol older than 2.15"));
        }
        if h.loadflags & 1 == 0 {
            return Err(bad("not a bzImage (LOADED_HIGH unset)"));
        }
        if h.xloadflags & 1 == 0 {
            return Err(bad("kernel has no 64-bit entry (XLF_KERNEL_64 unset)"));
        }
        if (h.setup_sects as usize + 1) * 512 >= image.len() {
            return Err(bad("truncated image"));
        }
        Ok(h)
    }

    pub fn protected_offset(&self) -> usize {
        (self.setup_sects as usize + 1) * 512
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub(crate) struct BootState {
    pub rip: u64,
    pub rsi: u64,
    pub cr3: u64,
    pub gdt_base: u64,
    pub gdt_limit: u16,
    pub cs: u16,
    pub ds: u16,
}

fn put(ram: &mut [u8], addr: u64, data: &[u8]) -> Result<(), Error> {
    let a = addr as usize;
    ram.get_mut(a..a + data.len())
        .ok_or_else(|| Error::BadConfig("guest memory too small".into()))?
        .copy_from_slice(data);
    Ok(())
}

pub(crate) fn load(
    ram: &mut [u8],
    kernel: &[u8],
    initrd: Option<&[u8]>,
    cmdline: &str,
) -> Result<BootState, Error> {
    let hdr = SetupHeader::parse(kernel)?;
    if cmdline.len() + 1 > hdr.cmdline_max as usize {
        return Err(Error::BadConfig("kernel command line too long".into()));
    }
    let ram_top = ram.len() as u64;
    let payload = &kernel[hdr.protected_offset()..];
    let load_addr = if hdr.relocatable {
        KERNEL_LOAD
    } else {
        hdr.pref_address
    };
    let kernel_end = load_addr + (hdr.init_size as u64).max(payload.len() as u64);
    put(ram, load_addr, payload)?;
    put(ram, CMDLINE_ADDR, cmdline.as_bytes())?;

    let mut bp = vec![0u8; 4096];
    let copy_end = (0x202 + kernel[0x201] as usize).min(0x290);
    bp[0x1f1..copy_end].copy_from_slice(&kernel[0x1f1..copy_end]);
    bp[0x210] = 0xff;
    bp[0x211] = hdr.loadflags | 0x80;
    bp[0x228..0x22c].copy_from_slice(&(CMDLINE_ADDR as u32).to_le_bytes());

    if let Some(rd) = initrd {
        let limit = ram_top.min(hdr.initrd_addr_max as u64 + 1);
        let addr = (limit - rd.len() as u64) & !0xfff;
        if limit < rd.len() as u64 || addr < kernel_end {
            return Err(Error::BadConfig("guest memory too small for initrd".into()));
        }
        put(ram, addr, rd)?;
        bp[0x218..0x21c].copy_from_slice(&(addr as u32).to_le_bytes());
        bp[0x21c..0x220].copy_from_slice(&(rd.len() as u32).to_le_bytes());
    } else if kernel_end > ram_top {
        return Err(Error::BadConfig("guest memory too small for kernel".into()));
    }

    let e820: [(u64, u64, u32); 3] = [
        (0, LOW_RAM_END, 1),
        (LOW_RAM_END, 0x10_0000 - LOW_RAM_END, 2),
        (0x10_0000, ram_top - 0x10_0000, 1),
    ];
    bp[0x1e8] = e820.len() as u8;
    for (i, (a, s, t)) in e820.iter().enumerate() {
        let o = 0x2d0 + i * 20;
        bp[o..o + 8].copy_from_slice(&a.to_le_bytes());
        bp[o + 8..o + 16].copy_from_slice(&s.to_le_bytes());
        bp[o + 16..o + 20].copy_from_slice(&t.to_le_bytes());
    }
    put(ram, BOOT_PARAMS, &bp)?;

    let gdt: [u64; 4] = [0, 0, 0x00af_9a00_0000_ffff, 0x00cf_9200_0000_ffff];
    let gdt_bytes: Vec<u8> = gdt.iter().flat_map(|e| e.to_le_bytes()).collect();
    put(ram, GDT_ADDR, &gdt_bytes)?;

    let mut pt = vec![0u8; 6 * 4096];
    let ent = |v: u64| v.to_le_bytes();
    pt[0..8].copy_from_slice(&ent((PML4 + 0x1000) | 3));
    for i in 0..4u64 {
        let o = 0x1000 + i as usize * 8;
        pt[o..o + 8].copy_from_slice(&ent((PML4 + 0x2000 + i * 0x1000) | 3));
        for j in 0..512u64 {
            let o = 0x2000 + i as usize * 0x1000 + j as usize * 8;
            let pa = (i * 512 + j) << 21;
            pt[o..o + 8].copy_from_slice(&ent(pa | 0x83));
        }
    }
    put(ram, PML4, &pt)?;

    Ok(BootState {
        rip: load_addr + 0x200,
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
        assert!(SetupHeader::parse(&[0u8; 4096]).is_err());
    }

    #[test]
    fn parses_debian_kernel() {
        let Some(k) = cached_kernel() else { return };
        let h = SetupHeader::parse(&k).unwrap();
        assert_eq!(h.version, 0x020f);
        assert!(h.relocatable);
        assert_eq!(h.pref_address, 0x100_0000);
        assert_eq!(h.xloadflags & 1, 1);
    }

    #[test]
    fn loads_debian_kernel() {
        let Some(k) = cached_kernel() else { return };
        let mut ram = vec![0u8; 512 << 20];
        let st = load(&mut ram, &k, None, "console=ttyS0").unwrap();
        assert_eq!(st.rip, KERNEL_LOAD + 0x200);
        assert_eq!(&ram[CMDLINE_ADDR as usize..][..13], b"console=ttyS0");
    }
}
