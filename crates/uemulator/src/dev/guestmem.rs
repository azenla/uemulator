pub(crate) struct GuestMem<'a> {
    ram: &'a mut [u8],
    code: &'a [u8],
    pub hits: Vec<usize>,
}

impl<'a> GuestMem<'a> {
    pub fn new(ram: &'a mut [u8], code: &'a [u8]) -> Self {
        GuestMem {
            ram,
            code,
            hits: Vec::new(),
        }
    }

    pub fn read(&self, pa: u64, buf: &mut [u8]) -> bool {
        match usize::try_from(pa)
            .ok()
            .and_then(|a| self.ram.get(a..a.checked_add(buf.len())?))
        {
            Some(src) => {
                buf.copy_from_slice(src);
                true
            }
            None => false,
        }
    }

    pub fn write(&mut self, pa: u64, data: &[u8]) -> bool {
        let Ok(a) = usize::try_from(pa) else {
            return false;
        };
        let Some(end) = a.checked_add(data.len()) else {
            return false;
        };
        if end > self.ram.len() || data.is_empty() {
            return end <= self.ram.len();
        }
        for p in (a >> 12)..=((end - 1) >> 12) {
            if self.code.get(p).is_some_and(|&c| c != 0) && !self.hits.contains(&p) {
                self.hits.push(p);
            }
        }
        self.ram[a..end].copy_from_slice(data);
        true
    }

    pub fn u16(&self, pa: u64) -> u16 {
        let mut b = [0u8; 2];
        self.read(pa, &mut b);
        u16::from_le_bytes(b)
    }

    pub fn u32(&self, pa: u64) -> u32 {
        let mut b = [0u8; 4];
        self.read(pa, &mut b);
        u32::from_le_bytes(b)
    }

    pub fn u64(&self, pa: u64) -> u64 {
        let mut b = [0u8; 8];
        self.read(pa, &mut b);
        u64::from_le_bytes(b)
    }

    pub fn put_u16(&mut self, pa: u64, v: u16) {
        self.write(pa, &v.to_le_bytes());
    }

    pub fn put_u32(&mut self, pa: u64, v: u32) {
        self.write(pa, &v.to_le_bytes());
    }
}

#[cfg(test)]
pub(crate) fn test_mem(ram: &mut [u8]) -> GuestMem<'_> {
    GuestMem::new(ram, &[])
}
