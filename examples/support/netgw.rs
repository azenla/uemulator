pub const GW_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
pub const GW_IP: [u8; 4] = [10, 0, 2, 2];

fn csum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    for c in data.chunks(2) {
        sum += u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]) as u32;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

pub fn handle(f: &[u8]) -> Option<Vec<u8>> {
    if f.len() < 14 {
        return None;
    }
    let src: [u8; 6] = f[6..12].try_into().unwrap();
    match u16::from_be_bytes([f[12], f[13]]) {
        0x0806 if f.len() >= 42 && f[20..22] == [0, 1] && f[38..42] == GW_IP => {
            let mut r = Vec::with_capacity(42);
            r.extend_from_slice(&src);
            r.extend_from_slice(&GW_MAC);
            r.extend_from_slice(&[0x08, 0x06, 0, 1, 8, 0, 6, 4, 0, 2]);
            r.extend_from_slice(&GW_MAC);
            r.extend_from_slice(&GW_IP);
            r.extend_from_slice(&f[22..32]);
            Some(r)
        }
        0x0800 if f.len() >= 34 && f[23] == 1 && f[30..34] == GW_IP && f[34] == 8 => {
            let mut r = f.to_vec();
            r[0..6].copy_from_slice(&src);
            r[6..12].copy_from_slice(&GW_MAC);
            let (a, b) = (f[26..30].to_vec(), f[30..34].to_vec());
            r[26..30].copy_from_slice(&b);
            r[30..34].copy_from_slice(&a);
            r[34] = 0;
            r[36] = 0;
            r[37] = 0;
            let c = csum(&r[34..]);
            r[36..38].copy_from_slice(&c.to_be_bytes());
            Some(r)
        }
        _ => None,
    }
}
