//! User-mode network stack (SLIRP-style) behind a virtio-net device: DHCP, DNS forwarding, ICMP
//! echo, and UDP/TCP NAT onto ordinary host sockets. The guest sees 10.0.2.0/24 with the gateway at
//! 10.0.2.2 (which also reaches the host's loopback) and DNS at 10.0.2.3.

use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpStream, UdpSocket};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use uemulator::NetHandle;

const GW_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
const GW_IP: [u8; 4] = [10, 0, 2, 2];
const DNS_IP: [u8; 4] = [10, 0, 2, 3];
const GUEST_IP: [u8; 4] = [10, 0, 2, 15];
const BROADCAST: [u8; 6] = [0xff; 6];

const MSS: usize = 1460;
const WBUF_MAX: usize = 65535;
const OUT_MAX: usize = 512;
const UDP_IDLE: Duration = Duration::from_secs(60);

const FIN: u8 = 0x01;
const SYN: u8 = 0x02;
const RST: u8 = 0x04;
const PSH: u8 = 0x08;
const ACK: u8 = 0x10;

fn sum(mut acc: u32, data: &[u8]) -> u32 {
    for c in data.chunks(2) {
        acc += u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]) as u32;
    }
    acc
}

fn fold(mut acc: u32) -> u16 {
    while acc >> 16 != 0 {
        acc = (acc & 0xffff) + (acc >> 16);
    }
    !(acc as u16)
}

fn l4_csum(src: [u8; 4], dst: [u8; 4], proto: u8, seg: &[u8]) -> u16 {
    let mut acc = sum(0, &src) + sum(0, &dst) + proto as u32 + seg.len() as u32;
    acc = sum(acc, seg);
    fold(acc)
}

fn eth(dst: [u8; 6], src: [u8; 6], ty: u16, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(14 + payload.len());
    f.extend_from_slice(&dst);
    f.extend_from_slice(&src);
    f.extend_from_slice(&ty.to_be_bytes());
    f.extend_from_slice(payload);
    f
}

fn ipv4(id: u16, src: [u8; 4], dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
    let mut p = vec![0x45, 0];
    p.extend_from_slice(&(20 + payload.len() as u16).to_be_bytes());
    p.extend_from_slice(&id.to_be_bytes());
    p.extend_from_slice(&[0x40, 0, 64, proto, 0, 0]);
    p.extend_from_slice(&src);
    p.extend_from_slice(&dst);
    let c = fold(sum(0, &p));
    p[10..12].copy_from_slice(&c.to_be_bytes());
    p.extend_from_slice(payload);
    p
}

fn udp_segment(src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16, data: &[u8]) -> Vec<u8> {
    let mut s = Vec::with_capacity(8 + data.len());
    s.extend_from_slice(&sport.to_be_bytes());
    s.extend_from_slice(&dport.to_be_bytes());
    s.extend_from_slice(&(8 + data.len() as u16).to_be_bytes());
    s.extend_from_slice(&[0, 0]);
    s.extend_from_slice(data);
    let c = match l4_csum(src, dst, 17, &s) {
        0 => 0xffff,
        c => c,
    };
    s[6..8].copy_from_slice(&c.to_be_bytes());
    s
}

struct TcpHdr {
    src: [u8; 4],
    dst: [u8; 4],
    sport: u16,
    dport: u16,
    seq: u32,
    ack: u32,
    flags: u8,
    win: u16,
    mss: bool,
}

fn tcp_segment(h: &TcpHdr, data: &[u8]) -> Vec<u8> {
    let hl = if h.mss { 24 } else { 20 };
    let mut s = Vec::with_capacity(hl + data.len());
    s.extend_from_slice(&h.sport.to_be_bytes());
    s.extend_from_slice(&h.dport.to_be_bytes());
    s.extend_from_slice(&h.seq.to_be_bytes());
    s.extend_from_slice(&h.ack.to_be_bytes());
    s.push((hl as u8 / 4) << 4);
    s.push(h.flags);
    s.extend_from_slice(&h.win.to_be_bytes());
    s.extend_from_slice(&[0, 0, 0, 0]);
    if h.mss {
        s.extend_from_slice(&[2, 4]);
        s.extend_from_slice(&(MSS as u16).to_be_bytes());
    }
    s.extend_from_slice(data);
    let c = l4_csum(h.src, h.dst, 6, &s);
    s[16..18].copy_from_slice(&c.to_be_bytes());
    s
}

struct Ip<'a> {
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    payload: &'a [u8],
}

fn parse_ip(f: &[u8]) -> Option<Ip<'_>> {
    if f.len() < 34 || f[12..14] != [8, 0] || f[14] >> 4 != 4 {
        return None;
    }
    let ihl = (f[14] & 15) as usize * 4;
    let total = u16::from_be_bytes([f[16], f[17]]) as usize;
    let frag = u16::from_be_bytes([f[20], f[21]]) & 0x3fff;
    if ihl < 20 || total < ihl || 14 + total > f.len() || frag != 0 {
        return None;
    }
    Some(Ip {
        src: f[26..30].try_into().ok()?,
        dst: f[30..34].try_into().ok()?,
        proto: f[23],
        payload: &f[14 + ihl..14 + total],
    })
}

fn seq_lt(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) < 0
}

type Key = (u16, [u8; 4], u16);
/// IP protocol, source, destination, payload.
type Frame = (u8, [u8; 4], [u8; 4], Vec<u8>);

fn host_addr(ip: [u8; 4], port: u16) -> SocketAddr {
    let ip = if ip == GW_IP { [127, 0, 0, 1] } else { ip };
    SocketAddr::from((Ipv4Addr::from(ip), port))
}

fn system_dns() -> SocketAddr {
    let conf = std::fs::read_to_string("/etc/resolv.conf").unwrap_or_default();
    conf.lines()
        .filter_map(|l| l.strip_prefix("nameserver"))
        .filter_map(|a| a.trim().parse::<Ipv4Addr>().ok())
        .next()
        .map_or_else(
            || SocketAddr::from(([1, 1, 1, 1], 53)),
            |a| SocketAddr::from((a, 53)),
        )
}

struct UdpFlow {
    sock: UdpSocket,
    guest_ip: [u8; 4],
    last: Instant,
}

enum Conn {
    Connecting,
    Open(TcpStream),
}

struct TcpFlow {
    conn: Conn,
    guest_ip: [u8; 4],
    mss: usize,
    /// Next sequence number expected from the guest.
    rcv_nxt: u32,
    snd_nxt: u32,
    snd_una: u32,
    guest_win: u32,
    wbuf: VecDeque<u8>,
    guest_fin: bool,
    shut: bool,
    fin_sent: bool,
    dead: bool,
}

pub struct UserNet {
    guest_mac: [u8; 6],
    out: VecDeque<Vec<u8>>,
    udp: HashMap<Key, UdpFlow>,
    tcp: HashMap<Key, TcpFlow>,
    dns: SocketAddr,
    ip_id: u16,
    isn: u32,
    connect_tx: Sender<(Key, io::Result<TcpStream>)>,
    connect_rx: Receiver<(Key, io::Result<TcpStream>)>,
}

impl Default for UserNet {
    fn default() -> Self {
        Self::new()
    }
}

impl UserNet {
    pub fn new() -> Self {
        let (connect_tx, connect_rx) = channel();
        let isn = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.subsec_nanos());
        UserNet {
            guest_mac: [0x52, 0x54, 0, 0x12, 0x34, 0x56],
            out: VecDeque::new(),
            udp: HashMap::new(),
            tcp: HashMap::new(),
            dns: system_dns(),
            ip_id: 1,
            isn,
            connect_tx,
            connect_rx,
        }
    }

    fn emit_ip(&mut self, src: [u8; 4], dst: [u8; 4], proto: u8, payload: &[u8]) {
        self.ip_id = self.ip_id.wrapping_add(1);
        let p = ipv4(self.ip_id, src, dst, proto, payload);
        self.out.push_back(eth(self.guest_mac, GW_MAC, 0x0800, &p));
    }

    pub fn input(&mut self, f: &[u8]) {
        if f.len() < 14 {
            return;
        }
        self.guest_mac = f[6..12].try_into().unwrap();
        match u16::from_be_bytes([f[12], f[13]]) {
            0x0806 => self.arp(f),
            0x0800 => {
                if let Some(ip) = parse_ip(f) {
                    match ip.proto {
                        1 => self.icmp(&ip),
                        6 => self.tcp_in(&ip),
                        17 => self.udp_in(&ip),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn arp(&mut self, f: &[u8]) {
        if f.len() < 42 || f[20..22] != [0, 1] {
            return;
        }
        let target = &f[38..42];
        if target[..3] != [10, 0, 2] || target[3] == GUEST_IP[3] {
            return;
        }
        let mut p = vec![0, 1, 8, 0, 6, 4, 0, 2];
        p.extend_from_slice(&GW_MAC);
        p.extend_from_slice(target);
        p.extend_from_slice(&f[22..28]);
        p.extend_from_slice(&f[28..32]);
        self.out.push_back(eth(self.guest_mac, GW_MAC, 0x0806, &p));
    }

    fn icmp(&mut self, ip: &Ip) {
        if ip.payload.len() < 8 || ip.payload[0] != 8 {
            return;
        }
        let mut r = ip.payload.to_vec();
        r[0] = 0;
        r[2..4].copy_from_slice(&[0, 0]);
        let c = fold(sum(0, &r));
        r[2..4].copy_from_slice(&c.to_be_bytes());
        self.emit_ip(ip.dst, ip.src, 1, &r);
    }

    fn udp_in(&mut self, ip: &Ip) {
        let p = ip.payload;
        if p.len() < 8 {
            return;
        }
        let (sport, dport) = (
            u16::from_be_bytes([p[0], p[1]]),
            u16::from_be_bytes([p[2], p[3]]),
        );
        let len = (u16::from_be_bytes([p[4], p[5]]) as usize).clamp(8, p.len());
        let data = &p[8..len];
        if dport == 67 {
            self.dhcp(data);
            return;
        }
        let key = (sport, ip.dst, dport);
        if !self.udp.contains_key(&key) {
            let target = if ip.dst == DNS_IP {
                self.dns
            } else {
                host_addr(ip.dst, dport)
            };
            let Ok(sock) = UdpSocket::bind("0.0.0.0:0") else {
                return;
            };
            if sock.connect(target).is_err() || sock.set_nonblocking(true).is_err() {
                return;
            }
            self.udp.insert(
                key,
                UdpFlow {
                    sock,
                    guest_ip: ip.src,
                    last: Instant::now(),
                },
            );
        }
        let fl = self.udp.get_mut(&key).unwrap();
        fl.last = Instant::now();
        let _ = fl.sock.send(data);
    }

    fn dhcp(&mut self, d: &[u8]) {
        if d.len() < 241 || d[0] != 1 || d[236..240] != [0x63, 0x82, 0x53, 0x63] {
            return;
        }
        let mut ty = 0;
        let mut i = 240;
        while i + 1 < d.len() && d[i] != 255 {
            if d[i] == 0 {
                i += 1;
                continue;
            }
            let l = d[i + 1] as usize;
            if d[i] == 53 && l >= 1 && i + 2 < d.len() {
                ty = d[i + 2];
            }
            i += 2 + l;
        }
        let reply = match ty {
            1 => 2,
            3 => 5,
            _ => return,
        };
        let mut r = vec![0u8; 240];
        r[0] = 2;
        r[1] = 1;
        r[2] = 6;
        r[4..8].copy_from_slice(&d[4..8]);
        r[10..12].copy_from_slice(&d[10..12]);
        r[16..20].copy_from_slice(&GUEST_IP);
        r[20..24].copy_from_slice(&GW_IP);
        r[28..44].copy_from_slice(&d[28..44]);
        r[236..240].copy_from_slice(&[0x63, 0x82, 0x53, 0x63]);
        let opts: &[(u8, &[u8])] = &[
            (53, &[reply]),
            (54, &GW_IP),
            (51, &86400u32.to_be_bytes()),
            (1, &[255, 255, 255, 0]),
            (3, &GW_IP),
            (6, &DNS_IP),
            (28, &[10, 0, 2, 255]),
        ];
        for (code, v) in opts {
            r.push(*code);
            r.push(v.len() as u8);
            r.extend_from_slice(v);
        }
        r.push(255);
        r.resize(300, 0);
        let u = udp_segment(GW_IP, [255; 4], 67, 68, &r);
        self.ip_id = self.ip_id.wrapping_add(1);
        let p = ipv4(self.ip_id, GW_IP, [255; 4], 17, &u);
        self.out.push_back(eth(BROADCAST, GW_MAC, 0x0800, &p));
    }

    fn rst_for(&mut self, ip: &Ip, ports: (u16, u16), seq_ack: (u32, u32), flags: u8, len: usize) {
        let ((sport, dport), (seq, ack)) = (ports, seq_ack);
        if flags & RST != 0 {
            return;
        }
        let (rseq, rack, rflags) = if flags & ACK != 0 {
            (ack, 0, RST)
        } else {
            let n = len as u32 + (flags & SYN != 0) as u32 + (flags & FIN != 0) as u32;
            (0, seq.wrapping_add(n), RST | ACK)
        };
        let h = TcpHdr {
            src: ip.dst,
            dst: ip.src,
            sport: dport,
            dport: sport,
            seq: rseq,
            ack: rack,
            flags: rflags,
            win: 0,
            mss: false,
        };
        let s = tcp_segment(&h, &[]);
        self.emit_ip(ip.dst, ip.src, 6, &s);
    }

    fn tcp_in(&mut self, ip: &Ip) {
        let p = ip.payload;
        if p.len() < 20 {
            return;
        }
        let sport = u16::from_be_bytes([p[0], p[1]]);
        let dport = u16::from_be_bytes([p[2], p[3]]);
        let seq = u32::from_be_bytes(p[4..8].try_into().unwrap());
        let ack = u32::from_be_bytes(p[8..12].try_into().unwrap());
        let off = (p[12] >> 4) as usize * 4;
        let flags = p[13];
        let win = u16::from_be_bytes([p[14], p[15]]) as u32;
        if off < 20 || off > p.len() {
            return;
        }
        let data = &p[off..];
        let key = (sport, ip.dst, dport);

        if !self.tcp.contains_key(&key) {
            if flags & SYN == 0 || flags & ACK != 0 {
                self.rst_for(ip, (sport, dport), (seq, ack), flags, data.len());
                return;
            }
            let mut mss = MSS;
            let mut o = &p[20..off];
            while o.len() >= 2 && o[0] != 0 {
                if o[0] == 1 {
                    o = &o[1..];
                    continue;
                }
                let l = o[1] as usize;
                if l < 2 || l > o.len() {
                    break;
                }
                if o[0] == 2 && l == 4 {
                    mss = mss.min(u16::from_be_bytes([o[2], o[3]]) as usize);
                }
                o = &o[l..];
            }
            self.isn = self.isn.wrapping_add(0x10000);
            self.tcp.insert(
                key,
                TcpFlow {
                    conn: Conn::Connecting,
                    guest_ip: ip.src,
                    mss: mss.max(64),
                    rcv_nxt: seq.wrapping_add(1),
                    snd_nxt: self.isn,
                    snd_una: self.isn,
                    guest_win: win,
                    wbuf: VecDeque::new(),
                    guest_fin: false,
                    shut: false,
                    fin_sent: false,
                    dead: false,
                },
            );
            let target = if ip.dst == DNS_IP {
                self.dns
            } else {
                host_addr(ip.dst, dport)
            };
            let tx = self.connect_tx.clone();
            std::thread::spawn(move || {
                let r = TcpStream::connect_timeout(&target, Duration::from_secs(10));
                let _ = tx.send((key, r));
            });
            return;
        }

        let fl = self.tcp.get_mut(&key).unwrap();
        if flags & RST != 0 {
            fl.dead = true;
            return;
        }
        if !matches!(fl.conn, Conn::Open(_)) || flags & SYN != 0 {
            return;
        }
        if flags & ACK != 0 {
            if !seq_lt(ack, fl.snd_una) && !seq_lt(fl.snd_nxt, ack) {
                fl.snd_una = ack;
            }
            fl.guest_win = win;
        }
        let mut need_ack = false;
        if !data.is_empty() || flags & FIN != 0 {
            need_ack = true;
            let skip = fl.rcv_nxt.wrapping_sub(seq);
            if (skip as i32) >= 0 && (skip as usize) <= data.len() {
                let fresh = &data[skip as usize..];
                fl.wbuf.extend(fresh);
                fl.rcv_nxt = fl.rcv_nxt.wrapping_add(fresh.len() as u32);
                if flags & FIN != 0 && !fl.guest_fin {
                    fl.guest_fin = true;
                    fl.rcv_nxt = fl.rcv_nxt.wrapping_add(1);
                }
            }
        }
        if fl.fin_sent && fl.guest_fin && fl.snd_una == fl.snd_nxt {
            fl.dead = true;
        }
        if need_ack {
            let h = Self::hdr(
                &key,
                (&fl.guest_ip, fl.rcv_nxt, fl.wbuf.len()),
                ACK,
                fl.snd_nxt,
                false,
            );
            let s = tcp_segment(&h, &[]);
            let g = fl.guest_ip;
            self.emit_ip(key.1, g, 6, &s);
        }
    }

    fn hdr(key: &Key, fl: (&[u8; 4], u32, usize), flags: u8, seq: u32, mss: bool) -> TcpHdr {
        TcpHdr {
            src: key.1,
            dst: *fl.0,
            sport: key.2,
            dport: key.0,
            seq,
            ack: fl.1,
            flags,
            win: WBUF_MAX.saturating_sub(fl.2) as u16,
            mss,
        }
    }

    /// Moves data between host sockets and the guest; call regularly.
    pub fn poll(&mut self) {
        while let Ok((key, res)) = self.connect_rx.try_recv() {
            let Some(fl) = self.tcp.get_mut(&key) else {
                continue;
            };
            match res {
                Ok(s) if s.set_nonblocking(true).is_ok() => {
                    let _ = s.set_nodelay(true);
                    fl.conn = Conn::Open(s);
                    let h = Self::hdr(
                        &key,
                        (&fl.guest_ip, fl.rcv_nxt, fl.wbuf.len()),
                        SYN | ACK,
                        fl.snd_nxt,
                        true,
                    );
                    fl.snd_nxt = fl.snd_nxt.wrapping_add(1);
                    let (s, g) = (tcp_segment(&h, &[]), fl.guest_ip);
                    self.emit_ip(key.1, g, 6, &s);
                }
                _ => {
                    let mut h = Self::hdr(
                        &key,
                        (&fl.guest_ip, fl.rcv_nxt, fl.wbuf.len()),
                        RST | ACK,
                        0,
                        false,
                    );
                    h.win = 0;
                    let (s, g) = (tcp_segment(&h, &[]), fl.guest_ip);
                    fl.dead = true;
                    self.emit_ip(key.1, g, 6, &s);
                }
            }
        }

        let mut frames: Vec<Frame> = Vec::new();
        let room = OUT_MAX.saturating_sub(self.out.len());
        let mut budget = room;
        for (key, fl) in self.tcp.iter_mut() {
            let Conn::Open(stream) = &mut fl.conn else {
                continue;
            };
            while !fl.wbuf.is_empty() {
                let (a, _) = fl.wbuf.as_slices();
                match stream.write(a) {
                    Ok(0) => break,
                    Ok(n) => {
                        fl.wbuf.drain(..n);
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => {
                        fl.dead = true;
                        break;
                    }
                }
            }
            if fl.guest_fin && fl.wbuf.is_empty() && !fl.shut {
                let _ = stream.shutdown(Shutdown::Write);
                fl.shut = true;
            }
            let mut buf = [0u8; MSS];
            while !fl.dead && !fl.fin_sent && budget > 0 {
                let inflight = fl.snd_nxt.wrapping_sub(fl.snd_una);
                let avail = fl.guest_win.saturating_sub(inflight) as usize;
                if avail == 0 {
                    break;
                }
                let want = avail.min(fl.mss).min(buf.len());
                match stream.read(&mut buf[..want]) {
                    Ok(0) => {
                        let h = Self::hdr(
                            key,
                            (&fl.guest_ip, fl.rcv_nxt, fl.wbuf.len()),
                            FIN | ACK,
                            fl.snd_nxt,
                            false,
                        );
                        fl.snd_nxt = fl.snd_nxt.wrapping_add(1);
                        fl.fin_sent = true;
                        frames.push((6, key.1, fl.guest_ip, tcp_segment(&h, &[])));
                        budget -= 1;
                    }
                    Ok(n) => {
                        let h = Self::hdr(
                            key,
                            (&fl.guest_ip, fl.rcv_nxt, fl.wbuf.len()),
                            PSH | ACK,
                            fl.snd_nxt,
                            false,
                        );
                        fl.snd_nxt = fl.snd_nxt.wrapping_add(n as u32);
                        frames.push((6, key.1, fl.guest_ip, tcp_segment(&h, &buf[..n])));
                        budget -= 1;
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => {
                        let h = Self::hdr(
                            key,
                            (&fl.guest_ip, fl.rcv_nxt, fl.wbuf.len()),
                            RST | ACK,
                            fl.snd_nxt,
                            false,
                        );
                        frames.push((6, key.1, fl.guest_ip, tcp_segment(&h, &[])));
                        fl.dead = true;
                    }
                }
            }
        }
        self.tcp.retain(|_, f| !f.dead);

        let now = Instant::now();
        let mut buf = [0u8; 65536];
        for (key, fl) in self.udp.iter_mut() {
            while let Ok(n) = fl.sock.recv(&mut buf) {
                fl.last = now;
                let u = udp_segment(key.1, fl.guest_ip, key.2, key.0, &buf[..n]);
                frames.push((17, key.1, fl.guest_ip, u));
            }
        }
        self.udp
            .retain(|_, f| now.duration_since(f.last) < UDP_IDLE);

        for (proto, src, dst, seg) in frames {
            self.emit_ip(src, dst, proto, &seg);
        }
    }

    /// Frames ready for the guest, oldest first.
    pub fn pop(&mut self) -> Option<Vec<u8>> {
        self.out.pop_front()
    }

    pub fn unpop(&mut self, f: Vec<u8>) {
        self.out.push_front(f);
    }
}

pub fn spawn(net: NetHandle) {
    std::thread::spawn(move || {
        let mut un = UserNet::new();
        loop {
            if let Some(f) = net.recv_frame(Duration::from_millis(2)) {
                un.input(&f);
                while let Some(f) = net.try_recv_frame() {
                    un.input(&f);
                }
            }
            un.poll();
            while let Some(f) = un.pop() {
                if !net.send_frame(&f) {
                    un.unpop(f);
                    break;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, UdpSocket};

    const GMAC: [u8; 6] = [0x52, 0x54, 0, 0x12, 0x34, 0x56];

    fn guest_ip_frame(dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
        let p = ipv4(1, GUEST_IP, dst, proto, payload);
        eth(GW_MAC, GMAC, 0x0800, &p)
    }

    fn drain(un: &mut UserNet) -> Vec<Vec<u8>> {
        std::iter::from_fn(|| un.pop()).collect()
    }

    fn wait_for(un: &mut UserNet, mut pred: impl FnMut(&[u8]) -> bool) -> Vec<u8> {
        let end = Instant::now() + Duration::from_secs(5);
        while Instant::now() < end {
            un.poll();
            while let Some(f) = un.pop() {
                if pred(&f) {
                    return f;
                }
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for frame");
    }

    fn dhcp_msg(ty: u8) -> Vec<u8> {
        let mut d = vec![0u8; 240];
        d[0] = 1;
        d[1] = 1;
        d[2] = 6;
        d[4..8].copy_from_slice(&[1, 2, 3, 4]);
        d[28..34].copy_from_slice(&GMAC);
        d[236..240].copy_from_slice(&[0x63, 0x82, 0x53, 0x63]);
        d.extend_from_slice(&[53, 1, ty, 255]);
        d
    }

    #[test]
    fn dhcp_offer_and_ack() {
        let mut un = UserNet::new();
        for (req, reply) in [(1u8, 2u8), (3, 5)] {
            let u = udp_segment([0; 4], [255; 4], 68, 67, &dhcp_msg(req));
            let p = ipv4(1, [0; 4], [255; 4], 17, &u);
            un.input(&eth(BROADCAST, GMAC, 0x0800, &p));
            let out = drain(&mut un);
            assert_eq!(out.len(), 1);
            let ip = parse_ip(&out[0]).unwrap();
            assert_eq!(ip.proto, 17);
            let d = &ip.payload[8..];
            assert_eq!(d[0], 2);
            assert_eq!(d[4..8], [1, 2, 3, 4]);
            assert_eq!(d[16..20], GUEST_IP);
            assert_eq!(d[240..243], [53, 1, reply]);
        }
    }

    #[test]
    fn arp_and_ping() {
        let mut un = UserNet::new();
        let mut arp = vec![0, 1, 8, 0, 6, 4, 0, 1];
        arp.extend_from_slice(&GMAC);
        arp.extend_from_slice(&GUEST_IP);
        arp.extend_from_slice(&[0; 6]);
        arp.extend_from_slice(&GW_IP);
        un.input(&eth(BROADCAST, GMAC, 0x0806, &arp));
        let out = drain(&mut un);
        assert_eq!(out[0][20..22], [0, 2]);
        assert_eq!(out[0][22..28], GW_MAC);

        let mut echo = vec![8, 0, 0, 0, 0, 1, 0, 1, 9, 9];
        let c = fold(sum(0, &echo));
        echo[2..4].copy_from_slice(&c.to_be_bytes());
        un.input(&guest_ip_frame([8, 8, 8, 8], 1, &echo));
        let out = drain(&mut un);
        let ip = parse_ip(&out[0]).unwrap();
        assert_eq!((ip.src, ip.dst, ip.payload[0]), ([8, 8, 8, 8], GUEST_IP, 0));
        assert_eq!(fold(sum(0, ip.payload)), 0);
    }

    #[test]
    fn udp_nat_roundtrip() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let port = server.local_addr().unwrap().port();
        let mut un = UserNet::new();
        let u = udp_segment(GUEST_IP, [127, 0, 0, 1], 4000, port, b"ping");
        un.input(&guest_ip_frame([127, 0, 0, 1], 17, &u));
        let mut buf = [0u8; 64];
        let (n, from) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"ping");
        server.send_to(b"pong", from).unwrap();
        let f = wait_for(&mut un, |f| parse_ip(f).is_some_and(|i| i.proto == 17));
        let ip = parse_ip(&f).unwrap();
        assert_eq!(ip.src, [127, 0, 0, 1]);
        assert_eq!(u16::from_be_bytes([ip.payload[0], ip.payload[1]]), port);
        assert_eq!(u16::from_be_bytes([ip.payload[2], ip.payload[3]]), 4000);
        assert_eq!(&ip.payload[8..], b"pong");
        assert_eq!(l4_csum(ip.src, ip.dst, 17, ip.payload), 0);
    }

    struct Guest {
        sport: u16,
        dport: u16,
        seq: u32,
        ack: u32,
    }

    impl Guest {
        fn send(&mut self, un: &mut UserNet, flags: u8, data: &[u8]) {
            let h = TcpHdr {
                src: GUEST_IP,
                dst: [127, 0, 0, 1],
                sport: self.sport,
                dport: self.dport,
                seq: self.seq,
                ack: self.ack,
                flags,
                win: 60000,
                mss: false,
            };
            let s = tcp_segment(&h, data);
            un.input(&guest_ip_frame([127, 0, 0, 1], 6, &s));
            self.seq = self
                .seq
                .wrapping_add(data.len() as u32 + (flags & (SYN | FIN) != 0) as u32);
        }
    }

    fn tcp_of(f: &[u8]) -> Option<(u8, u32, u32, Vec<u8>)> {
        let ip = parse_ip(f)?;
        if ip.proto != 6 {
            return None;
        }
        let p = ip.payload;
        assert_eq!(l4_csum(ip.src, ip.dst, 6, p), 0, "bad tcp checksum");
        let off = (p[12] >> 4) as usize * 4;
        Some((
            p[13],
            u32::from_be_bytes(p[4..8].try_into().unwrap()),
            u32::from_be_bytes(p[8..12].try_into().unwrap()),
            p[off..].to_vec(),
        ))
    }

    #[test]
    fn tcp_proxy_echo_and_close() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            loop {
                match s.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => s.write_all(&buf[..n]).unwrap(),
                }
            }
        });
        let mut un = UserNet::new();
        let mut g = Guest {
            sport: 5000,
            dport: port,
            seq: 1000,
            ack: 0,
        };
        g.send(&mut un, SYN, &[]);
        let f = wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| t.0 == SYN | ACK));
        let (_, srv_seq, ack, _) = tcp_of(&f).unwrap();
        assert_eq!(ack, 1001);
        g.ack = srv_seq.wrapping_add(1);
        g.send(&mut un, ACK, &[]);

        let payload: Vec<u8> = (0..5000u32).map(|i| i as u8).collect();
        g.send(&mut un, ACK | PSH, &payload);
        let mut echoed = Vec::new();
        let mut next = g.ack;
        while echoed.len() < payload.len() {
            let f = wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| !t.3.is_empty()));
            let (_, seq, _, data) = tcp_of(&f).unwrap();
            assert_eq!(seq, next, "in-order delivery");
            next = next.wrapping_add(data.len() as u32);
            echoed.extend_from_slice(&data);
            g.ack = next;
            g.send(&mut un, ACK, &[]);
        }
        assert_eq!(echoed, payload);

        g.send(&mut un, FIN | ACK, &[]);
        let f = wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| t.0 & FIN != 0));
        let (_, seq, _, _) = tcp_of(&f).unwrap();
        g.ack = seq.wrapping_add(1);
        g.send(&mut un, ACK, &[]);
        un.poll();
        assert!(un.tcp.is_empty(), "flow closed after both FINs are acked");
    }

    #[test]
    fn tcp_refused_gets_rst() {
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let mut un = UserNet::new();
        let mut g = Guest {
            sport: 5001,
            dport: port,
            seq: 7,
            ack: 0,
        };
        g.send(&mut un, SYN, &[]);
        wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| t.0 & RST != 0));
    }
}
