//! User-mode network stack (SLIRP-style) behind a virtio-net device: DHCP, DNS forwarding, ICMP
//! echo, and UDP/TCP NAT onto ordinary host sockets. The guest sees 10.0.2.0/24 with the gateway at
//! 10.0.2.2 (which also reaches the host's loopback) and DNS at 10.0.2.3.
//!
//! TCP is a terminating proxy, not a packet forwarder: each guest connection becomes a host
//! `TcpStream`. The virtual link never drops or reorders frames, so there is no retransmission
//! timer, no congestion control, no window scaling, no SACK/timestamps, no urgent data and no
//! TIME_WAIT. Out-of-order guest segments are dropped and answered with a duplicate ACK. IPv4
//! fragments, IPv6 and checksums on received packets are not handled or verified.

use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpStream, UdpSocket};
use std::ops::Range;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use uemulator::NetHandle;

use crate::run::GUEST_MAC;

/// 52:55 followed by the gateway IP, the way QEMU's slirp derives its MAC.
pub(crate) const GATEWAY_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
/// Also the DHCP server and default router. Connections to it are redirected to host loopback.
pub(crate) const GATEWAY_IP: [u8; 4] = [10, 0, 2, 2];
const DNS_IP: [u8; 4] = [10, 0, 2, 3];
/// The only address DHCP hands out.
const GUEST_IP: [u8; 4] = [10, 0, 2, 15];
const SUBNET_PREFIX: [u8; 3] = [10, 0, 2];
const SUBNET_MASK: [u8; 4] = [255, 255, 255, 0];
const SUBNET_BROADCAST: [u8; 4] = [10, 0, 2, 255];
const LIMITED_BROADCAST_IP: [u8; 4] = [255; 4];
const BROADCAST_MAC: [u8; 6] = [0xff; 6];
const LOOPBACK_IP: [u8; 4] = [127, 0, 0, 1];
/// Used when the host has no readable `/etc/resolv.conf` (always the case on Windows).
const FALLBACK_DNS: [u8; 4] = [1, 1, 1, 1];
const DNS_PORT: u16 = 53;

/// Largest TCP payload per segment we send: Ethernet MTU 1500 minus 20-byte IPv4 and 20-byte TCP
/// headers. Also advertised in our SYN-ACK.
const MSS: usize = 1460;
/// A guest that advertises a tiny (or zero) MSS would make the per-read size 0, and a zero-length
/// read is indistinguishable from EOF.
const MIN_MSS: usize = 64;
/// Largest window we can advertise without window scaling (16-bit field). The advertised window
/// shrinks by the bytes still waiting to be written to the host socket, which throttles the guest.
const ADVERTISED_WINDOW_MAX: usize = 65535;
/// `poll` stops reading host sockets once this many frames are already queued for the guest, so a
/// fast host sender cannot grow `out` without bound while the guest is slow to take frames.
const OUT_QUEUE_LIMIT: usize = 512;
const UDP_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
const TCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Longest wait for a guest frame in the worker loop, which is also how often host sockets are
/// polled (they are nonblocking and have no wakeup path into this thread).
const POLL_INTERVAL: Duration = Duration::from_millis(2);
/// Step between initial sequence numbers of successive connections. Only needs to keep them
/// distinct; this is not a security boundary.
const ISN_STEP: u32 = 0x10000;
const UDP_RECV_BUF_LEN: usize = 65536;

const ETH_HEADER_LEN: usize = 14;
const ETH_SRC_MAC: Range<usize> = 6..12;
const ETH_TYPE: Range<usize> = 12..14;
const ETHERTYPE_IPV4: u16 = 0x0800;
const ETHERTYPE_ARP: u16 = 0x0806;

const IP_PROTO_ICMP: u8 = 1;
const IP_PROTO_TCP: u8 = 6;
const IP_PROTO_UDP: u8 = 17;

/// Offsets inside the IPv4 header (after the Ethernet header). Only fixed-size headers (IHL 5)
/// are generated.
const IPV4_HEADER_LEN: usize = 20;
const IPV4_VERSION: u8 = 4;
/// Version 4 in the high nibble, header length in 32-bit words (5) in the low one.
const IPV4_VERSION_IHL_NO_OPTIONS: u8 = 0x45;
const IPV4_TOTAL_LEN: Range<usize> = 2..4;
const IPV4_FLAGS_FRAGMENT: Range<usize> = 6..8;
const IPV4_PROTOCOL: usize = 9;
const IPV4_CHECKSUM: Range<usize> = 10..12;
const IPV4_SRC: Range<usize> = 12..16;
const IPV4_DST: Range<usize> = 16..20;
/// "More fragments" flag plus the fragment offset. The reserved bit and Don't Fragment (0x4000)
/// are excluded so DF packets still pass.
const IPV4_FRAGMENT_MASK: u16 = 0x3fff;
const IPV4_FLAG_DONT_FRAGMENT: u16 = 0x4000;
const IPV4_TTL: u8 = 64;

/// Offsets inside the ARP packet (after the Ethernet header).
const ARP_PACKET_LEN: usize = 28;
const ARP_OPCODE: Range<usize> = 6..8;
const ARP_SENDER_MAC: Range<usize> = 8..14;
const ARP_SENDER_IP: Range<usize> = 14..18;
const ARP_TARGET_IP: Range<usize> = 24..28;
/// Hardware type Ethernet (1), protocol type IPv4 (0x0800), address lengths 6 and 4.
const ARP_ETHERNET_IPV4: [u8; 6] = [0, 1, 8, 0, 6, 4];
const ARP_OP_REQUEST: u16 = 1;
const ARP_OP_REPLY: u16 = 2;

const ICMP_ECHO_REPLY: u8 = 0;
const ICMP_ECHO_REQUEST: u8 = 8;
/// Type, code, checksum, identifier, sequence.
const ICMP_ECHO_HEADER_LEN: usize = 8;
const ICMP_CHECKSUM: Range<usize> = 2..4;

const UDP_HEADER_LEN: usize = 8;
const UDP_SRC_PORT: Range<usize> = 0..2;
const UDP_DST_PORT: Range<usize> = 2..4;
const UDP_LENGTH: Range<usize> = 4..6;
const UDP_CHECKSUM: Range<usize> = 6..8;

const TCP_HEADER_LEN: usize = 20;
const TCP_SRC_PORT: Range<usize> = 0..2;
const TCP_DST_PORT: Range<usize> = 2..4;
const TCP_SEQ: Range<usize> = 4..8;
const TCP_ACK: Range<usize> = 8..12;
/// High nibble is the header length in 32-bit words.
const TCP_DATA_OFFSET: usize = 12;
const TCP_FLAGS: usize = 13;
const TCP_WINDOW: Range<usize> = 14..16;
const TCP_CHECKSUM: Range<usize> = 16..18;
const TCP_OPT_END: u8 = 0;
const TCP_OPT_NOP: u8 = 1;
const TCP_OPT_MSS: u8 = 2;
const TCP_OPT_MSS_LEN: usize = 4;

const FIN: u8 = 0x01;
const SYN: u8 = 0x02;
const RST: u8 = 0x04;
const PSH: u8 = 0x08;
const ACK: u8 = 0x10;

const DHCP_SERVER_PORT: u16 = 67;
const DHCP_CLIENT_PORT: u16 = 68;
/// Offsets inside the BOOTP/DHCP message (RFC 2131 section 2).
const BOOTP_OP: usize = 0;
const BOOTP_HTYPE: usize = 1;
const BOOTP_HLEN: usize = 2;
const BOOTP_XID: Range<usize> = 4..8;
const BOOTP_FLAGS: Range<usize> = 10..12;
const BOOTP_YIADDR: Range<usize> = 16..20;
const BOOTP_SIADDR: Range<usize> = 20..24;
const BOOTP_CHADDR: Range<usize> = 28..44;
const BOOTP_MAGIC_COOKIE: Range<usize> = 236..240;
const DHCP_OPTIONS_OFFSET: usize = 240;
const DHCP_MAGIC_COOKIE: [u8; 4] = [0x63, 0x82, 0x53, 0x63];
const BOOTP_REQUEST: u8 = 1;
const BOOTP_REPLY: u8 = 2;
const HTYPE_ETHERNET: u8 = 1;
const HLEN_ETHERNET: u8 = 6;
/// BOOTP replies are padded to the 300 bytes of the original fixed-format packet (RFC 951) for
/// the benefit of old clients.
const BOOTP_MIN_LEN: usize = 300;
const DHCP_OPT_PAD: u8 = 0;
const DHCP_OPT_SUBNET_MASK: u8 = 1;
const DHCP_OPT_ROUTER: u8 = 3;
const DHCP_OPT_DNS_SERVER: u8 = 6;
const DHCP_OPT_BROADCAST_ADDRESS: u8 = 28;
const DHCP_OPT_LEASE_TIME: u8 = 51;
const DHCP_OPT_MESSAGE_TYPE: u8 = 53;
const DHCP_OPT_SERVER_ID: u8 = 54;
const DHCP_OPT_END: u8 = 255;
const DHCP_DISCOVER: u8 = 1;
const DHCP_OFFER: u8 = 2;
const DHCP_REQUEST: u8 = 3;
const DHCP_ACK: u8 = 5;
const DHCP_LEASE_SECS: u32 = 24 * 60 * 60;

/// Adds `data` as big-endian 16-bit words (a trailing odd byte is padded with zero) into a
/// running ones-complement sum. See `checksum_finish`.
fn checksum_add(mut acc: u32, data: &[u8]) -> u32 {
    for pair in data.chunks(2) {
        acc += u16::from_be_bytes([pair[0], *pair.get(1).unwrap_or(&0)]) as u32;
    }
    acc
}

/// Folds the carries back in and complements: the Internet checksum (RFC 1071). A packet whose
/// stored checksum is included in the sum folds to 0.
fn checksum_finish(mut acc: u32) -> u16 {
    while acc >> 16 != 0 {
        acc = (acc & 0xffff) + (acc >> 16);
    }
    !(acc as u16)
}

pub(crate) fn internet_checksum(data: &[u8]) -> u16 {
    checksum_finish(checksum_add(0, data))
}

/// TCP/UDP checksum: covers the IPv4 pseudo-header (addresses, protocol, segment length) too.
fn transport_checksum(src: [u8; 4], dst: [u8; 4], proto: u8, segment: &[u8]) -> u16 {
    let mut acc =
        checksum_add(0, &src) + checksum_add(0, &dst) + proto as u32 + segment.len() as u32;
    acc = checksum_add(acc, segment);
    checksum_finish(acc)
}

fn be_u16(bytes: &[u8]) -> u16 {
    u16::from_be_bytes([bytes[0], bytes[1]])
}

fn be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn ethernet_frame(dst: [u8; 6], src: [u8; 6], ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(ETH_HEADER_LEN + payload.len());
    frame.extend_from_slice(&dst);
    frame.extend_from_slice(&src);
    frame.extend_from_slice(&ethertype.to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

fn ipv4_packet(id: u16, src: [u8; 4], dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
    let mut packet = vec![IPV4_VERSION_IHL_NO_OPTIONS, 0];
    packet.extend_from_slice(&(IPV4_HEADER_LEN as u16 + payload.len() as u16).to_be_bytes());
    packet.extend_from_slice(&id.to_be_bytes());
    packet.extend_from_slice(&IPV4_FLAG_DONT_FRAGMENT.to_be_bytes());
    packet.push(IPV4_TTL);
    packet.push(proto);
    packet.extend_from_slice(&[0, 0]);
    packet.extend_from_slice(&src);
    packet.extend_from_slice(&dst);
    let checksum = internet_checksum(&packet);
    packet[IPV4_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
    packet.extend_from_slice(payload);
    packet
}

fn udp_segment(src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16, data: &[u8]) -> Vec<u8> {
    let mut segment = Vec::with_capacity(UDP_HEADER_LEN + data.len());
    segment.extend_from_slice(&sport.to_be_bytes());
    segment.extend_from_slice(&dport.to_be_bytes());
    segment.extend_from_slice(&(UDP_HEADER_LEN as u16 + data.len() as u16).to_be_bytes());
    segment.extend_from_slice(&[0, 0]);
    segment.extend_from_slice(data);
    // A computed checksum of 0 is sent as 0xffff because 0 means "no checksum" in UDP (RFC 768).
    let checksum = match transport_checksum(src, dst, IP_PROTO_UDP, &segment) {
        0 => 0xffff,
        checksum => checksum,
    };
    segment[UDP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
    segment
}

#[derive(Clone, Copy)]
struct TcpHeader {
    src: [u8; 4],
    dst: [u8; 4],
    sport: u16,
    dport: u16,
    seq: u32,
    ack: u32,
    flags: u8,
    window: u16,
    /// Append the MSS option; only SYN segments carry it.
    advertise_mss: bool,
}

fn tcp_segment(header: &TcpHeader, data: &[u8]) -> Vec<u8> {
    let header_len = TCP_HEADER_LEN
        + if header.advertise_mss {
            TCP_OPT_MSS_LEN
        } else {
            0
        };
    let mut segment = Vec::with_capacity(header_len + data.len());
    segment.extend_from_slice(&header.sport.to_be_bytes());
    segment.extend_from_slice(&header.dport.to_be_bytes());
    segment.extend_from_slice(&header.seq.to_be_bytes());
    segment.extend_from_slice(&header.ack.to_be_bytes());
    segment.push((header_len as u8 / 4) << 4);
    segment.push(header.flags);
    segment.extend_from_slice(&header.window.to_be_bytes());
    // Checksum placeholder and urgent pointer.
    segment.extend_from_slice(&[0, 0, 0, 0]);
    if header.advertise_mss {
        segment.extend_from_slice(&[TCP_OPT_MSS, TCP_OPT_MSS_LEN as u8]);
        segment.extend_from_slice(&(MSS as u16).to_be_bytes());
    }
    segment.extend_from_slice(data);
    let checksum = transport_checksum(header.src, header.dst, IP_PROTO_TCP, &segment);
    segment[TCP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
    segment
}

struct Ipv4Packet<'a> {
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    payload: &'a [u8],
}

/// Parses an Ethernet frame carrying an unfragmented IPv4 packet. Trailing Ethernet padding is
/// cut off using the IPv4 total length.
fn parse_ipv4(frame: &[u8]) -> Option<Ipv4Packet<'_>> {
    if frame.len() < ETH_HEADER_LEN + IPV4_HEADER_LEN
        || frame[ETH_TYPE] != ETHERTYPE_IPV4.to_be_bytes()
    {
        return None;
    }
    let ip = &frame[ETH_HEADER_LEN..];
    if ip[0] >> 4 != IPV4_VERSION {
        return None;
    }
    let header_len = (ip[0] & 0xf) as usize * 4;
    let total_len = be_u16(&ip[IPV4_TOTAL_LEN]) as usize;
    let fragment = be_u16(&ip[IPV4_FLAGS_FRAGMENT]) & IPV4_FRAGMENT_MASK;
    if header_len < IPV4_HEADER_LEN
        || total_len < header_len
        || total_len > ip.len()
        || fragment != 0
    {
        return None;
    }
    Some(Ipv4Packet {
        src: ip[IPV4_SRC].try_into().ok()?,
        dst: ip[IPV4_DST].try_into().ok()?,
        proto: ip[IPV4_PROTOCOL],
        payload: &ip[header_len..total_len],
    })
}

struct TcpSegment<'a> {
    sport: u16,
    dport: u16,
    seq: u32,
    ack: u32,
    flags: u8,
    window: u32,
    options: &'a [u8],
    data: &'a [u8],
}

impl<'a> TcpSegment<'a> {
    fn parse(payload: &'a [u8]) -> Option<Self> {
        if payload.len() < TCP_HEADER_LEN {
            return None;
        }
        let header_len = (payload[TCP_DATA_OFFSET] >> 4) as usize * 4;
        if header_len < TCP_HEADER_LEN || header_len > payload.len() {
            return None;
        }
        Some(TcpSegment {
            sport: be_u16(&payload[TCP_SRC_PORT]),
            dport: be_u16(&payload[TCP_DST_PORT]),
            seq: be_u32(&payload[TCP_SEQ]),
            ack: be_u32(&payload[TCP_ACK]),
            flags: payload[TCP_FLAGS],
            window: be_u16(&payload[TCP_WINDOW]) as u32,
            options: &payload[TCP_HEADER_LEN..header_len],
            data: &payload[header_len..],
        })
    }
}

/// MSS to use towards the guest: our own `MSS` lowered to what the guest's SYN options ask for.
fn negotiated_mss(options: &[u8]) -> usize {
    let mut mss = MSS;
    let mut rest = options;
    while rest.len() >= 2 && rest[0] != TCP_OPT_END {
        if rest[0] == TCP_OPT_NOP {
            rest = &rest[1..];
            continue;
        }
        let len = rest[1] as usize;
        if len < 2 || len > rest.len() {
            break;
        }
        if rest[0] == TCP_OPT_MSS && len == TCP_OPT_MSS_LEN {
            mss = mss.min(be_u16(&rest[2..4]) as usize);
        }
        rest = &rest[len..];
    }
    mss.max(MIN_MSS)
}

/// Modular sequence-number comparison (RFC 793 section 3.3): true when `a` is before `b`.
fn seq_before(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) < 0
}

/// Flow identity from the guest's point of view.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FlowKey {
    guest_port: u16,
    remote_ip: [u8; 4],
    remote_port: u16,
}

/// An IP payload waiting to be wrapped and queued by `emit_ip`. Collected first because the flow
/// tables are borrowed while polling.
struct PendingPacket {
    proto: u8,
    src: [u8; 4],
    dst: [u8; 4],
    payload: Vec<u8>,
}

/// The gateway address stands for the host itself.
fn host_addr(ip: [u8; 4], port: u16) -> SocketAddr {
    let ip = if ip == GATEWAY_IP { LOOPBACK_IP } else { ip };
    SocketAddr::from((Ipv4Addr::from(ip), port))
}

/// Where traffic for `DNS_IP` goes: the host's first IPv4 nameserver.
fn system_dns() -> SocketAddr {
    let conf = std::fs::read_to_string("/etc/resolv.conf").unwrap_or_default();
    conf.lines()
        .filter_map(|line| line.strip_prefix("nameserver"))
        .filter_map(|addr| addr.trim().parse::<Ipv4Addr>().ok())
        .next()
        .map_or_else(
            || SocketAddr::from((FALLBACK_DNS, DNS_PORT)),
            |addr| SocketAddr::from((addr, DNS_PORT)),
        )
}

struct UdpFlow {
    /// Connected to the NAT target, so only its replies are received.
    sock: UdpSocket,
    guest_ip: [u8; 4],
    last_activity: Instant,
}

enum Conn {
    /// The host `connect` runs on a worker thread; the result arrives through `connect_rx`.
    Connecting,
    Open(TcpStream),
}

/// Sequence variable names follow RFC 793: `snd_*` is our (host side) send direction towards the
/// guest, `rcv_*` is what we have received from the guest.
struct TcpFlow {
    conn: Conn,
    guest_ip: [u8; 4],
    mss: usize,
    /// Next sequence number expected from the guest.
    rcv_nxt: u32,
    /// Next sequence number we will send; a SYN or FIN consumes one.
    snd_nxt: u32,
    /// Oldest sequence number the guest has not acknowledged yet.
    snd_una: u32,
    /// Window the guest last advertised. Never scaled, because our SYN-ACK has no window scale
    /// option and the guest therefore cannot use one either.
    guest_window: u32,
    /// Guest data not yet accepted by the host socket.
    to_host: VecDeque<u8>,
    guest_fin: bool,
    /// The host socket's write side has been shut down after the guest's FIN.
    host_write_shutdown: bool,
    fin_sent: bool,
    /// Removed at the end of the next `poll`.
    dead: bool,
}

/// Window to advertise given how many guest bytes are still buffered for the host socket.
fn advertised_window(buffered: usize) -> u16 {
    ADVERTISED_WINDOW_MAX.saturating_sub(buffered) as u16
}

impl TcpFlow {
    fn header(&self, key: &FlowKey, flags: u8, seq: u32, advertise_mss: bool) -> TcpHeader {
        TcpHeader {
            src: key.remote_ip,
            dst: self.guest_ip,
            sport: key.remote_port,
            dport: key.guest_port,
            seq,
            ack: self.rcv_nxt,
            flags,
            window: advertised_window(self.to_host.len()),
            advertise_mss,
        }
    }
}

pub struct UserNet {
    /// Learned from the source MAC of every guest frame; frames the stack originates (replies,
    /// host data) go to it.
    guest_mac: [u8; 6],
    /// Frames queued for the guest, oldest first.
    out: VecDeque<Vec<u8>>,
    udp: HashMap<FlowKey, UdpFlow>,
    tcp: HashMap<FlowKey, TcpFlow>,
    dns: SocketAddr,
    ip_id: u16,
    /// Initial sequence number of the most recent connection.
    isn: u32,
    connect_tx: Sender<(FlowKey, io::Result<TcpStream>)>,
    connect_rx: Receiver<(FlowKey, io::Result<TcpStream>)>,
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
            guest_mac: GUEST_MAC,
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
        let packet = ipv4_packet(self.ip_id, src, dst, proto, payload);
        self.out.push_back(ethernet_frame(
            self.guest_mac,
            GATEWAY_MAC,
            ETHERTYPE_IPV4,
            &packet,
        ));
    }

    pub fn input(&mut self, frame: &[u8]) {
        if frame.len() < ETH_HEADER_LEN {
            return;
        }
        self.guest_mac = frame[ETH_SRC_MAC].try_into().unwrap();
        match be_u16(&frame[ETH_TYPE]) {
            ETHERTYPE_ARP => self.arp(frame),
            ETHERTYPE_IPV4 => {
                if let Some(ip) = parse_ipv4(frame) {
                    match ip.proto {
                        IP_PROTO_ICMP => self.icmp(&ip),
                        IP_PROTO_TCP => self.tcp_in(&ip),
                        IP_PROTO_UDP => self.udp_in(&ip),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    /// Answers requests for any address in the subnet except the guest's own, so the gateway, DNS
    /// and every NAT destination resolve to `GATEWAY_MAC`. Skipping the guest's address keeps its
    /// duplicate-address probe from being answered as a conflict.
    fn arp(&mut self, frame: &[u8]) {
        if frame.len() < ETH_HEADER_LEN + ARP_PACKET_LEN {
            return;
        }
        let arp = &frame[ETH_HEADER_LEN..];
        if be_u16(&arp[ARP_OPCODE]) != ARP_OP_REQUEST {
            return;
        }
        let target_ip = &arp[ARP_TARGET_IP];
        if target_ip[..3] != SUBNET_PREFIX || target_ip[3] == GUEST_IP[3] {
            return;
        }
        let mut reply = ARP_ETHERNET_IPV4.to_vec();
        reply.extend_from_slice(&ARP_OP_REPLY.to_be_bytes());
        reply.extend_from_slice(&GATEWAY_MAC);
        reply.extend_from_slice(target_ip);
        reply.extend_from_slice(&arp[ARP_SENDER_MAC]);
        reply.extend_from_slice(&arp[ARP_SENDER_IP]);
        self.out.push_back(ethernet_frame(
            self.guest_mac,
            GATEWAY_MAC,
            ETHERTYPE_ARP,
            &reply,
        ));
    }

    /// Echo requests to any destination are answered locally and never leave the host, so ping
    /// only proves the virtual link works.
    fn icmp(&mut self, ip: &Ipv4Packet) {
        if ip.payload.len() < ICMP_ECHO_HEADER_LEN || ip.payload[0] != ICMP_ECHO_REQUEST {
            return;
        }
        let mut reply = ip.payload.to_vec();
        reply[0] = ICMP_ECHO_REPLY;
        reply[ICMP_CHECKSUM].copy_from_slice(&[0, 0]);
        let checksum = internet_checksum(&reply);
        reply[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
        self.emit_ip(ip.dst, ip.src, IP_PROTO_ICMP, &reply);
    }

    fn udp_in(&mut self, ip: &Ipv4Packet) {
        let datagram = ip.payload;
        if datagram.len() < UDP_HEADER_LEN {
            return;
        }
        let sport = be_u16(&datagram[UDP_SRC_PORT]);
        let dport = be_u16(&datagram[UDP_DST_PORT]);
        let len = (be_u16(&datagram[UDP_LENGTH]) as usize).clamp(UDP_HEADER_LEN, datagram.len());
        let data = &datagram[UDP_HEADER_LEN..len];
        if dport == DHCP_SERVER_PORT {
            self.dhcp(data);
            return;
        }
        let key = FlowKey {
            guest_port: sport,
            remote_ip: ip.dst,
            remote_port: dport,
        };
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
                    last_activity: Instant::now(),
                },
            );
        }
        let flow = self.udp.get_mut(&key).unwrap();
        flow.last_activity = Instant::now();
        let _ = flow.sock.send(data);
    }

    /// Hands out `GUEST_IP` to anyone: DISCOVER gets an OFFER and REQUEST gets an ACK without
    /// checking the requested address. Other message types (RELEASE, INFORM, DECLINE) are
    /// ignored. Replies are always broadcast because the client may not have an address yet.
    fn dhcp(&mut self, request: &[u8]) {
        if request.len() <= DHCP_OPTIONS_OFFSET
            || request[BOOTP_OP] != BOOTP_REQUEST
            || request[BOOTP_MAGIC_COOKIE] != DHCP_MAGIC_COOKIE
        {
            return;
        }
        let mut message_type = 0;
        let mut pos = DHCP_OPTIONS_OFFSET;
        while pos + 1 < request.len() && request[pos] != DHCP_OPT_END {
            if request[pos] == DHCP_OPT_PAD {
                pos += 1;
                continue;
            }
            let option_len = request[pos + 1] as usize;
            if request[pos] == DHCP_OPT_MESSAGE_TYPE && option_len >= 1 && pos + 2 < request.len() {
                message_type = request[pos + 2];
            }
            pos += 2 + option_len;
        }
        let reply_type = match message_type {
            DHCP_DISCOVER => DHCP_OFFER,
            DHCP_REQUEST => DHCP_ACK,
            _ => return,
        };
        let mut reply = vec![0u8; DHCP_OPTIONS_OFFSET];
        reply[BOOTP_OP] = BOOTP_REPLY;
        reply[BOOTP_HTYPE] = HTYPE_ETHERNET;
        reply[BOOTP_HLEN] = HLEN_ETHERNET;
        reply[BOOTP_XID].copy_from_slice(&request[BOOTP_XID]);
        reply[BOOTP_FLAGS].copy_from_slice(&request[BOOTP_FLAGS]);
        reply[BOOTP_YIADDR].copy_from_slice(&GUEST_IP);
        reply[BOOTP_SIADDR].copy_from_slice(&GATEWAY_IP);
        reply[BOOTP_CHADDR].copy_from_slice(&request[BOOTP_CHADDR]);
        reply[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
        let options: &[(u8, &[u8])] = &[
            (DHCP_OPT_MESSAGE_TYPE, &[reply_type]),
            (DHCP_OPT_SERVER_ID, &GATEWAY_IP),
            (DHCP_OPT_LEASE_TIME, &DHCP_LEASE_SECS.to_be_bytes()),
            (DHCP_OPT_SUBNET_MASK, &SUBNET_MASK),
            (DHCP_OPT_ROUTER, &GATEWAY_IP),
            (DHCP_OPT_DNS_SERVER, &DNS_IP),
            (DHCP_OPT_BROADCAST_ADDRESS, &SUBNET_BROADCAST),
        ];
        for (code, value) in options {
            reply.push(*code);
            reply.push(value.len() as u8);
            reply.extend_from_slice(value);
        }
        reply.push(DHCP_OPT_END);
        reply.resize(BOOTP_MIN_LEN, 0);
        let datagram = udp_segment(
            GATEWAY_IP,
            LIMITED_BROADCAST_IP,
            DHCP_SERVER_PORT,
            DHCP_CLIENT_PORT,
            &reply,
        );
        self.ip_id = self.ip_id.wrapping_add(1);
        let packet = ipv4_packet(
            self.ip_id,
            GATEWAY_IP,
            LIMITED_BROADCAST_IP,
            IP_PROTO_UDP,
            &datagram,
        );
        self.out.push_back(ethernet_frame(
            BROADCAST_MAC,
            GATEWAY_MAC,
            ETHERTYPE_IPV4,
            &packet,
        ));
    }

    /// Reset for a segment that matches no flow (RFC 793 section 3.4, "reset generation"). Never
    /// answers a RST.
    fn send_reset(&mut self, ip: &Ipv4Packet, segment: &TcpSegment) {
        if segment.flags & RST != 0 {
            return;
        }
        let (seq, ack, flags) = if segment.flags & ACK != 0 {
            (segment.ack, 0, RST)
        } else {
            // SYN and FIN each occupy one sequence number.
            let seq_len = segment.data.len() as u32
                + (segment.flags & SYN != 0) as u32
                + (segment.flags & FIN != 0) as u32;
            (0, segment.seq.wrapping_add(seq_len), RST | ACK)
        };
        let header = TcpHeader {
            src: ip.dst,
            dst: ip.src,
            sport: segment.dport,
            dport: segment.sport,
            seq,
            ack,
            flags,
            window: 0,
            advertise_mss: false,
        };
        let reply = tcp_segment(&header, &[]);
        self.emit_ip(ip.dst, ip.src, IP_PROTO_TCP, &reply);
    }

    fn tcp_in(&mut self, ip: &Ipv4Packet) {
        let Some(segment) = TcpSegment::parse(ip.payload) else {
            return;
        };
        let key = FlowKey {
            guest_port: segment.sport,
            remote_ip: ip.dst,
            remote_port: segment.dport,
        };

        if !self.tcp.contains_key(&key) {
            if segment.flags & SYN == 0 || segment.flags & ACK != 0 {
                self.send_reset(ip, &segment);
                return;
            }
            self.isn = self.isn.wrapping_add(ISN_STEP);
            self.tcp.insert(
                key,
                TcpFlow {
                    conn: Conn::Connecting,
                    guest_ip: ip.src,
                    mss: negotiated_mss(segment.options),
                    rcv_nxt: segment.seq.wrapping_add(1),
                    snd_nxt: self.isn,
                    snd_una: self.isn,
                    guest_window: segment.window,
                    to_host: VecDeque::new(),
                    guest_fin: false,
                    host_write_shutdown: false,
                    fin_sent: false,
                    dead: false,
                },
            );
            let target = if ip.dst == DNS_IP {
                self.dns
            } else {
                host_addr(ip.dst, segment.dport)
            };
            let connect_tx = self.connect_tx.clone();
            std::thread::spawn(move || {
                let result = TcpStream::connect_timeout(&target, TCP_CONNECT_TIMEOUT);
                let _ = connect_tx.send((key, result));
            });
            return;
        }

        let flow = self.tcp.get_mut(&key).unwrap();
        if segment.flags & RST != 0 {
            flow.dead = true;
            return;
        }
        // A repeated SYN while connecting is a guest retransmit; the SYN-ACK follows once the host
        // connect finishes.
        if !matches!(flow.conn, Conn::Open(_)) || segment.flags & SYN != 0 {
            return;
        }
        if segment.flags & ACK != 0 {
            if !seq_before(segment.ack, flow.snd_una) && !seq_before(flow.snd_nxt, segment.ack) {
                flow.snd_una = segment.ack;
            }
            flow.guest_window = segment.window;
        }
        let mut need_ack = false;
        if !segment.data.is_empty() || segment.flags & FIN != 0 {
            need_ack = true;
            // Bytes at the start of the segment we already have (guest retransmit). A segment
            // starting beyond `rcv_nxt` leaves a gap and is dropped, so `rcv_nxt` in the ACK
            // below asks for the missing data again.
            let already_received = flow.rcv_nxt.wrapping_sub(segment.seq);
            if (already_received as i32) >= 0 && (already_received as usize) <= segment.data.len() {
                let fresh = &segment.data[already_received as usize..];
                flow.to_host.extend(fresh);
                flow.rcv_nxt = flow.rcv_nxt.wrapping_add(fresh.len() as u32);
                if segment.flags & FIN != 0 && !flow.guest_fin {
                    flow.guest_fin = true;
                    flow.rcv_nxt = flow.rcv_nxt.wrapping_add(1);
                }
            }
        }
        // Both directions closed and our FIN acknowledged: nothing left to wait for.
        if flow.fin_sent && flow.guest_fin && flow.snd_una == flow.snd_nxt {
            flow.dead = true;
        }
        if need_ack {
            let header = flow.header(&key, ACK, flow.snd_nxt, false);
            let reply = tcp_segment(&header, &[]);
            let guest_ip = flow.guest_ip;
            self.emit_ip(key.remote_ip, guest_ip, IP_PROTO_TCP, &reply);
        }
    }

    /// Moves data between host sockets and the guest; call regularly.
    pub fn poll(&mut self) {
        while let Ok((key, result)) = self.connect_rx.try_recv() {
            let Some(flow) = self.tcp.get_mut(&key) else {
                continue;
            };
            match result {
                Ok(stream) if stream.set_nonblocking(true).is_ok() => {
                    let _ = stream.set_nodelay(true);
                    flow.conn = Conn::Open(stream);
                    let header = flow.header(&key, SYN | ACK, flow.snd_nxt, true);
                    flow.snd_nxt = flow.snd_nxt.wrapping_add(1);
                    let (reply, guest_ip) = (tcp_segment(&header, &[]), flow.guest_ip);
                    self.emit_ip(key.remote_ip, guest_ip, IP_PROTO_TCP, &reply);
                }
                _ => {
                    // Host connect failed (refused, unreachable or timed out): reset the guest's SYN.
                    let mut header = flow.header(&key, RST | ACK, 0, false);
                    header.window = 0;
                    let (reply, guest_ip) = (tcp_segment(&header, &[]), flow.guest_ip);
                    flow.dead = true;
                    self.emit_ip(key.remote_ip, guest_ip, IP_PROTO_TCP, &reply);
                }
            }
        }

        let mut packets: Vec<PendingPacket> = Vec::new();
        let mut budget = OUT_QUEUE_LIMIT.saturating_sub(self.out.len());
        for (key, flow) in self.tcp.iter_mut() {
            let Conn::Open(stream) = &mut flow.conn else {
                continue;
            };
            while !flow.to_host.is_empty() {
                let (contiguous, _) = flow.to_host.as_slices();
                match stream.write(contiguous) {
                    Ok(0) => break,
                    Ok(written) => {
                        flow.to_host.drain(..written);
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => {
                        flow.dead = true;
                        break;
                    }
                }
            }
            if flow.guest_fin && flow.to_host.is_empty() && !flow.host_write_shutdown {
                let _ = stream.shutdown(Shutdown::Write);
                flow.host_write_shutdown = true;
            }
            // Built from fields because `stream` keeps `flow.conn` mutably borrowed. The window and
            // ack are fixed for this pass: nothing below touches `to_host` or `rcv_nxt`.
            let segment_template = TcpHeader {
                src: key.remote_ip,
                dst: flow.guest_ip,
                sport: key.remote_port,
                dport: key.guest_port,
                seq: 0,
                ack: flow.rcv_nxt,
                flags: 0,
                window: advertised_window(flow.to_host.len()),
                advertise_mss: false,
            };
            let mut read_buf = [0u8; MSS];
            while !flow.dead && !flow.fin_sent && budget > 0 {
                let in_flight = flow.snd_nxt.wrapping_sub(flow.snd_una);
                let window_left = flow.guest_window.saturating_sub(in_flight) as usize;
                if window_left == 0 {
                    break;
                }
                let want = window_left.min(flow.mss).min(read_buf.len());
                match stream.read(&mut read_buf[..want]) {
                    Ok(0) => {
                        let header = TcpHeader {
                            flags: FIN | ACK,
                            seq: flow.snd_nxt,
                            ..segment_template
                        };
                        flow.snd_nxt = flow.snd_nxt.wrapping_add(1);
                        flow.fin_sent = true;
                        packets.push(PendingPacket {
                            proto: IP_PROTO_TCP,
                            src: key.remote_ip,
                            dst: flow.guest_ip,
                            payload: tcp_segment(&header, &[]),
                        });
                        budget -= 1;
                    }
                    Ok(n) => {
                        let header = TcpHeader {
                            flags: PSH | ACK,
                            seq: flow.snd_nxt,
                            ..segment_template
                        };
                        flow.snd_nxt = flow.snd_nxt.wrapping_add(n as u32);
                        packets.push(PendingPacket {
                            proto: IP_PROTO_TCP,
                            src: key.remote_ip,
                            dst: flow.guest_ip,
                            payload: tcp_segment(&header, &read_buf[..n]),
                        });
                        budget -= 1;
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => {
                        let header = TcpHeader {
                            flags: RST | ACK,
                            seq: flow.snd_nxt,
                            ..segment_template
                        };
                        packets.push(PendingPacket {
                            proto: IP_PROTO_TCP,
                            src: key.remote_ip,
                            dst: flow.guest_ip,
                            payload: tcp_segment(&header, &[]),
                        });
                        flow.dead = true;
                    }
                }
            }
        }
        self.tcp.retain(|_, flow| !flow.dead);

        let now = Instant::now();
        let mut recv_buf = [0u8; UDP_RECV_BUF_LEN];
        for (key, flow) in self.udp.iter_mut() {
            while let Ok(n) = flow.sock.recv(&mut recv_buf) {
                flow.last_activity = now;
                packets.push(PendingPacket {
                    proto: IP_PROTO_UDP,
                    src: key.remote_ip,
                    dst: flow.guest_ip,
                    payload: udp_segment(
                        key.remote_ip,
                        flow.guest_ip,
                        key.remote_port,
                        key.guest_port,
                        &recv_buf[..n],
                    ),
                });
            }
        }
        self.udp
            .retain(|_, flow| now.duration_since(flow.last_activity) < UDP_IDLE_TIMEOUT);

        for packet in packets {
            self.emit_ip(packet.src, packet.dst, packet.proto, &packet.payload);
        }
    }

    pub fn pop_frame(&mut self) -> Option<Vec<u8>> {
        self.out.pop_front()
    }

    /// Puts a frame the guest could not take yet back at the head, keeping frame order.
    pub fn requeue_frame(&mut self, frame: Vec<u8>) {
        self.out.push_front(frame);
    }
}

pub fn spawn(net: NetHandle) {
    std::thread::spawn(move || {
        let mut usernet = UserNet::new();
        loop {
            if let Some(frame) = net.recv_frame(POLL_INTERVAL) {
                usernet.input(&frame);
                while let Some(frame) = net.try_recv_frame() {
                    usernet.input(&frame);
                }
            }
            usernet.poll();
            while let Some(frame) = usernet.pop_frame() {
                if !net.send_frame(&frame) {
                    usernet.requeue_frame(frame);
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

    fn guest_ip_frame(dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
        let p = ipv4_packet(1, GUEST_IP, dst, proto, payload);
        ethernet_frame(GATEWAY_MAC, GUEST_MAC, 0x0800, &p)
    }

    fn drain(un: &mut UserNet) -> Vec<Vec<u8>> {
        std::iter::from_fn(|| un.pop_frame()).collect()
    }

    fn wait_for(un: &mut UserNet, mut pred: impl FnMut(&[u8]) -> bool) -> Vec<u8> {
        let end = Instant::now() + Duration::from_secs(5);
        while Instant::now() < end {
            un.poll();
            while let Some(f) = un.pop_frame() {
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
        d[28..34].copy_from_slice(&GUEST_MAC);
        d[236..240].copy_from_slice(&[0x63, 0x82, 0x53, 0x63]);
        d.extend_from_slice(&[53, 1, ty, 255]);
        d
    }

    #[test]
    fn dhcp_offer_and_ack() {
        let mut un = UserNet::new();
        for (req, reply) in [(1u8, 2u8), (3, 5)] {
            let u = udp_segment([0; 4], [255; 4], 68, 67, &dhcp_msg(req));
            let p = ipv4_packet(1, [0; 4], [255; 4], 17, &u);
            un.input(&ethernet_frame(BROADCAST_MAC, GUEST_MAC, 0x0800, &p));
            let out = drain(&mut un);
            assert_eq!(out.len(), 1);
            let ip = parse_ipv4(&out[0]).unwrap();
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
        arp.extend_from_slice(&GUEST_MAC);
        arp.extend_from_slice(&GUEST_IP);
        arp.extend_from_slice(&[0; 6]);
        arp.extend_from_slice(&GATEWAY_IP);
        un.input(&ethernet_frame(BROADCAST_MAC, GUEST_MAC, 0x0806, &arp));
        let out = drain(&mut un);
        assert_eq!(out[0][20..22], [0, 2]);
        assert_eq!(out[0][22..28], GATEWAY_MAC);

        let mut echo = vec![8, 0, 0, 0, 0, 1, 0, 1, 9, 9];
        let c = internet_checksum(&echo);
        echo[2..4].copy_from_slice(&c.to_be_bytes());
        un.input(&guest_ip_frame([8, 8, 8, 8], 1, &echo));
        let out = drain(&mut un);
        let ip = parse_ipv4(&out[0]).unwrap();
        assert_eq!((ip.src, ip.dst, ip.payload[0]), ([8, 8, 8, 8], GUEST_IP, 0));
        assert_eq!(internet_checksum(ip.payload), 0);
    }

    #[test]
    fn udp_nat_roundtrip() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let port = server.local_addr().unwrap().port();
        let mut un = UserNet::new();
        let u = udp_segment(GUEST_IP, LOOPBACK_IP, 4000, port, b"ping");
        un.input(&guest_ip_frame(LOOPBACK_IP, 17, &u));
        let mut buf = [0u8; 64];
        let (n, from) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"ping");
        server.send_to(b"pong", from).unwrap();
        let f = wait_for(&mut un, |f| parse_ipv4(f).is_some_and(|i| i.proto == 17));
        let ip = parse_ipv4(&f).unwrap();
        assert_eq!(ip.src, LOOPBACK_IP);
        assert_eq!(u16::from_be_bytes([ip.payload[0], ip.payload[1]]), port);
        assert_eq!(u16::from_be_bytes([ip.payload[2], ip.payload[3]]), 4000);
        assert_eq!(&ip.payload[8..], b"pong");
        assert_eq!(transport_checksum(ip.src, ip.dst, 17, ip.payload), 0);
    }

    struct Guest {
        sport: u16,
        dport: u16,
        seq: u32,
        ack: u32,
    }

    impl Guest {
        fn send(&mut self, un: &mut UserNet, flags: u8, data: &[u8]) {
            let h = TcpHeader {
                src: GUEST_IP,
                dst: LOOPBACK_IP,
                sport: self.sport,
                dport: self.dport,
                seq: self.seq,
                ack: self.ack,
                flags,
                window: 60000,
                advertise_mss: false,
            };
            let s = tcp_segment(&h, data);
            un.input(&guest_ip_frame(LOOPBACK_IP, 6, &s));
            self.seq = self
                .seq
                .wrapping_add(data.len() as u32 + (flags & (SYN | FIN) != 0) as u32);
        }
    }

    fn tcp_of(f: &[u8]) -> Option<(u8, u32, u32, Vec<u8>)> {
        let ip = parse_ipv4(f)?;
        if ip.proto != 6 {
            return None;
        }
        let p = ip.payload;
        assert_eq!(
            transport_checksum(ip.src, ip.dst, 6, p),
            0,
            "bad tcp checksum"
        );
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
