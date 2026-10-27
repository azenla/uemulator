//! `--net-loop`: a minimal gateway that only answers ARP and ICMP echo for 10.0.2.2. Used by the
//! tests and for checking the virtio-net path; it does no DHCP and forwards nothing.

// Standalone on purpose: `crates/uemulator/tests/devices.rs` includes this file with `#[path]`, so
// it cannot depend on anything else in the CLI crate (the gateway constants here duplicate the
// ones in `usernet.rs`).

use std::ops::Range;

const GATEWAY_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
const GATEWAY_IP: [u8; 4] = [10, 0, 2, 2];

const ETH_HEADER_LEN: usize = 14;
const ETH_SRC_MAC: Range<usize> = 6..12;
const ETH_DST_MAC: Range<usize> = 0..6;
const ETH_TYPE: Range<usize> = 12..14;
const ETHERTYPE_IPV4: u16 = 0x0800;
const ETHERTYPE_ARP: u16 = 0x0806;

/// Offsets are from the start of the Ethernet frame. The ARP packet starts at 14.
const ARP_FRAME_LEN: usize = 42;
const ARP_OPCODE: Range<usize> = 20..22;
/// Sender MAC and IP, which become the target of the reply.
const ARP_SENDER: Range<usize> = 22..32;
const ARP_TARGET_IP: Range<usize> = 38..42;
const ARP_REQUEST: [u8; 2] = [0, 1];
/// Ethernet/IPv4 hardware and protocol types, address lengths 6 and 4, then opcode 2 (reply).
const ARP_REPLY_HEADER: [u8; 8] = [0, 1, 8, 0, 6, 4, 0, 2];

/// A fixed 20-byte IPv4 header is assumed, so the ICMP header starts at frame offset 34.
const IPV4_PROTOCOL: usize = 23;
const IPV4_SRC: Range<usize> = 26..30;
const IPV4_DST: Range<usize> = 30..34;
/// Ethernet header plus the fixed IPv4 header.
const IPV4_FRAME_MIN_LEN: usize = 34;
const IP_PROTO_ICMP: u8 = 1;
const ICMP_TYPE: usize = 34;
const ICMP_CHECKSUM: Range<usize> = 36..38;
const ICMP_ECHO_REQUEST: u8 = 8;
const ICMP_ECHO_REPLY: u8 = 0;

/// RFC 1071 ones'-complement sum.
fn internet_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    for chunk in data.chunks(2) {
        sum += u16::from_be_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]) as u32;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

pub fn handle(frame: &[u8]) -> Option<Vec<u8>> {
    if frame.len() < ETH_HEADER_LEN {
        return None;
    }
    let src_mac: [u8; 6] = frame[ETH_SRC_MAC].try_into().unwrap();
    let ethertype = u16::from_be_bytes(frame[ETH_TYPE].try_into().unwrap());
    match ethertype {
        ETHERTYPE_ARP
            if frame.len() >= ARP_FRAME_LEN
                && frame[ARP_OPCODE] == ARP_REQUEST
                && frame[ARP_TARGET_IP] == GATEWAY_IP =>
        {
            let mut reply = Vec::with_capacity(ARP_FRAME_LEN);
            reply.extend_from_slice(&src_mac);
            reply.extend_from_slice(&GATEWAY_MAC);
            reply.extend_from_slice(&ETHERTYPE_ARP.to_be_bytes());
            reply.extend_from_slice(&ARP_REPLY_HEADER);
            reply.extend_from_slice(&GATEWAY_MAC);
            reply.extend_from_slice(&GATEWAY_IP);
            reply.extend_from_slice(&frame[ARP_SENDER]);
            Some(reply)
        }
        ETHERTYPE_IPV4
            if frame.len() >= IPV4_FRAME_MIN_LEN
                && frame[IPV4_PROTOCOL] == IP_PROTO_ICMP
                && frame[IPV4_DST] == GATEWAY_IP
                && frame[ICMP_TYPE] == ICMP_ECHO_REQUEST =>
        {
            let mut reply = frame.to_vec();
            reply[ETH_DST_MAC].copy_from_slice(&src_mac);
            reply[ETH_SRC_MAC].copy_from_slice(&GATEWAY_MAC);
            // Swapping source and destination leaves the IPv4 header checksum valid.
            let (guest_ip, gateway_ip) = (frame[IPV4_SRC].to_vec(), frame[IPV4_DST].to_vec());
            reply[IPV4_SRC].copy_from_slice(&gateway_ip);
            reply[IPV4_DST].copy_from_slice(&guest_ip);
            reply[ICMP_TYPE] = ICMP_ECHO_REPLY;
            reply[ICMP_CHECKSUM].copy_from_slice(&[0, 0]);
            let checksum = internet_checksum(&reply[ICMP_TYPE..]);
            reply[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
            Some(reply)
        }
        _ => None,
    }
}
