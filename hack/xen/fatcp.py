#!/usr/bin/env python3
"""Copy one file (8.3 names only) out of a FAT32 partition of a raw GPT disk image.

usage: fatcp.py DISK PARTITION_NUMBER PATH OUT    e.g. fatcp.py disk-0.img 1 EFI/BOOT/BOOTX64.EFI out.efi
"""
import struct
import sys

SECTOR = 512
disk, partition, path, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
image = open(disk, "rb")


def read(offset, length):
    image.seek(offset)
    return image.read(length)


# GPT entries start at LBA 2; the entry's first LBA is at +32.
entry = read(2 * SECTOR + (partition - 1) * 128, 128)
base = struct.unpack_from("<Q", entry, 32)[0] * SECTOR
boot = read(base, SECTOR)
bytes_per_sector, sectors_per_cluster, reserved, fats = struct.unpack_from("<HBHB", boot, 11)
sectors_per_fat, root_cluster = struct.unpack_from("<I", boot, 36)[0], struct.unpack_from("<I", boot, 44)[0]
cluster_bytes = bytes_per_sector * sectors_per_cluster
fat_offset = base + reserved * bytes_per_sector
data_offset = base + (reserved + fats * sectors_per_fat) * bytes_per_sector


def next_cluster(cluster):
    return struct.unpack("<I", read(fat_offset + cluster * 4, 4))[0] & 0x0FFFFFFF


def chain(cluster):
    while cluster < 0x0FFFFFF8:
        yield cluster
        cluster = next_cluster(cluster)


def cluster_data(cluster):
    return read(data_offset + (cluster - 2) * cluster_bytes, cluster_bytes)


def lookup(cluster, name):
    for c in chain(cluster):
        data = cluster_data(c)
        for i in range(0, len(data), 32):
            entry = data[i : i + 32]
            if entry[0] == 0 or entry[0] == 0xE5 or entry[11] == 0x0F:
                continue
            short = entry[:8].decode().rstrip() + ("." + entry[8:11].decode().rstrip() if entry[8:11].strip() else "")
            if short.upper() == name.upper():
                first = (struct.unpack_from("<H", entry, 20)[0] << 16) | struct.unpack_from("<H", entry, 26)[0]
                return first, struct.unpack_from("<I", entry, 28)[0]
    raise SystemExit("not found: " + name)


cluster = root_cluster
size = 0
for part in path.split("/"):
    cluster, size = lookup(cluster, part)
content = b"".join(cluster_data(c) for c in chain(cluster))[:size]
open(out, "wb").write(content)
