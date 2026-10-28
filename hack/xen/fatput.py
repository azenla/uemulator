#!/usr/bin/env python3
"""Overwrite a file of a FAT32 partition of a raw GPT disk image in place, from the host.

usage: fatput.py DISK PARTITION_NUMBER PATH SRC
The file must already exist and be at least as large as SRC (hack/xen/grubcfg.sh creates xen.cfg
as 8 KiB of padding); the rest is filled with newlines and the directory size is kept. Names
are 8.3 only.
"""
import struct
import sys

SECTOR = 512
disk, partition, path, source = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
image = open(disk, "r+b")


def read(offset, length):
    image.seek(offset)
    return image.read(length)


def write(offset, data):
    image.seek(offset)
    image.write(data)


entry = read(2 * SECTOR + (partition - 1) * 128, 128)
base = struct.unpack_from("<Q", entry, 32)[0] * SECTOR
boot = read(base, SECTOR)
bytes_per_sector, sectors_per_cluster, reserved, fats = struct.unpack_from("<HBHB", boot, 11)
sectors_per_fat = struct.unpack_from("<I", boot, 36)[0]
root_cluster = struct.unpack_from("<I", boot, 44)[0]
cluster_bytes = bytes_per_sector * sectors_per_cluster
fat_offset = base + reserved * bytes_per_sector
data_offset = base + (reserved + fats * sectors_per_fat) * bytes_per_sector


def chain(cluster):
    while cluster < 0x0FFFFFF8:
        yield cluster
        cluster = struct.unpack("<I", read(fat_offset + cluster * 4, 4))[0] & 0x0FFFFFFF


def lookup(cluster, name):
    for c in chain(cluster):
        offset = data_offset + (c - 2) * cluster_bytes
        data = read(offset, cluster_bytes)
        for i in range(0, len(data), 32):
            record = data[i : i + 32]
            if record[0] == 0 or record[0] == 0xE5 or record[11] == 0x0F:
                continue
            short = record[:8].decode().rstrip() + ("." + record[8:11].decode().rstrip() if record[8:11].strip() else "")
            if short.upper() == name.upper():
                first = (struct.unpack_from("<H", record, 20)[0] << 16) | struct.unpack_from("<H", record, 26)[0]
                return first, struct.unpack_from("<I", record, 28)[0]
    raise SystemExit("not found: " + name)


cluster = root_cluster
size = 0
for part in path.split("/"):
    cluster, size = lookup(cluster, part)
content = open(source, "rb").read()
if len(content) > size:
    raise SystemExit(f"{source} is {len(content)} bytes, the file holds {size}")
content = content.ljust(size, b"\n")
for c in chain(cluster):
    chunk, content = content[:cluster_bytes], content[cluster_bytes:]
    write(data_offset + (c - 2) * cluster_bytes, chunk)
