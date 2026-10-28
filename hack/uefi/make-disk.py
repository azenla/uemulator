#!/usr/bin/env python3
"""Builds a GPT disk image with one FAT32 EFI system partition, for the uemulator UEFI firmware.

    hack/uefi/make-disk.py OUT.img --size 128 \
        EFI/BOOT/BOOTX64.EFI=path/to/vmlinuz initramfs=path/to/initramfs

Each DEST=SRC argument copies the host file SRC to DEST on the ESP (directories are created).
Needs only python3, so it runs the same on macOS and Linux. The FAT writer is deliberately
plain: 512-byte clusters, one cluster chain per file, long names with generated 8.3 aliases.
"""
import argparse
import struct
import sys
import uuid
import zlib

SECTOR = 512
ESP_TYPE = uuid.UUID("C12A7328-F81F-11D2-BA4B-00A0C93EC93B")
FIRST_PARTITION_LBA = 2048
ATTR_DIRECTORY = 0x10
ATTR_ARCHIVE = 0x20
ATTR_LONG_NAME = 0x0F
LFN_OFFSETS = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30]


def lfn_checksum(short):
    total = 0
    for byte in short:
        total = (((total & 1) << 7) | (total >> 1)) + byte & 0xFF
    return total


class Fat32:
    def __init__(self, sectors):
        self.sectors = sectors
        self.reserved = 32
        # One sector per cluster. FAT32 needs at least 65525 clusters, so very small images
        # are refused rather than silently becoming FAT16.
        self.fat_sectors = (sectors // 128) + 2
        self.data_sector = self.reserved + 2 * self.fat_sectors
        self.clusters = sectors - self.data_sector
        if self.clusters < 65525:
            sys.exit("ESP too small for FAT32: use --size 40 or more")
        self.image = bytearray(sectors * SECTOR)
        self.next_cluster = 2
        self.fat = [0] * (self.clusters + 2)
        self.fat[0] = 0x0FFFFFF8
        self.fat[1] = 0x0FFFFFFF
        self.root = {}

    def allocate(self, count):
        count = max(count, 1)
        chain = list(range(self.next_cluster, self.next_cluster + count))
        self.next_cluster += count
        if self.next_cluster > self.clusters + 2:
            sys.exit("ESP is full: use a larger --size")
        for index, cluster in enumerate(chain):
            self.fat[cluster] = chain[index + 1] if index + 1 < len(chain) else 0x0FFFFFFF
        return chain

    def write_chain(self, chain, data):
        for index, cluster in enumerate(chain):
            at = (self.data_sector + cluster - 2) * SECTOR
            chunk = data[index * SECTOR:(index + 1) * SECTOR]
            self.image[at:at + len(chunk)] = chunk

    def add(self, path, data):
        parts = [part for part in path.split("/") if part]
        node = self.root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = data

    @staticmethod
    def short_entry(name11, attributes, cluster, size):
        raw = bytearray(32)
        raw[0:11] = name11
        raw[11] = attributes
        struct.pack_into("<H", raw, 20, cluster >> 16)
        struct.pack_into("<H", raw, 26, cluster & 0xFFFF)
        struct.pack_into("<I", raw, 28, size)
        return bytes(raw)

    def directory_bytes(self, entries, own, parent):
        out = bytearray()
        if own is not None:
            out += self.short_entry(b".          ", ATTR_DIRECTORY, own, 0)
            out += self.short_entry(b"..         ", ATTR_DIRECTORY, parent, 0)
        serial = 0
        for name, (attributes, cluster, size) in entries:
            stem, _, extension = name.rpartition(".") if "." in name else (name, "", "")
            fits = (len(stem) <= 8 and len(extension) <= 3 and name == name.upper()
                    and all(c.isalnum() or c in "._" for c in name))
            short = bytearray(b" " * 11)
            if fits:
                short[0:len(stem)] = stem.encode()
                short[8:8 + len(extension)] = extension.encode()
            else:
                serial += 1
                tail = ("~%d" % serial).encode()
                base = "".join(c for c in stem.upper() if c.isalnum()).encode()[:8 - len(tail)]
                short[0:len(base)] = base
                short[len(base):len(base) + len(tail)] = tail
                short[8:8 + len(extension[:3])] = extension[:3].upper().encode()
                units = name.encode("utf-16-le")
                units = [struct.unpack_from("<H", units, i)[0] for i in range(0, len(units), 2)]
                chunks = [units[i:i + 13] for i in range(0, len(units), 13)]
                checksum = lfn_checksum(short)
                for index in reversed(range(len(chunks))):
                    chunk = chunks[index]
                    raw = bytearray(32)
                    raw[0] = (index + 1) | (0x40 if index == len(chunks) - 1 else 0)
                    raw[11] = ATTR_LONG_NAME
                    raw[13] = checksum
                    for slot, offset in enumerate(LFN_OFFSETS):
                        unit = chunk[slot] if slot < len(chunk) else (0 if slot == len(chunk) else 0xFFFF)
                        struct.pack_into("<H", raw, offset, unit)
                    out += raw
            out += self.short_entry(bytes(short), attributes, cluster, size)
        return bytes(out)

    def place(self, tree, own, parent):
        """Writes the directory `tree` at cluster `own` (already allocated) and its children."""
        entries = []
        for name, value in tree.items():
            if isinstance(value, dict):
                chain = self.allocate(1)
                self.place(value, chain[0], own)
                entries.append((name, (ATTR_DIRECTORY, chain[0], 0)))
            else:
                if value:
                    chain = self.allocate(-(-len(value) // SECTOR))
                    self.write_chain(chain, value)
                    entries.append((name, (ATTR_ARCHIVE, chain[0], len(value))))
                else:
                    entries.append((name, (ATTR_ARCHIVE, 0, 0)))
        data = self.directory_bytes(entries, own if own != 2 else None, 0 if parent == 2 else parent)
        chain = [own]
        needed = -(-len(data) // SECTOR)
        if needed > 1:
            extra = self.allocate(needed - 1)
            self.fat[own] = extra[0]
            chain += extra
        self.write_chain(chain, data)

    def build(self):
        root_chain = self.allocate(1)
        self.place(self.root, root_chain[0], 2)
        boot = bytearray(SECTOR)
        boot[0:3] = b"\xeb\x58\x90"
        boot[3:11] = b"UEMU 1.0"
        struct.pack_into("<HBHBHHBHHHII", boot, 11, SECTOR, 1, self.reserved, 2, 0, 0, 0xF8, 0, 63, 255, 0, self.sectors)
        struct.pack_into("<IHHIHH", boot, 36, self.fat_sectors, 0, 0, 2, 1, 6)
        boot[64] = 0x80
        boot[66] = 0x29
        struct.pack_into("<I", boot, 67, 0x554D4531)
        boot[71:82] = b"UEMULATOR  "
        boot[82:90] = b"FAT32   "
        boot[510:512] = b"\x55\xaa"
        info = bytearray(SECTOR)
        info[0:4] = b"RRaA"
        info[484:488] = b"rrAa"
        struct.pack_into("<II", info, 488, 0xFFFFFFFF, 0xFFFFFFFF)
        info[510:512] = b"\x55\xaa"
        self.image[0:SECTOR] = boot
        self.image[SECTOR:2 * SECTOR] = info
        self.image[6 * SECTOR:7 * SECTOR] = boot
        fat = struct.pack("<%dI" % len(self.fat), *self.fat)
        for copy in range(2):
            at = (self.reserved + copy * self.fat_sectors) * SECTOR
            self.image[at:at + len(fat)] = fat
        return bytes(self.image)


def gpt_disk(partition, total_sectors):
    disk = bytearray(total_sectors * SECTOR)
    first, last = FIRST_PARTITION_LBA, FIRST_PARTITION_LBA + len(partition) // SECTOR - 1
    mbr = bytearray(SECTOR)
    struct.pack_into("<B3sB3sII", mbr, 446 + 0, 0, b"\x00\x02\x00", 0xEE, b"\xff\xff\xff", 1, min(total_sectors - 1, 0xFFFFFFFF))
    mbr[510:512] = b"\x55\xaa"
    disk[0:SECTOR] = mbr
    entry = bytearray(128)
    entry[0:16] = ESP_TYPE.bytes_le
    entry[16:32] = uuid.uuid4().bytes_le
    struct.pack_into("<QQQ", entry, 32, first, last, 0)
    name = "EFI system partition".encode("utf-16-le")
    entry[56:56 + len(name)] = name
    entries = bytes(entry) + bytes(127 * 128)
    disk_guid = uuid.uuid4().bytes_le

    def header(my_lba, other_lba, entries_lba):
        raw = bytearray(92)
        raw[0:8] = b"EFI PART"
        struct.pack_into("<IIIIQQQQ", raw, 8, 0x10000, 92, 0, 0, my_lba, other_lba, 34, total_sectors - 34)
        raw[56:72] = disk_guid
        struct.pack_into("<QII I", raw, 72, entries_lba, 128, 128, zlib.crc32(entries))
        struct.pack_into("<I", raw, 16, zlib.crc32(bytes(raw)))
        return bytes(raw)

    disk[SECTOR:SECTOR + 92] = header(1, total_sectors - 1, 2)
    disk[2 * SECTOR:2 * SECTOR + len(entries)] = entries
    backup_entries_lba = total_sectors - 33
    disk[backup_entries_lba * SECTOR:backup_entries_lba * SECTOR + len(entries)] = entries
    disk[(total_sectors - 1) * SECTOR:(total_sectors - 1) * SECTOR + 92] = header(total_sectors - 1, 1, backup_entries_lba)
    disk[first * SECTOR:first * SECTOR + len(partition)] = partition
    return bytes(disk)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=128, help="disk size in MiB")
    parser.add_argument("files", nargs="*", metavar="DEST=SRC")
    args = parser.parse_args()
    total_sectors = args.size * 1024 * 1024 // SECTOR
    partition_sectors = total_sectors - FIRST_PARTITION_LBA - 34
    fat = Fat32(partition_sectors)
    for item in args.files:
        destination, _, source = item.partition("=")
        with open(source, "rb") as handle:
            fat.add(destination, handle.read())
    with open(args.out, "wb") as out:
        out.write(gpt_disk(fat.build(), total_sectors))


if __name__ == "__main__":
    main()
