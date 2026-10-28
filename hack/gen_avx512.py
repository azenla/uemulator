#!/usr/bin/env python3
"""Generates crates/uemulator/src/cpu/avx512_encodings.rs: assembled AVX-512 test encodings.

Each form becomes a `Variants` constant holding the instruction in every shape the tests exercise
(xmm/ymm/zmm x plain, merge-masked, zero-masked, memory, broadcast, high registers). The form list
is hack/avx512_forms.py. Needs clang and objdump. Run from the repository root:
    python3 hack/gen_avx512.py
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTHS = [("x", "xmm", 128), ("y", "ymm", 256), ("z", "zmm", 512)]
SHAPES = ["p", "k", "kz", "m", "mk", "b", "bk"]

FORMS = []
SINGLES = []


def form(name, mnemonic, kind="bin", bits=32, imm=None):
    """kind: bin = dst, src1, src2; un = dst, src2; cmp/cmpun: the destination is k2; widen<N>/narrow<N>:
    the narrow operand is 1/N of the vector. bits is the broadcast element size (0: none)."""
    FORMS.append((name, mnemonic, kind, bits, imm))


def single(name, asm):
    """One hand-written instruction, emitted as a plain byte constant."""
    SINGLES.append((name, asm))


exec(open(os.path.join(HERE, "avx512_forms.py")).read())


def vector_name(bits):
    return {128: "xmm", 256: "ymm", 512: "zmm"}[max(128, min(512, bits))]


def shape_asm(mnemonic, kind, bits, imm, length, shape):
    _, reg, vl = length
    dst, s1, s2 = f"{reg}0", f"{reg}1", f"{reg}2"
    mask = ""
    if shape in ("k", "mk", "bk"):
        mask = "{k1}"
    if shape == "kz":
        mask = "{k1}{z}"
    if kind.startswith("widen"):
        # r/m operand is vl/ratio wide
        s2 = vector_name(vl // int(kind[5:])) + "2"
    if kind.startswith("narrow"):
        # the destination is the narrow one, the source a full register
        dst = vector_name(vl // int(kind[6:])) + "0"
        s2 = f"{reg}1"
    if kind.startswith("cvtnarrow"):
        # a narrowing conversion: narrow register destination, full-width register/memory source
        dst = vector_name(vl // int(kind[9:])) + "0"
    if shape in ("m", "mk"):
        if kind.startswith("narrow"):
            dst = "[rdi]"
        elif kind.startswith("cvtnarrow"):
            s2 = f"{vector_name(vl)}word ptr [rsi]".replace("mmword", "mmword")
        else:
            s2 = "[rsi]"
    if shape in ("b", "bk"):
        s2 = f"[rsi]{{1to{vl // bits}}}"
        if kind.startswith("cvtnarrow"):
            s2 = f"qword ptr [rsi]{{1to{vl // bits}}}"
    if kind in ("cmp", "cmpun"):
        dst = "k2"
    operands = [f"{dst}{mask}"]
    if kind in ("bin", "cmp"):
        operands.append(s1)
    operands.append(s2)
    if imm is not None:
        operands.append(str(imm))
    # clang prefers the shorter VEX encoding when one exists; the tests want the EVEX one
    return f"{{evex}} {mnemonic} " + ", ".join(operands)


def collect():
    items = []
    for name, mnemonic, kind, bits, imm in FORMS:
        for length in LENGTHS:
            for shape in SHAPES:
                if shape in ("b", "bk") and bits == 0:
                    continue
                if shape == "kz" and kind in ("cmp", "cmpun"):
                    continue
                if shape in ("b", "bk") and (kind.startswith("widen") or kind.startswith("narrow")):
                    continue
                items.append((f"{name}.{length[0]}.{shape}", shape_asm(mnemonic, kind, bits, imm, length, shape)))
        high = shape_asm(mnemonic, kind, bits, imm, LENGTHS[2], "p")
        high = re.sub(r"(xmm|ymm|zmm)([012])\b", lambda m: f"{m.group(1)}2{m.group(2)}", high)
        high = high.replace("k2", "k3") if kind in ("cmp", "cmpun") else high
        items.append((f"{name}.high", high))
    for name, asm in SINGLES:
        items.append((f"{name}.single", asm))
    return items


def assemble(items):
    """Assembles every item; one the assembler rejects (a shape that does not exist for the
    instruction, e.g. an xmm form of vshuff32x4) is dropped and reported as absent."""
    encodings = {}
    pending = list(items)
    while pending:
        lines = [".intel_syntax noprefix", ".text"]
        for n, (_, asm) in enumerate(pending):
            lines += [f"v{n}:", asm, "hlt"]
        with tempfile.TemporaryDirectory() as d:
            src, obj = os.path.join(d, "a.s"), os.path.join(d, "a.o")
            open(src, "w").write("\n".join(lines) + "\n")
            r = subprocess.run(["clang", "-target", "x86_64-unknown-none-elf", "-c", src, "-o", obj],
                               capture_output=True, text=True)
            if r.returncode:
                bad = {(int(m) - 4) // 3 for m in re.findall(r"a\.s:(\d+):\d+: error", r.stderr)}
                if not bad:
                    sys.exit(r.stderr)
                pending = [item for n, item in enumerate(pending) if n not in bad]
                continue
            dump = subprocess.run(["objdump", "-d", obj], capture_output=True, text=True).stdout
        out = {}
        current, data = None, []
        for line in dump.splitlines():
            m = re.match(r"^[0-9a-f]+ <v(\d+)>:", line)
            if m:
                current, data = int(m.group(1)), []
                continue
            if current is None:
                continue
            m = re.match(r"^\s*[0-9a-f]+:\s((?:[0-9a-f]{2}\s)+)\s*(.*)$", line)
            if not m:
                continue
            data += [int(b, 16) for b in m.group(1).split()]
            if m.group(2).strip() == "hlt":
                out[current] = data
                current = None
        for n, (key, _) in enumerate(pending):
            encodings[key] = out[n]
        break
    return encodings


def rust_bytes(b):
    return "&[" + ", ".join(f"0x{x:02x}" for x in b) + "]"


def main():
    items = collect()
    enc = assemble(items)
    out = ["// @generated by hack/gen_avx512.py from the assembler; do not edit.", "",
           "// Not every shape of every form is used by a test.",
           "#![allow(dead_code)]", "",
           "pub(crate) struct Variants {",
           "    /// Each shape is indexed by vector length: xmm, ymm, zmm (empty when it does not exist)."]
    for s in SHAPES:
        out.append(f"    pub {s}: [&'static [u8]; 3],")
    out += ["    pub high: &'static [u8],", "}", ""]
    for name, _, _, _, _ in FORMS:
        out.append(f"pub(crate) const {name}: Variants = Variants {{")
        for s in SHAPES:
            parts = []
            for length in LENGTHS:
                key = f"{name}.{length[0]}.{s}"
                parts.append(rust_bytes(enc[key]) if key in enc else "&[]")
            out.append(f"    {s}: [{', '.join(parts)}],")
        out.append(f"    high: {rust_bytes(enc.get(name + '.high', []))},")
        out.append("};")
    for name, asm in SINGLES:
        if name + ".single" not in enc:
            sys.exit(f"the assembler rejected {name}: {asm}")
        out.append(f"pub(crate) const {name}: &[u8] = {rust_bytes(enc[name + '.single'])}; // {asm}")
    dest = os.path.join(HERE, "..", "crates", "uemulator", "src", "cpu", "avx512_encodings.rs")
    open(dest, "w").write("\n".join(out) + "\n")


main()
