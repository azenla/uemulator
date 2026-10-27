#!/bin/sh
# Assembles Intel-syntax x86_64 (stdin or a file) and prints the bytes as a Rust array, for
# hand-written test code. Needs clang and objdump (both ship with the Xcode tools).
#   echo 'mov rax, 5; add rax, 7; hlt' | hack/asm.sh
set -e

dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
{
    echo '.intel_syntax noprefix'
    if [ -n "$1" ]; then cat "$1"; else cat | tr ';' '\n'; fi
} >"$dir/in.s"
clang -target x86_64-unknown-none-elf -c "$dir/in.s" -o "$dir/in.o"
objdump -s -j .text "$dir/in.o" |
    awk '/^ [0-9a-f]+ / { for (i = 2; i <= 5; i++) if ($i ~ /^[0-9a-f]+$/) hex = hex $i }
         END {
             printf "[";
             for (i = 1; i <= length(hex); i += 2) printf "%s0x%s", (i > 1 ? ", " : ""), substr(hex, i, 2);
             print "]"
         }'
