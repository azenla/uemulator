#!/usr/bin/env bash
# Wall seconds from process start to the kernel's "Freeing initrd memory" line: the initramfs
# unpack dominates it. Prints `unpack <seconds>`. KERNEL and INITRAMFS default to the `test`
# machine's artifacts; UEMULATOR_BIN picks another binary.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
home="${UEMULATOR_HOME:-$HOME/.uemulator}"
kernel="${KERNEL:-$home/machines/test/artifacts/vmlinuz}"
initramfs="${INITRAMFS:-$home/machines/test/artifacts/initrd.img}"
bin="${UEMULATOR_BIN:-$root/target/release/uemulator}"
log="$(mktemp)"
trap 'kill "$pid" 2>/dev/null || true; rm -f "$log"' EXIT
start=$(python3 -c 'import time; print(time.time())')
"$bin" advanced boot --kernel "$kernel" --initramfs "$initramfs" --console serial >"$log" 2>&1 </dev/null &
pid=$!
until grep -q "Freeing initrd memory" "$log"; do
    kill -0 "$pid" 2>/dev/null || { echo "emulator exited before the unpack finished" >&2; exit 1; }
    sleep 0.1
done
python3 -c "import time; print('unpack %.2f' % (time.time() - $start))"
