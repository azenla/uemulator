#!/bin/sh
# Runs a real Linux KVM (kvm_amd, inside the emulated guest) against the emulator's SVM: the guest
# kernel loads kvm.ko/kvm-amd.ko and /kvmtest starts a vCPU in real mode, 32-bit protected mode,
# long mode, PAE paging and 32-bit paging, each printing a marker through port 0x3f8 and halting.
# Not a unit test (it downloads Alpine's modloop and takes a minute): run it after changing
# crates/uemulator/src/cpu/svm.rs, the paging code or the 16/32-bit modes.
#
# Needs: an `x86_64-linux-musl-gcc` (or a Linux host gcc), `unsquashfs` (brew install squashfs),
# `bsdtar` or `cpio`, and the Alpine artifacts of a machine made with
# `cargo run --release -- quick create alpine --install alpine` (UEMULATOR_MACHINE names another).
#   hack/kvm/run.sh            # expects "KVMTEST done failures=0"
set -e

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cache=${UEMULATOR_HOME:-$HOME/.uemulator}/machines/${UEMULATOR_MACHINE:-alpine}/artifacts
work=${KVM_TEST_DIR:-/tmp/uemulator-kvm-test}
base=https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot

[ -f "$cache/vmlinuz-virt" ] || { echo "run 'uemulator quick create alpine --install alpine' once to fetch $cache" >&2; exit 1; }
mkdir -p "$work"
cd "$work"

[ -f modloop-virt ] || curl -fL -o modloop-virt "$base/modloop-virt"
version=$(unsquashfs -l modloop-virt | sed -n 's|^squashfs-root/modules/\([^/]*\)/kernel/arch/x86/kvm/kvm.ko$|\1|p')
rm -rf mods extra
unsquashfs -q -d mods modloop-virt \
    "modules/$version/kernel/arch/x86/kvm/kvm.ko" \
    "modules/$version/kernel/arch/x86/kvm/kvm-amd.ko" \
    "modules/$version/kernel/virt/lib/irqbypass.ko" >/dev/null
mkdir extra
cp mods/modules/"$version"/kernel/arch/x86/kvm/kvm.ko mods/modules/"$version"/kernel/arch/x86/kvm/kvm-amd.ko \
    mods/modules/"$version"/kernel/virt/lib/irqbypass.ko extra/

cc=${CC:-x86_64-linux-musl-gcc}
"$cc" -static -O1 -o extra/kvmtest "$here/kvmtest.c"
cat >extra/init2 <<'INIT'
#!/bin/busybox sh
B=/bin/busybox
$B mount -t proc proc /proc
$B mount -t sysfs sysfs /sys
$B mount -t devtmpfs dev /dev
$B insmod /irqbypass.ko
$B insmod /kvm.ko
$B insmod /kvm-amd.ko
$B dmesg | $B grep -i "kvm_amd"
/kvmtest
$B poweroff -f
INIT
chmod +x extra/init2 extra/kvmtest
(cd extra && bsdtar --format newc -cf - . | gzip -9) >extra.cpio.gz
# The kernel unpacks concatenated initramfs archives in order.
cat "$cache/initramfs-virt" extra.cpio.gz >initramfs-kvm

(sleep "${KVM_TEST_SECONDS:-90}") | timeout "${KVM_TEST_SECONDS:-90}" \
    "$root/target/release/uemulator" boot --kernel "$cache/vmlinuz-virt" --initramfs initramfs-kvm \
    --console serial -m 512 --cmdline "console=ttyS0 rdinit=/init2" 2>&1 | grep -a "kvm_amd\|KVMTEST"
