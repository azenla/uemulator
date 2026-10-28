#!/bin/sh
# Boots the UEFI application in QEMU, accelerated when the host allows it.
#
#   hack/uefi/boot.sh [x86_64|aarch64] [-- extra kernel command line words]
#
# The architecture is the one the UEFI application is built for, and defaults to the host's.
# The guest is always an x86_64 Linux kernel. Only a same-architecture QEMU can use KVM or HVF,
# so on an Apple Silicon Mac the aarch64 build is the accelerated one.
#
# Environment: UEFI_ACCEL (auto, hv or emu: hardware virtualization or the emulator), KERNEL, INITRD (default: the artifacts of the `alpine` machine, or the one UEMULATOR_MACHINE names), GUEST_MEM (MiB, the
# emulated machine's RAM), QEMU_ACCEL (kvm, hvf, tcg), QEMU_FIRMWARE, QEMU_MEMORY (MiB, the
# QEMU VM's RAM), QEMU_EXTRA_ARGS, SKIP_BUILD=1.
set -e

cd "$(dirname "${0}")/../.." || exit 1

HOST_ARCH="$(uname -m)"
[ "${HOST_ARCH}" = "arm64" ] && HOST_ARCH="aarch64"
[ "${HOST_ARCH}" = "amd64" ] && HOST_ARCH="x86_64"

TARGET_ARCH="${HOST_ARCH}"
case "${1}" in
x86_64 | aarch64)
	TARGET_ARCH="${1}"
	shift
	;;
esac
[ "${1}" = "--" ] && shift

RUST_TARGET="${TARGET_ARCH}-unknown-uefi"
ARTIFACTS="${UEMULATOR_HOME:-${HOME}/.uemulator}/machines/${UEMULATOR_MACHINE:-alpine}/artifacts"
KERNEL="${KERNEL:-${ARTIFACTS}/vmlinuz-virt}"
INITRD="${INITRD:-${ARTIFACTS}/initramfs-virt}"
GUEST_MEM="${GUEST_MEM:-1024}"
QEMU_MEMORY="${QEMU_MEMORY:-4096}"

for f in "${KERNEL}" "${INITRD}"; do
	if [ ! -f "${f}" ]; then
		echo "missing ${f}: run 'cargo run --release -- quick create alpine --install alpine' once, or set KERNEL and INITRD" >&2
		exit 1
	fi
done

if [ -z "${QEMU_ACCEL}" ] && [ "${TARGET_ARCH}" = "${HOST_ARCH}" ]; then
	if [ -e /dev/kvm ]; then
		QEMU_ACCEL="kvm"
	elif [ "$(uname)" = "Darwin" ] && [ "$(sysctl -n kern.hv_support 2>/dev/null || true)" = "1" ]; then
		QEMU_ACCEL="hvf"
	fi
fi
QEMU_ACCEL="${QEMU_ACCEL:-tcg}"
if [ "${QEMU_ACCEL}" != "tcg" ] && [ "${TARGET_ARCH}" != "${HOST_ARCH}" ]; then
	echo "${QEMU_ACCEL} cannot run a ${TARGET_ARCH} guest on a ${HOST_ARCH} host" >&2
	exit 1
fi

if [ "${SKIP_BUILD}" != "1" ]; then
	cargo build --release --target "${RUST_TARGET}" -p uemulator-uefi
fi

if [ "${TARGET_ARCH}" = "x86_64" ]; then
	BOOT_NAME="BOOTX64.EFI"
	FIRMWARE_NAME="edk2-x86_64-code.fd"
else
	BOOT_NAME="BOOTAA64.EFI"
	FIRMWARE_NAME="edk2-aarch64-code.fd"
fi

if [ -z "${QEMU_FIRMWARE}" ]; then
	for dir in \
		"$(brew --prefix qemu 2>/dev/null || true)/share/qemu" \
		/usr/share/qemu /usr/share/edk2/ovmf /usr/share/edk2/aarch64 /usr/share/OVMF /usr/share/AAVMF; do
		for name in "${FIRMWARE_NAME}" OVMF_CODE.fd OVMF.fd QEMU_EFI.fd AAVMF_CODE.fd; do
			if [ -f "${dir}/${name}" ]; then
				QEMU_FIRMWARE="${dir}/${name}"
				break 2
			fi
		done
	done
fi
if [ -z "${QEMU_FIRMWARE}" ]; then
	echo "no UEFI firmware found, set QEMU_FIRMWARE" >&2
	exit 1
fi

OUT="target/uefi/${TARGET_ARCH}"
ESP="${OUT}/esp"
rm -rf "${ESP}"
mkdir -p "${ESP}/EFI/BOOT"
cp "target/${RUST_TARGET}/release/uemulator-uefi.efi" "${ESP}/EFI/BOOT/${BOOT_NAME}"
ln "${KERNEL}" "${ESP}/vmlinuz" 2>/dev/null || cp "${KERNEL}" "${ESP}/vmlinuz"
ln "${INITRD}" "${ESP}/initramfs" 2>/dev/null || cp "${INITRD}" "${ESP}/initramfs"
{
	echo "mem=${GUEST_MEM}"
	[ -n "${UEFI_ACCEL}" ] && echo "accel=${UEFI_ACCEL}"
	[ "${#}" -gt 0 ] && echo "-- ${*}"
} >"${ESP}/uemulator.cfg"

# The aarch64 firmware needs a flash image of exactly 64 MiB.
FIRMWARE="${OUT}/firmware.fd"
cp "${QEMU_FIRMWARE}" "${FIRMWARE}"
if [ "${TARGET_ARCH}" = "aarch64" ]; then
	dd if=/dev/zero of="${FIRMWARE}" bs=1 count=1 seek=67108863 conv=notrunc >/dev/null 2>&1
fi

if [ "${TARGET_ARCH}" = "x86_64" ]; then
	# TCG can emulate SVM (not VMX), which lets the hypervisor path run without hardware support.
	CPU="max,svm=on"
	[ "${QEMU_ACCEL}" != "tcg" ] && CPU="host"
	set -- qemu-system-x86_64 -M q35 -cpu "${CPU}"
else
	CPU="max"
	[ "${QEMU_ACCEL}" != "tcg" ] && CPU="host"
	set -- qemu-system-aarch64 -M virt -cpu "${CPU}"
fi

# shellcheck disable=SC2086
exec "${@}" \
	-accel "${QEMU_ACCEL}" \
	-nodefaults -nographic -serial mon:stdio \
	-m "${QEMU_MEMORY}" -smp 1 \
	-drive "if=pflash,format=raw,readonly=on,file=${FIRMWARE}" \
	-drive "if=none,id=esp,format=raw,file=fat:rw:${ESP}" \
	-device virtio-blk-pci,drive=esp \
	${QEMU_EXTRA_ARGS}
