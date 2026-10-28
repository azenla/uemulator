#!wait 900
modprobe ext4
modprobe vfat
modprobe nls_cp437
modprobe nls_iso8859_1
mount /dev/vda2 /mnt
mount /dev/vda1 /mnt/boot/efi
mkdir -p /mnt/boot/efi/EFI/xen
cp /mnt/usr/lib/efi/xen.efi /mnt/boot/efi/EFI/xen/xen.efi
cp /mnt/boot/vmlinuz-lts /mnt/boot/efi/EFI/xen/vmlinuz
cp /mnt/boot/initramfs-lts /mnt/boot/efi/EFI/xen/initrd
printf '[global]\ndefault=alpine\n[alpine]\noptions=dom0_mem=1024M,max:1024M console=com1 com1=115200,8n1 loglvl=all guest_loglvl=all iommu=debug dom0=pvh\nkernel=vmlinuz root=/dev/vda2 rootfstype=ext4 rootdelay=5 rw console=hvc0\nramdisk=initrd\n' > /mnt/boot/efi/EFI/xen/xen.cfg
ls -l /mnt/boot/efi/EFI/xen
sync
umount /mnt/boot/efi
umount /mnt
