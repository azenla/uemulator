modprobe ext4
modprobe vfat
mount /dev/vda2 /mnt
mount /dev/vda1 /mnt/boot/efi
printf 'set timeout=0\nserial --unit=0 --speed=115200\nterminal_input serial\nterminal_output serial\ninsmod part_gpt\ninsmod fat\nsearch --no-floppy --file --set=esp /xen.cfg\nsource ($esp)/xen.cfg\n' > /mnt/boot/grub/grub.cfg
dd if=/dev/zero bs=1024 count=8 | tr '\0' '#' > /mnt/boot/efi/xen.cfg
cat /mnt/boot/grub/grub.cfg
sync
umount /mnt/boot/efi
umount /mnt
