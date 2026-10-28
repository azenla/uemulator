modprobe ext4
modprobe vfat
mount /dev/vda2 /mnt
ls -l /mnt/boot /mnt/boot/grub
printf 'set timeout=0\nserial --unit=0 --speed=115200\nterminal_input serial\nterminal_output serial\nmenuentry "Xen" {\n  insmod part_gpt\n  insmod ext2\n  search --no-floppy --file --set=root /boot/xen.gz\n  multiboot2 /boot/xen.gz dom0_mem=1024M,max:1024M console=com1 com1=115200,8n1 loglvl=all guest_loglvl=all iommu=debug\n  module2 /boot/vmlinuz-lts root=/dev/vda2 rw console=hvc0 earlyprintk=xen\n  module2 /boot/initramfs-lts\n}\n' > /mnt/boot/grub/grub.cfg
cat /mnt/boot/grub/grub.cfg
sync
umount /mnt
