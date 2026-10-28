#!wait 1800
apk add e2fsprogs dosfstools sfdisk grub-efi
printf 'label: gpt\n,256M,U\n,,L\n' | sfdisk /dev/vda
mdev -s
modprobe ext4
modprobe vfat
modprobe nls_cp437
modprobe nls_iso8859_1
mkfs.vfat -F 32 /dev/vda1
mkfs.ext4 -F -q /dev/vda2
mount /dev/vda2 /mnt
mkdir -p /mnt/boot/efi
mount /dev/vda1 /mnt/boot/efi
cp /etc/apk/repositories /tmp/repos
echo https://dl-cdn.alpinelinux.org/alpine/latest-stable/community >> /tmp/repos
apk --root /mnt --initdb --repositories-file /tmp/repos --keys-dir /etc/apk/keys add alpine-base linux-lts linux-firmware-none e2fsprogs xen xen-hypervisor xen-qemu grub-efi openrc busybox-openrc
ls -l /mnt/boot /mnt/usr/lib/xen/boot /mnt/boot/efi
for d in dev proc sys; do mount --bind /$d /mnt/$d; done
cp /etc/resolv.conf /mnt/etc/resolv.conf
cp /tmp/repos /mnt/etc/apk/repositories
echo 'root:xen' | chroot /mnt chpasswd
echo localhost > /mnt/etc/hostname
printf 'auto lo\niface lo inet loopback\nauto eth0\niface eth0 inet dhcp\n' > /mnt/etc/network/interfaces
printf '/dev/vda2 / ext4 defaults 0 1\n/dev/vda1 /boot/efi vfat defaults 0 2\n' > /mnt/etc/fstab
printf 'hvc0::respawn:/sbin/getty 38400 hvc0\nttyS0::respawn:/sbin/getty 115200 ttyS0\n' >> /mnt/etc/inittab
for s in devfs dmesg mdev hwdrivers; do chroot /mnt rc-update add $s sysinit; done
for s in modules sysctl hostname bootmisc syslog networking; do chroot /mnt rc-update add $s boot; done
for s in xenstored xenconsoled; do chroot /mnt rc-update add $s default; done
grub-install --target=x86_64-efi --efi-directory=/mnt/boot/efi --boot-directory=/mnt/boot --removable --no-nvram
ls -R /mnt/boot/efi
sync
umount /mnt/boot/efi
umount /mnt/dev /mnt/proc /mnt/sys
umount /mnt
