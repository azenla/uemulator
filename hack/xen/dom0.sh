#!wait 1200
uname -a
xl info
xl list
printf 'name = "d1"\ntype = "pvh"\nmemory = 256\nvcpus = 1\nkernel = "/boot/vmlinuz-lts"\nramdisk = "/boot/initramfs-lts"\nextra = "console=hvc0"\n' > /root/d1.cfg
timeout 240 xl create -c /root/d1.cfg
xl list
xl dmesg | tail -25
