/* The image the library loads at FIRMWARE_BASE (crates/uemulator/src/boot/uefi.rs): a flat
   binary whose first 24 bytes are the header the loader checks. The zero-initialised data is
   not in the file; the header's footprint tells the library to reserve it. */
ENTRY(_start)

SECTIONS
{
    . = 0x100000;
    __image_start = .;

    .header : {
        BYTE(0x55) BYTE(0x45) BYTE(0x4d) BYTE(0x55) BYTE(0x46) BYTE(0x57) BYTE(0x36) BYTE(0x34)
        LONG(1)
        LONG(_start - __image_start)
        QUAD(__image_end - __image_start)
    }
    .text : ALIGN(16) { *(.text._start) *(.text .text.*) }
    .rodata : ALIGN(16) { *(.rodata .rodata.*) }
    .data : ALIGN(16) { *(.data .data.*) }
    .bss : ALIGN(4096) {
        __bss_start = .;
        *(.bss .bss.*) *(COMMON)
        . = ALIGN(4096);
    }
    __image_end = .;

    /DISCARD/ : { *(.comment) *(.eh_frame*) *(.note*) *(.got*) }
}
