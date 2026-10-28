# The instruction list for hack/gen_avx512.py (exec'd by it): form(NAME, mnemonic, kind, bits, imm).
for n, m, b in [("VPADDD", "vpaddd", 32), ("VPADDQ", "vpaddq", 64), ("VPSUBD", "vpsubd", 32),
                ("VPSUBQ", "vpsubq", 64), ("VPANDD", "vpandd", 32), ("VPANDQ", "vpandq", 64),
                ("VPORD", "vpord", 32), ("VPXORQ", "vpxorq", 64), ("VPANDNQ", "vpandnq", 64),
                ("VPADDB", "vpaddb", 0), ("VPADDW", "vpaddw", 0), ("VPMULLD", "vpmulld", 32),
                ("VADDPS", "vaddps", 32), ("VADDPD", "vaddpd", 64), ("VMULPS", "vmulps", 32),
                ("VSUBPD", "vsubpd", 64), ("VDIVPS", "vdivps", 32), ("VMINPS", "vminps", 32),
                ("VMAXPD", "vmaxpd", 64), ("VPUNPCKLDQ", "vpunpckldq", 32),
                ("VPUNPCKHQDQ", "vpunpckhqdq", 64), ("VUNPCKLPS", "vunpcklps", 32),
                ("VPSHUFB", "vpshufb", 0), ("VPMAXSD", "vpmaxsd", 32), ("VPMINUD", "vpminud", 32)]:
    form(n, m, "bin", b)
for n, m, b in [("VSQRTPS", "vsqrtps", 32), ("VSQRTPD", "vsqrtpd", 64), ("VPABSD", "vpabsd", 32),
                ("VCVTDQ2PS", "vcvtdq2ps", 32), ("VMOVSLDUP", "vmovsldup", 0)]:
    form(n, m, "un", b)
form("VSHUFPS", "vshufps", "bin", 32, 0x1b)
form("VPSHUFD", "vpshufd", "un", 32, 0x1b)
form("VPALIGNR", "vpalignr", "bin", 0, 5)
for n, m in [("VMOVDQA32", "vmovdqa32"), ("VMOVDQU32", "vmovdqu32"), ("VMOVDQU64", "vmovdqu64"),
             ("VMOVDQU8", "vmovdqu8"), ("VMOVDQU16", "vmovdqu16"), ("VMOVUPS", "vmovups"),
             ("VMOVAPD", "vmovapd")]:
    form(n, m, "un", 0)

single("VPADDD_DISP8", "vpaddd zmm0, zmm1, [rsi+128]")
single("VPADDD_DISP8_NEGATIVE", "vpaddd zmm0, zmm1, [rsi-64]")
single("VPADDD_BCAST_DISP8", "vpaddd zmm0, zmm1, [rsi+8]{1to16}")
single("VPADDD_Y_DISP8", "vpaddd ymm0, ymm1, [rsi+64]")
single("VPADDD_X_DISP8", "vpaddd xmm0, xmm1, [rsi+16]")
single("VADDPS_RN_SAE", "vaddps zmm0, zmm1, zmm2, {rn-sae}")
single("VADDPS_RD_SAE", "vaddps zmm0, zmm1, zmm2, {rd-sae}")
single("VADDPS_RU_SAE", "vaddps zmm0, zmm1, zmm2, {ru-sae}")
single("VADDPS_RZ_SAE", "vaddps zmm0, zmm1, zmm2, {rz-sae}")
single("VADDPS_K1_RD_SAE", "vaddps zmm0{k1}{z}, zmm1, zmm2, {rd-sae}")
single("VMAXPS_SAE", "vmaxps zmm0, zmm1, zmm2, {sae}")
single("VADDSS_RD_SAE", "vaddss xmm0, xmm1, xmm2, {rd-sae}")
single("VADDSS", "vaddss xmm0, xmm1, xmm2")
single("VADDSS_K1", "vaddss xmm0{k1}, xmm1, xmm2")
single("VADDSS_K1Z", "vaddss xmm0{k1}{z}, xmm1, xmm2")
single("VADDSD_M", "vaddsd xmm0, xmm1, [rsi]")
single("VADDSD_M_DISP8", "vaddsd xmm0, xmm1, [rsi+16]")
single("VPSLLD_COUNT_Z", "vpslld zmm0, zmm1, xmm2")
single("VPSLLD_COUNT_Z_M", "vpslld zmm0, zmm1, [rsi]")
single("VPSLLQ_COUNT_Y_K", "vpsllq ymm0{k1}, ymm1, xmm2")
single("VMOVD_X_EAX", "vmovd xmm0, eax")
single("VMOVQ_X_RAX", "vmovq xmm0, rax")
single("VMOVD_EAX_X", "vmovd eax, xmm1")
single("VMOVQ_RAX_X", "vmovq rax, xmm1")
single("VMOVD_X_M", "vmovd xmm0, [rsi]")
single("VMOVQ_X_X", "vmovq xmm0, xmm1")
single("VMOVQ_X_M", "vmovq xmm0, [rsi]")
single("VMOVQ_M_X", "vmovq [rdi], xmm1")
single("VMOVSS_X_X_X", "vmovss xmm0, xmm1, xmm2")
single("VMOVSS_X_M", "vmovss xmm0, [rsi]")
single("VMOVSS_X_M_K", "vmovss xmm0{k1}, [rsi]")
single("VMOVSS_X_M_KZ", "vmovss xmm0{k1}{z}, [rsi]")
single("VMOVSS_M_X_K", "vmovss [rdi]{k1}, xmm1")
single("VMOVSD_X_X_X_K", "vmovsd xmm0{k1}, xmm1, xmm2")
single("VMOVSD_X_M", "vmovsd xmm0, [rsi]")
single("VMOVNTDQ_M_Z", "vmovntdq [rdi], zmm1")
single("VMOVAPS_M_Z_K", "vmovaps [rdi]{k1}, zmm1")
single("VMOVUPS_M_Z_K", "vmovups [rdi]{k1}, zmm1")
single("VMOVUPS_Z_M_K", "vmovups zmm0{k1}, [rsi]")
single("VMOVUPS_Z_M_KZ", "vmovups zmm0{k1}{z}, [rsi]")
single("VMOVDQU8_M_Z_K", "vmovdqu8 [rdi]{k1}, zmm1")
single("VMOVDQU16_Z_M_KZ", "vmovdqu16 zmm0{k1}{z}, [rsi]")
single("VMOVDQA64_Z_M", "vmovdqa64 zmm0, [rsi]")
single("VMOVDQA64_M_Z", "vmovdqa64 [rdi], zmm1")
single("VMOVDQU64_HIGH", "vmovdqu64 zmm30, zmm17")
single("VMOVDQU64_HIGH_M", "vmovdqu64 zmm31, [rsi]")
single("VMOVDQU64_M_HIGH", "vmovdqu64 [rdi], zmm29")
for w in "bwdq":
    for op in ["kand", "kandn", "kor", "kxnor", "kxor", "kadd"]:
        single(f"{op}{w}".upper(), f"{op}{w} k1, k2, k3")
    for op in ["knot", "kortest", "ktest"]:
        single(f"{op}{w}".upper(), f"{op}{w} k1, k2")
    single(f"KMOV{w}_KK".upper(), f"kmov{w} k1, k2")
    single(f"KMOV{w}_KM".upper(), f"kmov{w} k1, [rsi]")
    single(f"KMOV{w}_MK".upper(), f"kmov{w} [rdi], k1")
    single(f"KMOV{w}_KR".upper(), f"kmov{w} k1, {'rax' if w == 'q' else 'eax'}")
    single(f"KMOV{w}_RK".upper(), f"kmov{w} {'rax' if w == 'q' else 'eax'}, k1")
    single(f"KSHIFTL{w}_3".upper(), f"kshiftl{w} k1, k2, 3")
    single(f"KSHIFTR{w}_3".upper(), f"kshiftr{w} k1, k2, 3")
    single(f"KSHIFTL{w}_9".upper(), f"kshiftl{w} k1, k2, 9")
    single(f"KSHIFTR{w}_9".upper(), f"kshiftr{w} k1, k2, 9")
    single(f"KSHIFTL{w}_70".upper(), f"kshiftl{w} k1, k2, 70")
single("KUNPCKBW", "kunpckbw k1, k2, k3")
single("KUNPCKWD", "kunpckwd k1, k2, k3")
single("KUNPCKDQ", "kunpckdq k1, k2, k3")
single("VPGATHERDD_HIGH_INDEX", "vpgatherdd zmm0{k1}, [rsi + zmm20*4]")
