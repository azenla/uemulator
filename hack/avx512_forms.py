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

# ---- stage 2: integer
for n, m, b in [("VPMINSQ", "vpminsq", 64), ("VPMAXSQ", "vpmaxsq", 64), ("VPMINUQ", "vpminuq", 64),
                ("VPMAXUQ", "vpmaxuq", 64), ("VPMULLQ", "vpmullq", 64), ("VPSRLVD", "vpsrlvd", 32),
                ("VPSRLVQ", "vpsrlvq", 64), ("VPSRAVD", "vpsravd", 32), ("VPSRAVQ", "vpsravq", 64),
                ("VPSLLVD", "vpsllvd", 32), ("VPSLLVQ", "vpsllvq", 64), ("VPSRLVW", "vpsrlvw", 0),
                ("VPSRAVW", "vpsravw", 0), ("VPSLLVW", "vpsllvw", 0), ("VPRORVD", "vprorvd", 32),
                ("VPROLVD", "vprolvd", 32), ("VPRORVQ", "vprorvq", 64), ("VPROLVQ", "vprolvq", 64),
                ("VPMULUDQ", "vpmuludq", 64), ("VPMULDQ", "vpmuldq", 64), ("VPMADDWD", "vpmaddwd", 0),
                ("VPMADDUBSW", "vpmaddubsw", 0), ("VPMULHRSW", "vpmulhrsw", 0), ("VPACKSSDW", "vpackssdw", 32),
                ("VPACKUSWB", "vpackuswb", 0), ("VPUNPCKLBW", "vpunpcklbw", 0), ("VPUNPCKHWD", "vpunpckhwd", 0),
                ("VPAVGB", "vpavgb", 0), ("VPSUBUSW", "vpsubusw", 0), ("VPADDSB", "vpaddsb", 0),
                ("VPMAXUB", "vpmaxub", 0), ("VPMINSW", "vpminsw", 0), ("VPMULHUW", "vpmulhuw", 0),
                ("VPMULLW", "vpmullw", 0), ("VPSHUFB_", "vpshufb", 0), ("VPANDND", "vpandnd", 32),
                ("VPXORD", "vpxord", 32), ("VPORQ", "vporq", 64), ("VPADDUSB", "vpaddusb", 0),
                ("VPSUBSB", "vpsubsb", 0), ("VPMINUW", "vpminuw", 0), ("VPMAXSB", "vpmaxsb", 0)]:
    form(n, m, "bin", b)
for n, m, b in [("VPABSQ", "vpabsq", 64), ("VPLZCNTD", "vplzcntd", 32), ("VPLZCNTQ", "vplzcntq", 64),
                ("VPOPCNTD", "vpopcntd", 32), ("VPOPCNTQ", "vpopcntq", 64), ("VPOPCNTB", "vpopcntb", 0),
                ("VPOPCNTW", "vpopcntw", 0), ("VPCONFLICTD", "vpconflictd", 32),
                ("VPCONFLICTQ", "vpconflictq", 64), ("VPABSB", "vpabsb", 0), ("VPABSW", "vpabsw", 0)]:
    form(n, m, "un", b)
for n, m, b, i in [("VPSRLD_I", "vpsrld", 32, 5), ("VPSLLD_I", "vpslld", 32, 7), ("VPSRAD_I", "vpsrad", 32, 9),
                   ("VPSRAQ_I", "vpsraq", 64, 3), ("VPSRLQ_I", "vpsrlq", 64, 11), ("VPSLLQ_I", "vpsllq", 64, 63),
                   ("VPROLD_I", "vprold", 32, 5), ("VPRORD_I", "vprord", 32, 9), ("VPROLQ_I", "vprolq", 64, 17),
                   ("VPRORQ_I", "vprorq", 64, 40), ("VPSRLW_I", "vpsrlw", 0, 3), ("VPSRAW_I", "vpsraw", 0, 20),
                   ("VPSLLW_I", "vpsllw", 0, 4), ("VPSLLD_BIG_I", "vpslld", 32, 40), ("VPSRAD_BIG_I", "vpsrad", 32, 70)]:
    form(n, m, "un", b, i)
single("VPSRAQ_COUNT_Z", "vpsraq zmm0, zmm1, xmm2")
single("VPSRAQ_COUNT_Y_K", "vpsraq ymm0{k1}, ymm1, xmm2")
single("VPSRAQ_COUNT_Z_M", "vpsraq zmm0, zmm1, [rsi]")
single("VPSLLDQ_Z", "vpslldq zmm0, zmm2, 5")
single("VPSRLDQ_Z", "vpsrldq zmm0, zmm2, 3")
single("VPSLLDQ_Y_M", "vpslldq ymm0, [rsi], 4")
for n, m, b in [("VPCMPEQB", "vpcmpeqb", 0), ("VPCMPEQW", "vpcmpeqw", 0), ("VPCMPEQD", "vpcmpeqd", 32),
                ("VPCMPEQQ", "vpcmpeqq", 64), ("VPCMPGTB", "vpcmpgtb", 0), ("VPCMPGTW", "vpcmpgtw", 0),
                ("VPCMPGTD", "vpcmpgtd", 32), ("VPCMPGTQ", "vpcmpgtq", 64), ("VPTESTMD", "vptestmd", 32),
                ("VPTESTMQ", "vptestmq", 64), ("VPTESTMB", "vptestmb", 0), ("VPTESTMW", "vptestmw", 0),
                ("VPTESTNMD", "vptestnmd", 32), ("VPTESTNMQ", "vptestnmq", 64), ("VPTESTNMB", "vptestnmb", 0)]:
    form(n, m, "cmp", b)
for suffix, bits in [("B", 0), ("W", 0), ("D", 32), ("Q", 64)]:
    for u in ("", "U"):
        for predicate in range(8):
            form(f"VPCMP{u}{suffix}_{predicate}", f"vpcmp{u.lower()}{suffix.lower()}", "cmp", bits, predicate)
form("VPTERNLOGD", "vpternlogd", "bin", 32, 0xca)
form("VPTERNLOGQ", "vpternlogq", "bin", 64, 0x96)
form("VALIGND", "valignd", "bin", 32, 3)
form("VALIGNQ", "valignq", "bin", 64, 5)
form("VSHUFF32X4", "vshuff32x4", "bin", 32, 0xb1)
form("VSHUFI64X2", "vshufi64x2", "bin", 64, 0x1b)
for n, m, b in [("VPERMD", "vpermd", 32), ("VPERMQ_V", "vpermq", 64), ("VPERMPS", "vpermps", 32),
                ("VPERMPD_V", "vpermpd", 64), ("VPERMI2D", "vpermi2d", 32), ("VPERMI2Q", "vpermi2q", 64),
                ("VPERMI2PS", "vpermi2ps", 32), ("VPERMT2D", "vpermt2d", 32), ("VPERMT2Q", "vpermt2q", 64),
                ("VPERMT2PD", "vpermt2pd", 64), ("VPERMILPS_V", "vpermilps", 32), ("VPERMILPD_V", "vpermilpd", 64),
                ("VPERMW", "vpermw", 0), ("VPERMB", "vpermb", 0), ("VPERMI2W", "vpermi2w", 0),
                ("VPERMI2B", "vpermi2b", 0), ("VPERMT2B", "vpermt2b", 0), ("VPERMT2W", "vpermt2w", 0),
                ("VPBLENDMD", "vpblendmd", 32), ("VPBLENDMQ", "vpblendmq", 64), ("VPBLENDMB", "vpblendmb", 0),
                ("VPBLENDMW", "vpblendmw", 0), ("VBLENDMPS", "vblendmps", 32)]:
    form(n, m, "bin", b)
form("VPERMQ_I", "vpermq", "un", 64, 0x1b)
form("VPERMPD_I", "vpermpd", "un", 64, 0xc6)
form("VPERMILPS_I", "vpermilps", "un", 32, 0x1b)
form("VPERMILPD_I", "vpermilpd", "un", 64, 0x5)
for n, m, r in [("VPMOVZXBW", "vpmovzxbw", 2), ("VPMOVSXBD", "vpmovsxbd", 4), ("VPMOVZXBQ", "vpmovzxbq", 8),
                ("VPMOVSXWD", "vpmovsxwd", 2), ("VPMOVZXWQ", "vpmovzxwq", 4), ("VPMOVSXDQ", "vpmovsxdq", 2),
                ("VPMOVZXDQ", "vpmovzxdq", 2), ("VPMOVSXBW", "vpmovsxbw", 2)]:
    form(n, m, f"widen{r}", 0)
for n, m, r in [("VPMOVDB", "vpmovdb", 4), ("VPMOVQD", "vpmovqd", 2), ("VPMOVWB", "vpmovwb", 2),
                ("VPMOVQB", "vpmovqb", 8), ("VPMOVQW", "vpmovqw", 4), ("VPMOVDW", "vpmovdw", 2),
                ("VPMOVSDB", "vpmovsdb", 4), ("VPMOVUSDB", "vpmovusdb", 4), ("VPMOVSQD", "vpmovsqd", 2),
                ("VPMOVUSQW", "vpmovusqw", 4), ("VPMOVSWB", "vpmovswb", 2), ("VPMOVUSWB", "vpmovuswb", 2),
                ("VPMOVUSQD", "vpmovusqd", 2), ("VPMOVSQB", "vpmovsqb", 8)]:
    form(n, m, f"narrow{r}", 0)
for n, asm in [("VPCOMPRESSD_R", "vpcompressd zmm0{k1}, zmm1"), ("VPCOMPRESSD_RZ", "vpcompressd zmm0{k1}{z}, zmm1"),
               ("VPCOMPRESSD_M", "vpcompressd [rdi]{k1}, zmm1"), ("VCOMPRESSPD_M", "vcompresspd [rdi]{k1}, zmm1"),
               ("VPCOMPRESSQ_Y_R", "vpcompressq ymm0{k1}, ymm1"), ("VPCOMPRESSB_M", "vpcompressb [rdi]{k1}, zmm1"),
               ("VPCOMPRESSW_R", "vpcompressw zmm0{k1}, zmm1"), ("VPEXPANDD_R", "vpexpandd zmm0{k1}, zmm1"),
               ("VPEXPANDD_RZ", "vpexpandd zmm0{k1}{z}, zmm1"), ("VPEXPANDD_M", "vpexpandd zmm0{k1}, [rsi]"),
               ("VPEXPANDD_MZ", "vpexpandd zmm0{k1}{z}, [rsi]"), ("VEXPANDPS_M", "vexpandps zmm0{k1}, [rsi]"),
               ("VPEXPANDQ_Y_M", "vpexpandq ymm0{k1}, [rsi]"), ("VPEXPANDB_M", "vpexpandb zmm0{k1}, [rsi]"),
               ("VPEXPANDW_R", "vpexpandw zmm0{k1}, zmm1"),
               ("VPMOVM2B", "vpmovm2b zmm0, k1"), ("VPMOVM2W", "vpmovm2w ymm0, k1"), ("VPMOVM2D", "vpmovm2d zmm0, k1"),
               ("VPMOVM2Q", "vpmovm2q xmm0, k1"), ("VPMOVB2M", "vpmovb2m k2, zmm1"), ("VPMOVW2M", "vpmovw2m k2, ymm1"),
               ("VPMOVD2M", "vpmovd2m k2, zmm1"), ("VPMOVQ2M", "vpmovq2m k2, xmm1"),
               ("VPMADD52LUQ", "vpmadd52luq zmm0, zmm1, zmm2"), ("VPMADD52HUQ", "vpmadd52huq zmm0{k1}, zmm1, zmm2"),
               ("VPDPBUSD", "vpdpbusd zmm0, zmm1, zmm2"), ("VPDPBUSDS", "vpdpbusds zmm0, zmm1, zmm2"),
               ("VPDPWSSD", "vpdpwssd zmm0, zmm1, zmm2"), ("VPDPWSSDS", "vpdpwssds zmm0{k1}, zmm1, zmm2"),
               ("VPSHUFBITQMB", "vpshufbitqmb k2, zmm1, zmm2"), ("VPMULTISHIFTQB", "vpmultishiftqb zmm0, zmm1, zmm2"),
               ("VPSHLDW", "vpshldw zmm0, zmm1, zmm2, 5"), ("VPSHLDD", "vpshldd zmm0, zmm1, zmm2, 9"),
               ("VPSHRDQ", "vpshrdq zmm0, zmm1, zmm2, 13"), ("VPSHRDW", "vpshrdw zmm0, zmm1, zmm2, 3"),
               ("VPSHLDVD", "vpshldvd zmm0, zmm1, zmm2"), ("VPSHRDVQ", "vpshrdvq zmm0, zmm1, zmm2"),
               ("VPSHLDVW", "vpshldvw zmm0, zmm1, zmm2"), ("VPSHRDVD", "vpshrdvd zmm0, zmm1, zmm2"),
               ("VGF2P8MULB", "vgf2p8mulb zmm0, zmm1, zmm2"), ("VGF2P8AFFINEQB", "vgf2p8affineqb zmm0, zmm1, zmm2, 0x63"),
               ("VGF2P8AFFINEINVQB", "vgf2p8affineinvqb zmm0, zmm1, zmm2, 0x63"),
               ("VPMOVZXBW_M_Z", "vpmovzxbw zmm0{k1}{z}, [rsi]"),
               ("VPSRLW_COUNT_Z_M", "vpsrlw zmm0, zmm1, [rsi]")]:
    single(n, asm)

# ---- stage 2: floating point
for form_kind in ("132", "213", "231"):
    for op in ("fmadd", "fmsub", "fnmadd", "fnmsub", "fmaddsub", "fmsubadd"):
        form(f"V{op.upper()}{form_kind}PS", f"v{op}{form_kind}ps", "bin", 32)
        form(f"V{op.upper()}{form_kind}PD", f"v{op}{form_kind}pd", "bin", 64)
    for op in ("fmadd", "fmsub", "fnmadd", "fnmsub"):
        single(f"V{op.upper()}{form_kind}SS", f"v{op}{form_kind}ss xmm0, xmm1, xmm2")
        single(f"V{op.upper()}{form_kind}SD", f"v{op}{form_kind}sd xmm0, xmm1, xmm2")
single("VFMADD231PS_RZ", "vfmadd231ps zmm0, zmm1, zmm2, {rz-sae}")
single("VFMADD231PS_RU", "vfmadd231ps zmm0, zmm1, zmm2, {ru-sae}")
single("VFMADD231SS_K", "vfmadd231ss xmm0{k1}, xmm1, xmm2")
single("VFMADD231SS_KZ", "vfmadd231ss xmm0{k1}{z}, xmm1, xmm2")
single("VFMADD231SD_RD", "vfmadd231sd xmm0, xmm1, xmm2, {rd-sae}")
single("VFMADD213SS_M", "vfmadd213ss xmm0, xmm1, [rsi]")
for n, m, b in [("VRCP14PS", "vrcp14ps", 32), ("VRCP14PD", "vrcp14pd", 64), ("VRSQRT14PS", "vrsqrt14ps", 32),
                ("VRSQRT14PD", "vrsqrt14pd", 64), ("VGETEXPPS", "vgetexpps", 32), ("VGETEXPPD", "vgetexppd", 64)]:
    form(n, m, "un", b)
form("VGETMANTPS", "vgetmantps", "un", 32, 0)
form("VGETMANTPD_HALF", "vgetmantpd", "un", 64, 1)
form("VGETMANTPS_34", "vgetmantps", "un", 32, 3)
form("VGETMANTPS_NOSIGN", "vgetmantps", "un", 32, 4)
form("VGETMANTPS_NANNEG", "vgetmantps", "un", 32, 8)
form("VSCALEFPS", "vscalefps", "bin", 32)
form("VSCALEFPD", "vscalefpd", "bin", 64)
form("VFIXUPIMMPS", "vfixupimmps", "bin", 32, 0)
form("VFIXUPIMMPD_FLAGS", "vfixupimmpd", "bin", 64, 0xff)
form("VRNDSCALEPS", "vrndscaleps", "un", 32, 0x0b)
form("VRNDSCALEPS_SCALE", "vrndscaleps", "un", 32, 0x41)
form("VRNDSCALEPD", "vrndscalepd", "un", 64, 0x09)
form("VRNDSCALEPD_MXCSR", "vrndscalepd", "un", 64, 0x04)
form("VREDUCEPS", "vreduceps", "un", 32, 0x0b)
form("VREDUCEPD", "vreducepd", "un", 64, 0x08)
form("VRANGEPS_MIN", "vrangeps", "bin", 32, 0)
form("VRANGEPD_MAXABS_NEG", "vrangepd", "bin", 64, 0x0f)
form("VRANGEPS_MINABS", "vrangeps", "bin", 32, 0x06)
form("VFPCLASSPS", "vfpclassps", "cmpun", 32, 0xff)
form("VFPCLASSPD_NAN", "vfpclasspd", "cmpun", 64, 0x81)
form("VFPCLASSPS_INF", "vfpclassps", "cmpun", 32, 0x18)
form("VFPCLASSPS_NEG", "vfpclassps", "cmpun", 32, 0x56)
for p_ in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 21, 24, 29, 31):
    form(f"VCMPPS_{p_}", "vcmpps", "cmp", 32, p_)
    form(f"VCMPPD_{p_}", "vcmppd", "cmp", 64, p_)
for n, m, k, b in [("VCVTUDQ2PS", "vcvtudq2ps", "un", 32), ("VCVTPS2UDQ", "vcvtps2udq", "un", 32),
                   ("VCVTTPS2UDQ", "vcvttps2udq", "un", 32), ("VCVTPS2DQ", "vcvtps2dq", "un", 32),
                   ("VCVTTPS2DQ", "vcvttps2dq", "un", 32), ("VCVTQQ2PD", "vcvtqq2pd", "un", 64),
                   ("VCVTUQQ2PD", "vcvtuqq2pd", "un", 64), ("VCVTPD2QQ", "vcvtpd2qq", "un", 64),
                   ("VCVTTPD2QQ", "vcvttpd2qq", "un", 64), ("VCVTPD2UQQ", "vcvtpd2uqq", "un", 64),
                   ("VCVTTPD2UQQ", "vcvttpd2uqq", "un", 64),
                   ("VCVTDQ2PD", "vcvtdq2pd", "widen2", 0), ("VCVTUDQ2PD", "vcvtudq2pd", "widen2", 0),
                   ("VCVTPS2PD", "vcvtps2pd", "widen2", 0), ("VCVTPS2QQ", "vcvtps2qq", "widen2", 0),
                   ("VCVTTPS2UQQ", "vcvttps2uqq", "widen2", 0), ("VCVTPH2PS", "vcvtph2ps", "widen2", 0),
                   ("VCVTPD2PS", "vcvtpd2ps", "cvtnarrow2", 64), ("VCVTPD2DQ", "vcvtpd2dq", "cvtnarrow2", 64),
                   ("VCVTTPD2UDQ", "vcvttpd2udq", "cvtnarrow2", 64), ("VCVTQQ2PS", "vcvtqq2ps", "cvtnarrow2", 64),
                   ("VCVTUQQ2PS", "vcvtuqq2ps", "cvtnarrow2", 64), ("VCVTTPD2DQ", "vcvttpd2dq", "cvtnarrow2", 64)]:
    form(n, m, k, b)
form("VCVTPS2PH", "vcvtps2ph", "narrow2", 0, 0)
form("VCVTPS2PH_UP", "vcvtps2ph", "narrow2", 0, 2)
for n, asm in [("VCVTSI2SS_Q", "vcvtsi2ss xmm0, xmm1, rax"), ("VCVTSI2SD_L", "vcvtsi2sd xmm0, xmm1, eax"),
               ("VCVTSI2SS_L_RD", "vcvtsi2ss xmm0, xmm1, {rd-sae}, eax"),
               ("VCVTUSI2SD_Q", "vcvtusi2sd xmm0, xmm1, rax"), ("VCVTUSI2SS_L", "vcvtusi2ss xmm0, xmm1, eax"),
               ("VCVTSI2SD_M", "vcvtsi2sd xmm0, xmm1, dword ptr [rsi]"),
               ("VCVTSS2SI_Q", "vcvtss2si rax, xmm2"), ("VCVTSS2SI_L_RU", "vcvtss2si eax, xmm2, {ru-sae}"),
               ("VCVTTSD2SI_L", "vcvttsd2si eax, xmm2"), ("VCVTTSD2SI_Q", "vcvttsd2si rax, xmm2"),
               ("VCVTSD2SI_Q", "vcvtsd2si rax, xmm2"),
               ("VCVTSS2USI_Q", "vcvtss2usi rax, xmm2"), ("VCVTTSD2USI_Q", "vcvttsd2usi rax, xmm2"),
               ("VCVTTSS2USI_L", "vcvttss2usi eax, xmm2"), ("VCVTSD2USI_L", "vcvtsd2usi eax, xmm2"),
               ("VCVTSS2SD", "vcvtss2sd xmm0, xmm1, xmm2"), ("VCVTSD2SS", "vcvtsd2ss xmm0, xmm1, xmm2"),
               ("VCVTSD2SS_RD", "vcvtsd2ss xmm0, xmm1, xmm2, {rd-sae}"),
               ("VCVTSS2SD_K", "vcvtss2sd xmm0{k1}{z}, xmm1, xmm2"),
               ("VRCP14SS", "vrcp14ss xmm0, xmm1, xmm2"), ("VRSQRT14SD", "vrsqrt14sd xmm0, xmm1, xmm2"),
               ("VGETEXPSD", "vgetexpsd xmm0, xmm1, xmm2"), ("VGETEXPSS_K", "vgetexpss xmm0{k1}, xmm1, xmm2"),
               ("VGETMANTSS", "vgetmantss xmm0, xmm1, xmm2, 0"), ("VSCALEFSS", "vscalefss xmm0, xmm1, xmm2"),
               ("VSCALEFSD", "vscalefsd xmm0, xmm1, xmm2"),
               ("VRNDSCALESD", "vrndscalesd xmm0, xmm1, xmm2, 0x0b"), ("VRNDSCALESS_K", "vrndscaless xmm0{k1}, xmm1, xmm2, 0x09"),
               ("VREDUCESD", "vreducesd xmm0, xmm1, xmm2, 0x0b"), ("VRANGESS", "vrangess xmm0, xmm1, xmm2, 1"),
               ("VFIXUPIMMSS", "vfixupimmss xmm0, xmm1, xmm2, 0"), ("VFPCLASSSD", "vfpclasssd k2, xmm2, 0xc3"),
               ("VFPCLASSSS_M", "vfpclassss k2, dword ptr [rsi], 0x18"),
               ("VCOMISS_SAE", "vcomiss xmm1, xmm2, {sae}"), ("VUCOMISD", "vucomisd xmm1, xmm2"),
               ("VCOMISD_M", "vcomisd xmm1, qword ptr [rsi]"),
               ("VCMPSS_1", "vcmpss k2, xmm1, xmm2, 1"), ("VCMPSD_12_K", "vcmpsd k2{k1}, xmm1, xmm2, 12"),
               ("VCMPSS_SAE", "vcmpss k2, xmm1, xmm2, {sae}, 1"),
               ("VADDPS_ZERO_MASK_NAN", "vaddps zmm0{k1}{z}, zmm1, zmm2")]:
    single(n, asm)

single("VEX_PINSRD_EAX", "vpinsrd xmm0, xmm1, eax, 1")
single("VEX_PINSRQ_RAX", "vpinsrq xmm0, xmm1, rax, 1")
single("VEX_PINSRW_EAX", "vpinsrw xmm0, xmm1, eax, 2")
single("VEX_PINSRB_EAX", "vpinsrb xmm0, xmm1, eax, 3")
single("VEX_CVTSI2SD_RAX", "vcvtsi2sd xmm0, xmm1, rax")

# ---- stage 2: broadcasts, inserts, extracts, gathers
for n, m in [("VPBROADCASTD", "vpbroadcastd"), ("VPBROADCASTQ", "vpbroadcastq"), ("VPBROADCASTB", "vpbroadcastb"),
             ("VPBROADCASTW", "vpbroadcastw"), ("VBROADCASTSS", "vbroadcastss"), ("VBROADCASTSD", "vbroadcastsd")]:
    form(n, m, "widen999", 0)
form("VBROADCASTF32X2", "vbroadcastf32x2", "widen999", 0)
form("VBROADCASTI32X2", "vbroadcasti32x2", "widen999", 0)
for n, asm in [("VPBROADCASTD_GPR", "vpbroadcastd zmm0{k1}, eax"), ("VPBROADCASTQ_GPR", "vpbroadcastq zmm0, rax"),
               ("VPBROADCASTB_GPR", "vpbroadcastb ymm0{k1}{z}, eax"), ("VPBROADCASTW_GPR", "vpbroadcastw xmm0, eax"),
               ("VBROADCASTF32X4", "vbroadcastf32x4 zmm0{k1}, xmmword ptr [rsi]"),
               ("VBROADCASTF32X4_Y", "vbroadcastf32x4 ymm0, xmmword ptr [rsi]"),
               ("VBROADCASTI64X2", "vbroadcasti64x2 zmm0{k1}{z}, xmmword ptr [rsi]"),
               ("VBROADCASTF64X4", "vbroadcastf64x4 zmm0, ymmword ptr [rsi]"),
               ("VBROADCASTI32X8", "vbroadcasti32x8 zmm0{k1}, ymmword ptr [rsi]"),
               ("VBROADCASTI32X4_DISP8", "vbroadcasti32x4 zmm0, xmmword ptr [rsi+32]"),
               ("VMOVNTDQA_Z", "vmovntdqa zmm0, zmmword ptr [rsi]"),
               ("VINSERTF32X4_Z", "vinsertf32x4 zmm0, zmm1, xmm2, 2"),
               ("VINSERTF32X4_ZK", "vinsertf32x4 zmm0{k1}{z}, zmm1, xmm2, 3"),
               ("VINSERTI64X2_ZM", "vinserti64x2 zmm0{k1}, zmm1, xmmword ptr [rsi], 1"),
               ("VINSERTI32X4_Y", "vinserti32x4 ymm0, ymm1, xmm2, 1"),
               ("VINSERTF64X4_Z", "vinsertf64x4 zmm0, zmm1, ymm2, 1"),
               ("VINSERTI32X8_ZK", "vinserti32x8 zmm0{k1}, zmm1, ymm2, 0"),
               ("VEXTRACTF32X4_Z", "vextractf32x4 xmm0, zmm1, 3"),
               ("VEXTRACTF32X4_ZK", "vextractf32x4 xmm0{k1}{z}, zmm1, 2"),
               ("VEXTRACTI64X2_ZMEM", "vextracti64x2 xmmword ptr [rdi]{k1}, zmm1, 1"),
               ("VEXTRACTI32X4_Y", "vextracti32x4 xmm0, ymm1, 1"),
               ("VEXTRACTF64X4_Z", "vextractf64x4 ymm0, zmm1, 1"),
               ("VEXTRACTI64X4_ZMEM", "vextracti64x4 ymmword ptr [rdi]{k1}, zmm1, 0"),
               ("VEXTRACTI32X8_ZK", "vextracti32x8 ymm0{k1}, zmm1, 1"),
               ("VPEXTRB_EAX", "{evex} vpextrb eax, xmm1, 5"), ("VPEXTRB_MEM", "{evex} vpextrb byte ptr [rdi], xmm1, 9"),
               ("VPEXTRW_EAX", "{evex} vpextrw eax, xmm1, 3"), ("VPEXTRW_MEM", "{evex} vpextrw word ptr [rdi], xmm1, 6"),
               ("VPEXTRD_EAX", "{evex} vpextrd eax, xmm1, 2"), ("VPEXTRQ_RAX", "{evex} vpextrq rax, xmm1, 1"),
               ("VPEXTRQ_MEM", "{evex} vpextrq qword ptr [rdi], xmm1, 0"),
               ("VEXTRACTPS_EAX", "{evex} vextractps eax, xmm1, 2"),
               ("VPINSRB_HIGH", "{evex} vpinsrb xmm16, xmm17, eax, 7"), ("VPINSRW_HIGH", "{evex} vpinsrw xmm16, xmm17, eax, 4"),
               ("VPINSRD_HIGH", "{evex} vpinsrd xmm16, xmm17, eax, 3"), ("VPINSRQ_HIGH", "{evex} vpinsrq xmm16, xmm17, rax, 1"),
               ("VPINSRD_MEM", "{evex} vpinsrd xmm0, xmm1, dword ptr [rsi], 2"),
               ("VINSERTPS_REG", "{evex} vinsertps xmm0, xmm1, xmm2, 0x9d"),
               ("VINSERTPS_MEM", "{evex} vinsertps xmm0, xmm1, dword ptr [rsi], 0x20"),
               ("VMOVLPS_LOAD", "{evex} vmovlps xmm0, xmm1, qword ptr [rsi]"),
               ("VMOVHPD_LOAD", "{evex} vmovhpd xmm0, xmm1, qword ptr [rsi]"),
               ("VMOVLPD_STORE", "{evex} vmovlpd qword ptr [rdi], xmm1"),
               ("VMOVHPS_STORE", "{evex} vmovhps qword ptr [rdi], xmm1"),
               ("VMOVHLPS", "{evex} vmovhlps xmm0, xmm1, xmm2"), ("VMOVLHPS", "{evex} vmovlhps xmm0, xmm1, xmm2"),
               ("VPGATHERDD_Z", "vpgatherdd zmm0{k1}, [rsi + zmm2*4]"),
               ("VPGATHERDD_Y", "vpgatherdd ymm0{k1}, [rsi + ymm2*4]"),
               ("VPGATHERDD_X", "vpgatherdd xmm0{k1}, [rsi + xmm2*4]"),
               ("VPGATHERDD_DISP8", "vpgatherdd zmm0{k1}, [rsi + zmm2*4 + 8]"),
               ("VPGATHERDD_HIGH", "vpgatherdd zmm16{k1}, [rsi + zmm20*2]"),
               ("VPGATHERDQ_Z", "vpgatherdq zmm0{k1}, [rsi + ymm2*8]"),
               ("VPGATHERQD_Z", "vpgatherqd ymm0{k1}, [rsi + zmm2*4]"),
               ("VPGATHERQQ_Z", "vpgatherqq zmm0{k1}, [rsi + zmm2*8]"),
               ("VGATHERDPS_Z", "vgatherdps zmm0{k1}, [rsi + zmm2*4]"),
               ("VGATHERQPD_Z", "vgatherqpd zmm0{k1}, [rsi + zmm2*8]"),
               ("VGATHERQPS_Z", "vgatherqps ymm0{k1}, [rsi + zmm2*4]"),
               ("VPSCATTERDD_Z", "vpscatterdd [rdi + zmm2*4]{k1}, zmm0"),
               ("VPSCATTERDQ_Z", "vpscatterdq [rdi + ymm2*8]{k1}, zmm0"),
               ("VPSCATTERQD_Z", "vpscatterqd [rdi + zmm2*4]{k1}, ymm0"),
               ("VSCATTERQPD_Z", "vscatterqpd [rdi + zmm2*8]{k1}, zmm0"),
               ("VSCATTERDPS_Z", "vscatterdps [rdi + zmm2*4]{k1}, zmm0")]:
    single(n, asm)
form("VPSADBW", "vpsadbw", "bin", 0)
form("VAESENC", "vaesenc", "bin", 0)
form("VAESDECLAST", "vaesdeclast", "bin", 0)
form("VPCLMULQDQ", "vpclmulqdq", "bin", 0, 0x11)

for n, asm in [("VEX_VAESENC_Y", "vaesenc ymm0, ymm1, ymm2"), ("VEX_VAESDECLAST_YM", "vaesdeclast ymm0, ymm1, [rsi]"),
               ("VEX_VPCLMULQDQ_Y", "vpclmulqdq ymm0, ymm1, ymm2, 0x11"), ("VEX_GF2P8MULB_Y", "vgf2p8mulb ymm0, ymm1, ymm2"),
               ("VEX_GF2P8AFFINEQB_Y", "vgf2p8affineqb ymm0, ymm1, ymm2, 0x63"),
               ("VEX_GF2P8AFFINEINVQB_X", "vgf2p8affineinvqb xmm0, xmm1, xmm2, 0x63"),
               ("SSE_GF2P8MULB", "gf2p8mulb xmm0, xmm1"), ("SSE_GF2P8AFFINEQB", "gf2p8affineqb xmm0, xmm1, 0x63"),
               ("SSE_GF2P8AFFINEINVQB", "gf2p8affineinvqb xmm0, xmm1, 0x63"),
               ("VEX_AESIMC_X", "vaesimc xmm0, xmm1"), ("VEX_VAESENC_X", "vaesenc xmm0, xmm1, xmm2")]:
    single(n, asm)

single("VPBROADCASTMB2Q_Z", "vpbroadcastmb2q zmm0, k1")
single("VPBROADCASTMW2D_Y", "vpbroadcastmw2d ymm0, k1")
single("VPBROADCASTMW2D_HIGH", "vpbroadcastmw2d zmm20, k1")
