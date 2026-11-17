"""Where does the W4 kernel lose time? MODE 0 = load packed weights only, 1 = + unpack/convert, 2 = full (dot + scale)."""
import re, torch, triton, triton.language as tl
from bench_w4 import graph_time
from w4kernels import quantize_w4, to_t, G_SIZE, F8

@triton.jit
def k(X, QPT, S, Z, OUT, M, N, K, G, GPS, MODE: tl.constexpr, A8: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GS: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M); rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); rw = tl.arange(0, GS // 8); rk0 = tl.arange(0, GS)
    mmask = rm < M; K8 = K // 8; sh = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N, GS // 8), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_N, GS), dtype=tl.float32)
    g0 = pid_k * GPS; g1 = tl.minimum(g0 + GPS, G)
    for g in range(g0, g1):
        wp = tl.load(QPT + rn[:, None] * K8 + (g * (GS // 8) + rw)[None, :])
        if MODE == 0:
            acc1 += wp
        else:
            q = tl.reshape((wp[:, :, None] >> sh[None, None, :]) & 0xF, (BLOCK_N, GS))
            zz = tl.load(Z + g * N + rn).to(tl.int32)
            w = (q - zz[:, None]).to(tl.float32)
            if A8:
                w = w.to(tl.float8e4b8)
            else:
                w = w.to(tl.bfloat16)
            if MODE == 1:
                acc2 += w.to(tl.float32)
            else:
                xt = tl.load(X + rm[None, :] * K + (g * GS + rk0)[:, None], mask=mmask[None, :], other=0.0)
                d = tl.dot(w, xt, out_dtype=tl.float32)
                sc = tl.load(S + g * N + rn).to(tl.float32)
                acc += d * sc[:, None]
    if MODE == 0:
        tl.store(OUT + pid_k * N + rn, tl.sum(acc1, 1).to(tl.float32))
    elif MODE == 1:
        tl.store(OUT + pid_k * N + rn, tl.sum(acc2, 1))
    else:
        tl.atomic_add(OUT + rm[None, :] * N + rn[:, None], acc, mask=mmask[None, :])

N, K = 5120, 25600; G = K // G_SIZE
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
qp, s, z = quantize_w4(W); qpt = to_t(qp); nbytes = qpt.numel() * 4
x = torch.randn(1, K, device="cuda").to(F8); out = torch.zeros(64, N, device="cuda")
for bn, sp, nw in [(64, 8, 2), (64, 8, 4), (128, 16, 4), (32, 8, 2)]:
    gps = triton.cdiv(G, sp)
    line = f"bn={bn} split={sp} nw={nw}:"
    for mode in (0, 1, 2):
        f = lambda: k[(N // bn, sp)](x, qpt, s, z, out, 1, N, K, G, gps, MODE=mode, A8=True, BLOCK_M=16, BLOCK_N=bn, GS=G_SIZE, num_warps=nw)
        t = graph_time(f)
        h = k[(N // bn, sp)](x, qpt, s, z, out, 1, N, K, G, gps, MODE=mode, A8=True, BLOCK_M=16, BLOCK_N=bn, GS=G_SIZE, num_warps=nw)
        asm = h.asm["amdgcn"]
        lds = len(re.findall(r"\bds_(read|write|load|store)", asm)); spill = len(re.findall(r"scratch_|buffer_store_dword.*off", asm))
        vg = re.search(r"\.vgpr_count:\s*(\d+)", asm)
        line += f" | mode{mode} {t*1e3:6.1f} us {nbytes/(t*1e-3)/1e12:.2f} TB/s lds={lds} spill={spill} vgpr={vg.group(1) if vg else '?'}"
    print(line, flush=True)
