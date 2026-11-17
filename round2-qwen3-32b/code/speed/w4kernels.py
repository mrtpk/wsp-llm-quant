"""W4 GEMM kernels for decode (2026-11-15, after checkpoint d). Triton, gfx942.

Storage (= GPTQ W4 g128 asymmetric, the round-2 W4 finals and the GPTQ W4A16 baseline):
  qp [K/8, N] int32  : 8 x 4-bit codes along K (GPTQ order: code of row k at bits 4*(k%8))
  s  [G, N]   bf16   : group scale, G = K/128
  z  [G, N]   int8   : group zero point (0..15)
  W[k, n] = (q[k, n] - z[k//128, n]) * s[k//128, n]
Bytes: 4 bits + (16 + 8)/128 = 4.19 bits/weight (P4 accounting counts 4.25: 16-bit zero).

Kernel: one program = BLOCK_N output columns x a range of K groups (split-K, fp32 atomic add).
Per group of 128: d = dot(x_blk, (q - z)) with (q - z) in [-15, 15], exact in FP8 e4m3fnuz and in bf16;
acc += d * s[g]. So the group scale is applied in fp32 after the MFMA (no rounding of W itself).
  ACT_FP8 = True : x is FP8 (per-token scale xs applied at the end) -> FP8 MFMA      (W4A8)
  ACT_FP8 = False: x is bf16                                          -> bf16 MFMA   (W4A16)
M (tokens) is padded to BLOCK_M = 16 by masking (MFMA minimum).
"""
import torch, triton, triton.language as tl

G_SIZE = 128
F8 = torch.float8_e4m3fnuz
F8MAX = torch.finfo(F8).max


@triton.jit
def _w4_kernel(X, XS, QP, S, Z, OUT, M, N, K, G, GPS,
               ACT_FP8: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk0 = tl.arange(0, GS)
    mmask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    g0 = pid_k * GPS
    g1 = tl.minimum(g0 + GPS, G)
    for g in range(g0, g1):
        rk = g * GS + rk0
        x = tl.load(X + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0)
        # load each packed int32 ONCE ([GS/8, BN]) and unpack with shifts + reshape (fix 2026-11-15:
        # the first version gathered every word 8 times, 7-17% of bandwidth)
        rp = g * (GS // 8) + tl.arange(0, GS // 8)
        wp = tl.load(QP + rp[:, None] * N + rn[None, :])
        sh = (tl.arange(0, 8) * 4)
        q = tl.reshape((wp[:, None, :] >> sh[None, :, None]) & 0xF, (GS, BLOCK_N))
        zz = tl.load(Z + g * N + rn).to(tl.int32)
        w = (q - zz[None, :]).to(tl.float32)
        if ACT_FP8:
            w = w.to(tl.float8e4b8)
        else:
            w = w.to(tl.bfloat16)
        d = tl.dot(x, w, out_dtype=tl.float32)
        sc = tl.load(S + g * N + rn).to(tl.float32)
        acc += d * sc[None, :]
    if ACT_FP8:
        xs = tl.load(XS + rm, mask=mmask, other=0.0)
        acc = acc * xs[:, None]
    tl.atomic_add(OUT + rm[:, None] * N + rn[None, :], acc, mask=mmask[:, None])


def quantize_w4(W):
    """RTN asymmetric 4-bit, group 128 along K. W [N, K] -> (qp, s, z) in the storage above, plus q for checks."""
    N, K = W.shape
    G = K // G_SIZE
    Wg = W.float().view(N, G, G_SIZE)
    lo, hi = Wg.amin(-1), Wg.amax(-1)
    s = ((hi - lo) / 15).clamp(min=1e-8)
    z = (-lo / s).round().clamp(0, 15)
    q = (Wg / s[..., None] + z[..., None]).round().clamp(0, 15).to(torch.int32).view(N, K)
    s = s.to(torch.bfloat16)                                    # stored precision
    qt = q.t().contiguous().view(K // 8, 8, N)                  # [K/8, 8, N]
    qp = torch.zeros(K // 8, N, dtype=torch.int64, device=W.device)
    for j in range(8):
        qp |= qt[:, j, :].to(torch.int64) << (4 * j)
    qp = torch.where(qp >= 2**31, qp - 2**32, qp).to(torch.int32).contiguous()
    return qp, s.t().contiguous(), z.to(torch.int8).t().contiguous()


def dequant_w4(qp, s, z):
    """Reference: exact W [N, K] (fp32) from the storage."""
    K8, N = qp.shape
    K = K8 * 8
    q = torch.stack([(qp >> (4 * j)) & 0xF for j in range(8)], 1).view(K, N)
    g = torch.arange(K, device=qp.device) // G_SIZE
    return ((q.float() - z.float()[g]) * s.float()[g]).t()


# split-K / tile config per (N, K) and token-count class; chosen by bench_configs() (see NOTES)
CONFIGS = {}


def default_cfg(N, K, M):
    G = K // G_SIZE
    bn = 64
    target = 4 * 304
    split = max(1, min(G, target // max(1, N // bn)))
    return bn, split


def w4_gemm(x, qp, s, z, act_fp8, cfg=None):
    """x [M, K] bf16 -> y [M, N] bf16. W4A8 if act_fp8 (dynamic per-token FP8 on x), else W4A16."""
    M, K = x.shape
    N = qp.shape[1]
    G = K // G_SIZE
    c = cfg or CONFIGS.get((N, K, act_fp8, M <= 1)) or default_cfg(N, K, M)
    bn, split = c[0], c[1]
    nw, ns = (c[2], c[3]) if len(c) > 2 else (4, 1)
    gps = triton.cdiv(G, split)
    split = triton.cdiv(G, gps)
    if act_fp8:
        xs = x.float().abs().amax(1).clamp(min=1e-8) / F8MAX
        xq = (x.float() / xs[:, None]).to(F8)
    else:
        xs, xq = x, x.contiguous()
    out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    for m0 in range(0, M, 16):
        mm = min(16, M - m0)
        _w4_kernel[(N // bn, split)](xq[m0:], xs[m0:] if act_fp8 else xs, qp, s, z, out[m0:], mm, N, K, G, gps,
                                     ACT_FP8=act_fp8, BLOCK_M=16, BLOCK_N=bn, GS=G_SIZE, num_warps=nw, num_stages=ns)
    return out.to(torch.bfloat16)


def ref_gemm(x, qp, s, z, act_fp8):
    """Reference in fp32 with the SAME quantized weights and the same activation quantization."""
    W = dequant_w4(qp, s, z)
    if act_fp8:
        xs = x.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / F8MAX
        xf = (x.float() / xs).to(F8).float() * xs
    else:
        xf = x.float()
    return (xf @ W.t()).to(torch.bfloat16)


if __name__ == "__main__":
    # kernel-level check (model-level real-text gate is in decode_loop.py)
    torch.manual_seed(0)
    for N, K in [(10240, 5120), (5120, 8192), (51200, 5120), (5120, 25600)]:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        qp, s, z = quantize_w4(W)
        Wd = dequant_w4(qp, s, z)
        werr = ((Wd - W.float()).norm() / W.float().norm()).item()
        for M in (1, 16):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            for a8 in (True, False):
                y = w4_gemm(x, qp, s, z, a8).float(); r = ref_gemm(x, qp, s, z, a8).float()
                e = ((y - r).norm() / r.norm()).item()
                print(f"N={N} K={K} M={M} {'W4A8 ' if a8 else 'W4A16'}: kernel vs ref rel err {e:.2e} {'OK' if e < 1e-2 else 'WRONG'}  (W4 quant err {werr:.3f})", flush=True)


# ---------------------------------------------------------------- v2 (2026-11-15): transposed storage, k-contiguous words
# Same bytes, different order: qpt [N, K/8] int32, word i of row n holds k = 8i..8i+7 (bits 4*(k%8)).
# y^T = W^T x^T: the unpacked weight tile [BLOCK_N, GS] is the MFMA A operand with consecutive k per lane,
# so unpacking needs no layout change through LDS (v1 reshaped a [GS/8, 8, BN] tile -> 11-22% of bandwidth).
@triton.jit
def _w4t_kernel(X, XS, QPT, S, Z, OUT, M, N, K, G, GPS,
                ACT_FP8: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = tl.arange(0, GS // 8)
    rk0 = tl.arange(0, GS)
    mmask = rm < M
    K8 = K // 8
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    sh = tl.arange(0, 8) * 4
    g0 = pid_k * GPS
    g1 = tl.minimum(g0 + GPS, G)
    for g in range(g0, g1):
        wp = tl.load(QPT + rn[:, None] * K8 + (g * (GS // 8) + rw)[None, :])          # [BN, GS/8]
        q = tl.reshape((wp[:, :, None] >> sh[None, None, :]) & 0xF, (BLOCK_N, GS))     # k-contiguous per lane
        zz = tl.load(Z + g * N + rn).to(tl.int32)
        w = (q - zz[:, None]).to(tl.float32)
        rk = g * GS + rk0
        xt = tl.load(X + rm[None, :] * K + rk[:, None], mask=mmask[None, :], other=0.0)  # [GS, BM]
        if ACT_FP8:
            w = w.to(tl.float8e4b8)
        else:
            w = w.to(tl.bfloat16)
        d = tl.dot(w, xt, out_dtype=tl.float32)                                         # [BN, BM]
        sc = tl.load(S + g * N + rn).to(tl.float32)
        acc += d * sc[:, None]
    if ACT_FP8:
        xs = tl.load(XS + rm, mask=mmask, other=0.0)
        acc = acc * xs[None, :]
    tl.atomic_add(OUT + rm[None, :] * N + rn[:, None], acc, mask=mmask[None, :])


def to_t(qp):
    """v1 storage qp [K/8, N] -> v2 storage qpt [N, K/8] (same codes, k-contiguous words)."""
    K8, N = qp.shape
    K = K8 * 8
    q = torch.stack([(qp >> (4 * j)) & 0xF for j in range(8)], 1).view(K, N).t().contiguous()   # [N, K]
    q = q.view(N, K8, 8).to(torch.int64)
    w = torch.zeros(N, K8, dtype=torch.int64, device=qp.device)
    for j in range(8):
        w |= q[:, :, j] << (4 * j)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32).contiguous()


CONFIGS_T = {}


def w4t_gemm(x, qpt, s, z, act_fp8, cfg=None):
    M, K = x.shape
    N = qpt.shape[0]
    G = K // G_SIZE
    c = cfg or CONFIGS_T.get((N, K, act_fp8, M <= 1)) or (64, max(1, min(G, (4 * 304) // max(1, N // 64))), 4, 1)
    bn, split, nw, ns = c
    gps = triton.cdiv(G, split); split = triton.cdiv(G, gps)
    if act_fp8:
        xs = x.float().abs().amax(1).clamp(min=1e-8) / F8MAX
        xq = (x.float() / xs[:, None]).to(F8)
    else:
        xs, xq = x, x.contiguous()
    out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    for m0 in range(0, M, 16):
        mm = min(16, M - m0)
        _w4t_kernel[(N // bn, split)](xq[m0:], xs[m0:] if act_fp8 else xs, qpt, s, z, out[m0:], mm, N, K, G, gps,
                                      ACT_FP8=act_fp8, BLOCK_M=16, BLOCK_N=bn, GS=G_SIZE, num_warps=nw, num_stages=ns)
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------- v3 (2026-11-15): GEMV for decode, no layout change
# Diagnosis (speed/diag_w4.py, down_proj shape, M=1): loads alone 4.0 TB/s (86%); unpack via reshape and the MFMA
# operand conversion go through LDS (68-196 ds_ ops) -> 0.5-0.7 TB/s. Here: same v2 storage (qpt [N, K/8], word i
# holds k = 8i..8i+7). Nibble plane j of a [BN, GS/8] word tile = weights at k = 8i + j; it is multiplied element-wise
# with x[8i + j] (strided load) and accumulated in registers; one reduction at the end. FMA on the vector units, no MFMA.
# Values are identical to W4A8 (x FP8 per token, (q - z) exact, group scale in fp32) or W4A16 (x bf16).
@triton.jit
def _w4_gemv_kernel(X, XS, QPT, S, Z, OUT, N, K, G, GPS, ROWS,
                    ACT_FP8: tl.constexpr, BLOCK_N: tl.constexpr, GS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    m = tl.program_id(2)                     # one token row per grid slice (decode: ROWS small)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = tl.arange(0, GS // 8)
    K8 = K // 8
    acc = tl.zeros((BLOCK_N, GS // 8), dtype=tl.float32)
    g0 = pid_k * GPS
    g1 = tl.minimum(g0 + GPS, G)
    for g in range(g0, g1):
        wp = tl.load(QPT + rn[:, None] * K8 + (g * (GS // 8) + rw)[None, :])         # [BN, GS/8]
        zz = tl.load(Z + g * N + rn).to(tl.float32)
        sc = tl.load(S + g * N + rn).to(tl.float32)
        part = tl.zeros((BLOCK_N, GS // 8), dtype=tl.float32)
        xsum = tl.zeros((GS // 8,), dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(X + m * K + g * GS + rw * 8 + j).to(tl.float32)                 # [GS/8]
            qj = ((wp >> (4 * j)) & 0xF).to(tl.float32)
            part += qj * xj[None, :]
            xsum += xj
        part -= zz[:, None] * xsum[None, :]                                           # sum (q - z) x
        acc += part * sc[:, None]
    y = tl.sum(acc, 1)
    if ACT_FP8:
        y = y * tl.load(XS + m)
    tl.atomic_add(OUT + m * N + rn, y)


CONFIGS_GEMV = {}


def w4_gemv(x, qpt, s, z, act_fp8, cfg=None):
    M, K = x.shape
    N = qpt.shape[0]
    G = K // G_SIZE
    bn, split, nw = cfg or CONFIGS_GEMV.get((N, K, act_fp8)) or (64, max(1, min(G, (4 * 304) // max(1, N // 64))), 4)
    gps = triton.cdiv(G, split); split = triton.cdiv(G, gps)
    if act_fp8:
        xs = x.float().abs().amax(1).clamp(min=1e-8) / F8MAX
        xq = (x.float() / xs[:, None]).to(F8)
    else:
        xs, xq = x, x.contiguous()
    out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    _w4_gemv_kernel[(N // bn, split, M)](xq, xs, qpt, s, z, out, N, K, G, gps, M,
                                         ACT_FP8=act_fp8, BLOCK_N=bn, GS=G_SIZE, num_warps=nw)
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------- v4 (2026-11-15): GEMV, wider K tile per load
# v3 reached 22-47% at M=1 with a [BN, 16]-word tile (64 B per row per load) and few waves per CU, so few bytes are
# in flight per wave. v4 loads [BN, GPI*16] words (GPI groups at once; 256 B per row for GPI=4). The scale and zero of
# each word are gathered per word (word w belongs to group w // 16; 16 lanes read the same address -> cache hits).
# Arithmetic per group is the same as v3: sum_k (q - z) x  then * s in fp32.
@triton.jit
def _w4_gemv4_kernel(X, XS, QPT, S, Z, OUT, N, K, G, IPS,
                     ACT_FP8: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, GS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    m = tl.program_id(2)
    KW: tl.constexpr = GPI * (GS // 8)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rw = tl.arange(0, KW)
    K8 = K // 8
    NIT = G // GPI
    i0 = pid_k * IPS
    i1 = tl.minimum(i0 + IPS, NIT)
    acc = tl.zeros((BLOCK_N, KW), dtype=tl.float32)
    for it in range(i0, i1):
        w0 = it * KW
        wp = tl.load(QPT + rn[:, None] * K8 + (w0 + rw)[None, :])                          # [BN, KW]
        gw = it * GPI + rw // (GS // 8)                                                    # group of each word
        zz = tl.load(Z + gw[None, :] * N + rn[:, None]).to(tl.float32)                     # [BN, KW] (gather)
        sc = tl.load(S + gw[None, :] * N + rn[:, None]).to(tl.float32)
        part = tl.zeros((BLOCK_N, KW), dtype=tl.float32)
        xsum = tl.zeros((KW,), dtype=tl.float32)
        for j in tl.static_range(8):
            xj = tl.load(X + m * K + (w0 + rw) * 8 + j).to(tl.float32)
            qj = ((wp >> (4 * j)) & 0xF).to(tl.float32)
            part += qj * xj[None, :]
            xsum += xj
        acc += (part - zz * xsum[None, :]) * sc
    y = tl.sum(acc, 1)
    if ACT_FP8:
        y = y * tl.load(XS + m)
    tl.atomic_add(OUT + m * N + rn, y)


CONFIGS_GEMV4 = {}


def w4_gemv4(x, qpt, s, z, act_fp8, cfg=None):
    M, K = x.shape
    N = qpt.shape[0]
    G = K // G_SIZE
    bn, gpi, split, nw = cfg or CONFIGS_GEMV4.get((N, K, act_fp8)) or (64, 4, 4, 4)
    nit = G // gpi
    ips = triton.cdiv(nit, split); split = triton.cdiv(nit, ips)
    if act_fp8:
        xs = x.float().abs().amax(1).clamp(min=1e-8) / F8MAX
        xq = (x.float() / xs[:, None]).to(F8)
    else:
        xs, xq = x, x.contiguous()
    out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    _w4_gemv4_kernel[(N // bn, split, M)](xq, xs, qpt, s, z, out, N, K, G, ips,
                                          ACT_FP8=act_fp8, BLOCK_N=bn, GPI=gpi, GS=G_SIZE, num_warps=nw)
    return out.to(torch.bfloat16)
