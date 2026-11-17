"""v3 GEMV: correctness vs fp32 reference (same weights) + graph-timed config search, M = 1 (and M = 16 for comparison)."""
import json, torch, triton
from w4kernels import _w4_gemv_kernel, quantize_w4, to_t, w4_gemv, ref_gemm, G_SIZE, F8
from bench_w4 import graph_time
BW = 4.66e12; best = {}
for N, K in [(10240, 5120), (5120, 8192), (51200, 5120), (5120, 25600)]:
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    qp, s, z = quantize_w4(W); qpt = to_t(qp); G = K // G_SIZE
    nbytes = qpt.numel() * 4 + s.numel() * 2 + z.numel()
    for M in (1, 16):
        for a8 in (True, False):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            r = ref_gemm(x, qp, s, z, a8).float()
            e = ((w4_gemv(x, qpt, s, z, a8).float() - r).norm() / r.norm()).item()
            xq = x.to(F8) if a8 else x; xs = torch.ones(M, device="cuda"); out = torch.zeros(M, N, device="cuda")
            res = []
            for bn in (16, 32, 64, 128):
                for sp in (1, 2, 4, 8, 16, 32):
                    for nw in (1, 2, 4, 8):
                        if sp > G: continue
                        gps = triton.cdiv(G, sp); spp = triton.cdiv(G, gps)
                        f = lambda: _w4_gemv_kernel[(N // bn, spp, M)](xq, xs, qpt, s, z, out, N, K, G, gps, M, ACT_FP8=a8, BLOCK_N=bn, GS=G_SIZE, num_warps=nw)
                        try: res.append((graph_time(f), bn, sp, nw))
                        except Exception: pass
            res.sort(); t, bn, sp, nw = res[0]
            best[f"{N},{K},{int(a8)},{M}"] = [bn, sp, nw, t * 1e3, nbytes / (t * 1e-3) / BW, e]
            print(f"N={N:6d} K={K:6d} M={M:2d} {'W4A8 ' if a8 else 'W4A16'}: err {e:.1e} {'OK' if e < 1e-2 else 'WRONG'} | bn={bn} split={sp} nw={nw} {t*1e3:6.1f} us {nbytes/(t*1e-3)/1e12:.2f} TB/s ({100*nbytes/(t*1e-3)/BW:.0f}%)", flush=True)
json.dump(best, open("/results/speed_gemv_configs.json", "w"), indent=1)
