"""v2 kernel: correctness vs fp32 reference (same weights) + graph-timed config search. Writes /results/speed_w4t_configs.json"""
import json, torch, triton
from w4kernels import _w4t_kernel, quantize_w4, to_t, w4t_gemm, ref_gemm, G_SIZE, F8
from bench_w4 import graph_time
BW = 4.66e12
best = {}
for N, K in [(10240, 5120), (5120, 8192), (51200, 5120), (5120, 25600)]:
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    qp, s, z = quantize_w4(W); qpt = to_t(qp); G = K // G_SIZE
    nbytes = qpt.numel() * 4 + s.numel() * 2 + z.numel()
    for M in (1, 16):
        for a8 in (True, False):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            e = ((w4t_gemm(x, qpt, s, z, a8).float() - ref_gemm(x, qp, s, z, a8).float()).norm() / ref_gemm(x, qp, s, z, a8).float().norm()).item()
            xq = x.to(F8) if a8 else x; xs = torch.ones(M, device="cuda"); out = torch.zeros(M, N, device="cuda")
            res = []
            for bn in (32, 64, 128):
                for sp in (1, 2, 4, 8, 16, 32):
                    for nw in (2, 4, 8):
                        for ns in (1, 2):
                            if sp > G: continue
                            gps = triton.cdiv(G, sp); spp = triton.cdiv(G, gps)
                            f = lambda: _w4t_kernel[(N // bn, spp)](xq, xs, qpt, s, z, out, M, N, K, G, gps, ACT_FP8=a8, BLOCK_M=16, BLOCK_N=bn, GS=G_SIZE, num_warps=nw, num_stages=ns)
                            try: res.append((graph_time(f), bn, sp, nw, ns))
                            except Exception: pass
            res.sort(); t, bn, sp, nw, ns = res[0]
            best[f"{N},{K},{int(a8)},{int(M <= 1)}"] = [bn, sp, nw, ns, t * 1e3, nbytes / (t * 1e-3) / BW, e]
            print(f"N={N:6d} K={K:6d} M={M:2d} {'W4A8 ' if a8 else 'W4A16'}: err {e:.1e} {'OK' if e < 1e-2 else 'WRONG'} | bn={bn} split={sp} nw={nw} ns={ns} {t*1e3:6.1f} us {nbytes/(t*1e-3)/1e12:.2f} TB/s ({100*nbytes/(t*1e-3)/BW:.0f}%)", flush=True)
json.dump(best, open("/results/speed_w4t_configs.json", "w"), indent=1)
