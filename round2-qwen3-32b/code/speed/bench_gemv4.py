"""v4 GEMV: correctness vs fp32 reference (same weights) + graph-timed config search, M = 1 only."""
import json, torch, triton
from w4kernels import _w4_gemv4_kernel, quantize_w4, to_t, w4_gemv4, ref_gemm, G_SIZE, F8
from bench_w4 import graph_time
BW = 4.66e12; best = {}
for N, K in [(10240, 5120), (5120, 8192), (51200, 5120), (5120, 25600)]:
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    qp, s, z = quantize_w4(W); qpt = to_t(qp); G = K // G_SIZE
    nbytes = qpt.numel() * 4 + s.numel() * 2 + z.numel()
    M = 1
    for a8 in (True, False):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        r = ref_gemm(x, qp, s, z, a8).float()
        xq = x.to(F8) if a8 else x; xs = torch.ones(M, device="cuda"); out = torch.zeros(M, N, device="cuda")
        res = []
        for bn in (16, 32, 64, 128):
            for gpi in (1, 2, 4, 8):
                if bn * gpi > 512 or G % gpi: continue
                nit = G // gpi
                for sp in (1, 2, 4, 8, 16, 32):
                    if sp > nit: continue
                    for nw in (1, 2, 4):
                        ips = triton.cdiv(nit, sp); spp = triton.cdiv(nit, ips)
                        f = lambda: _w4_gemv4_kernel[(N // bn, spp, M)](xq, xs, qpt, s, z, out, N, K, G, ips, ACT_FP8=a8, BLOCK_N=bn, GPI=gpi, GS=G_SIZE, num_warps=nw)
                        try: res.append((graph_time(f, n=20), bn, gpi, sp, nw))
                        except Exception: pass
        res.sort(); t, bn, gpi, sp, nw = res[0]
        e = ((w4_gemv4(x, qpt, s, z, a8, cfg=(bn, gpi, sp, nw)).float() - r).norm() / r.norm()).item()
        best[f"{N},{K},{int(a8)}"] = [bn, gpi, sp, nw, t * 1e3, nbytes / (t * 1e-3) / BW, e]
        print(f"N={N:6d} K={K:6d} M=1 {'W4A8 ' if a8 else 'W4A16'}: err(best cfg) {e:.1e} {'OK' if e < 1e-2 else 'WRONG'} | bn={bn} gpi={gpi} split={sp} nw={nw} {t*1e3:6.1f} us {nbytes/(t*1e-3)/1e12:.2f} TB/s ({100*nbytes/(t*1e-3)/BW:.0f}%)", flush=True)
        json.dump(best, open("/results/speed_gemv4_configs.json", "w"), indent=1)
