"""Pick (BLOCK_N, split-K) per shape for the W4 kernel; report achieved weight bandwidth. Kernel-only timing (triton do_bench)."""
import json, torch, triton
from w4kernels import _w4_kernel, quantize_w4, G_SIZE, F8
BW = 4.66e12


def graph_time(f, n=50):
    """ms per launch: n launches captured in one HIP graph, median of 20 replays."""
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) / n)
    ts.sort(); return ts[10]
def main():
    best = {}
    for N, K in [(10240, 5120), (5120, 8192), (51200, 5120), (5120, 25600)]:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        qp, s, z = quantize_w4(W); G = K // G_SIZE
        nbytes = qp.numel() * 4 + s.numel() * 2 + z.numel()
        for M in (1, 16):
            for a8 in (True, False):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                xq = x.to(F8) if a8 else x; xs = torch.ones(M, device="cuda")
                out = torch.zeros(M, N, device="cuda")
                res = []
                for bn, split, nw, ns in [(bn, sp, nw, ns) for bn in (32, 64, 128) for sp in (1, 2, 4, 8, 16, 32, 64) for nw in (2, 4, 8) for ns in (1, 2)]:
                    if True:
                        if split > G: continue
                        gps = triton.cdiv(G, split); sp = triton.cdiv(G, gps)
                        f = lambda: _w4_kernel[(N // bn, sp)](xq, xs, qp, s, z, out, M, N, K, G, gps, ACT_FP8=a8, BLOCK_M=16, BLOCK_N=bn, GS=G_SIZE, num_warps=nw, num_stages=ns)
                        try:
                            t = graph_time(f)            # fix 2026-11-15: do_bench counted ~60 us Python launch cost per call
                        except Exception as e:
                            continue
                        res.append((t, bn, split, nw, ns))
                res.sort()
                if not a8 and M == 1:
                    xb = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    tb = graph_time(lambda: torch.nn.functional.linear(xb, W))
                    print(f"   ref bf16 F.linear N={N} K={K} M=1: {tb*1e3:.1f} us, {N*K*2/(tb*1e-3)/1e12:.2f} TB/s", flush=True)
                t, bn, split, nw, ns = res[0]
                key = f"{N},{K},{M},{'a8' if a8 else 'a16'}"
                best[key] = dict(bn=bn, split=split, nw=nw, ns=ns, us=t * 1e3, bw_frac=nbytes / (t * 1e-3) / BW)
                print(f"N={N:6d} K={K:6d} M={M:2d} {'W4A8 ' if a8 else 'W4A16'}: best bn={bn:3d} split={split:2d} nw={nw} ns={ns} {t*1e3:7.1f} us  {nbytes/(t*1e-3)/1e12:.2f} TB/s ({100*nbytes/(t*1e-3)/BW:.0f}% of 4.66)", flush=True)
    json.dump(best, open("/results/speed_w4_configs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
