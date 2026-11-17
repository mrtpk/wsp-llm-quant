"""Checkpoint (b), 2026-11-15: memory-bandwidth upper limit for batch-1/16 decode per format.
Bytes per decode step = all linear weights (incl. scales, exceptions; lm_head as stated) + KV cache read at context L.
Embedding (1 row), norms and activations are ignored (< 0.1%). Upper limit tok/s = bandwidth / bytes (per sequence: x batch)."""
import json, torch, time
# measured achievable read bandwidth on this GPU
x = torch.empty(4 * 2**30, dtype=torch.uint8, device="cuda").view(torch.float32)  # 4 GiB... use 16 GiB below
x = torch.ones(4 * 2**30, dtype=torch.float32, device="cuda")  # 16 GiB
for _ in range(3): x.sum()
torch.cuda.synchronize(); ts = []
for _ in range(10):
    t = time.perf_counter(); x.sum(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
ts.sort(); bw_read = x.numel() * 4 / ts[len(ts) // 2]
y = torch.empty_like(x); torch.cuda.synchronize(); ts = []
for _ in range(10):
    t = time.perf_counter(); y.copy_(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
ts.sort(); bw_copy = 2 * x.numel() * 4 / ts[len(ts) // 2]
del x, y
SPEC = 6.0e12
print(f"HBM: spec 6.0 TB/s; measured read (sum) {bw_read/1e12:.2f} TB/s, copy {bw_copy/1e12:.2f} TB/s")
# Qwen3-32B linear sizes
H, I, NQ, NKV, D, L_, V = 5120, 25600, 64, 8, 128, 64, 151936
body = L_ * (H * NQ * D + 2 * H * NKV * D + NQ * D * H + 3 * H * I)
lm = V * H
tot = body + lm
print(f"params: decoder linears {body/1e9:.3f} B, lm_head {lm/1e9:.3f} B, total {tot/1e9:.3f} B")
R = "/models/tmp/wsp2/results/p4"
def get(k):
    r = json.load(open(f"{R}/{k}/s0.json")); return r["wbits"], r["wbits_body"], r["kvbits"]
FORMATS = [("bf16", None), ("fp8_W8A8", "fp8_W8A8"), ("rtn_int8_W8A8", "rtn_int8_W8A8"), ("gptq_W4g128 (W4A16)", "gptq_W4g128"),
           ("W4A8 finals (gptq/awq asym Afp8)", "gptq_W4g128_L63d8_asym15_Afp8"), ("mix W3 late34", "mix_asym_W3_late34"),
           ("mix W3g64 late44 (+Afp8)", "mix_asym_W3g64_late44"), ("ours W4A4KV4", "ours_W4A4KV4_g32"), ("quarot W4A4KV4", "quarot_W4A4KV4")]
out = []
CTX = 2048
print(f"\ncontext {CTX}; bound = measured read BW / bytes; columns: lm_head int8 (our accounting) | lm_head bf16 (engine default)")
print(f"{'format':36s} {'W':>6s} {'KV':>5s} {'GB w':>6s} {'GB kv/seq':>9s} | {'b1 tok/s':>8s} {'b16 tok/s':>9s} | {'b1 bf16-lm':>10s}")
for name, k in FORMATS:
    if k is None: wb, wbody, kvb = 16.0, 16.0, 16.0; wlm8 = (body * 16 + lm * 8) / 8
    else: wb, wbody, kvb = get(k); wlm8 = wb * tot / 8
    wlm16 = (wbody * body + 16 * lm) / 8 if k else body * 2 + lm * 2
    if k is None: wlm8 = (body * 16 + lm * 8) / 8
    kv = 2 * NKV * D * L_ * kvb / 8 * CTX
    b1 = bw_read / (wlm8 + kv); b16 = 16 * bw_read / (wlm8 + 16 * kv); b1h = bw_read / (wlm16 + kv)
    out.append(dict(format=name, key=k, wbits=wb, wbits_body=wbody, kvbits=kvb, gb_w_lm8=wlm8 / 1e9, gb_w_lm16=wlm16 / 1e9, gb_kv_seq=kv / 1e9, b1=b1, b16=b16, b1_lm16=b1h))
    print(f"{name:36s} {wb:6.3f} {kvb:5.2f} {wlm8/1e9:6.2f} {kv/1e9:9.3f} | {b1:8.1f} {b16:9.1f} | {b1h:10.1f}")
json.dump(dict(bw_read=bw_read, bw_copy=bw_copy, ctx=CTX, rows=out), open("/results/speed_roofline.json", "w"), indent=1)
