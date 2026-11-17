"""Step (c) short check, 2026-11-15: does SGLang's GPTQ W4 kernel (sgl_kernel.gptq_gemm, exllama-style) run correctly on gfx942?
GPTQ v1 packing: qweight int32 [K/8, N] (8 x 4-bit along K), qzeros int32 [K/g, N/8] (stored zero - 1), scales fp16 [K/g, N]."""
import torch
from sgl_kernel import gptq_gemm, gptq_shuffle
torch.manual_seed(0)
def pack_k(q):   # q [K, N] uint4 -> [K/8, N] int32
    K, N = q.shape; q = q.view(K // 8, 8, N).to(torch.int64); out = torch.zeros(K // 8, N, dtype=torch.int64, device=q.device)
    for j in range(8): out |= q[:, j, :] << (4 * j)
    return out.to(torch.uint32).view(torch.int32) if hasattr(torch, "uint32") else (out - (out >= 2**31) * 2**32).to(torch.int32)
def pack_n(z):   # z [G, N] -> [G, N/8]
    G, N = z.shape; z = z.view(G, N // 8, 8).to(torch.int64); out = torch.zeros(G, N // 8, dtype=torch.int64, device=z.device)
    for j in range(8): out |= z[:, :, j] << (4 * j)
    return (out - (out >= 2**31) * 2**32).to(torch.int32)
ok_all = True
for (M, K, N) in [(1, 5120, 8192), (16, 5120, 25600), (64, 25600, 5120)]:
    g = 128; G = K // g
    q = torch.randint(0, 16, (K, N), device="cuda"); z = torch.randint(1, 15, (G, N), device="cuda")
    s = (torch.rand(G, N, device="cuda") * 0.01 + 0.001).half()
    W = ((q - z.repeat_interleave(g, 0)).half() * s.repeat_interleave(g, 0)).float()
    qw = pack_k(q).contiguous(); qz = pack_n(z - 1).contiguous(); gidx = (torch.arange(K, device="cuda") // g).int()
    gptq_shuffle(qw, torch.empty(0, dtype=torch.int, device="cuda"), 4)
    x = torch.randn(M, K, device="cuda").half()
    y = gptq_gemm(x, qw, qz, s, torch.empty(0, dtype=torch.int, device="cuda"), True, 4).float()
    ref = x.float() @ W
    rel = ((y - ref).norm() / ref.norm()).item(); ok = rel < 1e-2; ok_all &= ok
    print(f"gptq_gemm M={M} K={K} N={N}: rel err {rel:.2e} {'OK' if ok else 'WRONG'}")
print("GPTQ kernel on gfx942:", "PASS" if ok_all else "FAIL")
