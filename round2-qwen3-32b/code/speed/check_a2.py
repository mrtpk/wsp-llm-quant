import re, torch, triton, triton.language as tl
f8 = torch.float8_e4m3fnuz
@triton.jit
def k(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
    x = tl.load(a + rm[:, None] * K + rk[None, :]); y = tl.load(b + rk[:, None] * N + rn[None, :])
    tl.store(c + rm[:, None] * N + rn[None, :], tl.dot(x, y, out_dtype=tl.float32))
for dt, name in [(f8, "fp8"), (torch.int8, "int8")]:
    M = N = K = 64
    if dt == torch.int8:
        A = torch.randint(-8, 8, (M, K), device="cuda", dtype=dt); B = torch.randint(-8, 8, (K, N), device="cuda", dtype=dt); C = torch.empty(M, N, device="cuda", dtype=torch.int32)
        h = None
        @triton.jit
        def ki(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
            rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
            x = tl.load(a + rm[:, None] * K + rk[None, :]); y = tl.load(b + rk[:, None] * N + rn[None, :])
            tl.store(c + rm[:, None] * N + rn[None, :], tl.dot(x, y, out_dtype=tl.int32))
        h = ki[(1,)](A, B, C, M, N, K); ok = torch.equal(C.double(), A.double() @ B.double())
    else:
        A = torch.randn(M, K, device="cuda").to(dt); B = torch.randn(K, N, device="cuda").to(dt); C = torch.empty(M, N, device="cuda")
        h = k[(1,)](A, B, C, M, N, K); ok = (C - A.float() @ B.float()).abs().max().item() == 0
    asm = h.asm.get("amdgcn", "")
    print(f"triton {name} tl.dot correct={ok} mfma:", sorted(set(re.findall(r"v_mfma_\w+", asm))))
for m in (32, 64, 256):
    try:
        a8 = torch.randint(-8, 8, (m, 4096), device="cuda", dtype=torch.int8); b8 = torch.randint(-8, 8, (4096, 4096), device="cuda", dtype=torch.int8)
        r = torch._int_mm(a8, b8); print("torch._int_mm m", m, "exact", torch.equal(r.double(), a8.double() @ b8.double()))
    except Exception as e:
        print("torch._int_mm m", m, "FAIL", str(e)[:120])
try:
    a8 = torch.randint(-8, 8, (64, 4096), device="cuda", dtype=torch.int8); b8 = torch.randint(-8, 8, (4096, 4096), device="cuda", dtype=torch.int8)
    s = torch.ones(64, 1, device="cuda"); t = torch.ones(1, 4096, device="cuda")
    r = torch._scaled_mm(a8, b8.t().contiguous().t(), scale_a=s, scale_b=t, out_dtype=torch.bfloat16); print("torch._scaled_mm int8 OK")
except Exception as e:
    print("torch._scaled_mm int8 FAIL", str(e)[:150])
