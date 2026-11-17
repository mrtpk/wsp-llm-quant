"""Checkpoint (a), 2026-11-14: Triton FP8 tl.dot on gfx942, torch FP8/INT8 GEMMs, vLLM presence. Correctness only, no timing."""
import importlib, torch, triton, triton.language as tl
print("torch", torch.__version__, "triton", triton.__version__, "hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
for m in ["vllm", "aiter"]:
    try:
        mod = importlib.import_module(m); print(m, "OK", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__, str(e)[:120])
f8 = torch.float8_e4m3fnuz
print("fp8 dtype", f8)

@triton.jit
def k(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
    x = tl.load(a + rm[:, None] * K + rk[None, :]); y = tl.load(b + rk[:, None] * N + rn[None, :])
    tl.store(c + rm[:, None] * N + rn[None, :], tl.dot(x, y, out_dtype=tl.float32))

M = N = K = 64
A = torch.randn(M, K, device="cuda").to(f8); B = torch.randn(K, N, device="cuda").to(f8)
C = torch.empty(M, N, device="cuda")
try:
    k[(1,)](A, B, C, M, N, K)
    ref = A.float() @ B.float()
    print("triton fp8 tl.dot: OK, max abs err", (C - ref).abs().max().item())
    asm = next(iter(k.cache[0].values())).asm.get("amdgcn", "")
    import re
    print("fp8 mfma in ISA:", sorted(set(re.findall(r"v_mfma\w*(?:fp8|f8)\w*", asm)))[:5])
except Exception as e:
    print("triton fp8 tl.dot: FAIL", type(e).__name__, str(e)[:300])
try:
    sa = torch.tensor(1.0, device="cuda"); r = torch._scaled_mm(A, B.t().contiguous().t(), scale_a=sa, scale_b=sa, out_dtype=torch.bfloat16)
    print("torch._scaled_mm fp8: OK, max abs err", (r.float() - A.float() @ B.float()).abs().max().item())
except Exception as e:
    print("torch._scaled_mm fp8: FAIL", type(e).__name__, str(e)[:200])
try:
    a8 = torch.randint(-8, 8, (64, 64), device="cuda", dtype=torch.int8); b8 = torch.randint(-8, 8, (64, 64), device="cuda", dtype=torch.int8)
    r = torch._int_mm(a8, b8); print("torch._int_mm int8: OK, exact", torch.equal(r, (a8.int() @ b8.int())))
except Exception as e:
    print("torch._int_mm int8: FAIL", type(e).__name__, str(e)[:200])
