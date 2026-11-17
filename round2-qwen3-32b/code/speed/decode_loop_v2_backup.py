"""Decode-speed step (c), option 3 (2026-11-15): own minimal Qwen3-32B decode loop with NATIVE kernels only.

  python speed/decode_loop.py FORMAT     FORMAT in {bf16, fp8, int8}

- bf16: F.linear (hipBLAS/hipBLASLt).
- fp8 : W per-output-channel e4m3fnuz, A dynamic per-token e4m3fnuz, torch._scaled_mm with row-wise scales.
- int8: W per-output-channel int8, A dynamic per-token int8, torch._int_mm + scale epilogue (not fused).
qkv and gate/up are fused (as in SGLang). lm_head stays bf16 for every format (engine default; SGLang does the same).
Attention: SDPA over a static KV cache (bf16). One decode step = 64 layers + final norm + lm_head, captured in a HIP graph.

Self-checks (printed, and stored in the result):
 1. bf16 loop vs HF transformers on a real 48-token text: per-position logits, token by token.
 2. quantized formats: next-token top-1 agreement with the bf16 loop and mean KL on the same text.
Timing: KV filled to context CTX, batch 1 and 16, median of 100 graph replays after 20 warm-up replays.
Result: /results/speed_loop_<FORMAT>.json
"""
import json, math, sys, time, glob
import torch, torch.nn.functional as F
from safetensors import safe_open

FMT = sys.argv[1]
COMPILE = "nocompile" not in sys.argv
CTX = 2048
H, I, NQ, NKV, D, NL, V, EPS, THETA = 5120, 25600, 64, 8, 128, 64, 151936, 1e-6, 1e6
SNAP = glob.glob("/models/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/")[0]
dev = "cuda"
F8 = torch.float8_e4m3fnuz; F8MAX = torch.finfo(F8).max


def load_state():
    sd = {}
    for f in sorted(glob.glob(SNAP + "*.safetensors")):
        with safe_open(f, "pt", device=dev) as s:
            for k in s.keys():
                sd[k] = s.get_tensor(k)
    return sd


class Lin:
    """One (possibly fused) linear in one format."""
    def __init__(self, W, fmt):
        self.fmt, self.N = fmt, W.shape[0]
        if fmt == "bf16":
            self.W = W.contiguous()
        elif fmt == "fp8":
            s = W.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / F8MAX
            self.Wq = (W.float() / s).to(F8).t()            # [K, N] column-major view for _scaled_mm
            self.sw = s.t().contiguous()                       # [1, N]
        elif fmt == "int8":
            s = W.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
            self.Wq = (W.float() / s).round().clamp(-127, 127).to(torch.int8).t().contiguous()  # [K, N]
            self.sw = s.t().contiguous()

    def __call__(self, x):                                     # x [M, K] bf16
        if self.fmt == "bf16":
            return F.linear(x, self.W)
        if self.fmt == "fp8":
            sx = x.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / F8MAX
            xq = (x.float() / sx).to(F8)
            return torch._scaled_mm(xq, self.Wq, scale_a=sx, scale_b=self.sw, out_dtype=torch.bfloat16)
        sx = x.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
        xq = (x.float() / sx).round().clamp(-127, 127).to(torch.int8)
        M = xq.shape[0]
        if M <= 16:                                            # _int_mm needs M > 16 on this stack (checked): pad
            xq = F.pad(xq, (0, 0, 0, 17 - M))
        y = torch._int_mm(xq, self.Wq)[:M]
        return (y.float() * sx * self.sw).to(torch.bfloat16)


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


class Model:
    def __init__(self, sd, fmt):
        g = lambda k: sd.pop(k)
        self.emb = g("model.embed_tokens.weight")
        self.norm = g("model.norm.weight")
        self.lm = Lin(g("lm_head.weight"), "bf16")
        self.L = []
        for i in range(NL):
            p = f"model.layers.{i}."
            qkv = torch.cat([g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight")])
            gu = torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")])
            self.L.append(dict(ln1=g(p + "input_layernorm.weight"), ln2=g(p + "post_attention_layernorm.weight"),
                               qn=g(p + "self_attn.q_norm.weight"), kn=g(p + "self_attn.k_norm.weight"),
                               qkv=Lin(qkv, fmt), o=Lin(g(p + "self_attn.o_proj.weight"), fmt),
                               gu=Lin(gu, fmt), down=Lin(g(p + "mlp.down_proj.weight"), fmt)))
            del qkv, gu
            torch.cuda.empty_cache()
        inv = 1.0 / (THETA ** (torch.arange(0, D, 2, device=dev).float() / D))
        self.inv = inv

    def alloc(self, B, maxlen):
        self.K = [torch.zeros(B, NKV, maxlen, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
        self.Vc = [torch.zeros(B, NKV, maxlen, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]

    def step(self, tok, pos):
        """tok [B] int64, pos: python int (current position; KV 0..pos-1 valid). Returns logits [B, V]."""
        B = tok.shape[0]
        x = self.emb[tok]                                       # [B, H]
        f = pos * self.inv
        cos, sin = torch.cat([f, f]).cos().to(torch.bfloat16), torch.cat([f, f]).sin().to(torch.bfloat16)
        def rope(t):                                            # t [B, h, D]
            t1, t2 = t[..., :D // 2], t[..., D // 2:]
            return t * cos + torch.cat([-t2, t1], -1) * sin
        for i, l in enumerate(self.L):
            h = rms(x, l["ln1"])
            qkv = l["qkv"](h)
            q = qkv[:, :NQ * D].view(B, NQ, D); k = qkv[:, NQ * D:(NQ + NKV) * D].view(B, NKV, D); v = qkv[:, (NQ + NKV) * D:].view(B, NKV, D)
            q = rope(rms(q, l["qn"])); k = rope(rms(k, l["kn"]))
            self.K[i][:, :, pos] = k; self.Vc[i][:, :, pos] = v
            # decode attention as two batched matmuls over the GQA groups (fix 2026-11-15: SDPA used a prefill
            # flash kernel, ~110 us/layer at q_len 1)
            qg = q.view(B, NKV, NQ // NKV, D)
            sc = torch.matmul(qg, self.K[i][:, :, :pos + 1].transpose(-1, -2)).float() * (D ** -0.5)
            a = torch.matmul(sc.softmax(-1).to(torch.bfloat16), self.Vc[i][:, :, :pos + 1])
            x = x + l["o"](a.reshape(B, NQ * D))
            h = rms(x, l["ln2"])
            gu = l["gu"](h)
            x = x + l["down"](F.silu(gu[:, :I]) * gu[:, I:])
        return self.lm(rms(x, self.norm)).float()


def text_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    t = ("The history of the city goes back to a small fishing village on the river. In the twelfth century, "
         "merchants built a market and a stone bridge, and the town grew quickly. Today it is known for its "
         "university, its old harbour and a large festival every summer.")
    return tok(t, return_tensors="pt").input_ids[0, :48].to(dev)


def run_text(m, ids):
    m.alloc(1, len(ids) + 1)
    out = []
    for p in range(len(ids)):
        out.append(m.step(ids[p:p + 1], p)[0])
    return torch.stack(out)                                     # [T, V]


def main():
    res = {"format": FMT, "ctx": CTX, "lm_head": "bf16", "compile": COMPILE, "attn": "matmul", "checks": {}}
    ids = text_ids()
    sd = load_state()
    if FMT == "bf16":
        # self-check 1: HF reference on the same text (load HF after our model to share nothing)
        m = Model(sd, "bf16")
        with torch.no_grad():
            ours = run_text(m, ids)
        torch.save(ours.cpu(), "/results/speed_loop_bf16_textlogits.pt")
        del m; torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa")
        with torch.no_grad():
            ref = hf(ids[None]).logits[0].float()
        del hf; torch.cuda.empty_cache()
        ours = ours.to(dev)
        top1 = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
        kl = F.kl_div(ours.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean").item()
        rel = ((ours - ref).norm() / ref.norm()).item()
        res["checks"]["vs_hf"] = dict(top1_agree=top1, kl=kl, rel_err=rel)
        print(f"CHECK bf16 loop vs HF: top-1 agreement {top1:.3f}, KL {kl:.2e}, rel err {rel:.3e}", flush=True)
        sd = load_state()
        m = Model(sd, "bf16")
    else:
        m = Model(sd, FMT)
        with torch.no_grad():
            ours = run_text(m, ids)
        ref = torch.load("/results/speed_loop_bf16_textlogits.pt").to(dev)
        top1 = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
        kl = F.kl_div(ours.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean").item()
        res["checks"]["vs_bf16_loop"] = dict(top1_agree=top1, kl=kl)
        print(f"CHECK {FMT} loop vs bf16 loop: top-1 agreement {top1:.3f}, KL {kl:.3e}", flush=True)
    del sd; torch.cuda.empty_cache()
    res["weights_gb_alloc"] = torch.cuda.memory_allocated() / 1e9

    for B in (1, 16):
        m.alloc(B, CTX + 1)
        for i in range(NL):
            m.K[i].normal_(); m.Vc[i].normal_()
        tok = torch.randint(0, V, (B,), device=dev)
        step = torch.compile(m.step, dynamic=False) if COMPILE else m.step   # fuses the element-wise ops (fix 2026-11-15)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(st):
            for _ in range(3):
                step(tok, CTX)
        torch.cuda.current_stream().wait_stream(st)
        if COMPILE:                                            # self-check 3: compiled step == eager step
            with torch.no_grad():
                ye, yc = m.step(tok, CTX), step(tok, CTX)
            ce = ((yc - ye).norm() / ye.norm()).item(); ag = (yc.argmax(-1) == ye.argmax(-1)).float().mean().item()
            res["checks"][f"compile_vs_eager_b{B}"] = dict(rel_err=ce, top1_agree=ag)
            print(f"CHECK compiled vs eager b{B}: rel err {ce:.2e}, top-1 agree {ag:.3f}", flush=True)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = step(tok, CTX)
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(100):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1) / 1e3)
        ts.sort(); t = ts[50]
        res[f"b{B}"] = dict(step_s=t, tok_s=B / t, p10_s=ts[10], p90_s=ts[90])
        print(f"{FMT} batch {B} ctx {CTX}: step {t*1e3:.2f} ms -> {B/t:.1f} tok/s (p10 {ts[10]*1e3:.2f}, p90 {ts[90]*1e3:.2f} ms)", flush=True)
        del g, out; m.K = m.Vc = None; torch.cuda.empty_cache()
    json.dump(res, open(f"/results/speed_loop_{FMT}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
