r"""Simulated (fake) quantization for Qwen3 in plain transformers. Built in-house for round 2.

Spec (dict), used for weights, activations and KV:
  bits   : int; >= 16 means identity (exercises the code path, changes nothing)
  fmt    : "int"   uniform integer, float scale per group (sym or asym)
           "mxint" OCP-MX style: block of `group` elements, shared power-of-two scale, sym int elements
           "mxfp"  OCP-MX style minifloat elements (4: E2M1, 6: E2M3, 8: E4M3), power-of-two block scale
           "fp"    minifloat elements with a float (non power-of-two) scale per group (e.g. FP8 per channel)
  group  : 0 = whole row (per output channel for W, per token for A, per head-token for KV); else group size
  sym    : bool (int only)
  clip   : float ratio applied to the range, or "mse" (weights: per-group grid search)
  keep   : activations only: {"pos": [0, ...], "topk": k, "thresh": t}  tokens kept in high precision
           (pos: fixed positions; topk: k tokens with largest absmax per sequence; thresh: absmax > t * median)
  axis   : KV only: "token" (default, quantize along head_dim) or "channel" (along tokens, KIVI-style K)

Rules (list of dicts), applied in order, later rules override earlier ones per module:
  {"match": regex on module name (e.g. r"layers\.(\d+)\.mlp\.down_proj"), "w": spec, "a": spec}
  {"kv": regex on str(layer_idx), "k": spec, "v": spec}
Linear names: model.layers.{i}.self_attn.{q,k,v,o}_proj, model.layers.{i}.mlp.{gate,up,down}_proj.
lm_head is never matched by the helpers below (kept in bf16).
"""
import math, re
import torch

TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_FP = {4: (2, 1, 6.0), 6: (2, 3, 7.5), 8: (4, 3, 448.0)}  # bits -> (ebits, mbits, max)


def lin_name(i, t):
    return f"model.layers.{i}.{'self_attn' if t in TYPES[:4] else 'mlp'}.{t}"


def lin_regex(layers=None, types=None):
    L = r"\d+" if layers is None else "(" + "|".join(str(i) for i in layers) + ")"
    T = "|".join(types or TYPES)
    return rf"^model\.layers\.{L}\.(self_attn|mlp)\.({T})$"


# ---------------------------------------------------------------- element formats

def _fp_round(x, e, m, maxv):
    """Round to a minifloat grid (e exponent bits, m mantissa bits, saturating at maxv), with subnormals."""
    bias = 2 ** (e - 1) - 1
    ax = x.abs().clamp(max=maxv)
    ex = torch.floor(torch.log2(ax.clamp(min=1e-30))).clamp(min=1 - bias)
    step = torch.exp2(ex - m)
    return torch.sign(x) * (torch.round(ax / step) * step).clamp(max=maxv)


def _int_q(x, bits, sym, c):
    # x: (..., g) float32 ; c: clip ratio, scalar or broadcastable tensor
    if sym:
        qmax = 2 ** (bits - 1) - 1
        s = (x.abs().amax(-1, keepdim=True) * c / qmax).clamp(min=1e-10)
        return torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s
    qmax = 2 ** bits - 1
    mx = x.amax(-1, keepdim=True).clamp(min=0) * c
    mn = x.amin(-1, keepdim=True).clamp(max=0) * c
    s = ((mx - mn) / qmax).clamp(min=1e-10)
    z = torch.round(-mn / s)
    return (torch.clamp(torch.round(x / s) + z, 0, qmax) - z) * s


def _quant_groups(xg, spec, c=1.0):
    """xg: (..., g) float32 -> dequantized, same shape."""
    bits, fmt = spec["bits"], spec.get("fmt", "int")
    if fmt == "int":
        return _int_q(xg, bits, spec.get("sym", False), c)
    amax = xg.abs().amax(-1, keepdim=True) * c
    if fmt == "mxint":
        qmax = 2 ** (bits - 1) - 1
        # shared scale 2^k, smallest power of two with amax / 2^k <= qmax (no clipping of the block max)
        s = torch.exp2(torch.ceil(torch.log2((amax / qmax).clamp(min=1e-30))))
        return torch.clamp(torch.round(xg / s), -qmax - 1, qmax) * s
    e, m, maxv = _FP[bits]
    if fmt == "mxfp":
        # OCP MX: X = 2^(floor(log2 amax) - emax_elem); elements saturate at maxv
        emax = math.floor(math.log2(maxv))
        s = torch.exp2(torch.floor(torch.log2(amax.clamp(min=1e-30))) - emax)
        return _fp_round(xg / s, e, m, maxv) * s
    if fmt == "fp":
        s = (amax / maxv).clamp(min=1e-12)
        return _fp_round(xg / s, e, m, maxv) * s
    raise ValueError(fmt)


def fake_quant(x, spec):
    """Quantize along the last dim in groups of spec['group'] (0 = whole row). Returns x.dtype."""
    if spec is None or spec["bits"] >= 16:
        return x
    D = x.shape[-1]
    g = spec.get("group", 0) or D
    assert D % g == 0, (D, g)
    xf = x.float().reshape(*x.shape[:-1], D // g, g)
    c = spec.get("clip", 1.0)
    c = 1.0 if c == "mse" else c
    return _quant_groups(xf, spec, c).reshape(x.shape).to(x.dtype)


@torch.no_grad()
def quant_weight(W, spec, chunk=2048, grid=20, maxshrink=0.5):
    """W: (out, in). Groups along `in`. clip='mse': per-group grid search over the range shrink ratio."""
    if spec is None or spec["bits"] >= 16:
        return W
    if spec.get("clip", 1.0) != "mse":
        out = torch.empty_like(W)
        for r in range(0, W.shape[0], chunk):
            out[r: r + chunk] = fake_quant(W[r: r + chunk], spec)
        return out
    D = W.shape[1]
    g = spec.get("group", 0) or D
    out = torch.empty_like(W)
    for r in range(0, W.shape[0], chunk):
        xg = W[r: r + chunk].float().reshape(-1, D // g, g)
        best = None; besterr = None
        for k in range(grid):
            c = 1 - maxshrink * k / grid
            q = _quant_groups(xg, spec, c)
            err = (q - xg).pow(2).sum(-1, keepdim=True)
            if best is None:
                best, besterr = q, err
            else:
                better = err < besterr
                best = torch.where(better, q, best); besterr = torch.where(better, err, besterr)
        out[r: r + chunk] = best.reshape(-1, D).to(W.dtype)
    return out


def quant_act(x, spec):
    """x: (B, T, D). Dynamic quantization per token / per group, with optional kept (high-precision) tokens."""
    if spec is None or spec["bits"] >= 16:
        return x
    xq = fake_quant(x, spec)
    keep = spec.get("keep")
    if not keep:
        return xq
    B, T, D = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for p in keep.get("pos", []):
        if p < T:
            mask[:, p] = True
    tm = x.abs().amax(-1)  # (B, T)
    if keep.get("topk"):
        idx = tm.topk(min(keep["topk"], T), dim=1).indices
        mask.scatter_(1, idx, True)
    if keep.get("thresh"):
        mask |= tm > keep["thresh"] * tm.median(dim=1, keepdim=True).values
    return torch.where(mask[..., None], x, xq)


def quant_kv(x, spec):
    """x: (B, H, T, hd), K after QK-norm and RoPE, or V."""
    if spec is None or spec["bits"] >= 16:
        return x
    if spec.get("axis", "token") == "channel":
        xt = x.transpose(-1, -2)                     # (B, H, hd, T): groups of `group` tokens per channel
        T, g = xt.shape[-1], spec.get("group", 0) or xt.shape[-1]
        pad = (-T) % g
        if pad:                                      # last partial block: pad by repeating the edge value
            xt = torch.cat([xt, xt[..., -1:].expand(*xt.shape[:-1], pad)], -1)
        return fake_quant(xt, spec)[..., :T].transpose(-1, -2)
    xq = fake_quant(x, spec)
    keep = spec.get("keep")
    if keep and keep.get("pos"):
        for p in keep["pos"]:
            xq[:, :, p] = x[:, :, p]
    return xq


def had(x, b):
    """Normalized Walsh-Hadamard on contiguous blocks of size b (power of 2) along the last dim."""
    shape, D = x.shape, x.shape[-1]
    y = x.reshape(-1, D // b, b)
    h = 1
    while h < b:
        y = y.reshape(y.shape[0], D // b, b // (2 * h), 2, h)
        y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), dim=-2)
        h *= 2
    return (y.reshape(shape) / math.sqrt(b)).to(x.dtype)


# ---------------------------------------------------------------- attention wrapper (KV quant + probes)

_ORIG_SDPA = None


def install_attention():
    """Wrap transformers' sdpa attention once. Per-module attributes:
       fq_k, fq_v: KV specs; fq_probe: callable(module, q, k, v) for statistics."""
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    _ORIG_SDPA = sdpa_attention_forward

    def fq_sdpa(module, query, key, value, attention_mask, **kw):
        probe = getattr(module, "fq_probe", None)
        if probe is not None:
            probe(module, query, key, value)
        if getattr(module, "fq_rot_qk", False):   # online Hadamard on head_dim after RoPE: q.k unchanged
            query, key = had(query, query.shape[-1]), had(key, key.shape[-1])
        ks, vs = getattr(module, "fq_k", None), getattr(module, "fq_v", None)
        if ks is not None:
            key = quant_kv(key, ks)
        if vs is not None:
            value = quant_kv(value, vs)
        return _ORIG_SDPA(module, query, key, value, attention_mask, **kw)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = fq_sdpa


# ---------------------------------------------------------------- applying configurations

class FQ:
    """Apply / undo a quantization configuration on a loaded model. Original weights are backed up on CPU."""

    def __init__(self, model):
        install_attention()
        self.model = model
        self.mods = dict(model.named_modules())
        self.backup = {}
        self.hooks = []
        self.kv_layers = []

    def set_weight(self, name, spec):
        lin = self.mods[name]
        if name not in self.backup:
            self.backup[name] = lin.weight.data.to("cpu", copy=True)
        W = self.backup[name].to(lin.weight.device)
        lin.weight.data.copy_(quant_weight(W, spec))

    def set_act(self, name, spec):
        if spec is None:
            return
        h = self.mods[name].register_forward_pre_hook(lambda m, inp, s=spec: (quant_act(inp[0], s),) + tuple(inp[1:]))
        self.hooks.append(h)

    def set_kv(self, i, kspec, vspec):
        att = self.model.model.layers[i].self_attn
        att.fq_k, att.fq_v = kspec, vspec
        self.kv_layers.append(i)

    def apply(self, rules):
        wmap, amap = {}, {}
        nl = len(self.model.model.layers)
        for r in rules:
            if "match" in r:
                rx = re.compile(r["match"])
                for n in self.mods:
                    if rx.search(n) and n.startswith("model.layers.") and n.rsplit(".", 1)[-1] in TYPES:
                        if "w" in r:
                            wmap[n] = r["w"]
                        if "a" in r:
                            amap[n] = r["a"]
            if "kv" in r:
                rx = re.compile(r["kv"])
                for i in range(nl):
                    if rx.fullmatch(str(i)):
                        self.set_kv(i, r.get("k"), r.get("v"))
        for n, s in wmap.items():
            if s is not None:
                self.set_weight(n, s)
        for n, s in amap.items():
            self.set_act(n, s)
        return self

    def reset(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        for i in self.kv_layers:
            att = self.model.model.layers[i].self_attn
            att.fq_k = att.fq_v = None
        self.kv_layers = []
        for n, W in self.backup.items():
            self.mods[n].weight.data.copy_(W.to(self.mods[n].weight.device))
        self.backup = {}


# ---------------------------------------------------------------- cost accounting

def weight_bits(spec, in_features):
    """Average stored bits per weight incl. scales (16-bit) and zero-points (bits) per group."""
    if spec is None or spec["bits"] >= 16:
        return 16.0
    g = spec.get("group", 0) or in_features
    fmt = spec.get("fmt", "int")
    if fmt in ("mxint", "mxfp"):
        return spec["bits"] + 8 / g
    over = 16 + (0 if spec.get("sym", False) or fmt == "fp" else spec["bits"])
    return spec["bits"] + over / g
