"""P1 profiling study for Qwen3-32B (forward passes only). One sub-command per PLAN.md P1 item.

  attrib_type : error attribution per module type (W4 per-channel, W3 g128, A4 per-token) + KV + all-at-once
  attrib_layer: error attribution per layer (weights, activations, KV of one layer at a time)
  attrib_lt   : layer x type grid (A4 per-token, W3 g128), 16 c4dev windows
  sens_w      : weight bits x granularity x format per module type
  sens_a      : activation bits x granularity x format per module type
  outlier     : A4 / W4A4 / KV4 with massive-activation tokens kept in high precision
  actstats    : activation statistics at every linear input, residual stream, K/V; attention sinks
  wstats      : weight statistics, AWQ-style salience, spectrum of the quantization error (needs actstats)
  prop        : error propagation: quantize layer i only, relative hidden-state error at layers i..63

Results: /results/profile/<cmd>.json (+ .pt for tensors). Every eval run is resumable by key.
Metric for the attribution/sensitivity runs: C4 dev perplexity (first 64 c4 validation windows).
"""
import argparse, math, os, sys, time
import torch
from r2 import common, fq
from r2.fq import TYPES

P = f"{common.RESULTS}/profile"
os.makedirs(P, exist_ok=True)

W4pc = {"bits": 4, "sym": True}                      # RTN, per output channel, symmetric
W3g = {"bits": 3, "group": 128, "sym": False}         # RTN, g128, asymmetric
A4 = {"bits": 4, "sym": False}                        # dynamic per token, asymmetric
A8 = {"bits": 8, "sym": False}
KV4 = {"bits": 4, "sym": False}                       # per (token, head), asymmetric
ALL = fq.lin_regex()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Runner:
    def __init__(self, name, model, ids):
        self.path = f"{P}/{name}.json"
        self.res = common.read_json(self.path, {})
        self.model, self.ids = model, ids

    def __call__(self, key, rules, meta=None):
        if key in self.res:
            return self.res[key]
        t = time.time()
        q = fq.FQ(self.model).apply(rules)
        nll, cnt = common.nll_sum(self.model, self.ids)
        q.reset()
        r = dict(ppl=math.exp(nll / cnt), sec=round(time.time() - t, 1), **(meta or {}))
        self.res[key] = r
        common.write_json(self.path, self.res)
        base = self.res.get("bf16", {}).get("ppl")
        log(f"{key:40s} ppl {r['ppl']:.4f}" + (f"  d {r['ppl'] - base:+.4f}" if base else "") + f"  ({r['sec']}s)")
        return r


# ---------------------------------------------------------------- error attribution / sensitivity

def attrib_type(R):
    R("bf16", [])
    for nm, rules in [("all/W4pc", [{"match": ALL, "w": W4pc}]), ("all/W3g128", [{"match": ALL, "w": W3g}]),
                      ("all/A4", [{"match": ALL, "a": A4}]), ("all/A8", [{"match": ALL, "a": A8}]),
                      ("all/W4pcA4", [{"match": ALL, "w": W4pc, "a": A4}]),
                      ("all/W4pcA4KV4", [{"match": ALL, "w": W4pc, "a": A4}, {"kv": r"\d+", "k": KV4, "v": KV4}])]:
        R(nm, rules)
    for t in TYPES:
        rx = fq.lin_regex(types=[t])
        R(f"type/{t}/W4pc", [{"match": rx, "w": W4pc}])
        R(f"type/{t}/W3g128", [{"match": rx, "w": W3g}])
        R(f"type/{t}/A4", [{"match": rx, "a": A4}])
    for b in [4, 3, 2]:
        s = {"bits": b, "sym": False}
        R(f"kv/K{b}", [{"kv": r"\d+", "k": s}])
        R(f"kv/V{b}", [{"kv": r"\d+", "v": s}])
        R(f"kv/K{b}g32", [{"kv": r"\d+", "k": dict(s, group=32)}])
        R(f"kv/V{b}g32", [{"kv": r"\d+", "v": dict(s, group=32)}])
        R(f"kv/K{b}chan", [{"kv": r"\d+", "k": dict(s, axis="channel", group=128)}])


def attrib_layer(R, nl):
    R("bf16", [])
    for i in range(nl):
        rx = fq.lin_regex(layers=[i])
        R(f"layer/{i}/W4pc", [{"match": rx, "w": W4pc}])
        R(f"layer/{i}/A4", [{"match": rx, "a": A4}])
        R(f"layer/{i}/W3g128", [{"match": rx, "w": W3g}])
        R(f"layer/{i}/KV4", [{"kv": str(i), "k": KV4, "v": KV4}])


def attrib_lt(R, nl):
    R("bf16", [])
    for i in range(nl):
        for t in TYPES:
            rx = fq.lin_regex(layers=[i], types=[t])
            R(f"lt/{i}/{t}/A4", [{"match": rx, "a": A4}])
            R(f"lt/{i}/{t}/W3g128", [{"match": rx, "w": W3g}])


def sens_w(R):
    R("bf16", [])
    specs = {}
    for b in [8, 6, 4, 3, 2]:
        for g in [0, 128, 64, 32]:
            specs[f"int{b}_{'pc' if g == 0 else 'g%d' % g}"] = {"bits": b, "group": g, "sym": False}
    specs.update({
        "int4_g128_mse": {"bits": 4, "group": 128, "sym": False, "clip": "mse"},
        "int3_g128_mse": {"bits": 3, "group": 128, "sym": False, "clip": "mse"},
        "int4_pc_sym_mse": {"bits": 4, "sym": True, "clip": "mse"},
        "mxint4_b32": {"bits": 4, "group": 32, "fmt": "mxint"},
        "mxfp4_b32": {"bits": 4, "group": 32, "fmt": "mxfp"},
        "fp4_g16": {"bits": 4, "group": 16, "fmt": "fp"},
        "fp8_pc": {"bits": 8, "fmt": "fp"},
        "mxfp6_b32": {"bits": 6, "group": 32, "fmt": "mxfp"},
    })
    for t in TYPES:  # most informative first: 4 and 3 bits
        rx = fq.lin_regex(types=[t])
        for k in sorted(specs, key=lambda k: (specs[k]["bits"] not in (3, 4), k)):
            R(f"W/{t}/{k}", [{"match": rx, "w": specs[k]}], dict(wbits=fq.weight_bits(specs[k], 5120)))
    for k in ["int4_g128", "int4_g128_mse", "int3_g128", "int3_g128_mse", "mxfp4_b32", "fp4_g16", "int4_pc_sym_mse"]:
        R(f"W/all/{k}", [{"match": ALL, "w": specs[k]}], dict(wbits=fq.weight_bits(specs[k], 5120)))


def sens_a(R):
    R("bf16", [])
    specs = {}
    for b in [8, 6, 4]:
        for g in [0, 128, 64, 32]:
            specs[f"int{b}_{'pt' if g == 0 else 'g%d' % g}"] = {"bits": b, "group": g, "sym": False}
    specs.update({
        "mxint4_b32": {"bits": 4, "group": 32, "fmt": "mxint"},
        "mxfp4_b32": {"bits": 4, "group": 32, "fmt": "mxfp"},
        "fp4_g16": {"bits": 4, "group": 16, "fmt": "fp"},
        "mxfp6_b32": {"bits": 6, "group": 32, "fmt": "mxfp"},
        "mxint8_b32": {"bits": 8, "group": 32, "fmt": "mxint"},
        "fp8_pt": {"bits": 8, "fmt": "fp"},
        "int4_pt_clip0.9": {"bits": 4, "sym": False, "clip": 0.9},
        "int4_pt_keep0": {"bits": 4, "sym": False, "keep": {"pos": [0]}},
    })
    for t in TYPES:
        rx = fq.lin_regex(types=[t])
        for k in sorted(specs, key=lambda k: (specs[k]["bits"] != 4, k)):
            R(f"A/{t}/{k}", [{"match": rx, "a": specs[k]}])
    for k in ["int4_g128", "int4_g32", "mxfp4_b32", "mxint4_b32", "fp4_g16", "int6_pt", "fp8_pt", "int8_pt"]:
        R(f"A/all/{k}", [{"match": ALL, "a": specs[k]}])


def outlier(R):
    R("bf16", [])
    keeps = {"none": None, "pos0": {"pos": [0]}, "top1": {"topk": 1}, "top2": {"topk": 2}, "top4": {"topk": 4},
             "top8": {"topk": 8}, "th10": {"thresh": 10}, "th30": {"thresh": 30}, "pos0+th10": {"pos": [0], "thresh": 10}}
    for k, kp in keeps.items():
        R(f"A4/{k}", [{"match": ALL, "a": dict(A4, keep=kp)}])
    for k in ["none", "pos0", "top4", "th10"]:
        R(f"W4pcA4/{k}", [{"match": ALL, "w": W4pc, "a": dict(A4, keep=keeps[k])}])
        R(f"A4down/{k}", [{"match": fq.lin_regex(types=["down_proj"]), "a": dict(A4, keep=keeps[k])}])
        R(f"A4qkv/{k}", [{"match": fq.lin_regex(types=["q_proj", "k_proj", "v_proj"]), "a": dict(A4, keep=keeps[k])}])
    for k in ["none", "pos0"]:
        R(f"A6/{k}", [{"match": ALL, "a": {"bits": 6, "sym": False, "keep": keeps[k]}}])
        R(f"A8/{k}", [{"match": ALL, "a": dict(A8, keep=keeps[k])}])
        R(f"KV4/{k}", [{"kv": r"\d+", "k": dict(KV4, keep=keeps[k]), "v": dict(KV4, keep=keeps[k])}])
        R(f"KV3/{k}", [{"kv": r"\d+", "k": {"bits": 3, "sym": False, "keep": keeps[k]},
                        "v": {"bits": 3, "sym": False, "keep": keeps[k]}}])


# ---------------------------------------------------------------- statistics

def had(x, b):
    """Normalized Walsh-Hadamard on contiguous blocks of size b (power of 2) along the last dim."""
    shape = x.shape
    D = shape[-1]
    y = x.reshape(-1, D // b, b)
    h = 1
    while h < b:
        y = y.reshape(y.shape[0], D // b, b // (2 * h), 2, h)
        y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), dim=-2)
        h *= 2
    return (y.reshape(shape) / math.sqrt(b))


def pow2_block(D):
    return D & (-D)


SITES = ["attn_in", "o_in", "mlp_in", "down_in", "resid", "k", "v"]


@torch.no_grad()
def actstats(model, tok, nseq=64):
    out_pt, out_js = f"{P}/actstats.pt", f"{P}/actstats.json"
    if os.path.exists(out_js):
        log("actstats exists"); return
    fq.install_attention()
    calib = common.calib_data(tok, nseq)
    layers = model.model.layers
    nl = len(layers)
    dev = "cuda"
    acc = {}      # (site, i) -> dict of per-channel accumulators (float64)
    tok_amax = {s: torch.zeros(nl, nseq, calib.shape[1], dtype=torch.float16) for s in SITES}
    tok_rms = {s: torch.zeros(nl, nseq, calib.shape[1], dtype=torch.float16) for s in SITES}
    qerr = {}     # (site, i) -> {scheme: [err_energy, energy]}  (excluding position 0)
    sink = torch.zeros(nl, 64, 3)  # per layer/head: mass on pos 0, max mass on any other single key, count
    signs = {}
    S4g = {"bits": 4, "group": 128, "sym": False}
    MX = {"bits": 4, "group": 32, "fmt": "mxfp"}
    cur = {"j": 0}

    def record(site, i, x):
        # x: (T, D) float32
        j = cur["j"]
        D = x.shape[-1]
        a = acc.get((site, i))
        if a is None:
            a = acc[(site, i)] = {k: torch.zeros(D, dtype=torch.float64, device=dev)
                                  for k in ["s1", "s2", "s4", "sabs"]}
            a["amax"] = torch.zeros(D, device=dev); a["n"] = 0
        xd = x.double()
        a["s1"] += xd.sum(0); a["s2"] += (xd * xd).sum(0); a["s4"] += (xd ** 4).sum(0); a["sabs"] += xd.abs().sum(0)
        a["amax"] = torch.maximum(a["amax"], x.abs().amax(0)); a["n"] += x.shape[0]
        tok_amax[site][i, j, : x.shape[0]] = x.abs().amax(-1).half().cpu()
        tok_rms[site][i, j, : x.shape[0]] = x.pow(2).mean(-1).sqrt().half().cpu()
        if site in ("k", "v"):
            return
        # per-token quantization error of different A4 schemes (tokens 1..T-1)
        xs = x[1:]
        if D not in signs:
            g = torch.Generator(device="cpu").manual_seed(D)
            signs[D] = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float().to(dev)
        b = pow2_block(D)
        xh = had(xs * signs[D], b)
        e = qerr.setdefault((site, i), {})
        en = xs.pow(2).sum().item()
        for nm, y, sp in [("pt4", xs, A4), ("g128_4", xs, S4g), ("had_pt4", xh, A4), ("mxfp4", xs, MX),
                          ("had_mxfp4", xh, MX), ("pt8", xs, A8)]:
            err = (fq.fake_quant(y, sp) - y).pow(2).sum().item()
            v = e.setdefault(nm, [0.0, 0.0]); v[0] += err; v[1] += en

    hooks = []
    for i, L in enumerate(layers):
        for site, mod in [("attn_in", L.self_attn.q_proj), ("o_in", L.self_attn.o_proj),
                          ("mlp_in", L.mlp.gate_proj), ("down_in", L.mlp.down_proj)]:
            hooks.append(mod.register_forward_pre_hook(
                lambda m, inp, s=site, i=i: record(s, i, inp[0][0].float())))
        hooks.append(L.register_forward_hook(
            lambda m, inp, out, i=i: record("resid", i, (out[0] if isinstance(out, tuple) else out)[0].float())))

        def probe(mod, q, k, v, i=i):
            # k, v: (1, 8, T, 128) after QK-norm + RoPE (k); flatten heads -> channels
            T = k.shape[2]
            record("k", i, k[0].transpose(0, 1).reshape(T, -1).float())
            record("v", i, v[0].transpose(0, 1).reshape(T, -1).float())
            if cur["j"] < 8:  # attention sinks on the last 256 queries
                rep = q.shape[1] // k.shape[1]
                kk = k.repeat_interleave(rep, dim=1).float()
                qq = q[:, :, -256:].float()
                s = (qq @ kk.transpose(-1, -2)) * mod.scaling
                qpos = torch.arange(T - 256, T, device=q.device)
                s = s.masked_fill(torch.arange(T, device=q.device)[None, :] > qpos[:, None], float("-inf"))
                p = s.softmax(-1)[0]  # (H, 256, T)
                sink[i, :, 0] += p[..., 0].mean(-1).cpu()
                sink[i, :, 1] += p[..., 1:].amax(-1).mean(-1).cpu()
                sink[i, :, 2] += 1
        L.self_attn.fq_probe = probe

    t0 = time.time()
    for j in range(nseq):
        cur["j"] = j
        model(calib[j: j + 1].to(dev))
        if j % 8 == 0:
            log(f"actstats seq {j} ({time.time() - t0:.0f}s)")
    for h in hooks:
        h.remove()
    for L in layers:
        L.self_attn.fq_probe = None

    # ---- summaries
    tens = {"tok_amax": tok_amax, "tok_rms": tok_rms, "tokens": calib, "sink": sink, "chan": {}}
    js = {"nseq": nseq, "sites": {}}
    for (site, i), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = a["n"]
        mean = a["s1"] / n; m2 = a["s2"] / n
        var = (m2 - mean ** 2).clamp(min=1e-20)  # noqa: F841
        # kurtosis of each channel (about 0), excess not subtracted
        # kurtosis about 0 (raw 4th moment / m2^2); equals the usual kurtosis for zero-mean channels
        kurt0 = (a["s4"] / n) / (m2 ** 2).clamp(min=1e-30)
        rms = m2.sqrt(); amax = a["amax"].double(); meanabs = a["sabs"] / n
        tens["chan"][f"{site}/{i}"] = {"rms": rms.float().cpu(), "amax": amax.float().cpu(),
                                       "meanabs": meanabs.float().cpu(), "mean": mean.float().cpu(),
                                       "kurt0": kurt0.float().cpu()}
        ta = tok_amax[site][i].float(); tr = tok_rms[site][i].float()
        ratio = (ta[:, 1:] / tr[:, 1:].clamp(min=1e-6)).flatten()
        top = torch.topk(amax, 5)
        d = {
            "chan_amax_max": amax.max().item(), "chan_amax_median": amax.median().item(),
            "chan_rms_max_over_median": (rms.max() / rms.median()).item(),
            "top_chan": top.indices.tolist(), "top_chan_amax": [round(v, 2) for v in top.values.tolist()],
            "chan_kurt0_median": kurt0.median().item(), "chan_kurt0_max": kurt0.max().item(),
            "frac_chan_kurt0_gt10": (kurt0 > 10).float().mean().item(),
            "tok_amax_pos0_mean": ta[:, 0].mean().item(), "tok_amax_rest_median": ta[:, 1:].median().item(),
            "tok_amax_rest_max": ta[:, 1:].max().item(),
            "tok_rms_pos0_mean": tr[:, 0].mean().item(), "tok_rms_rest_median": tr[:, 1:].median().item(),
            "tok_maxrms_ratio_median": ratio.median().item(),
            "tok_maxrms_ratio_p99": ratio.quantile(0.99).item(),
            "frac_tok_amax_gt10x_median": (ta[:, 1:] > 10 * ta[:, 1:].median()).float().mean().item(),
        }
        if (site, i) in qerr:
            d["qrelerr"] = {k: math.sqrt(v[0] / v[1]) for k, v in qerr[(site, i)].items()}
        # massive tokens: top 8 (seq, pos) by absmax, with token ids
        flat = ta.flatten(); tv, ti = flat.topk(8)
        d["massive"] = [[int(ix // ta.shape[1]), int(ix % ta.shape[1]), int(calib[ix // ta.shape[1], ix % ta.shape[1]]),
                         round(float(v), 1)] for v, ix in zip(tv, ti)]
        js["sites"].setdefault(site, {})[str(i)] = d
    sk = sink[..., :2] / sink[..., 2:].clamp(min=1)
    js["sink_mass_pos0_by_layer"] = [round(x, 4) for x in sk[..., 0].mean(1).tolist()]
    js["sink_max_other_by_layer"] = [round(x, 4) for x in sk[..., 1].mean(1).tolist()]
    torch.save(tens, out_pt)
    common.write_json(out_js, js)
    log(f"actstats done {time.time() - t0:.0f}s")


@torch.no_grad()
def wstats(model):
    out_js, out_pt = f"{P}/wstats.json", f"{P}/wstats.pt"
    res = common.read_json(out_js, {})
    for _ in range(240):  # wait for actstats (salience needs activation stats)
        if os.path.exists(f"{P}/actstats.json"):
            break
        time.sleep(30)
    chan = torch.load(f"{P}/actstats.pt")["chan"]
    site_of = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
               "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in"}
    layers = model.model.layers
    nl = len(layers)
    W4g = {"bits": 4, "group": 128, "sym": False}
    spec_layers = [0, 1, 2, 3, 8, 16, 24, 32, 40, 48, 56, 60, 61, 62, 63]
    spectra = torch.load(out_pt) if os.path.exists(out_pt) else {}
    rand_cache = {}
    R_LIST = [1, 4, 16, 64, 256]

    def svals(M):
        M = M.double()
        G = M.T @ M if M.shape[0] >= M.shape[1] else M @ M.T
        return torch.linalg.eigvalsh(G).clamp(min=0).sqrt().flip(0).float()

    def energy_frac(sv):
        e = sv.double() ** 2; c = e.cumsum(0) / e.sum()
        return {str(r): round(c[r - 1].item(), 4) for r in R_LIST if r <= len(c)}

    for i in range(nl):
        for t in TYPES:
            key = f"{i}/{t}"
            if key in res:
                continue
            W = dict(layers[i].named_modules())[
                ("self_attn." if t in TYPES[:4] else "mlp.") + t].weight.data.float()
            c = chan[f"{site_of[t]}/{i}"]
            xr = c["rms"].to(W.device); xm = c["meanabs"].to(W.device)
            row_amax = W.abs().amax(1); col_amax = W.abs().amax(0); col_norm = W.norm(dim=0)
            kurt = ((W - W.mean()) ** 4).mean() / W.var() ** 2
            Q4 = fq.quant_weight(W, W4g); Q4pc = fq.quant_weight(W, W4pc)
            E = W - Q4
            # output error energy weighted by input activation RMS: ||E diag(xr)||^2 ~ E||E x||^2
            relw = lambda Q: ((W - Q) * xr).norm().item() / (W * xr).norm().item()  # noqa: F821  (W is deleted later in the loop)
            sal = xm * col_norm
            k1 = max(1, int(0.01 * W.shape[1]))
            top_sal = sal.topk(k1).indices; top_act = xm.topk(k1).indices; top_w = col_norm.topk(k1).indices
            d = {
                "row_amax_max_over_median": (row_amax.max() / row_amax.median()).item(),
                "col_amax_max_over_median": (col_amax.max() / col_amax.median()).item(),
                "kurtosis": kurt.item(),
                "relerr_w4g128": (E.norm() / W.norm()).item(), "relerr_w4pc": ((W - Q4pc).norm() / W.norm()).item(),
                "relerr_w4g128_actw": relw(Q4), "relerr_w4pc_actw": relw(Q4pc),
                "sal_top1pct_share": (sal[top_sal].sum() / sal.sum()).item(),
                "sal_top1pct_overlap_act": len(set(top_sal.tolist()) & set(top_act.tolist())) / k1,
                "sal_top1pct_overlap_wnorm": len(set(top_sal.tolist()) & set(top_w.tolist())) / k1,
                "top_sal_chan": top_sal[:8].tolist(),
            }
            if i in spec_layers:
                sv = svals(E); svw = svals(E * xr)
                shp = tuple(W.shape)
                if shp not in rand_cache:
                    g = torch.Generator(device="cpu").manual_seed(0)
                    rand_cache[shp] = svals(torch.randn(shp, generator=g).to(W.device))
                d["err_energy_topr"] = energy_frac(sv)
                d["err_actw_energy_topr"] = energy_frac(svw)
                d["random_energy_topr"] = energy_frac(rand_cache[shp])
                spectra[key] = {"sv": sv.cpu(), "sv_actw": svw.cpu()}
            res[key] = d
            del W, Q4, Q4pc, E
        common.write_json(out_js, res)
        torch.save(spectra, out_pt)
        log(f"wstats layer {i}")


@torch.no_grad()
def prop(model, tok):
    out = f"{P}/prop.json"
    res = common.read_json(out, {})
    ids = common.ppl_ids("c4dev", tok)[:8]
    layers = model.model.layers
    nl = len(layers)
    dev = "cuda"
    cap = []

    def hk(m, inp, o):
        cap.append((o[0] if isinstance(o, tuple) else o).detach())

    hooks = [L.register_forward_hook(hk) for L in layers]

    def run():
        hs, nll, logits_all = [], 0.0, []
        for j in range(0, ids.shape[0], 2):
            cap.clear()
            x = ids[j: j + 2].to(dev)
            lg = model(x).logits[:, :-1].float()
            nll += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                                                     reduction="sum").item()
            hs.append(torch.stack([c for c in cap]).cpu())  # (nl, 2, T, D)
            logits_all.append(lg.log_softmax(-1).to(torch.bfloat16).cpu())
        return torch.cat(hs, 1), nll / (ids.shape[0] * (ids.shape[1] - 1)), logits_all

    ref, ref_nll, ref_lp = run()
    res["bf16"] = {"nll": ref_nll}
    rnorm = torch.tensor([ref[l, :, 1:].to(dev).float().norm().item() for l in range(nl)])
    rnorm0 = torch.tensor([ref[l, :, 0].to(dev).float().norm().item() for l in range(nl)])
    starts = [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 62, 63]
    for setting, mk in [("W4pcA4", lambda rx: [{"match": rx, "w": W4pc, "a": A4}]),
                        ("A4", lambda rx: [{"match": rx, "a": A4}]),
                        ("W3g128", lambda rx: [{"match": rx, "w": W3g}])]:
        for i in starts + ["all"]:
            key = f"{setting}/{i}"
            if key in res:
                continue
            rx = fq.lin_regex() if i == "all" else fq.lin_regex(layers=[i])
            q = fq.FQ(model).apply(mk(rx))
            hq, nll, lp = run()
            q.reset()
            diff, diff0, cos = [], [], []
            for l in range(nl):
                a_, b_ = hq[l].to(dev).float(), ref[l].to(dev).float()
                diff.append((a_[:, 1:] - b_[:, 1:]).norm().item()); diff0.append((a_[:, 0] - b_[:, 0]).norm().item())
                cos.append(torch.nn.functional.cosine_similarity(a_[:, 1:], b_[:, 1:], dim=-1).mean().item())
            diff, diff0 = torch.tensor(diff), torch.tensor(diff0)
            kl = sum(((r.to(dev).float().exp() * (r.to(dev).float() - p.to(dev).float())).sum(-1).mean().item())
                     for r, p in zip(ref_lp, lp)) / len(lp)
            res[key] = {"relerr": (diff / rnorm).tolist(), "relerr_pos0": (diff0 / rnorm0.clamp(min=1e-6)).tolist(),
                        "cos": cos, "dnll": nll - ref_nll, "kl": kl}
            common.write_json(out, res)
            log(f"prop {key} dnll {nll - ref_nll:+.5f} kl {kl:.5f}")
            del hq, lp
    for h in hooks:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    a = ap.parse_args()
    tok = common.load_tokenizer()
    model = common.load_model()
    fq.install_attention()
    nl = len(model.model.layers)
    c4dev = common.ppl_ids("c4dev", tok)
    if a.cmd == "attrib_type":
        attrib_type(Runner("attrib_type", model, c4dev))
    elif a.cmd == "attrib_layer":
        attrib_layer(Runner("attrib_layer", model, c4dev), nl)
    elif a.cmd == "attrib_lt":
        attrib_lt(Runner("attrib_lt", model, c4dev[:16]), nl)
    elif a.cmd == "sens_w":
        sens_w(Runner("sens_w", model, c4dev))
    elif a.cmd == "sens_a":
        sens_a(Runner("sens_a", model, c4dev))
    elif a.cmd == "outlier":
        outlier(Runner("outlier", model, c4dev))
    elif a.cmd == "actstats":
        actstats(model, tok)
    elif a.cmd == "wstats":
        wstats(model)
    elif a.cmd == "prop":
        prop(model, tok)
    else:
        sys.exit(f"unknown {a.cmd}")
    print("peak_mem_gb", common.peak_gb(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
