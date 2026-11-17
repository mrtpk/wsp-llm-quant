"""Transforms and solvers for P2 builds (Qwen3, plain transformers). All gradient-free.

cfg["pipeline"] is a list of ops, applied in order:
  {"op": "scale", "mode": "awq"|"sq", "wspec": S, "aspec": S|None, "grid": 20, "nseq": 32, "sites": [...]}
        AWQ (mode awq: s = mean|x|^a) or SmoothQuant-style (mode sq: s = max|x|^a / max|w|^(1-a)) per-channel
        scales, a searched per site and layer on the output error of the quantized linear(s) (with act quant
        if aspec). Folded exactly into the previous op: input_layernorm -> qkv, post_attention_layernorm ->
        gate/up, v_proj rows -> o_proj cols (GQA-expanded), up_proj rows -> down_proj cols.
  {"op": "rotate", "r1": true, "r4": true, "seed": 0}
        QuaRot-style: fuse RMSNorm weights, rotate the residual stream by an orthogonal Q (5120 = 5 x 1024:
        kron(random 5x5 orthogonal, Hadamard 1024) with random signs); r4 = online block-Hadamard (1024) at
        the down_proj input, folded into down_proj weights.
  {"op": "qkrot"}   online Hadamard over head_dim on q and k after RoPE (scores unchanged; for K quant)
  {"op": "gptq", "wspec": S, "except": [[regex, S], ...], "act_order": true, "nsamples": 128, "seed": 0,
        "asym_alpha": 0.0, "percdamp": 0.01, "calib": "mix", "aq_rules": rules-for-act-quant-in-q-stream|None}
  {"op": "rtn", "rules": fq rules}          quantize weights in place (no backup)
  {"op": "lowrank", "match": regex, "rank": r, "nseq": 16}
        W <- Wq + L, L = rank-r minimiser of ||(W_orig - Wq - L) X^T|| (whitened SVD); counted as extra params
Returns info dict (extra_params, timings, chosen alphas, ...).
"""
import math, re, time
import torch
from r2 import common, fq
from r2.fq import TYPES

DEV = __import__("os").environ.get("WSP_DEV", "cuda")


def _lin(layer, t):
    return getattr(layer.self_attn if t in TYPES[:4] else layer.mlp, t)


def _name(i, t):
    return fq.lin_name(i, t)


# ---------------------------------------------------------------- layer-wise calibration plumbing

class _Stop(Exception):
    pass


@torch.no_grad()
def layer0_inputs(model, calib, bs=8):
    """Hidden states entering layer 0 for all calib rows, plus the kwargs the decoder layer is called with."""
    layers = model.model.layers
    store = {}

    def pre(mod, args, kwargs):
        store.setdefault("x", []).append(args[0] if args else kwargs["hidden_states"])
        if "kw" not in store:
            store["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise _Stop

    h = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
    for j in range(0, calib.shape[0], bs):
        try:
            model(calib[j: j + bs].to(DEV))
        except _Stop:
            pass
    h.remove()
    return torch.cat(store["x"], 0), store["kw"]


def run_layer(layer, X, kw, bs=8):
    out = torch.empty_like(X)
    for j in range(0, X.shape[0], bs):
        o = layer(X[j: j + bs], **kw)
        out[j: j + bs] = o[0] if isinstance(o, tuple) else o
    return out


# ---------------------------------------------------------------- quantization with fixed group params

def group_params(W, spec):
    """W (out, in) float -> (scale, zero) of shape (out, ngroups). zero=None for symmetric / float formats."""
    D = W.shape[1]
    g = spec.get("group", 0) or D
    Wg = W.reshape(W.shape[0], D // g, g)
    fmt, b = spec.get("fmt", "int"), spec["bits"]
    amax = Wg.abs().amax(-1)
    if fmt == "int":
        if spec.get("sym", False):
            qmax = 2 ** (b - 1) - 1
            return (amax / qmax).clamp(min=1e-10), None
        qmax = 2 ** b - 1
        mx = Wg.amax(-1).clamp(min=0); mn = Wg.amin(-1).clamp(max=0)
        s = ((mx - mn) / qmax).clamp(min=1e-10)
        return s, torch.round(-mn / s)
    if fmt == "mxint":
        qmax = 2 ** (b - 1) - 1
        return torch.exp2(torch.ceil(torch.log2((amax / qmax).clamp(min=1e-30)))), None
    e, m, maxv = fq._FP[b]
    if fmt == "mxfp":
        return torch.exp2(torch.floor(torch.log2(amax.clamp(min=1e-30))) - math.floor(math.log2(maxv))), None
    return (amax / maxv).clamp(min=1e-12), None


def q_with(w, s, z, spec):
    """Quantize column(s) w (out, k) with scale s (out, 1 or k) / zero z."""
    fmt, b = spec.get("fmt", "int"), spec["bits"]
    if fmt == "int":
        if z is None:
            qmax = 2 ** (b - 1) - 1
            return torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
        return (torch.clamp(torch.round(w / s) + z, 0, 2 ** b - 1) - z) * s
    if fmt == "mxint":
        qmax = 2 ** (b - 1) - 1
        return torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
    e, m, maxv = fq._FP[b]
    return fq._fp_round(w / s, e, m, maxv) * s


DAMP_LOG: list = []   # damping actually used whenever it had to be raised


@torch.no_grad()
def gptq_solve(W, H, spec, act_order=True, percdamp=0.01, blocksize=128):
    """Standard GPTQ with static (contiguous, pre-computed) group params. W (out, in) float32."""
    if spec is None or spec["bits"] >= 16:
        return W
    W = W.clone().float()
    H = H.clone()
    ncol = W.shape[1]
    g = spec.get("group", 0) or ncol
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    S, Z = group_params(W, spec)
    perm = torch.argsort(torch.diag(H), descending=True) if act_order else torch.arange(ncol, device=W.device)
    W = W[:, perm]; H = H[perm][:, perm]
    Hinv = None
    for k in range(5):   # float64 Cholesky; on failure raise the damping x10 (badly conditioned H from massive acts)
        Hd = H.double()
        Hd.diagonal().add_(percdamp * (10 ** k) * torch.mean(torch.diag(Hd)))
        try:
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True).float()
            break
        except torch._C._LinAlgError:
            print(f"gptq_solve: cholesky failed at percdamp={percdamp * 10 ** k:g}, retrying", flush=True)
    if Hinv is None:
        raise RuntimeError("gptq_solve: Hessian not positive-definite even with 1e4x damping")
    if k:
        DAMP_LOG.append(percdamp * 10 ** k)
    Q = torch.zeros_like(W)
    gidx = perm // g
    for i1 in range(0, ncol, blocksize):
        i2 = min(i1 + blocksize, ncol)
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]; d = Hinv1[i, i]
            gi = gidx[i1 + i]
            q = q_with(w[:, None], S[:, gi: gi + 1], None if Z is None else Z[:, gi: gi + 1], spec)[:, 0]
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err[:, None] @ Hinv1[i, i:][None, :]
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)]



def chol_damped(H, damp, what=""):
    """Lower Cholesky factor of H + damp*mean(diag)*I in float64; on failure retry with damp x10 (up to 4 times).
    Every escalation is recorded in DAMP_LOG (reported in the result's info)."""
    for k in range(5):
        Hd = H.double(); Hd.diagonal().add_(damp * (10 ** k) * Hd.diagonal().mean())
        try:
            L = torch.linalg.cholesky(Hd)
            if k:
                DAMP_LOG.append((what, damp * 10 ** k))
            return L
        except torch._C._LinAlgError:
            print(f"chol_damped[{what}]: failed at damp={damp * 10 ** k:g}, retrying", flush=True)
    raise RuntimeError(f"chol_damped[{what}]: not PD even at damp={damp * 1e4:g}")

def spec_for(name, default, exceptions):
    s = default
    for rx, sp in exceptions or []:
        if re.search(rx, name):
            s = sp
    return s


SEQ = [["q_proj", "k_proj", "v_proj"], ["o_proj"], ["gate_proj", "up_proj"], ["down_proj"]]


@torch.no_grad()
def gptq(model, tok, op, log):
    calib = common.calib_data(tok, op.get("nsamples", 128), seed=op.get("seed", 0), mix=op.get("calib", "mix"))
    X, kw = layer0_inputs(model, calib)
    alpha = op.get("asym_alpha", 0.0)
    Xfp = X.clone() if alpha > 0 else None
    aq = op.get("aq_rules")
    layers = model.model.layers
    for i, layer in enumerate(layers):
        t0 = time.time()
        fpw = {t: _lin(layer, t).weight.data.clone() for t in TYPES}
        for grp in SEQ:
            first = _lin(layer, grp[0])
            d = first.in_features
            H = torch.zeros(d, d, device=DEV); C = torch.zeros(d, d, device=DEV) if alpha > 0 else None
            buf = {}
            hk = first.register_forward_pre_hook(lambda m, a: buf.__setitem__("x", a[0].reshape(-1, a[0].shape[-1]).float()))
            for j in range(0, X.shape[0], 8):
                if alpha > 0:   # FP stream: original weights of this layer, no act quant
                    sw = {t: _lin(layer, t).weight.data for t in TYPES}
                    for t in TYPES: _lin(layer, t).weight.data = fpw[t]
                    layer(Xfp[j: j + 8], **kw); xf = buf["x"]
                    for t in TYPES: _lin(layer, t).weight.data = sw[t]
                q = fq.FQ(model).apply(aq) if aq else None   # optional act quant in the quantized stream
                layer(X[j: j + 8], **kw); xq = buf["x"]
                if q: q.reset()
                H += xq.T @ xq
                if C is not None:
                    C += xf.T @ xq
            hk.remove()
            ntok = X.shape[0] * X.shape[1]
            H /= ntok
            M = None
            if C is not None:
                C /= ntok
                M = (C - H).double() @ torch.cholesky_inverse(chol_damped(H, 0.01, "asym target"))
            for t in grp:
                lin = _lin(layer, t)
                W = fpw[t].float()
                if M is not None:
                    W = W + alpha * (W.double() @ M).float()
                sp = spec_for(_name(i, t), op["wspec"], op.get("except"))
                lin.weight.data = gptq_solve(W, 2 * H, sp, op.get("act_order", True), op.get("percdamp", 0.01)).to(lin.weight.dtype)
            del H, C, M
        if alpha > 0:
            qw = {t: _lin(layer, t).weight.data for t in TYPES}
            for t in TYPES: _lin(layer, t).weight.data = fpw[t]
            Xfp = run_layer(layer, Xfp, kw)
            for t in TYPES: _lin(layer, t).weight.data = qw[t]
        q = fq.FQ(model).apply(aq) if aq else None
        X = run_layer(layer, X, kw)
        if q: q.reset()
        del fpw
        torch.cuda.empty_cache()
        if i % 8 == 0 or i == len(layers) - 1:
            log(f"gptq layer {i} {time.time() - t0:.1f}s")
    return {"damp_raised": list(DAMP_LOG)}


# ---------------------------------------------------------------- AWQ / SmoothQuant scale search with folding

SITES = {  # name: (linears, kind of previous op)
    "attn_in": (["q_proj", "k_proj", "v_proj"], "norm_attn"),
    "v_o": (["o_proj"], "v_rows"),
    "mlp_in": (["gate_proj", "up_proj"], "norm_mlp"),
    "up_down": (["down_proj"], "up_rows"),
}


@torch.no_grad()
def scale_search(model, tok, op, log):
    calib = common.calib_data(tok, op.get("nseq", 32), seed=op.get("seed", 0) + 77)
    X, kw = layer0_inputs(model, calib)
    grid = op.get("grid", 20)
    mode, ws, asp = op.get("mode", "awq"), op["wspec"], op.get("aspec")
    sites = op.get("sites", list(SITES))
    layers_sel = op.get("layers")  # optional subset
    cfg = model.config
    rep = cfg.num_attention_heads // cfg.num_key_value_heads
    hd = cfg.head_dim
    chosen = {}
    ntok = op.get("ntok", 8192)
    for i, layer in enumerate(model.model.layers):
        if layers_sel is None or i in layers_sel:
            for site in sites:
                lins, prev = SITES[site]
                mods = [_lin(layer, t) for t in lins]
                buf = []
                hk = mods[0].register_forward_pre_hook(lambda m, a: buf.append(a[0].reshape(-1, a[0].shape[-1])))
                run_layer(layer, X, kw)
                hk.remove()
                xs = torch.cat(buf, 0)
                del buf
                xmean = xs.float().abs().mean(0); xmax = xs.float().abs().amax(0)
                g = torch.Generator(device="cpu").manual_seed(i)
                sub = xs[torch.randperm(xs.shape[0], generator=g)[:ntok].to(DEV)].float()
                del xs
                Ws = [m.weight.data.float() for m in mods]
                ref = [sub @ W.T for W in Ws]
                wmax = torch.stack([W.abs().amax(0) for W in Ws]).amax(0)
                best = (float("inf"), 0.0, None)
                for a in ([op["fixed_alpha"]] if "fixed_alpha" in op else [k / grid for k in range(grid + 1)]):
                    if mode == "awq":
                        s = xmean.clamp(min=1e-5) ** a
                    else:
                        s = (xmax.clamp(min=1e-5) ** a) / (wmax.clamp(min=1e-5) ** (1 - a))
                    s = s / (s.max() * s.min()).sqrt()
                    if prev == "v_rows":  # o input c = head*hd + d; heads k*rep..k*rep+rep-1 share v head k
                        s = s.view(-1, rep, hd).mean(1).repeat_interleave(rep, 0).reshape(-1)
                    xin = fq.fake_quant(sub / s, asp) if asp else sub / s
                    err = sum((r - xin @ fq.quant_weight(W * s, ws).float().T).pow(2).sum().item() for r, W in zip(ref, Ws))
                    if err < best[0]:
                        best = (err, a, s)
                err0 = best[0]  # noqa: F841
                s = best[2]
                # fold
                if prev == "v_rows":
                    # o input channel c belongs to q-head h = c // hd, using kv head h // rep. Make s consistent per kv group:
                    s8 = s.view(cfg.num_key_value_heads, rep, hd)
                    s_kv = s8.mean(1)  # (8, hd)
                    s = s_kv.repeat_interleave(rep, 0).reshape(-1)
                    layer.self_attn.v_proj.weight.data.div_(s_kv.reshape(-1, 1).to(layer.self_attn.v_proj.weight.dtype))
                    if layer.self_attn.v_proj.bias is not None:
                        layer.self_attn.v_proj.bias.data.div_(s_kv.reshape(-1).to(layer.self_attn.v_proj.bias.dtype))
                elif prev == "norm_attn":
                    layer.input_layernorm.weight.data.div_(s.to(layer.input_layernorm.weight.dtype))
                elif prev == "norm_mlp":
                    layer.post_attention_layernorm.weight.data.div_(s.to(layer.post_attention_layernorm.weight.dtype))
                elif prev == "up_rows":
                    layer.mlp.up_proj.weight.data.div_(s.reshape(-1, 1).to(layer.mlp.up_proj.weight.dtype))
                for m in mods:
                    m.weight.data.mul_(s.reshape(1, -1).to(m.weight.dtype))
                chosen[f"{i}/{site}"] = best[1]
                del sub, ref, Ws
        X = run_layer(layer, X, kw)
        if i % 8 == 0:
            log(f"scale layer {i}: " + " ".join(f"{k.split('/')[1]}={v:.2f}" for k, v in chosen.items() if k.startswith(f"{i}/")))
    return {"alphas": chosen}


# ---------------------------------------------------------------- rotation

def _had_matrix(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


@torch.no_grad()
def rotate(model, op, log):
    cfg = model.config
    assert not cfg.tie_word_embeddings
    D = cfg.hidden_size
    b = D & (-D)
    g = torch.Generator(device="cpu").manual_seed(op.get("seed", 0))
    R5 = torch.linalg.qr(torch.randn(D // b, D // b, generator=g, dtype=torch.float64))[0].contiguous()
    signs = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).double()
    Q = (signs[:, None] * torch.kron(R5, _had_matrix(b).double())).to(DEV)   # orthogonal D x D
    layers = model.model.layers

    def fuse(norm, lins):
        w = norm.weight.data.double()
        for l in lins:
            l.weight.data = (l.weight.data.double() * w[None, :]).to(l.weight.dtype)
        norm.weight.data.fill_(1.0)

    if op.get("r1", True):
        for L in layers:
            fuse(L.input_layernorm, [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj])
            fuse(L.post_attention_layernorm, [L.mlp.gate_proj, L.mlp.up_proj])
        fuse(model.model.norm, [model.lm_head])
        E = model.model.embed_tokens.weight
        E.data = (E.data.double() @ Q).to(E.dtype)
        model.lm_head.weight.data = (model.lm_head.weight.data.double() @ Q).to(model.lm_head.weight.dtype)
        for L in layers:
            for l in [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj, L.mlp.gate_proj, L.mlp.up_proj]:
                l.weight.data = (l.weight.data.double() @ Q).to(l.weight.dtype)
            for l in [L.self_attn.o_proj, L.mlp.down_proj]:
                l.weight.data = (Q.T @ l.weight.data.double()).to(l.weight.dtype)
                if l.bias is not None:
                    l.bias.data = (Q.T @ l.bias.data.double()).to(l.bias.dtype)
    if op.get("r4", True):
        from r2.fq import had
        F = layers[0].mlp.down_proj.in_features
        bb = F & (-F)
        for L in layers:
            dp = L.mlp.down_proj
            dp.weight.data = had(dp.weight.data.float(), bb).to(dp.weight.dtype)   # W H (H symmetric)
            dp.register_forward_pre_hook(lambda m, a, bb=bb: (had(a[0].float(), bb).to(a[0].dtype),) + tuple(a[1:]))
    return {}


def qkrot(model, op, log):
    for L in model.model.layers:
        L.self_attn.fq_rot_qk = True
    return {}


# ---------------------------------------------------------------- RTN in place, low-rank correction

@torch.no_grad()
def rtn(model, op, log, originals):
    names, wmap, _ = __import__("r2.p2", fromlist=["eff_specs"]).eff_specs(model, op["rules"])
    mods = dict(model.named_modules())
    for n in names:
        if n in wmap and wmap[n] is not None:
            originals.setdefault(n, mods[n].weight.data.clone())
            mods[n].weight.data.copy_(fq.quant_weight(mods[n].weight.data, wmap[n]))
    return {}


@torch.no_grad()
def lowrank(model, tok, op, log, originals):
    mods = dict(model.named_modules())
    targets = [n for n in mods if re.search(op["match"], n)]
    Hs = {n: torch.zeros(mods[n].in_features, mods[n].in_features, device=DEV, dtype=torch.float64) for n in targets}
    def acc(m, a, n):
        x = a[0].reshape(-1, a[0].shape[-1]).double()
        Hs[n].add_(x.T @ x)   # returns None: input unchanged
    hooks = [mods[n].register_forward_pre_hook(lambda m, a, n=n: acc(m, a, n)) for n in targets]
    calib = common.calib_data(tok, op.get("nseq", 16), seed=op.get("seed", 0) + 55)
    for j in range(calib.shape[0]):
        model(calib[j: j + 1].to(DEV))
    for h in hooks:
        h.remove()
    extra = 0
    r = op["rank"]
    for n in targets:
        lin = mods[n]
        W0 = originals[n].double(); Wq = lin.weight.data.double()
        H = Hs[n] / (calib.numel())
        Lc = chol_damped(H, 1e-4, "lowrank")                 # H = L L^T ; ||E X^T||^2 = ||E L||^2
        U, S, Vh = torch.linalg.svd((W0 - Wq) @ Lc, full_matrices=False)
        corr = (U[:, :r] * S[:r]) @ torch.linalg.solve_triangular(Lc.T, Vh[:r].T, upper=True).T
        lin.weight.data = (Wq + corr).to(lin.weight.dtype)
        extra += r * (lin.in_features + lin.out_features)
        tot = S.pow(2).sum().item()
        log(f"lowrank {n} r={r}: captured {S[:r].pow(2).sum().item() / tot:.3f} of weighted error energy")
    return {"extra_params": extra}


def run_pipeline(model, tok, cfg, log=print):
    info = {"extra_params": 0, "ops": []}
    originals = {}
    for op in cfg.get("pipeline", []):
        t = time.time()
        k = op["op"]
        if k == "scale":
            r = scale_search(model, tok, op, log)
        elif k == "rotate":
            r = rotate(model, op, log)
        elif k == "qkrot":
            r = qkrot(model, op, log)
        elif k == "gptq":
            for n, m in model.named_modules():
                if re.search(op.get("keep_orig", "(?!)"), n):
                    originals[n] = m.weight.data.clone()
            r = gptq(model, tok, op, log)
        elif k == "rtn":
            r = rtn(model, op, log, originals)
        elif k == "lowrank":
            r = lowrank(model, tok, op, log, originals)
        else:
            raise ValueError(k)
        info["extra_params"] += r.pop("extra_params", 0)
        info["ops"].append(dict(op=k, sec=round(time.time() - t, 1), **r))
        log(f"op {k} done {time.time() - t:.0f}s")
    return info
