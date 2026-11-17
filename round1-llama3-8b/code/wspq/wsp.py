"""WSP quantizer for Llama 3 8B at W4A4KV4. No gradients anywhere.

Pipeline (each stage optional, controlled by a JSON config):
  1. smooth_down: SmoothQuant-style per-channel scale on the down_proj input, folded
     into up_proj rows / down_proj columns (exact function; closed form, 1 number: alpha).
  2. fuse norms; rotate with R1 (residual) and R2 (per head) = random Hadamard from a seed,
     or best-of-N seeds by a W16A4KV4 proxy on practice data (1 discrete variable),
     or a rotation file.
  3. online Hadamard R4 (down_proj input) and R3 (q/k after RoPE) as in SpinQuant.
  4. activation / KV quantizers configured BEFORE weight calibration.
  5. weights: GPTQ, optionally with asymmetric calibration:
        min ||X W^T - Xq What^T||,  X = FP-chain input, Xq = quantized-chain input (A4)
     solved as  W* = W + a*(W C H^-1 - W),  C = X^T Xq, H = Xq^T Xq  (least squares, closed form)
     then GPTQ(W*, H). With a = 0 and Xq = X this is ordinary GPTQ.
"""
import argparse, copy, json, os, random, time
import torch
import torch.nn as nn
from wspq import common

from utils import fuse_norm_utils, hadamard_utils, quant_utils  # noqa: E402
from eval_utils import gptq_utils, rotation_utils  # noqa: E402

DEFAULT = dict(
    seed=0,
    # calibration
    calib="wikitext2", nsamples=128, calib_seqlen=2048,
    # rotation
    rot="hadamard",          # hadamard | file
    rot_file=None,
    r1_seed=None,            # None -> seed
    r1_search=0,             # N>0: best of N R1 seeds by proxy
    r2_seed=None,
    proxy_nseq=32,
    # smoothing of down_proj input (0 = off)
    smooth_down=0.0,         # exponent a of the scale (0 = off)
    smooth_mode="max",       # max | l2
    r4_sign_seed=None,       # random signs on the down_proj input before online Hadamard R4
    r3_sign_seed=None,       # random signs on q/k head dims (RoPE-pair consistent) before R3
    # quantizer settings (SpinQuant defaults for 4-4-4)
    w_bits=4, a_bits=4, k_bits=4, v_bits=4,
    w_clip=True, w_asym=False, a_asym=True, k_asym=True, v_asym=True,
    a_clip=1.0, k_clip=1.0, v_clip=1.0, k_groupsize=128, v_groupsize=128,
    # weight quantization
    wq="gptq",               # gptq | rtn
    act_order=False, percdamp=0.01,
    calib_actq=False,        # Hessian from A4-quantized inputs of the quantized chain
    asym_alpha=0.0,          # blend toward the least-squares target W C H^-1
    ls_damp=0.01,
)


# ---------------------------------------------------------------- data

def calib_data(cfg, tok):
    n, L, seed = cfg["nsamples"], cfg["calib_seqlen"], cfg["seed"]
    path = os.path.join(os.environ.get("WSP_DATA", "/models/tmp/wsp/data"), "wikitext2_train_ids.pt")
    if os.path.exists(path):
        ids = torch.load(path)
    else:
        import datasets
        d = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
        ids = tok("\n\n".join(d["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
        torch.save(ids, path)
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        i = rng.randint(0, ids.numel() - L - 1)
        rows.append(ids[i : i + L])
    if cfg["calib"] == "mix":
        c4 = c4_train_rows(tok, n // 2, L, seed)
        rows = rows[: n - n // 2] + c4
    return torch.stack(rows)


def c4_train_rows(tok, n, L, seed):
    import datasets
    d = datasets.load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
    rng = random.Random(seed + 1000)
    rows = []
    while len(rows) < n:
        t = d[rng.randint(0, len(d) - 1)]["text"]
        e = tok(t, add_special_tokens=False).input_ids
        if len(e) <= L:
            continue
        j = rng.randint(0, len(e) - L - 1)
        rows.append(torch.tensor(e[j : j + L]))
    return rows


def proxy_data(tok, nseq, L=2048):
    """Practice-set proxy: wikitext2 TRAIN windows from a fixed seed disjoint from calibration seeds."""
    cfg = dict(DEFAULT, nsamples=nseq, calib_seqlen=L, seed=987654, calib="wikitext2")
    return calib_data(cfg, tok)


# ---------------------------------------------------------------- transforms

@torch.no_grad()
def smooth_down_proj(model, calib, alpha, mode="max", sign_seed=None, nseq=32):
    """Scale intermediate channel c: x_c -> x_c / s_c (via up_proj rows), down[:, c] *= s_c.
    mode=max: SmoothQuant s = amax_x^a / amax_w^(1-a)   (SmoothRot)
    mode=l2:  s = (||X_c||_2 / ||W_c||_2)^a, a=0.5 is L2-SmoothRot's closed form
    mode=none: s = 1.  sign_seed: random +-1 per channel before the online Hadamard R4."""
    stat = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def h(mod, inp, i=i):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            a = x.abs().amax(0) if mode == "max" else (x * x).sum(0)
            stat[i] = (torch.maximum(stat[i], a) if mode == "max" else stat[i] + a) if i in stat else a
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(h))
    if mode != "none":
        for j in range(min(nseq, calib.shape[0])):
            model(calib[j : j + 1].cuda())
    for hk in hooks:
        hk.remove()
    g = torch.Generator(device="cpu")
    if sign_seed is not None:
        g.manual_seed(sign_seed)
    for i, layer in enumerate(model.model.layers):
        W = layer.mlp.down_proj.weight.data
        if mode == "max":
            wmax = W.abs().amax(0).float().clamp(min=1e-5)
            s = (stat[i].clamp(min=1e-5) ** alpha) / (wmax ** (1 - alpha))
        elif mode == "l2":
            s = (stat[i].sqrt().clamp(min=1e-5) / W.float().norm(dim=0).clamp(min=1e-5)) ** alpha
        else:
            s = torch.ones(W.shape[1], device=W.device)
        s = s / torch.exp(torch.log(s).mean())  # geometric-mean 1: keeps overall scale
        if sign_seed is not None:
            s = s * (torch.randint(0, 2, (W.shape[1],), generator=g) * 2 - 1).to(s)
        layer.mlp.up_proj.weight.data = (layer.mlp.up_proj.weight.data.float() / s[:, None]).to(W.dtype)
        layer.mlp.down_proj.weight.data = (W.float() * s[None, :]).to(W.dtype)


@torch.no_grad()
def qk_signs(model, seed):
    """q_h -> q_h * d, k_h -> k_h * d with d[j] = d[j + hd/2] (RoPE rotates pairs (j, j+hd/2)),
    so RoPE(q*d) = RoPE(q)*d and q.k is unchanged. Changes what the K Hadamard/quantizer sees."""
    hd = model.config.hidden_size // model.config.num_attention_heads
    g = torch.Generator(device="cpu").manual_seed(seed)
    half = (torch.randint(0, 2, (hd // 2,), generator=g) * 2 - 1).float()
    d = torch.cat([half, half])
    for layer in model.model.layers:
        for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj):
            W = lin.weight.data
            lin.weight.data = (W.float().view(-1, hd, W.shape[1]) * d.to(W.device)[None, :, None]).view_as(W).to(W.dtype)


def make_rotation_file(cfg, model, r1_seed, r2_seed, path):
    hd = model.config.hidden_size // model.config.num_attention_heads
    torch.manual_seed(r1_seed)
    R = {"R1": hadamard_utils.random_hadamard_matrix(model.config.hidden_size, "cuda").cpu()}
    torch.manual_seed(r2_seed)
    for i in range(model.config.num_hidden_layers):
        R[f"model.layers.{i}.self_attn.R2"] = hadamard_utils.random_hadamard_matrix(hd, "cuda").cpu()
    torch.save(R, path)
    return path


def spin_args(cfg, rot_path):
    return argparse.Namespace(
        rotate_mode="hadamard", optimized_rotation_path=rot_path, fp32_had=False,
        a_bits=cfg["a_bits"], a_groupsize=-1, a_asym=cfg["a_asym"], a_clip_ratio=cfg["a_clip"],
        v_bits=cfg["v_bits"], v_asym=cfg["v_asym"], v_clip_ratio=cfg["v_clip"],
        k_bits=cfg["k_bits"], k_groupsize=cfg["k_groupsize"], k_asym=cfg["k_asym"],
        k_clip_ratio=cfg["k_clip"], k_pre_rope=False, int8_down_proj=False,
        w_bits=cfg["w_bits"], w_asym=cfg["w_asym"], w_clip=cfg["w_clip"], w_groupsize=-1,
        nsamples=cfg["nsamples"], percdamp=cfg["percdamp"], act_order=cfg["act_order"],
        export_to_et=False)


def rotate_and_wrap(model, cfg, rot_path):
    """Norm fusion must already be done. Rotate, add wrappers, configure A/KV quantizers
    exactly as SpinQuant eval_utils/main.py:ptq_model does (same points, same groupings)."""
    args = spin_args(cfg, rot_path)
    rotation_utils.rotate_model(model, args)
    model.cuda()  # SpinQuant's rotate_* functions leave weights on CPU
    quant_utils.add_actquant(model)
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = False
    hd = model.config.hidden_size // model.config.num_attention_heads
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    for name in qlayers:
        bits, gs = args.a_bits, -1
        if "v_proj" in name and args.v_bits < 16:
            qlayers[name].out_quantizer.configure(bits=args.v_bits, groupsize=hd,
                                                  sym=not args.v_asym, clip_ratio=args.v_clip_ratio)
        if "o_proj" in name:
            gs = hd
        if "lm_head" in name:
            bits = 16
        qlayers[name].quantizer.configure(bits=bits, groupsize=gs, sym=not args.a_asym,
                                          clip_ratio=args.a_clip_ratio)
    if args.k_bits < 16:
        for layer in model.model.layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn, "apply_rotary_pos_emb", config=model.config,
                k_bits=args.k_bits, k_groupsize=args.k_groupsize,
                k_sym=not args.k_asym, k_clip_ratio=args.k_clip_ratio)
    return args


# ---------------------------------------------------------------- quantizer toggling

def _quantizers(layer):
    qs = []
    for m in layer.modules():
        if isinstance(m, quant_utils.ActQuantWrapper):
            qs += [m.quantizer, m.out_quantizer]
    w = getattr(layer.self_attn, "apply_rotary_pos_emb_qk_rotation_wrapper", None)
    if w is not None:
        qs.append(w.k_quantizer)
    return qs


class QuantOff:
    """Context: all activation/KV quantizers of a layer at 16 bits."""
    def __init__(self, layer):
        self.qs = _quantizers(layer)

    def __enter__(self):
        self.saved = [q.bits for q in self.qs]
        for q in self.qs:
            q.bits = 16

    def __exit__(self, *a):
        for q, b in zip(self.qs, self.saved):
            q.bits = b


# ---------------------------------------------------------------- weight quantization

SEQUENTIAL = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]


def _linear(layer, name):
    m = layer.get_submodule(name)
    return m.module if isinstance(m, quant_utils.ActQuantWrapper) else m


@torch.no_grad()
def calibrated_gptq(model, calib, cfg, args, log=print):
    """Layer-wise GPTQ with two streams: FP chain (original weights, no A/KV quant) and
    quantized chain (quantized weights so far, A/KV quant on)."""
    dev = "cuda"
    layers = model.model.layers
    n = calib.shape[0]
    # capture first-layer inputs
    inps, cache = [], {}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x, **kw):
            inps.append(x); cache.update(kw); raise ValueError

    layers[0] = Catcher(layers[0])
    for j in range(n):
        try:
            model(calib[j : j + 1].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].m
    kw = {k: cache[k] for k in ("attention_mask", "position_ids") if k in cache}
    if "position_embeddings" in cache:
        kw["position_embeddings"] = cache["position_embeddings"]
    x_fp = torch.cat(inps, 0)
    x_q = x_fp.clone()
    use_two = cfg["calib_actq"] or cfg["asym_alpha"] > 0

    def run(layer, X, quant_on):
        out = torch.empty_like(X)
        if quant_on:
            for j in range(n):
                out[j : j + 1] = layer(X[j : j + 1], **kw)[0]
        else:
            with QuantOff(layer):
                for j in range(n):
                    out[j : j + 1] = layer(X[j : j + 1], **kw)[0]
        return out

    for i, layer in enumerate(layers):
        t0 = time.time()
        fp_w = {nm: _linear(layer, nm).weight.data.clone() for g in SEQUENTIAL for nm in g}
        for group in SEQUENTIAL:
            lins = {nm: _linear(layer, nm) for nm in group}
            d = lins[group[0]].in_features
            H = torch.zeros(d, d, device=dev, dtype=torch.float32)
            C = torch.zeros(d, d, device=dev, dtype=torch.float32) if cfg["asym_alpha"] > 0 else None
            buf = {}
            first = lins[group[0]]

            def cap(key):
                def h(mod, inp):
                    buf[key] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                return h

            for j in range(n):
                if use_two:
                    # FP stream: FP weights, quantizers off
                    hk = first.register_forward_pre_hook(cap("fp"))
                    swap = {}
                    for g in SEQUENTIAL:
                        for nm in g:
                            m = _linear(layer, nm); swap[nm] = m.weight.data; m.weight.data = fp_w[nm]
                    with QuantOff(layer):
                        layer(x_fp[j : j + 1], **kw)
                    for nm, w in swap.items():
                        _linear(layer, nm).weight.data = w
                    hk.remove()
                    # quantized stream
                    hk = first.register_forward_pre_hook(cap("q"))
                    if cfg["calib_actq"]:
                        layer(x_q[j : j + 1], **kw)
                    else:
                        with QuantOff(layer):
                            layer(x_q[j : j + 1], **kw)
                    hk.remove()
                    xq = buf["q"]
                    H += xq.T @ xq
                    if C is not None:
                        C += buf["fp"].T @ xq
                else:
                    hk = first.register_forward_pre_hook(cap("q"))
                    with QuantOff(layer):
                        layer(x_q[j : j + 1], **kw)
                    hk.remove()
                    H += buf["q"].T @ buf["q"]
            ntok = n * x_q.shape[1]
            H /= ntok
            if C is not None:
                C /= ntok
                Hd = H.clone()
                Hd.diagonal().add_(cfg["ls_damp"] * Hd.diagonal().mean())
                L = torch.linalg.cholesky(Hd.double())
                Hinv = torch.cholesky_inverse(L)
                # min ||X W^T - Xq W*^T||^2 + lam ||W* - W||^2  ->  W* = W + W (C - H)(H + lam)^-1
                # (ridge toward W, not toward 0: W* = W exactly when Xq = X)
                M = ((C - H).double() @ Hinv)  # d x d ; W* = W + W M
            for nm in group:
                lin = lins[nm]
                W = fp_w[nm].float()
                if C is not None:
                    dW = (W.double() @ M).float()
                    log(f"  {nm}: |dW|/|W|={dW.norm() / W.norm():.4f} max|W*|/max|W|="
                        f"{(W + dW).abs().max() / W.abs().max():.3f}")
                    W = W + cfg["asym_alpha"] * dW
                lin.weight.data = W.to(lin.weight.dtype)
                if cfg["wq"] == "rtn":
                    q = quant_utils.WeightQuantizer()
                    q.configure(cfg["w_bits"], perchannel=True, sym=not cfg["w_asym"], mse=cfg["w_clip"])
                    q.find_params(W)
                    lin.weight.data = q.quantize(W).to(lin.weight.dtype)
                    continue
                g = gptq_utils.GPTQ(lin)
                g.H = (2 * H).clone()  # GPTQ convention H = 2/n X^T X
                g.quantizer = quant_utils.WeightQuantizer()
                g.quantizer.configure(cfg["w_bits"], perchannel=True, sym=not cfg["w_asym"], mse=cfg["w_clip"])
                g.fasterquant(percdamp=cfg["percdamp"], groupsize=-1, actorder=cfg["act_order"], static_groups=False)
                g.free()
            del H, C
        # advance both chains
        if use_two:
            qw = {nm: _linear(layer, nm).weight.data for g in SEQUENTIAL for nm in g}
            for nm in qw:
                _linear(layer, nm).weight.data = fp_w[nm]
            x_fp = run(layer, x_fp, quant_on=False)
            for nm, w in qw.items():
                _linear(layer, nm).weight.data = w
            x_q = run(layer, x_q, quant_on=cfg["calib_actq"])
        else:
            x_q = run(layer, x_q, quant_on=False)
        del fp_w
        torch.cuda.empty_cache()
        log(f"layer {i} done {time.time() - t0:.1f}s")


@torch.no_grad()
def rtn_all(model, cfg):
    for layer in model.model.layers:
        for g in SEQUENTIAL:
            for nm in g:
                lin = _linear(layer, nm)
                q = quant_utils.WeightQuantizer()
                q.configure(cfg["w_bits"], perchannel=True, sym=not cfg["w_asym"], mse=cfg["w_clip"])
                W = lin.weight.data.float()
                q.find_params(W)
                lin.weight.data = q.quantize(W).to(lin.weight.dtype)


# ---------------------------------------------------------------- build

def build(cfg, tok, log=print, workdir="/tmp/wsp"):
    os.makedirs(workdir, exist_ok=True)
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    calib = calib_data(cfg, tok)
    base = common.load_model().cuda()
    base.eval(); base.config.use_cache = False
    if cfg["smooth_down"] > 0 or cfg["r4_sign_seed"] is not None:
        mode = cfg["smooth_mode"] if cfg["smooth_down"] > 0 else "none"
        smooth_down_proj(base, calib, cfg["smooth_down"], mode=mode, sign_seed=cfg["r4_sign_seed"])
        log(f"down_proj transform mode={mode} a={cfg['smooth_down']} signs={cfg['r4_sign_seed']}")
    if cfg["r3_sign_seed"] is not None:
        qk_signs(base, cfg["r3_sign_seed"])
    fuse_norm_utils.fuse_layer_norms(base)
    info = {}
    r1 = cfg["seed"] if cfg["r1_seed"] is None else cfg["r1_seed"]
    r2 = cfg["seed"] if cfg["r2_seed"] is None else cfg["r2_seed"]
    if cfg["rot"] == "file":
        rot_path = cfg["rot_file"]
    elif cfg["r1_search"] > 0:
        px = proxy_data(tok, cfg["proxy_nseq"])
        cand = []
        pcfg = dict(cfg, w_bits=16)
        for k in range(cfg["r1_search"]):
            s1 = 1000 * cfg["seed"] + 17 + k
            m = copy.deepcopy(base).cpu() if False else copy.deepcopy(base)
            p = make_rotation_file(cfg, m, s1, s1, os.path.join(workdir, f"R_{s1}.bin"))
            rotate_and_wrap(m, pcfg, p)
            score = common.perplexity(m, px)
            cand.append((score, s1)); log(f"R1 candidate seed {s1}: proxy ppl {score:.4f}")
            del m; torch.cuda.empty_cache()
        cand.sort()
        info["r1_candidates"] = cand
        r1 = r2 = cand[0][1]
        rot_path = os.path.join(workdir, f"R_{r1}.bin")
    else:
        rot_path = make_rotation_file(cfg, base, r1, r2, os.path.join(workdir, f"R_{r1}_{r2}.bin"))
    info.update(r1_seed=r1, r2_seed=r2, rot_path=rot_path)
    args = rotate_and_wrap(base, cfg, rot_path)
    if cfg["w_bits"] < 16:
        if cfg["wq"] == "rtn" and not (cfg["calib_actq"] or cfg["asym_alpha"] > 0):
            rtn_all(base, cfg)
        else:
            calibrated_gptq(base, calib, cfg, args, log=log)
    return base, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="JSON string or file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppl", default="wikitext2,c4,fw25sel,fw25test")
    ap.add_argument("--proxy", action="store_true", help="also report the practice proxy ppl")
    ap.add_argument("--zs", action="store_true")
    ap.add_argument("--zs_limit", type=int, default=None)
    a = ap.parse_args()
    user = json.load(open(a.cfg)) if os.path.exists(a.cfg) else json.loads(a.cfg)
    unknown = set(user) - set(DEFAULT)
    assert not unknown, f"unknown config keys {unknown}"
    cfg = dict(DEFAULT, **user)
    t0 = time.time()
    tok = common.load_tokenizer()
    res = {"cfg": cfg, "ppl": {}}
    model, info = build(cfg, tok, workdir=os.path.join("/tmp/wsp", str(os.getpid())))
    res["info"] = info
    res["time_build_s"] = time.time() - t0
    common.write_json(a.out, res)
    if a.proxy:
        res["proxy_ppl"] = common.perplexity(model, proxy_data(tok, 32))
        print("PROXY", res["proxy_ppl"], flush=True)
    for name in [s for s in a.ppl.split(",") if s]:
        res["ppl"][name] = common.perplexity(model, common.ppl_ids(name, tok))
        print(f"PPL {name} = {res['ppl'][name]:.4f}", flush=True)
        common.write_json(a.out, res)
    if a.zs:
        res["zs"] = common.zeroshot(model, tok, limit=a.zs_limit)
        print("ZS", json.dumps(res["zs"]), flush=True)
    res["time_total_s"] = time.time() - t0
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    common.write_json(a.out, res)


if __name__ == "__main__":
    main()
