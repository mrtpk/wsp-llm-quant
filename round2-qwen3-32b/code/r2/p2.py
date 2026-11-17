"""P2 exploration runner.

  python -m r2.p2 sweep NAME        many RTN configs in one process (model loaded once); experiments in r2/exps.py
  python -m r2.p2 build KEY         one config with transforms/solvers (GPTQ, AWQ, smoothing, rotation, low-rank);
                                    the model is modified in place, so one config per process

Every result: c4dev + wt2train ppl (dev), optional task-dev proxy, and cost accounting:
  wbits   average stored bits per linear weight incl. group scales/zeros and bf16 exceptions (lm_head excluded)
  extra_params  extra bf16 parameters (low-rank factors), also folded into wbits_eff
  abits   activation spec summary, kvbits average KV bits per element incl. scales
Results: /results/p2/<name>.json (sweep) or /results/p2/build/<key>.json (build).
"""
import argparse, os, re, sys, time
import torch
from r2 import common, fq

OUT = f"{common.RESULTS}/p2"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def eff_specs(model, rules):
    """Per linear: effective (w, a) spec after applying rules in order (same semantics as fq.FQ.apply)."""
    w, a = {}, {}
    names = [n for n, m in model.named_modules() if n.startswith("model.layers.") and n.rsplit(".", 1)[-1] in fq.TYPES]
    for r in rules:
        if "match" in r:
            rx = re.compile(r["match"])
            for n in names:
                if rx.search(n):
                    if "w" in r:
                        w[n] = r["w"]
                    if "a" in r:
                        a[n] = r["a"]
    return names, w, a


def act_bits(spec, D):
    if spec is None or spec["bits"] >= 16:
        return 16.0
    g = spec.get("group", 0) or D
    fmt = spec.get("fmt", "int")
    if fmt in ("mxint", "mxfp"):
        return spec["bits"] + 8 / g
    return spec["bits"] + (16 + (0 if spec.get("sym", False) or fmt == "fp" else spec["bits"])) / g


LM_HEAD = {"bits": 8, "sym": True}   # user decision 2026-11-11: lm_head at 8 bits in every config, counted
LM_HEAD_A = None                     # lm_head input stays bf16 (counted as 16 bits in BitOps)


def apply_lm_head(q):
    q.set_weight("lm_head", LM_HEAD)
    return q


def cost(model, rules, extra_params=0):
    """wbits: avg stored bits/weight over all linears INCLUDING lm_head (8-bit). bitops: FLOP-weighted
    W bits x A bits per MAC (bf16 = 256). wbits_body: decoder linears only (for reference)."""
    names, w, a = eff_specs(model, rules)
    mods = dict(model.named_modules())
    tot = bits = abit_w = bitops = 0.0
    for n in names + ["lm_head"]:
        lin = mods[n]
        k = lin.weight.numel()
        ws = LM_HEAD if n == "lm_head" else w.get(n)
        as_ = LM_HEAD_A if n == "lm_head" else a.get(n)
        wb, ab = fq.weight_bits(ws, lin.in_features), act_bits(as_, lin.in_features)
        tot += k; bits += k * wb; abit_w += k * ab; bitops += k * wb * ab
        if n == names[-1]:
            body_tot, body_bits = tot, bits
    kv = []
    for r in rules:
        if "kv" in r:
            kv.append(r)
    nl = len(model.model.layers)
    kvb = []
    for i in range(nl):
        ks = vs = None
        for r in kv:
            if re.fullmatch(r["kv"], str(i)):
                ks, vs = r.get("k", ks), r.get("v", vs)
        def b(s):
            if s is None or s["bits"] >= 16:
                return 16.0
            g = s.get("group", 0) or (128 if s.get("axis", "token") == "token" else 2048)
            over = 16 + (0 if s.get("sym", False) or s.get("fmt", "int") == "fp" else s["bits"])
            return s["bits"] + over / g
        kvb.append((b(ks) + b(vs)) / 2)
    return dict(wbits=(bits + 16 * extra_params) / tot, wbits_noextra=bits / tot, wbits_body=body_bits / body_tot,
                extra_params=extra_params, abits_flopw=abit_w / tot, bitops=bitops / tot, kvbits=sum(kvb) / nl,
                lm_head="int8pc")


def evaluate(model, tok, taskdev=False):
    # selection uses c4dev + task-dev ONLY (user decision); wt2train is logged as a diagnostic, never used to select
    r = {s: common.perplexity(model, common.ppl_ids(s, tok)) for s in ["c4dev", "wt2train"]}
    if taskdev:
        td = common.taskdev(model, tok)
        r["taskdev"] = td["avg"]
        r["taskdev_detail"] = td
    return r


def sweep(name):
    from r2 import exps
    path = f"{OUT}/{name}.json"
    res = common.read_json(path, {})
    tok = common.load_tokenizer()
    model = common.load_model()
    fq.install_attention()
    for key, cfg in exps.SWEEPS[name]():
        if key in res:
            continue
        t = time.time()
        rules = cfg["rules"]
        q = apply_lm_head(fq.FQ(model).apply(rules))
        r = evaluate(model, tok, cfg.get("taskdev", False))
        q.reset()
        r.update(cost(model, rules), sec=round(time.time() - t, 1))
        res[key] = r
        common.write_json(path, res)
        log(f"{key:55s} c4dev {r['c4dev']:.4f} wt2 {r['wt2train']:.4f}" + (f" td {r['taskdev']:.2f}" if "taskdev" in r else "")
            + f"  W{r['wbits']:.3f} B{r['bitops']:.1f} KV{r['kvbits']:.2f} ({r['sec']}s)")


def build(key):
    """KEY or KEY@sN (N = calibration seed set on every pipeline op; post-freeze exploration, 2026-11-14)."""
    import copy
    from r2 import exps, solvers
    base, _, sd = key.partition("@s")
    cfg = copy.deepcopy(exps.BUILDS[base])
    if sd:
        for op in cfg.get("pipeline", []):
            op["seed"] = int(sd)
    path = f"{OUT}/build/{key}.json"
    if os.path.exists(path):
        log("exists", path); return
    tok = common.load_tokenizer()
    model = common.load_model()
    fq.install_attention()
    t = time.time()
    info = solvers.run_pipeline(model, tok, cfg, log=log)
    torch.cuda.empty_cache()   # release solver buffers before evaluation (two jobs share the GPU)
    log(f"after pipeline: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GB, peak {common.peak_gb():.1f} GB")
    rules = cfg.get("rules", [])
    apply_lm_head(fq.FQ(model).apply(rules))
    r = evaluate(model, tok, cfg.get("taskdev", True))
    r.update(cost(model, cfg.get("cost_rules", rules), info.get("extra_params", 0)), info=info,
             sec=round(time.time() - t, 1), cfg=cfg, peak_mem_gb=common.peak_gb())
    common.write_json(path, r)
    log(f"{key} c4dev {r['c4dev']:.4f} wt2 {r['wt2train']:.4f} td {r.get('taskdev', 0):.2f} W{r['wbits']:.3f}")


def rotdiag(name):
    """E5 diagnosis (2026-11-12): where does per-token A4 still break after R1+R4 (+ q/k Hadamard)? bf16 weights, c4dev only."""
    from r2 import exps, solvers
    path = f"{OUT}/{name}.json"
    res = common.read_json(path, {})
    tok = common.load_tokenizer()
    model = common.load_model()
    fq.install_attention()
    solvers.run_pipeline(model, tok, {"pipeline": [dict(op="rotate", r1=True, r4=True), dict(op="qkrot")]}, log=log)
    ids = common.ppl_ids("c4dev", tok)
    cfgs = exps.rotdiag2_cfgs() if name == "rotdiag2" else exps.rotdiag_cfgs()
    for key, rules in cfgs:
        if key in res:
            continue
        qk = not key.endswith("_noqkrot")
        for L in model.model.layers:
            L.self_attn.fq_rot_qk = qk
        q = apply_lm_head(fq.FQ(model).apply(rules))
        res[key] = {"c4dev": common.perplexity(model, ids)}
        q.reset()
        common.write_json(path, res)
        log(f"{key:40s} c4dev {res[key]['c4dev']:.4f}")


def _keep_traceback(name):
    """queue.sh overwrites the job log on every retry; keep each traceback in its own file."""
    import traceback
    def hook(et, ev, tb):
        d = f"{OUT}/errors"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/{name}.{int(time.time())}.txt", "w") as f:
            f.write("".join(traceback.format_exception(et, ev, tb)))
            if torch.cuda.is_available():
                f.write(f"\nallocated {torch.cuda.memory_allocated() / 2**30:.1f} GB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB\n")
        sys.__excepthook__(et, ev, tb)
    sys.excepthook = hook


if __name__ == "__main__":
    _keep_traceback("_".join(sys.argv[1:3]))
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("name")
    a = ap.parse_args()
    {"sweep": sweep, "build": build, "rotdiag": rotdiag}[a.mode](a.name)
    print("DONE", flush=True)
