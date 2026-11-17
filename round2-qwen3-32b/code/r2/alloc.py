"""Round 3 phase A (2026-11-15): automatic bit-allocation CONTROL at equal bits (plan: ../round3-phase-a/PLAN.md).

  python -m r2.alloc sens                 per-linear sensitivity -> /results/p3a/sens.json (resumable per layer)
  python -m r2.alloc alloc                greedy allocation for each target -> /results/p3a/alloc_<target>.json
  python -m r2.alloc run TARGET SEED      P4-style test evaluation of the frozen control -> /results/p3a/<ctl>/s<SEED>.json

All knobs are fixed here and in the plan BEFORE any control result exists:
  menu      W3g128 asym, W3g64 asym, W4g128 asym, INT8 per-channel sym (the formats of the agent's M finals)
  sens      quantize ONE linear with the round-2 GPTQ solver (act_order, damp 0.01, Hessian from the round-2 calibration
            set: 128 x 2048, mix, seed 0; all other linears bf16), KL(bf16 || quantized) of the final next-token
            distribution, mean over all tokens of 8 calibration windows (mix, seed 4242), whole step in fp32
  alloc     start at W3g128 everywhere; per linear the lower convex hull of (bits, KL); greedy: largest KL drop per
            added stored bit, skip upgrades that do not fit, until none fits; budget = the agent final's W (round-2
            cost function, int8 lm_head counted)
  pipeline  identical to the agent's M finals: one GPTQ op, base W3g128, per-linear exceptions, asym_alpha 0.15
"""
import hashlib, json, os, sys, time
import torch
from r2 import common, fq, p2
from r2.exps import I, _build, _gptq
from r2.fq import TYPES, lin_name, lin_regex

OUT = f"{common.RESULTS}/p3a"
MENU = {"W3g128": I(3, 128), "W3g64": I(3, 64), "W4g128": I(4, 128), "W8pc": I(8, sym=True)}
BASE = "W3g128"
TARGETS = {"mix_asym_W3_late34": "primary", "mix_asym_W3g64_late44": "secondary"}
N_KL, KL_SEED, CAL_SEED = 8, 4242, 0   # 8 (not 16) windows: fp32 tail costs more (fix 2026-11-15, see NOTES)


def _lin(layer, t):
    return getattr(layer.self_attn if t in TYPES[:4] else layer.mlp, t)


@torch.no_grad()
def tail_logprobs_kl(model, i, Xk, kw, ref):
    """Run layers i..end on Xk (KL windows), return mean KL(ref || q) over all tokens."""
    from r2.solvers import run_layer
    H = Xk
    for j in range(i, len(model.model.layers)):
        H = run_layer(model.model.layers[j], H, kw)
    tot, n = 0.0, 0
    for w in range(H.shape[0]):
        lq = torch.log_softmax(model.lm_head(model.model.norm(H[w])).float(), -1)
        lr = ref[w]
        tot += (lr.exp() * (lr - lq)).sum().item(); n += lq.shape[0]
    return tot / n


@torch.no_grad()
def sens():
    from r2.solvers import layer0_inputs, run_layer, gptq_solve, SEQ
    # FIX 2026-11-15 (before any control result): the first run in bf16 had a KL floor of ~1e-3 for ANY change of one
    # linear (int8 RTN of L0.q gave the same KL as W3): bf16 rounding noise carried through 64 layers. The whole
    # sensitivity step now runs in fp32 (W8 of L0.q -> 7.9e-7, W3 -> 1.3e-4). Quantized weights are rounded to bf16,
    # as stored by the pipeline. The bf16 result file (sens.json, 2 layers) is kept as a record and not used.
    path = f"{OUT}/sens32.json"
    res = common.read_json(path, {}) or {}
    tok = common.load_tokenizer(); model = common.load_model(dtype=torch.float32); fq.install_attention()
    calib = common.calib_data(tok, 128, seed=CAL_SEED)
    klw = common.calib_data(tok, N_KL, seed=KL_SEED)
    X, kw = layer0_inputs(model, calib)
    Xk, kwk = layer0_inputs(model, klw)
    # reference: log-probs of the bf16 model on the KL windows, stored in fp32 (~20 GB; bf16 storage would add a
    # KL noise floor of the order of the small KLs we must rank)
    H = _final_hidden(model, Xk, kwk)
    ref = torch.stack([torch.log_softmax(model.lm_head(model.model.norm(H[w])).float(), -1) for w in range(H.shape[0])])
    base_kl = tail_logprobs_kl(model, len(model.model.layers), H, kwk, ref)    # self-check: must be 0
    del H
    res["_meta"] = dict(menu=MENU, n_kl=N_KL, kl_seed=KL_SEED, cal_seed=CAL_SEED, kl_bf16_self=base_kl)
    p2.log(f"bf16 self-KL (must be ~0): {base_kl:.2e}")
    for i, layer in enumerate(model.model.layers):
        t0 = time.time()
        if all(lin_name(i, t) in res for t in TYPES):
            X = run_layer(layer, X, kw); Xk = run_layer(layer, Xk, kwk); continue
        # Hessians of the 4 input groups from bf16 inputs (same as solvers.gptq with all upstream exact)
        Hs = {}
        for grp in SEQ:
            first = _lin(layer, grp[0]); d = first.in_features
            Hg = torch.zeros(d, d, device=X.device); buf = {}
            hk = first.register_forward_pre_hook(lambda m, a: buf.__setitem__("x", a[0].reshape(-1, a[0].shape[-1]).float()))
            for j in range(0, X.shape[0], 8):
                layer(X[j: j + 8], **kw); Hg += buf["x"].T @ buf["x"]
            hk.remove()
            Hg /= X.shape[0] * X.shape[1]
            for t in grp:
                Hs[t] = Hg
        for t in TYPES:
            n = lin_name(i, t)
            if n in res:
                continue
            lin = _lin(layer, t); W0 = lin.weight.data.clone()
            r = dict(numel=W0.numel(), in_features=lin.in_features)
            for fk, spec in MENU.items():
                lin.weight.data = gptq_solve(W0.float(), 2 * Hs[t], spec, True, 0.01).to(torch.bfloat16).to(W0.dtype)
                r[fk] = tail_logprobs_kl(model, i, Xk, kwk, ref)
                lin.weight.data = W0
            res[n] = r
        del Hs; torch.cuda.empty_cache()
        X = run_layer(layer, X, kw); Xk = run_layer(layer, Xk, kwk)
        common.write_json(path, res)
        p2.log(f"sens layer {i} {time.time() - t0:.0f}s  " + " ".join(
            f"{t[:4]}:" + "/".join(f"{res[lin_name(i, t)][f]:.1e}" for f in MENU) for t in TYPES))
    p2.log("sens done")


@torch.no_grad()
def _final_hidden(model, Xk, kw):
    from r2.solvers import run_layer
    H = Xk
    for layer in model.model.layers:
        H = run_layer(layer, H, kw)
    return H


def wbits_of(assign, sens_d, lm_numel, lm_in):
    """Round-2 cost function (p2.cost): stored bits incl. scales/zeros, int8 lm_head counted."""
    tot = bits = 0.0
    for n, f in assign.items():
        k = sens_d[n]["numel"]; tot += k; bits += k * fq.weight_bits(MENU[f], sens_d[n]["in_features"])
    tot += lm_numel; bits += lm_numel * fq.weight_bits(p2.LM_HEAD, lm_in)
    return bits / tot


def hull(points):
    """points [(bits, kl, fmt)] sorted by bits -> lower convex hull starting at the cheapest point (KL decreasing)."""
    pts = sorted(points)
    h = [pts[0]]
    while True:
        cur = h[-1]; best = None
        for p in pts:
            if p[0] <= cur[0] or p[1] >= cur[1]:
                continue
            slope = (cur[1] - p[1]) / (p[0] - cur[0])
            if best is None or slope > best[0] + 1e-30:
                best = (slope, p)
        if best is None:
            return h
        h.append(best[1])


def alloc():
    sd = json.load(open(f"{OUT}/sens32.json"))
    sd = {k: v for k, v in sd.items() if not k.startswith("_")}
    assert len(sd) == 64 * 7, len(sd)
    summ = json.load(open(f"{common.RESULTS}/p4_summary/summary.json"))
    lm_numel, lm_in = 151936 * 5120, 5120
    for tgt in TARGETS:
        budget = summ[tgt]["W"]
        assign = {n: BASE for n in sd}
        hulls = {n: hull([(fq.weight_bits(MENU[f], v["in_features"]), v[f], f) for f in MENU]) for n, v in sd.items()}
        pos = {n: 0 for n in sd}
        if any(h[0][2] != BASE for h in hulls.values()):
            raise SystemExit("a hull does not start at the base format (base is not the cheapest) -> stop")
        tot_w = sum(v["numel"] for v in sd.values()) + lm_numel
        cur = wbits_of(assign, sd, lm_numel, lm_in)
        steps = 0
        while True:
            cand = []
            for n, h in hulls.items():
                if pos[n] + 1 < len(h):
                    a, b = h[pos[n]], h[pos[n] + 1]
                    dbits = (b[0] - a[0]) * sd[n]["numel"]
                    cand.append(((a[1] - b[1]) / dbits, n, dbits / tot_w, b[2]))
            if not cand:
                break
            cand.sort(reverse=True)
            took = False
            for ratio, n, dw, f in cand:
                if cur + dw <= budget + 1e-12:
                    assign[n] = f; pos[n] += 1; cur += dw; steps += 1; took = True
                    break
                hulls[n] = hulls[n][: pos[n] + 1]          # does not fit -> no more upgrades for this linear
            if not took:
                break
        W = wbits_of(assign, sd, lm_numel, lm_in)
        ok = abs(W - budget) <= 0.01
        counts = {}
        for n, f in assign.items():
            t = n.rsplit(".", 1)[-1]; counts.setdefault(t, {}).setdefault(f, 0); counts[t][f] += 1
        by_layer = {i: [assign[lin_name(i, t)] for t in TYPES] for i in range(64)}
        out = dict(target=tgt, role=TARGETS[tgt], budget_W=budget, W=W, within_001=ok, steps=steps,
                   predicted_sum_kl=sum(sd[n][f] for n, f in assign.items()), assign=assign, counts_by_type=counts,
                   by_layer=by_layer, menu=MENU)
        path = f"{OUT}/alloc_{tgt}.json"
        common.write_json(path, out)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        p2.log(f"alloc {tgt}: budget {budget:.4f} W {W:.4f} within 0.01: {ok}  steps {steps}  sha256 {h}")
        p2.log("  counts by type: " + json.dumps(counts))
        if not ok:
            raise SystemExit("cannot reach the target within 0.01 -> stop and report")


def control_cfg(tgt):
    a = json.load(open(f"{OUT}/alloc_{tgt}.json"))
    exc = [(lin_regex(layers=[int(n.split(".")[2])], types=[n.rsplit(".", 1)[-1]]), MENU[f])
           for n, f in a["assign"].items() if f != BASE]
    return _build([_gptq(MENU[BASE], exc, asym_alpha=0.15)], MENU[BASE], exc), a


def run(tgt, seed):
    from r2 import solvers
    cfg, a = control_cfg(tgt)
    key = f"ctl_{tgt}"
    path = f"{OUT}/{key}/s{seed}.json"
    if os.path.exists(path):
        p2.log("exists", path); return
    for op in cfg["pipeline"]:
        op["seed"] = seed
    tok = common.load_tokenizer(); model = common.load_model(); fq.install_attention()
    c = p2.cost(model, cfg["cost_rules"], 0)
    p2.log(f"{key}: W by p2.cost {c['wbits']:.4f} (alloc said {a['W']:.4f}, budget {a['budget_W']:.4f})")
    if abs(c["wbits"] - a["budget_W"]) > 0.01:
        raise SystemExit("p2.cost disagrees with the budget by > 0.01 -> stop")
    t = time.time()
    info = solvers.run_pipeline(model, tok, cfg, log=p2.log)
    torch.cuda.empty_cache()
    p2.apply_lm_head(fq.FQ(model).apply(cfg.get("rules", [])))
    r = {s: common.perplexity(model, common.ppl_ids(s, tok)) for s in ["wikitext2", "fw25test", "c4dev"]}
    p2.log(f"{key} s{seed} wt2 {r['wikitext2']:.4f} fw25 {r['fw25test']:.4f} c4dev {r['c4dev']:.4f}")
    zs = common.zeroshot(model, tok)
    r["zs_avg"] = zs["avg"]; r["zs"] = zs
    r.update(c, info=info, seed=seed, sec=round(time.time() - t, 1), alloc_sha256=hashlib.sha256(
        open(f"{OUT}/alloc_{tgt}.json", "rb").read()).hexdigest(), peak_mem_gb=common.peak_gb())
    common.write_json(path, r)
    p2.log(f"{key} s{seed} zs {zs['avg']:.2f} done")


if __name__ == "__main__":
    cmd = sys.argv[1]
    p2._keep_traceback("p3a_" + "_".join(sys.argv[1:4]))
    if cmd == "sens":
        sens()
    elif cmd == "alloc":
        alloc()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]))
    print("DONE", flush=True)
