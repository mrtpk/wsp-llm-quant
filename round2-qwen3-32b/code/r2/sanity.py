"""P0 sanity checks.
  --part ppl : bf16 perplexity on all sets (+ dev sets), timing, peak memory; simulated W16A16KV16 == bf16
               exactly; attention wrapper is really called; RTN W8A8 close to bf16.
  --part zs  : bf16 zero-shot (8 test tasks, reference only) and the task-dev proxy.
"""
import argparse, time
import torch
from r2 import common, fq

ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True)
a = ap.parse_args()
out = f"{common.RESULTS}/p0/sanity_{a.part}.json"
res = common.read_json(out, {})
tok = common.load_tokenizer()

if a.part == "data":  # CPU only: build and cache every dataset
    for s in ["wikitext2", "c4", "fw25sel", "fw25test", "wt2train"]:
        print(s, tuple(common.ppl_ids(s, tok).shape), flush=True)
    print("calib", tuple(common.calib_data(tok, 64).shape), tuple(common.calib_data(tok, 128).shape))
    raise SystemExit

model = common.load_model()
res["device"] = torch.cuda.get_device_name(0)
res["arch"] = torch.cuda.get_device_properties(0).gcnArchName
print(res["device"], res["arch"], flush=True)

if a.part == "ppl":
    sets = ["c4dev", "wt2train", "c4", "fw25sel", "wikitext2", "fw25test"]
    bf = res.setdefault("bf16", {})
    for s in sets:
        ids = common.ppl_ids(s, tok)
        torch.cuda.synchronize(); t = time.time()
        bf[s] = common.perplexity(model, ids)
        torch.cuda.synchronize()
        bf[s + "/sec_per_window"] = (time.time() - t) / ids.shape[0]
        print("bf16", s, bf[s], f"{bf[s + '/sec_per_window']:.3f}s/win", flush=True)
        res["peak_mem_gb_bf16"] = common.peak_gb(); common.write_json(out, res)
    c4dev = common.ppl_ids("c4dev", tok)
    # exactness: identity specs through every code path (weights, activation hooks, KV wrapper)
    calls = {"n": 0}
    for layer in model.model.layers:
        layer.self_attn.fq_probe = lambda *x: calls.__setitem__("n", calls["n"] + 1)
    nll0, _ = common.nll_sum(model, c4dev)
    I = {"bits": 16}
    q = fq.FQ(model).apply([{"match": fq.lin_regex(), "w": I, "a": I}, {"kv": r"\d+", "k": I, "v": I}])
    nll1, _ = common.nll_sum(model, c4dev)
    q.reset()
    for layer in model.model.layers:
        layer.self_attn.fq_probe = None
    res["w16a16_exact"] = dict(nll_bf16=nll0, nll_sim=nll1, equal=nll0 == nll1, attn_wrapper_calls=calls["n"])
    print("W16A16 exact:", res["w16a16_exact"], flush=True)
    assert calls["n"] > 0, "attention wrapper not called"
    # RTN W8A8 (per-channel sym weights, per-token sym activations), also KV8
    W8 = {"bits": 8, "sym": True}; A8 = {"bits": 8, "sym": True}
    for nm, rules in [("w8a8", [{"match": fq.lin_regex(), "w": W8, "a": A8}]),
                      ("w8a8kv8", [{"match": fq.lin_regex(), "w": W8, "a": A8},
                                   {"kv": r"\d+", "k": {"bits": 8}, "v": {"bits": 8}}]),
                      ("w8a8_fp8", [{"match": fq.lin_regex(), "w": {"bits": 8, "fmt": "fp"},
                                     "a": {"bits": 8, "fmt": "fp"}}])]:
        t = time.time(); q = fq.FQ(model).apply(rules); tq = time.time() - t
        res[nm] = {s: common.perplexity(model, common.ppl_ids(s, tok)) for s in ["c4dev", "wt2train"]}
        res[nm]["apply_sec"] = tq
        q.reset()
        print(nm, res[nm], flush=True)
        common.write_json(out, res)
    # reset really restores
    res["after_reset_c4dev"] = common.perplexity(model, c4dev)
    print("after reset", res["after_reset_c4dev"], flush=True)

elif a.part == "zs":
    t = time.time()
    res["taskdev_bf16"] = common.taskdev(model, tok); res["taskdev_sec"] = time.time() - t
    print("taskdev", res["taskdev_bf16"], flush=True); common.write_json(out, res)
    t = time.time()
    res["zs_bf16"] = common.zeroshot(model, tok); res["zs_sec"] = time.time() - t
    print("zeroshot", res["zs_bf16"], flush=True)

res["peak_mem_gb"] = common.peak_gb()
common.write_json(out, res)
print("DONE", flush=True)
