"""P4 confirmation runner: one (config, seed) per process.

  python -m r2.p4 KEY SEED

KEY must be in r2/finals.py FROZEN (finals + baselines, fixed before any test data is read).
The seed is the calibration draw: it is set on every op of the pipeline (GPTQ/AWQ/SQ calibration, rotation).
Evaluates the TEST sets (WikiText-2 test, fw25test, 8 zero-shot tasks) plus c4dev for reference.
Result: /results/p4/<KEY>/s<SEED>.json
"""
import copy, os, sys, time
import torch
from r2 import common, fq, p2

OUT = f"{common.RESULTS}/p4"


def main(key, seed):
    from r2 import finals, solvers
    if key not in finals.FROZEN:
        raise SystemExit(f"{key} is not in finals.FROZEN; P4 runs only frozen configs")
    cfg = copy.deepcopy(finals.FROZEN[key])
    path = f"{OUT}/{key}/s{seed}.json"
    if os.path.exists(path):
        p2.log("exists", path); return
    for op in cfg.get("pipeline", []):
        op["seed"] = seed
    tok = common.load_tokenizer()
    model = common.load_model()
    fq.install_attention()
    t = time.time()
    info = solvers.run_pipeline(model, tok, cfg, log=p2.log)
    torch.cuda.empty_cache()
    rules = cfg.get("rules", [])
    q = fq.FQ(model).apply(rules)
    if cfg.get("lm_head", "int8") == "int8":
        p2.apply_lm_head(q)
    r = {s: common.perplexity(model, common.ppl_ids(s, tok)) for s in ["wikitext2", "fw25test", "c4dev"]}
    p2.log(f"{key} s{seed} wt2 {r['wikitext2']:.4f} fw25 {r['fw25test']:.4f} c4dev {r['c4dev']:.4f}")
    zs = common.zeroshot(model, tok)
    r["zs_avg"] = zs["avg"]; r["zs"] = zs
    cost_rules = cfg.get("cost_rules", rules)
    r.update(p2.cost(model, cost_rules, info.get("extra_params", 0)) if cfg.get("lm_head", "int8") == "int8" else {},
             info=info, seed=seed, sec=round(time.time() - t, 1), cfg=cfg, peak_mem_gb=common.peak_gb())
    common.write_json(path, r)
    p2.log(f"{key} s{seed} zs {zs['avg']:.2f} done")


if __name__ == "__main__":
    p2._keep_traceback("p4_" + "_".join(sys.argv[1:3]))
    main(sys.argv[1], int(sys.argv[2]))
    print("DONE", flush=True)
