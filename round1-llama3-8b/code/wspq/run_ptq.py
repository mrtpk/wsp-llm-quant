"""Baseline runner: SpinQuant ptq_model() + our evaluation.

Usage (inside container, 1 GPU):
  python -m wspq.run_ptq --out /results/x.json [--zs] [--ppl wikitext2,c4,...] -- <SpinQuant ptq args>
Everything after "--" goes unchanged to SpinQuant's arg parser (process_args_ptq).
"""
import argparse, sys, time, json
import torch
from wspq import common


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ours = argparse.ArgumentParser()
    ours.add_argument("--out", required=True)
    ours.add_argument("--ppl", default="wikitext2,c4,fw25sel,fw25test")
    ours.add_argument("--zs", action="store_true")
    ours.add_argument("--zs_limit", type=int, default=None)
    ours.add_argument("--zs_bs", type=int, default=32)
    ours.add_argument("--fp", action="store_true", help="no quantization at all")
    a = ours.parse_args(argv[:split])
    spin_argv = argv[split + 1 :]
    sys.argv = [sys.argv[0], "--input_model", common.MODEL, "--model_max_length", "2048",
                "--bf16", "True", "--fp16", "False", "--output_dir", "/tmp/out"] + spin_argv
    from utils.process_args import process_args_ptq
    from eval_utils.main import ptq_model
    model_args, training_args, ptq_args = process_args_ptq()

    t0 = time.time()
    tok = common.load_tokenizer()
    model = common.load_model()
    model.cuda()
    if not a.fp:
        model = ptq_model(ptq_args, model, model_args)
    model.cuda()
    model.config.use_cache = False
    t_ptq = time.time() - t0
    res = {"argv": spin_argv, "ptq_args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                          for k, v in vars(ptq_args).items()},
           "fp": a.fp, "time_ptq_s": t_ptq, "ppl": {}}
    for name in [s for s in a.ppl.split(",") if s]:
        ids = common.ppl_ids(name, tok)
        res["ppl"][name] = common.perplexity(model, ids)
        print(f"PPL {name} = {res['ppl'][name]:.4f}", flush=True)
        common.write_json(a.out, res)
    if a.zs:
        res["zs"] = common.zeroshot(model, tok, bs=a.zs_bs, limit=a.zs_limit)
        print("ZS", json.dumps(res["zs"]), flush=True)
    res["time_total_s"] = time.time() - t0
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    common.write_json(a.out, res)


if __name__ == "__main__":
    main()
