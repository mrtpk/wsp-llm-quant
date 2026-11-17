"""Round-2 shared loading, data and evaluation for Qwen/Qwen3-32B (plain transformers 5.22).

Data split policy (PLAN.md section 5):
  dev:       c4 (validation windows; c4dev = first 64), wikitext2-train proxy, task-dev proxy
  selection: fw25sel
  test:      wikitext2 (test), fw25test, zero-shot TASKS  -- final runs (and P0 bf16 reference) only
"""
import json, os, random, time
import torch

MODEL = os.environ.get("WSP_MODEL", "Qwen/Qwen3-32B")
REVISION = os.environ.get("WSP_REVISION", "9216db5781bf21249d130ec9da846c4624c16137")
DATA = os.environ.get("WSP_DATA", "/models/tmp/wsp2/data")
RESULTS = os.environ.get("WSP_RESULTS", "/results")
TASKS = ["boolq", "piqa", "social_iqa", "hellaswag", "winogrande",
         "arc_easy", "arc_challenge", "openbookqa"]
DEV_TASKS = ["boolq_dev", "arc_easy_dev", "arc_challenge_dev", "piqa_dev", "winogrande_dev"]
TASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
SEQLEN = 2048


def load_model(dtype=torch.bfloat16, device="cuda:0"):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype, device_map=device,
                                             attn_implementation="sdpa")
    m.eval()
    m.config.use_cache = False
    print(f"loaded {type(m).__name__} in {time.time() - t0:.0f}s", flush=True)
    return m


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def doc_sep(tok):
    """Pretraining document separator for Qwen (<|endoftext|>), not the chat eos <|im_end|>."""
    return tok.convert_tokens_to_ids("<|endoftext|>")


# ---------------------------------------------------------------- perplexity data

def _cache(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, f"{name}.pt")


def ppl_ids(name, tok, seqlen=SEQLEN):
    """LongTensor (n, seqlen). name in wikitext2 | c4 | c4dev | fw25sel | fw25test | wt2train."""
    if name == "c4dev":
        return ppl_ids("c4", tok, seqlen)[:64]
    if name == "wt2train":
        return proxy_data(tok, 32, seqlen)
    path = _cache(f"ppl_{name}_{seqlen}")
    if os.path.exists(path):
        return torch.load(path)
    import datasets
    if name == "wikitext2":
        d = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
        ids = tok("\n\n".join(d["text"]), return_tensors="pt", add_special_tokens=False).input_ids
        n = ids.shape[1] // seqlen
        out = ids[:, : n * seqlen].view(n, seqlen)
    elif name == "c4":
        d = datasets.load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                                  split="validation")
        random.seed(0)
        rows = []
        while len(rows) < 256:
            i = random.randint(0, len(d) - 1)
            enc = tok(d[i]["text"], return_tensors="pt", add_special_tokens=False).input_ids
            if enc.shape[1] <= seqlen:
                continue
            j = random.randint(0, enc.shape[1] - seqlen - 1)
            rows.append(enc[:, j: j + seqlen])
        out = torch.cat(rows, 0)
    elif name.startswith("fw25"):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("HuggingFaceFW/fineweb", repo_type="dataset",
                            filename="data/CC-MAIN-2025-26/000_00000.parquet")
        texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        half = len(texts) // 2
        texts = texts[:half] if name == "fw25sel" else texts[half:]
        target = 160 if name == "fw25sel" else 320
        sep = doc_sep(tok)
        buf, rows = [], []
        for t in texts:
            buf.extend(tok(t, add_special_tokens=False).input_ids + [sep])
            while len(buf) >= seqlen and len(rows) < target:
                rows.append(torch.tensor(buf[:seqlen]).unsqueeze(0)); buf = buf[seqlen:]
            if len(rows) >= target:
                break
        out = torch.cat(rows, 0)
    else:
        raise ValueError(name)
    torch.save(out, path)
    return out


def _wt2_train_ids(tok):
    path = _cache("wikitext2_train_ids")
    if os.path.exists(path):
        return torch.load(path)
    import datasets
    d = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
    ids = tok("\n\n".join(d["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    torch.save(ids, path)
    return ids


def _wt2_rows(tok, n, L, seed):
    ids = _wt2_train_ids(tok)
    rng = random.Random(seed)
    return [ids[(i := rng.randint(0, ids.numel() - L - 1)): i + L] for _ in range(n)]


def _c4_train_rows(tok, n, L, seed):
    path = _cache(f"c4train_{n}_{L}_{seed}")
    if os.path.exists(path):
        return list(torch.load(path))
    import datasets
    d = datasets.load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
    rng = random.Random(seed + 1000)
    rows = []
    while len(rows) < n:
        e = tok(d[rng.randint(0, len(d) - 1)]["text"], add_special_tokens=False).input_ids
        if len(e) <= L:
            continue
        j = rng.randint(0, len(e) - L - 1)
        rows.append(torch.tensor(e[j: j + L]))
    torch.save(torch.stack(rows), path)
    return rows


def calib_data(tok, n=128, L=SEQLEN, seed=0, mix="mix"):
    """Calibration windows. mix: half WikiText-2 train + half C4 train (round-1 lesson: WikiText alone overfits)."""
    if mix == "wikitext2":
        return torch.stack(_wt2_rows(tok, n, L, seed))
    if mix == "c4":
        return torch.stack(_c4_train_rows(tok, n, L, seed))
    rows = _wt2_rows(tok, n - n // 2, L, seed) + _c4_train_rows(tok, n // 2, L, seed)
    return torch.stack(rows)


def proxy_data(tok, nseq=32, L=SEQLEN):
    """WikiText-2 train proxy: fixed seed, disjoint draw from calibration seeds."""
    return torch.stack(_wt2_rows(tok, nseq, L, 987654))


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def nll_sum(model, ids, bs=8, chunk=1024):
    """Sum of next-token NLL. The lm_head + loss run over blocks of `chunk` tokens, so full-vocabulary
    logits (bs x 2048 x 152k) are never materialised at once (was a ~10-20 GB transient at bs 8)."""
    model.eval()
    dev = next(model.parameters()).device
    nll, cnt = 0.0, 0
    for i in range(0, ids.shape[0], bs):
        x = ids[i: i + bs].to(dev)
        h = model.model(input_ids=x).last_hidden_state[:, :-1].reshape(-1, model.config.hidden_size)
        y = x[:, 1:].reshape(-1)
        for j in range(0, h.shape[0], chunk):
            logits = model.lm_head(h[j: j + chunk]).float()
            nll += torch.nn.functional.cross_entropy(logits, y[j: j + chunk], reduction="sum").item()
        cnt += y.numel()
        del h
    return nll, cnt


def perplexity(model, ids, bs=8):
    nll, cnt = nll_sum(model, ids, bs)
    return float(torch.exp(torch.tensor(nll / cnt, dtype=torch.float64)))


def _lm(model, tok, bs):
    from lm_eval.models.huggingface import HFLM
    return HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=SEQLEN)


def _collect(res, tasks):
    out = {}
    for t in tasks:
        r = res[t]
        out[t] = 100 * r.get("acc_norm,none", r["acc,none"])
        out[t + "/acc"] = 100 * r["acc,none"]
    out["avg"] = sum(out[t] for t in tasks) / len(tasks)
    return out


def zeroshot(model, tok, tasks=TASKS, bs=16, limit=None):
    """TEST tasks. Final runs (and the P0 bf16 reference) only."""
    import lm_eval
    res = lm_eval.simple_evaluate(_lm(model, tok, bs), tasks=tasks, num_fewshot=0, limit=limit,
                                  log_samples=False, bootstrap_iters=0)["results"]
    return _collect(res, tasks)


def taskdev(model, tok, tasks=DEV_TASKS, bs=16):
    """Task-dev proxy: train splits of boolq/arc_e/arc_c/piqa/winogrande, 500 sampled items each (seed 0).
    Disjoint from the splits lm-eval uses as test."""
    import lm_eval
    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=TASK_DIR)
    res = lm_eval.simple_evaluate(_lm(model, tok, bs), tasks=tasks, num_fewshot=0, task_manager=tm,
                                  log_samples=False, bootstrap_iters=0)["results"]
    return _collect(res, tasks)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30
