"""Shared loading and evaluation. SpinQuant repo is mounted at /spinquant."""
import json, os, random, sys
import torch

SPIN = os.environ.get("SPINQUANT_DIR", "/spinquant")
if SPIN not in sys.path:
    sys.path.insert(0, SPIN)

MODEL = os.environ.get("WSP_MODEL", "meta-llama/Meta-Llama-3-8B")
REVISION = os.environ.get("WSP_REVISION", "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920")
TASKS = ["boolq", "piqa", "social_iqa", "hellaswag", "winogrande",
         "arc_easy", "arc_challenge", "openbookqa"]
PPL_SETS = {"wikitext2": "test", "c4": "validation", "fw25sel": "selection", "fw25test": "test"}


def load_model(dtype=torch.bfloat16):
    import transformers
    from eval_utils.modeling_llama import LlamaForCausalLM
    config = transformers.AutoConfig.from_pretrained(MODEL, revision=REVISION)
    model = LlamaForCausalLM.from_pretrained(MODEL, revision=REVISION, config=config,
                                             torch_dtype=dtype)
    model.seqlen = 2048
    return model


def load_tokenizer():
    import transformers
    return transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True)


# ---------------- perplexity data ----------------

def _cache_path(name):
    d = os.environ.get("WSP_DATA", "/models/tmp/wsp/data")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.pt")


def ppl_ids(name, tok, seqlen=2048):
    """Return a LongTensor (n, seqlen) of eval sequences. Cached on disk."""
    path = _cache_path(f"ppl_{name}_{seqlen}")
    if os.path.exists(path):
        return torch.load(path)
    import datasets
    if name == "wikitext2":
        # Same as SpinQuant/QuaRot/GPTQ: "\n\n".join(test), no BOS, non-overlapping chunks.
        d = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
        ids = tok("\n\n".join(d["text"]), return_tensors="pt", add_special_tokens=False).input_ids
        n = ids.shape[1] // seqlen
        out = ids[:, : n * seqlen].view(n, seqlen)
    elif name == "c4":
        # GPTQ convention: c4 en validation shard 0, 256 random windows, seed 0.
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
            rows.append(enc[:, j : j + seqlen])
        out = torch.cat(rows, 0)
    elif name.startswith("fw25"):
        # Web text crawled in 2025 (after Llama 3 release). Public, but unseen by the model.
        # Disjoint document ranges: selection = docs[0:N/2], test = docs[N/2:].
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("HuggingFaceFW/fineweb", repo_type="dataset",
                            filename="data/CC-MAIN-2025-26/000_00000.parquet")
        texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        half = len(texts) // 2
        texts = texts[:half] if name == "fw25sel" else texts[half:]
        target = 160 if name == "fw25sel" else 320  # x 2048 tokens: ~330k / ~655k tokens
        buf, rows = [], []
        for t in texts:
            buf.extend(tok(t, add_special_tokens=False).input_ids + [tok.eos_token_id])
            while len(buf) >= seqlen and len(rows) < target:
                rows.append(torch.tensor(buf[:seqlen]).unsqueeze(0)); buf = buf[seqlen:]
            if len(rows) >= target:
                break
        out = torch.cat(rows, 0)
    else:
        raise ValueError(name)
    torch.save(out, path)
    return out


@torch.no_grad()
def perplexity(model, ids, bs=8):
    model.eval()
    dev = next(model.parameters()).device
    nll, cnt = 0.0, 0
    for i in range(0, ids.shape[0], bs):
        x = ids[i : i + bs].to(dev)
        logits = model(x).logits[:, :-1].float()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item(); cnt += x[:, 1:].numel()
    return float(torch.exp(torch.tensor(nll / cnt)))


def zeroshot(model, tok, tasks=TASKS, bs=32, limit=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=2048)
    res = lm_eval.simple_evaluate(lm, tasks=tasks, num_fewshot=0, limit=limit,
                                  log_samples=False, bootstrap_iters=0)["results"]
    out = {}
    for t in tasks:
        r = res[t]
        # QuaRot/SpinQuant convention: acc_norm when the task has it, else acc.
        out[t] = 100 * r.get("acc_norm,none", r["acc,none"])
        out[t + "/acc"] = 100 * r["acc,none"]
    out["avg"] = sum(out[t] for t in tasks) / len(tasks)
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
