"""ROCm support check: load a model, WikiText-2 test ppl on N windows, timing, kernel path, peak memory.
--no-fla blocks flash-linear-attention so transformers uses its torch fallback."""
import argparse, sys, time, json
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--no-fla", action="store_true")
ap.add_argument("--n", type=int, default=16); ap.add_argument("--out")
a = ap.parse_args()
if a.no_fla:
    for k in ["fla", "fla.ops", "fla.ops.gated_delta_rule", "fla.modules"]: sys.modules[k] = None  # type: ignore[assignment]
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
print("torch", torch.__version__, "transformers", transformers.__version__, "device", torch.cuda.get_device_name(0), flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
model.eval(); print(type(model).__name__, f"load {time.time()-t0:.0f}s", flush=True)
mods = sys.modules[type(model).__module__]
for nm in ["torch_chunk_gated_delta_rule", "causal_conv1d_fn"]:
    f = getattr(mods, nm, None)
    if f is not None: print("kernel", nm, "->", getattr(f, "__module__", "?"), getattr(f, "__qualname__", f), flush=True)
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids
L = 2048; n = min(a.n, ids.shape[1] // L)
nll = 0.0; times = []
with torch.no_grad():
    for i in range(n):
        x = ids[:, i*L:(i+1)*L].cuda()
        torch.cuda.synchronize(); s = time.time()
        logits = model(x).logits.float()
        torch.cuda.synchronize(); times.append(time.time() - s)
        nll += torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
        if i == 0: first_logits = logits[0, :64].cpu()
ppl = float(torch.exp(torch.tensor(nll / (n * (L - 1)))))
res = dict(model=a.model, no_fla=a.no_fla, cls=type(model).__name__, windows=n, ppl=ppl,
           sec_per_window_first=times[0], sec_per_window_rest=sum(times[1:]) / max(1, len(times) - 1),
           peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
print("RESULT", json.dumps(res), flush=True)
if a.out: torch.save(dict(res=res, logits=first_logits), a.out)
