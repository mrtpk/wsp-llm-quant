# Round 2 handoff: a message to my future self

> **Update 2026-11-11:** round 2 = **Qwen3-32B**, in `../round2-qwen3-32b/`, plan in `PLAN.md` there. Round 3 = **Qwen3.6-27B**. The ROCm check passed for both (see `../round2-qwen3-32b/NOTES.md`). The user asked to prepare only; do not start execution until they say go.

Written 2026-11-10 at the end of round 1. Context will be compacted or cleared, so this file plus `NOTES.md` and `behaviour.md` (how the user wants research done; not published) is the whole state. Read all three before doing anything.

## 1. What the user wants in round 2

The user's goal: the same accuracy or benchmark scores with less compute or memory. Follow that goal; the method is free.

- **Do not stay inside SpinQuant.** Look at AWQ, GPTQ, block/group quantization (per-group scales, MX-style block formats), SmoothQuant, mixed precision, and outlier-token handling. Take ideas from them and build hybrids.
- **Study the model first.** We have the base weights and free forward passes. Measure how the activations flow before designing anything (section 4).
- **Be creative.** Run many different experiments. Don't assume a method will fail.
- **The format may change.** W4A4KV4 in SpinQuant's exact layout is no longer required. The goal is the accuracy of FP16 (or of a strong baseline) at lower memory or compute.
- **Agree the scoring with the user before final runs.** A changed format needs an explicit cost metric (section 5). Propose it to the user before reading the test sets.

Round-1 verdict (user): about 2% better perplexity than SpinQuant is a gain that can be reported, but not a real win given the capability and resources. Do not repeat round 1's mistake: tuning knobs of one published pipeline with no profiling.

## 2. Hard constraints (still in force)

- **One GPU (GPU 1) only since 2026-11-11, at most 2 jobs (don't overload the icpc host).** (Earlier: GPUs 1 and 2, 3 jobs each.) Never use GPU 0, 2 or 3. GPUs 4–7 belong to another user. This is also in memory: `wsp-gpu-budget`.
- **Long processes in tmux.** Plain `sleep` in the foreground is blocked; use `until` loops in background Bash.
- **Never write HF_TOKEN** to files. It is in the environment, and drun.sh passes it through.
- **Keep the test sets clean.** WikiText-2 test, fw25test and the zero-shot tasks are read only for final runs. Development uses C4 validation and the WikiText-2-train proxy; selection uses fw25sel.
- **Gradient training:** the round-1 plan forbade it. Round 2 has not decided. Ask the user if a method needs it.
- **Licences:** SpinQuant is CC BY-NC 4.0 (non-commercial).

## 3. Infrastructure that works (reuse it)

Paths: `wsp-quant/` = this folder (`round1-llama3-8b/` in the public repository).

- **Image** `wsp-quant:rocm7.2`, built from `docker/Dockerfile`.
  - torch 2.10.0+rocm7.2.4, py3.12, transformers 4.44.2, lm_eval 0.4.4, datasets 2.20.0.
  - `fast_hadamard_transform` is a pure-torch shim in `docker/`, tested against an exact reference.
- **`code/drun.sh GPU cmd...`** runs a command in the container on one physical GPU.
  - Render nodes: 1→152, 2→144.
  - Mounts: code at /code, results at /results, /models, and SpinQuant read-only at /spinquant.
  - Sets the allocator to expandable segments and `OMP_NUM_THREADS=8`.
- **`code/queue.sh GPU QUEUEFILE`** works through a TSV of `NAME<TAB>CMD` lines.
  - It claims a job with `results/NAME.lock` (atomic mkdir) and marks it finished with `NAME.done`.
  - On failure it renames the lock to `NAME.failed.<ts>`. A failed job is retried automatically on the next scan.
  - It re-reads the file after each job, so lines can be appended while it runs, and it exits when the queue is empty.
  - Logs: `/models/tmp/wsp/logs/NAME.log` and `queue_g{G}.log`.
- **`code/start_queues.sh QUEUEFILE [SLOTS=3] [GPUS="1 2"]`** starts the 6 workers.
- **Pitfalls:**
  - `tmux has-session -t X` prefix-matches, so a waiter named X-something waits on itself. Use `-t =X`.
  - A waiter that tests for `.failed*` exits on the first failure, even though the queue retries the job.
  - `__pycache__` is root-owned, from the containers; `py_compile` on the host fails. Use `ast.parse` instead.
- **`code/wspq/common.py`:**
  - `load_model()` loads SpinQuant's LlamaForCausalLM; meta-llama/Meta-Llama-3-8B @ 8cde5ca8 is the base model, not instruct. `load_tokenizer()`.
  - `ppl_ids(name, tok)` for name ∈ {wikitext2, c4, fw25sel, fw25test}; cached in `/models/tmp/wsp/data/`.
  - `perplexity(model, ids)`: seqlen 2048, no BOS.
  - `zeroshot(model, tok)`: lm-eval HFLM on 8 tasks; acc_norm where available, else acc. Takes about 6 minutes.
  - `write_json`.
  - For a plain HF model, `load_model` needs a variant without SpinQuant's class. Easy to add.
- **`code/wspq/wsp.py`** (round-1 method, config-driven).
  - Includes `calibrated_gptq`, which keeps a two-stream FP/quantized calibration and an asymmetric LS target. Reusable as a GPTQ/GPTAQ solver.
  - Run as `--cfg JSON --out --ppl --proxy --zs`.
- **`code/wspq/run_ptq.py`** runs the SpinQuant `ptq_model` baseline with our evaluation. Arguments after `--` go to SpinQuant. The exact baseline command lines are in `code/queue_other.tsv` and `code/queue_spin.tsv`.
- **Results:**
  - `code/summarize.py` prints a table of all result JSONs.
  - `code/passtest.py` runs the round-1 pass test.
  - Result JSONs: `/models/tmp/wsp/results/*.json`. Since 2026-11-10 they include `peak_mem_gb`.
- **Measured costs:**
  - A full W4A4 build + all perplexity sets + zero-shot takes about 55–75 minutes per job.
  - Peak memory 54 GB allocated / 72 GB reserved, so 3 jobs per GPU fit in 256 GB.
  - SpinQuant rotation training: about 41 minutes per seed. Trained rotations: `results/spin_rot/s{0..3}/R.bin`.

## 4. Do this first in round 2: a profiling study (forward passes only, about 2–3 hours)

Write `code/wspq/profile.py`. Save the results to `/models/tmp/wsp/profile/` as .pt/.json files, and write a short summary into NOTES.

1. **Activation statistics** at each linear input, each layer, on ~64 × 2048 calibration tokens (C4 + WikiText train):
   - per-channel absmax/RMS and kurtosis;
   - per-token absmax (look for the first token, delimiters and "massive activations");
   - the residual stream norm by depth;
   - the attention-sink pattern.
2. **Error attribution.** Quantize one thing at a time and measure the change in C4 and proxy perplexity:
   - each module type (q/k/v/o/gate/up/down, the KV cache, the activations into each);
   - each layer.
   This gives the ranking of where the bits matter.
3. **Error propagation.** Quantize layer i only and track the hidden-state relative error at layers i..31. Does it grow or decay?
4. **Sensitivity curves.** For the top modules, perplexity at 8/6/4/3 bits for weights and for activations separately. Also group size (32/64/128/channel) and with or without an outlier-token exception.
5. **Weight statistics.** Per-channel ranges, low-rank structure of the quantization error, and the AWQ-style salient channels (activation-weighted).

Use these measurements to design the method, and write down the reasoning: measurement → design choice.

## 5. Candidate directions (hypotheses to test, not conclusions)

- **Weight-only W4A16 or W3A16.** AWQ / GPTQ with group 128 or 64 saves memory and decode bandwidth. Compare against W4A4 on a cost-accuracy plot.
- **Mixed precision guided by the error attribution.** Spend bits where they matter, for example 8-bit down_proj input or 8-bit for the first and last layers, 4-bit elsewhere.
- **Outlier tokens.** Keep a few massive-activation tokens (BOS, sinks) in higher precision or as a precomputed prefix, as PrefixQuant does, instead of rotating everything.
- **Block formats.** MXFP4/MXINT4-style shared scales per 32 elements, for activations as well as weights. Compare with per-token scales.
- **Hybrids.**
  - AWQ scaling + GPTQ + a rotation only where the profile shows heavy-tailed activations.
  - SmoothQuant-style migration per layer, with the strength picked by measurement.
  - Low-rank FP16 error correction, but count its cost.
- **Round-1 parts to keep.** The asymmetric GPTQ target (α≈0.15) and the activation clip (0.85) carried most of round 1's gain, so reuse them as components. The R1 seed search added nothing.

**Hardware check (verify before relying on it).** I believe MI300-class MFMA has INT8 and FP8 but no native INT4 matmul. If so, the real compute saving on MI325X comes from 8-bit and FP8 GEMMs, and 4-bit mainly saves memory and bandwidth. Check the CDNA3 ISA or the rocBLAS/hipBLASLt docs.

**Cost metric for the user to approve.** Report average bits per weight (including scales and zero-points), bits per activation, bits per KV entry, plus any extra FP16 parameters or FLOPs. Then compare methods on accuracy at equal cost, or cost at equal accuracy (for example within 1% perplexity of FP16, or within 1 zero-shot point).

## 6. Evaluation lessons from round 1

- **Perplexity under-predicts zero-shot damage.** SpinQuant s0 had normal perplexity but lost 4–6 points on boolq and ARC. Development needs a task proxy that keeps the test clean, for example the train or validation splits of boolq/arc/piqa/winogrande. Agree it with the user.
- **Calibrating on WikiText alone overfits WikiText**, and the WikiText-train proxy then rewards it. C4 val is the honest development metric for perplexity.
- **Use more baseline seeds (≥8), fixed in advance**, if any pass test uses 2σ of baseline seeds. With 4 seeds, a single bad seed set the margin.
- **Reference numbers (our harness):**
  - FP16: WikiText-2 6.138, C4 8.885, fw25sel 9.052, fw25test 9.198, zero-shot 67.09. The paper reports 69.6 zero-shot; the gap comes from the harness.
  - SpinQuant W4A4KV4 (mean over 4 seeds): WikiText-2 7.370, C4 10.951, fw25test 10.870, zero-shot 62.71.
  - Round-1 method (mean over 4 seeds): 7.220, 10.610, 10.481, 64.45.
  - No-rotation W4A4: RTN WikiText-2 214, GPTQ 176 (collapse).

## 7. First actions after reading this

1. Re-read `NOTES.md` and this file. Check that no jobs are running: `docker ps | grep wspq`, `tmux ls`.
2. Confirm the open decisions with the user, briefly, in one message:
   - the target cost metric or budget;
   - whether gradient training is allowed;
   - the zero-shot development proxy.
3. Write and run the profiling study (section 4) on GPUs 1 and 2, 3 jobs each.
4. Write the findings into NOTES, then design and run a broad set of experiments (AWQ, GPTQ, group sizes, mixed precision, outlier tokens, hybrids) through `start_queues.sh`.
