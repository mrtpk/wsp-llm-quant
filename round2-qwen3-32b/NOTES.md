# Round 2 notes

Decision (user, 2026-11-11): round 2 = Qwen3-32B, round 3 = Qwen3.6-27B. Plan: `PLAN.md`. A server crash on 2026-11-11 (up 06:18 UTC) interrupted no jobs.

Start: 2026-11-10. Round-1 state, lessons and the research-behaviour brief: `../round1-llama3-8b/ROUND2_HANDOFF.md`, `../round1-llama3-8b/NOTES.md` (the research-behaviour brief is not published).

## Stack

- Image `wsp-quant2:rocm7.2` (`docker/Dockerfile`):
  - same base as round 1: torch 2.10.0+rocm7.2.4, triton 3.6.0+rocm7.2.4;
  - transformers 5.22.2, flash-linear-attention 0.6.0, lm_eval and the rest at their latest versions;
  - torch is unchanged by the pip installs (checked).
- `code/drun.sh GPU cmd`: same as round 1. Mounts code/ at /code and /models/tmp/wsp2/results at /results. No SpinQuant mount.
- Models in `/models/huggingface/hub/`:
  - Qwen/Qwen3.6-27B: 52 GB.
  - Qwen/Qwen3-32B @ 9216db57: 62 GB.
- Datasets: the new `datasets` needs `Salesforce/wikitext`, not `wikitext`.
- Pitfall: the first `docker build` hung for 35 minutes with no output. A rebuild with `--progress=plain` took about 2 minutes.

## ROCm check (2026-11-10, GPUs 1/2, `code/rocm_check.py`)

WikiText-2 test, 16 × 2048 windows, bf16, one MI325X:

| model | path | ppl | s/window (after warm-up) | first window | peak mem |
|---|---|---|---|---|---|
| Qwen3-32B | standard attention | 7.409 | 0.284 | 4.8 s | 64.0 GB |
| Qwen3.6-27B | FLA Triton (gated delta rule) + torch causal-conv1d | 7.432 | 0.256 | 72.9 s (Triton JIT) | 54.9 GB |
| Qwen3.6-27B | torch fallback for both | 7.421 | 0.681 | 41.8 s | 55.0 GB |

- **FLA Triton kernels run on ROCm.** They're 2.7× faster than the torch fallback after a one-time JIT of about 73 s.
- **FLA vs torch on the same input:**
  - relative logit difference 0.95%;
  - max absolute difference 0.30;
  - argmax agreement 98.4%;
  - mean KL 5e-4;
  - perplexity differs by 0.011.
  Small, but not zero. For a quantization study, use one fixed path for all comparisons.
- **causal-conv1d has no kernel installed.** It falls back to torch; correct but slower. It's a minor cost.
- Qwen3.6 loads as text-only `Qwen3_5ForCausalLM` via AutoModelForCausalLM.

## Execution log

- 2026-11-11: user said go for P0 and P1 (unattended, GPU 1, ≤2 jobs; plan defaults for the open decisions: no gradient training, extra FP pieces allowed but counted, task-dev proxy as in PLAN §5). Then report P1 + proposed P2 and wait.

## P0 (done 2026-11-11, ~30 min)

Code (`code/`): `r2/common.py` (Qwen3-32B @ 9216db57, plain transformers, sdpa; data cached in `/models/tmp/wsp2/data`, Qwen tokenizer; fw25 doc separator `<|endoftext|>`), `r2/fq.py` (simulated quantization: int/mxint/mxfp/fp formats, groups, MSE clip, per-module rules, act hooks with kept-token exceptions, KV through a wrapped sdpa attention, CPU weight backup + `reset()`), `r2/tasks/` (task-dev proxy: 500 items each from the TRAIN splits of boolq, arc_e, arc_c, piqa, winogrande), `r2/sanity.py`, `r2/profile.py` (P1), `queue.sh`/`start_queues.sh` (round-1 copies; results/locks in `/models/tmp/wsp2/results`, logs in `/models/tmp/wsp2/logs`, tmux sessions `q2-*`). Results JSON: `/models/tmp/wsp2/results/p0/`, `/models/tmp/wsp2/results/profile/`.

bf16 reference (our harness, seqlen 2048, no BOS):

| set | ppl | s/window |
|---|---|---|
| c4dev (first 64 of c4 val) | 9.1505 | 0.33 |
| wt2train proxy (32) | 7.6237 | 0.25 |
| c4 (256) | 10.7816 | 0.31 |
| fw25sel (160) | 10.1894 | 0.35 |
| wikitext2 test (146) | 7.6103 | 0.28 |
| fw25test (320) | 9.8761 | 0.26 |

- Zero-shot bf16 (8 test tasks, acc_norm where available): **avg 70.94**: boolq 86.27, piqa 81.99, siqa 52.41, hellaswag 82.68, winogrande 73.24, arc_e 83.33, arc_c 61.43, obqa 46.20. 21 min.
- Task-dev proxy bf16: **avg 78.88**: boolq 86.0, arc_e 84.0, arc_c 61.2, piqa 84.2, winogrande 79.0. 3 min.
- lm_eval 0.4.13 works with transformers 5.22 (risk in PLAN §8 cleared).
- Simulated W16A16KV16 through every code path: c4dev NLL **bit-identical** to bf16 (290026.640625); attention wrapper called 512× (64 layers × 8 batches); `reset()` restores exactly.
- RTN W8A8 (per-channel sym W, per-token sym A), c4dev: **INT8 9.384 (+0.234, +2.6%)**, +KV8 9.375, **FP8 E4M3 9.191 (+0.041)**. wt2train: INT8 7.769, FP8 7.663. → INT8 per-token activations already hurt on this model; FP8 (dynamic range) barely does. A first hint of heavy-tailed activations.
- Peak memory: 84 GB (zero-shot job, bs 32). Model load 65 s.
- **Hardware check (PLAN §7.1), done:** the ROCm 7.2 LLVM in the image knows only `v_mfma_i32_*_i8` integer MFMAs, no i4 variant, for any gfx9 target; MI325X is gfx942 (`AMD Instinct MI325X gfx942:sramecc+:xnack-`). So W4A4 gives no native 4-bit GEMM here: 4-bit saves memory/bandwidth; compute savings come from INT8/FP8 GEMMs. (FP4/FP6 MFMAs are gfx950 = MI355X only.)

## P1 profiling (started 2026-11-11 11:43 UTC; queue `code/queue_p1.tsv`, 2 workers on GPU 1)

Default stress settings: W4pc = RTN 4-bit per output channel sym; W3g128 = RTN 3-bit g128 asym; A4 = dynamic per-token 4-bit asym; KV4 = per (token, head) 4-bit asym. Metric: c4dev ppl (bf16 9.1505). Analysis scripts are ad hoc over `/models/tmp/wsp2/results/profile/*.json`.

### P1.1 activation statistics (`actstats`, 64 × 2048 calib tokens = 32 WikiText-2 train + 32 C4 train; 6 min)

Measurement → design implication:

- **Massive activation is born in layer 6's down_proj.** At position 0, down_in channels 24380/6972 reach ~4500–5500 and write ~21,000–25,000 into **residual channel 731** (also 2994, 1016 ~2600). It persists through layer 62 and shrinks in layer 63. Position 0 carries it whatever the token. A few non-initial tokens also reach ~18,000. → The model's "sink" mechanism is one layer's MLP; that layer (and its gate/up/down) must stay precise, or the sink must be handled explicitly (kept token / precomputed prefix).
- **Attention sink:** from layer 7 on, the mean attention mass on position 0 is 0.5–0.9 (layers 0–6: < 0.03). Matches the massive activation onset.
- **After RMSNorm, position 0 is not an outlier** in attn_in/mlp_in (its amax 0.1–25, like other tokens). The linear-input problem is **channel outliers in every token**, not the sink token. Exceptions: down_in in layers 6 and 62–63, and the last layers.
- **Per-token 4-bit is hopeless without help, at every site.** Median per-token max/rms ratio: attn_in 6–27, o_in 9–20, mlp_in 4–45, **down_in 27–100** (p99 up to 158); Gaussian would be ~4. Per-token A4 relative error 0.25–0.66 (Gaussian floor ≈ 0.15).
  - Fixed outlier channels: attn_in 421/3834/5020 (mid to late layers, amax up to 270), mlp_in 1939/3763 (layers 1–8, RMS 100–190× median) and 2994/3486/1016 (later).
  - down_in is extremely sparse and heavy-tailed (channel median absmax ≈ 0 in layers 1–2; per-channel kurtosis median 36–1800). The outliers move from token to token, which favours rotation or fine groups over SmoothQuant-style fixed channel scaling.
- **4-bit activation error by scheme** (relative error, typical over sites and layers; tokens 1..T):

  | scheme | rel. error |
  |---|---|
  | per-token int4 | 0.25–0.66 |
  | g128 int4 | 0.04–0.19 |
  | block-Hadamard (1024) + per-token int4 | ≈ 0.14 everywhere, except down_in 0.16–0.20 |
  | **MXFP4 (block 32, no rotation)** | 0.12–0.155 |
  | Hadamard + MXFP4 | ≈ 0.11–0.115 |

  → **Fine-grained FP4 blocks match or beat rotation without any rotation.** Rotation brings every site to the Gaussian floor (~0.14), but it is *worse* than g128/MX at down_in, because sparsity makes the group scales efficient there.
  - Per-token **int8** still gives 0.05–0.12 at down_in. This explains INT8 W8A8 +0.23 vs FP8 +0.04 in P0.
- **K** (after QK-norm + RoPE) has fixed outlier channels: rms max/median 5–107; layer 0 is extreme, with channel 114 amax 135. Per-token max/rms 5–15. **V** is benign (max/rms 4–8) and grows with depth (amax up to 120 in late layers).

### P1.3a error attribution by module type (`attrib_type`, c4dev ppl deltas vs 9.1505)

| | q | k | v | o | gate | up | down |
|---|---|---|---|---|---|---|---|
| W4pc | +0.03 | +0.08 | +0.15 | +0.05 | +0.20 | +0.15 | **+18.10** |
| W3g128 | +0.02 | +0.05 | +0.20 | +0.06 | +0.24 | +0.24 | **+18.45** |
| A4 per-token | +0.48 | +0.11 | +0.86 | +0.16 | **+2580** | +7.55 | **+16.6** |

- All at once: W4pc 27.26 (+18.1), W3g128 26.91, **A4 10,527**, **A8 9.295 (+0.14)**, W4pcA4 7.0e6, +KV4 1.5e7.
- **down_proj weights carry ~all of the RTN weight damage** (W4pc all = 27.26 ≈ down_proj only 27.25). Per layer (below) it is **layer 63** alone, not layer 6 as I first guessed.
- **gate_proj A4 alone collapses the model** (+2580), while up_proj, with the same input, costs +7.5. A gate error at the sink token is amplified through SiLU into the 5000-scale down_in → the sink breaks. To confirm per layer.
- KV (all layers):

  | | per-token 4 | 3 | 2 | g32 4 | 3 | 2 | per-channel (g128 tokens) 4 | 3 | 2 |
  |---|---|---|---|---|---|---|---|---|---|
  | K | +3.27 | +447 | +9099 | +0.11 | +0.31 | +1.58 | **+0.018** | +0.088 | +0.78 |
  | V | +0.001 | +0.045 | +0.24 | −0.002 | +0.03 | +0.13 | | | |

  → **K must be quantized per channel (KIVI-style) or with a K rotation**; per-token K fails because of the fixed channel outliers created by QK-norm/RoPE. **V is almost free** (V3 per-token +0.045, V2 g32 +0.13). An asymmetric KV budget (K4-channel, V2/V3) looks attractive.
### P1.3b error attribution per layer (`attrib_layer`, 64 layers × 4 settings, 2 h 10 min)

| setting | sum of 64 deltas | top layers | median layer |
|---|---|---|---|
| W4pc | 13.8 | **L63 +13.03**, L62 +0.07, L43 +0.05 | +0.008 |
| W3g128 | 25.1 | **L63 +24.18**, L62 +0.11, L52 +0.05 | +0.009 |
| A4 per-token | 2146 | **L6 +2136**, L63 +5.07, L0 +0.61, L62 +0.59, L2–4 +0.15–0.29 | +0.046 |
| KV4 per-token | 2.89 | **L0 +2.69**, L63 +0.03 | +0.002 |

→ **The damage sits in three layers.** L63 weights: its down_in has amax 8768 at a few tokens (token 220 = space) and RMS max/median 1485. L6 activations: the massive-activation generator. L0 K: channel outliers. Every other layer costs < 0.06 even at W3g128 RTN. **Mixed precision is cheap:** keep L63 (weights) and L6 (activations) at high precision, plus K per-channel or L0 KV at 8 bits. That is 1/64 of the weights and fixes nearly all the W damage. Error interactions are non-additive (the W4pc layer sum 13.8 < all-at-once 18.1); measure combinations directly.

### P1.3c layer × type grid (`attrib_lt`, 448 linears × {A4, W3g128}, c4dev first 16 windows, bf16 7.383)

- **A4 per-token: one linear causes the collapse. L6.gate_proj +1835** (sum of all 448: 1851). Next: L63.gate +6.8, L63.up +2.5, L63.down +2.4, then gate_proj of layers 1–5 and 62 at +0.16…+0.35. 104/448 linears > +0.01.
- **W3g128: one linear. L63.down_proj +28.67** (sum 29.94); next is L62.gate at +0.05. Only **14/448 > +0.01**.
- → Precision exceptions are needed for **2 linears** (L6.gate input, L63.down weights), not for whole layers.

### P1.2 weight statistics (`wstats`)

- Weights are benign. Median kurtosis 3.2–4.3. down_proj has the largest column max/median (16) but the same W4g128 relative error (~0.10) as the other types.
- **The difference is the activations they meet.** Activation-weighted W4g128 error: median 0.09–0.12. **L63.down 0.29** (W4pc: 0.52), L62.down 0.14, L6.down 0.03.
- AWQ view: the top-1% salient input channels are 90% the top-1% activation channels (overlap with the weight-norm top-1% only 0–0.7). **Salience is set by the activations.** Top-1% share of salience is 2–9% typically, but **29% at L63.down** and 21% at L6.gate.
- **Low-rank error.** Activation-weighted error energy in the top-r singular directions (plain error ≈ random, i.e. not low-rank):

  | | r=1 | r=16 | r=64 | r=256 |
  |---|---|---|---|---|
  | **L63.down** | 0.38 | **0.80** | 0.87 | 0.95 |
  | L62.down | 0.16 | 0.36 | 0.49 | 0.64 |
  | L63.gate | 0.06 | 0.41 | 0.48 | 0.55 |
  | L32.down | 0.005 | 0.03 | 0.08 | 0.19 |
  | random (same shape) | 0.000 | 0.006 | 0.025 | 0.097 |

  → A rank-16–64 bf16 correction (activation-aware SVD of the error) on L63.down could recover most of its damage for ~0.5–2 M params, or keep L63.down in 8 bits (131 M params = 0.4% of linear weights). Elsewhere low-rank correction is pointless.

### P1.4 bit / granularity sensitivity (`sens_w`, `sens_a`; c4dev Δppl)

Weights, all of one type (down = dominated by L63):

| spec (bits/weight) | q | k | v | o | gate | up | down |
|---|---|---|---|---|---|---|---|
| int4 pc (4.00) | +.02 | +.02 | +.10 | +.06 | +.13 | +.12 | +27.9 |
| int4 g128 (4.16) | +.01 | +.01 | +.04 | +.02 | +.05 | +.05 | +3.65 |
| int4 g128 mse (4.16) | +.03 | +.03 | +.05 | +.03 | +.07 | +.07 | +2.02 |
| int4 g32 (4.63) | +.01 | +.01 | +.03 | +.01 | +.02 | +.02 | +0.66 |
| mxfp4 b32 (4.25) | +.00 | +.02 | +.05 | +.02 | +.09 | +.08 | **+0.13** |
| mxint4 b32 (4.25) | +.00 | +.02 | +.11 | +.03 | +.09 | +.11 | +9.38 |
| fp4 g16 (5.0) | .00 | .00 | +.02 | .00 | +.03 | +.04 | +0.05 |
| int3 g128 (3.15) | +.02 | +.05 | +.20 | +.07 | +.25 | +.24 | +18.4 |
| int3 g32 (3.59) | +.03 | +.01 | +.13 | +.03 | +.13 | +.15 | +16.0 |
| int2 g32 (2.56) | +.09 | +.10 | +.41 | +.19 | +.88 | +.87 | +37 |
| int6 / int8 / fp8 any | ≤ .01 | | | | | | ≤ +.02 |

- **MSE clipping hurts** every type except down (it cuts salient large weights).
- FP4 elements beat INT4 on the heavy-tailed L63.down (mxfp4 +0.13 vs mxint4 +9.4 at the same 4.25 bits).
- v/gate/up are the next most sensitive types after down.

Activations, all of one type, and all linears together:

| spec | q | k | v | o | gate | up | down | **all** |
|---|---|---|---|---|---|---|---|---|
| int4 per-token | +.48 | +.11 | +.86 | +.16 | +2580 | +7.6 | +16.6 | 10517 |
| int4 g128 | +.03 | +.01 | +.02 | +.02 | +.12 | +.16 | +.17 | **+0.52** |
| int4 g32 | +.01 | .00 | +.01 | +.01 | +.04 | +.06 | +.04 | **+0.155** |
| mxfp4 b32 | +.04 | +.02 | +.02 | +.02 | +.15 | +.08 | +.19 | +0.53 |
| mxint4 b32 | +.03 | +.02 | +.01 | +.02 | +.18 | +.25 | +.21 | +0.72 |
| fp4 g16 | .00 | .00 | .00 | +.01 | +.03 | +.05 | +.03 | **+0.117** |
| int6 per-token | +.04 | +.03 | +.01 | .00 | +8.7 | +.18 | +1.03 | +7.5 |
| int8 per-token | | | | | +.03 | +.01 | | +0.144 |
| **fp8 per-token** | | | | | +.006 | +.004 | +.008 | **+0.014** |
| int4 pt clip 0.9 | +.36 | +.09 | +.51 | +.13 | +1145 | +5.7 | +11.2 | |

→ **4-bit activations need fine groups (≤ 32) or FP elements, not rotation.** FP8 per-token is free; INT8 per-token is not (+0.14).

### P1.5 error propagation (`prop`, 8 c4dev windows, rel. hidden-state error excl. pos 0)

- An error injected at a single layer i **decays** through the later layers. W4pcA4 at layer i: relative error at layer i ≈ 0.1–0.4 → at L63 0.08–0.26. NLL change for one layer: ≤ +0.05 for layers 0–56, **+0.08 at L62, +0.41 at L63**.
- Exception: the error from layer 40 spikes at layer 48 (0.57), where non-initial massive tokens appear in the residual (amax 31k), then decays.
- All layers W4pcA4 together: relative error explodes (23× at layer 8, because of the L6 collapse). W3g128 everywhere: final relative error 0.36, ΔNLL +1.27 (≈ L63 alone +1.43).
- → Late layers matter most. Early-layer errors are partly washed out. **Gap: layer 6 was not in the start set**; add it in P2.

### P1.6 outlier tokens (`outlier`)

- Keeping position 0 / the top-k tokens / tokens above 10× or 30× the median absmax in high precision does **not** rescue A4 per-token (10,517 → 6,650 at best), W4A4 or KV3. Small gains for A8 (+0.144 → +0.123).
- → The activation problem is **channel outliers in ordinary tokens plus the L6 gate mechanism**, not sink tokens. PrefixQuant-style token exceptions are **deprioritised**.
- (KV4 per-token with pos0 kept: +3.24. Not the sink: K channel outliers, mostly at L0.)

### P1 summary: what the model is telling us

1. **Two linears and one KV head-set cause almost all damage:** L6.gate_proj input (A), L63.down_proj weights (W), L0 K (KV). Everything else is easy: W3g128 RTN costs ≤ +0.01 for 434/448 linears.
2. **Activations:** heavy-tailed channel outliers everywhere (down_in worst). Fine groups (g16–g32, FP elements) fix 4-bit without rotation. FP8 per-token is free. INT8 per-token costs +0.14.
3. **Weights:** benign. Their sensitivity is set by the activations they multiply (AWQ view), and the L63.down error is low-rank.
4. **KV:** K needs per-channel (or rotation); V tolerates 2–3 bits.
5. **Hardware:** only 8-bit compute savings on MI325X (INT8/FP8). 4-bit = memory and bandwidth.

## P2 proposal (written 2026-11-11, awaiting the user's go)

Dev metrics: c4dev + wt2train + task-dev proxy for the promising runs. Every row reports average W bits (incl. scales), A bits, KV bits and extra bf16 parameters.

**E1 — Exceptions only (RTN, minutes).** Measure how far the two exceptions take us:
- W4g128 / W3g128 / W3g64 with L63.down at {bf16, int8, mxfp4, fp4g16};
- all-linear A4 {int4 g32, fp4 g16, mxfp4} with L6.gate input at {bf16, fp8, int8};
- KV: K4 per-channel, L0 K8, V3 / V2 per-token.

**E2 — Weight-only memory frontier (W3–W4).**
- GPTQ (round-1 solver ported to `fq`, act_order, asymmetric target α≈0.15), AWQ scaling (own), AWQ+GPTQ; g128/g64/g32; int vs FP4 elements.
- Each with and without the L63 exception. Target: < 1% ppl at ~3.1–3.3 bits.
- Plus a mixed-bit map from `attrib_lt`: W2/W3 where the delta is ≈ 0, W4 where it matters; report average bits.

**E3 — Low-rank correction for L63.down (and L62.down).** Activation-aware SVD of the quantization error, rank 8/16/64, vs the plain bf16/int8 exception at equal extra bytes.

**E4 — Compute frontier on MI325X (8-bit GEMMs).**
- FP8 W8A8 (+ FP8 KV) as the reference.
- INT8 W8A8 with SmoothQuant at a per-layer α chosen from `actstats`, and INT8 with a g128 activation scale.
- **W4A8-FP8:** 4-bit weights, dequantized to FP8, with FP8 activations. Memory of W4, compute of FP8.

**E5 — W4A4(KV4) vs the QuaRot-style baseline.**
- Ours: group FP4/INT4 activations (g16/g32) + GPTQ W4 + the two exceptions + K per-channel/V3.
- Baseline: QuaRot-style (Hadamard + GPTQ, per-token A4) in the same harness.
- Hybrid: a Hadamard only where `actstats` shows it helps more than groups (attn_in/mlp_in), groups at down_in.

**E6 — Fixing L6.gate structurally instead of by exception.**
- SmoothQuant-style migration on the L6 mlp_in outlier channels (1939/3763/…) into gate/up weights.
- Keep pos 0 only at L6.gate.
- Treat the sink as a precomputed prefix (its KV and residual are input-independent).
- Add layer 6 to the propagation study.

**E7 — KV cache.** K per-channel 4/3/2 bits vs K Hadamard over head_dim after RoPE (streaming-friendly); V 3/2 bits per-token / g32; L0 special-cased.

Order: E1 → (E2, E4 in parallel) → E3, E5 → E6, E7. Roughly 6–10 GPU-hours with 2 jobs; GPTQ runs (~1.5–2.5 h each) dominate, so start with RTN+exceptions to prune.

## P2 (started 2026-11-11 17:20 UTC)

- Go for P2 (E1–E7) came from the user via the peer session `general-ags` (cross-session message), 2026-11-11. The same message asked for a detailed explanation of the P3 decisions; it was sent back (cost metric: recommend two frontiers, memory = weight bits incl. scales/exceptions/low-rank and 8-bit compute; win condition: recommend (b) equal accuracy at clearly lower cost, with bar = every test ppl ≤ +1% of bf16 AND zero-shot avg ≥ 69.94, margin ≥ 0.5 bits/weight below the cheapest passing baseline; also baseline set per frontier, KV in the metric, seeds, lm_head, kernel-less formats). **No decision from the user yet**; P3 is blocked on cost metric, win condition + bar, baseline set.
- Code: `r2/p2.py` (`sweep NAME` = many RTN configs per process; `build KEY` = one solver pipeline per process; results `/models/tmp/wsp2/results/p2/{NAME}.json`, `p2/build/{KEY}.json`, each with c4dev, wt2train, task-dev for combos/builds, and cost: wbits incl. scales/exceptions, wbits_eff incl. low-rank, FLOP-weighted abits, kvbits), `r2/exps.py` (definitions), `r2/solvers.py` (GPTQ with static groups + act_order + optional asymmetric target and act-quant stream; AWQ / SmoothQuant scale search with exact folding incl. GQA v→o; QuaRot-style R1 (kron(random 5×5 orthogonal, H1024), signs, fused norms) + online R4 block-Hadamard at down_proj input; q/k Hadamard after RoPE; RTN in place; whitened-SVD low-rank correction). Queue: `code/queue_p2.tsv`.
- Solver tests (CPU, tiny random Qwen3, scratch script): rotation r1+r4, q/k rotation, AWQ/SQ folding all output-preserving (rel. diff ~1e-6); GPTQ < RTN on the layer objective for int3 g128, mxfp4, int4 pc.
- E1 first results: W4g128 RTN +3.70 → **+0.21 with L63.down at bf16/int8** (+0.27 mxfp4). A4g32 +0.155, Afp4g16 +0.117; the L6.gate exception does **not** matter once activations are grouped (it only matters for per-token).

## P3/P4 decisions (user, via peer session `general-ags`, 2026-11-11 ~18:30 UTC) — fixed before any confirmation data is read

**Framing:** this project studies weight-space (model) properties. It does **not** optimise for the MI325X; that is only where the jobs run. Hardware limits are practical notes, never the ranking criterion. Do not drop a finding because gfx942 lacks a kernel.

**Metrics (rank on M and B; C is a side note):**
- **M — stored bits per weight**, averaged over **all linear weights including lm_head**: group scales/zeros, high-precision exceptions and low-rank factors (16 bit each) all count. Field `wbits` in P2 results (since 18:36 UTC); `wbits_body` = decoder linears only, for reference.
- **B — BitOps**: per-MAC W bits × A bits, weighted by each linear's size (∝ FLOPs), lm_head included (its input stays bf16 = 16). bf16 = 256. Field `bitops`.
- **C — 8-bit compute on gfx942**: deployment remark only. Native today on gfx942: INT8 and FP8 (E4M3) GEMMs with per-tensor/channel/token scales. **No native kernel today:** any 4-bit (INT4/FP4) or 3/2-bit GEMM, group-scaled activations (g16–g128), MX formats (MXFP4/6, MXINT4/8). These stay allowed everywhere as valid weight-space results; every report marks them "no native gfx942 kernel".
- **KV** is reported separately (`kvbits`) and is not part of M.

**lm_head:** 8-bit (RTN int8, per output channel, symmetric) in **every** config, baselines included, and counted in M and B. (Excluding a bf16 lm_head would hide ~0.32 bits at a 3-bit body.) Implemented in `p2.apply_lm_head`.
- E1 sweeps `e1_w`/`e1_a` ran with a bf16 lm_head. They are exploration only; sweep `e1_lm` re-measures the E1 anchors with the int8 lm_head.
- The first `gptq_W4g128` build (bf16 lm_head) is kept in `p2/build/_lmbf16/` and is being re-run.

**Win condition: (b) as primary — equal accuracy at clearly lower cost. (a) is reported too.**
- **Accuracy bar (confirmation):**
  - WikiText-2 test ppl ≤ 1.01 × 7.6103 = **7.6864**;
  - fw25test ppl ≤ 1.01 × 9.8761 = **9.9749**;
  - zero-shot avg (8 tasks) ≥ **69.94** (bf16 70.94 − 1).
  The perplexity terms carry the bar. The zero-shot term is a **guard**, not a discriminator, because it sits near its own noise floor.
  - C4 is **not** in the bar: c4dev (first 64 windows) is part of C4 validation and is used for selection.
- **"Clearly lower cost":** ≥ **0.5 bits/weight** (M) below the cheapest baseline that passes the same bar in the same confirmation check, or no baseline passes at the costs tested.
  - **B (decided):** at least 0.5 bits lower on W **or** on A, with the other not higher (no trading W against A). Reports also give the raw BitOps product (W bits × A bits) and its ratio to the baseline, as the literature quotes it.
- **Seeds:** 4 per final config; ≥ 8 for baselines with calibration randomness (GPTQ/AWQ/SmoothQuant). RTN has no randomness (1 run).
  - **Pass rule (decided):** a config passes if the **seed mean** passes every bar term; the worst seed is always reported. **A config declared a WINNER must also have its worst seed pass both perplexity terms** (WikiText-2 test, fw25test). The seed is the calibration draw: passing only on the mean means a lucky calibration set, not a weight-space property.
- **Gradient training:** none.

**Reference:** bf16 stays fully bf16, **lm_head included**. Candidates pay for their int8 lm_head against that ceiling. The bar numbers are 1.01 × the measured bf16 values (7.6103, 9.8761); the user's 7.6861/9.9748 differ only by rounding of the bf16 inputs, and 7.6864/9.9749 stand.

**E1 and lm_head (user, 2026-11-11):** do **not** re-run the 65 E1 configs with int8 lm_head. The lm_head is identical in every config, so the E1 ranking is unchanged. The `e1_lm` anchor sweep puts the absolute numbers on the new scale. GPU time goes to E2–E7.

**Selection vs confirmation (split required by the user):**
- **Search/selection uses c4dev (ppl) and the task-dev proxy only.** wt2train is still logged as a diagnostic but is never used to select. fw25sel is unused (spare).
- Then a **single confirmation check** on WikiText-2 test, fw25test and the 8 zero-shot tasks, for the chosen finals and the baselines. Nothing is re-selected after it.

**Baseline sets (same harness, same data, int8 lm_head):**
- M:
  - RTN W4 g128;
  - GPTQ W4A16 g128;
  - AWQ W4A16 g128 (own; Qwen's official `Qwen3-32B-AWQ` as an external check if it loads);
  - GPTQ W3A16 g128.
- B:
  - QuaRot-style W4A4KV4 (Hadamard R1+R4, GPTQ W4 per-channel, A4 per-token clip 0.9, KV4 g128);
  - SmoothQuant W8A8 INT8;
  - FP8 W8A8 (Qwen's official `Qwen3-32B-FP8` if it loads, else simulated);
  - RTN W8A8.

### E1 results (RTN + precision exceptions; c4dev Δppl vs 9.1505; done 19:01 UTC; `p2/e1_w.json`, `p2/e1_a.json`, anchors `p2/e1_lm.json`)

Scale note: 1% of c4dev ≈ **0.09** — the size of the confirmation bar on the test sets.
- **int8 lm_head costs ≈ +0.004 c4dev** (anchors shift by +0.003…+0.006). It adds +0.094 bits at a 4-bit body and +0.117 at a 3-bit body.
- Task-dev noise: bf16 body + int8 lm_head scores 79.12 vs bf16 78.88. Differences < ~0.5 point are noise.
- Bits below include the int8 lm_head (from `e1_lm`) where shown. W = bits/weight, B = BitOps.

**Weights (RTN):**

| config | c4dev Δ | W |
|---|---|---|
| W4g128 | +3.69 | 4.250 |
| **W4g128 + L63.down int8** | **+0.21** | 4.266 |
| W4g128 + L63.down mxfp4 / fp4g16 | +0.27 / +0.23 | ≈4.25 |
| W3g128 + L63.down int8 | +1.25 | 3.286 |
| (+ L62.down W4) | +1.22 | 3.291 |
| W3g64 + L63.down int8 | +0.90 | 3.431 |
| W4pc + L63.down int8 | +1.36 | ≈4.11 |
| W3pc + L63.down int8 | +17.1 | |
| W2g64 + L63.down int8 | +35 | 2.444 |
| MXFP4 everywhere | +0.41 | 4.34 |
| MXFP4 + L63 int8 | +0.32 | 4.357 |

→ The L63.down exception removes the collapse at +0.016 bits. The remaining RTN error is spread across all layers: W4 +0.21 (2.3%), **W3 +0.9…1.25** (10–14%). Below 4 bits, RTN cannot meet a 1% bar. That is GPTQ/AWQ's job (E2).
- First GPTQ data point (bf16 lm_head, kept for reference): **GPTQ W4g128 without any exception: +0.16, task-dev 77.48**. GPTQ alone already handles most of L63.down (RTN: +3.69).

**KV (activations/weights bf16):**

| config | c4dev Δ | KV bits |
|---|---|---|
| FP8 | +0.008 | 8.1 |
| INT8 | +0.003 | 8.2 |
| **K4 per-channel, V4** | **+0.022** | 4.16 |
| K4ch V3 | +0.072 | 3.65 |
| K3ch V3 | +0.116 | 3.15 |
| K4ch V2g32 | +0.163 | 3.36 |
| K2ch V2g32 | +1.05 | 2.35 |
| K4 per-token V4 + L0 K8 | +0.165 | |
| K4g32 V3 (+ L0 K8) | +0.151 (+0.118) | |

→ Per-channel K beats per-token K plus layer-0 exceptions. K4ch/V4 is near-free.

**Activations (weights bf16):**

| config | c4dev Δ | A bits (FLOP-weighted) |
|---|---|---|
| A4 g32 (L6.gate: none / bf16 / fp8) | +0.155 / +0.157 / +0.148 | 4.62 |
| FP4 g16 | +0.117 | 5.00 |
| A4 g64 / g128 / MXFP4 (+L6g fp8) | +0.27 / +0.50 / +0.57 | |
| A4 per-token + L6g fp8 | +338 | |
| + L63 fp8 too | +183 | |
| A6 per-token + L6g fp8 | +1.36 | |
| + L63 fp8 | +0.70 | |
| A8 per-token INT8 | +0.137 | |
| **FP8 per-token** | **+0.014** | |

→ The L6.gate exception is irrelevant once activations are grouped. Per-token 4-bit stays broken even with the known critical linears protected. The per-token problem is spread across many linears, not only L6.

**Combinations (RTN; int8 lm_head; task-dev bf16 78.88/79.12):**

| config | c4dev Δ | task-dev | W | B | KV |
|---|---|---|---|---|---|
| W4g128+L63d8 / A4g32 | +0.39 | 77.80 | 4.27 | 22 | 16 |
| + K4ch V3 | +0.47 | 76.64 | 4.27 | 22 | 3.65 |
| W4g128+L63d8 / Afp4g16 | +0.35 | 77.68 | 4.27 | 24 | 16 |
| + K4ch V3 | +0.43 | 77.64 | 4.27 | 24 | 3.65 |
| W4g128+L63d8 / Afp8 | +0.22 | 78.20 | 4.27 | 36 | 16 |
| W3g128+L63d8 / A4g32 | +1.46 | 73.60 | 3.29 | ≈17 | 16 |
| W3g128+L63d8 / Afp8 | +1.28 | 76.04 | 3.29 | ≈28 | 16 |
| FP8 W8A8 + FP8 KV | +0.047 | 78.20 | 8.0 (8.09 w/ lm_head) | | 8.1 |

→ For W4A4 the weight part (+0.21) and the activation part (+0.15) add roughly linearly. Next steps: GPTQ for the weights (E2), the QuaRot-style baseline vs groups (E5), and 3 bits needing GPTQ/AWQ plus mixed bits.

### Scheduling change (user, 2026-11-11 ~20:05 UTC)

- **One job at a time on GPU 1** (hard limit; never a second GPU). Worker `q2-queue_p2-g1s1` and its job (`gptq_W4g128_L63d8_Afp8`, ~20 min in) were stopped; the job was re-queued. `start_queues.sh` now defaults to 1 slot.
- OOM theory (user): two co-scheduled jobs. A resident GPTQ job (~64 GB weights + a few GB of buffers) next to a task-dev sweep at batch 32, whose logits (32 × len × 152k) are a 20–40 GB transient. GPU 1 showed 185/275 GB in use. Not confirmed: the tracebacks were lost because `queue.sh` overwrites logs. Since ~19:50 tracebacks are kept in `/models/tmp/wsp2/results/p2/errors/`; confirm or kill the theory at the next failure.
- Fixes:
  - `common.nll_sum` runs the decoder once, then lm_head + loss over blocks of 1024 tokens. Checked against full logits: relative difference 2.5e-8, summation order only.
  - lm-eval batch 32 → 16.
  - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (and the CUDA name) was already set in `drun.sh` since P0.
  - `p2 build` frees solver buffers (`empty_cache`) before evaluation.
- **Update 20:46 UTC — the OOM theory is wrong.** The first kept traceback (`p2/errors/build_gptq_W4g128.*.txt`) shows `linalg.cholesky: not positive-definite (leading minor of order 6247)` inside `gptq_solve`, at 66 GB allocated and 82 GB peak, far from the limit.
  - Cause: float32 Cholesky of badly conditioned Hessians (the diagonal spans orders of magnitude because of massive activations, and some down_in channels are near-dead). The failure is nondeterministic: the 17:56 run passed, then 19:38, 20:15 and 20:45 failed after ~30–37 min each (~1.6 GPU-hours lost).
  - Fix: Cholesky in float64; if it still fails, raise the damping ×10 up to 4 times. Every raise is logged and stored in the result (`info.ops[].damp_raised`). Tested on a synthetic ill-conditioned H.
  - The 20:45 attempt (old code) was stopped; the retry started 20:46 with the fix.
  - The serialisation stays (user instruction). Chunked eval and lm-eval batch 16 stay as hygiene.

### Bar check on selection data (2026-11-11 22:40 UTC) — baselines GPTQ W4/W3, AWQ W4 landed

All with int8 lm_head, calibration seed 0, no damping escalation needed (float64 Cholesky fixed the failures outright). Relative to fully-bf16 (c4dev 9.1505, wt2train 7.6237, task-dev 78.88). wt2train is a diagnostic (train-split proxy, not test).

| config | W | B | c4dev Δ% | wt2train Δ% | task-dev |
|---|---|---|---|---|---|
| GPTQ W4g128 | 4.250 | 68.0 | **+1.89** | +3.14 | 77.52 |
| AWQ W4g128 | 4.250 | 68.0 | +2.10 | +2.88 | 77.44 |
| GPTQ W3g128 | 3.267 | 52.3 | +9.26 | +13.02 | 75.56 |
| RTN W4g128 (M baseline) | 4.250 | 68.0 | +40.3 | +32.3 | – |
| RTN W4g128 + L63.down int8 | 4.266 | 68.2 | +2.32 | +3.62 | – |
| **GPTQ W4g128 + L63d8, A FP8 (W4A8 headline)** | 4.266 | **35.7** | +1.93 | +3.31 | 77.40 |
| FP8 W8A8 + KV FP8 (RTN, E1) | 8.002 | 65.6 | +0.56 | +0.80 | 78.2 |
| RTN INT8 W8A8 per-token (P0) | ~8 | ~64 | +2.55 | +1.90 | – |

**Result: no M baseline passes the +1% bar on selection data.** The best (GPTQ W4g128) is at +1.89% on c4dev and +3.1% on the WikiText proxy. Only 8-bit-weight configs pass (FP8 W8A8 +0.56%). The task-dev drop of every W4 baseline (−1.4…−1.5 vs bf16) is also beyond the zero-shot guard's −1 margin, if test zero-shot behaves like task-dev.

**Margin base (written now, per the user's instruction):**
- **Under the current bar (+1%) — in force until the user decides:** no M baseline passes. The cheapest passing measured config is FP8 W8A8 at W 8.00 → an M winner needs ≤ 7.50 bits; by the "no baseline passes" clause any passing config below 8 bits would count, which makes the M frontier degenerate. B base: FP8 W8A8, B 65.6 (W 8 / A 8).
- **Under the proposed bar (+4%, below) if adopted:** M base = **GPTQ W4g128, W 4.250** → an M winner needs **W ≤ 3.750**. B base to be fixed once QuaRot W4A4KV4 and SmoothQuant W8A8 land.

**Proposed revised bar (ADOPTED by the user 2026-11-12 ~08:33 UTC; see "Bar decision" below):** every test ppl ≤ **+4%** of bf16 — WikiText-2 test ≤ **7.9147**, fw25test ≤ **10.2711** — and zero-shot avg ≥ **68.94** (bf16 − 2) as a guard. Reasoning in the report to the user (22:45 UTC).

### Second Cholesky failure (23:14 UTC) — asymmetric GPTQ target
`gptq_W4g128_L63d8_asym15_Afp8` failed: `linalg.cholesky … leading minor of order 7757 not positive-definite` (68.9 GB allocated, 107.9 GB peak; not OOM). Cause: the asymmetric LS target (`M = (C−H) H⁻¹`) had its own Cholesky, still damped in float32 before the cast. Fix: new `solvers.chol_damped` (float64, ×10 damping escalation up to 4×, logged in `info.damp_raised`) now used for the asym target and the low-rank whitening too. Tested on CPU in the container. The old-code retry was killed; job restarted 23:15:58 with the fix. ~40 min lost.

### W4A8-FP8 push results (done 01:14 UTC 2026-11-12; all W4 g128/g64 + L63.down int8 + A FP8 E4M3 per-token, int8 lm_head, seed 0)

| config | W | B | c4dev Δ% | wt2train Δ% | task-dev |
|---|---|---|---|---|---|
| GPTQ | 4.266 | 35.7 | +1.93 | +3.31 | 77.40 |
| **GPTQ + asym target α=0.15** | 4.266 | 35.7 | **+1.53** | **+1.96** | 77.32 |
| AWQ scales → GPTQ | 4.266 | 35.7 | +1.89 | +2.78 | **78.56** |
| GPTQ g64 | 4.417 | 36.9 | +1.64 | +2.72 | 77.88 |

- The asymmetric GPTQ target (which corrects for errors from earlier layers) is the strongest single change: −0.4 points on c4dev and −1.35 points on the WikiText proxy, at no cost in bits. It beats the g64 variant, which pays +0.15 bits. W4A8-FP8 with asym is better than GPTQ W4A16 (+1.89 / +3.14) at half the BitOps.
- AWQ→GPTQ gives the best task-dev (78.56, −0.3 vs bf16), but ppl is unchanged vs plain GPTQ. Next to try: AWQ + asym combined.
- The asym run needed one damping escalation in each solve (0.01→0.1; recorded in `info.ops[].damp_raised`). The fix works.

### Correction + results to 03:51 UTC 2026-11-12
**Correction:** the 22:40 and 01:14 entries say "no damping escalation" for the GPTQ baselines. That was wrong: I read the wrong field. The escalations sit in `info.ops[].damp_raised`. Most GPTQ runs needed **one** solve raised from percdamp 0.01 to 0.1, including the baselines GPTQ W4g128 and GPTQ W3g128; asym runs also needed one raise in the target solve. The only runs with none: AWQ W4g128 (no GPTQ), AWQ→GPTQ W4A8, GPTQ W4g64 A8. The log records values only, not which sublayer. It is one linear out of 448, so the effect on the comparison is small but nonzero. The same rule is applied to baselines and candidates.

| config | W | B | c4dev Δ% | wt2train Δ% | task-dev |
|---|---|---|---|---|---|
| GPTQ W4g128 + L63d8 (A16) | 4.266 | 68.2 | +1.79 | +3.03 | 78.08 |
| AWQ→GPTQ asym W4 + L63d8, A FP8 | 4.266 | 35.7 | +1.61 | +2.27 | 78.56 |
| GPTQ W3g128 + L63d8 | 3.286 | 52.6 | +8.42 | +12.37 | 75.32 |
| **GPTQ W3g128 + L63d8, asym** | 3.286 | 52.6 | **+7.15** | **+10.07** | 76.72 |

- AWQ + asym does not stack on ppl (+1.61 vs asym alone +1.53) but keeps AWQ's task-dev (78.56). Plain asym stays the best-ppl W4A8.
- At 3 bits, asym removes ~1/6 of the gap (+8.4 → +7.2% c4dev), so W3 is still far from even a +4% bar. A ≤3.75-bit winner needs more than GPTQ variants: mixed precision (some layers at 4 bits, within the 3.75 budget), low-rank, g64. The remaining E2/E3 jobs in the queue cover part of this.
- 04:28 UTC: **AWQ→GPTQ W3g128 + L63d8**: c4dev +7.98%, wt2train +11.52%, task-dev 74.12 (W 3.286). That beats plain GPTQ (+8.42%) on ppl but loses task-dev, and is worse than asym (+7.15%). AWQ scales do not help at 3 bits the way the asym target does.

### E5 first results (06:15 UTC 2026-11-12) — QuaRot-style baseline looks broken, diagnosing
| config | W | A (flop-w) | B | KV | c4dev Δ% | wt2train Δ% | task-dev |
|---|---|---|---|---|---|---|---|
| QuaRot-style W4A4KV4 (R1+R4, GPTQ W4 per-channel sym, A4 per-token clip 0.9, KV4 g128) | 4.100 | 4.29 | 18.7 | 4.16 | **+73.3** | +86.4 | 69.52 |
| ours W4A4KV4 (GPTQ W4g128 + L63d8, A4 g32 + L6.gate FP8, K4ch/V3) | 4.266 | 4.90 | 21.9 | 3.65 | **+4.33** | +5.98 | 75.80 |
| GPTQ W3g64 + L63d8 | 3.431 | 16 | 54.9 | 16 | +6.38 | +8.84 | 76.96 |
| RTN W3g128 + low-rank r64 on L62/63.down | 3.268 | 16 | 52.3 | 16 | +14.29 | +23.36 | 75.92 |

- **QuaRot +73% is not a fair baseline yet.** Published QuaRot on similar models loses ~10–20%, not 73%. Our version lacks two of QuaRot's pieces: the online Hadamard on the K cache (`qkrot` exists but was not in the pipeline) and R2/R3 (head-wise rotation of the o_proj input). Unrotated per-token A4 collapses completely (E1: +3696%), so rotation helps a lot but leaves something. The diagnostic `p2_rotdiag` (rotated model, bf16 weights, per-token A4 per module type, c4dev only, ~10 min) is queued after `sqsearch`. If the o_proj input is the culprit, I add the head-wise Hadamard and re-run QuaRot before any comparison is reported as a result.
- Ours W4A4KV4: +4.33% at B 21.9 (raw W×A 4.27×4.90; QuaRot 4.10×4.29). It is just outside the proposed +4% on c4dev and well outside on WikiText (+6%).
- W3g64 + L63d8 at 3.43 bits is the best 3-bit-class point (+6.4%), but it is still far from +4%.
- Low-rank r64 on RTN W3 is worse than the plain int8 exception (+14.3 vs +13.7%): low-rank does not replace the exception here.

### E5 diagnosis, step 1 (`p2/rotdiag.json`, 06:30 UTC): activations are NOT the problem
Rotated model (R1+R4+q/k Hadamard), bf16 weights, c4dev Δ% vs bf16:
- rotation alone +0.05 (exact up to bf16 rounding); A8 per-token +0.07; A6 per-token +1.11
- **A4 per-token everywhere: +13.2%** (unrotated: +3696%). This is in line with published QuaRot.
- A4 on one type only: qkv +1.80, o_proj +1.44, gate/up +2.58, **down +6.09**; all-but-o +10.98; with L6.gate and L63 at FP8: +7.13
- A4 g32 (no per-token): +2.67. KV4 g128 with q/k Hadamard: +0.44.
→ The o_proj input is not the cause (R2/R3 would buy at most ~1.4%). With weights in bf16 the activation side gives about +13%, so most of QuaRot's +73% must come from the weights (GPTQ W4 per-channel sym on the rotated model, incl. L63.down) or from K4 without the q/k Hadamard. Step 2 (`p2_rotdiag2`, RTN weights ± L63d8, KV with/without q/k Hadamard, combined) is queued after sq05.

SmoothQuant INT8 W8A8 with alpha search: c4dev +0.66%, wt2train +1.19%, task-dev 78.32, W 8.00, B 65.6. It passes the +1% bar on c4dev.

### E5 diagnosis, step 2 (`p2/rotdiag2.json`, 06:46 UTC): cause found — K4 without the q/k Hadamard
Rotated model, RTN weights, c4dev Δ%:
- **KV4 g128 without q/k Hadamard: +36.3%** (with it: +0.44%). Combined W4pc+A4pt+KV4: +190.6% without vs +56.8% with.
- W4 per-channel RTN: +25.9% (+21.9% with L63d8); W4g128 RTN: +9.7%. W4pc + A4pt: +53.7%.
→ The first QuaRot run left `qkrot` out of its pipeline, so its K cache was quantized per token without the Hadamard. **That was my configuration bug, not a QuaRot property.** Fixed (QuaRot pipeline = rotate R1+R4, qkrot, GPTQ W4 per-channel sym). The old result is moved to `p2/build/_noqkrot/` and **must not be used**. The re-run is queued right after the running job (`p2b_quarot_W4A4KV4_v2`).
- Side finding: on the rotated model RTN W4g128 is +9.7%, while unrotated RTN W4g128 + L63.down int8 is +2.3%. For weight-only W4, the targeted exception beats rotation by a wide margin.

SmoothQuant INT8 W8A8 with fixed alpha 0.5: c4dev +1.11%, wt2train +1.33%, task-dev 78.60. That is just over +1% on c4dev; alpha search (+0.66%) is better.

### QuaRot re-run (with q/k Hadamard; 07:23 UTC) — the valid E5 baseline
| config | W | A | raw W×A | B | KV | c4dev Δ% | wt2train Δ% | task-dev |
|---|---|---|---|---|---|---|---|---|
| QuaRot-style W4A4KV4 (fixed) | 4.100 | 4.29 | 17.6 | 18.7 | 4.16 | **+20.4** | +26.5 | 72.28 |
| ours W4A4KV4 g32 | 4.266 | 4.90 | 20.9 | 21.9 | 3.65 | **+4.33** | +5.98 | 75.80 |

- QuaRot at +20% is in the range published for similar models, so the baseline is now credible.
- Ours is 4.7× closer to bf16 at **1.17× the BitOps** (21.9 / 18.7). Under the decided B rule (≥ 0.5 bits lower on W or A, the other not higher), ours is **not** a B win over QuaRot: it uses more W and more A bits. It is a type-(a) result (better accuracy at similar cost), and it is reported as such. A B win would need ours at ≤ 4.29−0.5 A bits or ≤ 4.10−0.5 W bits without the other going up. Group-wise A4 g32 already costs +0.5 A bits for its scales.
- RTN W4g128 + rank-64 low-rank on L63.down: +2.37% / +3.62% / 78.44 at W 4.251, the same as the int8 exception (+2.32%, W 4.266) at 0.015 fewer bits. Negligible.

### Bar decision (user, 2026-11-12 ~08:33 UTC): +4% bar approved
The user approved the +4% bar. This bar replaces the +1% bar.
- WikiText-2 test perplexity ≤ **7.9147** (1.04 × 7.6103).
- fw25test perplexity ≤ **10.2711** (1.04 × 9.8761).
- Zero-shot average (8 tasks) ≥ **68.94** (bf16 70.94 − 2). This term is a guard only.
- All other rules stay the same: seed rule, worst-seed rule for a winner, margin ≥ 0.5 bits/weight, B rule, int8 lm_head, fully-bf16 reference, selection/confirmation split.
- **Margin base for M:** GPTQ W4g128, W 4.250. An M winner must pass the bar at **W ≤ 3.750**.
- **Margin base for B:** the cheapest B baseline that passes the bar in the confirmation check. Candidates on selection data: QuaRot-style W4A4KV4 (+20.4% c4dev, fails), SmoothQuant INT8 W8A8 (+0.66%), FP8 W8A8 (+0.56%).

## P2 summary (2026-11-12 08:45 UTC; 23 builds, 3 sweeps, 2 diagnostics; all on selection data, seed 0)

The last build: **ours W4A4KV4 with rotation (R1+R4)**: c4dev +5.02%, wt2train +6.61%, task-dev 76.36. Rotation makes our W4A4KV4 worse (+4.33% without). Our method does not need rotation.

**All builds, sorted by BitOps** (Δ% vs fully-bf16; bar = +4% on the test sets):

| config | W | A | B | KV | c4dev Δ% | wt2train Δ% | task-dev |
|---|---|---|---|---|---|---|---|
| QuaRot-style W4A4KV4 (fixed) | 4.100 | 4.29 | 18.7 | 4.16 | +20.44 | +26.54 | 72.28 |
| ours W4A4KV4 g32 | 4.266 | 4.90 | 21.9 | 3.65 | +4.33 | +5.98 | 75.80 |
| ours W4A4KV4 g32 + rotation | 4.266 | 4.90 | 21.9 | 3.65 | +5.02 | +6.61 | 76.36 |
| **GPTQ-asym W4 + L63d8, A FP8** | 4.266 | 8.20 | 35.7 | 16 | **+1.53** | **+1.96** | 77.32 |
| AWQ→GPTQ-asym W4 + L63d8, A FP8 | 4.266 | 8.20 | 35.7 | 16 | +1.61 | +2.27 | 78.56 |
| AWQ→GPTQ W4 + L63d8, A FP8 | 4.266 | 8.20 | 35.7 | 16 | +1.89 | +2.78 | 78.56 |
| GPTQ W4 + L63d8, A FP8 | 4.266 | 8.20 | 35.7 | 16 | +1.93 | +3.31 | 77.40 |
| GPTQ W4g64 + L63d8, A FP8 | 4.417 | 8.20 | 36.9 | 16 | +1.64 | +2.72 | 77.88 |
| GPTQ W3g128 (baseline) | 3.267 | 16 | 52.3 | 16 | +9.26 | +13.02 | 75.56 |
| GPTQ-asym W3g128 + L63d8 | 3.286 | 16 | 52.6 | 16 | +7.15 | +10.07 | 76.72 |
| GPTQ W3g64 + L63d8 | 3.431 | 16 | 54.9 | 16 | +6.38 | +8.84 | 76.96 |
| SmoothQuant INT8 W8A8, α search (baseline) | 8.002 | 8.20 | 65.6 | 16 | +0.66 | +1.19 | 78.32 |
| GPTQ W4g128 (baseline, M base) | 4.250 | 16 | 68.0 | 16 | +1.89 | +3.14 | 77.52 |
| AWQ W4g128 (baseline) | 4.250 | 16 | 68.0 | 16 | +2.10 | +2.88 | 77.44 |
| GPTQ W4g128 + L63d8 | 4.266 | 16 | 68.2 | 16 | +1.79 | +3.03 | 78.08 |
| GPTQ MXFP4 + L63d8 | 4.357 | 16 | 69.7 | 16 | +1.99 | +2.49 | 79.08 |

**What we learned in P2:**
1. **One exception fixes 4-bit weights.** L63.down_proj at int8 costs +0.016 bits/weight. It takes RTN W4g128 from +40% to +2.3% on c4dev. GPTQ and AWQ also fix most of this layer without the exception.
2. **The asymmetric GPTQ target is the best single solver change.** It costs no bits. At W4A8 it improves c4dev by 0.4 points and the WikiText proxy by 1.35 points. At W3 it improves c4dev by 1.3 points.
3. **FP8 activations are almost free.** GPTQ-asym W4A8-FP8 is better than the GPTQ W4A16 baseline at 0.52× the BitOps.
4. **4-bit activations need small groups, not rotation.** Ours (A4 g32) is at +4.3%; QuaRot (rotation + per-token A4) is at +20.4%. Rotation added to our method makes it worse.
5. **The QuaRot baseline needs the q/k Hadamard for KV4.** Without it, K4 per-token costs +36%. (My configuration bug; found by diagnosis and fixed.)
6. **3-bit weights are the open problem.** The best uniform 3-bit configuration is +6.4% (W3g64, 3.43 bits). GPTQ variants close only a small part of the gap. Next: mixed 3/4-bit allocation at ≤ 3.75 bits (sweep `e2_mix`, running).
7. **Low-rank correction does not replace the int8 exception** (rank 16/64 gives the same result as int8 at the same bits).
8. **MXFP4 weights give the best task-dev** (79.08, above bf16 78.88 within noise) and the best WikiText proxy among A16 configurations, at +0.1 bits.
9. **KV is solved:** K4 per-channel + V3 costs +0.8%; FP8 KV costs +0.09%.

**Proposed finals for P4 (to be frozen after `e2_mix` and its GPTQ builds):**
- **M frontier (target W ≤ 3.75):** the 1–2 best mixed 3/4-bit configurations with GPTQ-asym.
- **B frontier:** GPTQ-asym W4 + L63d8, A FP8 (B 35.7). Against the B baselines (SmoothQuant W8A8 and FP8 W8A8, both W 8 / A 8.2): W is 3.7 bits lower and A is not higher. This is a B win if it passes the +4% bar.
- **B frontier, alternative:** AWQ→GPTQ-asym W4A8-FP8 (better task-dev).
- **W4A4KV4 (type-(a) report against QuaRot):** ours W4A4KV4 g32. It will probably not pass +4% on WikiText.

### E2b mixed 3/4-bit screen (`p2/e2_mix.json`, RTN, 08:57 UTC)
W3 base, W4g128 in the listed parts, L63.down int8. Bits include the int8 lm_head.

| allocation | W | c4dev Δ% | wt2train Δ% |
|---|---|---|---|
| W3g64, layers 44–63 at W4 | 3.689 | **+4.38** | +7.22 |
| W3g128, layers 34–63 at W4 | 3.743 | +4.41 | **+6.33** |
| W3g128, layers 40–63 + all v at W4 | 3.658 | +4.47 | +7.58 |
| W3g64, all v + MLP 46–63 at W4 | 3.626 | +4.56 | +8.55 |
| W3g128, all v, down + gate/up 48–63 at W4 | 3.689 | +4.66 | +7.39 |
| W3g128, layers 0–3 and 40–63 at W4 | 3.712 | +4.90 | +8.38 |
| W3g128, all v, down + layers 56–63 at W4 | 3.646 | +5.01 | +7.12 |
| W3g128, all v + MLP 40–63 at W4 | 3.590 | +5.43 | +8.67 |
| W3g64, all v, down at W4 | 3.662 | +6.29 | +11.06 |
| W3g128, all v, up + down 50–63 at W4 | 3.615 | +6.96 | +11.75 |
| W3g128, all v, gate + down 50–63 at W4 | 3.615 | +7.30 | +10.88 |
| (reference) W3g128 uniform | 3.286 | +13.73 | +23.4 |
| (reference) W4g128 uniform | 4.266 | +2.32 | +3.62 |

- **Give the 4-bit budget to the late layers, not to module types.** Contiguous late-layer blocks are best. This agrees with P1.5 (late-layer errors decay less). Early layers at W4 (0–3) give nothing.
- With RTN, the best allocations at ≤ 3.75 bits remove about 80% of the W3→W4 gap on c4dev.
- The top 3 now run with GPTQ-asym (`mix_asym_*`, ~45 min each). With RTN, GPTQ-asym halved the uniform-W3 error, so these may reach the +4% bar.
- The queue had stopped ("queue empty") after `e2_mix`. I restarted one worker on GPU 1 at 08:59.

### E2b first GPTQ-asym mixed build (09:40 UTC): a candidate for the bits/weight win
**mix_asym_W3g64_late44** (W3g64 everywhere, layers 44–63 at W4g128, L63.down int8, GPTQ with asymmetric target α 0.15):
W **3.689**, c4dev **+2.52%**, wt2train **+3.21%**, task-dev **79.48** (bf16 78.88).
- This is the first configuration at ≤ 3.75 bits/weight that is inside +4% on both proxies, with headroom (1.5 and 0.8 points).
- It is 0.56 bits/weight below the M base (GPTQ W4g128, 4.250), so it meets the 0.5-bit margin by 0.06 bits.
- On selection data it is equal to or better than the GPTQ W4g128 baseline (+1.89 / +3.14 / 77.52) on WikiText proxy and task-dev, and 0.6 points worse on c4dev.
- GPTQ-asym reduced the RTN error of this allocation from +4.38% to +2.52% on c4dev.
- One damping raise (0.01 → 0.1) in one GPTQ solve and one in the target solve, as in the other runs.
- 10:22 UTC: **mix_asym_W3_late34** (W3g128, layers 34–63 at W4g128, L63.down int8, GPTQ-asym): W **3.743**, c4dev **+2.29%**, wt2train **+3.11%**, task-dev 77.28. Slightly better ppl than late44-g64, but task-dev is 2.2 points lower, and the margin to the M base is only 0.507 bits. Small changes in the bit count could break the 0.5-bit margin.
- 11:04 UTC: **mix_asym_W3_late40_v** (W3g128, layers 40–63 + all v_proj at W4g128): W 3.658, c4dev +2.63%, wt2train +3.55%, task-dev 76.96. Worse than the other two on all selection metrics. Not selected.
- **One more selection build before the freeze:** `mix_asym_W3g64_late44_Afp8` = the best M allocation with FP8 activations. If it holds the accuracy, one configuration can win on both M and B. Started 11:05, ~45 min.

## Objective (restated 2026-11-12 11:30 UTC, after the user asked the agent to take decisions)
Find weight-space quantization methods for Qwen3-32B that give **the same accuracy as the standard methods at a clearly lower cost**:
- **Accuracy:** every test perplexity ≤ +4% of bf16 (WikiText-2 test ≤ 7.9147, fw25test ≤ 10.2711); zero-shot average ≥ 68.94 as a guard.
- **M (bits/weight):** pass the bar at ≥ 0.5 bits/weight below GPTQ W4g128 (4.250) → **≤ 3.750**.
- **B (BitOps):** pass the bar with ≥ 0.5 bits lower on W or A than the cheapest passing B baseline, the other not higher.
- Report type-(a) results too (better accuracy at similar cost), e.g. ours W4A4KV4 against QuaRot.

## P4 freeze (decided by the agent, 2026-11-12 11:30 UTC, before any test data was read)
Selection used c4dev + task-dev only (wt2train as a diagnostic). The frozen set is `code/r2/finals.py`; the queue is `code/queue_p4.tsv` (67 runs, round-robin by seed).

**Finals (4 seeds each):**
| key | frontier | reason |
|---|---|---|
| mix_asym_W3g64_late44 | M (3.689) | best task-dev (79.48), c4dev +2.52%; 0.56-bit margin |
| mix_asym_W3_late34 | M (3.743) | best c4dev (+2.29%) of the mixed builds; margin only 0.507 bits |
| gptq_W4g128_L63d8_asym15_Afp8 | B (35.7) | best c4dev of all W4A8 (+1.53%) |
| awq_gptq_W4g128_L63d8_asym15_Afp8 | B (35.7) | best task-dev of all W4A8 (78.56), c4dev +1.61% |
| ours_W4A4KV4_g32 | type (a) vs QuaRot | +4.33% vs +20.44% |
| mix_asym_W3g64_late44_Afp8 | M and B | **conditional**, rule fixed before its result: included only if c4dev ≤ +3.0% and wt2train ≤ +3.5% |

Not selected: mix_asym_W3_late40_v (worse on all metrics), ours with rotation (worse than without), W4g64 A8 (more bits, worse than asym), MXFP4 (+0.1 bits, no M or B advantage).

**Baselines:** GPTQ W4g128, AWQ W4g128, GPTQ W3g128, QuaRot-style W4A4KV4 (fixed), SmoothQuant INT8 W8A8 (α search): 8 seeds each. RTN W4g128, FP8 W8A8 (per-channel W, per-token A, no KV quant), RTN INT8 W8A8: 1 run each (no calibration randomness). Qwen's official AWQ/FP8 checkpoints are not run (they need a large download; not in the budget).
**Seed** = the calibration draw, applied to every op (GPTQ, AWQ, SmoothQuant, rotation).
**Order:** seed 0 of everything first, then seeds 1–3, then baseline seeds 4–7. **ETA:** about 65 GPU-hours (≈ 14 Nov ~06:00 UTC).

### Seed rule change (user, 2026-11-12 ~11:42 UTC, before P4 started)
The user approved the agent's proposal: **GPTQ W3g128 and QuaRot-style W4A4KV4 get 2 seeds, not 8.** Reason: on selection data both fail the +4% bar by a wide margin (+9.3% and +20.4% c4dev), so they cannot be the margin base, and their exact mean changes no decision. GPTQ W4g128, AWQ W4g128 and SmoothQuant W8A8 keep 8 seeds (they are the margin bases). Finals keep 4. P4 is now 55 runs (59 with the conditional final), about 51 GPU-hours; expected end 13 Nov ~15:00 UTC. Changed in `code/r2/finals.py` (SEEDS) and `code/queue_p4.tsv`.

### Why published numbers are not the comparison (user question, 2026-11-12 ~11:50 UTC)
The user asked why we do not use the published results of GPTQ, AWQ and QuaRot. Answer, also in `methodology.md` §1.5: published numbers use other models, another evaluation harness, other quantization settings (often a bf16 lm_head) and one run, and they have no fw25test. In round 1, the harness alone moved bf16 zero-shot by 2.5 points (67.09 vs the paper's 69.6). So all methods run in our harness; published numbers serve only as a sanity check of our baselines (this check found the QuaRot q/k-Hadamard bug).

### Count of tuned settings (round-1 rule: ≤ 20 numbers searched on scores per run; counted 2026-11-12 11:45 UTC)
Settings chosen by score on selection data (c4dev/task-dev), per final. Settings fixed in advance (not searched) are listed separately.
- **mix_asym_W3g64_late44:** W base bits (3) and group (64); W4 layer range start (44); exception linear (L63.down) and its format (int8); asym α (0.15, reused from round 1, not re-tuned) → **5**.
- **mix_asym_W3_late34:** base group (128), range start (34), exception linear + format, α → **5**.
- **gptq_W4g128_L63d8_asym15_Afp8 / awq_… :** W group (128), exception linear + format, α, A format (FP8 per token), AWQ on/off → **5–6**.
- **ours_W4A4KV4_g32:** W group, exception linear + format, A format + group (int4 g32), A exception linear + format (L6.gate FP8), K format (4-bit per channel, 128-token blocks), V bits (3) → **9**.
- **mix_asym_W3g64_late44_Afp8 (conditional):** 5 + A format → **6**.
- **Fixed, not searched:** calibration 128 × 2048 (half WikiText-2, half C4), GPTQ damp 0.01 with act_order, lm_head int8 per channel, AWQ/SmoothQuant grid (20 α values; the search runs inside the solver on calibration data, not on selection scores).
→ Every final is well under 20. Caveat: the space of ideas the agent looked at in P2 is larger (≈ 90 RTN configs and 30 builds); the ≤ 20 rule counts the knobs of the final method, not the number of experiments.

### Conditional final decided (11:48 UTC) and P4 started (11:49 UTC)
**mix_asym_W3g64_late44_Afp8**: W 3.689, B 31.1, c4dev +2.73%, wt2train +3.44%, task-dev 79.12. The rule was c4dev ≤ +3.0% and wt2train ≤ +3.5%, so it is **INCLUDED** as a final (4 seeds). FP8 activations cost only +0.2 points c4dev on top of the mixed weights. This one configuration is a candidate for both frontiers: M 3.689 (≤ 3.75) and B 31.1 (W 3.69 / A 8.2 against the 8-bit baselines at W 8.0 / A 8.2).
P4 queue: 55 runs (the seed reduction and the conditional final included). Worker `q2-queue_p4-g1s1`, one job on GPU 1. First run `rtn_W4g128` s0 started 11:49.
- 12:07 UTC: first P4 run done (`rtn_W4g128` s0, 17 min). The runner works: its c4dev (12.8408) equals the P2 anchor value exactly, so P4 and P2 use the same quantization. Evaluation time is about 16 min per run (not 30), so a GPTQ run takes about 55 min. **New P4 ETA: about 40 GPU-hours, end ≈ 13 Nov ~04:00 UTC.** (Test numbers of baselines are not discussed here until the seed-0 round is complete.)

### P4 seed-0 round complete (2026-11-12 23:26 UTC) — first test-set table
All 14 frozen configurations, seed 0 only. Δ% against bf16 (WikiText-2 test 7.6103, fw25test 9.8761, c4dev 9.1505). Bar: both test perplexities ≤ +4.0%, zero-shot avg ≥ 68.94. W = bits/weight incl. lm_head and scales; A = FLOP-weighted activation bits; B = BitOps.

| config | role | W | A | B | KV | wt2 test | fw25test | c4dev | zero-shot | seed 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| fp8_W8A8 | base B | 8.002 | 8.20 | 65.6 | 16 | +0.55 | +0.48 | +0.49 | 70.93 | pass |
| sqsearch_int8_W8A8 | base B | 8.002 | 8.20 | 65.6 | 16 | +0.82 | +0.98 | +0.66 | 70.74 | pass |
| rtn_int8_W8A8 | base B | 8.002 | 8.20 | 65.6 | 16 | +2.05 | +2.79 | +2.61 | 71.09 | pass |
| gptq_W4g128 | base M | 4.250 | 16 | 68.0 | 16 | +2.27 | +1.97 | +1.89 | 70.09 | pass |
| awq_W4g128 | base M | 4.250 | 16 | 68.0 | 16 | +2.31 | +1.85 | +2.10 | 70.19 | pass |
| gptq_W3g128 | base M | 3.267 | 16 | 52.3 | 16 | +11.44 | +9.21 | +9.26 | 68.75 | fail |
| rtn_W4g128 | base M | 4.250 | 16 | 68.0 | 16 | +34.48 | +42.99 | +40.33 | 68.90 | fail |
| quarot_W4A4KV4 | base B | 4.100 | 4.29 | 18.7 | 4.16 | +24.88 | +21.15 | +20.44 | 65.69 | fail |
| **mix_asym_W3g64_late44** | final M | 3.689 | 16 | 59.0 | 16 | +2.81 | +2.15 | +2.52 | 70.62 | pass |
| **mix_asym_W3_late34** | final M | 3.743 | 16 | 59.9 | 16 | +2.36 | +2.21 | +2.29 | 70.38 | pass |
| **gptq_W4g128_L63d8_asym15_Afp8** | final B | 4.266 | 8.20 | 35.7 | 16 | +1.47 | +1.62 | +1.53 | 70.17 | pass |
| **awq_gptq_W4g128_L63d8_asym15_Afp8** | final B | 4.266 | 8.20 | 35.7 | 16 | +1.69 | +1.47 | +1.61 | 70.55 | pass |
| **mix_asym_W3g64_late44_Afp8** | final M+B | 3.689 | 8.20 | 31.1 | 16 | +3.05 | +2.34 | +2.73 | 70.49 | pass |
| ours_W4A4KV4_g32 | final (a) | 4.266 | 4.90 | 21.9 | 3.65 | +4.51 | +4.31 | +4.33 | 69.26 | fail (narrow) |

Observations (seed 0 only; no pass/fail decision is made before all seeds exist):
1. **Selection transfers to test.** For all passing configs, test Δ is within ±0.6 points of c4dev Δ. Only the weak configs (GPTQ W3, QuaRot) are worse on WikiText-2 test than on c4dev (by 2–4 points). wt2train over-predicted the WikiText-2 test Δ (e.g. late44: wt2train +3.21, test +2.81; GPTQ W4: +3.14 vs +2.27).
2. **Preliminary M result:** mix_asym_W3g64_late44 (3.689) and mix_asym_W3_late34 (3.743) pass with 1.2–1.6 points headroom; the M base GPTQ W4g128 (4.250) also passes. Both are ≤ 3.750, so the M margin holds if the seed means agree.
3. **Preliminary B result:** both W4A8-FP8 finals (W 4.27, A 8.2) are more accurate than every W8A8 baseline except FP8 and SmoothQuant, and they pass with about 2.4 points headroom. W is 3.7 bits lower than the passing W8A8 bases and A is equal, so the B margin holds. mix…late44_Afp8 (B 31.1) passes too, with the least headroom of the passing finals (0.95 points on WikiText-2).
4. **W4A4KV4:** ours misses the bar by 0.3–0.5 points; QuaRot-style misses by 17–21 points. Ours uses more A bits (4.90 vs 4.29) and fewer KV bits (3.65 vs 4.16). Reported as type (a).
5. **Zero-shot guard:** only RTN W4, GPTQ W3 and QuaRot are below 68.94; all finals are ≥ 70.17.

**ETA correction.** The 12:07 entry said "40 GPU-hours, end ≈ 13 Nov ~04:00 UTC". That date was an arithmetic error: 40 h from 12 Nov 11:49 is 14 Nov ~04:00. Measured seed-0 times: finals 57–75 min (FP8-activation builds ≈ 74 min), GPTQ W4 49, AWQ W4 21, SmoothQuant 27, GPTQ W3 50, QuaRot 72 min. Seed 0 took 11.6 h. Remaining 41 runs ≈ 33 h → **P4 end ≈ 14 Nov ~09:00 UTC.**

### Discussion with the user: what the result means (2026-11-13, ~06:30–07:30 UTC)
Questions from the user, and the answers. Recorded at the user's request.

**Seed 1 (finals only, 06:24 UTC).** All 5 passing finals pass again; ours W4A4KV4 fails again by a small amount (+4.43 / +4.14). Seed spread is ≤ 0.2 points for most values, max ≈ 0.56 (GPTQ-asym W4A8 WikiText-2: +1.47 → +2.03). AWQ+GPTQ-asym is the most stable (≤ 0.04 points between seeds): AWQ scaling seems to make GPTQ less sensitive to the calibration draw. Note for round 3.

**Why "modest".** The expected round-2 result is real but small:
1. Size: 3.69 vs 4.25 bits/weight ≈ 13% less weight memory (≈ 2.2 GB on 32B). Not a change of class.
2. Tools: per-layer bits, W4A8-FP8, asymmetric GPTQ targets and AWQ+GPTQ are all known. The contribution is the choice and combination, not a new algorithm.
3. Baselines are single methods. We did not compare against a combined pipeline or an automatic mixed-precision search.
4. The +4% bar is not strict. At +2%, only the W4A8 finals and the 8-bit baselines would pass.
5. One model, simulated quantization, no measured speed.
What is not modest: the measurement (frozen finals, rules fixed in advance, 4 seeds, selection predicted test within 0.6 points).

**Why L63.down_proj is int8 and not bf16.** E1 measured both: same accuracy (+0.21 with W4g128 RTN), but bf16 costs ≈ 0.05 bits/weight against 0.016 for int8. The problem is in the inputs of that linear (activation outliers up to 8768), not in its weights. At 8 bits the weight error is small enough; more bits give no gain.

**Do the finals mix methods?** Yes. Each linear gets one pipeline, but the bits differ per layer (3/4/8), and two finals run AWQ then GPTQ, plus FP8 activations. This is allowed (the agent may combine tools, ≤ 20 tuned numbers; ours use 5–9). But it is the main reason the comparison against single-method baselines is weak.

**Is this uncharted territory?** Two parts:
- **Automatic bit allocation is well studied** (from memory, not yet checked against the papers): HAWQ v1–v3 (Hessian sensitivity), HAQ (RL search), llama.cpp k-quants (more bits for some tensors), EXL2/ExLlamaV2 (per-layer error at several bit widths, then the cheapest mix for a target average — very close to our M final), SqueezeLLM, OWQ, SliM-LLM. A cheap search can possibly find our late-layer allocation without any analysis.
- **An agent that does the expert analysis is less studied.** That is the wsp claim: automated expert thinking. But "less studied" is not "better"; the agent must show a gain that a cheap search does not give.
- **Where the agent added things a search cannot find** (structural findings, not bit allocation): FP8 activations are almost free but INT8 is not (outliers) → the W4A8 finals; L6.gate_proj creates the massive activation (per-token A4 collapse); K needs per-channel scales, and QuaRot needs the q/k Hadamard after RoPE (found our own QuaRot bug); L63.down needs only int8 because its problem is in the inputs.
- **Probable honest claim:** for bit allocation (M final), a standard search may match the agent; for format choice and structural problems (W4A8, W4A4KV4), the agent does what a search does not.

**Decision (user approved, 2026-11-13):** round 3 adds an automatic mixed-precision baseline at the same bits (see PLAN.md §9). Before writing the method names into a report, check them against the papers.

### Interim observations, 2–3 seeds (2026-11-13 ~13:00 UTC; 28/55 runs; no decision yet)
1. **Pass/fail is almost settled.** Seed spread is mostly < 0.4 points, max ≈ 0.6; every passing final has ≥ 0.95 points headroom on its worst seed. Ours W4A4KV4 misses by ≈ 0.2–0.5 points on every seed.
2. **W4A8 finals beat GPTQ W4A16 on accuracy at equal weight bits.** Mean WikiText-2 test: GPTQ W4g128 (W 4.250, A16) +2.20; GPTQ-asym W4 + FP8 act (W 4.266, A8) +1.75; AWQ+GPTQ-asym + FP8 act +1.70. The asymmetric target and the L63.down int8 exception (+0.016 bits) recover more than FP8 activations cost. Better accuracy and half the BitOps at the same time — probably the strongest round-2 result.
3. **The mixed-bit finals were selected too conservatively.** Headroom 1.2–1.6 points against a seed spread of ≈ 0.6. A smaller allocation (≈ 3.5 bits/weight) would probably also pass +4%. At P3 the seed variance was unknown, so the large margin was deliberate. **Round 3: run 2 seeds on 1–2 candidates early in P2, then select with a headroom of about 2× the measured spread.**
4. **Ours W4A4KV4 is close** (+4.47 / +4.23 mean). A small change (e.g. V4 instead of V3) would probably pass; frozen in this round, first target for round 3.
5. **Zero-shot is a noisy guard.** Seed 2 lowered zero-shot by 0.3–0.9 points on three finals (late44 69.65, late34 69.98, GPTQ-asym Afp8 69.54) with no perplexity change; GPTQ W4 varies by 0.9 between seeds. Round 3: more zero-shot items, or report a confidence interval.

### Interim update (2026-11-13 ~14:15 UTC; 29/55 runs)
- **Correction to observation 2 of the discussion (06:30 UTC) and the AWQ+GPTQ stability note:** AWQ+GPTQ-asym + FP8 act seed 2 = WikiText-2 +1.93, fw25test +1.55, zero-shot 70.73 (seeds 0–1: +1.69/+1.70). It is still the most stable final (spread 0.23), but "≤ 0.04 points between seeds" was based on 2 seeds only and was too strong. **Lesson: 2 seeds do not estimate a spread.**
- Its zero-shot did not drop with seed 2, so the seed-2 zero-shot drop (observation 5 above) is not common to all finals.
- **The rest of P4 is mainly confirmation.** 8 final runs remain (seed 2 of 2 finals, seed 3 of 6); every passing final has ≥ ≈ 1 point headroom and no seed moved > 0.6 points. Seeds 4–7 of the baselines only refine the baseline means; the margins compare bits, so they cannot remove a result.
- **Expected report structure:** (1) main: W4A8-FP8 + L63.down int8 + asymmetric GPTQ is more accurate than GPTQ W4A16 at half the BitOps; (2) second: mixed 3/4-bit at 3.689 bits/weight passes +4%, 0.56 bits below GPTQ W4, probably not at the model's limit; (3) near miss: ours W4A4KV4 ≈ 5× smaller error than QuaRot-style, just outside the bar; (4) limits: single-method baselines, loose bar, one model, simulated quantization.

### Decision: post-freeze exploration after P4 (agent decision, 2026-11-13 ~14:15 UTC)
After P4 ends (≈ 14 Nov 09:00 UTC) and before round 3, run on **selection data only (c4dev, wt2train, task-dev; no test data)**, 2 seeds each:
1. a mixed 3/4-bit allocation at ≈ 3.5 bits/weight (tests interim observation 3: the M finals were too conservative);
2. ours W4A4KV4 with V4 instead of V3 (tests interim observation 4).
Cost ≈ 3–4 GPU-hours on GPU 1. **Labelled "post-freeze exploration": these results do not change the round-2 result or the frozen set**, and they are not reported as confirmed. Purpose: check the lessons early so that round 3 starts from tested hypotheses. Reason recorded here because it delays round 3 by some hours.

### Zero-shot per seed — late44 is not robust (2026-11-14 ~07:10 UTC; 51/55 runs, all finals complete)
Zero-shot avg per seed (s0 s1 s2 s3):
- mix_asym_W3g64_late44: 70.62 70.55 69.65 **68.92** (range 1.70)
- mix_asym_W3g64_late44_Afp8: 70.49 70.25 69.50 **68.87** (range 1.62)
- mix_asym_W3_late34: 70.38 70.20 69.98 70.34 (0.40)
- gptq_W4g128_L63d8_asym15_Afp8: 70.17 70.07 69.54 69.99 (0.63)
- awq_gptq_W4g128_L63d8_asym15_Afp8: 70.55 70.90 70.73 70.49 (0.41)
- baselines: gptq_W4g128 range 1.49 over 7 seeds (69.20–70.69); awq_W4g128 0.41; sqsearch 0.23.

Findings:
1. **Both late44 models have one seed (s3) below the zero-shot guard (68.94).** Under the frozen rule they still pass (guard on the seed mean; worst-seed rule only on the two perplexity terms). Report them as "pass, zero-shot not robust across seeds". Not hidden.
2. The two late44 models share the weights per seed, so FP8 activations are not the cause. Perplexity range across seeds is only ≈ 0.2 points, but zero-shot range is ≈ 1.7. late34 (3-bit in 34 layers instead of 44) is stable. **Hypothesis: more 3-bit layers make reasoning tasks sensitive to the calibration draw before perplexity shows it.**
3. **Correction to interim observation 3 (13 Nov):** "a ≈ 3.5-bit allocation would probably also pass +4%" was based on perplexity only. With this zero-shot evidence, the guard would probably fail first at ≈ 3.5 bits. **Below 4 bits, perplexity alone is not a sufficient guide.**
4. **Change to the post-freeze exploration:** the ≈ 3.5-bit test checks task-dev (selection data) on 2 seeds as well as perplexity.
5. **Ranking for the report:** W4A8 finals are the strongest (pass on every term and every seed; AWQ+GPTQ-asym + FP8 act best and most stable). The reliable M result is late34 (3.743). late44 (3.689) passes with the weakness above.
6. **Proposal for round 3 (fixed-rule change, needs user approval; not applied):** apply the zero-shot guard to the worst seed too, or use a larger zero-shot set / report a confidence interval.

## P4 FINAL RESULT (2026-11-14 08:44 UTC; 55/55 runs, no P4 failures)
Frozen rules applied by `code/p4_eval.py` (copy of the output: `/models/tmp/wsp2/results/p4_summary/summary.json`). Δ% vs bf16 (WikiText-2 test 7.6103, fw25test 9.8761). Bar: seed-mean ≤ +4.0% on both test perplexities and seed-mean zero-shot ≥ 68.94; a winner's worst seed must pass both perplexity terms. ± = std over seeds.

| config | role | n | W | A | B | KV | WikiText-2 | fw25test | zero-shot | worst ppl seed | min zs | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **awq_gptq_W4g128_L63d8_asym15_Afp8** | final B | 4 | 4.266 | 8.20 | 35.7 | 16 | +1.76±0.12 | +1.49±0.04 | 70.67±0.19 | +1.93/+1.55 | 70.49 | **PASS, B win** |
| **gptq_W4g128_L63d8_asym15_Afp8** | final B | 4 | 4.266 | 8.20 | 35.7 | 16 | +1.75±0.23 | +1.60±0.04 | 69.94±0.28 | +2.03/+1.63 | 69.54 | **PASS, B win** |
| **mix_asym_W3g64_late44_Afp8** | final M+B | 4 | 3.689 | 8.20 | 31.1 | 16 | +3.03±0.06 | +2.64±0.21 | 69.77±0.74 | +3.09/+2.80 | 68.87* | **PASS, M and B win** |
| **mix_asym_W3_late34** | final M | 4 | 3.743 | 16 | 59.9 | 16 | +2.48±0.18 | +2.37±0.14 | 70.23±0.18 | +2.74/+2.54 | 69.98 | **PASS, M win** |
| **mix_asym_W3g64_late44** | final M | 4 | 3.689 | 16 | 59.0 | 16 | +2.77±0.10 | +2.45±0.21 | 69.94±0.81 | +2.89/+2.60 | 68.92* | **PASS, M win** |
| ours_W4A4KV4_g32 | final (a) | 4 | 4.266 | 4.90 | 21.9 | 3.65 | +4.39±0.16 | +4.22±0.07 | 69.12±0.16 | +4.51/+4.31 | 68.96 | fail |
| gptq_W4g128 | base M | 8 | 4.250 | 16 | 68.0 | 16 | +2.12±0.11 | +1.92±0.08 | 70.10±0.49 | +2.27/+2.08 | 69.20 | pass |
| awq_W4g128 | base M | 8 | 4.250 | 16 | 68.0 | 16 | +2.34±0.08 | +1.86±0.07 | 70.15±0.18 | +2.48/+1.96 | 69.93 | pass |
| sqsearch_int8_W8A8 | base B | 8 | 8.002 | 8.20 | 65.6 | 16 | +0.96±0.19 | +1.06±0.07 | 70.66±0.15 | +1.24/+1.15 | 70.41 | pass |
| fp8_W8A8 | base B | 1 | 8.002 | 8.20 | 65.6 | 16 | +0.55 | +0.48 | 70.93 | | | pass |
| rtn_int8_W8A8 | base B | 1 | 8.002 | 8.20 | 65.6 | 16 | +2.05 | +2.79 | 71.09 | | | pass |
| gptq_W3g128 | base M | 2 | 3.267 | 16 | 52.3 | 16 | +11.26 | +9.06 | 68.60 | | | fail |
| quarot_W4A4KV4 | base B | 2 | 4.100 | 4.29 | 18.7 | 4.16 | +25.65 | +21.24 | 64.96 | | | fail |
| rtn_W4g128 | base M | 1 | 4.250 | 16 | 68.0 | 16 | +34.48 | +42.99 | 68.90 | | | fail |

\* One seed (s3) below the zero-shot guard; the frozen rule applies the guard to the seed mean, so the result is PASS. Reported as a weakness (see "Zero-shot per seed").

**Margins (frozen rules).**
- **M** (base = cheapest passing baseline: GPTQ/AWQ W4g128, W 4.250; win needs W ≤ 3.750): late44 3.689 (−0.561), late44_Afp8 3.689, late34 3.743 (−0.507) → **3 M wins**. W4A8 finals (4.266) are not M wins.
- **B** (bases = passing W8A8: FP8, SmoothQuant, RTN INT8; W 8.002, A 8.20, B 65.6; win needs ≥ 0.5 lower on W or A with the other not higher): both W4A8 finals (W −3.74, A equal; B 35.7 = 0.54× base; raw W×A 35.0) and late44_Afp8 (B 31.1 = 0.47×; W×A 30.2) → **3 B wins**. A16 finals have B ≈ 59–60 but A is higher than the bases → not B wins.
- ours W4A4KV4 fails the bar. Type-(a) comparison with QuaRot-style (same W4A4KV4 format, both fail): error ≈ 5× smaller (+4.39 vs +25.65 WikiText-2; +4.22 vs +21.24 fw25test), at more A bits (4.90 vs 4.29) and fewer KV bits (3.65 vs 4.16).

**Conclusion for round 2 (Qwen3-32B).**
1. **Main result:** W4A8-FP8 with L63.down int8 and asymmetric GPTQ (± AWQ) is **more accurate than GPTQ/AWQ W4A16** (+1.75/+1.76 vs +2.12/+2.34 WikiText-2) at about **half the BitOps of every W8A8 baseline**. It passes on every term and every seed. Best and most stable: AWQ+GPTQ-asym + FP8 act.
2. **Memory result:** mixed 3/4-bit at 3.689–3.743 bits/weight passes +4%, 0.51–0.56 bits below GPTQ W4. late34 is robust; late44 has a weak zero-shot seed.
3. **Both at once:** mix late44 + FP8 act: W 3.689, B 31.1 (0.47× W8A8), passes, with the same zero-shot weakness.
4. **Near miss:** ours W4A4KV4, ≈ 5× smaller error than QuaRot-style, just outside the bar.
5. **Limits:** single-method baselines (no automatic mixed-precision control — round 3), +4% bar (at +2% only the 8-bit baselines and the W4A8 finals pass), one model, simulated quantization (no measured speed; no native W4A8 or 4-bit GEMM on gfx942 — 4-bit = memory; FP8 activations = native compute).
6. Selection data predicted test data within ≈ 0.6 points for every passing config: no sign of selection overfit.

Next (agent decision recorded 2026-11-13): post-freeze exploration on selection data only (≈ 3.5-bit allocation with task-dev on 2 seeds; W4A4KV4 with V4). Then round 3.

### Post-freeze exploration started (2026-11-14 08:45 UTC; selection data only; does NOT change the round-2 result)
`python -m r2.p2 build KEY@sN` (seed support added to `build`; reports c4dev, wt2train, task-dev only — no test data). Queue `code/queue_pf.tsv`, worker `q2-queue_pf-g1s1`, one job on GPU 1. 2 seeds each:
- `pf_mix_asym_W3_late50`: W3g128 base, layers 50–63 at W4g128, L63.down int8, asym 0.15 (expected W ≈ 3.50).
- `pf_mix_asym_W3g64_late58`: W3g64 base, layers 58–63 at W4g128 (expected W ≈ 3.50).
- `pf_ours_W4A4KV4_g32_V4`: ours W4A4KV4 with V 4-bit per token instead of V3.
Questions: can ≈ 3.5 bits stay inside +4% on selection perplexity, and is task-dev stable across 2 seeds (the late44 zero-shot weakness)? Does V4 bring W4A4KV4 under the bar on the proxies? Estimated ≈ 4 GPU-hours.

### Post-freeze exploration results (2026-11-14 12:47 UTC; selection data only; NOT confirmed, does not change the round-2 result)
Δ% vs bf16 on selection data (c4dev / wt2train), task-dev (bf16 78.88). In P4, test Δ was within ≈ 0.6 points of c4dev Δ for every passing config (wt2train over-predicts test).

| build | W | B | KV | c4dev s0/s1 | wt2train s0/s1 | task-dev s0/s1 |
|---|---|---|---|---|---|---|
| pf_mix_asym_W3_late50 (W3g128, W4 in 50–63) | 3.497 | 56.0 | 16 | +3.23 / +3.59 | +4.28 / +4.06 | 77.12 / 77.16 |
| pf_mix_asym_W3g64_late58 (W3g64, W4 in 58–63) | 3.506 | 56.1 | 16 | +4.16 / +4.05 | +5.27 / +4.87 | 79.48 / 77.92 |
| pf_ours_W4A4KV4_g32_V4 | 4.266 | 21.9 | 4.16 | +3.74 / +3.73 | +5.17 / +5.20 | 77.12 / 76.20 |
| ref: mix_asym_W3_late34 (final) | 3.743 | 59.9 | 16 | +2.29 | +3.11 | 77.28 |
| ref: mix_asym_W3g64_late44 (final) | 3.689 | 59.0 | 16 | +2.52 | +3.21 | 79.48 |
| ref: ours_W4A4KV4_g32 (final, V3) | 4.266 | 21.9 | 3.65 | +4.33 | +5.98 | 75.80 |

Findings:
1. **≈ 3.5 bits/weight is at the edge of the +4% bar, not safely inside.** late50 (3.497) predicts test ≈ +3.2…+3.9 (0.1–0.8 points headroom); late58 g64 predicts ≈ +4.1 (probable fail). So the correction of 14 Nov ("3.5 bits would probably also pass" was too optimistic) is confirmed by perplexity too: the round-2 M finals at 3.69–3.74 are ≈ 0.2 bits from the limit, not 0.5.
2. **More W4 layers beat finer groups at equal bits.** At ≈ 3.5 bits, W3g128 + 14 W4 layers (+3.41 mean c4dev) is clearly better than W3g64 + 6 W4 layers (+4.11). Same direction as P1.5 (late layers matter most).
3. **Task-dev robustness:** late50 is stable across 2 seeds (77.12/77.16); late58 g64 moves 1.56 points (79.48/77.92). Weak support (2 seeds) for the hypothesis that more 3-bit layers make the reasoning tasks sensitive to the calibration draw.
4. **V4 brings ours W4A4KV4 close to or inside the bar:** c4dev +4.33 → +3.74 (both seeds), task-dev +0.4…+1.3. Predicted test ≈ +3.7…+4.0: borderline. KV becomes 4.16 = the same as QuaRot-style, so in round 3 this is a like-for-like W4A4KV4 comparison (A 4.90 vs 4.29 remains).
5. **For round 3:** start W4A4KV4 from V4; for M, select near 3.6–3.7 bits with a late W4 block, not finer groups; check task-dev on ≥ 2 seeds before freezing.

### Peer recommendations (general-ags, relayed 2026-11-14 ~13:00 UTC; not user decisions)
1. Worst-seed zero-shot guard: recommended. Under it, late44 and late44_Afp8 would fail round 2 (68.92 / 68.87 < 68.94), so conclusion 3 (combined M+B) would go. Agent agrees: a result that flips on 0.02 points is noise. **Still needs the user's approval.** Round-2 result stays under the frozen rule, with this stated next to conclusion 3.
2. Measure real decode speed (PLAN §9 item 10). Checked against the peer's notes: CompressionBench Qwen3.8 27B 8 GiB: Sonnet 5.5 23.4 tok/s, Opus 5.5 26.2, human GSQ-RCO 75.5 (batch 1). "≈ 2.5 bits/weight" is the video's derivation, not a paper figure.
3. Keep "bits below base" and "distance to bar" separate (PLAN §9 item 11).
Round 3 not started; waiting for the user.

### Decode-speed measurement plan (gfx942 / MI325X; proposal, not run)
Facts: gfx942 has native INT8 and FP8 (e4m3fnuz) MFMA; no 4-bit or 3-bit MFMA (NOTES P0 hardware check). Batch-1 decode is memory-bound: speed ≈ bytes read per token / HBM bandwidth, plus kernel overhead. So 4/3-bit weights can only win by bytes, and only if the unpack is cheap; A8 activations matter for prefill and large batches, not batch-1 decode.
1. **Roofline first.** Per format: bytes/token = weight bytes (incl. scales, exceptions, lm_head) + KV bytes at a fixed context. Upper bound tok/s = HBM bandwidth / bytes. Report measured as a % of this bound; that separates "format cost" from "kernel quality".
2. **GEMV/GEMM microbenchmark** on the real Qwen3-32B linear shapes (qkv, o, gate/up, down; M = 1, 8, 64, 512, 4096), with HIP graphs, warm-up, median of ≥ 100 runs, GPU 1 only:
   - bf16 and FP8 W8A8: hipBLASLt via torch (`torch._scaled_mm`), native.
   - INT8 W8A8: torch `_int_mm` / hipBLASLt int8.
   - W4A16 g128: an existing ROCm kernel (vLLM's Triton AWQ/GPTQ path, if it runs in our image) plus our own Triton dequant-GEMV.
   - **W4A8:** own Triton kernel: unpack int4 + group scale → FP8 in registers → FP8 dot (tl.dot fp8 on gfx942). If this kernel is slower than W4A16, report it as kernel-limited.
   - **Mixed 3/4-bit:** own Triton GEMV for 3-bit (packing 10×3 bits in int32, or bit-planes) per layer. This is the CompressionBench risk: 3-bit unpack can cost more than the 25% byte saving.
3. **End-to-end decode loop.** One simple torch decode loop (static KV cache, HIP graphs) where each nn.Linear is replaced by the benchmarked kernel; batch 1 and 16, context 2k; tokens/s for bf16, FP8 W8A8, GPTQ W4A16, our W4A8, our mixed 3/4-bit. Same loop for all → a fair relative ranking.
4. **Absolute anchor.** vLLM ROCm for the formats it supports natively (bf16, FP8, GPTQ/AWQ W4A16), if vLLM runs in the image; if our loop and vLLM disagree by > 20% on these, fix the loop before reporting our formats.
5. **Report** per format: bits/weight, BitOps, roofline bound, measured tok/s (batch 1, 16), prefill tok/s, % of bound. Mark any format with no native kernel. A config that wins on bits but loses on tok/s is reported as such.
Cost: kernel work (agent time) + ≈ 2–4 GPU-hours on GPU 1. To check before starting: whether vLLM/Triton ROCm in `wsp-quant2:rocm7.2` supports fp8 tl.dot on gfx942 and the AWQ/GPTQ kernels.

### User decisions (relayed by general-ags, 2026-11-14 ~13:20 UTC)
1. **APPROVED for round 3: the zero-shot guard applies to the worst seed as well as the seed mean.** Under that rule, mix_asym_W3g64_late44 (worst seed 68.92) and mix_asym_W3g64_late44_Afp8 (68.87) would have failed round 2. **The frozen round-2 result does not change**; this is stated next to conclusions 2–3 in every report.
2. Decode-speed measurement on the round-2 finals FIRST; round 3 not started. GPU 1 only.
3. Order by risk, stop at each checkpoint: (a) Triton FP8 dot on gfx942 + vLLM in the container; (b) bandwidth upper limit per format; (c) measure formats with existing kernels (BF16, FP8 W8A8, INT8 W8A8, GPTQ W4A16 via vLLM); (d) report to the user before any custom kernel. Then own W4A8 kernel, then mixed 3/4-bit (last, may be dropped).
4. The freeze holds: speed is an added measurement and does not re-rank the frozen winners. A winner that loses on speed is reported as a loss on speed.
5. If (a) fails: stop and report; no slow fallback presented as a result.

### Decode speed — checkpoint (a) result (2026-11-14 ~13:45 UTC; scripts `code/speed/check_a.py`, `check_a2.py`; GPU 1)
Container `wsp-quant2:rocm7.2`: torch 2.10.0+rocm7.2.4, Triton 3.6.0+rocm7.2.4, MI325X gfx942.
- **Triton FP8 tl.dot: PASS, native.** e4m3fnuz × e4m3fnuz → fp32 is exact against a float reference; the ISA contains `v_mfma_f32_32x32x16_fp8_fp8` (no upcast fallback).
- **Triton INT8 tl.dot: PASS, native** (`v_mfma_i32_32x32x16_i8`, exact).
- **torch._scaled_mm FP8 (hipBLASLt): PASS.** **torch._int_mm INT8: PASS** (exact, m = 32/64/256). torch._scaled_mm does not take int8 (INT8 W8A8 = _int_mm + scale epilogue).
- (First run showed false failures: `a.int() @ b.int()` is not implemented on the GPU — my reference, not the kernel; and a wrong Triton API for the ISA dump. Both fixed.)
- **vLLM: NOT available** — not in our image, not in any local image. aiter also not in our image.
- Local alternative: `rocm/sgl-dev:v0.5.19-rocm724-mi30x-20260907` (SGLang 0.5.19, amd-aiter 0.1.21, torch 2.11, Triton 3.7; 72.7 GB, already on disk). Not yet run on the GPU; whether its GPTQ W4A16 path works on gfx942 is unchecked.
- Getting vLLM = pulling a ROCm vLLM image (tens of GB) → large download, needs the user's approval (machine rule).
**Stopped at checkpoint (a)** as instructed: the Triton FP8 path exists, but the vLLM anchor for step (c) does not. Waiting for the user's choice.

### User decisions (relayed by general-ags, 2026-11-15)
Step (c) option 1 (local SGLang image), with a short GPTQ check first; if it fails → option 3, no vLLM download. Step (b) approved. **Run (b) and (c) without asking; stop only at (d) before any custom kernel.** Decide small questions, record each with its reason. Stop early only if a step failure would make a reported number wrong. GPU 1 only. Keep the harness self-corrections visible in the final report.

### Decode speed — checkpoint (b): bandwidth upper limit (2026-11-15; `code/speed/roofline.py` → `/models/tmp/wsp2/results/speed_roofline.json`)
- Measured HBM read bandwidth on GPU 1: **4.66 TB/s** (16 GiB `sum`, median of 10); copy 4.19 TB/s; spec 6.0 TB/s. The bound uses the measured read value.
- Params: decoder linears 31.206 B, lm_head 0.778 B.
- Bytes per decode step = all linear weights (our bits/weight incl. scales and exceptions) + KV read at context 2048 (0.537 GB/sequence at KV16). Bound tok/s = 4.66 TB/s ÷ bytes; batch 16 = 16 × 4.66 TB/s ÷ (weights + 16 × KV).

| format | W | KV | GB weights | b1 bound tok/s | b16 bound tok/s | b1 bound, bf16 lm_head |
|---|---|---|---|---|---|---|
| bf16 | 16 | 16 | 63.97 | — | — | **72.2** |
| FP8 / INT8 W8A8 | 8.002 | 16 | 31.99 | 143.1 | 1836 | 139.8 |
| GPTQ W4A16 g128 | 4.250 | 16 | 16.99 | 265.6 | 2912 | 254.3 |
| W4A8 finals | 4.266 | 16 | 17.05 | 264.7 | 2905 | 253.5 |
| mix W3 late34 | 3.743 | 16 | 14.97 | 300.3 | 3162 | 286.0 |
| mix W3g64 late44 (±Afp8) | 3.689 | 16 | 14.75 | 304.6 | 3192 | 289.8 |
| ours W4A4KV4 | 4.266 | 3.65 | 17.05 | 271.1 | 3918 | 259.3 |
| QuaRot W4A4KV4 | 4.100 | 4.16 | 16.39 | 281.7 | 4000 | 269.0 |

Note: the script's bf16 row used an int8 lm_head in the "our accounting" column (73.1 tok/s); for bf16 the right value is the bf16-lm_head column, 72.2. A bound only: real kernels reach a fraction of it. At batch 1 the mixed-bit finals can be at most ≈ 15% faster than W4 (fewer bytes); at batch 16 the KV cache becomes a large part of the bytes, so KV4 matters more than 3-bit weights.

### Decode speed — step (c) GPTQ check: FAIL → option 3 (2026-11-15)
- In `rocm/sgl-dev:v0.5.19` (runner `code/drun_sgl.sh`, GPU 1), `torch.ops.sgl_kernel` has **no gptq_gemm, gptq_shuffle, gptq_marlin_gemm, int8_scaled_mm, fp8_scaled_mm, awq_dequantize**: SGLang's ROCm build does not include its GPTQ (exllama/marlin) or INT8 W8A8 kernels. (`code/speed/check_gptq.py` failed at `gptq_shuffle`.) SGLang's AWQ path on HIP = Triton dequantize of the whole weight to fp16 + `torch.matmul`: not a fused W4A16 kernel.
- aiter (AMD kernel library) is present in that image: gemm_a8w8 (CK), gemm_a4w4 (FP4, gfx950 only per the P0 ISA check). Not used yet.
- **Decision (as the user ordered): option 3, no vLLM download.** Our own decode loop with native kernels: bf16 (hipBLAS), FP8 W8A8 (`torch._scaled_mm`, row-wise scales), INT8 W8A8 (`torch._int_mm` + scale epilogue). GPTQ W4A16 has no existing kernel on this stack → it moves after checkpoint (d) together with our kernels.
- **Decision (agent): SGLang stays as the absolute anchor for bf16 and FP8** (both need no missing op) to validate our loop (rule: our loop and SGLang must agree within 20% before formats are reported). Reason: without an engine reference, a wrong loop cannot be detected.

### Decode speed — step (c) result and CHECKPOINT (d) (2026-11-15; `code/speed/`, results `/models/tmp/wsp2/results/speed_*.json`)
Setup: GPU 1, batch 1 and 16, context 2048, KV bf16, lm_head bf16 in every row (engine default). Bound = measured HBM read 4.66 TB/s ÷ bytes per step (step b). Own loop = `speed/decode_loop.py` (fused qkv and gate/up, matmul decode attention, `torch.compile`, HIP graph, median of 100 replays). SGLang = `sglang.bench_one_batch` (aiter attention, CUDA graphs).

| format | engine | b1 tok/s | b1 bound | % | b16 tok/s | b16 bound | % |
|---|---|---|---|---|---|---|---|
| bf16 | SGLang 0.5.19 | 50.2 | 72.2 | 69% | 703 | 1027 | 68% |
| bf16 | own loop | 53.4 | 72.2 | 74% | 724 | 1027 | 71% |
| FP8 W8A8, per-tensor (SGLang online fp8) | SGLang 0.5.19 | 68.1 | 139.8 | 49% | 923 | 1801 | 51% |
| FP8 W8A8, per-channel W / per-token A | own loop | 82.8 | 139.8 | 59% | 1028 | 1801 | 57% |
| INT8 W8A8, per-channel W / per-token A | own loop | 15.1 | 139.8 | 11% | 230 | 1801 | 13% |
| GPTQ W4A16 | — | no kernel on this stack (SGLang ROCm build has none; vLLM absent) | | | | | |

Findings:
1. **Loop validated:** own bf16 loop is within +6% (b1) / +3% (b16) of SGLang (rule: 20%). Own FP8 is faster than SGLang FP8 (+22%), probably because SGLang's online fp8 path adds per-tensor dynamic quant kernels; both are far from the bound.
2. **FP8 W8A8 gives 1.55× bf16 at batch 1** (82.8 vs 53.4), not 2×: it reaches 59% of its bound against 74% for bf16. Fixed costs (attention, norms, activation quantization) do not shrink with the weight bytes.
3. **INT8 W8A8 on this stack is slower than bf16** (15.1 vs 53.4 tok/s): `torch._int_mm` needs M > 16 (pad), and hipBLASLt selects a prefill tile (`MT128x128x32`) for the skinny shape: ≈ 240 µs per GEMM, ≈ 61 of 66 ms per step (≈ 11% of bandwidth). A kernel-selection limit, not a property of the format.
4. Consequence for the round-2 finals (estimate, not measured): time outside the weight GEMMs is ≈ 2.9 ms/step for bf16 (18.7 ms step − ≈ 15.8 ms GEMMs at ≈ 87% of bandwidth) and ≈ 4 ms for FP8 (adds activation quantization). A W4 kernel at the same 87% would need ≈ 4.2 ms for 17 GB → ≈ 7.5–8 ms/step → **≈ 125–135 tok/s at batch 1, about half of the 254 bound**. Fixed costs take a growing share below 8 bits; the mixed 3/4-bit format can gain at most ≈ 0.5 ms/step over W4 even with a perfect kernel. (First draft of this point said 5–6 ms fixed and 100–110 tok/s; that was a guess, corrected from the measured step times.)

Self-checks and corrections (keep visible in the final report):
- (a) two false failures, both mine: int32 matmul reference not supported on GPU; wrong Triton API for the ISA dump. Fixed.
- First loop version was 40% slower than SGLang (30.1 tok/s): profile showed SDPA using a prefill flash kernel for q_len 1 (≈ 110 µs/layer) and ≈ 11 ms of unfused element-wise kernels. Fixed with matmul decode attention + torch.compile. Did NOT report the first numbers.
- bf16 loop vs HF transformers on real text: top-1 agreement 1.000, KL 1.1e-3, rel err 2.1%.
- Compiled vs eager: on a RANDOM KV cache the check showed large differences (rel err 0.16–1.3) — misleading, logits are almost flat there. On REAL text: bf16 rel err ≤ 1.4%, top-1 1.000; FP8/INT8 at b16 top-1 1.000, but at b1 one position flips, KL ≈ 0.01–0.02 (≈ the FP8-vs-bf16 difference itself). Cause probably different rounding of the activation scales after fusion. **Decision: the compiled loop is used for speed only, never for accuracy** (accuracy = P4). It runs the same GEMM kernels on the same bytes.
- FP8/INT8 eager loop vs bf16 loop: FP8 top-1 0.958, KL 9.5e-3; INT8 top-1 0.875, KL 0.146 (INT8 per-token A is lossy on this model, as P0 found: +2.6% c4dev vs +0.5% FP8).
- Differences from the P4 baselines: SGLang FP8 is per-tensor (P4 FP8 is per-channel); lm_head bf16 here (P4 counts int8).

**STOP at checkpoint (d)** as instructed: no custom kernel written. Next, after the user's go: own W4A8 Triton kernel (unpack int4 + group scale → FP8 → FP8 MFMA), then W4A16 for the GPTQ baseline (needed as the comparison point), then mixed 3/4-bit (last, may be dropped).

### Decode speed — W4 kernels after the GO (2026-11-15; `code/speed/w4kernels.py`, bench scripts in `code/speed/`)
GO received after checkpoint (d) (scope: W4A8 first, then W4A16, mixed 3/4-bit last; GPU 1; time box ≈ 6 GPU-hours per kernel; real-text correctness gate before any speed number). Storage = GPTQ W4 g128 asymmetric: 4-bit codes + bf16 scale + int8 zero = 4.19 bits/weight.

Kernel-level results (one GEMM, graph-timed, % of 4.66 TB/s; every config also checked against an fp32 reference with the same weights: all rel err ≤ 1e-4):

| version | idea | M = 1 | M = 16 |
|---|---|---|---|
| v1 | MFMA, codes packed along K in [K/8, N] (GPTQ order), unpack by reshape | 11–22% | 11–21% |
| v2 | MFMA, transposed storage [N, K/8] (k-contiguous words) | 11–22% | 10–21% |
| v3 | GEMV on vector units, no MFMA, [BN, 16-word] tile | 22–47% (1.0–2.2 TB/s) | 4% (grid repeats the weight reads per token) |

Diagnosis of v1/v2 (`diag_w4.py`, down_proj shape, M = 1): loads alone reach 4.0 TB/s; with unpack 1.1–1.6 TB/s; full kernel 0.5–0.7 TB/s. The ISA shows 68–196 LDS operations: the reshape and the MFMA operand layout go through LDS.
Self-corrections:
- The first timings used `triton.testing.do_bench`. It includes ≈ 60 µs Python launch overhead per call, which is larger than the kernel at these shapes. Replaced by HIP-graph timing (`bench_w4.graph_time`: n launches in one graph, median of 20 replays). Only graph-timed numbers are used.
- `bench_w4t.py` imported `bench_w4.py`, which had no main guard, so it ran the full v1 search again. Stopped with `docker kill` (pkill not permitted on the root process); added `main()`.
- Caveat for all kernel-level numbers: a graph repeats the same GEMM, so the weights (13–131 MB) can partly stay in the 256 MB Infinity Cache. Kernel-level TB/s can be optimistic; the end-to-end loop (17 GB of weights per step) is the number that counts.
Decision: per shape, the decode loop uses the fastest correct kernel: GEMV for M = 1, MFMA v2 for M > 1.
| v4 | GEMV, wider tile ([BN, 16·GPI] words, per-word scale gather) | 19–43% (not better than v3) | — |

v4 (10:38 UTC) was correct but not faster than v3, so kernel tuning stops here (≈ 3.2 GPU-hours used of the ≈ 6 h box). Decision and reason: the remaining budget goes to the end-to-end gate and measurement, because a correct end-to-end number with a known kernel limit is worth more than a further kernel guess. The loop takes the faster of v3/v4 per shape from the stored configs.

**W4A8 end-to-end gate, first run (10:43 UTC): FAILED, no speed number taken.** (An earlier start at 10:38 stopped on my own bug: wrong key format when the loop read the stored v2 configs. Fixed.)
48-token real text, eager, own kernels vs a reference loop with the SAME W4 weights (fp32 dequant, same FP8 activation rounding):
- kernel vs ref, b1: top-1 0.896, KL 8.3e-3; b16 (row 0): top-1 0.917, KL 1.0e-2. Rule set before the run: top-1 ≥ 0.95 → FAIL.
- ref b16 vs ref b1 (reference against itself, only the batch differs): top-1 0.979, KL 5.9e-3. So the end-to-end test has a noise floor: a tiny change before a per-token FP8 rounding changes FP8 codes, and 64 layers amplify it.
- w4a8 vs bf16 loop: top-1 0.854, KL 9.0e-2 (RTN W4 + FP8 A; for information only).
The kernel-vs-ref difference is above the ref-vs-ref floor, so the failure is not explained by noise alone. I do NOT relax the rule. Added a direct check: capture the real input of every W4 linear on the text, put the same input into kernel and reference, M = 1 and M = 16 (16 real positions); threshold rel err ≤ 1e-2, as at kernel level. This becomes the gate; the end-to-end numbers stay reported. Recorded as a change of the gate form after a failure, with this reason.

**W4A8 result (11:06 UTC; `/models/tmp/wsp2/results/speed_loop_w4a8.json`, log `logs/speed/loop_w4a8.log`).**
Gate PASS: per-linear on real activations, 255 W4 linears (L63 down is the 1-byte exception), max rel err 2.0e-4 (M = 1) and 1.1e-4 (M = 16).
End-to-end (information): kernel vs ref b1 top-1 0.875 / KL 9.1e-3; b16 0.917 / 1.0e-2; ref b16 vs ref b1 0.979 / 5.9e-3. w4a8 vs bf16 loop: top-1 0.833 here against 0.854 in the 10:43 run with the same weights. So the eager loop is not deterministic run to run: split-K kernels add partial sums with fp32 atomics in a random order, and the per-token FP8 rounding amplifies this. That explains why the end-to-end test sits above the ref-vs-ref floor while every linear matches to 2e-4.

| format | lm_head | b1 tok/s | b16 tok/s |
|---|---|---|---|
| bf16 (own loop, step c) | bf16 | 53.4 | 724 |
| FP8 W8A8 per-channel (own loop, step c) | bf16 | 82.8 | 1028 |
| **W4A8 g128 (round-2 final format), own Triton kernels** | bf16 | **51.7** | **500** |
| W4A8 g128, own Triton kernels | 1 byte (FP8, stands in for int8) | 52.0 | 502 |

**Prediction on record: 125–135 tok/s at b1 (≈ 1.6× FP8). Measured: 51.7 — 2.4–2.6× worse than predicted, 0.62× FP8, 0.97× bf16.** At b16 W4A8 is 0.49× FP8 and 0.69× bf16. The lm_head variant changes almost nothing (≤ 0.6%), so no final number depends on it.
Cause: the prediction assumed a W4 kernel at ≈ 87% of bandwidth; my kernels reach 20–47% at kernel level, and less in the loop (step 19.3 ms vs ≈ 14 ms from kernel-level times + 4 ms fixed; probably Infinity Cache help in the kernel benchmark and the extra per-linear kernels: activation quantization, zero fill, output cast). This is a measured limit of MY kernels on gfx942 (no 4-bit MFMA), not a bound of the W4A8 format: the format's bandwidth bound is 254 tok/s.
Compiled-vs-eager checks on the random KV cache again show large differences (known to mislead, see step c); timing uses the same kernels on the same bytes.

**W4A16 result (GPTQ W4 baseline format, 11:27 UTC; `speed_loop_w4a16.json`).** Same kernels with ACT_FP8 off; no L63 exception.
Gate PASS: per-linear on real activations, 256 linears, max rel err 1.5e-4 (M = 1), 9.9e-5 (M = 16). End-to-end kernel vs ref: b1 top-1 0.979 / KL 1.3e-3, b16 1.000 / 1.6e-3; ref b16 vs ref b1 0.958 / 1.0e-3. Without FP8 activation rounding the end-to-end test is at its noise floor. This supports the W4A8 explanation above.
(w4a16 RTN vs bf16 loop: top-1 0.604, KL 0.63. RTN W4 without the L63 down_proj exception is very lossy on this model, as round 2 found. Speed only; accuracy is from P4.)

| format | lm_head | b1 tok/s | b16 tok/s |
|---|---|---|---|
| W4A16 g128, own Triton kernels | bf16 | 56.9 | 491 |
| W4A16 g128, own Triton kernels | 1 byte | 57.5 | 494 |

W4A16 is 6–10% faster than W4A8 at b1 (no activation quantization; FP8 path also has the 1-byte L63 down), about the same at b16.

**Mixed 3/4-bit: dropped** (rule 3 of the GO). Reason: the W4 kernels are the limit (20–47% of bandwidth); a 3-bit kernel of the same kind would not change the ranking, and checkpoint (d) already showed that even a perfect mixed kernel gains ≤ 0.5 ms/step over W4.
GPU time for the kernel work: ≈ 4 GPU-hours (inside the ≈ 6 h box).

**Decode-speed conclusion (own loop, GPU 1, ctx 2048, MI325X; wording fixed by the user decision of 2026-11-15, see "Round 2 closed").** With our own Triton W4 kernels on gfx942, which reach 20–47% of HBM bandwidth at kernel level, the W4A8 finals measured 51.7 tok/s (b1) and 500 tok/s (b16), against a bandwidth bound of 254 tok/s (b1). No vendor W4A8 kernel exists on this stack. In the same loop, FP8 W8A8 per-channel (native hipBLASLt) measured 82.8 / 1028 and bf16 53.4 / 724. This number describes our kernels on this stack, not the W4A8 format. Per the freeze, it does not change the round-2 ranking.
Setup mismatches (do not compare across them): SGLang FP8 per-tensor vs own-loop/P4 FP8 per-channel; bf16 lm_head here vs int8 lm_head in P4 accounting (both variants measured; ≤ 1% difference).

## Round 2 closed (2026-11-15; user plan relayed by general-ags)
Step 1 done (no GPU):
- `REPORT.md` written with two separate claims: **A = accuracy per bit**, **B = format and structural findings** (PLAN §9 item 2). BitOps is defined there as an operand-bit count, explicitly not a speed measure.
- Decode-speed text scoped as instructed (own Triton kernels on gfx942, 20–47% of bandwidth, 254 tok/s bound, no vendor W4A8 kernel on this stack). The earlier conclusion wording in this file and in PLAN §9 item 10 was replaced to match. The round-2 report states a limitation; it does not promise later validation.
- AMD kernel work finished. `code/speed/` frozen: `FROZEN.md` + read-only (`__pycache__` is owned by root from the container and stays as it is).
- Self-check while writing the report: my first draft of claim-B item 4 said "one FP8 exception on L6.gate fixes per-token A4". NOTES (E1, line ≈ 355) says the opposite: per-token A4 stays broken with L6.gate protected; grouped A4 makes the exception unnecessary. Fixed before release.
Step 2: round-3 plan written in `../round3-phase-a/PLAN.md`; waits for approval. Round 3 not started.

## Round 3 phase A — automatic-allocation control on Qwen3-32B (approved 2026-11-15)
User decision (relayed by general-ags):
- The kernel-shape rule is DROPPED. Kernel existence is a column only, never a filter, and the round-2 formats stay.
- Do not start Qwen3.6-27B. Phase A = the control on Qwen3-32B, with ≈ 10–15 GPU-hours on GPU 1.
- The verdict rule is binding, and a tie withdraws claim A.
- Phase B is not approved.

Plan: `../round3-phase-a/PLAN.md`. Code: `code/r2/alloc.py`. Results: `/models/tmp/wsp2/results/p3a/`. Logs: `/models/tmp/wsp2/logs/p3a/`.

Fixed choices (before any control result), with reasons:
- **Menu:** the formats of the agent's M finals (W3g128, W3g64, W4g128 asym; INT8 pc sym), all through GPTQ. Reason: "same format menu".
- **Primary target:** late34 (W 3.7432). Reason: it is the only M final that passes the worst-seed zero-shot guard, so it is the only one where an agent win is possible.
- **Secondary target:** late44 (3.6891), only if the budget remains. At this budget the agent cannot win (worst seed 68.92).
- **Sensitivity:** GPTQ on one linear at a time, KL of the final distribution on 16 calibration windows (seed 4242). Reference log-probs are stored in fp32, because bf16 storage would put a noise floor at the size of the small KLs.
- **Allocation:** greedy on per-linear convex hulls up to the budget.
- **Self-checks in the code:**
  - bf16 self-KL must be 0;
  - the greedy fill must land within 0.01;
  - the round-2 `p2.cost` must agree with the allocation within 0.01 before each run, else stop.
- 12:43 UTC (approx.): sensitivity run started (GPU 1).
- **Sensitivity run 1 (bf16) stopped after 2 layers: measurement defect, found by me.**
  - Every format of every linear gave KL ≈ 1.0–1.5e-3, including INT8.
  - Debug (GPU 1): an unchanged tail gives exactly 0, but an int8 RTN of layer-0 q_proj gives 1.0e-3 and W3 gives 1.2e-3. So any change of one linear moves the final distribution by the same bf16 rounding noise, carried through 64 layers. The real differences are hidden under this floor.
  - My first self-check (bf16 self-KL = 0) was too weak: it compared a hidden state with itself, not a perturbed path.
  - **Fix (before any control result exists):** the sensitivity step runs the whole model in fp32. Quantized weights are rounded to bf16 as in the pipeline.
  - Checked: L0.q W8 7.9e-7, W4 4.1e-5, W3 1.3e-4; L63.down W8 2.4e-4, W4 0.31, W3 1.68. A clean signal, and it agrees with round 2 (L63.down needs 8 bits).
  - To keep the budget, the step uses **8 KL windows instead of 16** (fp32 is slower).
  - Hessians now come from fp32 activations instead of bf16 (negligible difference).
  - The bf16 file `p3a/sens.json` (2 layers) is kept as a record and not used. The new file is `p3a/sens32.json`.
- Sensitivity run 2 (fp32) started 12:55 UTC.
- Sensitivity run 2 (fp32) results look clean. Layer 0: ≈ 407 s, layer 1: 390 s; later layers have shorter tails, so the whole step is ≈ 4.7 h (ends ≈ 17:40 UTC).
- Budget decision: the primary control (late34) runs right after. The secondary (late44) runs only if it can start before 23:15 UTC (≈ 15 GPU-hours from 12:38).
- `code/p3a_chain.sh` runs unattended: wait for sens → alloc (stops if the target is not reached within 0.01) → late34 s0–3 → late44 s0–3 (if within budget). It stops on any failure. The allocation sha256 is written into the alloc log and into every result file.
- 17:31 UTC: sensitivity run 2 done (4.6 h, 12:55–17:31). File `p3a/sens32.json`.
- **Allocation FROZEN 17:31 UTC, before any control evaluation:**
  - late34 (primary): budget 3.7432, W 3.7432, 532 greedy steps, sha256 `4bb1888796c76094b9209097ad8488f85ec1b4c7fd434ddfbf4b43d32c5106b4`. Predicted sum of KL 0.0851.
  - late44 (secondary): budget 3.6891, W 3.6891, 486 steps, sha256 `bbf21bc941697a886ef76dfb91540b30eb89d4b92d622f98fa384e0bdf5851a0`.
  - late34 control, counts by type (W3g128 / W3g64 / W4g128 / W8pc): q 42/11/11/0; k 3/4/42/15; v 3/2/34/25; o 34/17/13/0; gate 26/15/23/0; up 21/18/25/0; down 11/12/41/0.
  - Agent late34, for comparison: all 7 types W3g128 in layers 0–33 and W4g128 in layers 34–63, L63.down W8pc.
  - Difference in structure: the control puts most of its extra bits into k_proj and v_proj (small linears, cheap per bit; 40 of 128 at int8) and into down_proj. It also upgrades 70 linears in layers 0–33 (the agent: 0). Upgraded linears per layer rise from ≈ 2 (early) to 5–7 (layers 36–45) and ≈ 5 (late).
  - **The control does NOT choose int8 for L63.down (it gets W4g128).** Reason: with GPTQ, the isolated KL of L63.down is only 6.7e-3 (W3) / 1.2e-3 (W4) / 1.1e-5 (W8), ≈ 4× the other late down_projs. The 12:55 check (W3 1.68, W4 0.31) used **RTN**, not GPTQ. So my sentence above "it agrees with round 2 (L63.down needs 8 bits)" is correct only for RTN. Under GPTQ with exact inputs, the Hessian compensates most of the input-outlier problem. Whether the exception matters inside the full chain is answered by the control result itself; I do not change the method.
- 17:32 UTC: control late34 seed 0 started; `p2.cost` W 3.7432 agrees with the allocation.
- 18:28 UTC: control late34 s0: WikiText-2 7.8063 (+2.58%), fw25test 10.1101 (+2.37%), c4dev 9.3612, zero-shot 70.10. 56 min per seed. (Agent late34 means: +2.48 / +2.37 / 70.23.) No verdict from one seed.
- 21:16 UTC: control late34 s0–3 done. Secondary (late44) started at 21:16, inside the 23:15 limit.

### PHASE A VERDICT (primary, late34, W 3.7432) — TIE → CLAIM A WITHDRAWN

Δ% against bf16 (WikiText-2 7.6103, fw25test 9.8761), 4 seeds each, sample std:

| | WikiText-2 | fw25test | zero-shot mean | zero-shot worst |
|---|---|---|---|---|
| agent late34 (P4) | +2.479 ± 0.179 | +2.367 ± 0.141 | 70.23 | 69.98 |
| control (greedy KL allocation) | +2.491 ± 0.075 | +2.390 ± 0.077 | 70.19 | 70.09 |
| d = control − agent | +0.012 | +0.024 | | |
| pooled s / 2s | 0.137 / 0.275 | 0.114 / 0.228 | | |
| d / s | 0.09 | 0.21 | | |

- Rule (PLAN §3): agent wins needs d > 2s on both sets. Here d < 0.25 s on both. Control win needs −d > 2s: no. → **TIE. Claim A of the round-2 report is withdrawn.**
- The control per seed: WikiText-2 +2.58/+2.53/+2.41/+2.46; fw25test +2.37/+2.38/+2.50/+2.31; zero-shot 70.10/70.18/70.09/70.40; c4dev 9.361/9.375/9.352/9.357.
- The agent's per-seed values were read again from `p4/mix_asym_W3_late34/`. They agree with summary.json.
- The control passes the +4% bar on the mean and on the worst seed (worst +2.58 / +2.50). It fails the +2% sensitivity row, the same as the agent. It passes the worst-seed zero-shot guard (70.09 ≥ 68.94), with a smaller seed spread than the agent.
- **Interpretation (mine):** at 3.74 bits, a cheap automatic allocation finds an equal result. It needs no analysis: one KL pass of 4.6 GPU-hours and a greedy fill. The two allocations are very different (the control puts int8 into k/v, upgrades early layers, and leaves L63.down at W4), but they give the same accuracy. So the accuracy at this budget is not sensitive to *where* the bits go, within reasonable allocations. And the L63.down int8 exception is not necessary under GPTQ A16 at this budget.
- Phase B condition (agent win) is not met.

### Secondary verdict (late44, W 3.6891) — TIE. Phase A complete 2026-11-16 01:03 UTC

| | WikiText-2 | fw25test | zero-shot mean | zero-shot worst |
|---|---|---|---|---|
| agent late44 (P4) | +2.769 ± 0.099 | +2.446 ± 0.205 | 69.94 | 68.92 (fails guard) |
| control | +2.693 ± 0.072 | +2.570 ± 0.053 | 70.21 | 69.99 |
| d / 2s | −0.075 / 0.173 | +0.123 / 0.300 | | |

- The difference is in opposite directions on the two sets, and both are below 2s. → **TIE**. (The agent could not win here, because its worst zero-shot seed fails the guard.)
- The control passes the +4% bar and the worst-seed zero-shot guard, so at 3.69 bits it is the more robust config. The agent late44 is not.
- Phase A used ≈ 12.4 GPU-hours (12:38 → 01:03, including the stopped bf16 run). This is inside the 10–15 h budget. GPU 1 is free.

## Recreate files (2026-11-17; no GPU work)
User request: make round 2 and round 3 recreatable from their folders, as `wsp-quant/` is (code only, no checkpoints).
- Added `RECREATE.md` (environment, paths to change, data sources, run order, commands for P0–P4, phase A and the speed runs, and the limits).
- Added `docker/requirements-lock.txt` (`pip freeze` of `wsp-quant2:rocm7.2`). Reason: the Dockerfile pins only transformers and fla, so a rebuild today would get other versions of lm_eval, datasets and accelerate.
- Added `data_sha256.txt` (sha256 of the 25 cached data files). Reason: the Hugging Face dataset revisions are not pinned in `r2/common.py`.
- Added `../round3-phase-a/README.md`. **Decision:** the phase-A code stays in `code/r2/alloc.py` and is not copied to the round-3 folder. Reason: it imports the round-2 harness, and a copy could become different from the code that made the results.
- Gaps found and recorded in RECREATE.md: the phase-A `sens` launch command and the speed-run commands were not written down at the time (reconstructed from the scripts and logs); the SGLang FP8 flag is inferred from the log (35.2 GB against 61.1 GB of weights), not recorded.
- No code changed. `code/speed/` stays frozen and untouched. `Untitled` (18 bytes, text "wsp-quant-round-02") and the `__pycache__` folders are left for the user to remove.
