# Methodology: post-training quantization research on one LLM

This document records the method that we used in round 2 (Qwen/Qwen3-32B). You can use the same method for a different model. The results are in `NOTES.md`. The plan is in `PLAN.md`. The code is in `code/r2/`.

The method has five phases. Each phase uses the result of the previous phase.

| phase | purpose | data | typical GPU time (32B, 1 GPU) |
|---|---|---|---|
| P0 | Make the harness and prove that it is correct | bf16 reference only | 1 h |
| P1 | Measure where the model is sensitive to quantization | dev data | 6–10 h |
| P2 | Explore many methods quickly and select candidates | selection data | 15–20 h |
| P3 | Fix the final configurations and the rules | none (decisions only) | 0 |
| P4 | Run seeds and one confirmation check on the test sets | test data, one time | 1–2 days |

## 1. Principles

1. **Study the model before you tune a method.** Find which layers, which module types and which tensors cause the error. Then select a method that fixes that cause. Do not start with a known method and tune its parameters.
2. **Keep the test data closed until the end.** Use dev data and selection data to make all decisions. Use the test data one time, in the confirmation check. Do not select again after the check.
3. **Write the rules before you see the results.** Record the accuracy bar, the cost metric, the margin, the baselines and the seed rule in `NOTES.md` before the confirmation check. If a rule must change, write the reason and get approval from the owner of the research.
4. **Count all the bits.** The cost includes group scales, zero points, high-precision exceptions, low-rank factors and the lm_head.
5. **Make the baselines fair.** Run the baselines in the same harness, with the same data and the same lm_head. If a baseline gives a result that is much worse than the published result, find the cause before you use the baseline.

   **Why we do not compare with published numbers directly.** A difference in measurement can be larger than a difference between methods:
   - **Different model.** Most papers use Llama models. Each model reacts differently (Qwen3-32B has one linear, L63.down_proj, that causes almost all 4-bit weight error).
   - **Different evaluation.** Sequence length, dataset and tool versions, acc or acc_norm, batch size and padding all change the numbers. In round 1 our bf16 Llama 3 8B zero-shot was 67.09; the SpinQuant paper gave 69.6. That 2.5-point difference comes from the harness only.
   - **Different quantization settings.** Papers often keep the lm_head in 16 bits and use other group sizes, clipping, calibration data and GPTQ options.
   - **New test data.** No paper reports fw25test.
   - **One run.** Papers usually give one run, so the noise is unknown.

   **Use published numbers as a sanity check only.** If our baseline is far from the published range, our baseline is probably wrong. In round 2 this check found a bug: our first QuaRot result was +73%; the published range is about +10–20%; after the fix it was +20%.
6. **Give the agent a control that does not think.** If the agent's result is a choice among known options (for example bits per layer), compare it with a cheap automatic search over the same options at the same cost (for example an EXL2- or HAWQ-style greedy allocation). Without this control, a better result than single-method baselines does not show that the analysis added value. Report bit-allocation results and structural findings (formats, exceptions, bugs found) as separate claims.
7. **Record everything.** Record each result, each failure, each wrong conclusion and each correction in `NOTES.md` on the same day.

## 2. Harness (P0)

### 2.1 Simulated quantization

We use simulated ("fake") quantization: quantize and dequantize in the forward pass, then compute in bf16. This lets you test any format, also formats that have no kernel on the hardware. Mark these formats as "no native kernel" in the reports.

The harness (`code/r2/fq.py`) has these parts:
- **Weights:** integer symmetric or asymmetric, group size 0 (per channel) / 32 / 64 / 128, optional MSE clip search. MX formats (MXINT, MXFP with E2M1/E2M3/E4M3 elements and shared 8-bit exponent). FP formats with a float scale.
- **Activations:** dynamic quantization per token or per group, in a forward pre-hook on each linear layer.
- **KV cache:** quantization inside a replacement attention function. K is quantized after QK-norm and RoPE. K can be per token, per group, or per channel over blocks of 128 tokens (KIVI-style). Pad the last block of tokens by edge replication, because the sequence length is not always a multiple of 128.
- **Rules:** an ordered list of `{match: regex, w: spec, a: spec}` and `{kv: layer regex, k: spec, v: spec}`. A later rule overrides an earlier rule. One rule language gives all exceptions and mixed precision.
- **Reset:** keep the original weights on the CPU, so one process can test many configurations without loading the model again.

### 2.2 Checks before any experiment

Do these checks and record the results:
1. W16A16 through the harness gives the same NLL as the plain model (bit-exact).
2. The attention wrapper is called for each layer (count the calls).
3. After `reset()`, perplexity returns to the bf16 value.
4. Every transform that must keep the output the same (rotation, scale folding) keeps it the same on a small random model of the same architecture (relative difference about 1e-6 in float64).
5. GPTQ gives a lower layer error than RTN on a small test.
6. Record the bf16 perplexity on every data set and the bf16 zero-shot accuracy. These are the reference values.

### 2.3 Evaluation

- **Perplexity:** windows of 2048 tokens. Compute the lm_head and the loss in blocks of 1024 tokens. Do not make the full-vocabulary logits for a full batch at one time (for a 152k vocabulary this is 10–20 GB).
- **Zero-shot:** lm-eval, loglikelihood scoring, no chat template, the same tasks as earlier rounds.
- **Task-dev proxy:** 500 items each from the train splits of boolq, arc_easy, arc_challenge, piqa and winogrande. These splits are not the splits that lm-eval uses as test. Perplexity alone does not show all zero-shot damage, so use this proxy for selection.

## 3. Data split

| role | data | use |
|---|---|---|
| calibration | 128 windows: half WikiText-2 train, half C4 train | GPTQ, AWQ, SmoothQuant, low-rank. WikiText alone overfits. |
| dev / selection | C4 validation, first 64 windows ("c4dev"); task-dev proxy | all decisions in P1 and P2 |
| diagnostic only | WikiText-2 train proxy (32 windows, seed not used for calibration) | shows if a method is weak on WikiText; never used to select |
| spare | fw25sel (first half of a FineWeb 2025 shard) | not used |
| test | WikiText-2 test, fw25test (second half of the shard), 8 zero-shot tasks | P4 confirmation only |

The C4 validation data is used to select. Thus C4 is not in the accuracy bar.

## 4. Profiling (P1)

Run these measurements. Each one answers one question.

1. **Activation statistics** (per linear input: absmax, RMS per channel, kurtosis, top channels, top tokens). Find massive activations and outlier channels.
2. **Weight statistics** (kurtosis, column max/median, Gram-matrix spectrum). Find if the weights or the activations that meet the weights cause the error. Compute the activation-weighted weight error.
3. **Error attribution.** Quantize one part only and measure the perplexity change:
   - by module type (q, k, v, o, gate, up, down), for W4, W3, A4 and KV4;
   - by layer (all 64 layers);
   - by layer × type (all 448 linears) for the two most important settings.

   On Qwen3-32B this found that one linear (L63.down_proj) causes almost all weight error, and one linear input (L6.gate_proj) causes the A4 per-token collapse.
4. **Bit and granularity sensitivity** per module type: W 8/6/4/3/2 bits, A 8/6/4 bits, per channel / per token / groups 32/64/128 / MX blocks.
5. **Error propagation.** Quantize layer i only. Measure the relative hidden-state error at all later layers. Find if errors grow or decay.
6. **Outlier tokens.** Keep the sink token or the top tokens in high precision. Find if the problem is tokens or channels.

Write a short summary: "what the model is telling us". Use this summary to order the P2 experiments.

## 5. Exploration (P2)

### 5.1 Order

1. **Cheap screens first.** Use RTN with precision exceptions in a sweep (many configurations in one process; about 1–2 minutes each). Remove the bad directions.
2. **Solver builds next.** One configuration per process (GPTQ, AWQ, SmoothQuant, rotation, low-rank). GPTQ on a 32B model takes 35–45 minutes.
3. **Baselines early.** The accuracy bar and the margin depend on the baselines. Run them first, then do the bar check (section 6.2).

### 5.2 Methods in the harness (`code/r2/solvers.py`)

- **GPTQ:** sequential over the layers, act_order, static groups. Optional asymmetric target: the calibration runs two streams (full precision and quantized). The target for layer i corrects the error from layers 0..i−1 (strength α = 0.15).
- **AWQ and SmoothQuant:** scale search, folded exactly into the previous RMSNorm or the previous linear (v → o with GQA expansion, up → down).
- **Rotation (QuaRot-style):** R1 = random signs × kron(random orthogonal, Hadamard), RMSNorm weights fused first. Online block Hadamard at the down_proj input (R4). **Online Hadamard on q and k after RoPE when you quantize the K cache.**
- **Low-rank correction:** SVD of the activation-whitened quantization error, rank r, stored in bf16 and counted in the bits.

### 5.3 Cost metrics

- **M, stored bits per weight:** the average over all linear weights, including the lm_head (int8 per channel in every configuration). Include scales, zero points, exceptions and low-rank factors (16 bits each).
- **B, BitOps:** W bits × A bits per multiply-accumulate, weighted by the size of each linear. The lm_head input is bf16 (16 bits). bf16 = 256.
- **KV bits:** reported separately. Not part of M.
- **Hardware note (C):** say which formats have a native kernel. This is a note only, not a ranking metric.

## 6. Rules for the final decision (P3)

### 6.1 Rules to fix before the confirmation check

- **Accuracy bar:** every test perplexity ≤ (1 + x) × bf16. Zero-shot average ≥ bf16 − y, as a guard only. The bf16 reference is fully bf16 (also the lm_head).
- **Win condition:** equal accuracy (passes the bar) at a lower cost.
- **Margin:** ≥ 0.5 bits/weight below the cheapest baseline that passes the bar. For BitOps: ≥ 0.5 bits lower on W or on A, and the other not higher. Also report the raw W × A product and its ratio to the baseline.
- **Seeds:** the seed is the calibration draw. 4 seeds per final configuration. 8 or more for baselines with calibration randomness. RTN needs 1 run.
- **Pass rule:** the mean of the seeds passes all bar terms. A winner must also pass both perplexity terms with its worst seed. Always report the worst seed.

### 6.2 Bar check before the confirmation

When the main baselines are complete, check them against the bar on the selection data.
- If no baseline passes, the bar is too strict for the research question. Then the margin is measured from a high-cost baseline, and the comparison has no value.
- Do not change the bar yourself. Propose a new bar with the reasons and wait for the decision.
- In round 2 the +1% bar passed only 8-bit weights. The best 4-bit baseline (GPTQ W4g128) was at +1.9% on C4 dev and +3.1% on the WikiText train proxy. The user approved +4%. Set the bar from the worst proxy, not from the best one, and leave headroom for seed noise.

## 7. Pitfalls that we found (and how to prevent them)

| pitfall | effect | prevention |
|---|---|---|
| GPTQ Cholesky in float32 on an ill-conditioned Hessian (massive activations, near-zero channels) | random "not positive-definite" failures; we first thought it was out-of-memory | Cholesky in float64; if it fails, increase damping ×10 (up to 4 times); record each increase in the result. Use one helper for every Cholesky (also the asymmetric target and low-rank). |
| Reading the traceback late | 1.6 GPU-hours lost on a wrong theory (OOM) | Write each traceback and the memory state to its own file. The queue overwrites the job log on retry. |
| K cache quantized per token without the q/k Hadamard in the QuaRot baseline | baseline at +73% instead of +20%; an unfair comparison | Diagnose a baseline that is far from its published result. Quantize one part at a time on the rotated model. |
| lm_head left in bf16 | hides about 0.1–0.3 bits/weight | Quantize the lm_head the same way in every configuration and count it. |
| KV per-channel groups with a sequence length that is not a multiple of the group | crash on task-dev prompts; the queue retried 8 times | Pad the last block. Test with odd lengths. |
| Forward hook that returns a tensor | the hook replaces the input of the layer | A statistics hook must return None. |
| Regex `$^` as "match nothing" | matches the root module name "" | Use `(?!)`. |
| Reading the wrong field of a result | wrong claim ("no damping increase") sent to the user | Print the full structure of a new result file one time before you summarise it. |
| Two jobs on one GPU | memory pressure, harder diagnosis | One job at a time, unless the owner allows more. |

## 8. Reports

- `NOTES.md`: one section per experiment group, with date and time (UTC), the table of results, and the conclusion ("→ ..."). Write corrections as new entries. Do not delete the old text; mark it as wrong.
- Every result row: c4dev Δ%, WikiText proxy Δ%, task-dev, M, B, KV bits, extra parameters, damping increases.
- Every decision of the user: the date, the exact decision, and the effect on the plan.
- `methodology.md` (this file): update it when the method changes.
