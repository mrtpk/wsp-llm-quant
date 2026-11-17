# Round 2 plan: Qwen3-32B

Status: **ROUND 2 CLOSED 2026-11-15.** P0–P4 done (P4 ended 2026-11-14 08:44 UTC); post-freeze exploration and decode-speed measurement done. Report: `REPORT.md` (two claims). Round-3 plan: `../round3-phase-a/PLAN.md` (waits for approval). Accuracy bar +4% (user, 2026-11-12).
Round 3 will repeat this on Qwen3.6-27B (hybrid linear attention).

Read first: `../round1-llama3-8b/ROUND2_HANDOFF.md` (infra and round-1 lessons), `NOTES.md` (round-2 stack and ROCm check).

## 1. Goal

Goal: the same accuracy or benchmark scores with less compute or memory. Find the best accuracy-per-cost point for Qwen3-32B on the MI325X. The method and format are free: AWQ, GPTQ, group/block formats, mixed precision, outlier handling, rotations, and hybrids of these. Design from measurements of the model, not from one published recipe.

What counts as a real win (to agree with the user, section 7):
- **(a)** At equal cost to a strong published point, clearly better accuracy. The published points are W4A16 g128 (AWQ/GPTQ) and W4A4KV4 (QuaRot-style).
- **(b)** At equal accuracy (for example within 1% perplexity and 1 zero-shot point of bf16), clearly lower cost.

## 2. Model facts (Qwen/Qwen3-32B @ 9216db5781bf21249d130ec9da846c4624c16137)

- **Post-trained** (hybrid thinking/instruct). No base checkpoint was released. Perplexity is still meaningful but higher than for a base model.
- **Shape:** dense, 64 layers, hidden 5120, MLP 25600 (SwiGLU), 64 query heads and 8 KV heads, head_dim 128. Vocabulary 151936.
- **QK-norm:** an RMSNorm per head on q and k *before* RoPE. Unlike Llama, this matters for K quantization and for any rotation of Q/K.
- **Measured** (`NOTES.md`): bf16 WikiText-2 perplexity 7.409 on the first 16 windows only, not the full test. 0.28 s per 2048-token window, 64 GB peak memory on one MI325X.
- Released by Qwen around 2025-04. The FineWeb CC-MAIN-2025-26 crawl (June 2025) is newer than the model, so the fw25 sets still work as unseen web text.

## 3. Phases

### P0. Port the infrastructure (about 1 hour, CPU and 1 GPU)

- `code/r2/common.py`:
  - load Qwen3-32B with plain transformers 5.22 (no SpinQuant);
  - perplexity sets retokenized with the Qwen tokenizer: wikitext2 test, c4 val (256 windows), fw25sel (160), fw25test (320). Cache them in `/models/tmp/wsp2/data/`;
  - `perplexity`, `zeroshot` (the same 8 lm-eval tasks), and calibration data (WikiText-2 train + C4 train).
- `code/r2/fq.py`, a simulated quantization library, built in-house:
  - weight quantization (RTN, bits, group size, sym/asym, clip search);
  - activation quantization (per token, per group, block/MX-style, static or dynamic, clip);
  - KV quantization through a registered attention function: K after QK-norm and RoPE, V as stored;
  - per-module configuration, so any mixed-precision map can be expressed.
- `code/queue.sh` and `start_queues.sh`, copied from round 1. Results go to `/models/tmp/wsp2/results`.
- **Sanity checks:**
  - full bf16 perplexity on all sets;
  - bf16 zero-shot;
  - simulated W16A16 must equal bf16 exactly;
  - RTN W8A8 should be close to bf16.

### P1. Profiling study (forward passes only, about 6–10 hours on GPU 1, 2 jobs)

On GPU 1. Each result goes into `NOTES.md` as "measurement → design implication".

1. **Activation statistics** at every linear input and in the residual stream, on 64 × 2048 calibration tokens:
   - per-channel absmax/RMS and kurtosis;
   - per-token absmax: massive activations, the first token, delimiters, attention sinks;
   - how these change with depth.
2. **Weight statistics:**
   - per-output-channel ranges;
   - salient input channels weighted by activation (the AWQ view);
   - the spectrum of the quantization error (is it low-rank?).
3. **Error attribution.** Quantize one thing at a time and measure the change in C4 dev perplexity (64 windows):
   - each module type, separately for weights and for activations: q, k, v, o, gate, up, down;
   - the KV cache;
   - each of the 64 layers.
4. **Bit and granularity sensitivity** for the top modules:
   - bits: W 8/6/4/3/2 and A 8/6/4;
   - granularity: per channel, per token, groups of 32/64/128, MX blocks of 32.
5. **Error propagation.** Quantize layer i only and track the hidden-state relative error at layers i..63, for about 8 values of i. Does the error grow or decay?
6. **Outlier tokens.** Perplexity with and without keeping the massive-activation tokens (BOS or sink, per the profile) in high precision.

### P2. Exploration (fast, broad, cheap; failures are fine)

All runs (on GPU 1) use dev metrics only: C4 dev, the WikiText-2-train proxy and a task-dev proxy (section 5). Candidate directions; the profile decides the order:
- **Weight-only:** RTN, GPTQ (round-1 solver, with act_order and the asymmetric target α≈0.15), AWQ-style scaling, and AWQ+GPTQ. Bits 4 and 3, group 128/64/32.
- **Weights and activations:**
  - SmoothQuant-style migration with per-layer strength from measurement;
  - Hadamard rotation only where activations are heavy-tailed (not everywhere);
  - outlier-token exceptions;
  - MX-style block formats for activations.
- **Mixed precision** from the error-attribution map: bits where they matter. Report the average bits.
- **KV cache:** 8/4 bits, per head or per group; K after the norm makes this different from Llama.
- **Hybrids** of whatever the profile shows works. Try ideas quickly in many directions rather than slowly in one.

### P3. Selection

Pick at most 10 candidates on fw25sel, scored by the cost metric agreed with the user. Fix the final configuration(s) and the pass bar before reading any test data.

### P4. Final evaluation

- 4 seeds per final configuration.
- Seeds for every baseline, number fixed in advance; at least 8 for the noisy ones.
- All perplexity sets plus zero-shot.
- Report the accuracy-versus-cost frontier and whether the agreed win condition is met.

## 4. Baselines (same harness, same data)

| baseline | format | source |
|---|---|---|
| bf16 | 16/16/16 | reference |
| RTN | W4A16 g128, W8A8 | own fq.py |
| GPTQ | W4A16 g128, W3A16 g128 | own (round-1 solver) |
| AWQ | W4A16 g128 | own implementation; also Qwen's official `Qwen/Qwen3-32B-AWQ` as an external check, if it loads on ROCm |
| FP8 | W8A8 FP8 | Qwen's official `Qwen/Qwen3-32B-FP8`, if it loads; else own simulation |
| SmoothQuant | W8A8 INT8 | own |
| QuaRot-style | W4A4KV4 (Hadamard + GPTQ) | own; SpinQuant/QuaRot code is Llama-only |

SpinQuant needs gradient training of rotations, so it's included only if the user allows training (section 7).

## 5. Evaluation and test hygiene

- **Development:** C4 val and the WikiText-2-train proxy (perplexity), plus the **task-dev proxy**. That proxy is the train or validation splits of boolq, arc_e, arc_c, piqa and winogrande, sampled at about 500 items each and disjoint from the test splits lm-eval uses. Round 1 showed perplexity misses zero-shot damage.
- **Selection:** fw25sel.
- **Test, final runs only:** WikiText-2 test, fw25test and the 8 zero-shot tasks (test/val splits as lm-eval defines them).
- **Zero-shot with a post-trained model:** use the same 8 tasks, no chat template, loglikelihood scoring, so the numbers stay comparable with round 1. Thinking mode doesn't apply to loglikelihood tasks.

## 6. Compute

- **One GPU only (GPU 1), from 2026-11-11, user.** **At most 2 concurrent jobs** (user: don't overload the icpc host). Machine-wide rule: no heavy load without an explicit ask.
- For 32B the weights alone are 64 GB. Two jobs fit easily in memory (about 70–100 GB each); the limit is the host, not GPU memory. Measure peak memory in P0 anyway.
- With one GPU, the time estimates in P1/P2 roughly double compared with two GPUs. With 2 jobs, P1 is about 6–10 hours. Order P1 so the most informative measurements come first (error attribution, then activation statistics).
- Wall-clock estimates: bf16 full perplexity over all sets about 5 minutes; zero-shot over 8 tasks about 20–30 minutes (4× round 1's model size); GPTQ on 64 layers about 1.5–2.5 hours.

## 7. Decisions needed from the user before P3 (P0–P2 can run without them)

**Decided 2026-11-11 — see NOTES.md "P3/P4 decisions".** Rank on M (bits/weight incl. int8 lm_head) and B (BitOps); C (8-bit compute) is a side note. Win = (b) at ≥0.5 bits, bar on WikiText-2 test + fw25test ppl (+1%) with zero-shot ≥69.94 as a guard. Selection on c4dev + task-dev only, one confirmation check after. The text below is the original question list.

1. **Cost metric and win condition:** (a) better accuracy at equal cost, (b) lower cost at equal accuracy, or both. Also which cost counts: memory (average weight bits + KV bits), compute (activation bits / GEMM type) or both.
   - Hardware note to verify in P0: MI300-class MFMA has INT8 and FP8 but, I believe, no native INT4 GEMM. If so, W4A4 saves memory but not compute on this GPU, and W8A8/FP8 is the real compute saving.
2. **Gradient training allowed?** For example SpinQuant-style rotation learning, or tuning only the quantization scales. Default: no, as in round 1.
3. **Extra full-precision pieces allowed?** For example outlier channels or tokens in bf16, or low-rank corrections. Default: allowed but counted in the cost.
4. **The task-dev proxy in section 5:** fine as proposed?

## 8. Risks

- **lm-eval with transformers 5.22 is untested.** The round-1 harness used 4.44. Check in P0 with bf16 zero-shot.
- **Official AWQ and FP8 checkpoints** may need kernels (autoawq, fp8 GEMM) that don't exist on ROCm. Fall back to simulated versions.
- **QK-norm before RoPE** changes where a Q/K rotation can be fused. A Llama-style R3 must sit after RoPE, online.
- **Post-trained model:** perplexity differences may matter less than task scores. Give the task-dev proxy real weight.

## 9. Additions for round 3 (decided 2026-11-13 with the user; reason in NOTES.md, "Discussion with the user")

1. **Automatic mixed-precision baseline at equal bits.** EXL2- or HAWQ-style: measure the error of each linear (or layer) at 3, 4 and 8 bits on calibration data, then a greedy fill to the same average bits/weight as our M final. Same harness, same GPTQ solver, same lm_head. Cost ≈ 2–3 GPU-hours. Fix it in the frozen set before P4.
   - If the agent's allocation is equal or worse: report it, and move the wsp claim to the structural findings (formats, exceptions, bugs found).
   - If it is better: the M claim becomes stronger.
2. **Split the claims in the report:** (i) bit allocation; (ii) format choice and structural findings. Give each its own control. **Done for round 2 (`REPORT.md`): claim A = accuracy per bit, claim B = format and structural findings.**
3. **Check the related-work names** (HAWQ, HAQ, k-quants, EXL2, SqueezeLLM, OWQ, SliM-LLM) against the papers before they go into a report.
4. **Report a stricter bar as a sensitivity row** (for example +2%), next to the agreed bar. Not a rule change: the agreed bar stays the decision rule.
5. **Measure the seed spread early** (2 seeds on 1–2 P2 candidates) and select with headroom ≈ 2× the spread, not a fixed large margin (round 2 left ≈ 1.2–1.6 points unused).
6. **Retry W4A4KV4 with V4** (or another small change) first; round 2 missed the +4% bar by ≈ 0.3 points.
7. **Zero-shot guard:** more items or a confidence interval; its seed noise is ≈ 1 point.
8. **APPROVED (user via general-ags, 2026-11-14): zero-shot guard on the worst seed as well as the mean, from round 3.** Round 2: two late44 finals had one seed at 68.92/68.87 against a 68.94 guard, while perplexity was stable. Below 4 bits, select with task-dev on ≥ 2 seeds, not perplexity only.
9. **From the post-freeze exploration (14 Nov):** W4A4KV4 starts from V4 (c4dev +3.74 vs +4.33 with V3); M allocations use a late W4 block rather than finer groups; ≈ 3.5 bits/weight is at the edge of +4% on Qwen3-32B.
10. **Measure real decode speed for the leading configs, not only BitOps** (proposed by general-ags 2026-11-14; agent agrees; method in NOTES.md "Decode-speed measurement plan"). Reason: CompressionBench (arXiv 2611.06858, notes in `~/scratchpad/quant/notes.md`): an agent-designed format on Qwen3.8 27B at 8 GiB passed on quality but decoded at 23.4 tok/s against 75.5 for the human release (batch 1, H200, vLLM), because unpacking/reconstruction cost more than the memory saving. BitOps did not predict it. **DONE 2026-11-15, closed:** with our own Triton W4 kernels on gfx942 (20–47% of HBM bandwidth), the W4A8 finals measured 51.7 tok/s b1 / 500 b16 against a 254 tok/s b1 bound; no vendor W4A8 kernel exists on this stack (FP8 W8A8 native 82.8 / 1028, bf16 53.4 / 724, same loop). The number describes our kernels, not the format. AMD kernel work finished; `code/speed` frozen. Details: NOTES "Decode speed — W4 kernels after the GO".
11. **Report two distances separately:** bits below the M base (round 2: 0.51–0.56 below GPTQ W4) and distance to the bar (round 2: ≈ 0.2 bits; ≈ 3.5 bits/weight is at the edge). Do not merge them into one "headroom" number.
