# LLM quantization research report: round 1, round 2 and round 3 phase A

Date: 2026-11-17. Period: 2026-11-09 to 2026-11-16.

Sources:
- Round 1: `round1-llama3-8b/NOTES.md`, `round1-llama3-8b/ROUND2_HANDOFF.md`, `round1-llama3-8b/wsp-rq2-quantization-plan.md`.
- Round 2: `round2-qwen3-32b/NOTES.md`, `PLAN.md`, `methodology.md`, `REPORT.md`. The result files (`p4_summary/summary.json`) are not in this repository.
- Phase A: `round3-phase-a/PLAN.md`, `round2-qwen3-32b/NOTES.md` (sections "Round 3 phase A" and "PHASE A VERDICT"). The result files (`p3a/`) are not in this repository.

All numbers come from these files. Values marked Δ% are perplexity changes against the unquantized model of the same round.

---

## 0. Summary

| | round 1 | round 2 | round 3 phase A |
|---|---|---|---|
| Model | Llama-3-8B (base) | Qwen3-32B (post-trained) | Qwen3-32B |
| Question | Can an agent make 4-bit W4A4KV4 better than SpinQuant without gradient training? | Find the best accuracy per cost, with a free choice of method and format | Is the agent's bit allocation better than a cheap automatic allocation at equal bits? |
| Result | Better perplexity than SpinQuant on every set, but the zero-shot margin failed. Overall FAIL under the protocol. The user judged that this is not a real win. | 3 memory (M) wins and 3 BitOps (B) wins under the frozen rules. The W4A4KV4 design missed the bar. Two claims: A (accuracy per bit) and B (format and structural findings). | TIE on both targets. **Claim A is withdrawn.** Claim B stays. |
| GPU time | about 1 day, GPUs 1–2 | about 4 days, GPU 1 | 12.4 GPU-hours, GPU 1 |

The main lessons:
1. Measurement of the model (round 2, P1) found the real causes of quantization error: two linears and one K cache. This made the format and structural findings possible.
2. The agent's per-layer bit allocation is not better than a greedy allocation from per-linear KL sensitivity. At about 3.7 bits per weight, the location of the extra bits has little effect on accuracy, within reasonable allocations.
3. A proxy metric (BitOps) is not a speed measure. Our own W4 kernels did not give a speed benefit on this GPU. This is a result about our kernels, not about the format.

---

## 1. Round 1: Llama-3-8B, W4A4KV4 against SpinQuant (2026-11-09 to 2026-11-10)

### 1.1 Objective

Make 4-bit weights, 4-bit activations and a 4-bit KV cache (W4A4KV4) on Llama-3-8B better than the published methods (SpinQuant, QuaRot), with these rules:
- no gradient training;
- forward passes are free;
- the agent can tune at most 20 numbers per run by a search on scores.

Pass condition: each of 4 runs of the agent's method is better than each run of each baseline, by a margin, on perplexity **and** on the zero-shot average. The margin is 2σ of the baseline seeds.

### 1.2 Method

**Setup:**
- Model: meta-llama/Meta-Llama-3-8B @ 8cde5ca8 (base model).
- SpinQuant code @ 8f47aa3f.
- Docker image `wsp-quant:rocm7.2` with torch 2.10 + ROCm 7.2, on AMD MI325X.
- Perplexity: sequence length 2048, no BOS.
- Zero-shot: lm-eval 0.4.4 on 8 tasks (boolq, piqa, social_iqa, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa); acc_norm where available, else acc.

**Data split:**
- Development: C4 validation and a WikiText-2-train proxy.
- Selection: fw25sel, the first half of FineWeb shard CC-MAIN-2025-26. This crawl is newer than Llama 3.
- Test, final runs only: WikiText-2 test, fw25test (the second half of the shard) and the 8 zero-shot tasks.

**Protocol defects that the agent found and fixed before the runs:**
- The SpinQuant `--seed` argument did not reach the trainer. The seed was always 42. Fixed in our copy.
- The paper's zero-shot reference (69.6) cannot be reproduced. Our bf16 value is 67.09, and the difference comes from the harness. So all comparisons use our own baselines.
- The plan's permission to start from published SpinQuant rotations contradicts "no gradient training".

**Baselines:** SpinQuant (4 trained rotation seeds), a tuned SpinQuant variant, QuaRot-style (random Hadamard), RTN, GPTQ.

**The agent's method** (selected on fw25sel from 4 candidates; 9 tuned variables, which is ≤ 20):
- random Hadamard R1/R2, best of 16 R1 seeds on a practice proxy;
- L2 scaling (exponent 0.5) before R4;
- GPTQ with act_order, A4-quantized calibration inputs and an **asymmetric least-squares target** (α = 0.15);
- 256 × 2048 mixed WikiText-2/C4 calibration;
- activation clip 0.85, K/V clip 0.95.

### 1.3 Findings

| metric | agent s0–s3 | best baseline run | margin needed | result |
|---|---|---|---|---|
| WikiText-2 | 7.210–7.230 | 7.347 (SpinQuant s0) | 0.100 | PASS |
| C4 | 10.582–10.677 | 10.907 (SpinQuant s3) | 0.100 | PASS |
| fw25test | 10.448–10.519 | 10.788 (SpinQuant s3) | 0.112 | PASS |
| zero-shot avg | 64.33–64.60 | 63.33 (tuned SpinQuant s3) | 2.198 | **FAIL** |

- References: FP16 WikiText-2 6.138, zero-shot 67.09. SpinQuant mean 7.370 / 62.71. Agent mean 7.220 / 64.45.
- The agent's method closes 12–23% more of the perplexity gap to FP16 than SpinQuant.
- On zero-shot, every run of the agent beats every baseline run (64.33 > 63.33), but not by the margin. One bad SpinQuant seed (s0, 61.06) made the margin large (2.2 points).
- The agent's zero-shot seed std is 0.11; SpinQuant's is 1.10.
- **Ablation:** the asymmetric GPTQ target and the activation clipping give most of the gain. The R1 seed search adds nothing.
- Perplexity does not show all zero-shot damage. SpinQuant s0 had normal perplexity but lost 4–6 points on boolq and ARC.

### 1.4 Verdict and lessons

- Under the protocol: **FAIL** (both perplexity and zero-shot are necessary).
- User verdict: about 2% better perplexity is a gain that can be reported, but it is not a real win for the available compute.
- The cause is in the approach:
  - the agent stayed inside SpinQuant's pipeline and format and tuned its knobs;
  - the agent did not measure the model (activation flow, outliers, where the error comes from);
  - the agent made the plan's rules more strict by itself.
- Carried into round 2: study the model first; free choice of method and format; agree the cost metric before the test; use more baseline seeds; use a zero-shot development proxy.

---

## 2. Round 2: Qwen3-32B, free method and format (2026-11-11 to 2026-11-15)

### 2.1 Objective

Goal: the same accuracy or benchmark scores with less compute or memory. Find the best accuracy-per-cost points for Qwen3-32B. The method and the format are free.

Fixed with the user before any test data was read:
- **Win type (b):** equal accuracy at a clearly lower cost. Type (a), better accuracy at equal cost, is also reported.
- **Cost metrics:**
  - **M**: stored bits per weight over all linears. It includes scales, zero points, exceptions, low-rank factors and an int8 lm_head, which is in every config.
  - **B (BitOps)**: W bits × A bits per multiply-accumulate, weighted by the size of each linear. bf16 = 256.
  - KV bits are reported separately. Kernel availability is a note only.
- **Accuracy bar:**
  - The first bar was +1%. On selection data, only 8-bit weights passed it, so the user approved **+4%**.
  - The +4% bar is: WikiText-2 test ≤ 7.9147, fw25test ≤ 10.2711, and zero-shot average ≥ 68.94 as a guard. The bf16 values are 7.6103, 9.8761 and 70.94.
- **Margins:**
  - M: ≥ 0.5 bits/weight below the cheapest passing baseline. GPTQ W4g128 has W 4.250, so a win needs W ≤ 3.750.
  - B: ≥ 0.5 bits lower on W or on A, and the other not higher.
- **Seeds:**
  - 4 per final; 8 for the baselines that are margin bases; 2 for the baselines that fail clearly; 1 for RTN.
  - The seed mean must pass, and a winner's worst seed must pass both perplexity terms.
- No gradient training.

### 2.2 Method (five phases; full text in `methodology.md`)

| phase | purpose | data |
|---|---|---|
| P0 | Harness and correctness checks | bf16 reference |
| P1 | Profiling: where is the model sensitive? | dev data |
| P2 | Broad exploration, baselines, bar check | selection data |
| P3 | Freeze the finals and rules | none |
| P4 | Seeds and one confirmation on the test sets | test data, one time |

**Harness (P0):**
- Simulated quantization in bf16 (`code/r2/fq.py`). It supports:
  - integer and FP formats, MX formats, groups and clip search, for weights;
  - per-token and per-group activations, in hooks on each linear;
  - a KV cache in a wrapped attention function, with K quantized after QK-norm and RoPE;
  - an ordered rule language for exceptions and mixed precision.
- Checks:
  - W16A16 through every code path gives an NLL that is bit-identical to bf16;
  - the attention wrapper is called 512 times (once per layer per batch), as expected;
  - `reset()` restores the model exactly;
  - the transforms (rotation, scale folding) keep the output the same on a small random model;
  - GPTQ has a lower layer error than RTN.
- Hardware check: gfx942 has INT8 and FP8 MFMA, but no 4-bit MFMA. So on this GPU, 4-bit saves memory and bandwidth only.

**Data split:**
- Calibration: 128 × 2048 tokens, half WikiText-2 train and half C4 train.
- Selection: c4dev (64 C4 validation windows) and a task-dev proxy (500 train-split items each of boolq, arc_e, arc_c, piqa, winogrande).
- Diagnostic only: a WikiText-2-train proxy.
- Test: WikiText-2 test, fw25test and the 8 zero-shot tasks, used one time in P4.

**Solvers (`code/r2/solvers.py`):**
- GPTQ with act_order, static groups and an optional asymmetric target (α = 0.15, from round 1);
- AWQ and SmoothQuant, with exact folding;
- QuaRot-style rotation;
- whitened-SVD low-rank correction.

### 2.3 Findings of the profiling (P1)

1. **Two linears and one K cache cause almost all damage.**
   - **L63.down_proj weights:**
     - With W3g128 RTN, this one linear gives +28.67 c4dev perplexity. The sum over all 448 linears is +29.94.
     - Its input activations go up to 8768.
   - **L6.gate_proj input:**
     - With per-token A4, this one linear gives +1835. All 448 linears together give +1851.
     - It creates the massive activation.
   - **L0 K:** it has channel outliers.
   - With W3g128 RTN, 434 of the 448 linears cost ≤ +0.01.
2. **Activations:** heavy-tailed channel outliers. Groups of 32 or FP elements fix 4-bit activations without rotation:
   - A4 g32: +0.155;
   - per-token A4: about +10,500.

   FP8 per-token is almost free (+0.014). INT8 per-token costs +0.14.
3. **KV:**
   - K needs per-channel scales (KIVI-style): K4 per channel costs +0.018, and K4 per token costs +3.27.
   - V tolerates 2–3 bits.
4. **Error propagation:** an error at one layer decays through the later layers, so the late layers matter most.
5. **Outlier tokens:** keeping the sink or the top tokens in high precision does not rescue A4. The problem is in the channels, not in the tokens.

### 2.4 Findings of the exploration (P2, selection data, seed 0)

- **One exception fixes 4-bit weights.** L63.down_proj at int8 costs +0.016 bits/weight. It takes RTN W4g128 from +40% to +2.3% c4dev.
- **The asymmetric GPTQ target is the best single solver change**, at no bit cost.
- **FP8 activations are almost free.** GPTQ-asym W4A8-FP8 is more accurate than the GPTQ W4A16 baseline, at 0.52× the BitOps.
- **4-bit activations need small groups, not rotation:**
  - our W4A4KV4 with A4 g32: +4.33% c4dev;
  - QuaRot-style: +20.44%;
  - our design with a rotation added: worse.
- **A defect in our own baseline was found and fixed.**
  - The first QuaRot-style result was +73%; the published range is about +10–20%.
  - Diagnosis: K4 per token without a q/k Hadamard after RoPE.
  - After the fix, the result was +20%.
- **3-bit weights are the open problem.** Uniform W3g64 gives +6.4%, so mixed 3/4-bit allocations were built (W3 base, a late W4 block, the L63.down int8 exception).
- Low-rank correction does not replace the int8 exception.
- MXFP4 weights give the best task-dev score, at +0.1 bits.

### 2.5 Confirmation results (P4, test sets, one time, 55 runs)

Δ% against bf16, seed means:

| config | W | A | B | WikiText-2 | fw25test | zero-shot (min) | result |
|---|---|---|---|---|---|---|---|
| awq_gptq_W4g128_L63d8_asym15_Afp8 | 4.266 | 8.20 | 35.7 | +1.76 | +1.49 | 70.67 (70.49) | B win |
| gptq_W4g128_L63d8_asym15_Afp8 | 4.266 | 8.20 | 35.7 | +1.75 | +1.60 | 69.94 (69.54) | B win |
| mix_asym_W3g64_late44_Afp8 | 3.689 | 8.20 | 31.1 | +3.03 | +2.64 | 69.77 (68.87*) | M and B win |
| mix_asym_W3_late34 | 3.743 | 16 | 59.9 | +2.48 | +2.37 | 70.23 (69.98) | M win |
| mix_asym_W3g64_late44 | 3.689 | 16 | 59.0 | +2.77 | +2.45 | 69.94 (68.92*) | M win |
| ours_W4A4KV4_g32 | 4.266 | 4.90 | 21.9 | +4.39 | +4.22 | 69.12 | fail |
| base: GPTQ W4g128 | 4.250 | 16 | 68.0 | +2.12 | +1.92 | 70.10 | pass |
| base: AWQ W4g128 | 4.250 | 16 | 68.0 | +2.34 | +1.86 | 70.15 | pass |
| base: SmoothQuant INT8 W8A8 | 8.002 | 8.20 | 65.6 | +0.96 | +1.06 | 70.66 | pass |
| base: FP8 W8A8 | 8.002 | 8.20 | 65.6 | +0.55 | +0.48 | 70.93 | pass |
| base: GPTQ W3g128 | 3.267 | 16 | 52.3 | +11.26 | +9.06 | 68.60 | fail |
| base: QuaRot-style W4A4KV4 | 4.100 | 4.29 | 18.7 | +25.65 | +21.24 | 64.96 | fail |
| base: RTN W4g128 | 4.250 | 16 | 68.0 | +34.48 | +42.99 | 68.90 | fail |

\* One seed is below the guard of 68.94. Under the frozen round-2 rule (the guard applies to the seed mean) these configs pass. Under the worst-seed guard, which the user approved for round 3, both late44 configs fail.

Other results:
- **W4A4KV4 (type (a), not a win):** our design is about 5× closer to bf16 than QuaRot-style. Both fail the bar.
- **Selection predicted test** within about 0.6 points for every passing config, so there is no sign of selection overfit.
- **Post-freeze exploration** (selection data only, not confirmed):
  - About 3.5 bits/weight is at the edge of the +4% bar. So the M finals are about 0.2 bits from the limit, not 0.5.
  - More W4 layers are better than finer groups at equal bits.
  - V4 brings our W4A4KV4 to about +3.7% c4dev.

### 2.6 Decode speed (added after the freeze; it does not change the ranking)

The user asked for a real speed check, because BitOps does not measure speed. The test used our own decode loop on GPU 1, with context 2048.

| format | kernel | b1 tok/s | b16 tok/s | b1 bandwidth bound |
|---|---|---|---|---|
| bf16 | hipBLASLt | 53.4 | 724 | 72.2 |
| FP8 W8A8 | hipBLASLt | 82.8 | 1028 | 139.8 |
| W4A8 g128 | our own Triton kernels | 51.7 | 500 | 254 |
| W4A16 g128 | our own Triton kernels | 56.9 | 491 | — |

- No vendor W4A8 kernel exists on this stack (ROCm 7.2, SGLang 0.5.19). Our kernels reach 20–47% of HBM bandwidth.
- The agent's prediction before the measurement was 125–135 tok/s at batch 1. The measured value is 51.7.
- Correctness gate: every W4 linear matches an fp32 reference on real activations, with a relative error ≤ 2e-4.
- The first end-to-end gate failed because of run-to-run non-determinism. It was replaced by a per-linear gate, and the change is recorded.
- **Scope:** these numbers describe our kernels, not the W4A8 format. The report makes no speed claim for W4A8 or for mixed 3/4-bit. The kernel work is finished, and `code/speed` is frozen.

### 2.7 Round-2 claims (REPORT.md)

- **Claim A, accuracy per bit:** the agent's configs pass the bar with fewer bits than the cheapest passing baselines. Its main limit was that no automatic bit-allocation control was run. Round 3 phase A tested this.
- **Claim B, format and structural findings:**
  1. FP8 activations cost almost no accuracy on this model, but INT8 activations do.
  2. Asymmetric GPTQ with the L63.down int8 exception and FP8 activations is more accurate than GPTQ/AWQ W4A16.
  3. The problem in L63.down is in its inputs.
  4. L6.gate_proj creates the massive activation. Grouped A4 (g32) works without an exception.
  5. K needs per-channel scales after QK-norm and RoPE, and a q/k rotation must come after RoPE.
  6. In the W4A4KV4 format, our design is about 5× closer to bf16 than QuaRot-style.
  7. Below 4 bits, zero-shot becomes sensitive to the calibration seed before perplexity shows it.
  8. More W4 layers are better than finer groups (not confirmed).

---

## 3. Round 3 phase A: automatic-allocation control on Qwen3-32B (2026-11-15 to 2026-11-16)

### 3.1 Objective

Find whether the agent's per-layer bit allocation (the round-2 M finals) is better than a standard automatic allocation that has:
- the same format menu;
- the same GPTQ pipeline;
- the same seeds;
- the same bits per weight.

The result decides claim A.

Scope decisions (user):
- Keep the round-2 formats. Kernel availability is a reported column only, never a filter.
- Do not start Qwen3.6-27B. Phase B (Qwen3.6-27B) is only possible after an agent win, and the user decides it.
- No kernel work and no speed experiments.
- Budget: about 10–15 GPU-hours, GPU 1 only.

### 3.2 Method (all knobs fixed before any control result)

1. **Format menu:**
   - W3 g128 asymmetric (3.148 bits);
   - W3 g64 asymmetric (3.297 bits);
   - W4 g128 asymmetric (4.156 bits);
   - INT8 per-channel symmetric (8.003 bits).

   The lm_head is int8 and counted, as in round 2.
2. **Sensitivity:**
   - Quantize each of the 448 decoder linears alone, in each format, with the round-2 GPTQ solver (act_order, damp 0.01, round-2 calibration set, seed 0).
   - Measure KL(bf16 ‖ quantized) of the final next-token distribution, as a mean over all tokens of 8 calibration windows (seed 4242, which is not used by the calibration).
   - No selection data and no test data are used.
3. **Allocation:**
   - Start with every linear at W3 g128.
   - Keep the lower convex hull of each linear's (bits, KL) points.
   - Greedy fill: always take the upgrade with the largest KL drop per added bit, until the budget is reached.
   - Target: within 0.01 bits of the agent's W.
4. **Targets:**
   - Primary: `mix_asym_W3_late34` (W 3.7432). It is the only M final that passes the worst-seed guard.
   - Secondary: `mix_asym_W3g64_late44` (W 3.6891), only if the budget remains. The agent cannot win at this target, because its worst zero-shot seed fails the guard.
5. **Pipeline:** identical to the agent's M finals: one GPTQ op, asym α 0.15, 128 calibration windows, seeds 0–3.
6. **Freeze:** the allocation file and its sha256 were written to NOTES before the first control evaluation. Then the test sets were used one time.
7. **Binding verdict rule:**
   - For each test set, d = mean Δ%(control) − mean Δ%(agent), and s = sqrt((s_agent² + s_control²)/2).
   - **AGENT WINS:** d > 2s on both sets, and the agent passes the zero-shot guard on the mean and on the worst seed.
   - **CONTROL WINS:** −d > 2s on both sets.
   - **Otherwise it is a TIE, and a tie withdraws claim A.**

**Measurement defect, found and fixed before any control result:**
- In the first sensitivity run (bf16), every format of every linear gave KL ≈ 1e-3, including int8. Any change of one linear moved the output by the same bf16 rounding noise through 64 layers.
- The fix: the whole sensitivity step runs in fp32, with 8 KL windows instead of 16 to keep the budget.
- After the fix, the signal was clean. For L0.q_proj: W8 7.9e-7, W4 4.1e-5, W3 1.3e-4.

### 3.3 Allocation chosen by the control

Counts for the late34 control, as W3g128 / W3g64 / W4g128 / INT8:

| type | q | k | v | o | gate | up | down |
|---|---|---|---|---|---|---|---|
| control | 42/11/11/0 | 3/4/42/15 | 3/2/34/25 | 34/17/13/0 | 26/15/23/0 | 21/18/25/0 | 11/12/41/0 |

The agent's late34 uses W3g128 in layers 0–33 and W4g128 in layers 34–63 for all types, with L63.down at INT8.

The control's allocation is very different:
- It puts most extra bits into k_proj and v_proj (small linears, cheap per bit). 40 of these 128 linears get int8.
- It also gives many extra bits to down_proj.
- It upgrades 70 linears in layers 0–33, where the agent upgrades none.
- **It keeps L63.down at W4, not int8.** With GPTQ and exact inputs, the KL of L63.down is only about 4× that of the other late down_projs. An earlier check had shown that L63.down needs 8 bits, but that check used RTN, not GPTQ. This correction is recorded in NOTES.

### 3.4 Results

**Primary target, late34 (W 3.7432):**

| | WikiText-2 | fw25test | zero-shot mean | zero-shot worst |
|---|---|---|---|---|
| agent late34 | +2.479 ± 0.179 | +2.367 ± 0.141 | 70.23 | 69.98 |
| control | +2.491 ± 0.075 | +2.390 ± 0.077 | 70.19 | 70.09 |
| d / 2s | +0.012 / 0.275 | +0.024 / 0.228 | | |

**Secondary target, late44 (W 3.6891):**

| | WikiText-2 | fw25test | zero-shot mean | zero-shot worst |
|---|---|---|---|---|
| agent late44 | +2.769 ± 0.099 | +2.446 ± 0.205 | 69.94 | 68.92 (fails the guard) |
| control | +2.693 ± 0.072 | +2.570 ± 0.053 | 70.21 | 69.99 |
| d / 2s | −0.075 / 0.173 | +0.123 / 0.300 | | |

### 3.5 Verdict

- **late34: TIE.** d < 0.25 s on both test sets.
- **late44: TIE.** The differences are below 2s, in opposite directions on the two sets.
- **Claim A is withdrawn.** An addendum in `round2-qwen3-32b/REPORT.md` records this.
- The control passes the +4% bar on the mean and on every seed, at both targets. It also passes the worst-seed zero-shot guard at both targets. At late44 it is the more robust config, and the agent's late44 is not robust.
- Neither the control nor the agent passes the +2% sensitivity row.
- Phase B needed an agent win, so it does not start. The next step is the user's decision.
- GPU time: about 12.4 hours, inside the 10–15 hour budget.

### 3.6 Interpretation

- A cheap automatic method (one 4.6 GPU-hour sensitivity pass and a greedy fill) gets the same accuracy as the agent's allocation, without the analysis.
- Two very different allocations give the same accuracy. So at about 3.7 bits/weight on this model, the location of the extra bits has little effect, within reasonable allocations.
- The L63.down int8 exception is not necessary with GPTQ and 16-bit activations at this budget. The exception was necessary for RTN. It also helped in the round-2 W4A8 finals, but phase A does not test that setup.
- Claim B is not affected. Its findings are about formats and model structure (FP8 against INT8 activations, grouped A4, per-channel K, the q/k rotation after RoPE, the massive-activation source), not about bit allocation.

---

## 4. Overall limits

- **Models:** Llama-3-8B in round 1, and Qwen3-32B in round 2 and phase A. The findings may not transfer to other models.
- **Quantization:** simulated (fake quantization in bf16). The speed numbers come only from our own kernels on one GPU type (gfx942).
- **Bar:** the +4% bar is not strict. At +2%, only the 8-bit baselines and the W4A8 finals pass.
- **Zero-shot:** the guard is noisy (about 1 point of seed noise).
- **Claim B:** it has no automatic control for "a search does not find this". Items 7 and 8 come from selection data or post-freeze runs.
- **The phase-A control:**
  - It assumes that the KL of single linears adds up, as HAWQ- and EXL2-style allocators also assume.
  - It was run at two bit budgets only.
  - It does not test the W4A8 setup.
