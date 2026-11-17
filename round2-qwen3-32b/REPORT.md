# Round 2 report: Qwen3-32B quantization (closed 2026-11-15)

Source of every number: `NOTES.md` ("P4 FINAL RESULT", "Post-freeze exploration results", "Decode speed" sections) and `/models/tmp/wsp2/results/p4_summary/summary.json`. The method is in `methodology.md`.

## Setup

- **Model:** Qwen/Qwen3-32B @ 9216db57, on one AMD MI325X (gfx942).
- **Quantization:** simulated (fake quantization in bf16). The accuracy numbers do not depend on a kernel.
- **Bar, fixed before P4:**
  - the seed-mean perplexity is at most +4.0% against bf16 on WikiText-2 test and on fw25test;
  - the seed-mean zero-shot average (8 tasks) is at least 68.94;
  - the worst seed of a winner passes both perplexity terms.
- **Selection and confirmation:**
  - the configs were selected on c4dev, wt2train and task-dev only;
  - 14 configs were frozen and then evaluated once on the test sets (P4), with 4 seeds for each final and up to 8 seeds for each baseline.
- **Cost units:**
  - **W** = bits per weight, including scales, exceptions and an int8 lm_head;
  - **A** = activation bits, weighted by FLOPs;
  - **B (BitOps)** = W bits × A bits per multiply-accumulate, weighted by the size of each linear.
  - BitOps counts the operand bits of the multiplications. **It is not a measure of speed. This report does not claim a speed benefit from BitOps.**

The report has two separate claims, A and B (PLAN.md §9 item 2). Each claim has its own evidence and its own limits.

---

## Claim A — accuracy per bit

**Statement:** on Qwen3-32B, the agent's configs pass the +4% bar with fewer weight bits, or fewer W×A bits, than the cheapest baselines that pass.

| config | W | A | B | WikiText-2 | fw25test | zero-shot | margin result |
|---|---|---|---|---|---|---|---|
| mix_asym_W3_late34 (W3 g128, W4 in layers 34–63, L63.down int8) | 3.743 | 16 | 59.9 | +2.48 | +2.37 | 70.23 | M win (−0.507 bits) |
| mix_asym_W3g64_late44 | 3.689 | 16 | 59.0 | +2.77 | +2.45 | 69.94* | M win (−0.561 bits) |
| mix_asym_W3g64_late44_Afp8 | 3.689 | 8.20 | 31.1 | +3.03 | +2.64 | 69.77* | M and B win |
| awq_gptq_W4g128_L63d8_asym15_Afp8 | 4.266 | 8.20 | 35.7 | +1.76 | +1.49 | 70.67 | B win |
| gptq_W4g128_L63d8_asym15_Afp8 | 4.266 | 8.20 | 35.7 | +1.75 | +1.60 | 69.94 | B win |
| base M: GPTQ W4 g128 | 4.250 | 16 | 68.0 | +2.12 | +1.92 | 70.10 | — |
| base M: AWQ W4 g128 | 4.250 | 16 | 68.0 | +2.34 | +1.86 | 70.15 | — |
| base B: SmoothQuant INT8 W8A8 | 8.002 | 8.20 | 65.6 | +0.96 | +1.06 | 70.66 | — |
| base B: FP8 W8A8 | 8.002 | 8.20 | 65.6 | +0.55 | +0.48 | 70.93 | — |

Values are seed means, Δ% against bf16. The margin rules were fixed before P4:
- **M:** W ≤ 3.750, which is 0.5 bits below GPTQ W4 g128 at 4.250.
- **B:** at least 0.5 bits lower on W or on A than the W8A8 bases, and the other not higher.

\* One seed (s3) of each late44 model is below the zero-shot guard (68.92 and 68.87, against 68.94). Under the frozen round-2 rule (guard on the seed mean) they pass. **Under the worst-seed guard, which is approved from round 3, both late44 models would fail.** Then the only robust M win is late34 (3.743), and the combined M+B row goes.

Two distances (keep them separate):
- **Bits below the M base:** 0.51–0.56.
- **Distance to the bar** (post-freeze, selection data only, not confirmed): about 0.2 bits. At ≈ 3.5 bits/weight, an allocation is at the edge of +4%.

**Limits of claim A:**
1. **No automatic bit-allocation control.** The M baselines use one method at one bit width. A standard per-layer search at equal bits (HAWQ- or EXL2-style) can possibly find a similar allocation without the agent's analysis. So round 2 does not show that the *agent's* allocation is better than a cheap search. Round 3 tests this as its main experiment.
2. The +4% bar is not strict. At +2%, only the 8-bit baselines and the W4A8 finals pass.
3. Only one model was tested, with simulated quantization.

---

## Claim B — format and structural findings

**Statement:** the agent measured the model and found format choices and structural problems that a bit-allocation search does not find. These findings explain most of the claim-A results.

1. **FP8 activations cost almost no accuracy on this model, but INT8 activations do.** In P0, FP8 per-token activations cost +0.5% c4dev and INT8 per-token activations cost +2.6%. The cause is activation outliers. This made the W4A8-FP8 finals possible.
2. **W4A8-FP8 with asymmetric GPTQ and an L63.down_proj int8 exception is more accurate than GPTQ/AWQ W4A16.** On WikiText-2: +1.75 / +1.76 against +2.12 / +2.34. W is about equal (4.266 against 4.250). The asymmetric GPTQ target and the exception (+0.016 bits) recover more accuracy than the FP8 activations cost. These finals pass on every term and every seed. AWQ+GPTQ-asym is the most stable (zero-shot range 0.41).
3. **The problem in L63.down_proj is in its inputs, not its weights.** Its input activations go up to 8768. An int8 version of this one linear gives the same accuracy as bf16 for 0.016 bits instead of ≈ 0.05.
4. **L6.gate_proj creates the massive activation.** When only this one linear gets per-token A4, perplexity damage is +1835 (all 448 linears together: +1851). But an exception on it does not repair per-token A4, because the remaining problem is spread over many linears. Grouped A4 (g32) works without the exception.
5. **K needs per-channel scales after QK-norm and RoPE.** A QuaRot-style rotation of q/k must come after RoPE. The agent found this bug in its own QuaRot baseline.
6. **W4A4KV4 (type (a), not a win):** the agent's design is about 5× closer to bf16 than the QuaRot-style baseline (+4.39 against +25.65 WikiText-2). Both fail the bar; the agent's design misses by ≈ 0.4 points. It uses more A bits (4.90 against 4.29) and fewer KV bits (3.65 against 4.16).
7. **Below 4 bits, perplexity alone is not enough.** More 3-bit layers made zero-shot sensitive to the calibration seed (range ≈ 1.7) before perplexity showed it (range ≈ 0.2).
8. **More W4 layers are better than finer groups at equal bits.** This was found on selection data after the freeze, and it is not confirmed.

**Limits of claim B:**
- These are findings on one model. Each finding has its own measurement in NOTES.md, but no automatic method was run as a control for "a search does not find this".
- Items 7 and 8 come from selection data or post-freeze runs and are not confirmed on test data.

---

## Decode-speed measurement (added after the freeze; does not change the ranking)

**Setup:** own decode loop, GPU 1, batch 1 and 16, context 2048, bf16 KV cache.

| format | kernel | b1 tok/s | b16 tok/s | b1 bandwidth bound |
|---|---|---|---|---|
| bf16 | hipBLASLt (native) | 53.4 | 724 | 72.2 |
| FP8 W8A8, per-channel W / per-token A | hipBLASLt (native) | 82.8 | 1028 | 139.8 |
| W4A8 g128 (round-2 finals' format) | **our own Triton kernels** | 51.7 | 500 | 254 |
| W4A16 g128 (GPTQ baseline format) | **our own Triton kernels** | 56.9 | 491 | — |

**Scope of the W4 rows:**
- These numbers describe our own Triton kernels on gfx942. At kernel level, they reach 20–47% of HBM bandwidth.
- The bound for W4A8 at batch 1 is 254 tok/s.
- No vendor W4A8 kernel exists on this software stack (ROCm 7.2, SGLang 0.5.19; SGLang's ROCm build also has no GPTQ kernel).
- The numbers do not measure the W4A8 format itself.
- Before the measurement, the agent predicted 125–135 tok/s at batch 1. The measured value was 51.7.

**Correctness gate:** every W4 linear matched an fp32 reference with the same weights on real-text activations. The largest relative error was 2.0e-4 (W4A8) and 1.5e-4 (W4A16).

**Setup differences** (do not compare across them):
- SGLang FP8 uses per-tensor scales; P4 and this loop use per-channel scales.
- This loop uses a bf16 lm_head, while P4 counts an int8 lm_head. Both variants were measured, and they differ by 1% or less.

**Self-corrections are in NOTES.md:**
- two false failures at checkpoint (a);
- a first loop version that was 40% slower than SGLang;
- a misleading compiled-vs-eager check on a random KV cache;
- the INT8 kernel-selection problem;
- timings that included Python launch overhead;
- an end-to-end gate that failed because of non-determinism. The gate form was changed to a per-linear check, and the reason is recorded.

**Limitation:** round 2 has no measured speed for the W4 formats with a tuned kernel. This report makes no claim about the decode speed of W4A8 or of the mixed 3/4-bit format.

---

## Overall limits

- **Model:** only one model was tested, with simulated quantization.
- **Baselines:** they are single methods, with no automatic mixed-precision control (see claim A, limit 1).
- **Bar:** the +4% bar is not strict.
- **Zero-shot guard:** it is noisy (seed noise ≈ 1 point). The worst-seed rule is approved for round 3.
- **Selection:** selection data predicted the test data within ≈ 0.6 points for every passing config, so there is no sign of selection overfit.

---

## Addendum 2026-11-15 (round 3 phase A): CLAIM A IS WITHDRAWN

- Round 3 phase A ran the automatic-allocation control for limit 1 of claim A, with:
  - per-linear KL sensitivity in fp32;
  - a greedy fill to the same W (3.7432);
  - the same format menu and the same GPTQ pipeline;
  - seeds 0–3.
- Result against late34: WikiText-2 +2.49 against +2.48, fw25test +2.39 against +2.37. The difference is below 0.25 pooled seed std on both sets.
- Under the binding rule, this is a **tie**. A tie withdraws claim A.
- The control also passes the bar and the worst-seed zero-shot guard (worst seed 70.09).
- Claim B and the decode-speed section are not changed by this result.
- Details: NOTES.md, "PHASE A VERDICT".
- Secondary control at late44 (W 3.6891): also a tie (WikiText-2 +2.69 against +2.77, fw25test +2.57 against +2.45; both differences are below 2s). The control passes the worst-seed zero-shot guard (69.99), and the agent's late44 does not (68.92).
