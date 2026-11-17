# RQ2 plan: quantization of Llama 3 8B with weight-space programming

Status: 2026-11-10. Draft. The target (section 3) is agreed. The other steps are not agreed.

Approval gate (decision of the user, 2026-11-10): the agent can prepare until 08:00 local time on 2026-11-10. Preparation is: this plan, code, and tests on the CPU. After that time, the agent waits. No experiment run starts without the approval of the user: no baseline run, no WSP run, no GPU job and no model download.

## 1. Goal

Question: can a WSP agent make 4-bit Llama 3 8B better than the published methods, with no gradient training?

Pass condition: at 4-4-4, each of 4 WSP runs is better than each run of each baseline, by a margin, on perplexity and on the zero-shot average (step 7).

The two rules of the research apply:

1. WSP is good only if it improves the metrics of the published study and of our baselines.
2. The agent does no gradient training. Forward passes are free. The agent can tune 20 numbers or fewer for each run by a search on scores.

## 2. Published numbers

Source: SpinQuant paper (arXiv 2405.16406, version 4, table 1 and table 2) and the README of `facebookresearch/SpinQuant`. Checked 2026-11-10. Model: Llama 3 8B. Each cell is: zero-shot average of 8 tasks / WikiText-2 perplexity. W-A-KV are the bits of the weights, the activations and the KV cache.

| Method | 4-4-16 | 4-4-4 | Source |
|---|---|---|---|
| Full precision (16-16-16) | 69.6 / 6.1 | | Paper, table 1 |
| RTN | 43.9 / 2e2 | 43.1 / 3e2 | Paper, table 1 |
| GPTQ | 40.6 / 2e2 | 40.5 / 2e2 | Paper, table 1 |
| QuaRot | 63.8 / 7.9 | 63.3 / 8.0 | Paper, table 5 |
| Random Hadamard, R1 and R2 (merged only) | 49.5 | 50.0 | Paper, table 2 |
| Random Hadamard, R1 to R4 | 64.2 | 63.9 | Paper, table 2 |
| SpinQuant, merged rotations only (`no_had`) | 51.9 / 18.6 | 52.6 / 18.6 | Paper, table 1 |
| SpinQuant, with online Hadamard (`had`) | 65.8 / 7.1 | 65.5 / 7.3 | Paper, table 1 |

Notes on the sources:

- All rows are from the paper. The README of the code gives some different numbers: SpinQuant at 4-4-4 is 65.2 there, and RTN at 4-4-16 is 38.5 / 9e2.
- The paper has no 4-16-16 column. The README gives 4-16-16: SpinQuant 68.5 / 6.4, QuaRot 68.4 / 6.4, GPTQ 66.5 / 7.2, RTN 65.4 / 7.8.

Facts about SpinQuant:

- It has four rotations. R1 (residual stream) and R2 (in each attention block) merge into the weights. R3 and R4 are Hadamard rotations that run at inference.
- It learns R1 and R2 with Cayley SGD: 100 iterations on 800 WikiText-2 samples, about 30 minutes for Llama 3 8B. This is gradient training (class D).
- The main-table rows use GPTQ for the weights.
- The 8 zero-shot tasks are BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge and OBQA.
- The paper used an internal Meta codebase. The public code is a Hugging Face reproduction, so our numbers can differ.
- The quantizer of the paper: activations with a dynamic minimum and maximum for each token, with no clipping. GPTQ uses 128 WikiText-2 samples of length 2048.

Not found in the sources:

- Whether the model is the base model or the Instruct model.
- The licence of the SpinQuant code (the README names a different project).

## 3. Target

Decision of the user, 2026-11-10: the plan does not choose the method. The agent decides. It can use merged rotations, online Hadamard rotations, a mix of them, or a different method.

The plan fixes only the bit setting and the number to beat:

| Item | Value |
|---|---|
| Bit setting | 4-4-4: weights, activations and KV cache at 4 bits |
| Number to beat | The best published result at 4-4-4: SpinQuant with online Hadamard, 65.5 / 7.3 |
| Gap to full precision | 4.1 zero-shot points, 1.2 perplexity |

For information: SpinQuant with merged rotations only gets 52.6 / 18.6 at 4-4-4. A random Hadamard for all four rotations gets 63.9. Thus the online rotations give most of the gain, and the learning adds 1.6 points.

Why merged rotations alone are weak: the input of the MLP down projection comes after an element-wise product (SiLU gate times up projection). A rotation does not pass through that product. Thus no rotation that merges into the weights can do the work of the online rotation R4. On that site, only a permutation, a sign change and a positive scale for each channel keep the function. The first draft of this plan proposed "merged rotations only" as the target. That target was close to a dead end.

The other bit settings are not part of this plan.

## 4. Steps

Each step ends with a check. Do not start a step before the check of the step before it passes.

### Step 1: decide the scope

Decide the items of section 6. Do not change this file after the user approves it.

### Step 2: build the stack

1. Build one ROCm image with PyTorch, `transformers`, `lm-eval` and the SpinQuant code. Pin each version and the code commit.
2. The SpinQuant code uses the `fast-hadamard-transform` package, which needs a CUDA build. Replace it with a Hadamard transform in plain PyTorch. The MLP width is 14336 = 28 × 512, which is not a power of two. Thus the replacement needs the size-28 Hadamard matrix of the QuaRot and SpinQuant code. Check the replacement against a matrix product for the sizes 128, 4096 and 14336.
3. Pin the model: the base model `meta-llama/Meta-Llama-3-8B` and its revision. The paper does not say base or Instruct.
4. Fix the evaluation settings: sequence length 2048 for perplexity, the `lm-eval` version, and the metric of each of the 8 tasks.
5. Write the complete quantizer configuration of the SpinQuant code as a list: which layers it quantizes, which layers stay in full precision, the points of the activation quantizers, the KV-cache quantizer, and the GPTQ settings. Step 5 uses this list.
6. Check the licence file of the SpinQuant code.

Check: the full-precision base model gives a WikiText-2 perplexity of 6.1 (tolerance 3%). If it does not, test the Instruct model. If neither gives 6.1, our own number is the reference.

### Step 3: reproduce the published numbers

Run these methods ourselves, in the `trainer` container, at 4-4-4. These are the baselines that WSP must beat:

| Baseline | Runs |
|---|---|
| SpinQuant with online Hadamard | 4 seeds |
| QuaRot | 4 seeds |
| Random Hadamard, R1 to R4 | 4 seeds |

Each seed also changes the calibration sample. Run RTN, GPTQ and SpinQuant `no_had` one time each, for information.

Check: each number is inside 3% of the published perplexity and inside 1.0 point of the published zero-shot average. If a number is outside, search for the cause for one working day at most. If the cause stays unknown, continue: our own numbers are then the baseline, and the report states the difference.

Margin (fixed here, before the WSP runs): the larger of 2 standard deviations of the 4 SpinQuant seeds, and a minimum of 0.5 zero-shot points and 0.1 perplexity.

### Step 4: make the test sets

| Set | Content | Who sees it |
|---|---|---|
| Practice | WikiText-2 train (the SpinQuant calibration source) and any public data | The agent |
| Selection | A private text set, for perplexity | Only `eval` |
| Primary test, perplexity | A second private text set of 300,000 tokens or more | Only `eval` |
| Primary test, zero-shot | The 8 public zero-shot tasks, complete | Only `eval`. Public, so the agent can know them |
| Secondary test | WikiText-2 test | Only `eval`. Public |

- The private text gives the perplexity for the accept decision. WikiText-2 gives the comparison with the published numbers.
- The zero-shot tasks stay public. Private items of the same difficulty are not practical, and a small private set gives too much noise. The risk is small: a quantization method does not learn task answers. The private perplexity set is the guard against a fit to public test text.

Check: across the 12 baseline runs of step 3, the perplexity on the private set and on WikiText-2 have a rank correlation of 0.8 or more.

### Step 5: define the rules of the bit setting in `eval`

The agent makes the quantized model itself, with a method of its choice. `eval` does not quantize. It checks only that an export obeys the bit setting. The rules are the same as in the SpinQuant code:

1. **Weights.** Each weight of the linear layers has 4 bits. The scales for each output channel stay in full precision, as in SpinQuant. The layers that SpinQuant keeps in full precision stay permitted in full precision.
2. **Activations.** The input of each linear layer has 4 bits, at the same points as in the SpinQuant code.
3. **KV cache.** The keys and values have 4 bits.
4. **Changes of the model graph.** They are permitted, for example online rotations. The export states each operation that it adds. `eval` records the extra parameters and the extra time for each token.

An export that breaks a rule gets the score of RTN.

Step 2 of the build must list the exact quantization points and the full-precision layers of the SpinQuant code. The agent gets this list.

Check: the SpinQuant models of step 3, sent through this path as exports, pass the rules and give the same numbers as in step 3.

### Step 6: run WSP

- 4 independent editor runs at 4-4-4. Each run has a new context and the same task message. The task message names the goal and the rules. It does not suggest a method.
- The agent gets: the base weights, the practice data, the SpinQuant and QuaRot papers and code, free forward passes, and the rule check of step 5 for its own tests.
- The agent can start from a published result, for example the SpinQuant or QuaRot models of step 3, or it can start from the base model. The result then states that start.
- Permitted: an analysis of the weights (class A), closed-form fits (class B, for example PCA, SVD, least squares, GPTQ), and a search over 20 numbers or fewer for each run.
- The search limit counts each searched variable, a number or a discrete choice. Example: a choice of one of two rotations for each of 32 layers is 32 variables. A solver with a fixed closed-form update in each step, and no step that accepts or rejects by a score, is class B.
- Not permitted: Cayley SGD, another optimizer, a training loop, fine-tuning.
- Each run can use the `select` call a fixed number of times (proposal: 10). The run then names one export.
- The harness keeps the code of each run. The gradient scan and the supervisor check it.

Check: each run names one export, and the code check finds no gradient training.

### Step 7: final test and decision

1. The `final` call scores the 4 WSP exports and the 4 SpinQuant seeds on the primary test, one time.
2. **Pass:** each WSP run is better than each of the 12 baseline runs of step 3, by the margin from step 3, on the private perplexity and on the zero-shot average.
   - Fallback rule, fixed here: if the 4 WSP runs have a spread larger than 3 times the margin, use a bootstrap interval (95%) on the difference of the means, against the SpinQuant runs. A pass needs an interval above the margin.
3. Report the extra parameters and the extra time for each token of each export, next to the scores. A result with no online operation that passes is a stronger result.
4. Report the secondary test next to it, and the published numbers.

### Step 8: report

1. Write the report: the method of each run in plain words, its class, its start, the numbers, and the cost (API tokens, forward GPU hours).
2. Classify each result: a known method rebuilt, a known method improved, or a new method.

## 5. Containers

The same as `wsp-research-plan.md`, section 4.3: `editor` (no GPU), `forward` (GPU), `trainer` (GPU, baselines only), `eval` (GPU), `ledger`, `proxy`, `supervisor`. RQ2 needs no hidden items, only the private test sets.

## 6. Protocol items

| Item | Value |
|---|---|
| Stop rule of a run | The budget ends, or the agent names its export |
| Budget of a run | API tokens and wall time. Open decision |
| `select` calls | 10 for each WSP run. Each baseline run gets 10 for a scripted tuning of its settings |
| Export type | A quantized checkpoint, with a list of each operation that it adds to the model graph |
| Score of a run with no export | The score of RTN |
| Excluded artifacts | None. Published SpinQuant and QuaRot models are a permitted start |

## 7. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | The target | Agreed 2026-11-10: 4-4-4, the method is free, the number to beat is SpinQuant with online Hadamard |
| 2 | Who quantizes | The agent, with a method of its choice. `eval` checks only the rules of the bit setting (step 5) |
| 3 | The source of the private text sets | Texts from after June 2026 that we collect and do not publish. The agent does not know which texts |
| 4 | The tolerances and the margin of steps 2 and 3, and the number of `select` calls | The proposals in the steps |
| 6 | The budget of a run (API tokens and wall time) | No proposal yet |
| 5 | The supervisor | A human runs RQ2 with scripts, with no supervisor agent. The track is small, and this removes one agent from the result |
