# Round 3 plan — PHASE A: automatic-allocation control on Qwen3-32B (APPROVED 2026-11-15; DONE 2026-11-16 01:03 UTC: TIE on both targets → claim A withdrawn)

Decision (user, relayed by general-ags, 2026-11-15):
- Round 3 starts with phase A on Qwen3-32B, the round-2 model. Its calibration, baselines, seeds and frozen rules exist.
- **Phase B (round 2 plus the control on Qwen3.6-27B) is NOT approved and has no compute plan.** It happens only if phase A gives an agent win, and the user decides it after the phase-A verdict.
- The earlier draft (not published) is superseded. Its kernel-shape rule is DROPPED: kernel existence is a reported column only, never a filter. The round-2 formats stay.
- **No kernel work and no speed experiments.**

Code: `../round2-qwen3-32b/code/r2/alloc.py` (same harness, same image, GPU 1). Results: `/models/tmp/wsp2/results/p3a/`. Log: NOTES.md (round 2 folder, section "Round 3 phase A").

## 1. Question

Does the agent's per-layer bit allocation (round-2 M final) beat a standard automatic allocation that has:
- the same format menu;
- the same GPTQ pipeline;
- the same seeds;
- the same bits per weight?

This decides claim A of the round-2 report ("accuracy per bit").

## 2. The control (all knobs fixed here, before any control result exists)

1. **Format menu:** the formats used by the agent's M finals, all through the same GPTQ solver:
   - W3 g128 asym (3.148 bits);
   - W3 g64 asym (3.297);
   - W4 g128 asym (4.156);
   - INT8 per-channel sym (8.003).

   The lm_head is int8 in every config (counted), as in round 2. bf16 is not on the menu: the agent's finals do not use it.
2. **Sensitivity:**
   - For each of the 448 decoder linears and each menu format, quantize that one linear alone with the round-2 GPTQ solver: act_order, damp 0.01, Hessian from the round-2 calibration set (128 × 2048, half WikiText-2 train and half C4 train, seed 0). All other linears stay bf16. (In isolation the asymmetric target has no effect, because the inputs are exact.)
   - Measure the **KL(bf16 ‖ quantized) of the final next-token distribution**, mean over all tokens of 8 calibration windows (WikiText-2 train + C4 train, seed 4242, disjoint from the calibration seeds 0–3). **Fix 2026-11-15, before any control result:** the whole sensitivity step runs in fp32, because in bf16 any change of one linear gave the same KL floor (≈ 1e-3); 16 → 8 windows to keep the budget (NOTES, "Sensitivity run 1").
   - No selection data and no test data are used.
3. **Allocation:**
   - Start with every linear at the cheapest format (W3 g128).
   - Per linear, keep the lower convex hull of its (bits, KL) points.
   - Greedy fill: always take the hull upgrade with the largest KL drop per added stored bit. If it does not fit, skip that linear and continue, until no upgrade fits.
   - The budget is the agent final's W, counted with the round-2 cost function (lm_head included). This assumes the KL of single linears adds up, as HAWQ and EXL2-style allocators also assume.
4. **Target:** |W_control − W_agent| ≤ 0.01.
   - **Primary:** `mix_asym_W3_late34`, W 3.7432. It is the only M final that passes the worst-seed zero-shot guard.
   - **Secondary, only if the budget remains:** `mix_asym_W3g64_late44`, W 3.6891. Its agent final fails the worst-seed guard (68.92), so at this budget the agent cannot win. Only a tie or a control win is possible.
5. **Pipeline per control:** identical to the agent's M finals. It is one GPTQ op with base W3 g128, per-linear exceptions from the allocation, asym_alpha 0.15, act_order, 128 calibration windows, and the seed set on the op. The lm_head is int8.
6. **Freeze:** the allocation file (and its sha256) is written to NOTES **before** the first control evaluation. Then 4 seeds (0–3, the same as the agent finals) are run on the test sets once, with the round-2 P4 evaluation: WikiText-2 test, fw25test, 8 zero-shot tasks, and c4dev for reference.

## 3. Verdict rule (binding)

For each test set, d = mean Δ%(control) − mean Δ%(agent), over 4 seeds each. s = pooled seed std = sqrt((s_agent² + s_control²) / 2), using the sample std.
- **AGENT WINS** if d > 2s on both test sets AND the agent final passes the zero-shot guard on the seed mean and on the worst seed (≥ 68.94).
- **CONTROL WINS** if −d > 2s on both test sets.
- **Otherwise it is a TIE. A tie WITHDRAWS claim A.**

The verdict is reported in all three cases. Also reported: whether the control passes the +4% bar and the +2% sensitivity row, its zero-shot mean and worst seed, and the per-type allocation it chose against the agent's.

## 4. Carried rules

- The +4% bar is unchanged, with a +2% sensitivity row.
- Worst-seed zero-shot guard.
- Test sets are touched once, after the freeze.
- Seed-spread headroom: not needed for the control, because it has no selection step.

## 5. Budget (GPU 1 only, one job at a time; ≈ 10–15 GPU-hours)

| step | estimate |
|---|---|
| sensitivity: 1792 GPTQ solves + partial forwards | ≈ 3–4.5 h (measured on the first layers, then confirmed in NOTES) |
| primary control: 4 seeds × ≈ 60–65 min | ≈ 4.3 h |
| secondary control | ≈ 4.3 h, only if the total stays ≤ 15 h |

If the sensitivity step makes the total exceed ≈ 15 h, the secondary control is dropped. That decision is recorded.

## 6. Stop conditions

- If the control cannot be built as specified, stop and report. Examples: the greedy fill cannot reach the target within 0.01, or a solver failure that would change the method.
- Report the verdict when phase A is done.
