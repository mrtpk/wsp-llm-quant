# wsp-llm-quant: an agent quantizes Llama-3-8B and Qwen3-32B

This repository holds the code, the plans, the lab notes and the reports of three rounds of LLM quantization research. A coding agent (Claude Code with Opus 5.5) wrote and ran all experiments on AMD MI325X GPUs. I set the goals and the rules, and I approved each decision that changed them.

The series is about weight-space programming: you change a trained network by reading and writing its weights directly, based on what each part computes. Here the agent measured where each linear of the model loses accuracy, and then chose a number format and a bit width for each linear. No round used gradient training.

Write-up: [Weight-Space Programming - Can an agent quantize an LLM better than a recipe?](https://github.com/mrtpk/kaizen/tree/master/learnings/20261117_wsp_llm_quant) (kaizen blog). Report: [`docs/REPORT.md`](docs/REPORT.md).

## Read this first

- **No result files are included.** The result JSONs stayed on the GPU machine. Every number in this repository comes from the lab notes (`NOTES.md`) and the reports. You cannot check a number from a file here; you can only run the experiment again.
- **The quantization is simulated** (fake quantization in bf16). The accuracy numbers do not depend on a kernel. The decode-speed numbers come from our own kernels on one GPU type.
- **No checkpoints and no data are included.** The code downloads the models and the data from Hugging Face.
- **To run again:** about 1 GPU-day for round 1, about 4 GPU-days for round 2, and about 12 GPU-hours for round 3 phase A, on MI325X.

## Results

### Round 1: Llama-3-8B, W4A4KV4, against SpinQuant

| | WikiText-2 ppl | C4 ppl | fw25test ppl | zero-shot avg |
|---|---|---|---|---|
| FP16 | 6.138 | | | 67.09 |
| agent, 4 seeds | 7.210–7.230 | 10.582–10.677 | 10.448–10.519 | 64.33–64.60 |
| best baseline run | 7.347 | 10.907 | 10.788 | 63.33 |
| margin needed | 0.100 | 0.100 | 0.112 | 2.198 |
| result | pass | pass | pass | **fail** |

FAIL under the protocol: it needed both perplexity and zero-shot. The agent stayed inside SpinQuant's pipeline and tuned its settings, without a measurement of the model.

### Round 2: Qwen3-32B, free method and format

Δ% perplexity against bf16, seed means. Bar: +4 % on WikiText-2 and fw25test, zero-shot ≥ 68.94. W = stored bits per weight; B = W bits × A bits (BitOps, not a speed measure).

| config | W | A | B | WikiText-2 | fw25test | zero-shot | result |
|---|---|---|---|---|---|---|---|
| gptq_W4g128_L63d8_asym15_Afp8 | 4.266 | 8.20 | 35.7 | +1.75 | +1.60 | 69.94 | B win |
| awq_gptq_W4g128_L63d8_asym15_Afp8 | 4.266 | 8.20 | 35.7 | +1.76 | +1.49 | 70.67 | B win |
| mix_asym_W3_late34 | 3.743 | 16 | 59.9 | +2.48 | +2.37 | 70.23 | M win |
| mix_asym_W3g64_late44 | 3.689 | 16 | 59.0 | +2.77 | +2.45 | 69.94* | M win |
| base: GPTQ W4g128 | 4.250 | 16 | 68.0 | +2.12 | +1.92 | 70.10 | pass |
| base: FP8 W8A8 | 8.002 | 8.20 | 65.6 | +0.55 | +0.48 | 70.93 | pass |
| base: QuaRot-style W4A4KV4 | 4.100 | 4.29 | 18.7 | +25.65 | +21.24 | 64.96 | fail |

\* One seed is below the zero-shot guard. Under the worst-seed guard of round 3, the late44 configs fail. The first bar was +1 %; on selection data only 8-bit weights passed it, so the bar moved to +4 % before any test data was read. Full table: `docs/REPORT.md`, section 2.5.

### Round 3 phase A: the agent's bit allocation against a greedy KL allocation

The control quantizes each of the 448 linears alone, measures the KL divergence of the output, and fills the bit budget greedily. Same formats, same GPTQ pipeline, same seeds, same bits per weight.

| W 3.7432 | WikiText-2 | fw25test | zero-shot worst seed |
|---|---|---|---|
| agent late34 | +2.479 ± 0.179 | +2.367 ± 0.141 | 69.98 |
| control | +2.491 ± 0.075 | +2.390 ± 0.077 | 70.09 |

TIE at both targets (3.7432 and 3.6891). Under the binding rule, a tie withdraws round 2's claim A (accuracy per bit). Claim B (format and structural findings) stays.

## Contents

| Path | Content |
|---|---|
| `docs/REPORT.md` | The report of all three rounds. Start here. |
| `round1-llama3-8b/` | Round 1: `NOTES.md` (results, ablation, retrospective), `wsp-rq2-quantization-plan.md` (the protocol), `ROUND2_HANDOFF.md` (lessons), `code/`, `docker/`. |
| `round1-llama3-8b/code/wspq/` | `wsp.py` (the agent's method), `run_ptq.py` (SpinQuant baselines with our evaluation), `optimize_rotation_seeded.py` (SpinQuant rotation training with a seed fix). |
| `round2-qwen3-32b/` | Round 2: `PLAN.md`, `methodology.md`, `NOTES.md` (the full lab log, also of round 3 phase A), `REPORT.md` (claims A and B, the addendum), `RECREATE.md` (how to run again). |
| `round2-qwen3-32b/code/r2/` | `fq.py` (simulated quantization), `solvers.py` (GPTQ, AWQ, SmoothQuant, rotation, low-rank), `profile.py` (P1 profiling), `p2.py`, `exps.py`, `finals.py`, `p4.py`, `alloc.py` (round 3 control), `tasks/` (task-dev proxy). |
| `round2-qwen3-32b/code/speed/` | Decode loop and W4 Triton kernels. Frozen, see `FROZEN.md`. |
| `round2-qwen3-32b/code/queue_*.tsv` | The exact job lines of each phase. |
| `round2-qwen3-32b/docker/` | Dockerfile and `requirements-lock.txt` (`pip freeze` of the image). |
| `round2-qwen3-32b/data_sha256.txt` | SHA-256 of each cached data file, to check your data. |
| `round2-qwen3-32b/logs/` | ROCm check logs before round 2. |
| `round3-phase-a/` | `PLAN.md` (the frozen control and the verdict rule). The code is `round2-qwen3-32b/code/r2/alloc.py`. |

Not included: the result JSONs, the checkpoints, the data caches, the SpinQuant source, the earlier draft of the round-3 plan, and a note of my instructions on research behaviour.

## Setup

| item | value |
|---|---|
| GPU | AMD MI325X (gfx942), one GPU per job |
| base image | `rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e` (torch 2.10.0+rocm7.2.4, Python 3.12) |
| round 1 | `round1-llama3-8b/docker/Dockerfile`, `meta-llama/Meta-Llama-3-8B` @ `8cde5ca8`, SpinQuant @ `8f47aa3f00e8662caf1a484153920a07e5281c3a` |
| round 2 and 3 | `round2-qwen3-32b/docker/Dockerfile`, `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137` |

1. Round 1 needs SpinQuant. Clone it and mount it at `/spinquant` (see `round1-llama3-8b/code/drun.sh`):
   `git clone https://github.com/facebookresearch/SpinQuant && git -C SpinQuant checkout 8f47aa3f00e8662caf1a484153920a07e5281c3a`
2. The round-2 Dockerfile pins only transformers and flash-linear-attention. For the same versions, install `round2-qwen3-32b/docker/requirements-lock.txt` in the image.
3. `HF_TOKEN` must be in the environment. `drun.sh` passes it into the container.
4. Change the machine paths before you run: the `RN` GPU map in `drun.sh`, and `/models/...` (Hugging Face cache, data, results, logs). `round2-qwen3-32b/RECREATE.md`, section 2, lists every file.
5. Run the jobs through the queue: `cd round2-qwen3-32b/code && ./queue.sh 1 queue_p0.tsv`. The order of the phases and the commands are in `RECREATE.md`, section 4.

Some scripts read results of earlier steps: `p4_eval.py` and `speed/roofline.py` need `results/p4/`. Run them after P4.

## Licence

The code and the documents of this repository: MIT (`LICENSE`).

## Licences of third-party material

- SpinQuant: CC BY-NC 4.0. It is not included. The round-1 scripts import it at run time, and `wspq/wsp.py`, `rotate_and_wrap`, repeats the wrapper setup steps of SpinQuant's `eval_utils/main.py:ptq_model`.
- `round1-llama3-8b/docker/fast_hadamard_transform/`: original pure-PyTorch code. It uses the package name of Dao-AILab's `fast-hadamard-transform` so that SpinQuant imports it.
- `round2-qwen3-32b/code/r2/tasks/`: task configs in the lm-evaluation-harness format (MIT), with its prompt templates, on the train splits.
- `meta-llama/Meta-Llama-3-8B`: Meta Llama 3 Community License. `Qwen/Qwen3-32B`: Apache-2.0. Not included.
- Data (WikiText-2, C4, FineWeb, the zero-shot tasks): downloaded by the code; see each dataset card.

## Reference

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming), Takehi, Teoh, Zhang and Khattab, MIT CSAIL.
- [SpinQuant](https://github.com/facebookresearch/SpinQuant), [QuaRot](https://github.com/spcl/QuaRot), GPTQ, AWQ, SmoothQuant.
