# How to recreate round 2 (Qwen3-32B) and round 3 phase A

This folder contains all code, configs and run commands for round 2 and round 3 phase A. It contains no checkpoints, no data and no results. The code makes all data from public Hugging Face datasets. The results go to a results folder outside this folder.

Written 2026-11-17. Results, decisions and dates are in `NOTES.md`. The method is in `methodology.md`.

## 1. Environment

| item | value |
|---|---|
| GPU | AMD MI325X (gfx942), one GPU per job, one job at a time |
| main image | `wsp-quant2:rocm7.2`, built from `docker/Dockerfile` (image id `sha256:33540fcdb013…`) |
| base image | `rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e` |
| exact package versions | `docker/requirements-lock.txt` (`pip freeze` of the image, 2026-11-17). Main versions: torch 2.10.0+rocm7.2.4, transformers 5.22.2, lm_eval 0.4.13, datasets 5.2.0, accelerate 1.15.0, flash-linear-attention 0.6.0 |
| SGLang image (speed step c only) | `rocm/sgl-dev:v0.5.19-rocm724-mi30x-20260907` (`rocm/sgl-dev@sha256:994593299b63cdd21fcdc2b863493f794e781f40c9b71881b86b1df63acd7b6b`) |
| model | `Qwen/Qwen3-32B` at revision `9216db5781bf21249d130ec9da846c4624c16137` (set in `code/r2/common.py`) |

The Dockerfile pins only transformers and flash-linear-attention. To get the same versions, install from the lock file:

```bash
cd docker
docker build -t wsp-quant2:rocm7.2 .
# for exact versions, add to the Dockerfile:  COPY requirements-lock.txt /tmp/ && pip install -r /tmp/requirements-lock.txt
```

`HF_TOKEN` must be in the environment. `drun.sh` passes it into the container. Do not write it into a file.

## 2. Local paths you must change

The run scripts contain the paths of the original machine. Change them before you run:

| file | path | meaning |
|---|---|---|
| `code/drun.sh`, `code/drun_sgl.sh` | `CODE=` (defaults to the folder of the script) | this `code/` folder |
| `code/drun.sh`, `code/drun_sgl.sh` | `declare -A RN=(...)` | GPU index to `/dev/dri/renderD*` map of your machine |
| `code/drun.sh`, `code/drun_sgl.sh`, `code/queue.sh`, `code/start_queues.sh`, `code/p3a_chain.sh`, `code/p4_eval.py`, `code/speed/roofline.py` | `/models/tmp/wsp2/{data,results,logs}` | data cache, results, logs |
| all | `/models/huggingface` | Hugging Face cache (`HF_HOME`) |

## 3. Data

`code/r2/common.py` makes every data set on first use and caches it in `WSP_DATA` (`/models/tmp/wsp2/data`):

| set | source |
|---|---|
| WikiText-2 test and train | `Salesforce/wikitext`, `wikitext-2-raw-v1` |
| C4 validation (c4dev = first 64 windows) | `allenai/c4`, `en/c4-validation.00000-of-00008.json.gz` |
| C4 train (calibration) | `allenai/c4`, `en/c4-train.00000-of-01024.json.gz`, sampled with the seed in the file name |
| fw25sel, fw25test | `HuggingFaceFW/fineweb`, crawl CC-MAIN-2025-26, shard `000_00000` (first half / second half of the documents) |
| task-dev proxy | `code/r2/tasks/*.yaml` (500 train-split items each of boolq, arc_easy, arc_challenge, piqa, winogrande) |

The dataset revisions are not pinned. To check that your data is the same, compare with `data_sha256.txt` (sha256 of each cached `.pt` file of the original run).

## 4. Run order and commands

All jobs run through the queue. A queue line is `NAME<TAB>COMMAND`. The command runs inside the container. A job is done when `results/NAME.done` exists.

```bash
cd code
./queue.sh 1 queue_p0.tsv     # GPU 1, one job at a time
```

| step | queue or command | results |
|---|---|---|
| P0 harness checks, bf16 reference | `queue_p0.tsv` | `p0/` |
| P1 profiling | `queue_p1.tsv` | `profile/` |
| P2 exploration, baselines, mixed builds, QuaRot diagnosis | `queue_p2.tsv` (33 jobs) | `p2/`, `p2b_*` |
| P4 confirmation on the test sets (55 runs) | `queue_p4.tsv` | `p4/<config>/s<seed>.json` |
| P4 summary table | `python code/p4_eval.py` (no GPU) | `p4_summary/summary.json` |
| post-freeze exploration (selection data only) | `queue_pf.tsv` | `pf_*` |

Rules: run P4 only after the freeze in `NOTES.md` ("P4 freeze"). Use the test sets one time.

### Round 3 phase A (automatic-allocation control)

Plan: `../round3-phase-a/PLAN.md`. Code: `code/r2/alloc.py`.

```bash
cd code
./drun.sh 1 bash -c "cd /code && python -m r2.alloc sens"   # per-linear KL sensitivity, fp32, about 4.6 h -> results/p3a/sens32.json
./p3a_chain.sh                                               # alloc -> control late34 s0-3 -> control late44 s0-3
```

- `p3a_chain.sh` waits until no `wsp2-g1-*` container runs, then checks for "sens done" in `logs/p3a/sens32.log`. If you start `sens` by hand, write its output to that log: `./drun.sh 1 bash -c "cd /code && python -m r2.alloc sens" > /models/tmp/wsp2/logs/p3a/sens32.log 2>&1`.
- The chain has a time rule for the secondary target (start before 23:15 UTC on day 15). This rule was for the original budget. To run the secondary, run its 4 seeds by hand: `./drun.sh 1 bash -c "cd /code && python -m r2.alloc run mix_asym_W3g64_late44 S"` for S = 0..3.
- Check the allocation against the frozen sha256 values in `NOTES.md` ("Allocation FROZEN"): late34 `4bb18887…`, late44 `bbf21bc9…`.
- `results/p3a/sens.json` is the stopped bf16 run (2 layers). It is a record only. The current code does not make it.

### Decode speed (after the freeze; `code/speed/` is frozen)

The commands were not written down at the time. The commands below come from the scripts and the logs. Run them in this order, because the loop reads the tuned kernel configs:

```bash
cd code
./drun.sh 1 python speed/roofline.py          # -> speed_roofline.json (needs results/p4/)
./drun.sh 1 python speed/bench_gemv.py        # -> speed_gemv_configs.json
./drun.sh 1 python speed/bench_gemv4.py       # -> speed_gemv4_configs.json
./drun.sh 1 python speed/bench_w4.py          # -> speed_w4_configs.json
./drun.sh 1 python speed/bench_w4t.py         # -> speed_w4t_configs.json
for f in bf16 fp8 int8 w4a16 w4a8; do ./drun.sh 1 python speed/decode_loop.py $f; done   # -> speed_loop_<f>.json
```

SGLang reference (bf16 and FP8), `sglang.bench_one_batch` with batch 1 and 16, input 2048, output 64:

```bash
./drun_sgl.sh 1 python -m sglang.bench_one_batch --model-path Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --batch-size 1 16 --input-len 2048 --output-len 64 --result-filename /results/speed_sgl_bf16.jsonl
# FP8: the same command with --quantization fp8 and --result-filename /results/speed_sgl_fp8.jsonl
```

The `--quantization fp8` flag is inferred (the log shows 35.2 GB of weights for FP8 against 61.1 GB for bf16). It is not recorded.

Diagnostic scripts with results only in `NOTES.md`: `speed/check_a.py`, `check_a2.py`, `check_gptq.py`, `check_compile.py`, `diag_w4.py`, `profile_loop.py`. `speed/decode_loop_v2_backup.py` is an older loop (logs `loop_*_v2.log`). It is not used for the reported numbers.

### Other

| item | command | result |
|---|---|---|
| ROCm and model checks before round 2 | `./drun.sh 1 python rocm_check.py Qwen/Qwen3-32B` (and Qwen3.6-27B, with and without `--no-fla`; see NOTES "ROCm check") | `rc_*.pt`, `logs/rc_*.log` |

## 5. What you cannot recreate exactly

- **Bit-exact numbers.** GPU kernels are not deterministic run to run. Expect small differences, below the seed spread in `NOTES.md`.
- **Short debug checks.** Some checks in `NOTES.md` ran as one-off commands, for example the bf16 KL floor check in phase A and the RTN check of L63.down. The code of each fix is in `code/`. The one-off commands are not.
- **Data, if the Hugging Face datasets change.** Use `data_sha256.txt` to check.
