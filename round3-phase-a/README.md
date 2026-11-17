# Round 3 — phase A (automatic-allocation control on Qwen3-32B)

Status: DONE 2026-11-16 01:03 UTC. TIE at both targets, so claim A of round 2 is withdrawn. Phase B (Qwen3.6-27B) is not approved.

| file | content |
|---|---|
| `PLAN.md` | the approved plan, the frozen control, the binding verdict rule |

## Where the rest is

Phase A uses the round-2 harness without changes, so its code stays in the round-2 folder. A copy here could become different from the code that made the results.

| item | location |
|---|---|
| code | `../round2-qwen3-32b/code/r2/alloc.py` (uses `r2/fq.py`, `r2/solvers.py`, `r2/common.py`, `r2/p2.py`) |
| run chain | `../round2-qwen3-32b/code/p3a_chain.sh` |
| environment, data and commands | `../round2-qwen3-32b/RECREATE.md`, section "Round 3 phase A" |
| log, frozen allocation sha256, verdict | `../round2-qwen3-32b/NOTES.md`, sections "Round 3 phase A", "PHASE A VERDICT", "Secondary verdict" |
| results (not in this folder) | `/models/tmp/wsp2/results/p3a/` |

## Recreate in short

```bash
cd ../round2-qwen3-32b/code
./drun.sh 1 bash -c "cd /code && python -m r2.alloc sens" > /models/tmp/wsp2/logs/p3a/sens32.log 2>&1   # about 4.6 h
./p3a_chain.sh   # allocation, then control late34 seeds 0-3, then late44 seeds 0-3
```

Before you run, read `../round2-qwen3-32b/RECREATE.md`. It lists the paths to change, the image and the time rule in `p3a_chain.sh`.
