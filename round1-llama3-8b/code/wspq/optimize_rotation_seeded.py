"""SpinQuant optimize_rotation.py with one fix: the seed reaches everything.

Upstream: ptq parser consumes --seed, so TrainingArguments.seed stays 42 (data order),
and R1/R2 init happens before Trainer seeds torch -> init is unseeded. Here --seed S
sets transformers.set_seed(S) before model build and TrainingArguments.seed = S.
Run with torchrun --nproc_per_node=1; use --gradient_accumulation_steps 8 to match
the paper's 8 GPUs x bs1 x 100 steps = 800 samples.
"""
import sys, os
sys.path.insert(0, os.environ.get("SPINQUANT_DIR", "/spinquant"))
import transformers
import optimize_rotation as orot
from utils import process_args as pa

_orig = pa.process_args_ptq


def _seeded():
    m, t, p = _orig()
    t.seed = p.seed
    t.data_seed = p.seed
    transformers.set_seed(p.seed)
    return m, t, p


orot.process_args_ptq = _seeded

if __name__ == "__main__":
    orot.train()
