#!/bin/bash
# drun.sh GPU_ID cmd...   -- run a command in the wsp-quant container on one physical GPU.
# SGLang variant (2026-11-15, decode-speed step c): same GPU mapping as drun.sh, image rocm/sgl-dev (SGLang 0.5.19, aiter).
# Physical GPU -> render node map (measured 2026-11-05, GPU map of the original machine).
set -euo pipefail
declare -A RN=([0]=136 [1]=152 [2]=144 [3]=128)
G=$1; shift
CODE=${CODE:-$(cd "$(dirname "$0")" && pwd)}   # this code/ folder
exec docker run --rm --name "wsp2sgl-g${G}-$$" \
  --device /dev/kfd --device /dev/dri/renderD${RN[$G]} \
  --security-opt seccomp=unconfined --ipc host --shm-size 32g \
  -e HIP_VISIBLE_DEVICES=0 -e HF_HOME=/models/huggingface -e HF_TOKEN \
  -e PYTHONPATH=/code/speed -e WSP_DATA=/models/tmp/wsp2/data \
  -e TOKENIZERS_PARALLELISM=false -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e OMP_NUM_THREADS=8 -e HF_DATASETS_TRUST_REMOTE_CODE=1 \
  -v /models:/models -v $CODE:/code \
  -v /models/tmp/wsp2/results:/results -w /code \
  --entrypoint "" rocm/sgl-dev:v0.5.19-rocm724-mi30x-20260907 "$@"
