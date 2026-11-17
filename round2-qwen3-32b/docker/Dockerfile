# Round-2 image: same ROCm/torch base as round 1, current transformers, flash-linear-attention (Triton).
FROM rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
RUN pip install --no-cache-dir "transformers==5.22.2" "accelerate" "datasets" "lm_eval" "sentencepiece" "safetensors" "scipy" "protobuf" \
 && pip install --no-cache-dir "flash-linear-attention==0.6.0" \
 && python -c "import torch, transformers, fla; print(torch.__version__, transformers.__version__, fla.__version__)"
