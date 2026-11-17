# wsp-quant image: SpinQuant / QuaRot reproduction + lm-eval on MI325X (gfx942), ROCm 7.2.
# Base = same pinned digest as an earlier image of the original machine (torch 2.10.0+rocm7.2.4, py3.12).
#   docker build -t wsp-quant:rocm7.2 .
FROM rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
ENV PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 VIRTUAL_ENV=/opt/venv
RUN pip install --no-cache-dir uv
# Versions from SpinQuant requirement.txt (commit 8f47aa3f) + lm-eval 0.4.4 (contemporary).
# torch is already installed in /opt/venv and satisfies every requirement, so it is kept.
RUN uv pip install --python /opt/venv/bin/python --no-cache \
    "transformers==4.44.2" "accelerate==0.34.2" "datasets==2.20.0" \
    "sentencepiece==0.2.0" "tensorboardX==2.6.2.2" "lm_eval==0.4.4" \
    "scipy==1.14.1" "safetensors==0.4.5" "protobuf==5.28.3"
COPY fast_hadamard_transform /opt/shim/fast_hadamard_transform
ENV PYTHONPATH=/opt/shim
WORKDIR /workspace
