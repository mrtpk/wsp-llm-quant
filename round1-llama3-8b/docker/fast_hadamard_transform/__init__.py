"""Pure-PyTorch stand-in for Dao-AILab/fast-hadamard-transform (CUDA-only).

hadamard_transform(x, scale=1.0) = scale * x @ H_n along the last dim, where H_n is
the unnormalized Sylvester Hadamard matrix and n is a power of two. Computed in
float32 with a cached matrix, then cast back to the input dtype.
"""
import torch

__version__ = "0.0-torch-shim"
_CACHE: dict = {}


def _sylvester(n, device):
    key = (n, str(device))
    H = _CACHE.get(key)
    if H is None:
        H = torch.ones(1, 1, dtype=torch.float32, device=device)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _CACHE[key] = H
    return H


def hadamard_transform(x, scale=1.0):
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"size {n} is not a power of two"
    H = _sylvester(n, x.device)
    y = x.float() @ H  # H is symmetric
    if scale != 1.0:
        y = y * scale
    return y.to(x.dtype)
