"""Plan step 2.2: shim Hadamard == explicit matrix product for n = 128, 4096, 14336."""
import sys, os, torch
sys.path.insert(0, os.environ.get("SPINQUANT_DIR", "/spinquant"))
from utils import hadamard_utils as hu


def explicit_had(n):
    hadK, K = hu.get_hadK(n)
    m = n // K
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < m:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    if K > 1:
        # matmul_hadU_cuda: x.view(-1,K,m) -> FWHT over m -> hadK @ . => kron(hadK, H_m)
        H = torch.kron(hadK.to(torch.float64), H)
    return H / n ** 0.5


def test_sizes():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    for n in (128, 4096, 14336):
        x = torch.randn(7, n, dtype=torch.float64)
        hadK, K = hu.get_hadK(n)
        y = hu.matmul_hadU_cuda(x.float().to(dev), hadK, K).double().cpu()
        H = explicit_had(n)
        ref = x @ H.T
        err = (y - ref).abs().max().item()
        orth = (H @ H.T - torch.eye(n, dtype=torch.float64)).abs().max().item()
        print(f"n={n} K={K} max_abs_err={err:.2e} orth_err={orth:.2e}")
        assert err < 1e-4 and orth < 1e-9
        # also against SpinQuant's pure-torch reference path
        y2 = hu.matmul_hadU(x.float()).double()
        assert (y2 - ref).abs().max().item() < 1e-4


if __name__ == "__main__":
    test_sizes(); print("PASS")
