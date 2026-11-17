"""Self-check 3b: compiled vs eager decode step on REAL text states (KV filled by the real 48-token prefix)."""
import sys; fmt = sys.argv[1]; sys.argv = ["x", fmt]
import torch, torch.nn.functional as F
sys.path.insert(0, "/code/speed")
import decode_loop as dl
ids = dl.text_ids()
m = dl.Model(dl.load_state(), fmt); torch.cuda.empty_cache()
step = torch.compile(m.step, dynamic=False)
for B in (1, 16):
    m.alloc(B, dl.CTX + 1)
    with torch.no_grad():
        for p in range(40):
            m.step(ids[p:p + 1].expand(B), p)
        res = []
        for p in range(40, 46):          # 6 positions; each pos is one specialization of the compiled graph
            t = ids[p:p + 1].expand(B)
            ye = m.step(t, p); yc = step(t, p)
            res.append((((yc - ye).norm() / ye.norm()).item(), (yc.argmax(-1) == ye.argmax(-1)).float().mean().item(),
                        F.kl_div(yc.log_softmax(-1), ye.log_softmax(-1), log_target=True, reduction="batchmean").item()))
    print(f"{fmt} B={B} real text, compiled vs eager: rel err max {max(r[0] for r in res):.2e}, top-1 agree min {min(r[1] for r in res):.3f}, KL max {max(r[2] for r in res):.2e}", flush=True)
