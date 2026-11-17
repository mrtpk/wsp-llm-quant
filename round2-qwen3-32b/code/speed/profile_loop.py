"""Profile one eager decode step of decode_loop (bf16, batch 1, ctx 2048): where does the time go?"""
import sys; sys.argv = ["x", sys.argv[1] if len(sys.argv) > 1 else "bf16"]
import torch
sys.path.insert(0, "/code/speed")
import decode_loop as dl
m = dl.Model(dl.load_state(), dl.FMT)
torch.cuda.empty_cache()
m.alloc(1, dl.CTX + 1)
tok = torch.randint(0, dl.V, (1,), device="cuda")
with torch.no_grad():
    for _ in range(3): m.step(tok, dl.CTX)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3): m.step(tok, dl.CTX)
        torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=90))
