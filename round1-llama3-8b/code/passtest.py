#!/usr/bin/env python3
"""Pass test: every WSP run beats every baseline run by the margin.
margin = max(2 * sample std of the 4 SpinQuant seeds, floor); floor = 0.1 ppl / 0.5 zero-shot points."""
import json, statistics as st
R = "/models/tmp/wsp/results"
BASE = ["randhad", "quarot", "spinquant", "spinquantT"]
SEEDS = range(4)

def get(name, m):
    d = json.load(open(f"{R}/{name}.json"))
    return d["zs"]["avg"] if m == "zs" else d["ppl"][m]

for m in ["wikitext2", "c4", "fw25test", "zs"]:
    lower_better = m != "zs"
    wsp = [get(f"wsp_s{s}", m) for s in SEEDS]
    base = {b: [get(f"{b}_s{s}", m) for s in SEEDS] for b in BASE}
    sq = base["spinquant"]
    margin = max(2 * st.stdev(sq), 0.5 if m == "zs" else 0.1)
    if lower_better:
        best_base = min(min(v) for v in base.values()); worst_wsp = max(wsp)
        ok = worst_wsp + margin < best_base
    else:
        best_base = max(max(v) for v in base.values()); worst_wsp = min(wsp)
        ok = worst_wsp - margin > best_base
    print(f"{m:10s} wsp={['%.3f' % x for x in wsp]} mean={st.mean(wsp):.3f}")
    for b, v in base.items():
        print(f"{'':10s} {b:11s}={['%.3f' % x for x in v]} mean={st.mean(v):.3f}")
    if lower_better:
        fp = json.load(open(f"{R}/fp16.json"))["ppl"][m] if __import__("os").path.exists(f"{R}/fp16.json") else None
        if fp:
            gw, gs = st.mean(wsp) - fp, st.mean(sq) - fp
            print(f"{'':10s} gap to FP: wsp {gw:.3f}, spinquant {gs:.3f}, closed {100*(1-gw/gs):.1f}% more")
    print(f"{'':10s} worst_wsp={worst_wsp:.3f} best_baseline={best_base:.3f} margin={margin:.3f} -> {'PASS' if ok else 'FAIL'}\n")
