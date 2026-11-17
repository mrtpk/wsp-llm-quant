"""P4 frozen set: finals + baselines. Fixed on 2026-11-12 BEFORE any test data was read (see NOTES.md "P4 freeze").
Do not add, remove or change an entry after the first P4 result exists. Selection used c4dev + task-dev only.
SEEDS: number of calibration seeds per entry (decided rule: 4 per final, >= 8 for baselines with calibration randomness, RTN 1).
"""
from r2.exps import BUILDS, ALL, I, FP

_rtn = lambda rules: {"pipeline": [], "rules": rules, "cost_rules": rules}

BASELINES = {
    # M frontier
    "rtn_W4g128": _rtn([{"match": ALL, "w": I(4, 128)}]),
    "gptq_W4g128": BUILDS["gptq_W4g128"],
    "awq_W4g128": BUILDS["awq_W4g128"],
    "gptq_W3g128": BUILDS["gptq_W3g128"],
    # B frontier
    "quarot_W4A4KV4": BUILDS["quarot_W4A4KV4"],
    "sqsearch_int8_W8A8": BUILDS["sqsearch_int8_W8A8"],
    "fp8_W8A8": _rtn([{"match": ALL, "w": FP(8), "a": FP(8)}]),
    "rtn_int8_W8A8": _rtn([{"match": ALL, "w": I(8, sym=True), "a": I(8, sym=True)}]),
}

FINALS = {
    # M: target W <= 3.75 inside +4%
    "mix_asym_W3g64_late44": BUILDS["mix_asym_W3g64_late44"],
    "mix_asym_W3_late34": BUILDS["mix_asym_W3_late34"],
    # B: W4A8-FP8
    "gptq_W4g128_L63d8_asym15_Afp8": BUILDS["gptq_W4g128_L63d8_asym15_Afp8"],
    "awq_gptq_W4g128_L63d8_asym15_Afp8": BUILDS["awq_gptq_W4g128_L63d8_asym15_Afp8"],
    # W4A4KV4, type-(a) report against QuaRot
    "ours_W4A4KV4_g32": BUILDS["ours_W4A4KV4_g32"],
}
# Conditional final, rule fixed at 11:30 UTC before its selection result exists:
# added only if c4dev <= +3.0% AND wt2train <= +3.5% (>= 0.5 point headroom to the +4% bar on both proxies).
CONDITIONAL = {"mix_asym_W3g64_late44_Afp8": BUILDS["mix_asym_W3g64_late44_Afp8"]}

import json as _json, os as _os
_p = "/results/p2/build/mix_asym_W3g64_late44_Afp8.json"
if _os.path.exists(_p):
    _r = _json.load(open(_p))
    if _r["c4dev"] <= 1.030 * 9.150500956608944 and _r["wt2train"] <= 1.035 * 7.623679020956867:
        FINALS.update(CONDITIONAL)

FROZEN = {**BASELINES, **FINALS}
SEEDS = {k: (1 if not v["pipeline"] else 8) for k, v in BASELINES.items()}
SEEDS.update({k: 4 for k in FINALS})
# User decision 2026-11-12 ~11:45 UTC: 2 seeds for the two baselines that fail the bar by a wide margin on selection data.
SEEDS.update({"gptq_W3g128": 2, "quarot_W4A4KV4": 2})
