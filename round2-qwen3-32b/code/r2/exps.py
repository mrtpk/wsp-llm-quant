"""P2 experiment definitions. SWEEPS[name]() -> list of (key, cfg); BUILDS[key] -> cfg for solvers.run_pipeline."""
from r2.fq import lin_regex

ALL = lin_regex()
L63D = lin_regex(layers=[63], types=["down_proj"])
L62D = lin_regex(layers=[62], types=["down_proj"])
L6G = lin_regex(layers=[6], types=["gate_proj"])
BF = {"bits": 16}


def I(b, g=0, sym=False, **kw):  # noqa: E743
    return dict(bits=b, group=g, sym=sym, **kw)


def FP(b, g=0):
    return dict(bits=b, group=g, fmt="fp")


def MXFP(b=4):
    return dict(bits=b, group=32, fmt="mxfp")


KV = r"\d+"


def kvr(k, v, layers=KV):
    return {"kv": layers, "k": k, "v": v}


Kch = lambda b: I(b, 128, axis="channel")    # per channel over groups of 128 tokens (KIVI-style)

# ---------------------------------------------------------------- E1: RTN + precision exceptions

W_EXC = {"none": None, "bf16": BF, "int8": I(8, sym=True), "mxfp4": MXFP(4), "fp4g16": FP(4, 16)}


def e1_w():
    out = []
    for wn, ws in [("W4g128", I(4, 128)), ("W3g128", I(3, 128)), ("W3g64", I(3, 64)), ("W4pc", I(4, sym=True)),
                   ("W3pc", I(3)), ("W2g64", I(2, 64)), ("mxfp4", MXFP(4))]:
        for en in (["none", "bf16", "int8", "mxfp4", "fp4g16"] if wn in ("W4g128", "W3g128") else ["none", "int8"]):
            rules = [{"match": ALL, "w": ws}]
            if W_EXC[en]:
                rules.append({"match": L63D, "w": W_EXC[en]})
            out.append((f"{wn}/L63d={en}", {"rules": rules}))
        if wn in ("W3g128", "W3g64", "W2g64"):
            out.append((f"{wn}/L63d=int8,L62d=W4g128", {"rules": [{"match": ALL, "w": ws}, {"match": L63D, "w": I(8, sym=True)},
                                                                  {"match": L62D, "w": I(4, 128)}]}))
    # KV
    for k, v, nm in [(Kch(4), I(4), "K4ch_V4"), (Kch(4), I(3), "K4ch_V3"), (Kch(4), I(2, 32), "K4ch_V2g32"),
                     (Kch(3), I(3), "K3ch_V3"), (Kch(2), I(2, 32), "K2ch_V2g32"), (I(4, 32), I(3), "K4g32_V3"),
                     (FP(8), FP(8), "KVfp8"), (I(8), I(8), "KV8")]:
        out.append((f"kv/{nm}", {"rules": [kvr(k, v)]}))
    out.append(("kv/K4pt_V4_L0K8", {"rules": [kvr(I(4), I(4)), kvr(I(8), I(4), "0")]}))
    out.append(("kv/K4g32_V3_L0K8", {"rules": [kvr(I(4, 32), I(3)), kvr(I(8), I(3), "0")]}))
    out.append(("kv/K3ch_V3_L0K8", {"rules": [kvr(Kch(3), I(3)), kvr(I(8), I(3), "0")]}))
    return out


A_EXC = {"none": None, "bf16": BF, "fp8": FP(8), "int8": I(8)}


def e1_a():
    out = []
    for an, asp in [("A4g32", I(4, 32)), ("Afp4g16", FP(4, 16)), ("Amxfp4", MXFP(4)), ("A4g128", I(4, 128)),
                    ("A4pt", I(4)), ("A6pt", I(6)), ("A8pt", I(8)), ("Afp8", FP(8)), ("A4g64", I(4, 64))]:
        for en in (["none", "bf16", "fp8", "int8"] if an in ("A4g32", "Afp4g16") else ["fp8"]):
            if an in ("A8pt", "Afp8") and en == "none":
                continue
            rules = [{"match": ALL, "a": asp}]
            if A_EXC[en]:
                rules.append({"match": L6G, "a": A_EXC[en]})
            out.append((f"{an}/L6g={en}", {"rules": rules}))
    # per-token A4 with both critical layers protected (does per-token become usable?)
    out.append(("A4pt/L6g=fp8,L63=fp8", {"rules": [{"match": ALL, "a": I(4)}, {"match": L6G, "a": FP(8)},
                                                    {"match": lin_regex(layers=[63]), "a": FP(8)}]}))
    out.append(("A6pt/L6g=fp8,L63=fp8", {"rules": [{"match": ALL, "a": I(6)}, {"match": L6G, "a": FP(8)},
                                                    {"match": lin_regex(layers=[63]), "a": FP(8)}]}))
    # combos (RTN everything) -- with task-dev proxy
    for wn, ws in [("W4g128", I(4, 128)), ("W3g128", I(3, 128))]:
        for an, asp in [("A4g32", I(4, 32)), ("Afp4g16", FP(4, 16)), ("Afp8", FP(8))]:
            for kn, kvrule in [("KV16", None), ("K4chV3", kvr(Kch(4), I(3)))]:
                rules = [{"match": ALL, "w": ws, "a": asp}, {"match": L63D, "w": I(8, sym=True)}]
                if asp["bits"] < 8:
                    rules.append({"match": L6G, "a": FP(8)})
                if kvrule:
                    rules.append(kvrule)
                out.append((f"combo/{wn}+L63d8/{an}+L6gfp8/{kn}", {"rules": rules, "taskdev": True}))
    out.append(("combo/fp8W8A8/KVfp8", {"rules": [{"match": ALL, "w": FP(8), "a": FP(8)}, kvr(FP(8), FP(8))], "taskdev": True}))
    out.append(("combo/bf16", {"rules": [], "taskdev": True}))
    return out


SWEEPS = {"e1_w": e1_w, "e1_a": e1_a}
BUILDS = {}


# ---------------------------------------------------------------- builds (E2-E5, first wave)

def _build(pipeline, wspec, wexc=(), arules=(), taskdev=True):
    """cost_rules declare the weight formats the solver produced, so cost() can count bits."""
    cost_rules = [{"match": ALL, "w": wspec}] + [{"match": rx, "w": s} for rx, s in wexc] + list(arules)
    return {"pipeline": pipeline, "rules": list(arules), "cost_rules": cost_rules, "taskdev": taskdev}


def _gptq(w, exc=(), **kw):
    return dict(op="gptq", wspec=w, **{"except": [list(e) for e in exc]}, **kw)


L63D8 = [(L63D, I(8, sym=True))]
A4G32_KV = [{"match": ALL, "a": I(4, 32)}, kvr(Kch(4), I(3))]
QUAROT_A = [{"match": ALL, "a": I(4, clip=0.9)}, kvr(I(4, 128), I(4, 128))]
W8A8_INT = [{"match": ALL, "a": I(8, sym=True)}]

BUILDS.update({
    # E2 weight-only
    "gptq_W4g128": _build([_gptq(I(4, 128))], I(4, 128)),
    "gptq_W4g128_L63d8": _build([_gptq(I(4, 128), L63D8)], I(4, 128), L63D8),
    "gptq_W3g128": _build([_gptq(I(3, 128))], I(3, 128)),
    "gptq_W3g128_L63d8": _build([_gptq(I(3, 128), L63D8)], I(3, 128), L63D8),
    "gptq_W3g64_L63d8": _build([_gptq(I(3, 64), L63D8)], I(3, 64), L63D8),
    "gptq_W3g128_L63d8_asym15": _build([_gptq(I(3, 128), L63D8, asym_alpha=0.15)], I(3, 128), L63D8),
    "awq_W4g128": _build([dict(op="scale", mode="awq", wspec=I(4, 128)), dict(op="rtn", rules=[{"match": ALL, "w": I(4, 128)}])], I(4, 128)),
    "awq_gptq_W3g128_L63d8": _build([dict(op="scale", mode="awq", wspec=I(3, 128)), _gptq(I(3, 128), L63D8)], I(3, 128), L63D8),
    "gptq_mxfp4_L63d8": _build([_gptq(MXFP(4), L63D8)], MXFP(4), L63D8),
    # E3 low-rank instead of the 8-bit exception (RTN, fast)
    "rtn_W4g128_L63dlr16": _build([dict(op="rtn", rules=[{"match": ALL, "w": I(4, 128)}]),
                                   dict(op="lowrank", match=L63D, rank=16)], I(4, 128)),
    "rtn_W4g128_L63dlr64": _build([dict(op="rtn", rules=[{"match": ALL, "w": I(4, 128)}]),
                                   dict(op="lowrank", match=L63D, rank=64)], I(4, 128)),
    "rtn_W3g128_L63d62dlr64": _build([dict(op="rtn", rules=[{"match": ALL, "w": I(3, 128)}]),
                                      dict(op="lowrank", match=lin_regex(layers=[62, 63], types=["down_proj"]), rank=64)], I(3, 128)),
    # E4 compute frontier
    "sq05_int8_W8A8": _build([dict(op="scale", mode="sq", fixed_alpha=0.5, wspec=I(8, sym=True), aspec=I(8, sym=True)),
                              dict(op="rtn", rules=[{"match": ALL, "w": I(8, sym=True)}])], I(8, sym=True), arules=W8A8_INT),
    "sqsearch_int8_W8A8": _build([dict(op="scale", mode="sq", wspec=I(8, sym=True), aspec=I(8, sym=True)),
                                  dict(op="rtn", rules=[{"match": ALL, "w": I(8, sym=True)}])], I(8, sym=True), arules=W8A8_INT),
    "gptq_W4g128_L63d8_Afp8": _build([_gptq(I(4, 128), L63D8)], I(4, 128), L63D8, arules=[{"match": ALL, "a": FP(8)}]),
    # E5 W4A4KV4: QuaRot-style baseline vs ours
    "quarot_W4A4KV4": _build([dict(op="rotate", r1=True, r4=True), dict(op="qkrot"), _gptq(I(4, sym=True))], I(4, sym=True), arules=QUAROT_A),
    "ours_W4A4KV4_g32": _build([_gptq(I(4, 128), L63D8)], I(4, 128), L63D8, arules=A4G32_KV),
    "ours_rot_W4A4KV4_g32": _build([dict(op="rotate", r1=True, r4=True), _gptq(I(4, 128), L63D8)], I(4, 128), L63D8, arules=A4G32_KV),
})


def e1_lm():
    """lm_head is now always int8 (p2.apply_lm_head). Re-measure the E1 anchors so E1 numbers can be corrected."""
    keep = {"W4g128/L63d=none", "W4g128/L63d=int8", "W3g128/L63d=int8", "W3g128/L63d=int8,L62d=W4g128",
            "W3g64/L63d=int8", "W2g64/L63d=int8", "mxfp4/L63d=int8", "kv/K4ch_V3"}
    out = [("bf16body", {"rules": [], "taskdev": True})]
    out += [(k, c) for k, c in e1_w() if k in keep]
    out += [(k, c) for k, c in e1_a() if k.startswith("combo/") or k in ("A4g32/L6g=none", "Afp4g16/L6g=none")]
    return out


SWEEPS["e1_lm"] = e1_lm


# ---------------------------------------------------------------- W4A8-FP8 push (user priority, 2026-11-11 ~19:45)
AFP8 = [{"match": ALL, "a": FP(8)}]
BUILDS.update({
    "gptq_W4g128_L63d8_asym15_Afp8": _build([_gptq(I(4, 128), L63D8, asym_alpha=0.15)], I(4, 128), L63D8, arules=AFP8),
    "awq_gptq_W4g128_L63d8_Afp8": _build([dict(op="scale", mode="awq", wspec=I(4, 128)), _gptq(I(4, 128), L63D8)],
                                         I(4, 128), L63D8, arules=AFP8),
    "gptq_W4g64_L63d8_Afp8": _build([_gptq(I(4, 64), L63D8)], I(4, 64), L63D8, arules=AFP8),
})

# W4A8 follow-up (2026-11-12 01:15): AWQ scales + asymmetric GPTQ target combined
BUILDS["awq_gptq_W4g128_L63d8_asym15_Afp8"] = _build([dict(op="scale", mode="awq", wspec=I(4, 128)),
                                                       _gptq(I(4, 128), L63D8, asym_alpha=0.15)], I(4, 128), L63D8, arules=AFP8)


def rotdiag_cfgs():
    """Per-token A4 (clip 0.9, as QuaRot) on a rotated model, by module type; bf16 weights."""
    A4 = I(4, clip=0.9)
    T = lambda *t: lin_regex(types=list(t))
    out = [("rot_only", []), ("A8pt_all", [{"match": ALL, "a": I(8)}]), ("A6pt_all", [{"match": ALL, "a": I(6, clip=0.9)}]),
           ("A4pt_all", [{"match": ALL, "a": A4}])]
    for nm, ts in [("qkv", ("q_proj", "k_proj", "v_proj")), ("o", ("o_proj",)), ("gateup", ("gate_proj", "up_proj")), ("down", ("down_proj",))]:
        out.append((f"A4pt_only_{nm}", [{"match": T(*ts), "a": A4}]))
    out.append(("A4pt_all_but_o", [{"match": ALL, "a": A4}, {"match": T("o_proj"), "a": BF}]))
    out.append(("A4pt_all_L6g_L63_fp8", [{"match": ALL, "a": A4}, {"match": L6G, "a": FP(8)}, {"match": lin_regex(layers=[63]), "a": FP(8)}]))
    out.append(("A4g32_all", [{"match": ALL, "a": I(4, 32)}]))
    out.append(("KV4g128_qkrot", [kvr(I(4, 128), I(4, 128))]))
    return out


def rotdiag2_cfgs():
    """Round 2 of the E5 diagnosis: activations are fine after rotation (A4pt all +13%), so test the weight and KV side (RTN)."""
    W4pc = I(4, sym=True); A4 = I(4, clip=0.9); KVq = kvr(I(4, 128), I(4, 128))
    L63d8 = {"match": L63D, "w": I(8, sym=True)}
    return [("W4pc", [{"match": ALL, "w": W4pc}]),
            ("W4pc_L63d8", [{"match": ALL, "w": W4pc}, L63d8]),
            ("W4g128", [{"match": ALL, "w": I(4, 128)}]),
            ("KV4g128_noqkrot", [KVq]),
            ("W4pc_A4pt", [{"match": ALL, "w": W4pc, "a": A4}]),
            ("W4pc_A4pt_KV4", [{"match": ALL, "w": W4pc, "a": A4}, KVq]),
            ("W4pc_A4pt_KV4_noqkrot", [{"match": ALL, "w": W4pc, "a": A4}, KVq]),
            ("W4pc_L63d8_A4pt_KV4", [{"match": ALL, "w": W4pc, "a": A4}, L63d8, KVq])]


# ---------------------------------------------------------------- E2b: mixed 3/4-bit weights at <= 3.75 bits/weight (user bar +4%, 2026-11-12)
def e2_mix():
    """RTN screen of bit allocations. W3 base everywhere, W4g128 where it matters, L63.down int8. The best 2-3 go to GPTQ-asym builds.
    Evidence for the allocation: P1.3a/P1.4 (v, gate, up, down are the sensitive types; q, k, o are cheap), P1.5 (late layers matter most)."""
    W4 = I(4, 128)
    rng = lambda a, b: list(range(a, b))
    def cfg(base, w4_parts):
        rules = [{"match": ALL, "w": base}]
        for layers, types in w4_parts:
            rules.append({"match": lin_regex(layers=layers, types=types), "w": W4})
        rules.append({"match": L63D, "w": I(8, sym=True)})
        return {"rules": rules, "taskdev": False}
    W3, W3g64 = I(3, 128), I(3, 64)
    MLP = ["gate_proj", "up_proj", "down_proj"]
    out = [
        ("W3/late34-63", cfg(W3, [(rng(34, 64), None)])),
        ("W3/late40-63+v", cfg(W3, [(rng(40, 64), None), (None, ["v_proj"])])),
        ("W3/v,down+gu48-63", cfg(W3, [(None, ["v_proj", "down_proj"]), (rng(48, 64), ["gate_proj", "up_proj"])])),
        ("W3/v+mlp40-63", cfg(W3, [(None, ["v_proj"]), (rng(40, 64), MLP)])),
        ("W3/v,down+all56-63", cfg(W3, [(None, ["v_proj", "down_proj"]), (rng(56, 64), None)])),
        ("W3/0-3,40-63", cfg(W3, [(rng(0, 4), None), (rng(40, 64), None)])),
        ("W3/v,up+down50-63", cfg(W3, [(None, ["v_proj", "up_proj"]), (rng(50, 64), ["down_proj"])])),
        ("W3/v,gate+down50-63", cfg(W3, [(None, ["v_proj", "gate_proj"]), (rng(50, 64), ["down_proj"])])),
        ("W3g64/late44-63", cfg(W3g64, [(rng(44, 64), None)])),
        ("W3g64/v,down", cfg(W3g64, [(None, ["v_proj", "down_proj"])])),
        ("W3g64/v+mlp46-63", cfg(W3g64, [(None, ["v_proj"]), (rng(46, 64), MLP)])),
    ]
    return out


SWEEPS["e2_mix"] = e2_mix


# E2b GPTQ-asym builds of the best RTN mixed allocations (screen e2_mix, 2026-11-12 08:57)
def _mix_build(base, w4_parts):
    exc = [(lin_regex(layers=l, types=t), I(4, 128)) for l, t in w4_parts] + L63D8
    return _build([_gptq(base, exc, asym_alpha=0.15)], base, exc)


BUILDS.update({
    "mix_asym_W3g64_late44": _mix_build(I(3, 64), [(list(range(44, 64)), None)]),
    "mix_asym_W3_late34": _mix_build(I(3, 128), [(list(range(34, 64)), None)]),
    "mix_asym_W3_late40_v": _mix_build(I(3, 128), [(list(range(40, 64)), None), (None, ["v_proj"])]),
})

# Combined: best M allocation + FP8 activations (one selection build before the finals are frozen)
_m = BUILDS["mix_asym_W3g64_late44"]
BUILDS["mix_asym_W3g64_late44_Afp8"] = {**_m, "rules": list(AFP8), "cost_rules": _m["cost_rules"] + list(AFP8)}


# Post-freeze exploration (2026-11-14, after P4; selection data only; does NOT change the round-2 result)
# 1. Is ~3.5 bits/weight possible at +4%, and is task-dev robust across seeds? (NOTES "Zero-shot per seed")
# 2. Does V4 instead of V3 bring ours W4A4KV4 inside the bar? (NOTES "Interim observations" 4)
BUILDS.update({
    "pf_mix_asym_W3_late50": _mix_build(I(3, 128), [(list(range(50, 64)), None)]),
    "pf_mix_asym_W3g64_late58": _mix_build(I(3, 64), [(list(range(58, 64)), None)]),
    "pf_ours_W4A4KV4_g32_V4": _build([_gptq(I(4, 128), L63D8)], I(4, 128), L63D8,
                                     arules=[{"match": ALL, "a": I(4, 32)}, kvr(Kch(4), I(4))]),
})
