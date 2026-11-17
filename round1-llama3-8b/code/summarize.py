#!/usr/bin/env python3
"""Print a table of all result JSONs in /models/tmp/wsp/results."""
import glob, json, os
R = "/models/tmp/wsp/results"
cols = ["proxy", "wikitext2", "c4", "fw25sel", "fw25test", "zs_avg"]
rows = []
for f in sorted(glob.glob(f"{R}/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    p = d.get("ppl", {})
    r = {"name": os.path.basename(f)[:-5], "proxy": d.get("proxy_ppl"),
         **{k: p.get(k) for k in cols[1:5]}, "zs_avg": (d.get("zs") or {}).get("avg")}
    rows.append(r)
print(f"{'name':32s}" + "".join(f"{c:>11s}" for c in cols))
for r in rows:
    print(f"{r['name']:32s}" + "".join(f"{r[c]:11.3f}" if isinstance(r[c], float) else f"{'-':>11s}" for c in cols))
