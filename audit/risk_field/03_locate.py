"""Step 1c: list all paths for each key risk-bearing basename, grouped by top dir.

Run:  python D:\\trae\\tool\\a\\cross\\_risk_audit_tmp\\03_locate.py
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "discovery.json"), encoding="utf-8") as f:
    d = json.load(f)

KEYS = [
    "flow_segments_eth.csv", "flow_segments_bnb.csv",
    "uot_marginals.csv", "src_flows_aggregated.csv",
    "candidate_eth_universe_flows.csv", "candidate_bnb_universe_flows.csv",
    "evidence_eth.csv", "evidence_bnb.csv",
    "flows.json",
]
want = set(k.lower() for k in KEYS)

by_base = defaultdict(list)
for h in d["strict_column_hits"]:
    b = os.path.basename(h["path"]).lower()
    if b in want:
        by_base[b].append(h)

for k in KEYS:
    items = by_base.get(k.lower(), [])
    print(f'### {k}   ({len(items)} copies)')
    # show the distinct "prefix" dirs, top 25 by shortest path
    seen = set()
    items_sorted = sorted(items, key=lambda x: (len(x["rel"]), x["rel"]))
    shown = 0
    for h in items_sorted:
        # collapse the deepest varying dir: keep path
        if h["rel"] in seen:
            continue
        seen.add(h["rel"])
        print(f'    {h["rel"]}  [{h["size"]} B]')
        shown += 1
        if shown >= 40:
            print(f'    ... ({len(items) - shown} more)')
            break
    print()
