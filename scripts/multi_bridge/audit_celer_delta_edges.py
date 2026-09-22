"""PHASE 6 (analysis): identify the exact Celer seed-46 edges that flipped frozen->new."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_46"
NEW = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "per_seed" / "Celer" / "seed_46"

for tag, root in (("frozen", FROZEN), ("new", NEW)):
    plan = pd.read_csv(root / "uot" / "uot_transport_plan.csv", dtype=str, keep_default_na=False)
    plan["_m"] = pd.to_numeric(plan["transport_mass"], errors="coerce").fillna(0.0)
    pos = set(zip(plan[plan["_m"] > 1e-9]["src_flow_id"], plan[plan["_m"] > 1e-9]["dst_flow_id"]))
    lab = pd.read_csv(root / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
    hits = {}
    for r in lab.itertuples():
        ls = str(r.label_source or "")
        if "split" in ls or "merge" in ls:
            hits[(str(r.src_flow_id), str(r.dst_flow_id))] = (str(r.src_flow_id), str(r.dst_flow_id)) in pos
    print(tag, "structural edges hit:", sum(hits.values()), "/", len(hits))

# diff
def hit_map(root):
    plan = pd.read_csv(root / "uot" / "uot_transport_plan.csv", dtype=str, keep_default_na=False)
    plan["_m"] = pd.to_numeric(plan["transport_mass"], errors="coerce").fillna(0.0)
    pos = set(zip(plan[plan["_m"] > 1e-9]["src_flow_id"], plan[plan["_m"] > 1e-9]["dst_flow_id"]))
    lab = pd.read_csv(root / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
    out = {}
    for r in lab.itertuples():
        ls = str(r.label_source or "")
        if "split" in ls or "merge" in ls:
            out[(str(r.src_flow_id), str(r.dst_flow_id))] = (str(r.src_flow_id), str(r.dst_flow_id)) in pos
    return out

f = hit_map(FROZEN)
n = hit_map(NEW)
flipped = [k for k in f if f[k] != n[k]]
print("\nflipped edges (frozen->new):")
for k in flipped:
    print("  ", k, "frozen hit:", f[k], "new hit:", n[k])
