"""Correctly identify frozen missed templates: which flows exist and their amounts."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"

lab = pd.read_csv(FROZEN / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
plan = pd.read_csv(FROZEN / "uot" / "uot_transport_plan.csv", dtype=str, keep_default_na=False)
plan["_m"] = pd.to_numeric(plan["transport_mass"], errors="coerce").fillna(0.0)
pos = set(zip(plan[plan["_m"] > 1e-9]["src_flow_id"], plan[plan["_m"] > 1e-9]["dst_flow_id"]))

eth = pd.read_csv(FROZEN / "flow_segments_eth_synth.csv", dtype=str, keep_default_na=False)
eth_ids = set(eth["flow_id"].astype(str))
eth_by_id = {str(r.flow_id): float(r.usd_amount_sum) for r in eth.itertuples()}
bnb_ids = set(pd.read_csv(FROZEN / "flow_segments_bnb_synth.csv", dtype=str, keep_default_na=False)["flow_id"].astype(str))

missed_tpls: dict[str, list[tuple[str, float]]] = {}
for r in lab.itertuples():
    ls = str(r.label_source or "")
    if ("split" not in ls and "merge" not in ls):
        continue
    sf, df = str(r.src_flow_id), str(r.dst_flow_id)
    if (sf, df) in pos:
        continue
    tpl = sf.split("__")[0]
    amt = eth_by_id.get(sf, float("nan"))
    missed_tpls.setdefault(tpl, []).append((sf in eth_ids, df in bnb_ids, amt))

for tpl, info in sorted(missed_tpls.items()):
    amts = sorted({a for _, _, a in info})
    print(f"{tpl}: n_edges={len(info)} src_in_csv={all(s for s,_,_ in info)} dst_in_csv={all(d for _,d,_ in info)} amounts={amts}")
