"""PHASE 6 (analysis): detail the Celer seed-46 flipped template."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_46"
NEW = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "per_seed" / "Celer" / "seed_46"

TPL = "eth_flow_edd8bb6d46fd4c"

for tag, root in (("frozen", FROZEN), ("new", NEW)):
    eth = pd.read_csv(root / "flow_segments_eth_synth.csv", dtype=str, keep_default_na=False)
    bnb = pd.read_csv(root / "flow_segments_bnb_synth.csv", dtype=str, keep_default_na=False)
    s = eth[eth["flow_id"].astype(str).str.startswith(TPL)]
    d = bnb[bnb["flow_id"].astype(str).str.startswith("bnb_flow_aa338ad6c65749")]
    print(f"== {tag}")
    print("  src flows:")
    print(s[["flow_id", "usd_amount_sum", "start_time", "aml_score_mean", "evidence_quality_mean", "address_set"]].to_string(index=False))
    print("  dst flows:")
    print(d[["flow_id", "usd_amount_sum", "start_time", "aml_score_mean", "evidence_quality_mean", "address_set"]].to_string(index=False))
    # plan mass on this template's split edges + top dsts for split_src
    plan = pd.read_csv(root / "uot" / "uot_transport_plan.csv", dtype=str, keep_default_na=False)
    plan["_m"] = pd.to_numeric(plan["transport_mass"], errors="coerce").fillna(0.0)
    sub = plan[plan["src_flow_id"].astype(str).str.startswith(TPL + "__synth_split_src")]
    sub = sub.sort_values("_m", ascending=False)
    print("  top transport for split_src:")
    print(sub[["dst_flow_id", "transport_mass", "cost_amount", "cost_time", "cost_risk", "cost_evidence"]].head(6).to_string(index=False))
    print()
