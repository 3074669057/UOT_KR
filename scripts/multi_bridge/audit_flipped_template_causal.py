"""PHASE 6 (close): did the flipped template's split edges carry causal penalty in the frozen run?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402

FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic"
d = FROZEN / "synthetic_eval_seed_46"

eth = flows_from_segment_export_csv(d / "flow_segments_eth_synth.csv", chain="ETH")
bnb = flows_from_segment_export_csv(d / "flow_segments_bnb_synth.csv", chain="BNB")
eid = {str(f.get("flow_id")): i for i, f in enumerate(eth)}
tid = {str(f.get("flow_id")): j for j, f in enumerate(bnb)}

lab = pd.read_csv(d / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
tpl_edges = [(str(r.src_flow_id), str(r.dst_flow_id)) for r in lab.itertuples()
             if str(r.label_source or "").replace("semi_synthetic_", "") in ("split", "merge")
             and str(r.src_flow_id).startswith("eth_flow_edd8bb6d46fd4c")]
print("template edges:", tpl_edges)

z = np.load(d / "uot" / "uot_cost_matrix.npz")
cp = np.asarray(z["causal_penalty"])
feas = np.asarray(z["feasible_flag"])
delay = np.asarray(z["delay_sec"])
Ceff = np.asarray(z["C_effective"])
for s, t in tpl_edges:
    i, j = eid[s], tid[t]
    print(f"edge {s.split('__')[-1]} -> {t.split('__')[-1]}: old delay={delay[i,j]:.0f} causal_pen={cp[i,j]:.3f} "
          f"feasible={feas[i,j]} C_eff={Ceff[i,j]:.4f}")
z.close()
print("\nold C_effective max:", Ceff.max(), "cells>1.0:", (Ceff > 1.0).sum())
# current delay for same edges
from cross.domain.uot.delay_policy import flow_pair_delay_sec
for s, t in tpl_edges:
    i, j = eid[s], tid[t]
    dcur = flow_pair_delay_sec(eth[i], bnb[j])
    print(f"current delay for {s.split('__')[-1]}: {dcur:.0f}")
