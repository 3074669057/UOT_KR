"""Deep compare: frozen vs new transport plan structure (density, argmax, truth-edge mass)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"
MINE = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "smoke" / "Celer" / "seed_42"

with np.load(FROZEN / "uot" / "uot_transport_matrix.npz") as z:
    p_f = z["P"]
with np.load(MINE / "uot" / "uot_transport_matrix.npz") as z:
    p_m = z["P"]

print("shapes:", p_f.shape, p_m.shape)
for tag, p in (("frozen", p_f), ("mine", p_m)):
    thr = 1e-9
    pos = (p > thr).sum()
    row_edges = (p > thr).sum(axis=1)
    col_edges = (p > thr).sum(axis=0)
    print(f"{tag}: sum={p.sum():.4f} pos_cells={pos} row_edges q=[{np.quantile(row_edges,0.1):.0f},{np.quantile(row_edges,0.5):.0f},{np.quantile(row_edges,0.9):.0f}] max_row_edges={row_edges.max()}")
    print(f"   row_max_mass q=[{np.quantile(p.max(axis=1),0.25):.4f},{np.quantile(p.max(axis=1),0.5):.4f},{np.quantile(p.max(axis=1),0.75):.4f}]")

# load truth pairs from synthetic labels
import pandas as pd
lab = pd.read_csv(FROZEN / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
# map flow ids to indices via segment CSVs
def ids(csv):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    return {str(r.flow_id): i for i, r in df.iterrows()}
# frozen ids
f_eth_ids = ids(FROZEN / "flow_segments_eth_synth.csv")
f_bnb_ids = ids(FROZEN / "flow_segments_bnb_synth.csv")
m_eth_ids = ids(MINE / "flow_segments_eth_synth.csv")
m_bnb_ids = ids(MINE / "flow_segments_bnb_synth.csv")

for tag, p, ei, bi in (("frozen", p_f, f_eth_ids, f_bnb_ids), ("mine", p_m, m_eth_ids, m_bnb_ids)):
    split_mass, merge_mass, decoy_mass, n_split, n_merge = 0.0, 0.0, 0.0, 0, 0
    miss_split = miss_merge = 0
    for _, r in lab.iterrows():
        s = str(r.src_flow_id); d = str(r.dst_flow_id)
        if s not in ei or d not in bi:
            continue
        m = p[ei[s], bi[d]]
        ls = str(r.label_source or "")
        if "split" in ls:
            n_split += 1; split_mass += m
            if m <= 1e-9: miss_split += 1
        elif "merge" in ls:
            n_merge += 1; merge_mass += m
            if m <= 1e-9: miss_merge += 1
        elif "noise" in ls:
            decoy_mass += m
    print(f"{tag}: truth edges n_split={n_split} n_merge={n_merge} missed_split={miss_split} missed_merge={miss_merge}")
    print(f"   mean mass on truth: split={split_mass/max(n_split,1):.6f} merge={merge_mass/max(n_merge,1):.6f} decoy_total={decoy_mass:.6f}")
