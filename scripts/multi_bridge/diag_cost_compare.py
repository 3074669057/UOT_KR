"""Compare frozen vs new synthetic cost matrices to diagnose plan density difference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402

FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"
MINE = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "smoke" / "Celer" / "seed_42"

with np.load(FROZEN / "uot" / "uot_cost_matrix.npz") as z:
    print("frozen npz keys:", list(z.keys()))
    c_frozen = z["C_effective"]

eth_f = flows_from_segment_export_csv(FROZEN / "flow_segments_eth_synth.csv", chain="ETH")
bnb_f = flows_from_segment_export_csv(FROZEN / "flow_segments_bnb_synth.csv", chain="BNB")
d_f = build_cost_matrix_decomposed(eth_f, bnb_f, weights=default_cost_weights(), use_graph=False, max_delay_sec=21600.0, causal_violation_penalty=5.0)

eth_m = flows_from_segment_export_csv(MINE / "flow_segments_eth_synth.csv", chain="ETH")
bnb_m = flows_from_segment_export_csv(MINE / "flow_segments_bnb_synth.csv", chain="BNB")
print("mine flows:", len(eth_m), len(bnb_m))
d = build_cost_matrix_decomposed(eth_m, bnb_m, weights=default_cost_weights(), use_graph=False, max_delay_sec=21600.0, causal_violation_penalty=5.0)
c_mine = np.asarray(d["C"], dtype=float)

def stats(name, c):
    c = c.ravel()
    print(f"{name}: shape={c.shape} mean={c.mean():.4f} q25={np.quantile(c,0.25):.4f} q50={np.quantile(c,0.5):.4f} q75={np.quantile(c,0.75):.4f} pct_lt_1.36={(c<1.36).mean():.3f} pct_lt_1.0={(c<1.0).mean():.3f} min={c.min():.4f} max={c.max():.4f}")

stats("frozen", c_frozen)
stats("mine", c_mine)

for comp in ("amount_cost", "time_cost", "route_cost", "risk_cost", "evidence_cost", "address_novelty_cost"):
    for tag, dd in (("mine", d), ("frozen", d_f)):
        if comp in dd and np.asarray(dd[comp]).shape == np.asarray(dd["C"]).shape:
            v = np.asarray(dd[comp]).ravel()
            print(f"{tag} {comp}: mean={v.mean():.4f} q50={np.quantile(v,0.5):.4f} uniq={len(np.unique(np.round(v,4)))}")
