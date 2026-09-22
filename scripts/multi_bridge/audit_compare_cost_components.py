"""PHASE 6 (deep): compare frozen-artifact cost matrix vs current-code cost on the same frozen inputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv  # noqa: E402
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights  # noqa: E402

FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic"

for seed in (42, 46):
    d = FROZEN / f"synthetic_eval_seed_{seed}"
    eth = flows_from_segment_export_csv(d / "flow_segments_eth_synth.csv", chain="ETH")
    bnb = flows_from_segment_export_csv(d / "flow_segments_bnb_synth.csv", chain="BNB")
    cur = build_cost_matrix_decomposed(eth, bnb, weights=default_cost_weights(), use_graph=False,
                                       max_delay_sec=21600.0, causal_violation_penalty=5.0)
    z = np.load(d / "uot" / "uot_cost_matrix.npz")
    old_C = np.asarray(z["C_effective"])
    print(f"== seed {seed}")
    print(f"  C: old mean={old_C.mean():.4f} min={old_C.min():.4f} | current mean={cur['C'].mean():.4f} min={cur['C'].min():.4f} maxdiff={np.abs(old_C-cur['C']).max():.4f}")
    for comp in ("amount_cost", "time_cost", "route_cost", "risk_cost", "evidence_cost", "address_novelty_cost"):
        if comp in z.files:
            old_v = np.asarray(z[comp])
            cur_v = np.asarray(cur[comp])
            print(f"  {comp}: old mean={old_v.mean():.4f} uniq={len(np.unique(np.round(old_v,4)))} | cur mean={cur_v.mean():.4f} uniq={len(np.unique(np.round(cur_v,4)))} | maxdiff={np.abs(old_v-cur_v).max():.4f}")
    z.close()
