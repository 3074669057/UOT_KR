"""PHASE 6 (close): where do frozen vs current time_cost differ (same inputs)? which cells."""
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

d = FROZEN / "synthetic_eval_seed_46"
eth = flows_from_segment_export_csv(d / "flow_segments_eth_synth.csv", chain="ETH")
bnb = flows_from_segment_export_csv(d / "flow_segments_bnb_synth.csv", chain="BNB")
cur = build_cost_matrix_decomposed(eth, bnb, weights=default_cost_weights(), use_graph=False,
                                   max_delay_sec=21600.0, causal_violation_penalty=5.0)
z = np.load(d / "uot" / "uot_cost_matrix.npz")
old_t = np.asarray(z["time_cost"])
old_d = np.asarray(z["delay_sec"])
cur_t = np.asarray(cur["time_cost"])
cur_d = np.asarray(cur["delay_sec"])
print("delay_sec: maxdiff", np.abs(old_d - cur_d).max(), "ndiff cells", (old_d != cur_d).sum(), "/", old_d.size)
print("time_cost: maxdiff", np.abs(old_t - cur_t).max(), "ndiff cells", (np.abs(old_t - cur_t) > 1e-9).sum())
# which delay cells differ
diff = np.abs(old_d - cur_d)
if diff.max() > 0:
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print("max delay diff at", idx, "old", old_d[idx], "new", cur_d[idx])
    # pattern of differing cells
    print("old delay values at diff cells:", np.unique(old_d[diff > 0])[:10], "new:", np.unique(cur_d[diff > 0])[:10])
# causal penalty in old npz?
old_cp = np.asarray(z["causal_penalty"])
print("old causal_penalty: uniq", np.unique(old_cp), "nonzero", (old_cp != 0).sum())
print("cur causal_penalty nonzero:", (np.asarray(cur['causal_penalty']) != 0).sum())
# was causal penalty inside old C_effective? check cells where old_cp>0
m = old_cp > 0
old_C = np.asarray(z["C_effective"])
print("old C at causal cells:", np.unique(np.round(old_C[m], 2))[:10], "mean", old_C[m].mean() if m.sum() else None)
z.close()
