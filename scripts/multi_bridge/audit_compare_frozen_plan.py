"""PHASE 6 (deep): compare frozen-artifact P vs current-code P on the same frozen inputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic"
AUDIT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges_audit"

for seed in (42, 46):
    with np.load(FROZEN / f"synthetic_eval_seed_{seed}" / "uot" / "uot_transport_matrix.npz") as z:
        P_old = z["P"]
    with np.load(AUDIT / "celer_regression_delta" / "frozen_inputs_current_code" / f"seed_{seed}" / "uot" / "uot_transport_matrix.npz") as z:
        P_new = z["P"]
    diff = np.abs(P_old - P_new)
    print(f"seed {seed}: shapes {P_old.shape} {P_new.shape} max_abs_diff={diff.max():.3e} "
          f"cells_diff_gt_1e-12={(diff > 1e-12).sum()} cells_diff_gt_1e-9={(diff > 1e-9).sum()}")
    if seed == 46:
        # which cells differ most
        idx = np.unravel_index(np.argsort(diff.ravel())[-6:], diff.shape)
        for i, j in zip(*idx):
            print(f"   top-diff cell ({i},{j}): old={P_old[i,j]:.3e} new={P_new[i,j]:.3e}")
