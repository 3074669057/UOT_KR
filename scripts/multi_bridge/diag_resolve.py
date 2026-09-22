"""Re-solve the frozen cost matrix with pot and compare density vs the frozen P."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"
MINE = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "smoke" / "Celer" / "seed_42"

with np.load(FROZEN / "uot" / "uot_cost_matrix.npz") as z:
    C_f = z["C_effective"]
with np.load(FROZEN / "uot" / "uot_transport_matrix.npz") as z:
    P_f = z["P"]
with np.load(MINE / "uot" / "uot_transport_matrix.npz") as z:
    P_m = z["P"]

diag = json.loads((FROZEN / "uot" / "uot_diagnostics.json").read_text(encoding="utf-8"))
sol = diag["solver"]
a = np.asarray(sol["source_mass_risk_weighted"], dtype=float)
b = np.asarray(sol["target_mass_evidence_weighted"], dtype=float)
print("frozen a sum:", a.sum(), "b sum:", b.sum(), "shapes:", a.shape, b.shape)

import ot
for reg, regm in ((0.05, 0.5), (0.05, 0.5)):
    p = ot.unbalanced.sinkhorn_unbalanced(a, b, C_f, reg=reg, reg_m=regm, numItermax=2000, stopThr=1e-9)
    p = np.asarray(p, dtype=float)
    print(f"re-solved frozen C: sum={p.sum():.4f} pos={(p>1e-9).sum()} vs frozen P pos={(P_f>1e-9).sum()}")
    print(f"  max abs diff vs frozen P: {np.abs(p-P_f).max():.3e}")

# also solve MY cost matrix with my masses to sanity check my plan
diag_m = json.loads((MINE / "uot" / "uot_diagnostics.json").read_text(encoding="utf-8"))
sol_m = diag_m["solver"]
a_m = np.asarray(sol_m["source_mass_risk_weighted"], dtype=float)
b_m = np.asarray(sol_m["target_mass_evidence_weighted"], dtype=float)
with np.load(MINE / "uot" / "uot_cost_matrix.npz") as z:
    C_m = z["C_effective"] if "C_effective" in z.files else None
print("mine cost npz keys:", None if C_m is None else "ok")
p_m2 = ot.unbalanced.sinkhorn_unbalanced(a_m, b_m, C_m, reg=0.05, reg_m=0.5, numItermax=2000, stopThr=1e-9)
print(f"re-solved mine C: pos={(p_m2>1e-9).sum()} vs stored mine P pos={(P_m>1e-9).sum()}")
print(f"  max abs diff vs mine P: {np.abs(p_m2-P_m).max():.3e}")
