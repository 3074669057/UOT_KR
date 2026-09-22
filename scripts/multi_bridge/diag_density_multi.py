"""Density driver for Multi: cost matrix stats + mass-swap solves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"
OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "smoke"

import ot

with np.load(FROZEN / "uot" / "uot_cost_matrix.npz") as z:
    C_f = z["C_effective"]
df = json.loads((FROZEN / "uot" / "uot_diagnostics.json").read_text(encoding="utf-8"))["solver"]
a_f = np.asarray(df["source_mass_risk_weighted"]); b_f = np.asarray(df["target_mass_evidence_weighted"])

for br in ("Multi", "Poly"):
    with np.load(OUT / br / "seed_42" / "uot" / "uot_cost_matrix.npz") as z:
        C_m = z["C_effective"]
    dm = json.loads((OUT / br / "seed_42" / "uot" / "uot_diagnostics.json").read_text(encoding="utf-8"))["solver"]
    a_m = np.asarray(dm["source_mass_risk_weighted"]); b_m = np.asarray(dm["target_mass_evidence_weighted"])
    print(f"== {br}")
    for tag, c in (("frozen", C_f), ("mine", C_m)):
        v = c.ravel()
        print(f"  {tag} C: mean={v.mean():.4f} q25={np.quantile(v,.25):.4f} q50={np.quantile(v,.5):.4f} min={v.min():.4f} max={v.max():.4f}")
    def dens(C, a, b):
        p = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=0.05, reg_m=0.5, numItermax=2000, stopThr=1e-9)
        return int((p > 1e-9).sum())
    print(f"  mine C + mine masses: {dens(C_m, a_m, b_m)}")
    print(f"  mine C + frozen masses: {dens(C_m, a_f, b_f)}")
    print(f"  frozen C + mine masses: {dens(C_f, a_m, b_m)}")
    # component means
    with np.load(OUT / br / "seed_42" / "uot" / "uot_cost_matrix.npz") as z:
        for comp in ("amount_cost", "time_cost", "route_cost", "risk_cost", "evidence_cost", "address_novelty_cost"):
            if comp in z.files:
                v = np.asarray(z[comp]).ravel()
                print(f"  {comp}: mean={v.mean():.4f} q50={np.quantile(v,.5):.4f}")
