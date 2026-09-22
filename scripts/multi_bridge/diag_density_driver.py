"""Isolate density driver: solve C matrices with uniform vs real masses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "out" / "paper_full_pipeline_run" / "synthetic" / "synthetic_eval_seed_42"
MINE = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "smoke" / "Celer" / "seed_42"

import ot

with np.load(FROZEN / "uot" / "uot_cost_matrix.npz") as z:
    C_f = z["C_effective"]
with np.load(MINE / "uot" / "uot_cost_matrix.npz") as z:
    C_m = z["C_effective"]

def solve_density(C, a, b):
    p = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=0.05, reg_m=0.5, numItermax=2000, stopThr=1e-9)
    p = np.asarray(p, dtype=float)
    return int((p > 1e-9).sum())

n = C_f.shape[0]
uni_a = np.ones(n) / n
uni_b = np.ones(n) / n

import json
df = json.loads((FROZEN / "uot" / "uot_diagnostics.json").read_text(encoding="utf-8"))["solver"]
dm = json.loads((MINE / "uot" / "uot_diagnostics.json").read_text(encoding="utf-8"))["solver"]
a_f = np.asarray(df["source_mass_risk_weighted"]); b_f = np.asarray(df["target_mass_evidence_weighted"])
a_m = np.asarray(dm["source_mass_risk_weighted"]); b_m = np.asarray(dm["target_mass_evidence_weighted"])

print("frozen C + frozen masses:", solve_density(C_f, a_f, b_f))
print("frozen C + uniform masses:", solve_density(C_f, uni_a, uni_b))
print("mine C + mine masses:    ", solve_density(C_m, a_m, b_m))
print("mine C + uniform masses: ", solve_density(C_m, uni_a, uni_b))
print("mine C + frozen masses:  ", solve_density(C_m, a_f, b_f))
print("frozen C + mine masses:  ", solve_density(C_f, a_m, b_m))
