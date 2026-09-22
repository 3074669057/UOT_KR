"""Marginal-pressure minimal intervention matrix (DIAGNOSTIC ONLY; dev 201-205).

M0: original a, original b (reuse the previous round's amount-free UOT plan).
M1: uniform a, original b.   M2: original a, uniform b.   M3: uniform a, uniform b.
Same amount-free kernel K, same reg/reg_m/solver/decoder everywhere. Purpose: locate
source vs destination marginal pressure in the K->pi ranking destruction. Uniform
marginals are a causal intervention ONLY — never a candidate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from diag.ctd_common import solve_uot_log  # noqa: E402
from diag2.tds_common import BRIDGES, DEV_SEEDS, REG, REG_M, TDS, load_cell  # noqa: E402


def main() -> int:
    out = TDS / "marginal_pressure"
    out.mkdir(parents=True, exist_ok=True)
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_cell(bridge, seed)
            C = cell["C_primary"]
            n, m = C.shape
            a0 = np.asarray(cell["a_rw"], dtype=float)
            b0 = np.asarray(cell["b_ev"], dtype=float)
            a0 = a0 / a0.sum()
            b0 = b0 / b0.sum()
            uni_a = np.ones(n) / n
            uni_b = np.ones(m) / m
            variants = {"M1_uniform_a": (uni_a, b0), "M2_uniform_b": (a0, uni_b),
                        "M3_uniform_ab": (uni_a, uni_b)}
            for vname, (a, b) in variants.items():
                cache = out / f"{bridge}_{seed}_{vname}.npz"
                if cache.is_file():
                    continue
                r = solve_uot_log(a, b, C, REG, REG_M)
                np.savez(cache, P=r["P"], a=a, b=b)
                (out / f"{bridge}_{seed}_{vname}_meta.json").write_text(json.dumps({
                    "variant": vname, "converged": r["converged"],
                    "final_err": r["final_err"], "diagnostic_only": True,
                    "residual_mass": float(max(0.0, 1.0 - r["P"].sum()))}, indent=1) + "\n",
                    encoding="utf-8")
                print(f"[marg-pressure] {bridge} s{seed} {vname} conv={r['converged']}",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
