"""Threshold Many-Match baseline for the structural mechanism study.

Rule: predicted edge (i,j) exists iff ``normalized_cost(i,j) <= tau``, decided
independently per cell, so 0 / 1 / many matches per source are all admissible
(1->N, N->1, N->M, and unmatched are representable).

Normalization: ``normalized_cost = ECDF_cal(C_ij)`` where ``ECDF_cal`` is the
empirical CDF of the calibration cost distribution (seeds 101-103, all three
bridges). The threshold grid is {0.05, 0.10, ..., 0.95} and the selected tau is
fixed on the calibration set ONLY (edge F1 criterion); it is never re-tuned on
the test seeds 42-46. The normalized-cost rule is exactly equivalent to
``C_ij <= Q_cal(tau)`` for the calibration quantile function Q_cal.

CLI (for inspection / small grids):
  python run_threshold_many_match.py --cost cost.npz --tau 0.3 --ecdf cal_ecdf.npz --out edges.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import STUDY  # noqa: E402

TAU_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))  # 0.05 .. 0.95


def ecdf_from_costs(costs: np.ndarray) -> dict[str, Any]:
    """Build a pre-registered cost ECDF from the calibration distribution.

    Returns ``{sorted: np.ndarray, min, max}``; ``ecdf(x) = fraction of sorted <= x``,
    ``cutoff(tau) = sorted[int(tau*n)]`` (the tau-quantile cost).
    """
    vals = np.asarray(costs, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    sorted_vals = np.sort(vals)
    return {
        "sorted": sorted_vals,
        "n": int(sorted_vals.size),
        "min": float(sorted_vals[0]) if sorted_vals.size else 0.0,
        "max": float(sorted_vals[-1]) if sorted_vals.size else 0.0,
    }


def cutoff_for_tau(ecdf: dict[str, Any], tau: float) -> float:
    sorted_vals = np.asarray(ecdf["sorted"], dtype=float)
    if sorted_vals.size == 0:
        return 0.0
    k = int(round(tau * sorted_vals.size)) - 1
    k = max(0, min(k, sorted_vals.size - 1))
    return float(sorted_vals[k])


def predict_edges(C: np.ndarray, cutoff: float, sids: list[str] | None = None,
                  tids: list[str] | None = None) -> list[tuple[str, str]]:
    C = np.asarray(C, dtype=float)
    edges: list[tuple[str, str]] = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if float(C[i, j]) <= cutoff:
                s = sids[i] if sids is not None else str(i)
                t = tids[j] if tids is not None else str(j)
                edges.append((s, t))
    return edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--ecdf", default=None, help="npz with key 'sorted' (calibration ECDF)")
    ap.add_argument("--sids", default=None, help="npz with key 'ids' (sources)")
    ap.add_argument("--tids", default=None, help="npz with key 'ids' (targets)")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    C = np.load(cli.cost)["C_effective"] if "C_effective" in np.load(cli.cost).files else None
    if C is None:
        C = np.load(cli.cost)["C"]
    cutoff = cli.cutoff
    if cutoff is None:
        if cli.tau is None or cli.ecdf is None:
            raise SystemExit("need --cutoff or (--tau + --ecdf)")
        ecdf = ecdf_from_costs(np.load(cli.ecdf)["sorted"])
        cutoff = cutoff_for_tau(ecdf, float(cli.tau))
    sids = tids = None
    if cli.sids:
        sids = [str(x) for x in np.load(cli.sids)["ids"]]
    if cli.tids:
        tids = [str(x) for x in np.load(cli.tids)["ids"]]
    edges = predict_edges(C, float(cutoff), sids, tids)
    pd.DataFrame([{"src_flow_id": s, "dst_flow_id": d, "cost": float(C[i, j])}
                  for i, (s, d) in enumerate(edges)], columns=["src_flow_id", "dst_flow_id", "cost"]).to_csv(
        cli.out, index=False)
    print(f"threshold-MM cutoff={cutoff:.6f} edges={len(edges)} -> {cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
