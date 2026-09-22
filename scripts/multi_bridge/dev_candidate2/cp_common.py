"""Dual-cancelled conditional-plan decoder candidate — shared machinery.

Frozen upstream: amount-free renormalized cost, frozen marginals, frozen UOT/BOT plans
(reused, never re-solved), D4 k=5. Only the correspondence RANKING SCORE changes:
  S_row(i,j) = P_ij / c_j   (cancels destination dual v; keeps source competition)
  S_col(i,j) = P_ij / r_i   (cancels source dual u; keeps destination competition)
Zero realized-mass endpoints: no support (ranked last); plain float64, no tunable epsilon.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

CP = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_candidate_dev"
AF = REPO / "out" / "multi_bridge_expansion" / "amount_free_candidate_dev"
TDS = REPO / "out" / "multi_bridge_expansion" / "transport_dual_scaling_diagnosis"

BRIDGES = ("Celer", "Multi", "Poly")
DEV_SEEDS = (201, 202, 203, 204, 205)
HOLDOUT_SEEDS = (301, 302, 303, 304, 305)  # NEVER generated / never read
K5 = 5


def load_cell(bridge: str, seed: int) -> dict[str, Any]:
    from baseline_mechanism.common import tpl_maps, truth_structure
    from dev_candidate.af_common import load_dev_cell as _ldc

    cell = _ldc(bridge, seed)
    c = np.load(AF / "plans" / bridge / f"seed_{seed}" / "costs.npz", allow_pickle=False)
    u = np.load(AF / "plans" / bridge / f"seed_{seed}" / "uot_primary.npz", allow_pickle=False)
    b = np.load(AF / "plans" / bridge / f"seed_{seed}" / "bot_primary.npz", allow_pickle=False)
    cell["C_primary"] = np.asarray(c["C_primary"], dtype=float)
    cell["P_uot"] = np.asarray(u["P"], dtype=float)
    cell["P_bot"] = np.asarray(b["P"], dtype=float)
    for plan, tag in (("UOT", "UOT"), ("BOT", "BOT")):
        z = np.load(TDS / "plans" / f"{bridge}_{seed}_{tag}.npz", allow_pickle=True)
        cell[f"K_{plan.lower()}"] = np.asarray(z["K"], dtype=float)
        cell[f"u_{plan.lower()}"] = np.asarray(z["u"], dtype=float)
        cell[f"v_{plan.lower()}"] = np.asarray(z["v"], dtype=float)
    return cell


def rank_desc(S: np.ndarray, axis: int) -> np.ndarray:
    """Dense rank, 1 = largest value; stable index tie-break."""
    S = np.asarray(S, dtype=float)
    if axis == 1:
        R = np.zeros_like(S, dtype=int)
        for i in range(S.shape[0]):
            order = np.lexsort((np.arange(S.shape[1]), -S[i]))
            R[i, order] = np.arange(1, S.shape[1] + 1)
        return R
    R = np.zeros_like(S, dtype=int)
    for j in range(S.shape[1]):
        order = np.lexsort((np.arange(S.shape[0]), -S[:, j]))
        R[order, j] = np.arange(1, S.shape[0] + 1)
    return R


def conditional_scores(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S_row = np.zeros_like(P, dtype=float)
    S_col = np.zeros_like(P, dtype=float)
    pos_c = c > 0
    pos_r = r > 0
    S_row[:, pos_c] = P[:, pos_c] / c[None, pos_c]
    S_col[pos_r, :] = P[pos_r, :] / r[pos_r, None]
    return S_row, S_col


def conditional_edges(P: np.ndarray, sids: list[str], tids: list[str],
                      k: int = K5) -> list[tuple[str, str]]:
    S_row, S_col = conditional_scores(P)
    rr = rank_desc(S_row, 1)
    cr = rank_desc(S_col, 0)
    edges = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                edges.append((sids[i], tids[j]))
    return edges


def mutual_top5_edges(S: np.ndarray, sids: list[str], tids: list[str],
                      k: int = K5) -> list[tuple[str, str]]:
    rr = rank_desc(S, 1)
    cr = rank_desc(S, 0)
    edges = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                edges.append((sids[i], tids[j]))
    return edges


def gt_cells(cell: dict[str, Any]) -> list[tuple[int, int]]:
    out = []
    for t, tr in cell["truth"].items():
        for s, d in tr["positive"]:
            out.append((cell["sids"].index(s), cell["tids"].index(d)))
    return list(set(out))


def role_of_edge(cell: dict[str, Any], i: int, j: int) -> str:
    s, d = cell["sids"][i], cell["tids"][j]
    for t, tr in cell["truth"].items():
        if (s, d) in tr["split"]:
            return "split"
        if (s, d) in tr["merge"]:
            return "merge"
        if (s, d) in tr["decoy"]:
            return "decoy"
    return "other"
