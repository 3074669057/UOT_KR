"""Shared machinery for the K->pi dual-scaling / marginal-pressure causal diagnosis.

Development seeds 201-205 only; amount-free renormalized cost (the hash-locked primary
candidate cost from the previous round); frozen UOT/BOT parameters; D4@5 frozen. Every
intervention here is DIAGNOSTIC ONLY and can never become a candidate or paper method.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

TDS = REPO / "out" / "multi_bridge_expansion" / "transport_dual_scaling_diagnosis"
AF = REPO / "out" / "multi_bridge_expansion" / "amount_free_candidate_dev"

BRIDGES = ("Celer", "Multi", "Poly")
DEV_SEEDS = (201, 202, 203, 204, 205)
HOLDOUT_SEEDS = (301, 302, 303, 304, 305)  # NEVER generated / never read
REG = 0.05
REG_M = 0.5
K5 = 5


def load_cell(bridge: str, seed: int) -> dict[str, Any]:
    """Previous round's amount-free dev cell (C_primary + UOT/BOT plans + marginals)."""
    from baseline_mechanism.common import tpl_maps, truth_structure
    from dev_candidate.af_common import load_dev_cell as _ldc

    cell = _ldc(bridge, seed)
    base = AF / "plans" / bridge / f"seed_{seed}"
    c = np.load(base / "costs.npz", allow_pickle=False)
    u = np.load(base / "uot_primary.npz", allow_pickle=False)
    b = np.load(base / "bot_primary.npz", allow_pickle=False)
    cell["C_primary"] = np.asarray(c["C_primary"], dtype=float)
    cell["P_uot"] = np.asarray(u["P"], dtype=float)
    cell["P_bot"] = np.asarray(b["P"], dtype=float)
    return cell


def rank_asc(S: np.ndarray, axis: int) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if axis == 1:
        R = np.zeros_like(S, dtype=int)
        for i in range(S.shape[0]):
            order = np.lexsort((np.arange(S.shape[1]), S[i]))
            R[i, order] = np.arange(1, S.shape[1] + 1)
        return R
    R = np.zeros_like(S, dtype=int)
    for j in range(S.shape[1]):
        order = np.lexsort((np.arange(S.shape[0]), S[:, j]))
        R[order, j] = np.arange(1, S.shape[0] + 1)
    return R


def effective_scalings(P: np.ndarray, K: np.ndarray, n_iter: int = 500) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Recover u, v (up to a global constant) with P ~= diag(u) K diag(v), over P>0 support,
    by iterative proportional fitting in log space. Returns (u, v, recon_error_stats)."""
    P = np.asarray(P, dtype=float)
    K = np.asarray(K, dtype=float)
    tiny = 1e-300
    mask = P > 1e-15
    # P_ij ~= u_i * K_ij * v_j  =>  log P - log K = log u + log v  on the support
    L = np.log(np.where(mask, P, tiny)) - np.log(np.where(mask, K, tiny))
    L = np.where(mask, L, np.nan)
    u = np.zeros(P.shape[0], dtype=float)
    v = np.zeros(P.shape[1], dtype=float)
    for _ in range(n_iter):
        v = np.nanmean(L - u[:, None], axis=0)
        v = np.where(np.isfinite(v), v, 0.0)
        u = np.nanmean(L - v[None, :], axis=1)
        u = np.where(np.isfinite(u), u, 0.0)
    # NOTE: no gauge re-centering — the decomposition is only defined up to a global
    # constant, and any asymmetric centering would introduce a spurious constant residual.
    resid = np.abs(L - u[:, None] - v[None, :])
    resid = resid[np.isfinite(resid)]
    stats = {"max": float(resid.max()) if resid.size else float("nan"),
             "median": float(np.median(resid)) if resid.size else float("nan"),
             "p95": float(np.percentile(resid, 95)) if resid.size else float("nan")}
    # reconstruction error of the multiplicative form on the support
    rec = np.exp(u)[:, None] * K * np.exp(v)[None, :]
    scale = P.sum() / max(rec.sum(), 1e-300)
    rec = rec * scale
    rel = np.abs(rec - P) / np.maximum(P, 1e-300)
    rel = rel[mask]
    stats["rel_max"] = float(rel.max()) if rel.size else float("nan")
    stats["rel_median"] = float(np.median(rel)) if rel.size else float("nan")
    stats["rel_p95"] = float(np.percentile(rel, 95)) if rel.size else float("nan")
    return np.exp(u), np.exp(v), stats


def d4_edges_on_score(S: np.ndarray, sids: list[str], tids: list[str], k: int = K5) -> list[tuple[str, str]]:
    """D4@k mutual top-k applied to an arbitrary score matrix (DIAGNOSTIC)."""
    rr = rank_asc(-S, 1)
    cr = rank_asc(-S, 0)
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


def role_of(cell: dict[str, Any], idx: int, side: str) -> str:
    """Template role of a node (split_parent / split_child_a/b / merge_parent_a/b /
    merge_dst / unmatched / decoy / other)."""
    sid = cell["sids"][idx] if side == "src" else cell["tids"][idx]
    for t, tr in cell["truth"].items():
        if side == "src":
            if sid in {s for s, _ in tr["split"]}:
                return "split_parent"
            ms = sorted(tr["merge_src"])
            if sid in ms:
                return f"merge_parent_{'a' if ms.index(sid) == 0 else 'b'}"
            if sid in tr["unmatched_src"]:
                return "unmatched"
            if any(sid == s for s, _ in tr["decoy"]):
                return "decoy"
        else:
            split_d = sorted(tr["split_dst"])
            if sid in split_d:
                return f"split_child_{'a' if split_d.index(sid) == 0 else 'b'}"
            if sid in tr["merge_dst"]:
                return "merge_dst"
            if sid in tr["hidden_dst"]:
                return "hidden_dst"
            if any(sid == d for _, d in tr["decoy"]):
                return "decoy"
    return "other"
