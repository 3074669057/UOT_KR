"""Shared machinery for the COST / FEATURE REPRESENTATION + ENTROPIC TRANSPORT diagnosis.

Development seeds: 201-205 (48 templates x 3 bridges, faithful generator, frozen
parameters). Seeds 42-46 are HISTORICAL LOCKED TEST only (never used for any selection
here); seeds 301-305 are reserved holdout and are NEVER generated or read.
All interventions in this round are DIAGNOSTIC ONLY and never become a final method.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

CTD = REPO / "out" / "multi_bridge_expansion" / "cost_transport_diagnosis"
PREV_AUDIT = REPO / "out" / "multi_bridge_expansion" / "decoder_plan_quality_audit"
ATTRIB_AUDIT = REPO / "out" / "multi_bridge_expansion" / "decoder_attribution_audit"

BRIDGES = ("Celer", "Multi", "Poly")
DEV_SEEDS = (201, 202, 203, 204, 205)
HOLDOUT_SEEDS = (301, 302, 303, 304, 305)  # NEVER generated / never read this round

REG_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)   # 0.05 = frozen reference
REGM_GRID = (0.10, 0.25, 0.50, 1.00, 2.00, 5.00)   # 0.50 = frozen reference
FROZEN_REG = 0.05
FROZEN_REGM = 0.5
DECODE_K = 5  # locked D4_mutrank@5 k (never modified)

# Effective frozen cost weights (graph merged into amount because use_graph=False).
FROZEN_WEIGHTS = {"amount": 0.40, "time": 0.25, "route": 0.15, "risk": 0.15,
                  "evidence": 0.05, "novelty": 0.05}
COMP_KEYS = ("amount_cost", "time_cost", "route_cost", "risk_cost", "evidence_cost",
             "address_novelty_cost")


def cost_d4_edges(C: np.ndarray, sids: list[str], tids: list[str], k: int = DECODE_K) -> list[tuple[str, str]]:
    """COST-space D4: mutual top-k by ascending total cost (stable index tie-break)."""
    rr = _rank_asc(C, "row")
    cr = _rank_asc(C, "col")
    edges: list[tuple[str, str]] = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                edges.append((sids[i], tids[j]))
    return edges


def _rank_asc(S: np.ndarray, along: str) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if along == "row":
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


def load_dev_cell(bridge: str, seed: int) -> dict[str, Any]:
    """Frozen-parameter development cell (frozen params, faithful generator)."""
    from baseline_mechanism.common import tpl_maps, truth_structure

    root = CTD / "plans" / "dev" / bridge / f"seed_{seed}"
    c = np.load(root / "cost.npz", allow_pickle=False)
    ids = np.load(root / "ids.npz", allow_pickle=True)
    u = np.load(root / "transport_uot.npz", allow_pickle=True)
    b = np.load(root / "transport_bot.npz", allow_pickle=True)
    labels = pd.read_csv(root / "labels.csv", dtype=str, keep_default_na=False)
    sids = [str(x) for x in ids["sids"]]
    tids = [str(x) for x in ids["tids"]]
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)
    return {
        "bridge": bridge, "seed": seed, "root": root,
        "C": np.asarray(c["C_effective"], dtype=float),
        "components": {k: np.asarray(c[k], dtype=float) for k in COMP_KEYS if k in c},
        "P_uot": np.asarray(u["P"], dtype=float),
        "logu_uot": np.asarray(u["logu"], dtype=float) if "logu" in u else None,
        "logv_uot": np.asarray(u["logv"], dtype=float) if "logv" in u else None,
        "P_bot": np.asarray(b["P"], dtype=float),
        "logu_bot": np.asarray(b["logu"], dtype=float) if "logu" in b else None,
        "logv_bot": np.asarray(b["logv"], dtype=float) if "logv" in b else None,
        "a_rw": np.asarray(u["a_rw"], dtype=float), "b_ev": np.asarray(u["b_ev"], dtype=float),
        "a0": np.asarray(u["a0"], dtype=float) if "a0" in u else None,
        "b0": np.asarray(u["b0"], dtype=float) if "b0" in u else None,
        "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "labels": labels, "truth": truth_structure(labels),
    }


def solve_uot_log(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float, reg_m: float,
                  numItermax: int = 20000) -> dict[str, Any]:
    """POT sinkhorn_unbalanced with log scaling vectors; records convergence. DIAGNOSTIC
    instrumentation only — does not change the transport solution."""
    import ot
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b
    P, log = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=float(reg), reg_m=float(reg_m),
                                               numItermax=numItermax, stopThr=1e-11,
                                               log=True)
    P = np.asarray(P, dtype=float)
    errs = np.asarray(log.get("err", [1.0]), dtype=float)
    # NOTE: POT's returned logu/logv are the raw Knopp iterates and do NOT directly
    # factorize the plan (a rank-1 correction is needed); the effective dual scalings are
    # recovered from P itself in run_kernel_transport_audit.py (IPF in log space).
    out: dict[str, Any] = {
        "P": P, "converged": bool(errs.ravel()[-1] < 1e-7), "final_err": float(errs.ravel()[-1]),
        "logu_raw": np.asarray(log.get("logu", np.full(P.shape[0], np.nan)), dtype=float),
        "logv_raw": np.asarray(log.get("logv", np.full(P.shape[1], np.nan)), dtype=float),
    }
    return out


def solve_bot_log(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float,
                  numItermax: int = 20000) -> dict[str, Any]:
    """Balanced entropic OT with log scaling vectors (mean-rescaled for conditioning;
    the rescaling is exact by entropic-OT scale invariance)."""
    import ot
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.sum() <= 0:
        a = np.ones_like(a) / max(len(a), 1)
    if b.sum() <= 0:
        b = np.ones_like(b) / max(len(b), 1)
    a = a / a.sum()
    b = b / b.sum()
    a = np.where(a < 1e-13, 0.0, a)
    b = np.where(b < 1e-13, 0.0, b)
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b
    kappa = max(float(np.mean(C)), 1e-9)
    P, log = ot.sinkhorn(a, b, C / kappa, reg=float(reg) / kappa, numItermax=numItermax,
                         stopThr=1e-11, log=True, method="sinkhorn")
    P = np.asarray(P, dtype=float)
    errs = np.asarray(log.get("err", [1.0]), dtype=float)
    row_res = float(np.abs(P.sum(axis=1) - a).max())
    col_res = float(np.abs(P.sum(axis=0) - b).max())
    out = {"P": P, "converged": bool(max(row_res, col_res) < 1e-6),
           "final_err": float(errs.ravel()[-1]),
           "row_residual": row_res, "col_residual": col_res,
           "u_pot": np.asarray(log.get("u", np.full(P.shape[0], np.nan)), dtype=float),
           "v_pot": np.asarray(log.get("v", np.full(P.shape[1], np.nan)), dtype=float)}
    return out


def marginal_variants(cell: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n, m = cell["C"].shape
    frozen_a = np.asarray(cell["a_rw"], dtype=float)
    frozen_b = np.asarray(cell["b_ev"], dtype=float)
    amount_a = np.asarray(cell["a0"], dtype=float) if cell.get("a0") is not None else frozen_a
    amount_b = np.asarray(cell["b0"], dtype=float) if cell.get("b0") is not None else frozen_b
    amount_a = amount_a / amount_a.sum() if amount_a.sum() > 0 else np.ones(n) / n
    amount_b = amount_b / amount_b.sum() if amount_b.sum() > 0 else np.ones(m) / m
    uniform_a = np.ones(n) / n
    uniform_b = np.ones(m) / m
    return {"frozen": (frozen_a, frozen_b), "amount": (amount_a, amount_b),
            "uniform": (uniform_a, uniform_b)}
