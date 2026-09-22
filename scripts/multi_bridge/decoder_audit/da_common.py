"""Shared machinery for the decoder / plan-quality / cost audit.

Everything here consumes FROZEN artifacts only:
- test seeds 42-46: frozen RC-UOT-Q transport plans (faithful pipeline) + the Balanced-OT
  plans computed in the previous study (same C, same marginals, strictly balanced);
- calibration seeds 101-103: plans generated with the SAME frozen feature pipeline and the
  SAME frozen UOT parameters (allowed: "可以使用完全相同的 frozen feature pipeline + 完全
  相同的 frozen UOT 参数生成 calibration plans。这不属于调参").

No UOT parameter, cost weight, feature builder, or synthetic task is modified anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

AUDIT = REPO / "out" / "multi_bridge_expansion" / "decoder_plan_quality_audit"
PREV_STUDY = REPO / "out" / "multi_bridge_expansion" / "structural_baseline_mechanism_study"
FROZEN = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"

BRIDGES = ("Celer", "Multi", "Poly")
CALIB_SEEDS = (101, 102, 103)
TEST_SEEDS = (42, 43, 44, 45, 46)

# Frozen RC-UOT-Q parameters (read-only reference; never modified)
FROZEN_PARAMS = {
    "uot_reg": 0.05, "uot_reg_m": 0.5, "uot_lambda_risk": 0.25,
    "uot_decode_threshold": 1e-9, "uot_max_delay_sec": 21600.0,
    "uot_causal_violation_penalty": 5.0, "uot_backend": "pot",
    "uot_allow_unmatched": True, "uot_use_graph_embedding": False,
}

# Frozen Threshold-MM calibration (previous study; NOT re-tuned this round)
THRESHOLD_MM_CUTOFF = 0.47762288884480164

# Pre-registered decoder grids (fixed before calibration)
ROW_TAUS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
K_GRID = (2, 3, 4, 5)      # k=1 is diagnostic-only (reintroduces one-to-one)
D5_TAUS = (0.01, 0.02, 0.05, 0.10)
D6_P = (0.50, 0.70, 0.80, 0.90, 0.95)
D6_GATE = (0.01, 0.02, 0.05)

DECODER_COMPLEXITY = {"D1": 1, "D2": 2, "D3": 2, "D4": 3, "D5": 4, "D6": 5}


def decoder_configs() -> list[dict[str, Any]]:
    """All (pre-registered) decoder configurations: name, rule, params."""
    out: list[dict[str, Any]] = [{"name": "D0_legacy", "family": "D0", "params": {"thr": 1e-9}}]
    for t in ROW_TAUS:
        out.append({"name": f"D1_row_share@{t}", "family": "D1", "params": {"tau": t}})
    for t in ROW_TAUS:
        out.append({"name": f"D2_joint@{t}", "family": "D2", "params": {"tau": t}})
    for t in ROW_TAUS:
        out.append({"name": f"D3_geo@{t}", "family": "D3", "params": {"tau": t}})
    for k in K_GRID:
        out.append({"name": f"D4_mutrank@{k}", "family": "D4", "params": {"k": k}})
    for k in K_GRID:
        for t in D5_TAUS:
            out.append({"name": f"D5_mutrank@{k}_share@{t}", "family": "D5",
                        "params": {"k": k, "tau": t}})
    for p in D6_P:
        for g in D6_GATE:
            out.append({"name": f"D6_cum@{p}_gate@{g}", "family": "D6",
                        "params": {"p": p, "gate": g}})
    return out


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_test_uot(bridge: str, seed: int) -> dict[str, Any]:
    """Frozen RC-UOT-Q instance (C_effective, components, P, marginals, labels, truth)."""
    from baseline_mechanism.common import load_frozen_instance, truth_structure

    inst = load_frozen_instance(bridge, seed)
    inst["truth"] = truth_structure(inst["labels"])
    return inst


def load_test_bot(bridge: str, seed: int) -> dict[str, Any]:
    """Balanced-OT plan from the previous study (same C/marginals as the frozen UOT run)."""
    from baseline_mechanism.common import load_frozen_instance, truth_structure

    inst = load_frozen_instance(bridge, seed)
    pz = np.load(PREV_STUDY / "per_seed" / bridge / f"seed_{seed}" / "balanced_ot_transport.npz",
                 allow_pickle=False)
    inst["P"] = np.asarray(pz["P"], dtype=float)
    inst["truth"] = truth_structure(inst["labels"])
    return inst


def load_cal_plans(bridge: str, seed: int) -> dict[str, Any]:
    """Calibration-seed plans generated this round (same frozen pipeline/parameters)."""
    from baseline_mechanism.common import truth_structure

    root = AUDIT / "plans" / "calibration" / bridge / f"seed_{seed}"
    c = np.load(root / "cost.npz", allow_pickle=False)
    ids = np.load(root / "ids.npz", allow_pickle=True)
    u = np.load(root / "transport_uot.npz", allow_pickle=True)
    b = np.load(root / "transport_bot.npz", allow_pickle=False)
    labels = pd.read_csv(root / "labels.csv", dtype=str, keep_default_na=False)
    sids = [str(x) for x in ids["sids"]]
    tids = [str(x) for x in ids["tids"]]
    from baseline_mechanism.common import tpl_maps
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)
    return {
        "bridge": bridge, "seed": seed, "root": root,
        "C": np.asarray(c["C_effective"], dtype=float),
        "components": {k: np.asarray(c[k], dtype=float) for k in
                       ("amount_cost", "time_cost", "delay_sec", "route_cost", "risk_cost",
                        "evidence_cost", "address_novelty_cost") if k in c},
        "P_uot": np.asarray(u["P"], dtype=float),
        "a_rw": np.asarray(u["a_rw"], dtype=float), "b_ev": np.asarray(u["b_ev"], dtype=float),
        "P_bot": np.asarray(b["P"], dtype=float),
        "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "labels": labels, "truth": truth_structure(labels),
    }


def load_stress_cell(ladder: str, level: str, bridge: str, seed: int) -> dict[str, Any]:
    from baseline_mechanism.common import truth_structure

    root = PREV_STUDY / "stress" / "instances" / ladder / level / bridge / f"seed_{seed}"
    c = np.load(root / "cost.npz", allow_pickle=False)
    ids = np.load(root / "ids.npz", allow_pickle=True)
    labels = pd.read_csv(root / "labels.csv", dtype=str, keep_default_na=False)
    sids = [str(x) for x in ids["sids"]]
    tids = [str(x) for x in ids["tids"]]
    from baseline_mechanism.common import tpl_maps
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)
    u = np.load(root / "transport_RC_UOT_Q.npz", allow_pickle=True)
    b = np.load(root / "transport_Balanced_OT.npz", allow_pickle=True)
    return {
        "bridge": bridge, "seed": seed, "C": np.asarray(c["C_effective"], dtype=float),
        "P_uot": np.asarray(u["P"], dtype=float),
        "P_bot": np.asarray(b["P"], dtype=float),
        "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "labels": labels, "truth": truth_structure(labels),
    }


# --------------------------------------------------------------------------- #
# Continuous scores
# --------------------------------------------------------------------------- #

def score_matrices(inst: dict[str, Any], P: np.ndarray) -> dict[str, np.ndarray]:
    """Six continuous scores (A-F of the audit spec)."""
    C = inst["C"]
    row = P.sum(axis=1, keepdims=True)
    col = P.sum(axis=0, keepdims=True)
    row_share = P / np.maximum(row, 1e-300)
    col_share = P / np.maximum(col, 1e-300)
    return {
        "neg_total_cost": -C,
        "raw_pi": P,
        "row_share": row_share,
        "col_share": col_share,
        "min_share": np.minimum(row_share, col_share),
        "geo_share": np.sqrt(row_share * col_share),
    }


# --------------------------------------------------------------------------- #
# Decoder family D0-D6 (all take the transport plan; shared by Balanced-OT / RC-UOT-Q)
# --------------------------------------------------------------------------- #

def _rank_matrix(score: np.ndarray, along: str) -> np.ndarray:
    """Dense rank (1 = best) of each cell within its row ('row') or column ('col').
    Ties broken by the column/row index for determinism."""
    S = np.asarray(score, dtype=float)
    if along == "row":
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


def decode(inst: dict[str, Any], P: np.ndarray, cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Apply one decoder config to a plan; returns decoded edges (src, dst)."""
    fam = cfg["family"]
    p = cfg["params"]
    sids, tids = inst["sids"], inst["tids"]
    n, m = P.shape
    row = P.sum(axis=1)
    col = P.sum(axis=0)
    row_share = P / np.maximum(row[:, None], 1e-300)
    col_share = P / np.maximum(col[None, :], 1e-300)
    edges: list[tuple[str, str]] = []
    if fam == "D0":
        mask = P >= float(p["thr"])
    elif fam == "D1":
        mask = row_share >= float(p["tau"])
    elif fam == "D2":
        mask = np.minimum(row_share, col_share) >= float(p["tau"])
    elif fam == "D3":
        mask = np.sqrt(row_share * col_share) >= float(p["tau"])
    elif fam == "D4":
        rr = _rank_matrix(P, "row")
        cr = _rank_matrix(P, "col")
        mask = (rr <= int(p["k"])) & (cr <= int(p["k"]))
    elif fam == "D5":
        rr = _rank_matrix(P, "row")
        cr = _rank_matrix(P, "col")
        mask = (rr <= int(p["k"])) & (cr <= int(p["k"])) & \
               (np.minimum(row_share, col_share) >= float(p["tau"]))
    elif fam == "D6":
        # cumulative row mass with an abstention gate (never forces an edge per source)
        gate = float(p["gate"]); target = float(p["p"])
        mask = np.zeros_like(P, dtype=bool)
        for i in range(n):
            order = np.lexsort((np.arange(m), -P[i]))
            cum = 0.0
            for j in order:
                s = row_share[i, j]
                if s < gate:
                    break  # remaining edges are below the gate (sorted desc)
                if cum >= target:
                    break
                mask[i, j] = True
                cum += s
    else:
        raise ValueError(f"unknown decoder family {fam}")
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                edges.append((sids[i], tids[j]))
    return edges


# --------------------------------------------------------------------------- #
# Extended evaluation (on top of the previous study's unified evaluator)
# --------------------------------------------------------------------------- #

def evaluate_edges(inst: dict[str, Any], edges: list[tuple[str, str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-template metrics incl. split/merge-specific edge F1 and overall exact recovery."""
    from baseline_mechanism.common import evaluate_method

    df, summ = evaluate_method(inst, inst["truth"], edges)
    edge_set = {(str(s), str(d)) for s, d in edges}
    out_by_src: dict[str, set[str]] = {}
    in_by_dst: dict[str, set[str]] = {}
    for s, d in edge_set:
        out_by_src.setdefault(s, set()).add(d)
        in_by_dst.setdefault(d, set()).add(s)
    rows = []
    for t, tr in sorted(inst["truth"].items()):
        split_gt = tr["split"]
        merge_gt = tr["merge"]
        # split-specific edge F1: edges touching the split structure
        sp = {s for s, _ in split_gt}
        sd = {d for _, d in split_gt}
        split_pred = {(s, d) for s, d in edge_set if s in sp or d in sd}
        tp = len(split_gt & split_pred)
        fp = len(split_pred - split_gt)
        fn = len(split_gt - split_pred)
        sf1 = 2 * tp / max(2 * tp + fp + fn, 1)
        # merge-specific edge F1
        mp = {s for s, _ in merge_gt}
        md = {d for _, d in merge_gt}
        merge_pred = {(s, d) for s, d in edge_set if s in mp or d in md}
        tp = len(merge_gt & merge_pred)
        fp = len(merge_pred - merge_gt)
        fn = len(merge_gt - merge_pred)
        mf1 = 2 * tp / max(2 * tp + fp + fn, 1)
        row = df[df["template_id"] == t].iloc[0].to_dict()
        row["split_edge_f1"] = sf1
        row["merge_edge_f1"] = mf1
        row["overall_exact"] = int(row["split_exact"] and row["merge_exact"])
        rows.append(row)
    df2 = pd.DataFrame(rows)
    summ2 = dict(summ)
    for col in ("split_edge_f1", "merge_edge_f1", "overall_exact"):
        summ2[col] = float(df2[col].mean())
    summ2["overall_exact_total"] = int(df2["overall_exact"].sum())
    return df2, summ2


def hier_agg(frames: list[pd.DataFrame], metric: str) -> dict[str, Any]:
    """template -> seed -> bridge -> macro-average over bridges (equal bridge weight)."""
    if not frames:
        return {"macro": float("nan"), "n": 0}
    seed_means: dict[str, list[float]] = {}
    for f in frames:
        if f.empty:
            continue
        seed_means.setdefault(str(f["seed"].iloc[0]), []).append(float(f[metric].mean()))
    bridge_means = []
    for s, vals in sorted(seed_means.items()):
        bridge_means.append(float(np.mean(vals)))
    return {"macro": float(np.mean(bridge_means)), "per_bridge": bridge_means,
            "n_templates": sum(len(f) for f in frames), "n_seeds": len(seed_means)}


def bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    m = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y_s = y[order]
    tp = np.cumsum(y_s).astype(float)
    prec = tp / np.arange(1, len(y_s) + 1)
    rec = tp / n_pos
    return float(np.sum(prec * y_s) / n_pos)


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Trapezoidal area under the (interpolated) precision-recall curve."""
    ap_ = average_precision(y, score)  # not used for area; kept for cross-check
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[order]).astype(float)
    fp = np.cumsum(~y[order]).astype(float)
    rec = tp / n_pos
    prec = tp / np.maximum(tp + fp, 1.0)
    # standard 11-point-free trapezoid on (rec, prec) with monotone prec
    rec_full = np.concatenate([[0.0], rec, [1.0]])
    prec_full = np.concatenate([[prec[0] if len(prec) else 0.0], prec, [0.0]])
    prec_mono = np.maximum.accumulate(prec_full[::-1])[::-1]
    return float(np.trapz(prec_mono, rec_full))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
