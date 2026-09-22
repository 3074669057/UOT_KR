"""Shared machinery for the baseline capability + structural recovery mechanism study.

All methods consume the SAME cost matrix ``C_effective`` (frozen ``max(C + bridge_prior_bonus, 0)``
from ``build_cost_matrix_decomposed`` with default weights, graph merged into amount) and the
same decoded-edge evaluator. RC-UOT-Q main results are read from the frozen faithful pipeline
artifacts (transport matrix decoded at 1e-9) and are never re-solved for the main comparison.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

FROZEN = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
STUDY = REPO / "out" / "multi_bridge_expansion" / "structural_baseline_mechanism_study"

BRIDGES = ("Celer", "Multi", "Poly")
TEST_SEEDS = (42, 43, 44, 45, 46)
CALIB_SEEDS = (101, 102, 103)
N_TEMPLATES = 48

FROZEN_PARAMS = {
    "uot_reg": 0.05,
    "uot_reg_m": 0.5,
    "uot_lambda_risk": 0.25,
    "uot_decode_threshold": 1e-9,
    "uot_max_delay_sec": 21600.0,
    "uot_causal_violation_penalty": 5.0,
    "uot_backend": "pot",
    "uot_allow_unmatched": True,
    "uot_use_graph_embedding": False,
}

METHODS = ("Connector-style", "ABCTracer-style", "Threshold-MM", "Balanced-OT", "RC-UOT-Q")


def tpl_of(flow_id: str) -> str:
    """Template key of a synthetic flow id.

    Frozen-pipeline ids look like ``<base>__synth_<role>`` (template = prefix before
    ``__synth``); stress-generator ids look like ``<base>__<role>`` (template = prefix
    before the LAST ``__``). A bare id with neither separator is its own key.
    """
    s = str(flow_id)
    if "__synth" in s:
        return s.split("__synth")[0]
    if "__" in s:
        return s.rsplit("__", 1)[0]
    return s


def tpl_maps(labels: pd.DataFrame, sids: list[str], tids: list[str]) -> tuple[list[str], list[str]]:
    """Template keys per flow. Source template = base of the source id; target template is
    taken from the label rows (a target belongs to the template of the row's source), with
    the target's own base as fallback."""
    t_of_dst: dict[str, str] = {}
    for _, r in labels.iterrows():
        t_of_dst[str(r.get("dst_flow_id") or "")] = tpl_of(str(r.get("src_flow_id") or ""))
    tpl_s = [tpl_of(s) for s in sids]
    tpl_t = [t_of_dst.get(t, tpl_of(t)) for t in tids]
    return tpl_s, tpl_t


def frozen_run_root(bridge: str, seed: int) -> Path:
    return FROZEN / "per_seed" / bridge / f"seed_{seed}"


def load_frozen_instance(bridge: str, seed: int) -> dict[str, Any]:
    """Load the frozen per-seed grid: C_effective, P, flow ids, marginals, labels."""
    root = frozen_run_root(bridge, seed)
    c_npz = np.load(root / "uot" / "uot_cost_matrix.npz", allow_pickle=False)
    p_npz = np.load(root / "uot" / "uot_transport_matrix.npz", allow_pickle=True)
    labels = pd.read_csv(root / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
    hints = json.loads((root / "labels" / "synthetic_uot_eval_metrics.json").read_text(encoding="utf-8"))
    C = np.asarray(c_npz["C_effective"], dtype=float)
    comp = {k: np.asarray(c_npz[k], dtype=float) for k in
            ("amount_cost", "time_cost", "delay_sec", "route_cost", "risk_cost",
             "evidence_cost", "address_novelty_cost") if k in c_npz}
    P = np.asarray(p_npz["P"], dtype=float)
    sids = [str(x) for x in p_npz["source_flow_ids"]]
    tids = [str(x) for x in p_npz["target_flow_ids"]]
    tpl_s, tpl_t = tpl_maps(labels, sids, tids)
    return {
        "root": root, "bridge": bridge, "seed": seed,
        "C": C, "components": comp, "P": P,
        "sids": sids, "tids": tids, "tpl_s": tpl_s, "tpl_t": tpl_t,
        "a_orig": np.asarray(p_npz["source_mass_original"], dtype=float),
        "a_rw": np.asarray(p_npz["source_mass_risk_weighted"], dtype=float),
        "b_orig": np.asarray(p_npz["target_mass_original"], dtype=float),
        "b_ev": np.asarray(p_npz["target_mass_evidence_weighted"], dtype=float),
        "labels": labels, "hints": hints,
    }


def truth_structure(labels: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-template ground truth from synthetic labels."""
    lab = labels.copy()
    lab["_tpl"] = lab["src_flow_id"].astype(str).map(tpl_of)
    lab["_ls"] = lab.get("label_source", "").fillna("").astype(str)
    out: dict[str, dict[str, Any]] = {}
    for t, g in lab.groupby("_tpl", sort=False):
        split = set(zip(g.loc[g["pattern_type"] == "one_to_many", "src_flow_id"],
                        g.loc[g["pattern_type"] == "one_to_many", "dst_flow_id"]))
        merge = set(zip(g.loc[g["pattern_type"] == "many_to_one", "src_flow_id"],
                        g.loc[g["pattern_type"] == "many_to_one", "dst_flow_id"]))
        decoy = set(zip(g.loc[g["_ls"].str.contains("delay_noise"), "src_flow_id"],
                        g.loc[g["_ls"].str.contains("delay_noise"), "dst_flow_id"]))
        un_rows = g[g["_ls"].str.contains("unmatched")]
        unmatched_src = {str(s) for s in un_rows["src_flow_id"]}
        hidden_dst = {str(d) for d in un_rows["dst_flow_id"]}
        all_src = set(g["src_flow_id"].astype(str))
        all_dst = set(g["dst_flow_id"].astype(str))
        split_src = {s for s, _ in split}
        split_dst = {d for _, d in split}
        merge_src = {s for s, _ in merge}
        merge_dst = {d for _, d in merge}
        positive = split | merge | decoy
        out[t] = {
            "split": split, "merge": merge, "decoy": decoy, "positive": positive,
            "unmatched_src": unmatched_src, "hidden_dst": hidden_dst,
            "split_src": split_src, "split_dst": split_dst,
            "merge_src": merge_src, "merge_dst": merge_dst,
            "all_src": all_src, "all_dst": all_dst,
        }
    return out


# --------------------------------------------------------------------------- #
# Method decoders (all take the same cost matrix / plan and return edge lists)
# --------------------------------------------------------------------------- #

def decode_connector(inst: dict[str, Any]) -> list[tuple[str, str]]:
    """Connector-style: per-source argmin of amount cost within the source's template
    (project's existing structural-baseline rule: ``run_one_to_one(use_time=False)``)."""
    C = inst["C"]
    amt = inst["components"]["amount_cost"]
    sids, tids = inst["sids"], inst["tids"]
    t_of_s = inst["tpl_s"]
    t_of_t = inst["tpl_t"]
    edges: list[tuple[str, str]] = []
    for i, s in enumerate(sids):
        cand = [j for j in range(C.shape[1]) if t_of_t[j] == t_of_s[i]]
        if not cand:
            continue
        j = int(min(cand, key=lambda jj: (float(amt[i, jj]), jj)))
        edges.append((s, tids[j]))
    return edges


def decode_abctracer(inst: dict[str, Any]) -> list[tuple[str, str]]:
    """ABCTracer-style: per-source argmin of 0.75*amount + 0.25*time within the
    source's template (project's existing structural-baseline rule:
    ``run_one_to_one(use_time=True)`` with ``tcost = min(|delay|/3600, 1)``)."""
    C = inst["C"]
    amt = inst["components"]["amount_cost"]
    delay = inst["components"]["delay_sec"]
    sids, tids = inst["sids"], inst["tids"]
    t_of_s = inst["tpl_s"]
    t_of_t = inst["tpl_t"]
    edges: list[tuple[str, str]] = []
    for i, s in enumerate(sids):
        cand = [j for j in range(C.shape[1]) if t_of_t[j] == t_of_s[i]]
        if not cand:
            continue
        best_j, best_c = None, float("inf")
        for j in cand:
            tcost = min(abs(float(delay[i, j])) / 3600.0, 1.0)
            c = 0.75 * float(amt[i, j]) + 0.25 * tcost
            if c < best_c or (c == best_c and (best_j is None or j < best_j)):
                best_c, best_j = c, j
        edges.append((s, tids[best_j]))
    return edges


def decode_threshold_mm(inst: dict[str, Any], cutoff: float) -> list[tuple[str, str]]:
    """Threshold-MM: edge iff normalized_cost(i,j) <= tau (implemented as C_ij <= cutoff,
    where cutoff = calibration-quantile of the cost ECDF for the selected tau)."""
    C = inst["C"]
    sids, tids = inst["sids"], inst["tids"]
    edges: list[tuple[str, str]] = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if float(C[i, j]) <= cutoff:
                edges.append((sids[i], tids[j]))
    return edges


def decode_plan(P: np.ndarray, sids: list[str], tids: list[str],
                threshold: float = 1e-9) -> list[tuple[str, str]]:
    """Semantic twin of ``decode_correspondence``: edge iff transport mass >= threshold."""
    P = np.asarray(P, dtype=float)
    edges: list[tuple[str, str]] = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if float(P[i, j]) >= threshold:
                edges.append((sids[i], tids[j]))
    return edges


def solve_balanced_ot(C: np.ndarray, a: np.ndarray, b: np.ndarray,
                      reg: float = 0.05) -> dict[str, Any]:
    """Strictly balanced entropic OT: min <P,C> + reg*H(P) s.t. P 1 = a, P^T 1 = b.

    a and b are each normalized to unit total mass (the task's feasibility rule:
    normalize both marginal sets to the same total mass). The cost matrix is rescaled
    by its mean before the solve (entropic OT is exactly scale-invariant in (C, reg);
    this only improves numerical conditioning, it does not change the solution).
    No dustbin / dummy node / partial-OT / unbalanced penalty is used.
    """
    import ot as _ot

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
    P, log = _ot.sinkhorn(a, b, C / kappa, reg=reg / kappa, numItermax=20000,
                          stopThr=1e-11, log=True, method="sinkhorn")
    P = np.asarray(P, dtype=float)
    row_res = float(np.abs(P.sum(axis=1) - a).max())
    col_res = float(np.abs(P.sum(axis=0) - b).max())
    # Residual-based convergence: POT's stopThr (1e-11) can stall on a few ill-conditioned
    # instances (Multi seeds 42/43 stall at err ~5e-8 with marginal residual ~2e-8 of unit
    # mass, far below the decode threshold 1e-9). Those are treated as converged.
    return {
        "P": P,
        "kappa": kappa,
        "converged": bool(max(row_res, col_res) < 1e-6),
        "final_err": float(np.asarray(log.get("err", [1.0])).ravel()[-1]),
        "row_residual": row_res,
        "col_residual": col_res,
    }


def solve_rc_uot(inst: dict[str, Any], C: np.ndarray, flows_s: list[dict[str, Any]],
                 flows_t: list[dict[str, Any]]) -> np.ndarray:
    """RC-UOT-Q on new data with the FROZEN parameters (used for stress ladders only;
    the main comparison reads the frozen transport matrix directly)."""
    from cross.domain.uot.uot_solver import solve_uot

    p, _ = solve_uot(
        flows_s, flows_t,
        reg=FROZEN_PARAMS["uot_reg"], reg_m=FROZEN_PARAMS["uot_reg_m"],
        weights=None, use_graph=False, backend="pot", solver_meta={},
        cost_matrix=C,
        max_delay_sec=FROZEN_PARAMS["uot_max_delay_sec"],
        causal_violation_penalty=FROZEN_PARAMS["uot_causal_violation_penalty"],
        causal_infeasible_delay_sec=None,
        lambda_risk=FROZEN_PARAMS["uot_lambda_risk"],
        use_risk_weighted_source_mass=True,
        use_evidence_weighted_target_mass=True,
    )
    return np.asarray(p, dtype=float)


# --------------------------------------------------------------------------- #
# Unified evaluator
# --------------------------------------------------------------------------- #

def evaluate_template(truth: dict[str, Any], pred_edges: set[tuple[str, str]],
                      sources_with_mass: set[str]) -> dict[str, Any]:
    split_gt = truth["split"]
    merge_gt = truth["merge"]
    positive = truth["positive"]
    split_src = truth["split_src"]
    merge_dst = truth["merge_dst"]
    split_dst = truth["split_dst"]
    merge_src = truth["merge_src"]

    out_by_src: dict[str, set[str]] = {}
    in_by_dst: dict[str, set[str]] = {}
    for s, d in pred_edges:
        out_by_src.setdefault(s, set()).add(d)
        in_by_dst.setdefault(d, set()).add(s)

    # exact split: split source emits exactly the two true dsts, no other source claims them
    split_exact = 0
    if split_src:
        s = next(iter(split_src))
        outs = out_by_src.get(s, set())
        claimers = {x for x, ds in out_by_src.items() if x != s and ds & split_dst}
        if outs == split_dst and not claimers:
            split_exact = 1
    # exact merge: exactly the two true sources claim merge dst; merge sources emit nothing else
    merge_exact = 0
    if merge_dst:
        d = next(iter(merge_dst))
        inn = in_by_dst.get(d, set())
        extras = [x for x in merge_src for dd in out_by_src.get(x, set()) if dd != d]
        if inn == merge_src and not extras:
            merge_exact = 1

    deg_split = int(len(out_by_src.get(next(iter(split_src)), set())) == 2) if split_src else 0
    deg_merge = int(len(in_by_dst.get(next(iter(merge_dst)), set())) == 2) if merge_dst else 0

    tp = len(positive & pred_edges)
    fp = len(pred_edges - positive)
    fn = len(positive - pred_edges)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    pos_srcs = {s for s, _ in positive}
    covered = len(pos_srcs & set(out_by_src)) / max(len(pos_srcs), 1)
    n_abstained = len([s for s in pos_srcs if s not in out_by_src])
    n_abstained_all = len([s for s in sources_with_mass if s not in out_by_src]) if sources_with_mass else 0

    return {
        "split_exact": split_exact, "merge_exact": merge_exact,
        "deg_acc_split": deg_split, "deg_acc_merge": deg_merge,
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
        "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
        "n_pred_edges": len(pred_edges),
        "coverage": covered,
        "n_abstained_pos_src": n_abstained,
        "n_abstained_all_src": n_abstained_all,
    }


def evaluate_method(inst: dict[str, Any], truth: dict[str, dict[str, Any]],
                    edges: list[tuple[str, str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one method's decoded edges on all templates; returns (per-template df, summary)."""
    edge_set = {(str(s), str(d)) for s, d in edges}
    by_src = {}
    for s, d in edge_set:
        by_src.setdefault(s, set()).add(d)
    rows = []
    for t, tr in sorted(truth.items()):
        local = {(s, d) for s, d in edge_set if tpl_of(s) == t}
        m = evaluate_template(tr, local, set(tr["all_src"]))
        m.update({"template_id": t})
        rows.append(m)
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {"n_templates": len(df), "n_pred_edges_total": len(edge_set)}
    for col in ("split_exact", "merge_exact", "deg_acc_split", "deg_acc_merge", "edge_precision",
                "edge_recall", "edge_f1", "coverage", "n_pred_edges"):
        summary[col] = float(df[col].mean()) if not df.empty else 0.0
    summary["edge_fp_total"] = int(df["edge_fp"].sum()) if not df.empty else 0
    summary["edge_tp_total"] = int(df["edge_tp"].sum()) if not df.empty else 0
    summary["edge_fn_total"] = int(df["edge_fn"].sum()) if not df.empty else 0
    return df, summary


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    means = np.array([rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(n_boot)])
    alpha = (1.0 - level) / 2.0
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))
