"""R7 pipeline: one cell -> one UOT solve -> every decoder -> family-level metrics.

Aggregation contract (specification section 18)
----------------------------------------------
``instance -> family mean -> 24-family equal-weight mean -> per (bridge, seed) macro``
then ``10 seeds -> bridge mean`` then ``3 bridges -> bridge-balanced overall``.

The primary statistical unit is the ``(bridge, seed)`` paired cell (30 of them); the
several hundred template instances are NEVER treated as independent samples.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from r7.r7_common import (BRIDGES, K_MAX, METHODS, N_FAMILIES, SUPPORT_THRESHOLD,
                          family_of_template, family_index_of_template, log)
from r7.r7_methods import (_edges_from_mask, SIGNAL_CONDITIONAL, SIGNAL_RAW,
                           SIGNAL_SUPPORT, SIGNAL_UOT_KR, adaptive_budgets, apply_rule,
                           dual_softmax_edges, hungarian_1to1, log_kernel,
                           oracle_1to1_ceiling, rank_desc, rank_desc_masked,
                           signal_matrices, threshold_mm_edges)

METRIC_COLUMNS = ("edge_f1", "edge_precision", "edge_recall", "split_exact",
                  "merge_exact", "overall_exact", "n_pred_edges", "edge_tp", "edge_fp",
                  "edge_fn", "split_edge_f1", "merge_edge_f1", "coverage")


# --------------------------------------------------------------------------- #
# prepared cell context -- rank ONCE per (cell, signal), reuse for every rule
# --------------------------------------------------------------------------- #

class CellContext:
    """Everything derived from one frozen UOT solve.

    The transport plan is solved once per cell (outside this class).  Ranking matrices
    are computed once per (cell, signal) and reused by every rule candidate, so a rule
    sweep never re-ranks and never re-solves.
    """

    def __init__(self, cell: dict[str, Any], epsilon: float):
        self.cell = cell
        self.bridge = cell["bridge"]
        self.seed = cell["seed"]
        self.C = np.asarray(cell["C_primary"], dtype=float)
        self.P = np.asarray(cell["P"], dtype=float)
        self.sids = cell["sids"]
        self.tids = cell["tids"]
        self.epsilon = float(epsilon)
        self.logK = log_kernel(self.C, self.epsilon)
        self.realized_rows = self.P.sum(axis=1)
        self.realized_cols = self.P.sum(axis=0)
        # Frozen transport marginals.  These are direct attributes so that every consumer
        # (executor diagnostics, representation analysis, dry-run harness) reads the same
        # objects the solver was given.
        self.a = np.asarray(cell["a"], dtype=float)
        self.b = np.asarray(cell["b"], dtype=float)
        self.support = self.P > SUPPORT_THRESHOLD
        self._signals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        self._ranks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._oracle: dict[str, Any] | None = None
        self._metric_cache: dict[tuple[str, str], dict[str, Any]] = {}

    # -- signals --------------------------------------------------------- #
    def signal(self, name: str):
        if name not in self._signals:
            self._signals[name] = signal_matrices(name, self.logK, self.P)
        return self._signals[name]

    def ranks(self, name: str):
        """Rank matrices for one signal (support-aware when the signal is support-based)."""
        if name not in self._ranks:
            row_s, col_s, sup = self.signal(name)
            if sup is None:
                rr = rank_desc(row_s, 1)
                cr = rank_desc(col_s, 0)
            else:
                rr = rank_desc_masked(row_s, sup, 1)
                cr = rank_desc_masked(col_s, sup, 0)
            self._ranks[name] = (rr, cr)
        return self._ranks[name]

    # -- predictions ------------------------------------------------------ #
    def edges(self, method: str, rule: dict[str, Any]) -> list[tuple[str, str]]:
        signal = SIGNAL_OF_METHOD[method]
        row_s, col_s, sup = self.signal(signal)
        rr, cr = self.ranks(signal)
        fam = rule["family"]

        if fam in ("R-const", "R-quantile"):
            k = int(rule["k"])
            mask = (rr <= k) & (cr <= k)
        elif fam == "R-adaptive":
            kr, kc = adaptive_budgets(self.realized_rows, self.realized_cols,
                                      float(rule["alpha"]), int(rule.get("k_max", K_MAX)))
            mask = (rr <= kr[:, None]) & (cr <= kc[None, :])
        elif fam == "R-threshold":
            log_theta = math.log(float(rule["theta"]))
            R = row_s if sup is None else np.where(sup, row_s, -np.inf)
            Cm = col_s if sup is None else np.where(sup, col_s, -np.inf)
            rmax = np.max(R, axis=1, keepdims=True)
            cmax = np.max(Cm, axis=0, keepdims=True)
            with np.errstate(invalid="ignore"):
                mask = ((R - rmax) >= log_theta - 1e-15) & \
                       ((Cm - cmax) >= log_theta - 1e-15) & \
                       np.isfinite(rmax) & np.isfinite(cmax)
        else:
            raise ValueError(f"unknown rule family {fam}")

        if sup is not None:
            mask = mask & sup
        return _edges_from_mask(mask, self.sids, self.tids)

    # -- metrics ---------------------------------------------------------- #
    def metrics(self, method: str, edges: Sequence[tuple[str, str]]) -> dict[str, Any]:
        key = (method, sha256_edges(edges))
        if key not in self._metric_cache:
            m = cell_metrics(self.cell, edges)
            m["family_edge_counts"] = family_edge_counts(self.cell, edges)
            m.pop("_per_template", None)
            self._metric_cache[key] = m
        return self._metric_cache[key]

    def oracle(self) -> dict[str, Any]:
        if self._oracle is None:
            self._oracle = oracle_ceiling_by_family(self.cell)
        return self._oracle


SIGNAL_OF_METHOD = {
    "UOT_KR": SIGNAL_UOT_KR,
    "RAW_UOT_PLAN": SIGNAL_RAW,
    "CONDITIONAL_UOT": SIGNAL_CONDITIONAL,
    "SUPPORT_PLUS_K": SIGNAL_SUPPORT,
}


def sha256_edges(edges: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for s, d in sorted((str(a), str(b)) for a, b in edges):
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
        h.update(d.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def evaluate_edges(cell: dict[str, Any], edges: Sequence[tuple[str, str]]
                   ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one prediction with the PROJECT'S frozen evaluator."""
    from decoder_audit.da_common import evaluate_edges as _frozen_eval

    df, summ = _frozen_eval(cell, list(edges))
    df = df.copy()
    df["family"] = df["template_id"].map(family_of_template)
    df["family_index"] = df["template_id"].map(family_index_of_template)
    return df, summ


def family_macro(df: pd.DataFrame, metric: str = "edge_f1") -> float:
    """Family mean first, then equal weight over the 24 families."""
    if df.empty:
        return float("nan")
    return float(df.groupby("family")[metric].mean().mean())


def cell_metrics(cell: dict[str, Any], edges: Sequence[tuple[str, str]]
                 ) -> dict[str, Any]:
    df, summ = evaluate_edges(cell, edges)
    out: dict[str, Any] = {"macro_edge_f1": family_macro(df, "edge_f1")}
    for m in METRIC_COLUMNS:
        if m in df.columns:
            out[f"family_mean_{m}"] = family_macro(df, m)
    out["n_templates"] = int(len(df))
    out["n_families"] = int(df["family"].nunique()) if not df.empty else 0
    out["n_pred_edges_total"] = int(len(set(map(tuple, edges))))
    out["_per_template"] = df
    return out


def family_edge_counts(cell: dict[str, Any], edges: Sequence[tuple[str, str]]
                       ) -> dict[str, int]:
    """Predicted edge count per family (used for the Threshold-MM budget calibration)."""
    counts: dict[str, int] = {f"r7fam{i:02d}": 0 for i in range(N_FAMILIES)}
    for s, _d in edges:
        counts[family_of_template(s)] = counts.get(family_of_template(s), 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# all methods for one cell
# --------------------------------------------------------------------------- #

def run_all_methods(cell: dict[str, Any], config: dict[str, Any],
                    *, methods: Sequence[str] = METHODS,
                    include_oracle: bool = True,
                    rule_override: dict[str, Any] | None = None,
                    ctx: "CellContext | None" = None
                    ) -> dict[str, Any]:
    """Predict with every requested arm on ONE already-solved cell.

    The transport plan is never re-solved here: every arm consumes ``cell["P"]`` and
    ``cell["C_primary"]``, and one ``CellContext`` is reused across the whole rule sweep.
    """
    if ctx is None:
        ctx = CellContext(cell, float(config["epsilon"]))
    C = ctx.C
    P = ctx.P
    sids, tids = ctx.sids, ctx.tids
    logK = ctx.logK

    rule = dict(rule_override if rule_override is not None else config["selected_rule"])

    results: dict[str, Any] = {
        "bridge": cell["bridge"], "seed": cell["seed"],
        "cost_matrix_sha256": cell["hashes"]["cost_matrix_sha256"],
        "plan_sha256": cell["hashes"]["plan_sha256"],
        "solver": {k: v for k, v in cell["solver"].items() if k != "kkt"},
        "rule_applied": rule,
    }
    predictions: dict[str, list[tuple[str, str]]] = {}

    for method in ("UOT_KR", "RAW_UOT_PLAN", "CONDITIONAL_UOT", "SUPPORT_PLUS_K"):
        if method not in methods:
            continue
        edges = ctx.edges(method, rule)
        if method == "SUPPORT_PLUS_K":
            bad = _support_violations(cell, edges)
            if bad:
                raise AssertionError(
                    f"SUPPORT_PLUS_K hard-filter violated: {len(bad)} selected edges with "
                    f"P <= {SUPPORT_THRESHOLD} (first: {bad[0]})")
        predictions[method] = edges
    if "SUPPORT_PLUS_K" in methods:
        results["support_filter_violations"] = 0

    if "HUNGARIAN_1TO1" in methods:
        edges, info = hungarian_1to1(C, sids, tids)
        predictions["HUNGARIAN_1TO1"] = edges
        results["hungarian"] = info

    if "THRESHOLD_MM" in methods:
        cutoff = float(config["threshold_mm_cutoff"])
        predictions["THRESHOLD_MM"] = threshold_mm_edges(C, sids, tids, cutoff)
        results["threshold_mm"] = {"cutoff": cutoff}

    if "DUAL_SOFTMAX" in methods:
        edges, info = dual_softmax_edges(logK, sids, tids,
                                         tau=float(config["dual_softmax_tau"]))
        predictions["DUAL_SOFTMAX"] = edges
        results["dual_softmax"] = info

    for method, edges in predictions.items():
        m = dict(ctx.metrics(method, edges))
        m["pred_edges"] = [list(e) for e in edges]
        results[method] = m

    if include_oracle:
        results["ORACLE_1TO1_CEILING"] = ctx.oracle()

    return results


def _support_violations(cell: dict[str, Any], edges: Sequence[tuple[str, str]]
                        ) -> list[tuple[str, str]]:
    """Assert ``selected edge => P_ij > SUPPORT_THRESHOLD`` (specification section 23)."""
    P = np.asarray(cell["P"], dtype=float)
    idx_s = {s: i for i, s in enumerate(cell["sids"])}
    idx_t = {d: j for j, d in enumerate(cell["tids"])}
    bad = []
    for s, d in edges:
        if P[idx_s[s], idx_t[d]] <= SUPPORT_THRESHOLD:
            bad.append((s, d))
    return bad


def oracle_ceiling_by_family(cell: dict[str, Any]) -> dict[str, Any]:
    """Per-template maximum-matching ceiling, family-averaged (diagnostic only)."""
    rows = []
    for t, tr in sorted(cell["truth"].items()):
        o = oracle_1to1_ceiling(tr["positive"])
        o["template_id"] = t
        o["family"] = family_of_template(t)
        rows.append(o)
    df = pd.DataFrame(rows)
    fam = df.groupby("family")[["f1", "recall", "T", "M"]].mean()
    return {
        "macro_f1": float(fam["f1"].mean()),
        "macro_recall": float(fam["recall"].mean()),
        "mean_T": float(fam["T"].mean()),
        "mean_M": float(fam["M"].mean()),
        "per_template": df.to_dict(orient="records"),
        "label_informed": True,
        "deployable": False,
        "part_of_holm_family": False,
    }


# --------------------------------------------------------------------------- #
# Threshold-MM budget calibration (specification section 21)
# --------------------------------------------------------------------------- #

def threshold_mm_calibrate(cells: Sequence[dict[str, Any]], rule: dict[str, Any],
                           config: dict[str, Any], q_grid: Sequence[float]
                           ) -> dict[str, Any]:
    """Choose the global Threshold-MM cutoff by EQUAL-BUDGET matching to UOT_KR.

    The target is the selected UOT_KR arm's ``bridge-balanced mean predicted edge count
    per family`` on the selection block.  The candidate cutoff is the ``q``-quantile of
    the POOLED selection-block cost ECDF, for ``q`` on the frozen grid.  The criterion is
    ``argmin |budget(cutoff) - budget(UOT_KR)|`` and it NEVER uses ground-truth F1.
    Ties go to the HIGHER threshold (deterministic conservative tie-break).
    """
    pooled = np.concatenate([np.asarray(c["C_primary"], dtype=float).ravel()
                             for c in cells])
    pooled = np.sort(pooled[np.isfinite(pooled)])
    target = _uot_kr_budget(cells, rule, config)

    cands = []
    for q in q_grid:
        k = int(round(float(q) * pooled.size)) - 1
        k = max(0, min(k, pooled.size - 1))
        cutoff = float(pooled[k])
        budgets = _threshold_budgets(cells, cutoff)
        cands.append({"q": float(q), "cutoff": cutoff,
                      "bridge_balanced_mean_edges_per_family": budgets["balanced"],
                      "abs_diff_vs_uot_kr": abs(budgets["balanced"] - target),
                      "per_bridge": budgets["per_bridge"]})
    best = min(cands, key=lambda c: (c["abs_diff_vs_uot_kr"], -c["cutoff"]))
    return {
        "method": "THRESHOLD_MM",
        "criterion": ("minimise |bridge-balanced mean predicted edge count per family - "
                      "selected UOT_KR budget|"),
        "uses_ground_truth_f1": False,
        "tie_break": "higher threshold",
        "uot_kr_target_budget": target,
        "selected_q": best["q"],
        "selected_cutoff": best["cutoff"],
        "selected_abs_diff": best["abs_diff_vs_uot_kr"],
        "n_candidates": len(cands),
        "candidates": cands,
    }


def _uot_kr_budget(cells: Sequence[dict[str, Any]], rule: dict[str, Any],
                   config: dict[str, Any]) -> float:
    per_bridge: dict[str, list[float]] = {b: [] for b in BRIDGES}
    for cell in cells:
        res = run_all_methods(cell, config, methods=("UOT_KR",),
                              include_oracle=False, rule_override=rule)
        counts = res["UOT_KR"]["family_edge_counts"]
        per_bridge[cell["bridge"]].append(float(np.mean(list(counts.values()))))
    return float(np.mean([np.mean(v) for v in per_bridge.values() if v]))


def _threshold_budgets(cells: Sequence[dict[str, Any]], cutoff: float) -> dict[str, Any]:
    per_bridge: dict[str, list[float]] = {b: [] for b in BRIDGES}
    for cell in cells:
        edges = threshold_mm_edges(np.asarray(cell["C_primary"], dtype=float),
                                   cell["sids"], cell["tids"], cutoff)
        counts = family_edge_counts(cell, edges)
        per_bridge[cell["bridge"]].append(float(np.mean(list(counts.values()))))
    means = {b: (float(np.mean(v)) if v else float("nan")) for b, v in per_bridge.items()}
    return {"per_bridge": means,
            "balanced": float(np.nanmean([means[b] for b in BRIDGES]))}


# --------------------------------------------------------------------------- #
# Dual-Softmax calibration (specification section 22)
# --------------------------------------------------------------------------- #

def dual_softmax_calibrate(cells: Sequence[dict[str, Any]], rule: dict[str, Any],
                           config: dict[str, Any], tau_grid: Sequence[float]
                           ) -> dict[str, Any]:
    """Fix the Dual-Softmax confidence threshold on the SELECTION block only.

    Pre-registered selection procedure: maximise the selection-block bridge-balanced mean
    family macro edge F1 over the frozen tau grid; ties go to the HIGHER tau.  The
    acceptance rule (bidirectional mutual nearest neighbour) is never tuned.  Dual-Softmax
    is a comparator, so calibrating it on the selection block can only make it stronger.
    """
    cands = []
    for tau in tau_grid:
        per_bridge: dict[str, list[float]] = {b: [] for b in BRIDGES}
        for cell in cells:
            logK = log_kernel(np.asarray(cell["C_primary"], dtype=float),
                              float(config["epsilon"]))
            edges, _info = dual_softmax_edges(logK, cell["sids"], cell["tids"], tau=tau)
            m = cell_metrics(cell, edges)
            per_bridge[cell["bridge"]].append(m["macro_edge_f1"])
        means = {b: (float(np.mean(v)) if v else float("nan")) for b, v in per_bridge.items()}
        cands.append({"tau": float(tau),
                      "bridge_balanced_macro_edge_f1": float(np.nanmean(list(means.values()))),
                      "per_bridge": means})
    best = max(cands, key=lambda c: (round(c["bridge_balanced_macro_edge_f1"], 12), c["tau"]))
    return {
        "method": "DUAL_SOFTMAX",
        "criterion": ("maximise selection-block bridge-balanced mean family macro edge F1 "
                      "over the frozen tau grid"),
        "tie_break": "higher tau",
        "acceptance": "bidirectional mutual nearest neighbour (never tuned)",
        "selected_tau": best["tau"],
        "selected_score": best["bridge_balanced_macro_edge_f1"],
        "n_candidates": len(cands),
        "candidates": cands,
        "historical_status": ("the R5B prior-art module was retired as mis-specified "
                              "(softmax applied to K instead of log K) and was never "
                              "executed; R7 provides the complete scheme"),
    }
