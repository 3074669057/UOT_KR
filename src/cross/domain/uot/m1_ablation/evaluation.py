"""Evaluation metrics with bootstrap CI and stratified analysis for M1 ablation.

Bootstrap CI is computed by re-sampling at the PAIR level and re-computing
the full metric (precision, recall, F1) on each bootstrap sample, matching
the original metric computation logic exactly.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cross.shared.normalize import norm_addr

StructureLabel = Literal["1-1", "m-1", "1-m", "m-n", "unmatched", "other"]


@dataclass
class BootstrapCI:
    """Bootstrap 95% percentile CI.

    The `mean` field is the POINT ESTIMATE from the original data.
    lower_95 and upper_95 are from the bootstrap distribution.
    """
    mean: float
    lower_95: float
    upper_95: float
    n_bootstrap: int = 1000
    seed: int = 42
    n_units: int = 0

    def to_dict(self) -> dict[str, float]:
        return {"mean": round(self.mean, 6), "ci_low": round(self.lower_95, 6), "ci_high": round(self.upper_95, 6)}


@dataclass
class MetricsResult:
    solver: str
    precision: float
    recall: float
    f1: float
    coverage: float
    abstention_rate: float
    tp: int
    fp: int
    fn: int
    n_predicted: int
    n_abstained: int
    n_ground_truth: int
    score_threshold: float | None = None
    precision_ci: BootstrapCI | None = None
    recall_ci: BootstrapCI | None = None
    f1_ci: BootstrapCI | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StratifiedResult:
    solver: str
    structure: StructureLabel
    support: int
    precision: float
    recall: float
    f1: float
    coverage: float
    abstention_rate: float
    tp: int
    fp: int
    fn: int
    precision_ci: BootstrapCI | None = None
    recall_ci: BootstrapCI | None = None
    f1_ci: BootstrapCI | None = None


def assign_structure_label_tx(
    ground_truth_edges: dict[str, str],
) -> dict[str, StructureLabel]:
    """Assign structure label per TRANSACTION PAIR (tx-level).

    1-1: src_tx maps to one dst_tx, and that dst_tx is mapped by exactly one src_tx
    m-1: src_tx maps to one dst_tx, but that dst_tx is mapped by multiple src_tx
    1-m: src_tx maps to multiple dst_tx (rare in Celer bridge)
    m-n: many-to-many
    unmatched: src_tx has no dst_tx mapping
    """
    # Build adjacency
    dst_to_srcs: dict[str, set[str]] = defaultdict(set)
    src_to_dsts: dict[str, set[str]] = defaultdict(set)
    for s, d in ground_truth_edges.items():
        src_to_dsts.setdefault(s, set()).add(d)
        dst_to_srcs.setdefault(d, set()).add(s)

    labels: dict[str, StructureLabel] = {}
    for s, d in ground_truth_edges.items():
        out_deg = len(src_to_dsts.get(s, set()))
        in_deg = len(dst_to_srcs.get(d, set()))
        if out_deg == 1 and in_deg == 1:
            labels[s] = "1-1"
        elif out_deg == 1 and in_deg > 1:
            labels[s] = "m-1"
        elif out_deg > 1 and in_deg == 1:
            labels[s] = "1-m"
        elif out_deg > 1 and in_deg > 1:
            labels[s] = "m-n"
        elif out_deg == 0 or in_deg == 0:
            labels[s] = "unmatched"
        else:
            labels[s] = "other"
    return labels


def assign_structure_label_flow(
    ground_truth_edges: dict[str, str],
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
) -> dict[str, StructureLabel]:
    """Assign structure label per source flow (flow-level, for paper comparison).

    Looks at which target flows the ground-truth transactions in each source flow map to.
    """
    # Map tx to flow
    tx_to_src_flow: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            h = norm_addr(str(txh))
            if h:
                tx_to_src_flow[h] = i

    tx_to_dst_flow: dict[str, set[int]] = {}
    for j, tf in enumerate(target_flows):
        for txh in tf.get("tx_hashes") or []:
            h = norm_addr(str(txh))
            if h:
                tx_to_dst_flow.setdefault(h, set()).add(j)

    # Build flow-level adjacency
    src_flow_to_dst_flows: dict[int, set[int]] = defaultdict(set)
    dst_flow_to_src_flows: dict[int, set[int]] = defaultdict(set)
    for s, d in ground_truth_edges.items():
        si = tx_to_src_flow.get(s, -1)
        djs = tx_to_dst_flow.get(d, set())
        if si >= 0 and djs:
            for dj in djs:
                src_flow_to_dst_flows.setdefault(si, set()).add(dj)
                dst_flow_to_src_flows.setdefault(dj, set()).add(si)

    labels: dict[str, StructureLabel] = {}
    for s, d in ground_truth_edges.items():
        si = tx_to_src_flow.get(s, -1)
        djs = tx_to_dst_flow.get(d, set())
        dj = next(iter(djs)) if djs else -1
        if si < 0 or dj < 0:
            labels[s] = "unmatched"
            continue
        fo = len(src_flow_to_dst_flows.get(si, set()))
        fi = len(dst_flow_to_src_flows.get(dj, set()))
        if fo == 1 and fi == 1:
            labels[s] = "1-1"
        elif fo == 1 and fi > 1:
            labels[s] = "m-1"
        elif fo > 1 and fi == 1:
            labels[s] = "1-m"
        elif fo > 1 and fi > 1:
            labels[s] = "m-n"
        else:
            labels[s] = "other"
    return labels


# Alias: paper uses tx-level topology (72.32% 1-1, 27.51% m-1, 0.17% 1-m)
def assign_structure_label(ground_truth_edges: dict[str, str], source_flows=None, target_flows=None) -> dict[str, StructureLabel]:
    return assign_structure_label_tx(ground_truth_edges)


def _compute_bootstrap_metrics(
    pair_results: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
    n_source_flows: int,
) -> tuple[BootstrapCI, BootstrapCI, BootstrapCI]:
    """Compute bootstrap CIs for precision, recall, F1 at the pair level.

    Each bootstrap sample re-samples pairs WITH REPLACEMENT and re-computes
    precision/recall/F1 using the EXACT same logic as compute_metrics.
    """
    n = len(pair_results)
    if n < 5:
        return (
            BootstrapCI(mean=0.0, lower_95=0.0, upper_95=0.0, n_bootstrap=n_bootstrap, seed=seed, n_units=n),
            BootstrapCI(mean=0.0, lower_95=0.0, upper_95=0.0, n_bootstrap=n_bootstrap, seed=seed, n_units=n),
            BootstrapCI(mean=0.0, lower_95=0.0, upper_95=0.0, n_bootstrap=n_bootstrap, seed=seed, n_units=n),
        )

    rng = np.random.RandomState(seed)
    prec_vals = np.zeros(n_bootstrap, dtype=float)
    rec_vals = np.zeros(n_bootstrap, dtype=float)
    f1_vals = np.zeros(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        sampled = [pair_results[i] for i in idx]
        # Re-compute metrics exactly like compute_metrics
        tp = sum(1 for p in sampled if p["pred_correct"])
        fp = sum(1 for p in sampled if not p["pred_correct"] and p["pred_dst"] and p["time_ok"])
        fn_ = sum(1 for p in sampled if not p["pred_correct"])
        n_pred = tp + fp
        prec_b = tp / max(n_pred, 1)
        rec_b = tp / max(n, 1)
        f1_b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-12)
        prec_vals[b] = prec_b
        rec_vals[b] = rec_b
        f1_vals[b] = f1_b

    # Point estimates from original data
    tp_orig = sum(1 for p in pair_results if p["pred_correct"])
    fp_orig = sum(1 for p in pair_results if not p["pred_correct"] and p["pred_dst"] and p["time_ok"])
    n_pred_orig = tp_orig + fp_orig
    prec_pt = tp_orig / max(n_pred_orig, 1)
    rec_pt = tp_orig / max(n, 1)
    f1_pt = 2 * prec_pt * rec_pt / max(prec_pt + rec_pt, 1e-12)

    return (
        BootstrapCI(mean=round(prec_pt, 6), lower_95=round(float(np.percentile(prec_vals, 2.5)), 6),
                     upper_95=round(float(np.percentile(prec_vals, 97.5)), 6),
                     n_bootstrap=n_bootstrap, seed=seed, n_units=n),
        BootstrapCI(mean=round(rec_pt, 6), lower_95=round(float(np.percentile(rec_vals, 2.5)), 6),
                     upper_95=round(float(np.percentile(rec_vals, 97.5)), 6),
                     n_bootstrap=n_bootstrap, seed=seed, n_units=n),
        BootstrapCI(mean=round(f1_pt, 6), lower_95=round(float(np.percentile(f1_vals, 2.5)), 6),
                     upper_95=round(float(np.percentile(f1_vals, 97.5)), 6),
                     n_bootstrap=n_bootstrap, seed=seed, n_units=n),
    )


def compute_metrics(
    *,
    mapping: dict[str, str | None],
    truth: dict[str, str],
    n_source_flows: int,
    n_abstained: int,
    n_predicted: int,
    solver_name: str = "",
    score_threshold: float | None = None,
    bootstrap_config: dict[str, Any] | None = None,
    eth_ts: dict[str, float] | None = None,
    bnb_ts: dict[str, float] | None = None,
) -> MetricsResult:
    """Compute overall precision/recall/F1/coverage/abstention.

    Precision = TP / (TP + FP) where:
      - TP: pred_dst matches gt_dst AND time_ok (ts_d >= ts_s)
      - FP: pred_dst != gt_dst OR not time_ok (but prediction exists)
      - FN: no prediction OR prediction abstained (counts as FN against ALL ground truth)

    This matches the original run_m1_analysis.py metric: pred_correct = is_gt AND time_ok.
    """
    bc = bootstrap_config or {}
    n_boot = int(bc.get("n_bootstrap", 1000))
    boot_seed = int(bc.get("seed", 42))

    n_gt = len(truth)
    pair_results: list[dict[str, Any]] = []

    for src_tx, gt_dst in truth.items():
        pred_dst = mapping.get(src_tx)
        # Determine time_ok
        time_ok = True
        if eth_ts and bnb_ts and pred_dst:
            ts_s = eth_ts.get(src_tx)
            ts_d = bnb_ts.get(norm_addr(str(pred_dst)))
            time_ok = ts_s is not None and ts_d is not None and float(ts_d - ts_s) >= 0

        if pred_dst is None:
            pair_results.append({
                "src_tx": src_tx, "gt_dst": gt_dst, "pred_dst": None,
                "pred_correct": False, "time_ok": False, "is_gt": False,
            })
        else:
            is_gt = norm_addr(str(pred_dst)) == norm_addr(str(gt_dst))
            pair_results.append({
                "src_tx": src_tx, "gt_dst": gt_dst, "pred_dst": pred_dst,
                "pred_correct": is_gt and time_ok, "time_ok": time_ok, "is_gt": is_gt,
            })

    tp = sum(1 for p in pair_results if p["pred_correct"])
    fp = sum(1 for p in pair_results if not p["pred_correct"] and p["pred_dst"] and p["time_ok"])
    fn = n_gt - tp
    n_pred = tp + fp

    precision = tp / max(n_pred, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    coverage = float(n_predicted / max(n_gt, 1))
    abstention_rate = float(n_abstained / max(n_gt, 1))

    prec_ci, rec_ci, f1_ci = _compute_bootstrap_metrics(pair_results, n_boot, boot_seed, n_source_flows)

    return MetricsResult(
        solver=solver_name,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        coverage=round(coverage, 6),
        abstention_rate=round(abstention_rate, 6),
        tp=tp, fp=fp, fn=fn,
        n_predicted=n_predicted,
        n_abstained=n_abstained,
        n_ground_truth=n_gt,
        score_threshold=score_threshold,
        precision_ci=prec_ci,
        recall_ci=rec_ci,
        f1_ci=f1_ci,
    )


def compute_stratified_metrics(
    *,
    mapping: dict[str, str | None],
    truth: dict[str, str],
    structure_labels: dict[str, StructureLabel],
    n_source_flows: int,
    n_abstained: int,
    n_predicted: int,
    solver_name: str = "",
    score_threshold: float | None = None,
    bootstrap_config: dict[str, Any] | None = None,
    eth_ts: dict[str, float] | None = None,
    bnb_ts: dict[str, float] | None = None,
) -> dict[StructureLabel, StratifiedResult]:
    """Compute per-structure-bucket metrics."""
    bc = bootstrap_config or {}
    n_boot = int(bc.get("n_bootstrap", 1000))
    boot_seed = int(bc.get("seed", 42))

    buckets: dict[StructureLabel, list[str]] = defaultdict(list)
    for src_tx in truth:
        label = structure_labels.get(src_tx, "other")
        buckets[label].append(src_tx)

    results: dict[StructureLabel, StratifiedResult] = {}
    for label, src_txs in sorted(buckets.items()):
        n = len(src_txs)
        pair_results: list[dict[str, Any]] = []
        for src_tx in src_txs:
            pred_dst = mapping.get(src_tx)
            gt_dst = truth[src_tx]
            time_ok = True
            if eth_ts and bnb_ts and pred_dst:
                ts_s = eth_ts.get(src_tx)
                ts_d = bnb_ts.get(norm_addr(str(pred_dst)))
                time_ok = ts_s is not None and ts_d is not None and float(ts_d - ts_s) >= 0

            if pred_dst is None:
                pair_results.append({"pred_correct": False, "pred_dst": None, "time_ok": False})
            else:
                is_gt = norm_addr(str(pred_dst)) == norm_addr(str(gt_dst))
                pair_results.append({"pred_correct": is_gt and time_ok, "pred_dst": pred_dst, "time_ok": time_ok})

        tp = sum(1 for p in pair_results if p["pred_correct"])
        fp = sum(1 for p in pair_results if not p["pred_correct"] and p["pred_dst"] and p["time_ok"])
        fn_ = n - tp
        n_pred = tp + fp

        precision = tp / max(n_pred, 1)
        recall = tp / max(n, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        coverage = float((n - sum(1 for p in pair_results if p["pred_dst"] is None)) / max(n, 1))
        abstention = float(sum(1 for p in pair_results if p["pred_dst"] is None) / max(n, 1))

        prec_ci = rec_ci = f1_ci = None
        if n >= 5:
            pci, rci, fci = _compute_bootstrap_metrics(pair_results, n_boot, boot_seed, n_source_flows)
            prec_ci, rec_ci, f1_ci = pci, rci, fci

        results[label] = StratifiedResult(
            solver=solver_name, structure=label, support=n,
            precision=round(precision, 6), recall=round(recall, 6), f1=round(f1, 6),
            coverage=round(coverage, 6), abstention_rate=round(abstention, 6),
            tp=tp, fp=fp, fn=fn_,
            precision_ci=prec_ci, recall_ci=rec_ci, f1_ci=f1_ci,
        )

    return results
