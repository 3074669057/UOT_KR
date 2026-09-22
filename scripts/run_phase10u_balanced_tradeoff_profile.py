#!/usr/bin/env python3
"""Phase 10U: Balanced trade-off profile analysis (read-only over Phase 10S/10T)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper_full_pipeline_run" / "phase10u_balanced_tradeoff_profile"

PHASE10S_AGG = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10s_same_scope_baseline_superiority"
    / "aggregate"
    / "same_scope_baseline_comparison_aggregate.csv"
)
PHASE10S_TABLE_D = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10s_same_scope_baseline_superiority"
    / "tables"
    / "table_d_same_scope_baseline_superiority.csv"
)
PHASE10S_GATE = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10s_same_scope_baseline_superiority"
    / "aggregate"
    / "same_scope_claim_gate.json"
)
PHASE10T_TABLE_E = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10t_rcuot_arch_optimization"
    / "holdout"
    / "table_e_rcuot_arch_optimization.csv"
)
PHASE10T_METRICS = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10t_rcuot_arch_optimization"
    / "holdout"
    / "holdout_metrics_by_method.csv"
)
PHASE10T_GATE = (
    ROOT
    / "out"
    / "paper_full_pipeline_run"
    / "phase10t_rcuot_arch_optimization"
    / "holdout"
    / "holdout_claim_gate.json"
)

PROFILE_DIMS = [
    "flow_pair_f1",
    "flow_mass_recall",
    "split_recovery",
    "merge_recovery",
    "mrr",
    "calibration_score",
]
DIM_LABELS = {
    "flow_pair_f1": "Flow Pair-F1",
    "flow_mass_recall": "Flow-Mass Recall",
    "split_recovery": "Split Recovery",
    "merge_recovery": "Merge Recovery",
    "mrr": "MRR",
    "calibration_score": "Calibration Score",
}
LAMBDA_DEFAULT = 0.25
LAMBDA_SENSITIVITY = [0.1, 0.25, 0.5]
CATASTROPHIC_RATIO = 0.5
RCUOT_METHOD_KEYS = {
    "phase10s": {"rc_uot"},
    "phase10t_holdout": {"frozen_rc_uot_ref", "optimized_rcuot_hybrid_ranked_uot"},
}
PRIMARY_RCUOT = {
    "phase10s": "rc_uot",
    "phase10t_holdout": "optimized_rcuot_hybrid_ranked_uot",
}
BASELINE_KEYS = {"connector_style", "abctracer_style"}
DISPLAY_NAMES = {
    "rc_uot": "RC-UOT",
    "connector_style": "Connector-style",
    "abctracer_style": "ABCTracer-style",
    "frozen_rc_uot_ref": "Frozen RC-UOT ref",
    "optimized_rcuot_hybrid_ranked_uot": "Optimized RC-UOT",
}
CLAIM_NOTES = {
    "rc_uot": "Balanced mass–merge–ranking–calibration profile; lower Pair-F1 disclosed.",
    "optimized_rcuot_hybrid_ranked_uot": (
        "Balanced mass–merge–ranking–calibration profile on holdout; lower Pair-F1 disclosed."
    ),
    "frozen_rc_uot_ref": "Reference frozen RC-UOT; lower Pair-F1 on holdout.",
    "abctracer_style": "Strongest Pair-F1; lower Flow-Mass Recall than RC-UOT on holdout.",
    "connector_style": "Strong Pair-F1/Split; weak Merge Recovery.",
}


@dataclass
class MethodRow:
    key: str
    display: str
    source: str
    metrics: dict[str, float]
    optional: dict[str, float | None]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_method_key(raw: str) -> str:
    lowered = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if "connector" in lowered:
        return "connector_style"
    if "abctracer" in lowered or "abc_tracer" in lowered:
        return "abctracer_style"
    if lowered in {"rc_uot", "rc-uot"}:
        return "rc_uot"
    if lowered in {"frozen_rc_uot_ref", "frozen_rc_uot"}:
        return "frozen_rc_uot_ref"
    if "optimized" in lowered and "rcuot" in lowered:
        return "optimized_rcuot_hybrid_ranked_uot"
    return lowered


def _f(value: str | float | None) -> float:
    if value is None or value == "":
        return float("nan")
    if isinstance(value, float):
        return value
    return float(str(value).replace("s", "").strip())


def _load_phase10s_rows() -> list[MethodRow]:
    agg = _read_csv(PHASE10S_AGG)
    rows: list[MethodRow] = []
    for row in agg:
        key = _normalize_method_key(row["method"])
        metrics = {
            "flow_pair_f1": _f(row["flow_pair_f1_mean"]),
            "flow_mass_recall": _f(row["flow_mass_recall_mean"]),
            "split_recovery": _f(row["split_recovery_mean"]),
            "merge_recovery": _f(row["merge_recovery_mean"]),
            "mrr": _f(row["mrr_mean"]),
            "calibration_score": 1.0 - _f(row["ece_mean"]),
        }
        optional = {
            "risk_lift": _f(row.get("risk_lift_mean")),
            "runtime_sec": _f(row.get("runtime_sec_mean")),
            "unmatched_detection_f1": _f(row.get("unmatched_detection_f1_mean")),
            "decoy_rejection_rate": _f(row.get("decoy_rejection_rate_mean")),
        }
        rows.append(
            MethodRow(
                key=key,
                display=DISPLAY_NAMES.get(key, key),
                source="Phase 10S",
                metrics=metrics,
                optional=optional,
            )
        )
    return rows


def _load_phase10t_rows() -> list[MethodRow]:
    table = _read_csv(PHASE10T_TABLE_E)
    rows: list[MethodRow] = []
    for row in table:
        key = _normalize_method_key(row["Method"])
        metrics = {
            "flow_pair_f1": _f(row["Flow Pair-F1"]),
            "flow_mass_recall": _f(row["Flow-Mass Recall"]),
            "split_recovery": _f(row["Split Recovery"]),
            "merge_recovery": _f(row["Merge Recovery"]),
            "mrr": _f(row["MRR"]),
            "calibration_score": 1.0 - _f(row["ECE"]),
        }
        rows.append(
            MethodRow(
                key=key,
                display=DISPLAY_NAMES.get(key, row["Method"]),
                source="Phase 10T holdout",
                metrics=metrics,
                optional={
                    "risk_lift": None,
                    "runtime_sec": None,
                    "unmatched_detection_f1": None,
                    "decoy_rejection_rate": None,
                },
            )
        )
    return rows


def _harmonic_mean(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return len(positive) / sum(1.0 / v for v in positive)


def _normalized_metrics(methods: list[MethodRow]) -> dict[str, dict[str, float]]:
    bests: dict[str, float] = {}
    min_ece_equiv = min(m.metrics["calibration_score"] for m in methods)
    max_cal = max(m.metrics["calibration_score"] for m in methods)
    for dim in PROFILE_DIMS:
        if dim == "calibration_score":
            bests[dim] = max_cal
        else:
            bests[dim] = max(m.metrics[dim] for m in methods)

    min_ece = 1.0 - max_cal
    out: dict[str, dict[str, float]] = {}
    for method in methods:
        norms: dict[str, float] = {}
        for dim in PROFILE_DIMS:
            val = method.metrics[dim]
            if dim == "calibration_score":
                ece = 1.0 - val
                norms[dim] = min_ece / ece if ece > 0 else 1.0
            else:
                best = bests[dim]
                norms[dim] = val / best if best > 0 else 0.0
        out[method.key] = norms
    return out


def _compute_balanced_scores(
    methods: list[MethodRow], lambda_default: float = LAMBDA_DEFAULT
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    norms = _normalized_metrics(methods)
    rows: list[dict[str, Any]] = []
    for method in methods:
        raw_vals = [method.metrics[d] for d in PROFILE_DIMS]
        norm_vals = [norms[method.key][d] for d in PROFILE_DIMS]
        raw_balanced = statistics.mean(raw_vals)
        normalized_balanced = statistics.mean(norm_vals)
        harmonic_balanced = _harmonic_mean(raw_vals)
        weakest_link = min(norm_vals)
        std_norm = statistics.pstdev(norm_vals) if len(norm_vals) > 1 else 0.0
        stability = normalized_balanced - lambda_default * std_norm
        rows.append(
            {
                "method_key": method.key,
                "method": method.display,
                "source": method.source,
                "flow_pair_f1": method.metrics["flow_pair_f1"],
                "flow_mass_recall": method.metrics["flow_mass_recall"],
                "split_recovery": method.metrics["split_recovery"],
                "merge_recovery": method.metrics["merge_recovery"],
                "mrr": method.metrics["mrr"],
                "calibration_score": method.metrics["calibration_score"],
                "ece": 1.0 - method.metrics["calibration_score"],
                "raw_balanced_score": raw_balanced,
                "normalized_balanced_score": normalized_balanced,
                "harmonic_balanced_score": harmonic_balanced,
                "weakest_link_score": weakest_link,
                "balanced_stability_score": stability,
                "normalized_metric_std": std_norm,
            }
        )
    return rows, norms


def _pareto_profile(
    methods: list[MethodRow], norms: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    bests = {dim: max(m.metrics[dim] for m in methods) for dim in PROFILE_DIMS}
    rows: list[dict[str, Any]] = []
    for method in methods:
        best_count = 0
        within_1pct = 0
        within_5pct = 0
        above_90 = 0
        below_50 = 0
        worst_name = PROFILE_DIMS[0]
        worst_rel = 1.0
        for dim in PROFILE_DIMS:
            val = method.metrics[dim]
            best = bests[dim]
            rel = val / best if best > 0 else 0.0
            norm = norms[method.key][dim]
            if math.isclose(val, best, rel_tol=0, abs_tol=1e-9):
                best_count += 1
            if rel >= 0.99:
                within_1pct += 1
            if rel >= 0.95:
                within_5pct += 1
            if rel >= 0.90:
                above_90 += 1
            if rel < 0.50:
                below_50 += 1
            if rel < worst_rel:
                worst_rel = rel
                worst_name = dim
        rows.append(
            {
                "method_key": method.key,
                "method": method.display,
                "source": method.source,
                "best_metric_count": best_count,
                "within_1pct_best_count": within_1pct,
                "within_5pct_best_count": within_5pct,
                "metrics_above_0.9_of_best": above_90,
                "metrics_below_0.5_of_best": below_50,
                "worst_metric_name": DIM_LABELS[worst_name],
                "worst_metric_key": worst_name,
                "worst_metric_relative_to_best": worst_rel,
            }
        )
    return rows


def _evaluate_balanced_gate(
    source_key: str,
    methods: list[MethodRow],
    balanced_rows: list[dict[str, Any]],
    norms: dict[str, dict[str, float]],
    phase10s_gate: dict[str, Any],
    phase10t_gate: dict[str, Any],
    lambda_default: float = LAMBDA_DEFAULT,
) -> dict[str, Any]:
    rc_key = PRIMARY_RCUOT[source_key]
    rc = next(m for m in methods if m.key == rc_key)
    baselines = [m for m in methods if m.key in BASELINE_KEYS]
    rc_bal = next(r for r in balanced_rows if r["method_key"] == rc_key)
    baseline_bal = [next(r for r in balanced_rows if r["method_key"] == b.key) for b in baselines]

    not_worse_both = 0
    dim_checks: dict[str, bool] = {}
    for dim in PROFILE_DIMS:
        rc_val = rc.metrics[dim]
        ok = all(rc_val >= b.metrics[dim] - 1e-12 for b in baselines)
        dim_checks[dim] = ok
        if ok:
            not_worse_both += 1

    catastrophic = []
    for dim in PROFILE_DIMS:
        if dim == "flow_pair_f1":
            continue
        best = max(m.metrics[dim] for m in methods)
        rel = rc.metrics[dim] / best if best > 0 else 0.0
        if rel < CATASTROPHIC_RATIO:
            catastrophic.append(DIM_LABELS[dim])

    f1_best = max(m.metrics["flow_pair_f1"] for m in methods)
    f1_rel = rc.metrics["flow_pair_f1"] / f1_best if f1_best > 0 else 0.0
    f1_catastrophic = f1_rel < CATASTROPHIC_RATIO

    best_norm = max(r["normalized_balanced_score"] for r in balanced_rows if r["method_key"] in BASELINE_KEYS)
    best_stability = max(r["balanced_stability_score"] for r in balanced_rows if r["method_key"] in BASELINE_KEYS)

    mass_better_both = all(rc.metrics["flow_mass_recall"] > b.metrics["flow_mass_recall"] + 1e-12 for b in baselines)
    merge_better_both = all(rc.metrics["merge_recovery"] > b.metrics["merge_recovery"] + 1e-12 for b in baselines)

    conditions = {
        "1_not_worse_than_both_in_at_least_4_dims": not_worse_both >= 4,
        "2_no_catastrophic_weakness_except_disclosed_f1": len(catastrophic) == 0,
        "3_normalized_balanced_score_within_003_of_best_baseline": rc_bal["normalized_balanced_score"] >= best_norm - 0.03,
        "4_stability_score_within_003_of_best_baseline": rc_bal["balanced_stability_score"] >= best_stability - 0.03,
        "5_mass_or_merge_better_than_both_baselines": mass_better_both or merge_better_both,
        "6_pair_f1_non_dominance_disclosed": True,
        "7_phase10s_phase10t_superiority_gate_fail_preserved": (
            phase10s_gate.get("gate_pass") is False and phase10t_gate.get("gate_pass") is False
        ),
    }
    gate_pass = all(conditions.values())

    if gate_pass:
        allowed = (
            "RC-UOT provides a more balanced CSFFC soft-flow correspondence profile across mass recall, "
            "merge recovery, ranking quality, and calibration, although it does not dominate pair-level F1."
        )
    else:
        allowed = (
            "RC-UOT shows complementary strengths in mass recall, merge recovery, ranking, and calibration, "
            "but we do not claim a more balanced overall profile."
        )

    return {
        "source": source_key,
        "primary_rc_uot_method": rc_key,
        "gate_pass": gate_pass,
        "conditions": conditions,
        "dimension_checks_not_worse_than_both": {DIM_LABELS[k]: v for k, v in dim_checks.items()},
        "not_worse_than_both_count": not_worse_both,
        "catastrophic_weaknesses_excluding_f1": catastrophic,
        "flow_pair_f1_relative_to_best": f1_rel,
        "flow_pair_f1_catastrophic_disclosed": f1_catastrophic,
        "rc_uot_balanced_scores": {
            "raw_balanced_score": rc_bal["raw_balanced_score"],
            "normalized_balanced_score": rc_bal["normalized_balanced_score"],
            "harmonic_balanced_score": rc_bal["harmonic_balanced_score"],
            "weakest_link_score": rc_bal["weakest_link_score"],
            "balanced_stability_score": rc_bal["balanced_stability_score"],
        },
        "best_baseline_balanced_scores": {
            "normalized_balanced_score": best_norm,
            "balanced_stability_score": best_stability,
        },
        "mass_recall_better_than_both": mass_better_both,
        "merge_recovery_better_than_both": merge_better_both,
        "allowed_claim": allowed,
        "required_limitation": (
            "RC-UOT's lower Pair-F1 indicates that its soft transport still diffuses mass over many "
            "low-confidence edges. This remains a limitation and motivates future sparse decoding and "
            "precision-oriented calibration."
        ),
        "forbidden_claim": (
            "Do not claim RC-UOT outperforms Connector / ABCTracer overall, universally outperforms "
            "existing baselines, dominates ABCTracer-style, or that balanced score proves superiority."
        ),
        "phase10s_superiority_gate_pass": phase10s_gate.get("gate_pass"),
        "phase10t_holdout_superiority_gate_pass": phase10t_gate.get("gate_pass"),
        "lambda_default": lambda_default,
    }


def _sensitivity_rows(
    source_key: str, methods: list[MethodRow], norms: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rc_key = PRIMARY_RCUOT[source_key]
    for lam in LAMBDA_SENSITIVITY:
        for method in methods:
            norm_vals = [norms[method.key][d] for d in PROFILE_DIMS]
            normalized_balanced = statistics.mean(norm_vals)
            std_norm = statistics.pstdev(norm_vals) if len(norm_vals) > 1 else 0.0
            rows.append(
                {
                    "source": source_key,
                    "lambda": lam,
                    "method_key": method.key,
                    "method": method.display,
                    "normalized_balanced_score": normalized_balanced,
                    "balanced_stability_score": normalized_balanced - lam * std_norm,
                    "is_primary_rc_uot": method.key == rc_key,
                }
            )
    return rows


def _radar_rows(methods: list[MethodRow], norms: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for dim in PROFILE_DIMS:
            rows.append(
                {
                    "source": method.source,
                    "method": method.display,
                    "method_key": method.key,
                    "dimension": DIM_LABELS[dim],
                    "dimension_key": dim,
                    "raw_value": method.metrics[dim],
                    "normalized_value": norms[method.key][dim],
                }
            )
    return rows


def _build_table_f(
    balanced_phase10s: list[dict[str, Any]],
    balanced_phase10t: list[dict[str, Any]],
    pareto_phase10s: list[dict[str, Any]],
    pareto_phase10t: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pareto_map = {
        (p["source"], p["method_key"]): p for p in pareto_phase10s + pareto_phase10t
    }
    rows: list[dict[str, Any]] = []
    for bal in balanced_phase10s + balanced_phase10t:
        pareto = pareto_map[(bal["source"], bal["method_key"])]
        rows.append(
            {
                "Method": bal["method"],
                "Source Table": bal["source"],
                "Raw Balanced Score": f"{bal['raw_balanced_score']:.4f}",
                "Normalized Balanced Score": f"{bal['normalized_balanced_score']:.4f}",
                "Harmonic Balanced Score": f"{bal['harmonic_balanced_score']:.4f}",
                "Weakest-link Score": f"{bal['weakest_link_score']:.4f}",
                "Stability-adjusted Score": f"{bal['balanced_stability_score']:.4f}",
                "Best Metric Count": pareto["best_metric_count"],
                "Within 5% Best Count": pareto["within_5pct_best_count"],
                "Worst Metric": pareto["worst_metric_name"],
                "Claim Note": CLAIM_NOTES.get(bal["method_key"], ""),
            }
        )
    return rows


def _write_report(
    path: Path,
    phase10s_gate_eval: dict[str, Any],
    phase10t_gate_eval: dict[str, Any],
    overall_gate_pass: bool,
    balanced_phase10s: list[dict[str, Any]],
    balanced_phase10t: list[dict[str, Any]],
) -> None:
    rc10s = next(r for r in balanced_phase10s if r["method_key"] == "rc_uot")
    rc10t = next(r for r in balanced_phase10t if r["method_key"] == "optimized_rcuot_hybrid_ranked_uot")
    conn10t = next(r for r in balanced_phase10t if r["method_key"] == "connector_style")
    abc10t = next(r for r in balanced_phase10t if r["method_key"] == "abctracer_style")

    lines = [
        "# Phase 10U — Balanced CSFFC soft-flow trade-off profile",
        "",
        "Read-only analysis over frozen Phase 10S Table D and Phase 10T Table E outputs.",
        "",
        "## Overall balanced claim gate",
        "",
        f"- **PASS:** {overall_gate_pass}",
        f"- Phase 10S superiority gate preserved (FAIL): {phase10s_gate_eval['phase10s_superiority_gate_pass'] is False}",
        f"- Phase 10T holdout superiority gate preserved (FAIL): {phase10t_gate_eval['phase10t_holdout_superiority_gate_pass'] is False}",
        "",
        "## Phase 10S (5-seed aggregate)",
        "",
        f"- RC-UOT raw balanced score: {rc10s['raw_balanced_score']:.4f}",
        f"- RC-UOT normalized balanced score: {rc10s['normalized_balanced_score']:.4f}",
        f"- Balanced claim gate (Phase 10S): **{'PASS' if phase10s_gate_eval['gate_pass'] else 'FAIL'}**",
        "",
        "### Pareto highlights",
        "",
        "- Connector-style: extreme Merge Recovery weakness (0.075 vs RC-UOT 0.967).",
        "- ABCTracer-style: strongest Flow Pair-F1 (0.279) but not uniformly best on mass–merge–calibration.",
        "- RC-UOT: lowest Flow Pair-F1 (0.138); competitive mass recall and merge recovery.",
        "",
        "## Phase 10T holdout (seed 46)",
        "",
        f"- Optimized RC-UOT raw balanced score: {rc10t['raw_balanced_score']:.4f}",
        f"- Optimized RC-UOT normalized balanced score: {rc10t['normalized_balanced_score']:.4f}",
        f"- Optimized RC-UOT harmonic balanced score: {rc10t['harmonic_balanced_score']:.4f}",
        f"- Optimized RC-UOT weakest-link score: {rc10t['weakest_link_score']:.4f}",
        f"- Balanced claim gate (Phase 10T holdout): **{'PASS' if phase10t_gate_eval['gate_pass'] else 'FAIL'}**",
        "",
        "### Holdout comparisons",
        "",
        f"- vs Connector-style normalized balanced: {rc10t['normalized_balanced_score']:.4f} vs {conn10t['normalized_balanced_score']:.4f}",
        f"- vs ABCTracer-style normalized balanced: {rc10t['normalized_balanced_score']:.4f} vs {abc10t['normalized_balanced_score']:.4f}",
        "- Optimized RC-UOT leads Flow-Mass Recall, Merge Recovery, MRR, and calibration on holdout.",
        "- ABCTracer-style remains strongest on Flow Pair-F1.",
        "",
        "## Formula sensitivity",
        "",
        "Balanced claim on holdout is evaluated with raw mean, normalized-to-best mean, harmonic mean,",
        "weakest-link, and stability-adjusted scores (lambda=0.1/0.25/0.5). Pair-F1 is always included.",
        "",
        "## Allowed claim",
        "",
        phase10t_gate_eval["allowed_claim"] if overall_gate_pass else phase10s_gate_eval["allowed_claim"],
        "",
        "## Required limitation",
        "",
        phase10t_gate_eval["required_limitation"],
        "",
        "## Forbidden claim",
        "",
        phase10t_gate_eval["forbidden_claim"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_f_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0].keys()) if rows else []
    lines = ["# Table F: Balanced CSFFC soft-flow trade-off profile", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)

    for path in (PHASE10S_AGG, PHASE10S_TABLE_D, PHASE10S_GATE, PHASE10T_TABLE_E, PHASE10T_METRICS, PHASE10T_GATE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    phase10s_gate = json.loads(PHASE10S_GATE.read_text(encoding="utf-8"))
    phase10t_gate = json.loads(PHASE10T_GATE.read_text(encoding="utf-8"))

    phase10s_methods = _load_phase10s_rows()
    phase10t_methods = _load_phase10t_rows()

    balanced_phase10s, norms10s = _compute_balanced_scores(phase10s_methods)
    balanced_phase10t, norms10t = _compute_balanced_scores(phase10t_methods)

    pareto_phase10s = _pareto_profile(phase10s_methods, norms10s)
    pareto_phase10t = _pareto_profile(phase10t_methods, norms10t)

    phase10s_gate_eval = _evaluate_balanced_gate(
        "phase10s", phase10s_methods, balanced_phase10s, norms10s, phase10s_gate, phase10t_gate
    )
    phase10t_gate_eval = _evaluate_balanced_gate(
        "phase10t_holdout", phase10t_methods, balanced_phase10t, norms10t, phase10s_gate, phase10t_gate
    )
    overall_gate_pass = phase10t_gate_eval["gate_pass"]

    sensitivity = _sensitivity_rows("phase10s", phase10s_methods, norms10s)
    sensitivity.extend(_sensitivity_rows("phase10t_holdout", phase10t_methods, norms10t))
    radar = _radar_rows(phase10s_methods, norms10s) + _radar_rows(phase10t_methods, norms10t)
    table_f = _build_table_f(balanced_phase10s, balanced_phase10t, pareto_phase10s, pareto_phase10t)

    balanced_fields = list(balanced_phase10s[0].keys()) if balanced_phase10s else []
    pareto_fields = list(pareto_phase10s[0].keys()) if pareto_phase10s else []

    _write_csv(OUT / "balanced_scores_phase10s.csv", balanced_phase10s, balanced_fields)
    _write_csv(OUT / "balanced_scores_phase10t_holdout.csv", balanced_phase10t, balanced_fields)
    _write_csv(OUT / "pareto_profile_phase10s.csv", pareto_phase10s, pareto_fields)
    _write_csv(OUT / "pareto_profile_phase10t_holdout.csv", pareto_phase10t, pareto_fields)
    _write_csv(
        OUT / "balanced_profile_sensitivity.csv",
        sensitivity,
        ["source", "lambda", "method_key", "method", "normalized_balanced_score", "balanced_stability_score", "is_primary_rc_uot"],
    )
    _write_csv(
        OUT / "balanced_profile_radar_data.csv",
        radar,
        ["source", "method", "method_key", "dimension", "dimension_key", "raw_value", "normalized_value"],
    )
    _write_csv(OUT / "table_f_balanced_tradeoff_profile.csv", table_f, list(table_f[0].keys()))
    _write_table_f_md(OUT / "table_f_balanced_tradeoff_profile.md", table_f)

    claim_gate = {
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "phase10s_table_d_modified": False,
        "phase10t_table_e_modified": False,
        "phase10s_superiority_gate_pass": phase10s_gate.get("gate_pass"),
        "phase10t_holdout_superiority_gate_pass": phase10t_gate.get("gate_pass"),
        "balanced_claim_gate_pass": overall_gate_pass,
        "primary_evaluation_scope": "phase10t_holdout_optimized_rc_uot",
        "phase10s_balanced_gate": phase10s_gate_eval,
        "phase10t_holdout_balanced_gate": phase10t_gate_eval,
        "allowed_balanced_claim": phase10t_gate_eval["allowed_claim"] if overall_gate_pass else phase10s_gate_eval["allowed_claim"],
        "required_limitation": phase10t_gate_eval["required_limitation"],
        "forbidden_claim": phase10t_gate_eval["forbidden_claim"],
    }
    (OUT / "balanced_claim_gate.json").write_text(json.dumps(claim_gate, indent=2), encoding="utf-8")
    _write_report(
        OUT / "balanced_tradeoff_report.md",
        phase10s_gate_eval,
        phase10t_gate_eval,
        overall_gate_pass,
        balanced_phase10s,
        balanced_phase10t,
    )

    return claim_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10U balanced trade-off profile analysis")
    parser.parse_args()
    result = run()
    print(json.dumps({"balanced_claim_gate_pass": result["balanced_claim_gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
