"""Root-cause diagnosis for negative delay / CVR in time-causal diagnostics."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import (
    audit_production_plan,
    compute_flow_boundary_variants,
    compute_gt_tx_pair_delays,
    compute_sign_sanity,
    delay_distribution,
    trace_current_delay_usage,
)
from cross.application.experiments.uot_cache_utils import load_flow_segments


def _resolve_bnb_path(bnb_arg: Path | None, run_dir: Path) -> tuple[Path | None, str | None]:
    if bnb_arg and Path(bnb_arg).is_file():
        return Path(bnb_arg), None
    manifest = run_dir / "experiment_manifest.json"
    if manifest.is_file():
        inputs = json.loads(manifest.read_text(encoding="utf-8")).get("inputs") or {}
        bnb = inputs.get("bnb_csv")
        if bnb and Path(bnb).is_file():
            return Path(bnb), None
    return None, "bnb_csv not found; pass --bnb or ensure experiment_manifest.json lists bnb_csv"


def _load_sweep_baseline(sensitivity_dir: Path | None) -> dict[str, Any] | None:
    if not sensitivity_dir or not Path(sensitivity_dir).is_dir():
        return None
    manifest = Path(sensitivity_dir) / "sensitivity_manifest.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    baseline = data.get("baseline_config")
    return {
        "source": str(manifest.resolve()),
        "baseline_config": baseline,
        "note": "Sweep CVR/delay are from proxy UOT re-solve, not headline production plan",
    }


def _classify_case(
    gt_stats: dict[str, Any],
    flow_variants: dict[str, Any],
    sign_sanity: dict[str, Any],
) -> dict[str, Any]:
    gt_med = gt_stats.get("median")
    legacy_med = flow_variants.get("dst_start_minus_src_end", {}).get("median")
    alt_med = flow_variants.get("dst_end_minus_src_start", {}).get("median")

    if gt_med is None:
        return {"case": "unresolved", "reason": "insufficient GT tx timestamps"}

    if sign_sanity.get("sign_reversed_if_src_minus_dst_positive_and_dst_minus_src_negative"):
        return {
            "case": 3,
            "label": "sign_error_suspected",
            "delay_direction_error_suspected": True,
        }

    if float(gt_med) < 0:
        return {
            "case": 2,
            "label": "gt_delay_median_negative",
            "delay_direction_error_suspected": True,
            "recommendation": "Stop sweep; inspect label direction / timestamp parsing / src-dst swap",
        }

    if legacy_med is not None and float(legacy_med) < 0 and float(gt_med) > 0:
        return {
            "case": 1,
            "label": "flow_boundary_mismatch",
            "delay_direction_error_suspected": False,
            "root_cause": (
                "GT tx-pair delay median is positive but legacy flow delay "
                "(dst_flow.start - src_flow.end) median is negative; "
                "causality penalty uses flow boundary unsuitable for aggregated source flows."
            ),
            "recommended_policy": "tx_if_available_else_flow_representative",
            "recommended_formula": "dst_flow.end_time - src_flow.start_time (flow representative)",
            "legacy_formula": "dst_flow.start_time - src_flow.end_time",
            "alt_variant_median_sec": {
                "dst_end_minus_src_start": alt_med,
                "dst_start_minus_src_end": legacy_med,
            },
        }

    return {
        "case": "other",
        "label": "no_single_flow_boundary_mismatch",
        "delay_direction_error_suspected": False,
    }


def _negative_gt_samples(gt_rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    neg = [r for r in gt_rows if float(r.get("gt_delay_tx_sec", 0)) < 0]
    neg.sort(key=lambda r: float(r["gt_delay_tx_sec"]))
    return neg[:limit]


def reconcile_unmatched_mass(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    eval_path = run_dir / "eval" / "uot_evaluation_metrics.json"
    if eval_path.is_file():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        um = ev.get("unmatched_mass") or {}
        entries.append(
            {
                "artifact": str(eval_path.relative_to(run_dir.parent) if run_dir.parent.name == "out" else eval_path.name),
                "field": "unmatched_mass_ratio",
                "value": ev.get("unmatched_mass_ratio"),
                "definition": "total_unmatched_source + total_unmatched_target over transported + unmatched (UOT marginals)",
                "formula": "(sum(unmatched_source) + sum(unmatched_target)) / (P.sum() + unmatched_total)",
                "paper_facing": True,
            }
        )
        entries.append(
            {
                "artifact": "eval/uot_evaluation_metrics.json",
                "field": "total_unmatched_source",
                "value": um.get("total_unmatched_source"),
                "definition": "source-side unmatched mass from UOT solver marginals",
            }
        )

    ab_path = run_dir / "ablation_results.csv"
    if ab_path.is_file():
        ab = pd.read_csv(ab_path)
        row = ab[ab["experiment_name"] == "leave_anchor_out_strict"]
        if not row.empty:
            entries.append(
                {
                    "artifact": "ablation_results.csv (leave_anchor_out_strict)",
                    "field": "unmatched_mass_ratio",
                    "value": float(row.iloc[0]["unmatched_mass_ratio"]),
                    "definition": "same as eval uot_evaluation_metrics via flow_metrics._soft_correspondence_proxies",
                    "paper_facing": True,
                }
            )

    fm_summary = out_dir / "flow_mass_calibration_summary.json"
    if fm_summary.is_file():
        fm = json.loads(fm_summary.read_text(encoding="utf-8"))
        entries.append(
            {
                "artifact": "flow_mass_calibration_summary.json",
                "field": "unmatched_mass_ratio",
                "value": fm.get("unmatched_mass_ratio"),
                "definition": "BUGGY PROXY: 1 - P.sum()/n_source_rows (not UOT marginal unmatched mass)",
                "paper_facing": False,
                "status": "deprecated_do_not_cite",
            }
        )
    else:
        p_path = run_dir / "uot" / "uot_transport_matrix.npz"
        if p_path.is_file():
            p = np.load(p_path)["P"]
            buggy = float(max(0.0, 1.0 - float(p.sum()) / max(p.shape[0], 1)))
            entries.append(
                {
                    "artifact": "derived from uot_transport_matrix.npz",
                    "field": "unmatched_mass_ratio_buggy_proxy",
                    "value": buggy,
                    "definition": "1 - P.sum()/n_source_rows",
                    "paper_facing": False,
                    "status": "deprecated_do_not_cite",
                }
            )

    paper_def = (
        "Paper-facing unmatched_mass_ratio = (total_unmatched_source + total_unmatched_target) / "
        "(transported_mass + total_unmatched), from UOT solver marginals (see eval/uot_evaluation_metrics.json "
        "and flow_metrics._soft_correspondence_proxies). Do NOT use flow_mass_calibration_summary "
        "1-P.sum()/n_rows proxy in manuscript."
    )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_facing_definition": paper_def,
        "paper_facing_value_leave_anchor_out_strict": next(
            (e["value"] for e in entries if e.get("paper_facing") and "leave_anchor_out" in str(e.get("artifact", ""))),
            next((e["value"] for e in entries if e.get("paper_facing")), None),
        ),
        "entries": entries,
    }

    (out_dir / "unmatched_mass_ratio_reconciliation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        "# Unmatched mass ratio reconciliation",
        "",
        f"Generated: {result['generated_at_utc']}",
        "",
        "## Paper-facing definition",
        "",
        paper_def,
        "",
        f"**Canonical value (leave-anchor-out strict):** {result['paper_facing_value_leave_anchor_out_strict']}",
        "",
        "| Artifact | Field | Value | Definition | Paper-facing |",
        "|----------|-------|------:|------------|--------------|",
    ]
    for e in entries:
        md.append(
            f"| {e.get('artifact','')} | {e.get('field','')} | {e.get('value','')} | "
            f"{e.get('definition','')} | {e.get('paper_facing', False)} |"
        )
    (out_dir / "unmatched_mass_ratio_reconciliation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


def run_delay_semantics_diagnosis(
    *,
    eth_path: Path,
    bnb_path: Path | None,
    label_path: Path,
    run_dir: Path,
    sensitivity_dir: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)

    bnb_resolved, bnb_err = _resolve_bnb_path(bnb_path, run_dir)
    eth_df = pd.read_csv(eth_path)
    label_df = pd.read_csv(label_path)

    missing_cols: list[str] = []
    if bnb_resolved is None:
        missing_cols.append(bnb_err or "bnb_csv")
        bnb_df = pd.DataFrame()
    else:
        bnb_df = pd.read_csv(bnb_resolved)

    eth_flows = load_flow_segments(run_dir / "uot" / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(run_dir / "uot" / "uot_flow_segments_bnb.csv")

    gt_rows, gt_stats, ts_missing = compute_gt_tx_pair_delays(eth_df, bnb_df, label_df)
    missing_cols.extend(ts_missing)
    flow_variants = compute_flow_boundary_variants(label_df, eth_flows, bnb_flows)
    sign_sanity = compute_sign_sanity(gt_rows)
    current_delay = trace_current_delay_usage()
    production_audit = audit_production_plan(
        run_dir=run_dir,
        label_df=label_df,
        eth_df=eth_df,
        bnb_df=bnb_df,
        eth_flows=eth_flows,
        bnb_flows=bnb_flows,
        eth_path=eth_path,
        bnb_path=bnb_resolved,
    )
    sweep_baseline = _load_sweep_baseline(sensitivity_dir)
    classification = _classify_case(gt_stats, flow_variants, sign_sanity)

    if classification.get("case") == 2:
        classification["negative_gt_samples"] = _negative_gt_samples(gt_rows, 20)

    unmatched_recon = reconcile_unmatched_mass(run_dir, out_dir.parent / "flow_mass_calibration")

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "eth": str(eth_path.resolve()),
            "bnb": str(bnb_resolved.resolve()) if bnb_resolved else None,
            "label": str(label_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "sensitivity_dir": str(sensitivity_dir.resolve()) if sensitivity_dir else None,
        },
        "missing_columns": missing_cols,
        "A_gt_tx_pair_delay": gt_stats,
        "B_flow_boundary_variants": flow_variants,
        "C_sign_sanity": sign_sanity,
        "current_delay_usage": current_delay,
        "current_sweep_formula_used": "dst_flow.start_time - src_flow.end_time (legacy_flow_boundary)",
        "production_plan_delay_audit": production_audit,
        "sweep_proxy_baseline": sweep_baseline,
        "classification": classification,
        "unmatched_mass_reconciliation": unmatched_recon,
        "paper_ready": False,
        "time_causal_sensitivity_status": "blocked_by_delay_semantics",
    }

    if classification.get("case") == 1:
        report["recommended_next_step"] = (
            "Re-run lightweight time/causal sweep with --time-delay-policy tx_if_available_else_flow_representative"
        )
    elif classification.get("case") == 2:
        report["recommended_next_step"] = "Do not write CVR conclusions; inspect GT label direction and timestamps"
    elif classification.get("case") == 3:
        report["recommended_next_step"] = "Fix delay sign in cost_matrix and re-solve from cache"

    (out_dir / "delay_semantics_diagnosis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = _write_diagnosis_md(report)
    (out_dir / "delay_semantics_diagnosis.md").write_text(md, encoding="utf-8")

    prod_md = _write_production_audit_md(production_audit)
    (out_dir / "production_plan_delay_audit.md").write_text(prod_md, encoding="utf-8")
    (out_dir / "production_plan_delay_audit.json").write_text(
        json.dumps(production_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir.parent / "flow_mass_calibration" / "production_plan_delay_audit.json").write_text(
        json.dumps(production_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return report


def _write_production_audit_md(audit: dict[str, Any]) -> str:
    lines = [
        "# Production plan delay audit",
        "",
        f"**Status:** {audit.get('production_cvr_status')}",
        "",
    ]
    if audit.get("production_cvr_status") != "ok":
        lines.append(f"Reason: {audit.get('reason', 'unknown')}")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Metric | Value |",
            "|--------|------:|",
            f"| production_pair_f1 | {audit.get('production_pair_f1')} |",
            f"| production_top1_recall | {audit.get('production_top1_recall')} |",
            f"| production_top3_recall | {audit.get('production_top3_recall')} |",
            f"| production_causality_violation_rate_tx_level | {audit.get('production_causality_violation_rate_tx_level')} |",
            f"| production_causality_violation_rate_flow_level | {audit.get('production_causality_violation_rate_flow_level')} |",
            f"| production_delay_tx_median_sec | {audit.get('production_delay_tx_median_sec')} |",
            f"| production_delay_flow_median_sec | {audit.get('production_delay_flow_median_sec')} |",
            "",
            "All metrics from the same headline decoded plan (`matching/matching_pairs.csv`).",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_diagnosis_md(report: dict[str, Any]) -> str:
    gt = report["A_gt_tx_pair_delay"]
    cls = report["classification"]
    prod = report["production_plan_delay_audit"]
    lines = [
        "# Delay semantics diagnosis",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Classification",
        "",
        f"- **Case:** {cls.get('case')} ({cls.get('label')})",
        f"- **Root cause:** {cls.get('root_cause', cls.get('reason', 'see JSON'))}",
        "",
        "## A. GT tx-pair delay (physical)",
        "",
        f"| n_pairs | median (sec) | negative_ratio | p05 | p95 |",
        f"|--------:|-------------:|---------------:|----:|----:|",
        f"| {gt.get('n_pairs')} | {gt.get('median')} | {gt.get('negative_ratio')} | {gt.get('p05')} | {gt.get('p95')} |",
        "",
        "## B. Flow boundary variants (labeled pairs)",
        "",
        "| Variant | median (sec) | negative_ratio |",
        "|---------|-------------:|---------------:|",
    ]
    for name, stats in report["B_flow_boundary_variants"].items():
        lines.append(f"| {name} | {stats.get('median')} | {stats.get('negative_ratio')} |")
    lines.extend(
        [
            "",
            f"**Current model uses:** `{report.get('current_sweep_formula_used')}`",
            "",
            "## Production plan (headline F1=0.321 same source)",
            "",
            f"- pair_f1: {prod.get('production_pair_f1')}",
            f"- CVR tx-level: {prod.get('production_causality_violation_rate_tx_level')}",
            f"- CVR flow-level (legacy boundary): {prod.get('production_causality_violation_rate_flow_level')}",
            f"- delay tx median: {prod.get('production_delay_tx_median_sec')} sec",
            f"- delay flow median: {prod.get('production_delay_flow_median_sec')} sec",
            "",
            "> Do NOT attach sweep proxy CVR (F1≈0.236) to headline production F1.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose delay semantics for time/causal diagnostics.")
    parser.add_argument("--eth", type=Path, required=True)
    parser.add_argument("--bnb", type=Path, default=None)
    parser.add_argument("--label", type=Path, default=Path("label/celer_label.csv"))
    parser.add_argument("--run-dir", type=Path, default=Path("out/leave_anchor_out_real"))
    parser.add_argument("--sensitivity-dir", type=Path, default=Path("out/time_causal_sensitivity"))
    parser.add_argument("--out", type=Path, default=Path("out/delay_semantics_diagnosis"))
    args = parser.parse_args()
    result = run_delay_semantics_diagnosis(
        eth_path=args.eth,
        bnb_path=args.bnb,
        label_path=args.label,
        run_dir=args.run_dir,
        sensitivity_dir=args.sensitivity_dir,
        out_dir=args.out,
    )
    print(json.dumps(
        {
            "classification": result["classification"],
            "gt_median_sec": result["A_gt_tx_pair_delay"].get("median"),
            "production_pair_f1": result["production_plan_delay_audit"].get("production_pair_f1"),
            "production_cvr_tx": result["production_plan_delay_audit"].get("production_causality_violation_rate_tx_level"),
            "paper_ready": result["paper_ready"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
