#!/usr/bin/env python
"""M1 Factorial Experiment: 2x3 = causal_mask(on/off) x solver(rc_uot, hungarian, greedy_nn).

Loads data from data/ directory via auto-discovery.

Usage:
    python scripts/run_m1_factorial.py --config config/experiments/m1_factorial.yaml --data-root data --auto-discover-data --output-dir results/m1_factorial
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="M1 Factorial Experiment")
    p.add_argument("--config", type=Path, default=Path("config/experiments/m1_factorial.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("results/m1_factorial"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--auto-discover-data", action="store_true", default=True)
    p.add_argument("--print-data-summary", action="store_true", default=True)
    p.add_argument("--limit-cases", type=int, default=None)
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from cross.domain.uot.m1_ablation.data_adapter import discover_data, print_data_summary
    from cross.domain.uot.m1_ablation import (
        FACTORIAL_SOLVERS, get_solver, decode_with_fixed_rc_uot_q, DecodeConfig,
        compute_metrics, compute_stratified_metrics, assign_structure_label,
    )

    data_cfg = cfg.get("data", {})
    prod_root = _REPO / "out" / "uot_delay_fixed_production"
    if not (prod_root / "uot" / "uot_cost_matrix.npz").is_file():
        prod_root = None

    print("=" * 70)
    print("M1 Factorial Experiment (2x3, REAL DATA)")
    print(f"  Data root: {args.data_root}")
    print(f"  Production root: {prod_root or 'NONE'}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    try:
        dataset = discover_data(
            data_root=args.data_root, production_root=prod_root,
            auto_discover=args.auto_discover_data, config=data_cfg,
        )
    except FileNotFoundError as e:
        print(f"\nDATA DISCOVERY FAILED: {e}")
        return 1

    if args.print_data_summary:
        summary_text = print_data_summary(dataset)
        print(summary_text)
        (output_dir / "data_summary.md").write_text(summary_text, encoding="utf-8")

    if args.limit_cases:
        gt_keys = list(dataset.ground_truth.keys())[:args.limit_cases]
        dataset.ground_truth = {k: dataset.ground_truth[k] for k in gt_keys}
        print(f"\n  LIMITED to {args.limit_cases} cases")

    factors = cfg.get("factors", {})
    causal_conditions = factors.get("causal_condition", ["no_causal_mask", "with_causal_mask"])
    solvers = factors.get("solver", list(FACTORIAL_SOLVERS))
    eval_cfg = cfg.get("evaluation", {})
    bootstrap_cfg = cfg.get("bootstrap", {"n": 1000, "ci": 0.95})
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    decode_config = DecodeConfig(
        strategy=cfg.get("decoder", {}).get("strategy", "joint_time_admissible_filter"),
        delay_policy=cfg.get("decoder", {}).get("delay_policy", "tx_if_available_else_flow_representative"),
        tx_decode_policy=cfg.get("decoder", {}).get("tx_decode_policy", "amount_nearest_positive_delay"),
    )

    # Paper pipeline: evaluate on ALL ground truth pairs
    all_truth = dict(dataset.ground_truth)
    structure_labels = assign_structure_label(dataset.ground_truth, dataset.source_flows, dataset.target_flows)

    all_rows = []
    all_stratified = []

    print(f"\n[2/4] Running {len(causal_conditions) * len(solvers)} conditions...")
    C = dataset.C
    n_sf = dataset.n_source

    for causal_cond in causal_conditions:
        use_mask = causal_cond == "with_causal_mask"
        feasible = dataset.feasible_mask if use_mask else np.ones(C.shape, dtype=bool)
        print(f"\n--- {causal_cond} (mask={'on' if use_mask else 'off'}) ---")

        for sname in solvers:
            try:
                solver = get_solver(sname)
            except ValueError:
                continue
            transport_cfg = cfg.get("transport", {})
            solver_cfg = {"temperature": float(transport_cfg.get("temperature", 1.0))}
            result = solver.solve(C=C, feasible_mask=feasible,
                                   source_mass=dataset.source_mass, target_mass=dataset.target_mass,
                                   config=solver_cfg)
            T = result.T

            decode_result = decode_with_fixed_rc_uot_q(
                T=T, C=C,
                source_flows=dataset.source_flows, target_flows=dataset.target_flows,
                src_all=dataset.src_all, dst_norm=dataset.dst_norm,
                truth=all_truth, eth_ts=dataset.eth_ts, bnb_ts=dataset.bnb_ts,
                config=decode_config,
            )

            metrics = compute_metrics(
                mapping=decode_result.mapping, truth=all_truth,
                n_source_flows=n_sf,
                n_abstained=decode_result.n_abstained, n_predicted=decode_result.n_predicted,
                solver_name=sname, bootstrap_config=bootstrap_cfg,
            )

            row = {
                "causal_condition": causal_cond, "solver": sname,
                "precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1,
                "coverage": metrics.coverage, "abstention_rate": metrics.abstention_rate,
                "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn,
                "n_predicted": metrics.n_predicted, "n_abstained": metrics.n_abstained,
            }
            if metrics.precision_ci:
                row["precision_ci_low"] = metrics.precision_ci.lower_95
                row["precision_ci_high"] = metrics.precision_ci.upper_95
            all_rows.append(row)
            print(f"  {sname}: P={metrics.precision:.4f} R={metrics.recall:.4f} F1={metrics.f1:.4f}")

            if eval_cfg.get("stratify_by_structure", True):
                strat = compute_stratified_metrics(
                    mapping=decode_result.mapping, truth=all_truth,
                    structure_labels=structure_labels,
                    n_source_flows=n_sf,
                    n_abstained=decode_result.n_abstained, n_predicted=decode_result.n_predicted,
                    solver_name=sname, bootstrap_config=bootstrap_cfg,
                )
                for label, sr in strat.items():
                    all_stratified.append({
                        "causal_condition": causal_cond, "solver": sname, "structure": label,
                        "support": sr.support, "precision": sr.precision, "recall": sr.recall,
                        "f1": sr.f1, "coverage": sr.coverage, "abstention_rate": sr.abstention_rate,
                        "tp": sr.tp, "fp": sr.fp, "fn": sr.fn,
                    })

    print(f"\n[3/4] Writing results...")
    _write_csv(all_rows, output_dir / "factorial_summary.csv")
    _write_csv(all_stratified, output_dir / "factorial_by_structure.csv")
    _generate_factorial_report(all_rows, all_stratified, output_dir)

    print(f"\n[4/4] Done. Results in {output_dir}")
    return 0


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {path} ({len(rows)} rows)")


def _generate_factorial_report(all_rows, all_stratified, output_dir):
    lines = [
        "# M1 Factorial Experiment Report",
        "",
        f"Generated: {_utc()}",
        "",
        "### Research Question",
        "Is the 0.589 → 0.889 precision jump mainly due to the shared temporal feasibility filter?",
        "",
        "## Overall Results",
        "",
        "| Causal Condition | Solver | Precision | Recall | F1 | Coverage | Abstention |",
        "|-----------------|--------|-----------|--------|----|----------|------------|",
    ]
    for r in all_rows:
        lines.append(f"| {r['causal_condition']} | {r['solver']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['coverage']:.4f} | {r['abstention_rate']:.4f} |")

    # Compute mask gain
    lines += ["", "### Causal Mask Gain (Δ Precision)", ""]
    for sname in sorted(set(r["solver"] for r in all_rows)):
        no = next((r for r in all_rows if r["causal_condition"] == "no_causal_mask" and r["solver"] == sname), None)
        wi = next((r for r in all_rows if r["causal_condition"] == "with_causal_mask" and r["solver"] == sname), None)
        if no and wi:
            gain = wi["precision"] - no["precision"]
            lines.append(f"- **{sname}**: {no['precision']:.4f} → {wi['precision']:.4f} ({gain:+.4f})")

    # Interpretation
    lines += ["", "## Interpretation", ""]
    wi_rows = [r for r in all_rows if r["causal_condition"] == "with_causal_mask"]
    no_rows = [r for r in all_rows if r["causal_condition"] == "no_causal_mask"]
    rc_w = next((r for r in wi_rows if r["solver"] == "rc_uot"), None)
    hung_w = next((r for r in wi_rows if r["solver"] == "hungarian"), None)

    if rc_w and hung_w:
        diff = abs(rc_w["precision"] - hung_w["precision"])
        if diff < 0.05:
            lines.append(
                "Under the causal mask, RC-UOT and Hungarian achieve very similar precision, "
                "confirming that the temporal admissibility filter accounts for the bulk of "
                "the precision improvement."
            )
        elif rc_w["precision"] > hung_w["precision"] + 0.03:
            lines.append(
                "RC-UOT achieves meaningfully higher precision than Hungarian under the causal mask, "
                "indicating genuine solver contribution beyond the filter."
            )

    mask_gains = {}
    for sname in sorted(set(r["solver"] for r in all_rows)):
        no = next((r for r in no_rows if r["solver"] == sname), None)
        wi = next((r for r in wi_rows if r["solver"] == sname), None)
        if no and wi:
            mask_gains[sname] = wi["precision"] - no["precision"]

    if all(v > 0.05 for v in mask_gains.values()):
        lines.append("The causal hard mask provides a large precision gain across all solvers, "
                     "confirming it as the primary driver of precision improvement.")

    lines += ["", "## Stratified Results", ""]
    for r in all_stratified:
        lines.append(
            f"| {r['causal_condition']} | {r['solver']} | {r['structure']} | "
            f"{r['support']} | P={r['precision']:.4f} | R={r['recall']:.4f} | F1={r['f1']:.4f} |"
        )

    (output_dir / "factorial_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {output_dir / 'factorial_report.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
