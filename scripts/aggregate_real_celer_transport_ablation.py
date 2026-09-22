#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Real Celer Transport Solver Ablation results into summary artifacts.

Reads out/real_celer_transport_ablation/runs/*/metrics.json and produces:
- summary.csv
- solver_ablation_table.md
- claim_boundary_update.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _load_metrics(run_dir: Path) -> dict[str, Any] | None:
    mp = run_dir / "metrics.json"
    if not mp.is_file():
        return None
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_paper_aligned(metrics_list: list[dict], run_root: Path) -> bool:
    """Check if the runs are paper-aligned (not dense_debug, audit would pass).

    First checks for table5_parity_report.json. If parity has not passed,
    results are NOT paper-aligned regardless of metrics.
    """
    parity_path = run_root / "table5_parity_report.json"
    if parity_path.is_file():
        with open(parity_path, "r", encoding="utf-8") as f:
            parity = json.load(f)
        if not parity.get("parity_pass", False):
            return False
        return True

    # Fallback: check metrics for paper alignment
    for m in metrics_list:
        cp_src = m.get("candidate_pool_source", "")
        if cp_src == "dense_debug":
            return False
        # Check after-filter CVR is computed correctly (not copying before-filter)
        cvr_before = m.get("causal_violation_rate_before_filter")
        cvr_after = m.get("causal_violation_rate_after_filter")
        if cvr_before is not None and cvr_after is not None and abs(cvr_before - cvr_after) < 1e-12 and cvr_before > 0.01:
            return False
        # Check table5_parity_pass field exists and is True
        if m.get("table5_parity_pass") is False:
            return False
    return True


def aggregate(run_root: Path) -> int:
    runs_dir = run_root / "runs"
    if not runs_dir.is_dir():
        print(f"ERROR: runs directory not found: {runs_dir}")
        return 1

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _load_metrics(run_dir)
        if m is None:
            print(f"[WARN] No metrics.json in {run_dir.name}")
            continue
        rows.append(m)

    if not rows:
        print("ERROR: No metrics found")
        return 1

    df = pd.DataFrame(rows)
    paper_aligned = _is_paper_aligned(rows, run_root)

    # Find rc_uot_full baseline
    rc_row = df[df["solver"] == "rc_uot_full"]
    # Paper-aligned metrics use "pair_f1", old diagnostic uses "pair_f1_after_filter"
    if "pair_f1_after_filter" in rc_row.columns and not rc_row.empty:
        rc_f1 = float(rc_row["pair_f1_after_filter"].values[0]) if rc_row["pair_f1_after_filter"].values[0] is not None else None
    elif "pair_f1" in rc_row.columns and not rc_row.empty:
        rc_f1 = float(rc_row["pair_f1"].values[0]) if rc_row["pair_f1"].values[0] is not None else None
    else:
        rc_f1 = None
    if "pair_precision_after_filter" in rc_row.columns and not rc_row.empty:
        rc_prec = float(rc_row["pair_precision_after_filter"].values[0]) if rc_row["pair_precision_after_filter"].values[0] is not None else None
    elif "pair_precision" in rc_row.columns and not rc_row.empty:
        rc_prec = float(rc_row["pair_precision"].values[0]) if rc_row["pair_precision"].values[0] is not None else None
    else:
        rc_prec = None
    if "pair_recall_after_filter" in rc_row.columns and not rc_row.empty:
        rc_rec = float(rc_row["pair_recall_after_filter"].values[0]) if rc_row["pair_recall_after_filter"].values[0] is not None else None
    elif "pair_recall" in rc_row.columns and not rc_row.empty:
        rc_rec = float(rc_row["pair_recall"].values[0]) if rc_row["pair_recall"].values[0] is not None else None
    else:
        rc_rec = None

    # Build summary columns
    summary_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        # Map paper-aligned field names to diagnostic names
        ppa = r.get("pair_precision_after_filter") or r.get("pair_precision")
        pra = r.get("pair_recall_after_filter") or r.get("pair_recall")
        pf1 = r.get("pair_f1_after_filter") or r.get("pair_f1")
        row: dict[str, Any] = {
            "run_id": r.get("run_id", ""),
            "solver": r.get("solver", ""),
            "skipped": r.get("skipped", False),
            "skip_reason": r.get("skip_reason", ""),
            "pair_precision_before_filter": r.get("pair_precision_before_filter"),
            "pair_recall_before_filter": r.get("pair_recall_before_filter"),
            "pair_f1_before_filter": r.get("pair_f1_before_filter"),
            "pair_precision_after_filter": ppa,
            "pair_recall_after_filter": pra,
            "pair_f1_after_filter": pf1,
        }
        r_f1_af = r.get("pair_f1_after_filter") or r.get("pair_f1")
        r_prec_af = r.get("pair_precision_after_filter") or r.get("pair_precision")
        r_rec_af = r.get("pair_recall_after_filter") or r.get("pair_recall")
        if rc_f1 is not None and r_f1_af is not None:
            row["delta_f1_vs_rc_uot_full"] = float(r_f1_af) - rc_f1
            row["delta_precision_vs_rc_uot_full"] = (
                float(r_prec_af) - rc_prec
                if r_prec_af is not None
                else None
            )
            row["delta_recall_vs_rc_uot_full"] = (
                float(r_rec_af) - rc_rec
                if r_rec_af is not None
                else None
            )
        else:
            row["delta_f1_vs_rc_uot_full"] = None
            row["delta_precision_vs_rc_uot_full"] = None
            row["delta_recall_vs_rc_uot_full"] = None

        row["top3_recall"] = r.get("top3_recall")
        row["causal_violation_rate_after_filter"] = r.get("causal_violation_rate_after_filter") or r.get("tx_level_cvr")
        row["abstention_rate"] = r.get("abstention_rate")
        row["runtime_sec"] = r.get("runtime_sec")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Write summary.csv
    csv_path = run_root / "summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(summary_df)} rows)")

    # Build solver_ablation_table.md
    transport_type_map = {
        "rc_uot_full": "Unbalanced OT",
        "rc_uot_numpy": "Unbalanced OT (numpy)",
        "balanced_sinkhorn": "Balanced Sinkhorn OT",
        "balanced_emd": "Balanced EMD",
        "hungarian": "Hungarian 1-to-1",
        "greedy_nn": "Greedy NN",
        "cost_ranking": "Cost ranking only",
    }

    lines: list[str] = []
    lines.append("# Real Celer Transport Solver Ablation")
    lines.append("")
    if not paper_aligned:
        lines.append("> **WARNING: These results are diagnostic only and are NOT aligned with the paper pipeline.**")
        lines.append("")
    lines.append(
        "| Solver / scorer | Transport type | Same cost C | Same time filter | Precision | Recall | F1 | Top-3 | tx-CVR | Abstention |"
    )
    lines.append(
        "| --------------- | -------------- | ----------: | ---------------: | --------: | -----: | -: | ----: | -----: | ---------: |"
    )

    for _, r in summary_df.iterrows():
        solver = r.get("solver", "")
        if r.get("skipped"):
            ttype = transport_type_map.get(solver, "-")
            lines.append(
                f"| {solver} | {ttype} | Yes | Yes | - (skipped) | - | - | - | - | - |"
            )
            continue
        ttype = transport_type_map.get(solver, "-")
        prec = f"{r['pair_precision_after_filter']:.3f}" if r.get("pair_precision_after_filter") is not None else "-"
        rec = f"{r['pair_recall_after_filter']:.3f}" if r.get("pair_recall_after_filter") is not None else "-"
        f1 = f"{r['pair_f1_after_filter']:.3f}" if r.get("pair_f1_after_filter") is not None else "-"
        top3 = f"{r['top3_recall']:.3f}" if r.get("top3_recall") is not None else "-"
        cvr = f"{r['causal_violation_rate_after_filter']:.3f}" if r.get("causal_violation_rate_after_filter") is not None else "-"
        abst = f"{r['abstention_rate']:.3f}" if r.get("abstention_rate") is not None else "-"
        lines.append(f"| {solver} | {ttype} | Yes | Yes | {prec} | {rec} | {f1} | {top3} | {cvr} | {abst} |")

    lines.append("")
    md_path = run_root / "solver_ablation_table.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")

    # Build claim_boundary_update.md
    claim_lines: list[str] = []
    claim_lines.append("# Claim Boundary Update - Real Celer Transport Solver Ablation")
    claim_lines.append("")

    if not paper_aligned:
        claim_lines.append(
            "**This run is diagnostic only and is not aligned with the paper pipeline.** "
            "The results below should NOT be used for M1 or any paper-facing claims. "
            "Re-run with `--candidate-pool-source phase10r` and ensure the time filter is functioning correctly "
            "(causal_violation_rate_after_filter must be 0, abstention_rate must be > 0)."
        )
    else:
        cost_ranking = df[df["solver"] == "cost_ranking"]
        if rc_row.empty:
            claim_lines.append("**RC-UOT full run missing. Cannot generate claim boundary.**")
        elif cost_ranking.empty:
            claim_lines.append("**Cost ranking baseline missing. Cannot generate claim boundary.**")
        else:
            rc_p = rc_prec if rc_prec is not None else 0
            rc_r = rc_rec if rc_rec is not None else 0
            rc_f1v = rc_f1 if rc_f1 is not None else 0

            cr_p_val = cost_ranking["pair_precision_after_filter"].values[0] if "pair_precision_after_filter" in cost_ranking.columns and cost_ranking["pair_precision_after_filter"].values[0] is not None else None
            cr_r_val = cost_ranking["pair_recall_after_filter"].values[0] if "pair_recall_after_filter" in cost_ranking.columns and cost_ranking["pair_recall_after_filter"].values[0] is not None else None
            cr_f1_val = cost_ranking["pair_f1_after_filter"].values[0] if "pair_f1_after_filter" in cost_ranking.columns and cost_ranking["pair_f1_after_filter"].values[0] is not None else None
            cr_p = float(cr_p_val) if cr_p_val is not None else 0
            cr_r = float(cr_r_val) if cr_r_val is not None else 0
            cr_f1v = float(cr_f1_val) if cr_f1_val is not None else 0

            claim_lines.append(
                f"Under the same candidate pool, cost matrix, quotient decoder, and joint "
                f"time-admissible filter, RC-UOT obtains P/R/F1 = {rc_p:.3f}/{rc_r:.3f}/{rc_f1v:.3f}. "
                f"The cost-ranking baseline obtains P/R/F1 = {cr_p:.3f}/{cr_r:.3f}/{cr_f1v:.3f}."
            )
            claim_lines.append("")

            delta_f1 = rc_f1v - cr_f1v
            if delta_f1 >= 0.03:
                claim_lines.append(
                    "RC-UOT retains a measurable solver-level contribution on Real Celer "
                    f"(Delta-F1 = {delta_f1:+.3f}). "
                    "Therefore, the Real-Celer high-precision operating point benefits from "
                    "unbalanced transport by a measurable margin."
                )
            else:
                claim_lines.append(
                    "On Real Celer, the high-precision point is mainly driven by the "
                    "evidence-constrained and time-admissible decoder; the split/merge "
                    "expressivity of UOT is therefore supported primarily by the controlled "
                    "stress tests, not by Real-Celer solver separation alone "
                    f"(Delta-F1 = {delta_f1:+.3f})."
                )
            claim_lines.append("")
            claim_lines.append(
                "This ablation separates the contribution of the transport solver from "
                "the contribution of the evidence-constrained decoder."
            )

    claim_path = run_root / "claim_boundary_update.md"
    claim_path.write_text("\n".join(claim_lines), encoding="utf-8")
    print(f"Wrote {claim_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate Real Celer Transport Solver Ablation")
    p.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "real_celer_transport_ablation",
        help="Run root directory",
    )
    args = p.parse_args(argv)
    return aggregate(Path(args.run_root))


if __name__ == "__main__":
    raise SystemExit(main())
