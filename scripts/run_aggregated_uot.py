#!/usr/bin/env python
"""Phase 3: Run RC-UOT on aggregated flow segments and compare with baseline."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"<REPO>")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.standalone_flow_uot import run_standalone_flow_uot

AGG_DIR = REPO / "out" / "bsc_open_independent_v1" / "stage5_6_flow_aggregation"
OUT_DIR = AGG_DIR / "rc_uot_results"
BASELINE_METRICS = REPO / "out" / "bsc_open_independent_v1" / "stage5_5_candidate_completion" / "rc_uot_results" / "w1.0h_r0.05_rm0.5" / "metrics.json"

SRC_FLOWS = AGG_DIR / "src_flows_aggregated.csv"
DST_FLOWS = AGG_DIR / "dst_flows_aggregated.csv"
FLOW_LABELS = AGG_DIR / "flow_labels_aggregated.csv"

COST_WEIGHTS = {
    "amount": 0.35, "time": 0.25, "route": 0.15,
    "risk": 0.15, "graph": 0.05, "evidence": 0.05,
}


def run_aggregated_uot():
    print("=== Phase 3: RC-UOT on Aggregated Flow Segments ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        {"reg": 0.05, "reg_m": 0.5, "label": "w3h_r0.05_rm0.5"},
        {"reg": 0.05, "reg_m": 1.0, "label": "w3h_r0.05_rm1.0"},
        {"reg": 0.01, "reg_m": 0.1, "label": "w3h_r0.01_rm0.1"},
    ]

    results = []
    for cfg in configs:
        label = cfg["label"]
        run_dir = OUT_DIR / label
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Config: {label} ---")
        t0 = time.time()

        result = run_standalone_flow_uot(
            out_dir=run_dir,
            src_flows_csv=SRC_FLOWS,
            dst_flows_csv=DST_FLOWS,
            flow_labels_csv=FLOW_LABELS,
            uot_reg=cfg["reg"],
            uot_reg_m=cfg["reg_m"],
            uot_decode_threshold=0.05,
            uot_cost_weights=COST_WEIGHTS,
            uot_backend="numpy",
            uot_max_delay_sec=21600.0,
            uot_causal_violation_penalty=5.0,
            uot_lambda_risk=0.25,
            uot_causal_infeasible_delay_sec=None,
            uot_export_matrix=False,
            uot_export_cost_components=False,
            uot_allow_unmatched=True,
            uot_use_graph_embedding=False,
            graph_ranker_checkpoint=None,
            uot_ablation="none",
            run_flow_baselines=True,
            flow_label_min_confidence=0.0,
            uot_flow_dst_top_k=200,
            uot_flow_max_matrix_cells=6_000_000,
        )

        elapsed = time.time() - t0

        metrics_path = run_dir / "uot_evaluation_metrics.json"
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        ab_path = run_dir / "ablation_metrics.csv"
        ablations = None
        if ab_path.exists():
            ablations = pd.read_csv(ab_path).to_dict("records")

        results.append({
            "config": label,
            "elapsed_s": elapsed,
            "metrics": metrics,
            "ablations": ablations,
        })

        f1 = metrics.get("flow_pair_f1", "N/A")
        prec = metrics.get("flow_pair_precision", "N/A")
        rec = metrics.get("flow_pair_recall", "N/A")
        mass_rec = metrics.get("flow_mass_recall", "N/A")
        top1 = metrics.get("top1_flow_correspondence_accuracy", "N/A")
        print(f"  F1={f1} P={prec} R={rec} mass_recall={mass_rec} top1={top1}")
        print(f"  Elapsed: {elapsed:.1f}s")

    # Compare with baseline
    print("\n=== Baseline Comparison ===")
    if BASELINE_METRICS.exists():
        with open(BASELINE_METRICS) as f:
            baseline = json.load(f)
        print(f"  Baseline (1-tx, w1.0h, r0.05 rm0.5):")
        print(f"    flow_pair_f1: {baseline.get('flow_pair_f1', 'N/A')}")
        print(f"    flow_pair_precision: {baseline.get('flow_pair_precision', 'N/A')}")
        print(f"    flow_pair_recall: {baseline.get('flow_pair_recall', 'N/A')}")
        print(f"    flow_mass_recall: {baseline.get('flow_mass_recall', 'N/A')}")
        print(f"    top1_flow_acc: {baseline.get('top1_flow_correspondence_accuracy', 'N/A')}")
    else:
        baseline = {}
        print("  Baseline metrics not found. Checking other paths...")
        alt = REPO / "out" / "bsc_open_independent_v1" / "stage5_5_candidate_completion" / "rc_uot_results" / "baseline_results.csv"
        if alt.exists():
            bl_df = pd.read_csv(alt)
            print(f"  Found baseline_results.csv: {bl_df.to_dict('records')}")

    # Summary
    summary = {
        "aggregated": results,
        "baseline": baseline if isinstance(baseline, dict) else str(baseline),
        "note": "Aggregated flows use 3h windows, grouped by sender/receiver+token+window",
    }
    with open(OUT_DIR / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    run_aggregated_uot()
