"""Stress regression with the LOCKED decoder — re-decodes the previous study's stress plans
(no re-solving; the frozen UOT/BOT transport matrices are reused).

Compares per ladder/level: RC-UOT-Q legacy (D0) vs RC-UOT-Q locked, Balanced-OT legacy vs
locked, and Threshold-MM frozen. Purpose: check whether the sparsification decoder broke
the unmatched / mass-mismatch / decoy robustness behavior observed in the previous study.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import decode_threshold_mm  # noqa: E402
from decoder_audit.da_common import (  # noqa: E402
    AUDIT, BRIDGES, CALIB_SEEDS, THRESHOLD_MM_CUTOFF, bootstrap_ci, decode, evaluate_edges,
    load_stress_cell,
)

LADDERS = {"mass": [0.0, 0.05, 0.10, 0.20, 0.40],
           "unmatched": [0.0, 0.1, 0.2, 0.3, 0.4],
           "decoy": [1, 2, 4, 8],
           "noise": [1, 2, 4]}


def main() -> int:
    lock_path = AUDIT / "calibration" / "locked_decoder.json"
    if not lock_path.is_file():
        raise SystemExit("locked_decoder.json missing — aborting stress regression.")
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_cfg = {"name": locked["decoder_name"], "family": locked["decoder_family"],
                "params": locked["params"]}
    legacy_cfg = {"name": "D0_legacy", "family": "D0", "params": {"thr": 1e-9}}
    out = AUDIT / "stress_regression"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ladder, levels in LADDERS.items():
        for level in levels:
            lv = str(level)  # stress instance dirs use the plain str(level) key
            for bridge in BRIDGES:
                for seed in CALIB_SEEDS:
                    cell = load_stress_cell(ladder, lv, bridge, seed)
                    variants = [
                        ("RC-UOT-Q legacy", cell["P_uot"], legacy_cfg),
                        ("RC-UOT-Q locked", cell["P_uot"], lock_cfg),
                        ("Balanced-OT legacy", cell["P_bot"], legacy_cfg),
                        ("Balanced-OT locked", cell["P_bot"], lock_cfg),
                    ]
                    for name, P, cfg in variants:
                        inst = dict(cell)
                        inst["P"] = P
                        edges = decode(inst, P, cfg)
                        _df, summ = evaluate_edges(inst, edges)
                        rows.append({
                            "bridge": bridge, "seed": seed, "ladder": ladder, "level": level,
                            "method": name,
                            "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                            "edge_precision": summ["edge_precision"], "edge_recall": summ["edge_recall"],
                            "edge_f1": summ["edge_f1"], "split_edge_f1": summ["split_edge_f1"],
                            "merge_edge_f1": summ["merge_edge_f1"],
                            "fp_per_template": summ["edge_fp_total"] / max(summ["n_templates"], 1),
                            "pred_edges_per_template": summ["n_pred_edges"],
                            "coverage": summ["coverage"],
                        })
                    inst = dict(cell)
                    edges = decode_threshold_mm(inst, THRESHOLD_MM_CUTOFF)
                    _df, summ = evaluate_edges(inst, edges)
                    rows.append({
                        "bridge": bridge, "seed": seed, "ladder": ladder, "level": level,
                        "method": "Threshold-MM frozen",
                        "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                        "edge_precision": summ["edge_precision"], "edge_recall": summ["edge_recall"],
                        "edge_f1": summ["edge_f1"], "split_edge_f1": summ["split_edge_f1"],
                        "merge_edge_f1": summ["merge_edge_f1"],
                        "fp_per_template": summ["edge_fp_total"] / max(summ["n_templates"], 1),
                        "pred_edges_per_template": summ["n_pred_edges"],
                        "coverage": summ["coverage"],
                    })
                print(f"[stress-reg] {ladder}={level} {bridge} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stress_decoder_comparison.csv", index=False)

    agg_rows: list[dict[str, Any]] = []
    for (ladder, level, method), g in df.groupby(["ladder", "level", "method"]):
        row = {"ladder": ladder, "level": level, "method": method,
               "n_cells": int(len(g))}
        for col in ("split_exact", "merge_exact", "edge_precision", "edge_recall", "edge_f1",
                    "split_edge_f1", "merge_edge_f1", "fp_per_template",
                    "pred_edges_per_template", "coverage"):
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            row[f"{col}_mean"] = float(v.mean())
            row[f"{col}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            lo, hi = bootstrap_ci(v, seed=7)
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "stress_decoder_comparison_aggregated.csv", index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
