from __future__ import annotations

from pathlib import Path

import pandas as pd

from cross.domain.analytics.aggregation import aggregate_relations
from cross.domain.evaluation.compare import compare_to_label
from cross.shared.label_availability import label_file_usable
from cross.shared.load_csv import load_label_csv
from cross.domain.path_a.pairing import run_path_a


def path_a_vs_label_cmp(pairs_df: pd.DataFrame, label_path: Path) -> dict:
    if not label_file_usable(label_path):
        return {
            "accuracy": None,
            "hits": None,
            "total_src_rows": None,
            "eval_numerator": None,
            "eval_denominator": None,
            "no_ground_truth": True,
            "mismatch_sample": [],
            "mismatch_count": 0,
            "metric_interpretation": (
                "No usable label CSV (missing or empty); Path A vs label accuracy is undefined."
            ),
            "label_file": str(label_path.resolve()),
        }
    labels_df = load_label_csv(label_path)
    cmp = compare_to_label(pairs_df, labels_df)
    cmp["no_ground_truth"] = False
    cmp["metric_interpretation"] = (
        "Path A pairs.csv vs label file: denominator = rows whose srcTxHash appears in labels; "
        "numerator = matching dstTxHash (same rule as Path B path_b_vs_label.json)."
    )
    cmp["label_file"] = str(label_path.resolve())
    return cmp


def run_path_a_and_aggregate(
    label_path: Path,
    eth_path: Path,
    bnb_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, pd.DataFrame]:
    pairs_df, stats_a = run_path_a(label_path, eth_path, bnb_df)
    summary, ambiguous, edges = aggregate_relations(pairs_df)
    return pairs_df, stats_a, summary, ambiguous, edges
