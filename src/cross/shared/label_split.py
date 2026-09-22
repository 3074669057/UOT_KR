"""Temporal train/test split of label CSV rows for leakage-safe Path B evaluation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cross.shared.load_csv import load_label_csv
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float
from cross.domain.path_a.pairing import eth_row_for_src


def split_labels_by_src_timestamp(
    labels: pd.DataFrame,
    eth_df: pd.DataFrame,
    *,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    rows: list[tuple[float, pd.Series]] = []
    for _, lab in labels.iterrows():
        src_h = norm_addr(lab.get("srcTxhash", ""))
        er = eth_row_for_src(eth_df, src_h)
        ts = safe_float(er.get("timeStamp"), 0.0) if er is not None else 0.0
        rows.append((ts, lab))
    rows.sort(key=lambda x: x[0])
    n = len(rows)
    if n < 2:
        raise ValueError("need at least 2 labeled pairs to split train/test")
    k = int(round(n * train_fraction))
    k = max(1, min(k, n - 1))
    train_df = pd.DataFrame([r[1] for r in rows[:k]]).reset_index(drop=True)
    test_df = pd.DataFrame([r[1] for r in rows[k:]]).reset_index(drop=True)
    return train_df, test_df


def load_split_from_paths(
    label_path: Path,
    eth_path: Path,
    *,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = load_label_csv(label_path)
    from cross.shared.load_csv import load_eth_cun

    eth_df = load_eth_cun(eth_path)
    return split_labels_by_src_timestamp(labels, eth_df, train_fraction=train_fraction)
