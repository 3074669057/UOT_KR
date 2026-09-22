"""Pairing accuracy vs label CSV."""
from __future__ import annotations

import pandas as pd

from cross.shared.normalize import norm_addr


def compare_to_label(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> dict:
    truth: dict[str, str] = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        truth[s] = d

    hits = 0
    total = 0
    mismatches = []
    if pred_df is None or pred_df.empty or "srcTxHash" not in pred_df.columns:
        return {
            "accuracy": 0.0,
            "hits": 0,
            "total_src_rows": 0,
            "eval_numerator": 0,
            "eval_denominator": 0,
            "mismatch_sample": [],
            "mismatch_count": 0,
        }

    pred_df = pred_df.copy()
    pred_df["srcTxHash"] = pred_df["srcTxHash"].map(norm_addr)
    pred_df["dstTxHash"] = pred_df["dstTxHash"].astype(str).map(norm_addr)
    for _, r in pred_df.iterrows():
        s = r["srcTxHash"]
        if not s or s == "nan" or s not in truth:
            continue
        total += 1
        exp = truth[s]
        got = norm_addr(str(r.get("dstTxHash", "") or "")) if str(r.get("dstTxHash", "") or "") else ""
        if exp == got:
            hits += 1
        else:
            mismatches.append({"srcTxHash": s, "expected": exp, "got": got})
    acc = hits / total if total else 0.0
    return {
        "accuracy": acc,
        "hits": hits,
        "total_src_rows": total,
        "eval_numerator": hits,
        "eval_denominator": total,
        "mismatch_sample": mismatches[:50],
        "mismatch_count": len(mismatches),
    }
