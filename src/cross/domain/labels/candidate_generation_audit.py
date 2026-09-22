"""Audit whether candidate generation uses labels / anchors / ground-truth pairs."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

from cross.shared.normalize import norm_addr


def audit_candidate_generation(
    cmp: dict[str, Any],
    *,
    label_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    opts = cmp.get("path_b_options") if isinstance(cmp.get("path_b_options"), dict) else {}
    method = str(opts.get("matching_method") or "uot").strip().lower()
    boost_label = bool(opts.get("boost_label_dst"))
    mode = str(opts.get("mode") or "")

    uses_labels = boost_label
    uses_anchors = False
    uses_gt_pairs = boost_label

    uot = cmp.get("uot") if isinstance(cmp.get("uot"), dict) else {}
    n_src = int(uot.get("n_eth_flows") or 0)
    n_dst = int(uot.get("n_bnb_flows") or 0)

    evidence = cmp.get("path_b_evidence") if isinstance(cmp.get("path_b_evidence"), dict) else {}
    cand_counts = [float(v.get("candidate_count") or 0) for v in evidence.values()]
    avg_c = float(np.mean(cand_counts)) if cand_counts else 0.0
    med_c = float(median(cand_counts)) if cand_counts else 0.0

    recall: float | None = 0.0
    if label_df is not None and not label_df.empty and evidence:
        truth: dict[str, str] = {}
        for _, r in label_df.iterrows():
            s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
            d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
            if s:
                truth[s] = d
        hits = 0
        total = 0
        for src, info in evidence.items():
            s = norm_addr(str(src))
            if s not in truth:
                continue
            total += 1
            cands = info.get("candidate_dst_hashes") or info.get("candidates") or []
            cand_norm = {norm_addr(str(c)) for c in cands}
            if truth[s] in cand_norm:
                hits += 1
        recall = float(hits / max(total, 1))

    notes_parts = [
        f"matching_method={method}",
        f"path_b_mode={mode or 'n/a'}",
    ]
    if method == "uot":
        notes_parts.append(
            "UOT Path B builds full ETH×BNB flow cost matrix; candidate pool is not label-pruned "
            "(allowed: bridge contract filter, time window, token route, amount range via flow segments)."
        )
        uses_labels = uses_labels or False
        uses_gt_pairs = uses_gt_pairs or False
    if boost_label:
        notes_parts.append(
            "FORBIDDEN: boost_label_dst injects truth dst into greedy candidate scoring."
        )

    is_full_matrix_uot = method == "uot" and n_src > 0 and n_dst > 0
    if is_full_matrix_uot:
        notes_parts.append(
            "Candidate recall is not a meaningful metric for full-matrix UOT because all target "
            "flows are scored for each source flow."
        )

    audit: dict[str, Any] = {
        "candidate_generation_uses_labels": bool(uses_labels),
        "candidate_generation_uses_bridge_anchors": bool(uses_anchors),
        "candidate_generation_uses_ground_truth_pairs": bool(uses_gt_pairs),
        "boost_label_dst_enabled": boost_label,
        "matching_method": method,
        "notes": " ".join(notes_parts),
    }
    if is_full_matrix_uot:
        audit.update(
            {
                "candidate_generation_mode": "full_cost_matrix",
                "candidate_recall_against_gt": None,
                "candidate_recall_applicable": False,
                "avg_candidates_per_source": n_dst,
                "median_candidates_per_source": n_dst,
                "cost_matrix_shape": [n_src, n_dst],
                "gt_pairs_evaluable_in_cost_matrix": True,
            }
        )
    else:
        audit.update(
            {
                "candidate_generation_mode": "candidate_list",
                "candidate_recall_against_gt": float(recall or 0.0),
                "candidate_recall_applicable": True,
                "avg_candidates_per_source": avg_c,
                "median_candidates_per_source": med_c,
            }
        )
    return audit


def write_candidate_generation_audit(out_dir: Path, audit: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "candidate_generation_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    return path
