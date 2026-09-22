"""Recompute top-k recall from saved UOT transport artifacts (no Path B re-run)."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.domain.evaluation.flow_metrics import topk_flow_accuracy
from cross.shared.normalize import norm_addr

PRIMARY_EXPERIMENTS: frozenset[str] = frozenset(
    {
        "baseline_original",
        "leave_key_out",
        "leave_anchor_out_strict",
    }
)


def parse_flow_tx_hashes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if text.startswith("["):
        import ast

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (SyntaxError, ValueError):
            pass
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def load_flow_segments(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    flows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        item["tx_hashes"] = parse_flow_tx_hashes(item.get("tx_hashes"))
        flows.append(item)
    return flows


def _truth_from_label_df(label_df: pd.DataFrame) -> dict[str, str]:
    truth: dict[str, str] = {}
    for _, row in label_df.iterrows():
        s = norm_addr(row.get("srcTxhash", row.get("srcTxHash", "")))
        d = norm_addr(row.get("dstTxhash", row.get("dstTxHash", "")))
        if s:
            truth[s] = d
    return truth


def _load_transport_matrix(out_dir: Path) -> np.ndarray | None:
    for rel in ("uot/uot_transport_matrix.npz", "uot/matching_transport_matrix.npz"):
        path = out_dir / rel
        if not path.is_file():
            continue
        data = np.load(path)
        for key in ("P", "transport", "transport_matrix"):
            if key in data:
                return np.asarray(data[key], dtype=float)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=float)
    return None


def recompute_topk_recall_from_artifacts(
    out_dir: Path,
    label_df: pd.DataFrame,
    *,
    k: int = 3,
    permuted_seed: int = 7,
    random_seed: int = 42,
) -> dict[str, float | None]:
    """Return recomputed top-k recall for paper-facing rows when transport artifacts exist."""
    out_dir = Path(out_dir)
    p = _load_transport_matrix(out_dir)
    eth_path = out_dir / "uot/uot_flow_segments_eth.csv"
    bnb_path = out_dir / "uot/uot_flow_segments_bnb.csv"
    if p is None or not eth_path.is_file() or not bnb_path.is_file() or label_df.empty:
        return {
            "primary_top3_recall": None,
            "negative_control_permuted_gt_top3_recall": None,
            "random_or_uniform_baseline_top3_recall": None,
        }

    eth_flows = load_flow_segments(eth_path)
    bnb_flows = load_flow_segments(bnb_path)
    if p.shape != (len(eth_flows), len(bnb_flows)):
        return {
            "primary_top3_recall": None,
            "negative_control_permuted_gt_top3_recall": None,
            "random_or_uniform_baseline_top3_recall": None,
        }

    primary = float(topk_flow_accuracy(p, eth_flows, bnb_flows, label_df, k=k))

    perm = label_df.copy()
    dst_col = next(c for c in perm.columns if str(c).lower() in ("dsttxhash", "dst_tx_hash"))
    vals = perm[dst_col].tolist()
    rng = random.Random(permuted_seed)
    rng.shuffle(vals)
    perm[dst_col] = vals
    permuted = float(topk_flow_accuracy(p, eth_flows, bnb_flows, perm, k=k))

    truth = _truth_from_label_df(label_df)
    tx_to_i: dict[str, int] = {}
    for i, sf in enumerate(eth_flows):
        for txh in sf.get("tx_hashes") or []:
            tx_to_i[norm_addr(str(txh))] = i

    p_rand = np.zeros_like(p)
    rng2 = random.Random(random_seed)
    for s in truth:
        i = tx_to_i.get(s, -1)
        if i < 0 or i >= p.shape[0]:
            continue
        j = rng2.randrange(p.shape[1])
        p_rand[i, :] = 0.0
        p_rand[i, j] = 1.0
    random_top3 = float(topk_flow_accuracy(p_rand, eth_flows, bnb_flows, label_df, k=k))

    return {
        "primary_top3_recall": primary,
        "negative_control_permuted_gt_top3_recall": permuted,
        "random_or_uniform_baseline_top3_recall": random_top3,
    }


def apply_recomputed_top3_to_rows(
    rows: list[dict[str, Any]],
    recompute: dict[str, float | None],
) -> None:
    primary = recompute.get("primary_top3_recall")
    if primary is not None:
        for name in PRIMARY_EXPERIMENTS:
            for row in rows:
                if row.get("experiment_name") == name:
                    row["top3_recall"] = primary
                    row["top3_flow_correspondence_accuracy"] = primary

    permuted = recompute.get("negative_control_permuted_gt_top3_recall")
    if permuted is not None:
        for row in rows:
            if row.get("experiment_name") == "negative_control_permuted_gt":
                row["top3_recall"] = permuted
                row["top3_flow_correspondence_accuracy"] = permuted

    random_top3 = recompute.get("random_or_uniform_baseline_top3_recall")
    if random_top3 is not None:
        for row in rows:
            if row.get("experiment_name") == "random_or_uniform_baseline":
                row["top3_recall"] = random_top3
                row["top3_flow_correspondence_accuracy"] = random_top3
