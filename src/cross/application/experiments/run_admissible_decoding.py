"""Admissible decoding / temporal filtering on fixed-delay UOT transport (no re-solve)."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup, delay_distribution
from cross.application.experiments.uot_cache_utils import load_flow_segments
from cross.domain.evaluation.flow_metrics import (
    flow_mass_recall,
    pair_precision_recall_f1,
    topk_flow_accuracy,
)
from cross.domain.uot.delay_policy import flow_pair_delay_sec
from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs
from cross.utils.safe_cast import safe_float

STRATEGIES: tuple[str, ...] = (
    "raw_argmax_fixed_delay",
    "flow_time_admissible_filter",
    "tx_time_admissible_filter",
    "joint_time_admissible_filter",
    "positive_delay_top3_rescue",
    "positive_delay_top5_rescue",
    "positive_delay_top10_rescue",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "method",
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "top1_recall",
    "top3_recall",
    "flow_level_recall",
    "flow_pair_cvr",
    "tx_level_cvr",
    "coverage",
    "abstention_rate",
    "n_predicted_pairs",
    "n_abstained",
    "median_flow_delay_sec",
    "median_tx_delay_sec",
    "p05_flow_delay_sec",
    "p05_tx_delay_sec",
    "flow_mass_recall",
)


def _load_transport(run_dir: Path) -> np.ndarray:
    for rel in ("matching_transport_matrix.npz", "uot/uot_transport_matrix.npz"):
        p = run_dir / rel
        if p.is_file():
            return np.asarray(np.load(p)["P"], dtype=float)
    raise FileNotFoundError(f"No transport matrix under {run_dir}")


def _truth_from_labels(label_df: pd.DataFrame) -> dict[str, str]:
    truth: dict[str, str] = {}
    for _, row in label_df.iterrows():
        s = norm_addr(row.get("srcTxhash", row.get("srcTxHash", "")))
        d = norm_addr(row.get("dstTxhash", row.get("dstTxHash", "")))
        if s:
            truth[s] = d
    return truth


def _tx_to_flow_index(source_flows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            out[norm_addr(str(txh))] = i
    return out


def _tx_to_target_flows(target_flows: list[dict[str, Any]]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for j, tf in enumerate(target_flows):
        for txh in tf.get("tx_hashes") or []:
            h = norm_addr(str(txh))
            out.setdefault(h, set()).add(j)
    return out


def _flow_admissible(
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    i: int,
    j: int,
) -> bool:
    if i < 0 or j < 0 or i >= len(eth_flows) or j >= len(bnb_flows):
        return False
    d = flow_pair_delay_sec(eth_flows[i], bnb_flows[j], policy="tx_if_available_else_flow_representative")
    return float(d) >= 0.0


def _tx_admissible(src_tx: str, dst_tx: str, eth_ts: dict[str, float], bnb_ts: dict[str, float]) -> bool:
    ts_s = eth_ts.get(norm_addr(src_tx))
    ts_d = bnb_ts.get(norm_addr(dst_tx))
    if ts_s is None or ts_d is None:
        return False
    return float(ts_d - ts_s) >= 0.0


def _strategy_k(strategy: str) -> int | None:
    if strategy == "positive_delay_top3_rescue":
        return 3
    if strategy == "positive_delay_top5_rescue":
        return 5
    if strategy == "positive_delay_top10_rescue":
        return 10
    return None


def _decode_src_tx(
    *,
    strategy: str,
    i: int,
    src_tx: str,
    src_ts: float,
    src_amt: float,
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    dst_norm: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> tuple[int, str | None, bool]:
    """Return (target_flow_j, dst_tx or None if abstain, abstained)."""
    row = p[i]
    order = np.argsort(-row).astype(int)
    if order.size == 0:
        return -1, None, True

    rescue_k = _strategy_k(strategy)

    def pick_tx(j: int) -> str:
        dst, _, _ = pick_dst_tx_in_flow(src_tx, src_ts, src_amt, bnb_flows[j], dst_norm, policy="legacy")
        return dst

    if strategy in ("raw_argmax_fixed_delay", "flow_time_admissible_filter", "tx_time_admissible_filter", "joint_time_admissible_filter"):
        j = int(order[0])
        dst = pick_tx(j)
        if strategy == "raw_argmax_fixed_delay":
            return j, dst or None, not bool(dst)

        if strategy == "flow_time_admissible_filter":
            if not _flow_admissible(eth_flows, bnb_flows, i, j):
                return j, None, True
            return j, dst or None, not bool(dst)

        if strategy == "tx_time_admissible_filter":
            if not dst or not _tx_admissible(src_tx, dst, eth_ts, bnb_ts):
                return j, None, True
            return j, dst, False

        # joint
        if not _flow_admissible(eth_flows, bnb_flows, i, j):
            return j, None, True
        if not dst or not _tx_admissible(src_tx, dst, eth_ts, bnb_ts):
            return j, None, True
        return j, dst, False

    if rescue_k is not None:
        for j in order[:rescue_k]:
            j = int(j)
            if _flow_admissible(eth_flows, bnb_flows, i, j):
                dst = pick_tx(j)
                if dst:
                    return j, dst, False
        return int(order[0]), None, True

    raise ValueError(f"Unknown strategy: {strategy}")


def _build_predictions(
    *,
    strategy: str,
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    truth: dict[str, str],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    external_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    """Per labeled src tx: dst hash or None (abstain)."""
    tx_to_i = _tx_to_flow_index(eth_flows)
    mapping: dict[str, str | None] = {}
    meta: dict[str, dict[str, Any]] = {}

    for src_tx in truth:
        if external_mapping is not None:
            mapping[src_tx] = external_mapping.get(src_tx)
            meta[src_tx] = {"abstained": not mapping[src_tx], "external": True}
            continue

        i = tx_to_i.get(src_tx, -1)
        if i < 0:
            mapping[src_tx] = None
            meta[src_tx] = {"abstained": True, "flow_i": -1, "flow_j": -1}
            continue

        row = src_all[src_all["txhash"].astype(str).map(norm_addr) == src_tx]
        if row.empty:
            s_ts, s_amt = eth_ts.get(src_tx, 0.0), 0.0
        else:
            r = row.iloc[0]
            s_ts = safe_float(r.get("timestamp"), eth_ts.get(src_tx, 0.0))
            s_amt = safe_float(r.get("args.amount"), 0.0)

        j, dst, abstained = _decode_src_tx(
            strategy=strategy,
            i=i,
            src_tx=src_tx,
            src_ts=float(s_ts),
            src_amt=float(s_amt),
            p=p,
            eth_flows=eth_flows,
            bnb_flows=bnb_flows,
            dst_norm=dst_norm,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
        )
        mapping[src_tx] = None if abstained or not dst else dst
        meta[src_tx] = {"abstained": abstained or not dst, "flow_i": i, "flow_j": j}

    return mapping, meta


def _evaluate_strategy(
    *,
    method: str,
    mapping: dict[str, str | None],
    meta: dict[str, dict[str, Any]],
    truth: dict[str, str],
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    tx_to_j: dict[str, set[int]],
) -> dict[str, Any]:
    pairs = [{"srcTxHash": s, "dstTxHash": d} for s, d in mapping.items() if d]
    pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["srcTxHash", "dstTxHash"])

    pr = pair_precision_recall_f1(pairs_df, label_df)
    top3 = topk_flow_accuracy(p, eth_flows, bnb_flows, label_df, k=3)
    fm = flow_mass_recall(p, eth_flows, bnb_flows, label_df)

    n_labeled = len(truth)
    n_abstained = sum(1 for s in truth if not mapping.get(s))
    n_predicted = n_labeled - n_abstained

    flow_hits = 0
    flow_eval = 0
    flow_delays: list[float] = []
    tx_delays: list[float] = []
    flow_viol = tx_viol = 0

    covered_flows: set[int] = set()
    for s, gt_d in truth.items():
        m = meta.get(s) or {}
        if m.get("abstained") or not mapping.get(s):
            continue
        i = int(m.get("flow_i", -1))
        j = int(m.get("flow_j", -1))
        pred_d = mapping[s]
        if i >= 0:
            covered_flows.add(i)

        if i >= 0 and j >= 0:
            fd = flow_pair_delay_sec(eth_flows[i], bnb_flows[j], policy="tx_if_available_else_flow_representative")
            flow_delays.append(fd)
            if fd < 0:
                flow_viol += 1

        ts_s = eth_ts.get(s)
        ts_d = bnb_ts.get(pred_d) if pred_d else None
        if ts_s is not None and ts_d is not None:
            td = float(ts_d - ts_s)
            tx_delays.append(td)
            if td < 0:
                tx_viol += 1

        gt_j = tx_to_j.get(gt_d, set())
        if j >= 0 and gt_j:
            flow_eval += 1
            if j in gt_j:
                flow_hits += 1

    flow_dist = delay_distribution(flow_delays)
    tx_dist = delay_distribution(tx_delays)
    n_flow_pairs = len(flow_delays)
    n_tx_pairs = len(tx_delays)

    coverage = float(len(covered_flows) / max(len(eth_flows), 1))
    abstention_rate = float(n_abstained / max(n_labeled, 1))
    n_gt = n_labeled
    tp = int(pr.get("tp") or 0)
    fp = int(pr.get("fp") or 0)

    return {
        "method": method,
        "pair_precision": pr.get("pair_precision"),
        "pair_recall": pr.get("pair_recall"),
        "pair_f1": pr.get("pair_f1"),
        "top1_recall": pr.get("pair_recall"),
        "top3_recall": top3,
        "flow_level_recall": float(flow_hits / max(flow_eval, 1)) if flow_eval else 0.0,
        "flow_pair_cvr": float(flow_viol / max(n_flow_pairs, 1)) if n_flow_pairs else 0.0,
        "tx_level_cvr": float(tx_viol / max(n_tx_pairs, 1)) if n_tx_pairs else 0.0,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "n_predicted_pairs": n_predicted,
        "n_abstained": n_abstained,
        "n_true_positive": tp,
        "n_false_positive": fp,
        "n_ground_truth_pairs": n_gt,
        "n_unrecovered_gt": n_gt - tp,
        "median_flow_delay_sec": flow_dist.get("median"),
        "median_tx_delay_sec": tx_dist.get("median"),
        "flow_mass_recall": fm,
    }


def _round3(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paper_branch(rows: list[dict[str, Any]]) -> str:
    raw = next((r for r in rows if r["method"] == "raw_argmax_fixed_delay"), None)
    flow_f = next((r for r in rows if r["method"] == "flow_time_admissible_filter"), None)
    rescue3 = next((r for r in rows if r["method"] == "positive_delay_top3_rescue"), None)

    if raw and flow_f:
        cvr_drop = float(raw.get("tx_level_cvr") or 1) - float(flow_f.get("tx_level_cvr") or 0)
        prec_ok = float(flow_f.get("pair_precision") or 0) >= float(raw.get("pair_precision") or 0) * 0.95
        recall_drop = float(raw.get("pair_recall") or 0) - float(flow_f.get("pair_recall") or 0)
        if cvr_drop > 0.05 and prec_ok and recall_drop > 0.01:
            return "case_a_admissible_gate"

    if rescue3 and raw:
        if float(rescue3.get("tx_level_cvr") or 1) < float(raw.get("tx_level_cvr") or 1) * 0.85:
            if float(rescue3.get("pair_recall") or 0) >= float(raw.get("pair_recall") or 0) * 0.9:
                return "case_b_topk_rescue"

    best_f1 = max((float(r.get("pair_f1") or 0) for r in rows if r["method"] in STRATEGIES), default=0.0)
    if best_f1 < 0.15:
        return "case_c_f1_collapse"
    return "case_a_admissible_gate"


def run_admissible_decoding(
    *,
    run_dir: Path,
    base_run: Path,
    label_path: Path,
    eth_path: Path,
    bnb_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    base_run = Path(base_run)

    p = _load_transport(run_dir)
    flow_root = run_dir / "uot" if (run_dir / "uot" / "uot_flow_segments_eth.csv").is_file() else base_run / "uot"
    eth_flows = load_flow_segments(flow_root / "uot_flow_segments_eth.csv")
    bnb_flows = load_flow_segments(flow_root / "uot_flow_segments_bnb.csv")
    label_df = pd.read_csv(label_path)
    truth = _truth_from_labels(label_df)

    eth_df = pd.read_csv(eth_path, low_memory=False)
    bnb_df = pd.read_csv(bnb_path, low_memory=False)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
    tx_to_j: dict[str, set[int]] = {}
    for j, tf in enumerate(bnb_flows):
        for txh in tf.get("tx_hashes") or []:
            h = norm_addr(str(txh))
            tx_to_j.setdefault(h, set()).add(j)

    flow_txs = {norm_addr(str(h)) for f in eth_flows for h in (f.get("tx_hashes") or [])}
    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs:
        src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        mapping, meta = _build_predictions(
            strategy=strategy,
            p=p,
            eth_flows=eth_flows,
            bnb_flows=bnb_flows,
            src_all=src_all,
            dst_norm=dst_norm,
            truth=truth,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
        )
        rows.append(
            _evaluate_strategy(
                method=strategy,
                mapping=mapping,
                meta=meta,
                truth=truth,
                p=p,
                eth_flows=eth_flows,
                bnb_flows=bnb_flows,
                label_df=label_df,
                eth_ts=eth_ts,
                bnb_ts=bnb_ts,
                tx_to_j=tx_to_j,
            )
        )

    # Controls: same decode as raw, permuted / random labels
    raw_mapping, raw_meta = _build_predictions(
        strategy="raw_argmax_fixed_delay",
        p=p,
        eth_flows=eth_flows,
        bnb_flows=bnb_flows,
        src_all=src_all,
        dst_norm=dst_norm,
        truth=truth,
        eth_ts=eth_ts,
        bnb_ts=bnb_ts,
    )
    perm_label = label_df.copy()
    dst_col = next(c for c in perm_label.columns if str(c).lower() in ("dsttxhash", "dst_tx_hash"))
    vals = perm_label[dst_col].tolist()
    rng = random.Random(7)
    rng.shuffle(vals)
    perm_label[dst_col] = vals
    perm_truth = _truth_from_labels(perm_label)
    rows.append(
        _evaluate_strategy(
            method="permuted_gt_control",
            mapping=raw_mapping,
            meta=raw_meta,
            truth=perm_truth,
            p=p,
            eth_flows=eth_flows,
            bnb_flows=bnb_flows,
            label_df=perm_label,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            tx_to_j=tx_to_j,
        )
    )

    branch = _paper_branch(rows)
    raw_row = next(r for r in rows if r["method"] == "raw_argmax_fixed_delay")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "base_run": str(base_run.resolve()),
        "n_source_flows": len(eth_flows),
        "n_ground_truth_pairs": len(truth),
        "paper_branch": branch,
        "raw_argmax_fixed_delay": raw_row,
        "strategies": {r["method"]: r for r in rows},
        "evaluation_object": "flow_correspondence_ranking_with_admissible_tx_projection",
        "notes": [
            "Transport plan unchanged; no UOT re-solve.",
            "top3_recall reflects transport ranking, not admissibility filter.",
            "CVR computed only on non-abstained predicted pairs.",
            "tx-level pair metrics are compatibility projection, not primary forensic object.",
        ],
    }

    (out_dir / "admissible_decoding_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from cross.application.experiments.admissible_decoding_paper_outputs import (
        refresh_admissible_decoding_paper_outputs,
    )

    refresh_result = refresh_admissible_decoding_paper_outputs(out_dir, n_source_flows=len(eth_flows))
    return refresh_result["summary"]


def main() -> int:
    p = argparse.ArgumentParser(description="Admissible decoding on fixed-delay UOT transport.")
    p.add_argument("--run-dir", type=Path, default=Path("out/uot_delay_fixed_production"))
    p.add_argument("--base-run", type=Path, default=Path("out/leave_anchor_out_real"))
    p.add_argument("--label", type=Path, required=True)
    p.add_argument("--eth", type=Path, required=True)
    p.add_argument("--bnb", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("out/admissible_decoding"))
    args = p.parse_args()
    result = run_admissible_decoding(
        run_dir=args.run_dir,
        base_run=args.base_run,
        label_path=args.label,
        eth_path=args.eth,
        bnb_path=args.bnb,
        out_dir=args.out,
    )
    print(json.dumps({"paper_branch": result["paper_branch"], "raw": result["raw_argmax_fixed_delay"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
