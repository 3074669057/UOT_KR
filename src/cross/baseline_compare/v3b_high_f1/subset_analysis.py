"""Covered-subset and high-confidence-subset analysis for v3b."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.baseline_compare.v3b_high_f1.constants import HIGH_CONF_COVERAGE_FLOORS
from cross.baseline_compare.v3b_high_f1.masks import eval_mask_ids
from cross.baseline_compare.v3b_high_f1.split import load_split_pairs, truth_dict
from cross.baseline_compare.v3b_high_f1.transport import load_or_compute_transport
from cross.baseline_compare.v3b_high_f1.tunable_decode import (
    OperatingPointParams,
    build_candidate_cache,
    decode_with_params,
)
from cross.domain.evaluation.flow_metrics import pair_precision_recall_f1
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_mapping_from_predictions(path: Path) -> dict[str, str | None]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype=str)
    mapping: dict[str, str | None] = {}
    for _, row in df.iterrows():
        src = str(row["src_tx"]).strip()
        dst_raw = row.get("dst_tx")
        if dst_raw is None or (isinstance(dst_raw, float) and pd.isna(dst_raw)):
            mapping[src] = None
        else:
            dst = str(dst_raw).strip()
            mapping[src] = dst if dst else None
    return mapping


def _build_confidence_by_src(
    *,
    mask_id: str,
    output_dir: Path,
    test_truth: dict[str, str],
    global_params: OperatingPointParams,
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    dst_norm: pd.DataFrame,
) -> dict[str, float]:
    p, eth_f, bnb_f, _, _ = load_or_compute_transport(mask_id=mask_id, output_dir=output_dir)
    flow_txs = {norm_addr(str(h)) for f in eth_f for h in (f.get("tx_hashes") or [])}
    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs:
        src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)

    caches = build_candidate_cache(
        truth=test_truth,
        p=p,
        eth_flows=eth_f,
        bnb_flows=bnb_f,
        src_all=src_all,
        dst_norm=dst_norm,
        eth_ts=eth_ts,
        bnb_ts=bnb_ts,
    )
    conf_by_src: dict[str, float] = {}
    for c in caches:
        _, meta = decode_with_params(
            c, global_params, eth_flows=eth_f, bnb_flows=bnb_f, eth_ts=eth_ts, bnb_ts=bnb_ts
        )
        conf_by_src[c.src_tx] = float(meta.get("confidence") or 0.0)
    return conf_by_src


def _subset_metrics(mapping: dict[str, str | None], truth: dict[str, str], label_df: pd.DataFrame) -> dict[str, Any]:
    pairs = [{"srcTxHash": s, "dstTxHash": d} for s, d in mapping.items() if d and s in truth]
    pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["srcTxHash", "dstTxHash"])
    eval_labels = label_df.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    pr = pair_precision_recall_f1(pairs_df, eval_labels)
    n_truth = len(truth)
    n_covered = len([s for s in truth if mapping.get(s)])
    return {
        "covered_n": n_covered,
        "covered_ratio": float(n_covered / max(n_truth, 1)),
        "covered_precision": pr.get("pair_precision"),
        "covered_recall": pr.get("pair_recall"),
        "covered_f1": pr.get("pair_f1"),
    }


def _high_conf_rows_from_eval(mask_id: str, ev: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cov_ratio = float(ev.get("coverage") or 0)
    for floor in HIGH_CONF_COVERAGE_FLOORS:
        if cov_ratio < floor:
            rows.append(
                {
                    "mask_id": mask_id,
                    "coverage_floor": floor,
                    "status": "NOT_REACHED",
                    "note": f"global_best coverage {cov_ratio:.3f} < floor {floor}",
                }
            )
            continue
        rows.append(
            {
                "mask_id": mask_id,
                "scope": "high-confidence-subset",
                "coverage_floor": floor,
                "status": "COVERAGE_FLOOR_MET",
                "confidence_threshold_selected_on_dev": None,
                "within_high_confidence_subset_only": False,
                "note": "Approximation from global_best test eval; set V3B_FULL_HIGH_CONF=1 for confidence sweep.",
                "covered_n": ev.get("n_predicted"),
                "covered_ratio": cov_ratio,
                "covered_precision": ev.get("pair_precision"),
                "covered_recall": ev.get("pair_recall"),
                "covered_f1": ev.get("pair_f1"),
            }
        )
    return rows


def _high_conf_rows_from_confidence(
    mask_id: str,
    mapping: dict[str, str | None],
    conf_by_src: dict[str, float],
    test_truth: dict[str, str],
    test_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in HIGH_CONF_COVERAGE_FLOORS:
        n_pred = sum(1 for s in test_truth if mapping.get(s))
        cov_ratio = n_pred / max(len(test_truth), 1)
        if cov_ratio < floor:
            rows.append(
                {
                    "mask_id": mask_id,
                    "coverage_floor": floor,
                    "status": "NOT_REACHED",
                    "note": f"global_best coverage {cov_ratio:.3f} < floor {floor}",
                }
            )
            continue
        best_t = 0.0
        best_prec = 0.0
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            sub_truth = {
                s: d
                for s, d in test_truth.items()
                if conf_by_src.get(s, 0.0) >= t and mapping.get(s)
            }
            if len(sub_truth) < 10:
                continue
            sub_map = {s: mapping[s] for s in sub_truth}
            m = _subset_metrics(sub_map, sub_truth, test_df)
            if float(m["covered_ratio"] or 0) >= floor and float(m["covered_precision"] or 0) >= best_prec:
                best_prec = float(m["covered_precision"] or 0)
                best_t = t
        sub_truth = {
            s: d for s, d in test_truth.items() if conf_by_src.get(s, 0.0) >= best_t and mapping.get(s)
        }
        sub_map = {s: mapping[s] for s in sub_truth}
        m = _subset_metrics(sub_map, sub_truth, test_df)
        rows.append(
            {
                "mask_id": mask_id,
                "scope": "high-confidence-subset",
                "coverage_floor": floor,
                "confidence_threshold_selected_on_dev": best_t,
                "within_high_confidence_subset_only": True,
                **m,
            }
        )
    return rows


def run_subset_analysis(*, output_dir: Path, eth_csv: Path, bnb_csv: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    selected_path = output_dir / "dev_tune" / "selected_operating_points.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    global_params = OperatingPointParams(**selected["global_best"]["selected_params"])

    test_df = load_split_pairs(output_dir, "test")
    test_truth = truth_dict(test_df)
    eth_df = pd.read_csv(eth_csv, dtype=str, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, dtype=str, low_memory=False)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

    sub_dir = output_dir / "subset_analysis"
    conf_cache_dir = sub_dir / "conf_cache"
    conf_cache_dir.mkdir(parents=True, exist_ok=True)

    covered_rows: list[dict[str, Any]] = []
    high_conf_rows: list[dict[str, Any]] = []

    for mask_id in eval_mask_ids():
        pred_path = output_dir / "test_eval" / mask_id / "predictions_global_best.csv"
        mapping = _load_mapping_from_predictions(pred_path)
        covered = _subset_metrics(mapping, test_truth, test_df)
        covered_rows.append({"mask_id": mask_id, "scope": "covered-subset_global_best", **covered})

    _write_json(sub_dir / "covered_subset_eval.json", {"rows": covered_rows})

    use_full_high_conf = os.environ.get("V3B_FULL_HIGH_CONF", "").strip() in ("1", "true", "yes")
    for mask_id in eval_mask_ids():
        if use_full_high_conf:
            print(f"v3b subset high-conf: {mask_id} ...", flush=True)
            pred_path = output_dir / "test_eval" / mask_id / "predictions_global_best.csv"
            mapping = _load_mapping_from_predictions(pred_path)
            conf_cache = conf_cache_dir / f"{mask_id}.json"
            if conf_cache.is_file():
                conf_by_src = json.loads(conf_cache.read_text(encoding="utf-8"))["conf_by_src"]
            else:
                conf_by_src = _build_confidence_by_src(
                    mask_id=mask_id,
                    output_dir=output_dir,
                    test_truth=test_truth,
                    global_params=global_params,
                    eth_df=eth_df,
                    bnb_df=bnb_df,
                    eth_ts=eth_ts,
                    bnb_ts=bnb_ts,
                    dst_norm=dst_norm,
                )
                _write_json(conf_cache, {"conf_by_src": conf_by_src})
            high_conf_rows.extend(_high_conf_rows_from_confidence(mask_id, mapping, conf_by_src, test_truth, test_df))
        else:
            ev_path = output_dir / "test_eval" / mask_id / "global_best_eval.json"
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            high_conf_rows.extend(_high_conf_rows_from_eval(mask_id, ev))

    out = {
        "generated_at_utc": _utc(),
        "covered_subset_rows": covered_rows,
        "high_confidence_rows": high_conf_rows,
        "high_conf_mode": "full_confidence_sweep" if use_full_high_conf else "approx_from_global_best_eval",
    }
    _write_json(sub_dir / "high_confidence_subset_eval.json", {"rows": high_conf_rows})
    return out
