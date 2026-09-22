#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Route A v2: symmetric masking with Phase 1-compatible Connector decimal bootstrap.

Outputs: cross/out/baseline_compare/routeA_symmetric_masking_v2/
Does NOT overwrite routeA_symmetric_masking/ (v1 rejected).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup  # noqa: E402
from cross.application.experiments.run_admissible_decoding import (  # noqa: E402
    _build_predictions,
    _evaluate_strategy,
    _truth_from_labels,
)
from cross.application.experiments.uot_cache_utils import (  # noqa: E402
    build_cost_matrix_from_components,
    load_cost_component_cache,
    load_flow_segments,
    solve_from_cache,
)
from cross.domain.evaluation.flow_metrics import pair_precision_recall_f1  # noqa: E402
from cross.domain.labels.anchor_masking import AnchorMaskMode, mask_matching_flows  # noqa: E402
from cross.domain.uot.cost_matrix import default_cost_weights  # noqa: E402
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY  # noqa: E402
from cross.shared.normalize import norm_addr  # noqa: E402
from cross.shared.transfers import bnb_df_to_dst_txs, eth_df_to_src_txs  # noqa: E402

BASE = REPO / "out" / "baseline_compare"
LABELS = BASE / "labels"
OUT = BASE / "routeA_symmetric_masking_v2"
OUT_V1 = BASE / "routeA_symmetric_masking"
PHASE1 = BASE / "connector_phase1"
RC_FROZEN = BASE / "rc_uot_q_frozen" / "rc_uot_q_reference_metrics.json"
UOT_BASE = REPO / "out" / "uot_delay_fixed_production"
CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"
CONNECTOR_SAMPLE = CONNECTOR_ROOT / "data" / "Validation" / "ETH-BNB" / "Celer" / "sample.json"
CONNECTOR_DST = CONNECTOR_ROOT / "core" / "dst_chain.py"
ETH_CSV = REPO / "in" / "Celer_ETH_cun.csv"
BNB_CSV = REPO / "label" / "tx" / "Celer_BNB_qu.csv"
N_GT = 7296

# Phase 1 canonical acceptance (hard)
P1_F1 = 0.9736140350877193
P1_N_PRED = 6954
P1_N_CORRECT = 6937
P1_N_NO_MATCH = 342
P1_F1_TOL = 1e-4

RC_STRATEGIES = (
    "raw_argmax_fixed_delay",
    "positive_delay_top3_rescue",
    "joint_time_admissible_filter",
)


class MaskLevel(StrEnum):
    FULL_NATIVE = "full_native"
    ID_ANCHOR_MASKED = "id_anchor_masked"
    NO_RECEIVER = "no_receiver"
    NO_AMOUNT = "no_amount"
    NO_RECEIVER_NO_AMOUNT = "no_receiver_no_amount"


MASK_ORDER: tuple[MaskLevel, ...] = (
    MaskLevel.FULL_NATIVE,
    MaskLevel.ID_ANCHOR_MASKED,
    MaskLevel.NO_RECEIVER,
    MaskLevel.NO_AMOUNT,
    MaskLevel.NO_RECEIVER_NO_AMOUNT,
)


@dataclass(frozen=True)
class MaskSpec:
    level: MaskLevel
    order: int
    connector_description: str
    rc_uot_q_description: str


MASK_SPECS: dict[MaskLevel, MaskSpec] = {
    MaskLevel.FULL_NATIVE: MaskSpec(MaskLevel.FULL_NATIVE, 0, "Native bridge deposit features.", "Frozen RC-UOT-Q reference."),
    MaskLevel.ID_ANCHOR_MASKED: MaskSpec(MaskLevel.ID_ANCHOR_MASKED, 1, "ID metadata masked; WL columns unchanged.", "leave_key_out + re-solve."),
    MaskLevel.NO_RECEIVER: MaskSpec(MaskLevel.NO_RECEIVER, 2, "args.receiver cleared.", "route/graph/novelty zero + address mask."),
    MaskLevel.NO_AMOUNT: MaskSpec(MaskLevel.NO_AMOUNT, 3, "args.amount=0.", "amount weight zero."),
    MaskLevel.NO_RECEIVER_NO_AMOUNT: MaskSpec(MaskLevel.NO_RECEIVER_NO_AMOUNT, 4, "receiver cleared + amount zero.", "combined."),
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    w = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(w.values())
    if total <= 0:
        return default_cost_weights()
    return {k: v / total for k, v in w.items()}


# --- Connector (Phase 1 bootstrap) ---------------------------------------------

def _load_withdraw_locator():
    if str(CONNECTOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CONNECTOR_ROOT))
    from core.dst_chain import WithdrawLocator  # type: ignore

    return WithdrawLocator


def _sample_map() -> dict[str, dict[str, Any]]:
    return {norm_addr(x["txhash"]): x for x in json.loads(CONNECTOR_SAMPLE.read_text(encoding="utf-8"))}


def _item_to_native_row(item: dict[str, Any]) -> dict[str, Any]:
    args = item.get("args") or {}
    return {
        "txhash": norm_addr(item.get("txhash", "")),
        "timestamp": float(item.get("timestamp", 0) or 0),
        "args.receiver": norm_addr(args.get("receiver", "")),
        "args.amount": float(args.get("amount", 0) or 0),
        "args.asset_s": str(args.get("asset_s", "") or "").strip().lower(),
        "args.srcChain": str(args.get("srcChain", "ETH") or "ETH"),
        "args.dstChain": str(args.get("dstChain", "BNB") or "BNB"),
    }


def _connector_row_for_level(item: dict[str, Any], level: MaskLevel) -> dict[str, Any]:
    row = _item_to_native_row(item)
    if level in (MaskLevel.FULL_NATIVE, MaskLevel.ID_ANCHOR_MASKED):
        return row
    if level == MaskLevel.NO_RECEIVER:
        row["args.receiver"] = ""
        return row
    if level == MaskLevel.NO_AMOUNT:
        row["args.amount"] = 0.0
        return row
    if level == MaskLevel.NO_RECEIVER_NO_AMOUNT:
        row["args.receiver"] = ""
        row["args.amount"] = 0.0
        return row
    raise ValueError(level)


def _bootstrap_decimal_dict_phase1(
    WithdrawLocator: Any,
    sample_map: dict[str, dict[str, Any]],
    gt_src: list[str],
    dst_df: pd.DataFrame,
) -> tuple[Any, dict[str, Any]]:
    """Phase 1 semantics: one bootstrap row per unique args.asset_s over ALL gt_src."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    asset_sources: list[dict[str, str]] = []
    for tx in gt_src:
        asset = str((sample_map[tx].get("args") or {}).get("asset_s", "") or "").strip().lower()
        if not asset or asset in seen:
            continue
        seen.add(asset)
        row = _item_to_native_row(sample_map[tx])
        rows.append(row)
        asset_sources.append({"asset_s": asset, "bootstrap_src_tx": tx, "srcChain": row["args.srcChain"]})

    if not rows:
        raise RuntimeError("No unique assets for decimal bootstrap")

    boot = WithdrawLocator(src_txs=pd.DataFrame(rows), dst_txs=dst_df)
    decimal_dict = boot.decimal_dict

    missing: list[dict[str, str]] = []
    warnings: list[str] = []
    for rec in asset_sources:
        chain = rec["srcChain"]
        asset = rec["asset_s"]
        chain_dict = decimal_dict.get(chain, {}) if isinstance(decimal_dict, dict) else {}
        if not chain_dict or asset not in chain_dict:
            missing.append(rec)

    audit = {
        "bootstrap_semantics": "phase1_unique_asset_s_over_all_gt_src",
        "n_gt_src": len(gt_src),
        "n_unique_assets": len(rows),
        "n_bootstrap_rows": len(rows),
        "forbidden_gt_src_slice": False,
        "asset_sources": asset_sources,
        "missing_decimals": missing,
        "decimal_bootstrap_status": "WARN" if missing else "PASS",
    }
    if missing:
        warnings.append(f"{len(missing)} assets missing from decimal_dict after bootstrap")
    audit["warnings"] = warnings
    return decimal_dict, audit


def _make_locator(WithdrawLocator: Any, src_row: pd.DataFrame, dst_df: pd.DataFrame, decimal_dict: Any) -> Any:
    loc = WithdrawLocator.__new__(WithdrawLocator)
    loc.src_txs = src_row
    loc.dst_txs = dst_df
    loc.src_tx_group = src_row.groupby(["args.srcChain", "args.dstChain"])
    loc.decimal_dict = decimal_dict
    return loc


def _blocked_status_for_level(level: MaskLevel, zero_after_receiver: bool) -> str:
    if zero_after_receiver and level in (MaskLevel.NO_RECEIVER, MaskLevel.NO_RECEIVER_NO_AMOUNT):
        return "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS"
    if zero_after_receiver:
        return "BLOCKED_BY_MASKING"
    return "BLOCKED_BY_MASKING"


def _run_connector_level(
    level: MaskLevel,
    gt_src: list[str],
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    decimal_dict: Any,
    decimal_audit: dict[str, Any],
) -> dict[str, Any]:
    level_dir = OUT / "connector" / level.value
    level_dir.mkdir(parents=True, exist_ok=True)
    WithdrawLocator = _load_withdraw_locator()

    audit = {
        "generated_at_utc": _utc(),
        "mask_level": level.value,
        "decimal_bootstrap": decimal_audit,
    }

    # Probe with shared decimal_dict
    src0 = gt_src[0]
    probe_row = pd.DataFrame([_connector_row_for_level(sample_map[src0], level)])
    loc0 = _make_locator(WithdrawLocator, probe_row, dst_df, decimal_dict)
    try:
        out = loc0.search_withdraw(fulloutput=True)
        recs, dbg = (out[0], out[1]) if isinstance(out, tuple) else (out, {})
        probe = {
            "runs_without_exception": True,
            "probe_has_prediction": bool(recs),
            "probe_debug": dbg.get(src0, {}) if isinstance(dbg, dict) else {},
        }
    except Exception as exc:
        probe = {"runs_without_exception": False, "probe_error": str(exc), "probe_debug": {}}

    audit["probe"] = probe
    zero_after_receiver = probe.get("probe_debug", {}).get("after_receiver_rows", -1) == 0

    if not probe.get("runs_without_exception") or zero_after_receiver:
        status = _blocked_status_for_level(level, zero_after_receiver)
        raw = {
            "method": "connector",
            "mask_level": level.value,
            "operating_point": "raw_top1",
            "status": status,
            "pair_precision": None,
            "pair_recall": None,
            "pair_f1": None,
            "n_predicted_pairs": 0,
            "n_no_match": N_GT,
            "tx_coverage": 0.0,
        }
        adm = {**raw, "operating_point": "top1_admissible_filter", "not_joint_time_admissible_filter": True}
        _write_json(level_dir / "masking_audit.json", audit)
        _write_json(level_dir / "raw_eval.json", raw)
        _write_json(level_dir / "top1_admissible_eval.json", adm)
        return {"raw": raw, "admissible": adm, "audit": audit, "predictions": []}

    predictions: list[dict[str, str]] = []
    no_match: list[str] = []
    t0 = time.time()
    for i, src_tx in enumerate(gt_src):
        src_row = pd.DataFrame([_connector_row_for_level(sample_map[src_tx], level)])
        loc = _make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
        recs = loc.search_withdraw()
        dst = norm_addr(recs[0].get("dstTxHash", "")) if recs else ""
        if dst:
            predictions.append({"src_tx": src_tx, "dst_tx": dst})
        else:
            no_match.append(src_tx)
        if (i + 1) % 1000 == 0:
            print(f"  connector v2 {level.value}: {i+1}/{len(gt_src)} {time.time()-t0:.1f}s", flush=True)

    if predictions:
        pd.DataFrame(predictions).to_csv(level_dir / "predictions_raw_top1.csv", index=False)
    if no_match:
        pd.DataFrame({"src_tx": no_match, "status": "no_match"}).to_csv(level_dir / "no_match_src_txs.csv", index=False)

    pred_df = pd.DataFrame([{"srcTxHash": p["src_tx"], "dstTxHash": p["dst_tx"]} for p in predictions])
    pr = pair_precision_recall_f1(
        pred_df, label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"})
    )

    if level in (MaskLevel.NO_RECEIVER, MaskLevel.NO_RECEIVER_NO_AMOUNT) and len(predictions) == 0:
        status = "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS"
    elif level == MaskLevel.NO_AMOUNT and len(predictions) == 0:
        status = "ZERO_PREDICTIONS_AFTER_AMOUNT_MASK"
    else:
        status = "ACCEPTED"

    raw = {
        "method": "connector",
        "mask_level": level.value,
        "operating_point": "raw_top1",
        "status": status,
        "pair_precision": pr["pair_precision"],
        "pair_recall": pr["pair_recall"],
        "pair_f1": pr["pair_f1"],
        "n_gt_pairs": N_GT,
        "n_predicted_pairs": len(predictions),
        "n_correct_pairs": int(pr["tp"]),
        "n_no_match": len(no_match),
        "tx_coverage": len(predictions) / N_GT,
        "metric_unit": "tx_pair_exact_match",
        "closed_set_warning": True,
        "connector_top3_status": "N/A",
        "joint_rc_uot_q_parity_status": "N/A",
    }

    truth = dict(zip(label_df["srcTxHash"].map(norm_addr), label_df["dstTxHash"].map(norm_addr)))
    pred_by_src = {p["src_tx"]: p["dst_tx"] for p in predictions}
    filtered: list[dict[str, str]] = []
    abstained = 0
    tx_viol = tx_eval = 0
    for src_tx in truth:
        pred_dst = pred_by_src.get(src_tx)
        if not pred_dst:
            abstained += 1
            continue
        ts_s, ts_d = eth_ts.get(src_tx), bnb_ts.get(pred_dst)
        if ts_s is None or ts_d is None or float(ts_d - ts_s) < 0:
            abstained += 1
            continue
        filtered.append({"srcTxHash": src_tx, "dstTxHash": pred_dst})
        tx_eval += 1
        if float(bnb_ts[pred_dst] - eth_ts[src_tx]) < 0:
            tx_viol += 1
    fdf = pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=["srcTxHash", "dstTxHash"])
    pr_f = pair_precision_recall_f1(fdf, label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"}))
    adm = {
        "method": "connector",
        "mask_level": level.value,
        "operating_point": "top1_admissible_filter",
        "not_joint_time_admissible_filter": True,
        "status": status,
        "filtered_precision": pr_f["pair_precision"],
        "filtered_recall": pr_f["pair_recall"],
        "filtered_f1": pr_f["pair_f1"],
        "n_abstained": abstained,
        "abstention_rate": abstained / N_GT,
        "tx_CVR": tx_viol / max(tx_eval, 1),
        "tx_coverage": len(filtered) / N_GT,
    }

    audit["connector_run_status"] = status
    audit["n_predictions"] = len(predictions)
    _write_json(level_dir / "masking_audit.json", audit)
    _write_json(level_dir / "raw_eval.json", raw)
    _write_json(level_dir / "top1_admissible_eval.json", adm)
    return {"raw": raw, "admissible": adm, "audit": audit, "predictions": predictions}


# --- RC-UOT-Q (same as v1, new OUT) ------------------------------------------

def _mask_flow_addresses(flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(flows)
    for f in out:
        f["addresses"] = ""
        f["address_features"] = ""
        f["address_count"] = 0
    return out


def _rc_weights_for_level(level: MaskLevel) -> dict[str, float]:
    w = dict(default_cost_weights())
    if level in (MaskLevel.NO_RECEIVER, MaskLevel.NO_RECEIVER_NO_AMOUNT):
        w["route"] = w["graph"] = w["novelty"] = 0.0
    if level in (MaskLevel.NO_AMOUNT, MaskLevel.NO_RECEIVER_NO_AMOUNT):
        w["amount"] = 0.0
    return _renormalize_weights(w)


def _prepare_rc_flows(level: MaskLevel, eth: list, bnb: list) -> tuple[list, list, dict]:
    meta: dict[str, Any] = {"mask_level": level.value}
    eth_f, bnb_f = deepcopy(eth), deepcopy(bnb)
    if level == MaskLevel.ID_ANCHOR_MASKED:
        eth_f, m1 = mask_matching_flows(eth_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
        bnb_f, m2 = mask_matching_flows(bnb_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
        meta["masked_flow_fields"] = sorted(set(m1 + m2))
    if level in (MaskLevel.NO_RECEIVER, MaskLevel.NO_RECEIVER_NO_AMOUNT):
        eth_f = _mask_flow_addresses(eth_f)
        bnb_f = _mask_flow_addresses(bnb_f)
    meta["cost_weights"] = _rc_weights_for_level(level)
    return eth_f, bnb_f, meta


def _rc_frozen_level_result(level: MaskLevel) -> dict[str, Any]:
    frozen = json.loads(RC_FROZEN.read_text(encoding="utf-8"))
    strategies = {}
    for op in RC_STRATEGIES:
        mt = frozen["methods"][op]["main_table"]
        strategies[op] = {**mt, "method": op, "source": "rc_uot_q_frozen/rc_uot_q_reference_metrics.json"}
    return {"mask_level": level.value, "status": "ACCEPTED_FROZEN_REFERENCE", "strategies": strategies}


def _run_rc_uot_q_level(level: MaskLevel, label_df: pd.DataFrame, eth_df: pd.DataFrame, bnb_df: pd.DataFrame) -> dict:
    level_dir = OUT / "rc_uot_q" / level.value
    level_dir.mkdir(parents=True, exist_ok=True)
    if level == MaskLevel.FULL_NATIVE:
        result = _rc_frozen_level_result(level)
        _write_json(level_dir / "decode_eval.json", result)
        return result

    cache = load_cost_component_cache(UOT_BASE)
    eth = load_flow_segments(UOT_BASE / "uot" / "uot_flow_segments_eth.csv")
    bnb = load_flow_segments(UOT_BASE / "uot" / "uot_flow_segments_bnb.csv")
    eth_f, bnb_f, mask_meta = _prepare_rc_flows(level, eth, bnb)
    c_mat = build_cost_matrix_from_components(
        cache["components"],
        time_weight=1.0,
        causal_weight=1.0,
        baseline_causal_penalty=float(cache["baseline_causal_penalty"]),
        max_delay_sec=float(cache["max_delay_sec"]),
        base_weights=mask_meta["cost_weights"],
        delay_policy=DEFAULT_TIME_DELAY_POLICY,
        source_flows=eth_f,
        target_flows=bnb_f,
    )
    p = solve_from_cache(cache, c_mat)
    np.savez_compressed(level_dir / "transport_matrix.npz", P=p)
    truth = _truth_from_labels(label_df)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
    tx_to_j: dict[str, set[int]] = {}
    for j, tf in enumerate(bnb_f):
        for txh in tf.get("tx_hashes") or []:
            tx_to_j.setdefault(norm_addr(str(txh)), set()).add(j)
    flow_txs = {norm_addr(str(h)) for f in eth_f for h in (f.get("tx_hashes") or [])}
    src_all = eth_df_to_src_txs(eth_df)
    if flow_txs:
        src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)
    strategies = {}
    for strategy in RC_STRATEGIES:
        mapping, meta = _build_predictions(
            strategy=strategy, p=p, eth_flows=eth_f, bnb_flows=bnb_f,
            src_all=src_all, dst_norm=dst_norm, truth=truth, eth_ts=eth_ts, bnb_ts=bnb_ts,
        )
        strategies[strategy] = _evaluate_strategy(
            method=strategy, mapping=mapping, meta=meta, truth=truth, p=p,
            eth_flows=eth_f, bnb_flows=bnb_f, label_df=label_df,
            eth_ts=eth_ts, bnb_ts=bnb_ts, tx_to_j=tx_to_j,
        )
    result = {"mask_level": level.value, "status": "ACCEPTED", "strategies": strategies}
    _write_json(level_dir / "masking_audit.json", mask_meta)
    _write_json(level_dir / "decode_eval.json", result)
    return result


# --- Audits & packaging --------------------------------------------------------

def _prediction_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    a = a.copy()
    b = b.copy()
    a["src_tx"] = a["src_tx"].map(norm_addr)
    a["dst_tx"] = a["dst_tx"].map(norm_addr)
    b["src_tx"] = b["src_tx"].map(norm_addr)
    b["dst_tx"] = b["dst_tx"].map(norm_addr)
    ma = dict(zip(a["src_tx"], a["dst_tx"]))
    mb = dict(zip(b["src_tx"], b["dst_tx"]))
    sa, sb = set(ma), set(mb)
    diff_dst = sorted(s for s in sa & sb if ma[s] != mb[s])
    return {
        "n_a": len(a),
        "n_b": len(b),
        "src_overlap": len(sa & sb),
        "only_a": len(sa - sb),
        "only_b": len(sb - sa),
        "diff_dst_count": len(diff_dst),
        "diff_dst_samples": diff_dst[:20],
        "equal": len(sa - sb) == 0 and len(sb - sa) == 0 and len(diff_dst) == 0,
    }


def _acceptance_check(full_native: dict[str, Any]) -> dict[str, Any]:
    raw = full_native["raw"]
    preds = full_native.get("predictions") or []
    p1 = pd.read_csv(PHASE1 / "pred_tx_pairs_connector_native_shared_pool_raw.csv", dtype=str)
    v2 = pd.DataFrame(preds)
    diff = _prediction_diff(
        p1.rename(columns={"src_tx": "src_tx", "dst_tx": "dst_tx"}),
        v2.rename(columns={"src_tx": "src_tx", "dst_tx": "dst_tx"}) if len(v2) else pd.DataFrame(columns=["src_tx", "dst_tx"]),
    )
    checks = {
        "n_predicted": raw["n_predicted_pairs"] == P1_N_PRED,
        "n_correct": raw["n_correct_pairs"] == P1_N_CORRECT,
        "n_no_match": raw["n_no_match"] == P1_N_NO_MATCH,
        "f1_tol": abs(raw["pair_f1"] - P1_F1) <= P1_F1_TOL,
        "prediction_set_equal": diff["equal"],
        "no_extra_vs_phase1": diff["only_b"] == 0,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "phase1_canonical": {"f1": P1_F1, "n_predicted": P1_N_PRED, "n_correct": P1_N_CORRECT, "n_no_match": P1_N_NO_MATCH},
        "v2_full_native": {
            "f1": raw["pair_f1"],
            "n_predicted": raw["n_predicted_pairs"],
            "n_correct": raw["n_correct_pairs"],
            "n_no_match": raw["n_no_match"],
        },
        "prediction_diff_vs_phase1": diff,
    }


def _candidate_pool_audit() -> dict[str, Any]:
    cand_path = LABELS / "candidate_bnb_universe_all_txs.csv"
    cand_df = pd.read_csv(cand_path, dtype=str)
    gt_dst = {norm_addr(x) for x in pd.read_csv(LABELS / "gt_tx_pairs.csv", dtype=str)["dst_tx_hash"]}
    cand_txs = {norm_addr(x) for x in cand_df["tx_hash"]}
    return {
        "candidate_pool_path": str(cand_path.relative_to(REPO)).replace("\\", "/"),
        "n_candidate_txs": len(cand_txs),
        "n_gt_dst_txs": len(gt_dst),
        "candidate_is_closed_gt_dst_set": cand_txs == gt_dst,
        "closed_set_warning": True,
        "candidate_pool_sha256": _sha256(cand_path),
    }


def _load_connector_results_from_disk() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for level in MASK_ORDER:
        level_dir = OUT / "connector" / level.value
        raw = json.loads((level_dir / "raw_eval.json").read_text(encoding="utf-8"))
        adm = json.loads((level_dir / "top1_admissible_eval.json").read_text(encoding="utf-8"))
        audit = json.loads((level_dir / "masking_audit.json").read_text(encoding="utf-8"))
        pred_path = level_dir / "predictions_raw_top1.csv"
        preds = pd.read_csv(pred_path, dtype=str).to_dict("records") if pred_path.is_file() else []
        results[level.value] = {"raw": raw, "admissible": adm, "audit": audit, "predictions": preds}
    return results


def _load_rc_results_from_disk() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for level in MASK_ORDER:
        path = OUT / "rc_uot_q" / level.value / "decode_eval.json"
        if path.is_file():
            results[level.value] = json.loads(path.read_text(encoding="utf-8"))
    return results


def _patch_connector_blocked_status() -> None:
    for level in (MaskLevel.NO_RECEIVER, MaskLevel.NO_RECEIVER_NO_AMOUNT):
        level_dir = OUT / "connector" / level.value
        raw_path = level_dir / "raw_eval.json"
        if not raw_path.is_file():
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if raw.get("n_predicted_pairs", -1) == 0 and raw.get("status") == "ACCEPTED":
            raw["status"] = "BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS"
            raw["pair_precision"] = None
            raw["pair_recall"] = None
            raw["pair_f1"] = None
            _write_json(raw_path, raw)
            adm_path = level_dir / "top1_admissible_eval.json"
            if adm_path.is_file():
                adm = json.loads(adm_path.read_text(encoding="utf-8"))
                adm["status"] = raw["status"]
                _write_json(adm_path, adm)
            audit_path = level_dir / "masking_audit.json"
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit["connector_run_status"] = raw["status"]
                _write_json(audit_path, audit)


def _write_v2_artifacts(
    curve_rows: list,
    acceptance: dict,
    decimal_audit: dict,
    id_diff: dict,
    v1_diff: dict,
    pool_audit: dict[str, Any],
) -> None:
    spec = {
        "version": "v2",
        "generated_at_utc": _utc(),
        "bootstrap_fix": "phase1_unique_asset_s_over_all_gt_src",
        "rejected_v1_dir": "routeA_symmetric_masking",
        "rejected_v1_reason": "decimal_bootstrap_inconsistency_gt_src_slice_20",
        "mask_levels": [l.value for l in MASK_ORDER],
        "acceptance": acceptance,
    }
    _write_json(OUT / "symmetric_masking_spec_v2.json", spec)

    protocol = OUT / "symmetric_masking_protocol_v2.md"
    protocol.write_text(
        "\n".join([
            "# Route A v2 — Symmetric masking protocol",
            "",
            f"Generated: {_utc()}",
            "",
            "**v2 fix:** Connector `decimal_dict` bootstrap matches Phase 1 (all unique `args.asset_s` over 7296 gt src).",
            "",
            "**v1 rejected:** `routeA_symmetric_masking/` — `rejected_due_to_decimal_bootstrap_inconsistency`.",
            "",
            "## Warnings",
            "",
            "- Closed-set pool (7296 candidate = GT dst set)",
            "- Connector top-3 / joint parity: N/A",
            "- ABCTracer: BLOCKED (no checkpoint)",
            "- Degradation/applicability curve — not a simple leaderboard",
            "",
        ]) + "\n",
        encoding="utf-8",
    )

    md = [
        "# Route A v2 degradation curve",
        "",
        "Headline: **Connector raw top-1** vs **RC-UOT-Q joint_time_admissible_filter**.",
        "",
        "This is a degradation / applicability curve, not a simple leaderboard.",
        "",
        "| order | mask_level | Connector raw F1 | RC-UOT-Q joint F1 | Connector > RC-UOT-Q? |",
        "|------:|------------|-----------------:|------------------:|:---------------------:|",
    ]
    for r in curve_rows:
        cr, jr = r.get("connector_raw_f1"), r.get("rc_uot_q_joint_f1")
        md.append(
            f"| {r['order']} | `{r['mask_level']}` | "
            f"{cr if cr is not None else 'N/A'} | {jr if jr is not None else 'N/A'} | "
            f"{'yes' if r.get('connector_beats_rc') else 'no'} |"
        )
    (OUT / "degradation_curve_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_json(OUT / "degradation_curve_v2.json", {"rows": curve_rows})

    report = [
        "# Route A v2 run report",
        "",
        f"Generated: {_utc()}",
        "",
        f"**Phase 1 acceptance:** {'PASS' if acceptance['all_pass'] else 'FAIL'}",
        "",
        "## Degradation curve",
        "",
    ]
    for r in curve_rows:
        report.append(
            f"- `{r['mask_level']}`: Connector raw={r.get('connector_raw_f1')}, "
            f"RC-UOT-Q joint={r.get('rc_uot_q_joint_f1')}, status={r.get('connector_raw_status')}"
        )
    report += [
        "",
        "## Notes",
        "",
        "- full_native / id_anchor_masked: Connector may exceed RC-UOT-Q under native bridge semantics (closed-set).",
        "- no_receiver / no_receiver_no_amount: Connector blocked by receiver semantics.",
        "- no_amount: Connector may collapse (ZERO_PREDICTIONS_AFTER_AMOUNT_MASK).",
        "",
    ]
    (OUT / "routeA_v2_run_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    audit_md = [
        "# Route A v2 consistency audit",
        "",
        f"Generated: {_utc()}",
        "",
        "## Phase 1 vs v2 full_native",
        "",
        f"**All pass:** {acceptance['all_pass']}",
        "",
        json.dumps(acceptance, indent=2),
        "",
        "## id_anchor_masked vs full_native",
        "",
        json.dumps(id_diff, indent=2),
        "",
        "## Rejected v1 vs v2 full_native",
        "",
        json.dumps(v1_diff, indent=2),
        "",
        "## Decimal bootstrap",
        "",
        json.dumps(decimal_audit, indent=2),
        "",
        "## Candidate pool",
        "",
        json.dumps(pool_audit, indent=2),
        "",
    ]
    (OUT / "routeA_v2_consistency_audit.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")
    _write_json(OUT / "phase1_vs_v2_prediction_equality_audit.json", acceptance)
    _write_json(OUT / "candidate_pool_audit.json", pool_audit)


def _build_curve_rows(connector_results: dict[str, Any], rc_results: dict[str, Any]) -> list[dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    for level in MASK_ORDER:
        conn = connector_results[level.value]
        raw = conn["raw"]
        rc = rc_results.get(level.value, {})
        joint = rc.get("strategies", {}).get("joint_time_admissible_filter", {})
        jf1 = joint.get("pair_f1")
        cr = raw.get("pair_f1")
        curve_rows.append({
            "order": MASK_SPECS[level].order,
            "mask_level": level.value,
            "connector_raw_f1": cr,
            "connector_raw_status": raw.get("status"),
            "connector_top1_adm_f1": conn["admissible"].get("filtered_f1"),
            "rc_uot_q_joint_f1": jf1,
            "rc_uot_q_joint_tx_cvr": joint.get("tx_level_cvr"),
            "rc_uot_q_joint_coverage": joint.get("coverage"),
            "connector_beats_rc": cr is not None and jf1 is not None and float(cr) > float(jf1),
            "abctracer_status": "BLOCKED",
        })
    return curve_rows


def _finalize_manifest(
    acceptance: dict,
    decimal_audit: dict,
    pool_audit: dict[str, Any],
) -> None:
    file_hashes = {}
    for p in OUT.rglob("*"):
        if p.is_file():
            file_hashes[str(p.relative_to(REPO)).replace("\\", "/")] = _sha256(p)

    manifest = {
        "generated_at_utc": _utc(),
        "version": "v2",
        "output_root": str(OUT.relative_to(REPO)).replace("\\", "/"),
        "rejected_prior": {
            "path": "out/baseline_compare/routeA_symmetric_masking",
            "status": "rejected_due_to_decimal_bootstrap_inconsistency",
            "rejected_f1": 0.9953379953379954,
            "rejected_n_predicted": 7290,
        },
        "phase1_canonical_connector_native": {
            "f1": P1_F1,
            "n_predicted": P1_N_PRED,
            "n_correct": P1_N_CORRECT,
            "n_no_match": P1_N_NO_MATCH,
            "source": "connector_phase1/connector_raw_eval.json",
        },
        "v2_full_native_acceptance": acceptance,
        "closed_set_warning": True,
        "no_top_k_warning": True,
        "connector_top3_status": "N/A",
        "abctracer_status": "BLOCKED",
        "abctracer_reason": "no official wgt.pth checkpoint",
        "candidate_pool_audit": pool_audit,
        "decimals_bootstrap_audit_path": "out/baseline_compare/routeA_symmetric_masking_v2/decimals_bootstrap_audit.json",
        "gt_tx_pairs_sha256": _sha256(LABELS / "gt_tx_pairs.csv"),
        "connector_dst_chain_sha256": _sha256(CONNECTOR_DST),
        "file_sha256": file_hashes,
        "manuscript_modified": False,
    }
    _write_json(OUT / "routeA_v2_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Route A v2 symmetric masking runner")
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Skip Connector re-run; load existing outputs, run missing RC-UOT-Q, write v2 artifacts",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool_audit = _candidate_pool_audit()

    gt = pd.read_csv(LABELS / "gt_tx_pairs.csv", dtype=str)
    gt_src = [norm_addr(x) for x in pd.read_csv(LABELS / "gt_src_txs.csv", dtype=str)["src_tx_hash"]]
    label_df = gt.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    eth_df = pd.read_csv(ETH_CSV, dtype=str, low_memory=False)
    bnb_raw = pd.read_csv(BNB_CSV, dtype=str, low_memory=False)
    bnb_raw["hash"] = bnb_raw["hash"].map(norm_addr)

    decimal_audit_path = OUT / "decimals_bootstrap_audit.json"
    if decimal_audit_path.is_file():
        decimal_audit = json.loads(decimal_audit_path.read_text(encoding="utf-8"))
    else:
        decimal_audit = {}

    if args.artifacts_only:
        print("Route A v2: artifacts-only resume...", flush=True)
        _patch_connector_blocked_status()
        connector_results = _load_connector_results_from_disk()
    else:
        print("Route A v2: bootstrap + run...", flush=True)
        sample_map = _sample_map()
        cand_txs = {norm_addr(x) for x in pd.read_csv(LABELS / "candidate_bnb_universe_all_txs.csv", dtype=str)["tx_hash"]}
        dst_df = bnb_df_to_dst_txs(bnb_raw[bnb_raw["hash"].isin(cand_txs)].copy())
        eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_raw)
        WithdrawLocator = _load_withdraw_locator()
        decimal_dict, decimal_audit = _bootstrap_decimal_dict_phase1(WithdrawLocator, sample_map, gt_src, dst_df)
        _write_json(decimal_audit_path, decimal_audit)
        connector_results = {}
        for level in MASK_ORDER:
            print(f"Route A v2 Connector @ {level.value}", flush=True)
            connector_results[level.value] = _run_connector_level(
                level, gt_src, sample_map, dst_df, label_df, eth_ts, bnb_ts, decimal_dict, decimal_audit
            )

    acceptance = _acceptance_check(connector_results["full_native"])
    if not acceptance["all_pass"]:
        _write_json(OUT / "acceptance_failure.json", acceptance)
        raise SystemExit(f"Route A v2 ACCEPTANCE FAIL: {acceptance['checks']}")

    print("Route A v2: full_native PASS vs Phase 1", flush=True)

    fn_preds = pd.DataFrame(connector_results["full_native"]["predictions"])
    ia_preds = pd.DataFrame(connector_results["id_anchor_masked"]["predictions"])
    id_diff = _prediction_diff(fn_preds, ia_preds) if len(fn_preds) and len(ia_preds) else {"equal": True, "note": "empty"}
    _write_json(OUT / "connector_id_anchor_vs_full_native_diff.json", id_diff)

    v1_path = OUT_V1 / "connector/full_native/predictions_raw_top1.csv"
    v1_diff = {"note": "v1 missing"}
    if v1_path.is_file():
        v1_diff = _prediction_diff(pd.read_csv(v1_path, dtype=str), fn_preds)
    _write_json(OUT / "rejected_v1_vs_v2_full_native_diff.json", v1_diff)

    rc_results = _load_rc_results_from_disk()
    for level in MASK_ORDER:
        if level.value in rc_results:
            continue
        print(f"Route A v2 RC-UOT-Q @ {level.value}", flush=True)
        rc_results[level.value] = _run_rc_uot_q_level(level, label_df, eth_df, bnb_raw)

    curve_rows = _build_curve_rows(connector_results, rc_results)
    _write_v2_artifacts(curve_rows, acceptance, decimal_audit, id_diff, v1_diff, pool_audit)
    _finalize_manifest(acceptance, decimal_audit, pool_audit)
    print("Route A v2 complete.", flush=True)


if __name__ == "__main__":
    main()
