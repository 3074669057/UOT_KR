#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Route A: symmetric masking protocol + degradation curve (RC-UOT-Q vs Connector).

All outputs under cross/out/baseline_compare/routeA_symmetric_masking/ only.
Does not modify manuscript, frozen Phase 1/2/3 JSON, final_paper_tables, or Connector core.
"""
from __future__ import annotations

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
OUT = BASE / "routeA_symmetric_masking"
RC_FROZEN = BASE / "rc_uot_q_frozen" / "rc_uot_q_reference_metrics.json"
UOT_BASE = REPO / "out" / "uot_delay_fixed_production"
CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"
CONNECTOR_SAMPLE = CONNECTOR_ROOT / "data" / "Validation" / "ETH-BNB" / "Celer" / "sample.json"
ETH_CSV = REPO / "in" / "Celer_ETH_cun.csv"
BNB_CSV = REPO / "label" / "tx" / "Celer_BNB_qu.csv"
N_GT = 7296

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
    connector_fields_masked: tuple[str, ...]
    connector_fields_retained: tuple[str, ...]
    rc_uot_q_actions: tuple[str, ...]


MASK_SPECS: dict[MaskLevel, MaskSpec] = {
    MaskLevel.FULL_NATIVE: MaskSpec(
        MaskLevel.FULL_NATIVE,
        0,
        "Native bridge deposit features (receiver, amount, asset, dstChain) unchanged.",
        "Baseline fixed-delay transport + admissible decoding (frozen reference).",
        (),
        ("args.receiver", "args.amount", "args.asset_s", "args.srcChain", "args.dstChain", "timestamp"),
        ("none",),
    ),
    MaskLevel.ID_ANCHOR_MASKED: MaskSpec(
        MaskLevel.ID_ANCHOR_MASKED,
        1,
        "Mask identity/bridge metadata (event, bridge, sender) not used by WithdrawLocator columns; "
        "bridge matching fields retained.",
        "leave_key_out flow-feature masking; cost weights unchanged; transport re-solved from cache.",
        ("event", "bridge", "args.sender"),
        ("args.receiver", "args.amount", "args.asset_s", "args.srcChain", "args.dstChain", "timestamp"),
        ("leave_key_out",),
    ),
    MaskLevel.NO_RECEIVER: MaskSpec(
        MaskLevel.NO_RECEIVER,
        2,
        "Clear args.receiver (WithdrawLocator _match_receiver cannot filter).",
        "Zero route/graph/address_novelty weights; mask flow address fields; re-solve transport.",
        ("args.receiver",),
        ("args.amount", "args.asset_s", "args.srcChain", "args.dstChain", "timestamp"),
        ("zero_route_graph_novelty", "mask_flow_addresses"),
    ),
    MaskLevel.NO_AMOUNT: MaskSpec(
        MaskLevel.NO_AMOUNT,
        3,
        "Set args.amount=0 (WithdrawLocator _match_amount filters args.amount>0).",
        "Zero amount cost weight; re-solve transport.",
        ("args.amount",),
        ("args.receiver", "args.asset_s", "args.srcChain", "args.dstChain", "timestamp"),
        ("zero_amount_weight",),
    ),
    MaskLevel.NO_RECEIVER_NO_AMOUNT: MaskSpec(
        MaskLevel.NO_RECEIVER_NO_AMOUNT,
        4,
        "Clear receiver and zero amount.",
        "Combine no_receiver + no_amount symmetric actions; re-solve transport.",
        ("args.receiver", "args.amount"),
        ("args.asset_s", "args.srcChain", "args.dstChain", "timestamp"),
        ("zero_route_graph_novelty", "mask_flow_addresses", "zero_amount_weight"),
    ),
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    w = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(w.values())
    if total <= 0:
        return default_cost_weights()
    return {k: v / total for k, v in w.items()}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _protocol_artifacts() -> None:
    spec = {
        "generated_at_utc": _utc(),
        "route": "A",
        "designation": "symmetric_masking_degradation_curve",
        "supersedes_note": "Old Phase 2/3 Table 6 fair-comparison line is superseded candidate; Route A is separate.",
        "shared_substrate": {
            "gt_tx_pairs": "labels/gt_tx_pairs.csv",
            "candidate_pool": "labels/candidate_bnb_universe_all_txs.csv",
            "n_gt_pairs": N_GT,
            "delay_policy": DEFAULT_TIME_DELAY_POLICY,
            "admissibility": "same top1 temporal filter for Connector; RC-UOT-Q frozen admissible strategies",
        },
        "mask_levels": [
            {
                "level": s.level.value,
                "order": s.order,
                "connector": {
                    "description": s.connector_description,
                    "fields_masked": list(s.connector_fields_masked),
                    "fields_retained": list(s.connector_fields_retained),
                },
                "rc_uot_q": {
                    "description": s.rc_uot_q_description,
                    "actions": list(s.rc_uot_q_actions),
                },
            }
            for s in (MASK_SPECS[l] for l in MASK_ORDER)
        ],
        "connector_operating_points": ["raw_top1", "top1_admissible_filter"],
        "connector_not_available": ["top3", "joint_time_admissible_filter_rc_uot_q_parity"],
        "abctracer_policy": "BLOCKED if no official checkpoint; no style baseline substitute",
        "honesty_policy": "Report all numeric outcomes even if Connector exceeds RC-UOT-Q at a level.",
    }
    _write_json(OUT / "symmetric_masking_spec.json", spec)

    md = [
        "# Route A — Symmetric masking protocol",
        "",
        f"Generated: {_utc()}",
        "",
        "New experimental branch under `routeA_symmetric_masking/`. Does **not** modify manuscript or old Phase 2/3 artifacts.",
        "",
        "## Shared evaluation substrate",
        "",
        "- GT: `labels/gt_tx_pairs.csv` (7296 tx-pairs)",
        "- Candidate pool: `labels/candidate_bnb_universe_all_txs.csv`",
        "- Delay policy: `tx_if_available_else_flow_representative`",
        "- Connector: original `WithdrawLocator` core; adapter-only masking",
        "- RC-UOT-Q: fixed-delay cost cache from `out/uot_delay_fixed_production`; re-solve only in Route A output dir",
        "",
        "## Degradation levels (presentation order)",
        "",
    ]
    for lvl in MASK_ORDER:
        s = MASK_SPECS[lvl]
        md += [
            f"### {s.order + 1}. `{lvl.value}`",
            "",
            f"- **Connector:** {s.connector_description}",
            f"- **RC-UOT-Q:** {s.rc_uot_q_description}",
            "",
        ]
    md += [
        "## Connector scope",
        "",
        "- Allowed: raw top-1, top1 admissible temporal filter",
        "- N/A: top-3, RC-UOT-Q joint parity (original core is top-1 only)",
        "",
        "## ABCTracer",
        "",
        "BLOCKED — no official checkpoint (`wgt.pth`). No Phase 7.5-style substitute.",
        "",
    ]
    (OUT / "symmetric_masking_protocol.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# --- Connector -----------------------------------------------------------------

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
    if level == MaskLevel.FULL_NATIVE:
        return row
    if level == MaskLevel.ID_ANCHOR_MASKED:
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


def _bootstrap_decimals(WithdrawLocator: Any, sample_map: dict[str, dict[str, Any]], gt_src: list[str], dst_df: pd.DataFrame, level: MaskLevel) -> Any:
    rows = [_connector_row_for_level(sample_map[s], level) for s in gt_src[: min(20, len(gt_src))]]
    boot = WithdrawLocator(src_txs=pd.DataFrame(rows), dst_txs=dst_df)
    return boot.decimal_dict


def _make_locator(WithdrawLocator: Any, src_row: pd.DataFrame, dst_df: pd.DataFrame, decimal_dict: Any) -> Any:
    loc = WithdrawLocator.__new__(WithdrawLocator)
    loc.src_txs = src_row
    loc.dst_txs = dst_df
    loc.src_tx_group = src_row.groupby(["args.srcChain", "args.dstChain"])
    loc.decimal_dict = decimal_dict
    return loc


def _connector_probe(WithdrawLocator: Any, sample_map: dict[str, dict[str, Any]], dst_df: pd.DataFrame, gt_src: list[str], level: MaskLevel) -> dict[str, Any]:
    src = gt_src[0]
    row = pd.DataFrame([_connector_row_for_level(sample_map[src], level)])
    try:
        loc = WithdrawLocator(src_txs=row, dst_txs=dst_df)
        out = loc.search_withdraw(fulloutput=True)
        recs, dbg = (out[0], out[1]) if isinstance(out, tuple) else (out, {})
        dst = norm_addr(recs[0].get("dstTxHash", "")) if recs else ""
        return {
            "runs_without_exception": True,
            "probe_has_prediction": bool(dst),
            "probe_debug": dbg.get(src, {}) if isinstance(dbg, dict) else {},
        }
    except Exception as exc:
        return {"runs_without_exception": False, "probe_has_prediction": False, "probe_error": str(exc)}


def _run_connector_level(
    level: MaskLevel,
    gt_src: list[str],
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> dict[str, Any]:
    level_dir = OUT / "connector" / level.value
    level_dir.mkdir(parents=True, exist_ok=True)
    spec = MASK_SPECS[level]
    audit = {
        "generated_at_utc": _utc(),
        "mask_level": level.value,
        "connector_fields_masked": list(spec.connector_fields_masked),
        "connector_fields_retained": list(spec.connector_fields_retained),
        "id_anchor_note": (
            "event/bridge/sender are not passed to WithdrawLocator adapter columns; "
            "id_anchor_masked is equivalent to native at matcher-input level."
            if level == MaskLevel.ID_ANCHOR_MASKED
            else None
        ),
    }
    _write_json(level_dir / "masking_audit.json", audit)

    WithdrawLocator = _load_withdraw_locator()
    probe = _connector_probe(WithdrawLocator, sample_map, dst_df, gt_src, level)
    audit["probe"] = probe

    zero_after_receiver = probe.get("probe_debug", {}).get("after_receiver_rows", -1) == 0 and not probe.get("probe_has_prediction")
    if not probe.get("runs_without_exception") or zero_after_receiver:
        status = "BLOCKED_BY_MASKING" if level != MaskLevel.FULL_NATIVE else "ERROR"
        reason = f"Probe produced no matches under {level.value}"
        if zero_after_receiver:
            reason = f"Zero rows after _match_receiver under {level.value}"
        raw = {
            "method": "connector",
            "mask_level": level.value,
            "operating_point": "raw_top1",
            "status": status,
            "blocked_reason": reason,
            "pair_precision": None,
            "pair_recall": None,
            "pair_f1": None,
            "n_predicted_pairs": 0,
            "n_no_match": N_GT,
            "tx_coverage": 0.0,
        }
        adm = {**raw, "operating_point": "top1_admissible_filter", "not_joint_time_admissible_filter": True}
        _write_json(level_dir / "raw_eval.json", raw)
        _write_json(level_dir / "top1_admissible_eval.json", adm)
        audit["connector_run_status"] = status
        _write_json(level_dir / "masking_audit.json", audit)
        return {"raw": raw, "admissible": adm, "audit": audit}

    decimal_dict = _bootstrap_decimals(WithdrawLocator, sample_map, gt_src, dst_df, level)
    predictions: list[dict[str, Any]] = []
    t0 = time.time()
    for i, src_tx in enumerate(gt_src):
        src_row = pd.DataFrame([_connector_row_for_level(sample_map[src_tx], level)])
        loc = _make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
        recs = loc.search_withdraw()
        dst = norm_addr(recs[0].get("dstTxHash", "")) if recs else ""
        if dst:
            predictions.append({"src_tx": src_tx, "dst_tx": dst, "rank": 1, "mask_level": level.value})
        if (i + 1) % 1000 == 0:
            print(f"  connector {level.value}: {i+1}/{len(gt_src)} elapsed={time.time()-t0:.1f}s", flush=True)

    if predictions:
        pd.DataFrame(predictions).to_csv(level_dir / "predictions_raw_top1.csv", index=False)

    pred_df = pd.DataFrame([{"srcTxHash": p["src_tx"], "dstTxHash": p["dst_tx"]} for p in predictions])
    pr = pair_precision_recall_f1(pred_df, label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"}))
    raw = {
        "method": "connector",
        "mask_level": level.value,
        "operating_point": "raw_top1",
        "status": "ACCEPTED",
        "pair_precision": pr["pair_precision"],
        "pair_recall": pr["pair_recall"],
        "pair_f1": pr["pair_f1"],
        "n_gt_pairs": N_GT,
        "n_predicted_pairs": len(predictions),
        "n_correct_pairs": int(pr["tp"]),
        "n_no_match": N_GT - len(predictions),
        "tx_coverage": len(predictions) / N_GT,
        "metric_unit": "tx_pair_exact_match",
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
        "status": "ACCEPTED",
        "filtered_precision": pr_f["pair_precision"],
        "filtered_recall": pr_f["pair_recall"],
        "filtered_f1": pr_f["pair_f1"],
        "n_abstained": abstained,
        "abstention_rate": abstained / N_GT,
        "tx_CVR": tx_viol / max(tx_eval, 1),
        "tx_coverage": len(filtered) / N_GT,
    }

    _write_json(level_dir / "raw_eval.json", raw)
    _write_json(level_dir / "top1_admissible_eval.json", adm)
    audit["connector_run_status"] = "RAN_WITH_PREDICTIONS"
    audit["n_predictions"] = len(predictions)
    _write_json(level_dir / "masking_audit.json", audit)
    return {"raw": raw, "admissible": adm, "audit": audit}


# --- RC-UOT-Q ------------------------------------------------------------------

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
        w["route"] = 0.0
        w["graph"] = 0.0
        w["novelty"] = 0.0
    if level in (MaskLevel.NO_AMOUNT, MaskLevel.NO_RECEIVER_NO_AMOUNT):
        w["amount"] = 0.0
    return _renormalize_weights(w)


def _prepare_rc_flows(level: MaskLevel, eth: list[dict[str, Any]], bnb: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {"mask_level": level.value, "actions": list(MASK_SPECS[level].rc_uot_q_actions)}
    eth_f, bnb_f = deepcopy(eth), deepcopy(bnb)
    if level == MaskLevel.ID_ANCHOR_MASKED:
        eth_f, m1 = mask_matching_flows(eth_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
        bnb_f, m2 = mask_matching_flows(bnb_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
        meta["masked_flow_fields"] = sorted(set(m1 + m2))
    if level in (MaskLevel.NO_RECEIVER, MaskLevel.NO_RECEIVER_NO_AMOUNT):
        eth_f = _mask_flow_addresses(eth_f)
        bnb_f = _mask_flow_addresses(bnb_f)
        meta["flow_address_fields_cleared"] = True
    meta["cost_weights"] = _rc_weights_for_level(level)
    return eth_f, bnb_f, meta


def _load_frozen_rc_metrics() -> dict[str, Any]:
    return json.loads(RC_FROZEN.read_text(encoding="utf-8"))


def _rc_frozen_level_result(level: MaskLevel) -> dict[str, Any]:
    frozen = _load_frozen_rc_metrics()
    strategies = {}
    for op in RC_STRATEGIES:
        mt = frozen["methods"][op]["main_table"]
        strategies[op] = {
            "method": op,
            "pair_precision": mt["pair_precision"],
            "pair_recall": mt["pair_recall"],
            "pair_f1": mt["pair_f1"],
            "tx_level_cvr": mt["tx_level_cvr"],
            "coverage": mt["coverage"],
            "abstention_rate": mt["abstention_rate"],
            "n_predicted_pairs": mt["n_true_positive"] + mt["n_false_positive"],
            "n_abstained": mt["n_abstained"],
            "source": "rc_uot_q_frozen/rc_uot_q_reference_metrics.json",
            "transport_reused": True,
        }
    return {
        "mask_level": level.value,
        "status": "ACCEPTED_FROZEN_REFERENCE",
        "strategies": strategies,
        "note": "full_native uses frozen fixed-delay admissible decoding; no re-solve.",
    }


def _run_rc_uot_q_level(
    level: MaskLevel,
    label_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
) -> dict[str, Any]:
    level_dir = OUT / "rc_uot_q" / level.value
    level_dir.mkdir(parents=True, exist_ok=True)

    if level == MaskLevel.FULL_NATIVE:
        result = _rc_frozen_level_result(level)
        _write_json(level_dir / "masking_audit.json", {"mask_level": level.value, "action": "frozen_reference"})
        _write_json(level_dir / "decode_eval.json", result)
        return result

    cache = load_cost_component_cache(UOT_BASE)
    eth = load_flow_segments(UOT_BASE / "uot" / "uot_flow_segments_eth.csv")
    bnb = load_flow_segments(UOT_BASE / "uot" / "uot_flow_segments_bnb.csv")
    eth_f, bnb_f, mask_meta = _prepare_rc_flows(level, eth, bnb)
    weights = mask_meta["cost_weights"]

    c_mat = build_cost_matrix_from_components(
        cache["components"],
        time_weight=1.0,
        causal_weight=1.0,
        baseline_causal_penalty=float(cache["baseline_causal_penalty"]),
        max_delay_sec=float(cache["max_delay_sec"]),
        base_weights=weights,
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

    strategies: dict[str, Any] = {}
    for strategy in RC_STRATEGIES:
        mapping, meta = _build_predictions(
            strategy=strategy,
            p=p,
            eth_flows=eth_f,
            bnb_flows=bnb_f,
            src_all=src_all,
            dst_norm=dst_norm,
            truth=truth,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
        )
        row = _evaluate_strategy(
            method=strategy,
            mapping=mapping,
            meta=meta,
            truth=truth,
            p=p,
            eth_flows=eth_f,
            bnb_flows=bnb_f,
            label_df=label_df,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            tx_to_j=tx_to_j,
        )
        strategies[strategy] = row

    mask_meta.update(
        {
            "generated_at_utc": _utc(),
            "base_run": str(UOT_BASE),
            "transport_resolved_in_routeA": True,
            "delay_policy": DEFAULT_TIME_DELAY_POLICY,
        }
    )
    result = {
        "mask_level": level.value,
        "status": "ACCEPTED",
        "strategies": strategies,
        "transport_shape": list(p.shape),
    }
    _write_json(level_dir / "masking_audit.json", mask_meta)
    _write_json(level_dir / "decode_eval.json", result)
    return result


# --- Degradation curve ---------------------------------------------------------

def _abctracer_blocked_row(level: MaskLevel) -> dict[str, Any]:
    return {
        "method": "ABCTracer",
        "mask_level": level.value,
        "operating_point": "original",
        "status": "BLOCKED",
        "blocked_reason": "No official checkpoint (wgt.pth) available",
        "pair_f1": None,
    }


def _curve_row(
    level: MaskLevel,
    connector_raw: dict[str, Any],
    connector_adm: dict[str, Any],
    rc: dict[str, Any],
) -> dict[str, Any]:
    joint = rc.get("strategies", {}).get("joint_time_admissible_filter", {})
    if "pair_f1" not in joint and joint:
        joint = {
            "pair_f1": joint.get("pair_f1"),
            "pair_precision": joint.get("pair_precision"),
            "pair_recall": joint.get("pair_recall"),
            "tx_level_cvr": joint.get("tx_level_cvr"),
            "coverage": joint.get("coverage"),
        }
    return {
        "mask_level": level.value,
        "order": MASK_SPECS[level].order,
        "connector_raw_top1_f1": connector_raw.get("pair_f1"),
        "connector_raw_top1_status": connector_raw.get("status"),
        "connector_top1_admissible_f1": connector_adm.get("filtered_f1"),
        "connector_top1_admissible_status": connector_adm.get("status"),
        "rc_uot_q_joint_f1": joint.get("pair_f1"),
        "rc_uot_q_joint_precision": joint.get("pair_precision"),
        "rc_uot_q_joint_recall": joint.get("pair_recall"),
        "rc_uot_q_joint_tx_cvr": joint.get("tx_level_cvr"),
        "rc_uot_q_joint_coverage": joint.get("coverage"),
        "rc_uot_q_status": rc.get("status"),
        "abctracer_status": "BLOCKED",
        "connector_beats_rc_uot_q_joint_f1": (
            connector_raw.get("pair_f1") is not None
            and joint.get("pair_f1") is not None
            and float(connector_raw["pair_f1"]) > float(joint["pair_f1"])
        ),
    }


def _write_curve_table(curve_rows: list[dict[str, Any]]) -> None:
    md = [
        "# Route A degradation curve",
        "",
        f"Generated: {_utc()}",
        "",
        "Honest comparison under symmetric masking levels. **Connector may exceed RC-UOT-Q** at some levels.",
        "",
        "| order | mask_level | Connector raw F1 | Connector top1 adm F1 | RC-UOT-Q joint F1 | RC-UOT-Q joint tx-CVR | Connector > RC-UOT-Q (raw vs joint)? |",
        "|------:|------------|-----------------:|----------------------:|------------------:|----------------------:|:------------------------------------:|",
    ]
    for r in curve_rows:
        cr = r["connector_raw_top1_f1"]
        ca = r["connector_top1_admissible_f1"]
        jr = r["rc_uot_q_joint_f1"]
        md.append(
            f"| {r['order']} | `{r['mask_level']}` | "
            f"{cr if cr is not None else 'N/A'} | {ca if ca is not None else 'N/A'} | "
            f"{jr if jr is not None else 'N/A'} | {r.get('rc_uot_q_joint_tx_cvr', 'N/A')} | "
            f"{'yes' if r.get('connector_beats_rc_uot_q_joint_f1') else 'no'} |"
        )
    (OUT / "degradation_curve.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_json(OUT / "degradation_curve.json", {"generated_at_utc": _utc(), "rows": curve_rows})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Route A: writing protocol...", flush=True)
    _protocol_artifacts()

    gt = pd.read_csv(LABELS / "gt_tx_pairs.csv", dtype=str)
    gt_src = [norm_addr(x) for x in pd.read_csv(LABELS / "gt_src_txs.csv", dtype=str)["src_tx_hash"]]
    label_df = gt.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    sample_map = _sample_map()

    cand_df = pd.read_csv(LABELS / "candidate_bnb_universe_all_txs.csv", dtype=str)
    cand_txs = {norm_addr(x) for x in cand_df["tx_hash"]}
    bnb_raw = pd.read_csv(BNB_CSV, dtype=str, low_memory=False)
    bnb_raw["hash"] = bnb_raw["hash"].map(norm_addr)
    dst_df = bnb_df_to_dst_txs(bnb_raw[bnb_raw["hash"].isin(cand_txs)].copy())
    eth_df = pd.read_csv(ETH_CSV, dtype=str, low_memory=False)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_raw)

    curve_rows: list[dict[str, Any]] = []
    all_results: dict[str, Any] = {"connector": {}, "rc_uot_q": {}, "abctracer": {}}

    for level in MASK_ORDER:
        print(f"Route A: Connector @ {level.value}...", flush=True)
        conn = _run_connector_level(level, gt_src, sample_map, dst_df, label_df, eth_ts, bnb_ts)
        all_results["connector"][level.value] = conn
        all_results["abctracer"][level.value] = _abctracer_blocked_row(level)

        print(f"Route A: RC-UOT-Q @ {level.value}...", flush=True)
        rc = _run_rc_uot_q_level(level, label_df, eth_df, bnb_raw)
        all_results["rc_uot_q"][level.value] = rc

        curve_rows.append(_curve_row(level, conn["raw"], conn["admissible"], rc))

    _write_curve_table(curve_rows)

    manifest = {
        "generated_at_utc": _utc(),
        "route": "A",
        "output_root": str(OUT.relative_to(REPO)),
        "supersedes": "Phase 2/3 Table 6 fair-comparison (candidate only; not used for Route A)",
        "manuscript_modified": False,
        "frozen_dirs_unmodified": [
            "out/final_paper_tables",
            "out/admissible_decoding",
            "out/uot_delay_fixed_production",
            "manuscript_final",
            "connector_phase1",
            "fair_main_compare",
        ],
        "degradation_curve": "degradation_curve.json",
        "protocol": "symmetric_masking_protocol.md",
    }
    _write_json(OUT / "manifest.json", manifest)

    report_lines = [
        "# Route A symmetric masking — run report",
        "",
        f"Generated: {_utc()}",
        "",
        "## Summary",
        "",
        "Degradation curve complete. See `degradation_curve.md`.",
        "",
    ]
    for r in curve_rows:
        flag = " **(Connector joint-beat flag on raw F1)**" if r.get("connector_beats_rc_uot_q_joint_f1") else ""
        report_lines.append(
            f"- `{r['mask_level']}`: Connector raw F1={r['connector_raw_top1_f1']}, "
            f"RC-UOT-Q joint F1={r['rc_uot_q_joint_f1']}{flag}"
        )
    (OUT / "routeA_run_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("Route A complete.", flush=True)


if __name__ == "__main__":
    main()
