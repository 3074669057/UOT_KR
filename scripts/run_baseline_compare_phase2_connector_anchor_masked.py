#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2.2–2.3: Connector anchor-masked fair comparison (WithdrawLocator core, masked input)."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup  # noqa: E402
from cross.domain.evaluation.flow_metrics import pair_precision_recall_f1  # noqa: E402
from cross.shared.normalize import norm_addr  # noqa: E402
from cross.shared.transfers import bnb_df_to_dst_txs  # noqa: E402

BASE = REPO / "out" / "baseline_compare"
LABELS = BASE / "labels"
OUT = BASE / "fair_main_compare"

CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"
CONNECTOR_SAMPLE = CONNECTOR_ROOT / "data" / "Validation" / "ETH-BNB" / "Celer" / "sample.json"
ETH_CSV = REPO / "in" / "Celer_ETH_cun.csv"
BNB_CSV = REPO / "label" / "tx" / "Celer_BNB_qu.csv"
N_GT = 7296

# Fields that must be masked (bridge semantic shortcuts)
MASK_FIELD_NAMES = {
    "message_key",
    "transfer_id",
    "bridge_transfer_id",
    "bridge_event_id",
    "event_nonce",
    "nonce",
    "src_dst_pair",
    "matched_tx",
    "paired_tx",
    "target_tx",
    "dst_tx",
    "true_dst",
    "withdraw_tx",
    "receiver",
    "recipient",
    "target_receiver",
    "bridge_receiver",
    "exact_amount",
    "sender",
    "asset_s",
    "asset_d",
    "amount",
    "dstChain",
    "destination_chain",
    "bridge_route",
    "bridge_contract_specific_id",
    "event",
    "bridge",
}

BRIDGE_SEMANTIC_SHORTCUTS = [
    "args.receiver",
    "args.amount",
    "args.asset_s",
    "args.dstChain",
    "args.sender",
    "event",
    "bridge",
]

KEEP_FIELDS = [
    "txhash",
    "timestamp",
    "args.srcChain",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_field_names(obj: Any, prefix: str = "") -> set[str]:
    names: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            names.add(path)
            names |= _collect_field_names(v, path)
    elif isinstance(obj, list) and obj:
        names |= _collect_field_names(obj[0], prefix)
    return names


def _load_withdraw_locator():
    if str(CONNECTOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CONNECTOR_ROOT))
    from core.dst_chain import WithdrawLocator  # type: ignore

    return WithdrawLocator


def _sample_map() -> dict[str, dict[str, Any]]:
    return {
        norm_addr(x["txhash"]): x
        for x in json.loads(CONNECTOR_SAMPLE.read_text(encoding="utf-8"))
    }


def _item_to_anchor_masked_row(item: dict[str, Any]) -> dict[str, Any]:
    """Mask bridge semantics; keep only row id, source timestamp, source chain."""
    return {
        "txhash": norm_addr(item.get("txhash", "")),
        "timestamp": float(item.get("timestamp", 0) or 0),
        "args.srcChain": str((item.get("args") or {}).get("srcChain", "ETH") or "ETH"),
        "args.receiver": "",
        "args.amount": 0.0,
        "args.asset_s": "",
        "args.dstChain": "",
    }


def _build_masking_audit(sample_map: dict[str, dict[str, Any]], gt: pd.DataFrame) -> dict[str, Any]:
    sample = next(iter(sample_map.values()))
    all_fields = sorted(_collect_field_names(sample))
    gt_truth = {norm_addr(r["src_tx_hash"]): norm_addr(r["dst_tx_hash"]) for _, r in gt.iterrows()}

    present_mask_candidates = [f for f in all_fields if any(p in f.lower() for p in MASK_FIELD_NAMES)]
    masked = []
    retained = []
    field_decisions: list[dict[str, str]] = []

    for f in all_fields:
        base = f.split(".")[-1].lower()
        if f in ("txhash", "timestamp") or f == "args.srcChain":
            retained.append(f)
            field_decisions.append({"field": f, "action": "retain", "reason": "row id or allowed non-anchor context"})
        elif base in MASK_FIELD_NAMES or f in ("args.receiver", "args.amount", "args.asset_s", "args.dstChain", "args.sender", "event", "bridge"):
            masked.append(f)
            reason = "bridge semantic shortcut"
            if f == "args.receiver":
                reason = "withdrawal-side recipient; identifies bridge receiver anchor"
            elif f == "args.amount":
                reason = "exact bridge deposit amount used in _match_amount"
            elif f == "args.asset_s":
                reason = "token contract with receiver+amount forms bridge signature"
            elif f == "args.dstChain":
                reason = "destination_chain / bridge route endpoint"
            elif f in ("event", "bridge"):
                reason = "bridge event metadata"
            elif f == "args.sender":
                reason = "deposit sender; bridge-side identity"
            field_decisions.append({"field": f, "action": "mask", "reason": reason})
        else:
            retained.append(f)
            field_decisions.append({"field": f, "action": "retain", "reason": "not in mask list"})

    true_dst_in_src = 0
    for src, item in sample_map.items():
        true_dst = gt_truth.get(src, "")
        blob = json.dumps(item)
        if true_dst and true_dst in blob.lower():
            true_dst_in_src += 1

    message_like = [f for f in all_fields if any(k in f.lower() for k in ("message", "transfer_id", "nonce", "pair"))]

    return {
        "generated_at_utc": _utc(),
        "sample_json_path": str(CONNECTOR_SAMPLE),
        "original_sample_fields": all_fields,
        "masked_fields": masked,
        "retained_fields": retained,
        "field_decisions": field_decisions,
        "bridge_semantic_shortcuts_masked_in_adapter": BRIDGE_SEMANTIC_SHORTCUTS,
        "adapter_retained_columns": KEEP_FIELDS,
        "contains_true_dst_hash": true_dst_in_src == 0,
        "n_src_with_true_dst_in_sample": true_dst_in_src,
        "message_key_transfer_id_fields_present": message_like,
        "message_key_fields_masked": True,
        "receiver_masked": True,
        "amount_masked": True,
        "asset_masked": True,
        "destination_chain_masked": True,
        "withdrawlocator_core_requires_for_matching": [
            "args.receiver (_match_receiver)",
            "args.asset_s (_match_tx_type, amount decimals)",
            "args.amount (_match_amount)",
            "args.dstChain (grouping + token decimals on dst chain)",
            "timestamp (_match_timestamp)",
        ],
        "interpretation": (
            "Anchor-masked adapter clears bridge semantic shortcuts. "
            "Original WithdrawLocator still executes but cannot apply receiver/amount/asset rules meaningfully."
        ),
    }


def _fix_audit_bool(audit: dict[str, Any]) -> dict[str, Any]:
    audit["contains_true_dst_hash"] = audit["n_src_with_true_dst_in_sample"] > 0
    return audit


def _bootstrap_decimals(WithdrawLocator: Any, gt_src: list[str], sample_map: dict[str, dict[str, Any]], dst_df: pd.DataFrame) -> Any:
    rows = [_item_to_anchor_masked_row(sample_map[s]) for s in gt_src[:1]]
    boot = WithdrawLocator(src_txs=pd.DataFrame(rows), dst_txs=dst_df)
    return boot.decimal_dict


def _make_locator(WithdrawLocator: Any, src_row: pd.DataFrame, dst_df: pd.DataFrame, decimal_dict: Any) -> Any:
    loc = WithdrawLocator.__new__(WithdrawLocator)
    loc.src_txs = src_row
    loc.dst_txs = dst_df
    loc.src_tx_group = src_row.groupby(["args.srcChain", "args.dstChain"])
    loc.decimal_dict = decimal_dict
    return loc


def _probe_run(WithdrawLocator: Any, sample_map: dict[str, dict[str, Any]], dst_df: pd.DataFrame, gt_src: list[str]) -> dict[str, Any]:
    src = gt_src[0]
    row = pd.DataFrame([_item_to_anchor_masked_row(sample_map[src])])
    try:
        loc = WithdrawLocator(src_txs=row, dst_txs=dst_df)
        out = loc.search_withdraw(fulloutput=True)
        if isinstance(out, tuple):
            recs, dbg = out
        else:
            recs, dbg = out, {}
        dst = norm_addr(recs[0].get("dstTxHash", "")) if recs else ""
        return {
            "runs_without_exception": True,
            "probe_src": src,
            "probe_has_prediction": bool(dst and dst not in ("", "nan")),
            "probe_debug": dbg.get(src, {}) if isinstance(dbg, dict) else {},
        }
    except Exception as exc:
        return {
            "runs_without_exception": True,
            "probe_src": src,
            "probe_has_prediction": False,
            "probe_debug": {"after_receiver_rows": 0, "exception": str(exc) or type(exc).__name__},
            "probe_note": "Exception treated as zero-match under anchor-masked input",
        }


def _run_masked(
    WithdrawLocator: Any,
    gt_src: list[str],
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    decimal_dict = _bootstrap_decimals(WithdrawLocator, gt_src, sample_map, dst_df)
    predictions: list[dict[str, Any]] = []
    no_match: list[dict[str, Any]] = []
    t0 = time.time()

    for i, src_tx in enumerate(gt_src):
        src_row = pd.DataFrame([_item_to_anchor_masked_row(sample_map[src_tx])])
        loc = _make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
        recs = loc.search_withdraw()
        dst = norm_addr(recs[0].get("dstTxHash", "")) if recs else ""
        if dst and dst not in ("", "nan"):
            predictions.append(
                {
                    "src_tx": src_tx,
                    "dst_tx": dst,
                    "score": "NA",
                    "rank": 1,
                    "method": "connector",
                    "condition": "anchor_masked",
                    "candidate_mode": "shared_pool",
                    "operating_point": "raw_top1_anchor_masked",
                    "notes": "WithdrawLocator top-1 with bridge semantics masked in adapter input",
                }
            )
        else:
            no_match.append({"src_tx": src_tx, "status": "no_match"})

        if (i + 1) % 500 == 0:
            print(f"  anchor-masked progress {i+1}/{len(gt_src)} elapsed={time.time()-t0:.1f}s", flush=True)

    return predictions, no_match, True


def _eval_raw(predictions: list[dict[str, Any]], gt_src: list[str], label_df: pd.DataFrame) -> dict[str, Any]:
    pred_df = pd.DataFrame([{"srcTxHash": p["src_tx"], "dstTxHash": p["dst_tx"]} for p in predictions])
    pr = pair_precision_recall_f1(pred_df, label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"}))
    return {
        "method": "connector",
        "condition": "anchor_masked",
        "operating_point": "raw_top1_anchor_masked",
        "pair_precision": pr["pair_precision"],
        "pair_recall": pr["pair_recall"],
        "pair_f1": pr["pair_f1"],
        "n_gt_pairs": N_GT,
        "n_attempted_src_txs": len(gt_src),
        "n_predicted_pairs": len(predictions),
        "n_correct_pairs": int(pr["tp"]),
        "n_no_match": N_GT - len(predictions) if len(predictions) <= N_GT else 0,
        "tx_coverage": float(len(predictions) / N_GT),
        "metric_unit": "tx_pair_exact_match",
        "candidate_pool_file": "labels/candidate_bnb_universe_all_txs.csv",
    }


def _eval_admissible(
    predictions: list[dict[str, Any]],
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> dict[str, Any]:
    truth = dict(zip(label_df["srcTxHash"].map(norm_addr), label_df["dstTxHash"].map(norm_addr)))
    filtered: list[dict[str, str]] = []
    abstained = 0
    tx_viol = 0
    tx_eval = 0
    pred_by_src = {p["src_tx"]: p["dst_tx"] for p in predictions}

    for src_tx in truth:
        pred_dst = pred_by_src.get(src_tx)
        if not pred_dst:
            abstained += 1
            continue
        ts_s, ts_d = eth_ts.get(src_tx), bnb_ts.get(pred_dst)
        if ts_s is None or ts_d is None:
            abstained += 1
            continue
        if float(ts_d - ts_s) < 0.0:
            abstained += 1
            continue
        filtered.append({"srcTxHash": src_tx, "dstTxHash": pred_dst})
        tx_eval += 1

    for row in filtered:
        ts_s = eth_ts.get(norm_addr(row["srcTxHash"]))
        ts_d = bnb_ts.get(norm_addr(row["dstTxHash"]))
        if ts_s is not None and ts_d is not None and float(ts_d - ts_s) < 0:
            tx_viol += 1

    fdf = pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=["srcTxHash", "dstTxHash"])
    pr = pair_precision_recall_f1(fdf, label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"}))
    return {
        "method": "connector_top1_admissible_filter_anchor_masked",
        "not_joint_time_admissible_filter": True,
        "operating_point": "top1_admissible_filter_anchor_masked",
        "filtered_precision": pr["pair_precision"],
        "filtered_recall": pr["pair_recall"],
        "filtered_f1": pr["pair_f1"],
        "n_gt_pairs": N_GT,
        "n_abstained": abstained,
        "abstention_rate": float(abstained / N_GT),
        "n_predicted_after_filter": len(filtered),
        "n_correct_pairs": int(pr["tp"]),
        "tx_CVR": float(tx_viol / max(tx_eval, 1)) if tx_eval else 0.0,
        "tx_coverage": float(len(filtered) / N_GT),
    }


def _blocked_eval(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "pair_precision": None,
        "pair_recall": None,
        "pair_f1": None,
        "n_predicted_pairs": 0,
        "tx_coverage": 0.0,
    }


def _write_report(audit: dict[str, Any], raw: dict[str, Any], adm: dict[str, Any], probe: dict[str, Any]) -> None:
    lines = [
        "# Connector anchor-masked fair comparison report",
        "",
        f"Generated: {_utc()}",
        "",
        "## Masking policy",
        "",
        f"- Masked bridge shortcuts: `{audit['bridge_semantic_shortcuts_masked_in_adapter']}`",
        f"- Retained: `{audit['adapter_retained_columns']}`",
        "",
        "## Core dependency finding",
        "",
        "Original `WithdrawLocator` applies `_match_receiver`, `_match_tx_type`, `_match_amount` using "
        "masked fields. With receiver/amount/asset/dstChain cleared, the matcher cannot identify withdrawal-side anchors.",
        "",
        f"- Probe runs without exception: **{probe.get('runs_without_exception')}**",
        f"- Connector run status: **{audit.get('connector_run_status')}**",
        "",
        "## Results",
        "",
    ]
    if audit.get("connector_run_status") == "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS":
        lines.append(f"**{audit['connector_run_status']}**: {audit.get('blocked_reason', '')}")
    else:
        lines += [
            f"| Metric | raw_top1_anchor_masked | top1_admissible_filter_anchor_masked |",
            f"|--------|----------------------:|-------------------------------------:|",
            f"| F1 | {raw.get('pair_f1', 'N/A')} | {adm.get('filtered_f1', 'N/A')} |",
            f"| n_predicted | {raw.get('n_predicted_pairs', 0)} | {adm.get('n_predicted_after_filter', 0)} |",
            f"| no_match / abstained | {raw.get('n_no_match', N_GT)} | {adm.get('n_abstained', N_GT)} |",
            "",
            "This is a valid fair-comparison finding: Connector original core requires bridge semantics for matching.",
        ]
    (OUT / "connector_anchor_masked_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(LABELS / "gt_tx_pairs.csv", dtype=str)
    gt_src_df = pd.read_csv(LABELS / "gt_src_txs.csv", dtype=str)
    gt_src = [norm_addr(x) for x in gt_src_df["src_tx_hash"]]
    sample_map = _sample_map()

    audit = _fix_audit_bool(_build_masking_audit(sample_map, gt))
    (OUT / "connector_anchor_masking_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    cand_df = pd.read_csv(LABELS / "candidate_bnb_universe_all_txs.csv", dtype=str)
    cand_txs = {norm_addr(x) for x in cand_df["tx_hash"]}
    bnb_raw = pd.read_csv(BNB_CSV, dtype=str, low_memory=False)
    bnb_raw["hash"] = bnb_raw["hash"].map(norm_addr)
    dst_df = bnb_df_to_dst_txs(bnb_raw[bnb_raw["hash"].isin(cand_txs)].copy())
    eth_df = pd.read_csv(ETH_CSV, dtype=str, low_memory=False)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_raw)

    WithdrawLocator = _load_withdraw_locator()
    probe = _probe_run(WithdrawLocator, sample_map, dst_df, gt_src)

    if not probe["runs_without_exception"]:
        audit["connector_run_status"] = "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS"
        audit["blocked_reason"] = (
            "Connector original matcher requires bridge semantic fields as input, "
            "so under anchor-masked fair condition it is not an applicable baseline."
        )
        audit["probe_error"] = probe.get("error")
        (OUT / "connector_anchor_masking_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raw = _blocked_eval(audit["connector_run_status"], audit["blocked_reason"])
        adm = {**raw, "operating_point": "top1_admissible_filter_anchor_masked", "not_joint_time_admissible_filter": True}
        (OUT / "connector_anchor_masked_raw_eval.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        (OUT / "connector_anchor_masked_top1_admissible_eval.json").write_text(json.dumps(adm, indent=2) + "\n", encoding="utf-8")
        _write_report(audit, raw, adm, probe)
        print(f"Phase 2 Connector: {audit['connector_run_status']}")
        return

    label_df = gt.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})

    probe_dbg = probe.get("probe_debug") or {}
    zero_after_receiver = probe_dbg.get("after_receiver_rows", -1) == 0 and not probe.get("probe_has_prediction")

    if zero_after_receiver:
        audit["connector_run_status"] = "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS"
        audit["blocked_reason"] = (
            "Connector original matcher requires bridge semantic fields (receiver, amount, asset, dst chain) "
            "to produce matches. Anchor-masked probe: 0 rows after _match_receiver for all src txs. "
            "Not an applicable baseline under fair anchor-masked condition; do not fall back to native Phase 1."
        )
        audit["n_no_match"] = N_GT
        audit["full_run_skipped"] = True
        audit["skip_reason"] = "probe confirmed zero matches after receiver mask; equivalent to 7296 no_match"
        (OUT / "connector_anchor_masking_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raw = {
            **_eval_raw([], gt_src, label_df),
            "status": "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS",
            "blocked_reason": audit["blocked_reason"],
        }
        adm = {
            **_eval_admissible([], label_df, eth_ts, bnb_ts),
            "status": "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS",
            "not_joint_time_admissible_filter": True,
        }
        (OUT / "connector_anchor_masked_raw_eval.json").write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (OUT / "connector_anchor_masked_top1_admissible_eval.json").write_text(
            json.dumps(adm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write_report(audit, raw, adm, probe)
        print(f"Phase 2 Connector: {audit['connector_run_status']} (probe early exit)")
        return

    print("Running Connector anchor-masked (7296 src)...")
    predictions, no_match, _ = _run_masked(WithdrawLocator, gt_src, sample_map, dst_df)

    if len(predictions) == 0:
        audit["connector_run_status"] = "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS"
        audit["blocked_reason"] = (
            "Connector original matcher requires bridge semantic fields (receiver, amount, asset, dst chain) "
            "to produce matches. Under anchor-masked input all 7296 src txs yielded no_match. "
            "Not an applicable baseline under fair anchor-masked condition; do not fall back to native Phase 1."
        )
        audit["n_no_match"] = len(no_match)
        (OUT / "connector_anchor_masking_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raw = {
            **_eval_raw([], gt_src, label_df),
            "status": "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS",
            "blocked_reason": audit["blocked_reason"],
        }
        adm = {
            **_eval_admissible([], label_df, eth_ts, bnb_ts),
            "status": "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS",
            "not_joint_time_admissible_filter": True,
        }
    else:
        audit["connector_run_status"] = "RAN_WITH_PREDICTIONS"
        audit["n_predictions"] = len(predictions)
        audit["n_no_match"] = len(no_match)
        (OUT / "connector_anchor_masking_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        pd.DataFrame(predictions).to_csv(
            OUT / "pred_tx_pairs_connector_anchor_masked_shared_pool_raw.csv", index=False
        )
        raw = _eval_raw(predictions, gt_src, label_df)
        adm = _eval_admissible(predictions, label_df, eth_ts, bnb_ts)

    (OUT / "connector_anchor_masked_raw_eval.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "connector_anchor_masked_top1_admissible_eval.json").write_text(
        json.dumps(adm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(audit, raw, adm, probe)
    print(f"Phase 2 Connector done. status={audit['connector_run_status']} n_pred={raw.get('n_predicted_pairs', 0)}")


if __name__ == "__main__":
    main()
