#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 1: Connector native_features + shared_pool raw top-1 (WithdrawLocator core only).

Outputs under cross/out/baseline_compare/connector_phase1/ only.
Does not modify Connector core or frozen RC-UOT-Q artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
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
OUT = BASE / "connector_phase1"

CONNECTOR_ROOT = REPO.parent / "Connector" / "Connector-main"
CONNECTOR_DST_CHAIN = CONNECTOR_ROOT / "core" / "dst_chain.py"
CONNECTOR_SAMPLE = CONNECTOR_ROOT / "data" / "Validation" / "ETH-BNB" / "Celer" / "sample.json"

ETH_CSV = REPO / "in" / "Celer_ETH_cun.csv"
BNB_CSV = REPO / "label" / "tx" / "Celer_BNB_qu.csv"
N_SRC_FLOWS = 3258
N_GT = 7296


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_connector_withdraw_locator():
    if str(CONNECTOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CONNECTOR_ROOT))
    from core.dst_chain import WithdrawLocator  # type: ignore

    return WithdrawLocator


def _sample_by_txhash() -> dict[str, dict[str, Any]]:
    items = json.loads(CONNECTOR_SAMPLE.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        h = norm_addr(item.get("txhash", ""))
        if h:
            out[h] = item
    return out


def _item_to_src_row(item: dict[str, Any]) -> dict[str, Any]:
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


def _field_audit(
    sample_map: dict[str, dict[str, Any]],
    gt_src: list[str],
    dst_df: pd.DataFrame,
    cand_txs: set[str],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
) -> dict[str, Any]:
    required_src = [
        "txhash",
        "timestamp",
        "args.receiver",
        "args.amount",
        "args.asset_s",
        "args.srcChain",
        "args.dstChain",
    ]
    required_dst = ["hash", "to", "value", "timeStamp", "contractAddress"]

    src_missing_sample: list[str] = []
    src_field_gaps: list[dict[str, Any]] = []
    for tx in gt_src:
        item = sample_map.get(tx)
        if item is None:
            src_missing_sample.append(tx)
            continue
        row = _item_to_src_row(item)
        gaps = [f for f in required_src if f not in row or row[f] in ("", None)]
        if row.get("timestamp") == 0.0:
            gaps.append("timestamp_zero")
        if gaps:
            src_field_gaps.append({"src_tx": tx, "missing_or_empty": gaps})

    dst_present = set(dst_df["hash"].astype(str).map(norm_addr))
    cand_missing_in_bnb = sorted(cand_txs - dst_present)
    dst_rows_per_hash = dst_df.groupby("hash").size().to_dict()

    ts_audit = {
        "gt_src_missing_eth_timestamp": [t for t in gt_src if t not in eth_ts],
        "candidate_dst_missing_bnb_timestamp": sorted(c for c in cand_txs if c not in bnb_ts),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "src_tx_source": {
            "primary": str(CONNECTOR_SAMPLE.relative_to(REPO.parent)),
            "description": "Connector native Validation sample (deposit args); not DepositLocator pipeline",
            "fields_used": required_src,
            "args_asset_d_present_in_eth_bnb_sample": False,
            "note": "ETH-BNB sample has no args.asset_d; WithdrawLocator _match_asset_type skipped when column absent (original behavior)",
        },
        "dst_tx_source": {
            "primary": str(BNB_CSV.relative_to(REPO)),
            "filtered_by": "labels/candidate_bnb_universe_all_txs.csv",
            "fields_used": required_dst,
            "adapter": "cross.shared.transfers.bnb_df_to_dst_txs (I/O only)",
        },
        "gt_src_in_sample": len(gt_src) - len(src_missing_sample),
        "gt_src_missing_from_sample": len(src_missing_sample),
        "src_field_gaps_count": len(src_field_gaps),
        "src_field_gaps_examples": src_field_gaps[:10],
        "candidate_pool_unique_dst_txs": len(cand_txs),
        "candidate_dst_present_in_bnb_csv": len(cand_txs) - len(cand_missing_in_bnb),
        "candidate_dst_missing_in_bnb_csv": cand_missing_in_bnb[:20],
        "n_candidate_dst_missing_in_bnb_csv": len(cand_missing_in_bnb),
        "dst_rows_per_hash_min": int(min(dst_rows_per_hash.values())) if dst_rows_per_hash else 0,
        "dst_rows_per_hash_max": int(max(dst_rows_per_hash.values())) if dst_rows_per_hash else 0,
        "timestamp_audit": ts_audit,
        "delay_table_source": {
            "eth_csv": str(ETH_CSV.relative_to(REPO)),
            "bnb_csv": str(BNB_CSV.relative_to(REPO)),
            "lookup_function": "cross.application.experiments.delay_semantics_utils._tx_timestamp_lookup",
            "same_as_admissible_decoding": True,
            "recomputed_uot_delay": False,
        },
        "blocking_issues": [],
    }


def _bootstrap_decimal_dict(WithdrawLocator: Any, sample_map: dict[str, dict[str, Any]], gt_src: list[str], dst_df: pd.DataFrame) -> Any:
    """Load token decimals once for all unique src assets (adapter-only; core unchanged)."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for tx in gt_src:
        asset = str((sample_map[tx].get("args") or {}).get("asset_s", "") or "").strip().lower()
        if asset in seen:
            continue
        seen.add(asset)
        rows.append(_item_to_src_row(sample_map[tx]))
    boot = WithdrawLocator(src_txs=pd.DataFrame(rows), dst_txs=dst_df)
    return boot.decimal_dict


def _make_locator(
    WithdrawLocator: Any,
    src_row: pd.DataFrame,
    dst_df: pd.DataFrame,
    decimal_dict: Any,
) -> Any:
    """Construct WithdrawLocator without repeated Config()/token CSV scans."""
    loc = WithdrawLocator.__new__(WithdrawLocator)
    loc.src_txs = src_row
    loc.dst_txs = dst_df
    loc.src_tx_group = src_row.groupby(["args.srcChain", "args.dstChain"])
    loc.decimal_dict = decimal_dict
    return loc


def _run_connector(
    WithdrawLocator: Any,
    gt_src: list[str],
    sample_map: dict[str, dict[str, Any]],
    dst_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    no_match: list[dict[str, Any]] = []
    decimal_dict = _bootstrap_decimal_dict(WithdrawLocator, sample_map, gt_src, dst_df)
    t0 = time.time()

    for i, src_tx in enumerate(gt_src):
        item = sample_map[src_tx]
        src_row = pd.DataFrame([_item_to_src_row(item)])
        locator = _make_locator(WithdrawLocator, src_row, dst_df, decimal_dict)
        recs = locator.search_withdraw()
        dst = ""
        if recs:
            dst = norm_addr(recs[0].get("dstTxHash", ""))
        if dst and dst not in ("", "nan"):
            predictions.append(
                {
                    "src_tx": src_tx,
                    "dst_tx": dst,
                    "score": "NA",
                    "rank": 1,
                    "method": "connector",
                    "condition": "native_features",
                    "candidate_mode": "shared_pool",
                    "notes": "WithdrawLocator.search_withdraw top-1; no numeric score in original core",
                }
            )
        else:
            no_match.append({"src_tx": src_tx, "status": "no_match"})

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  connector progress {i+1}/{len(gt_src)} elapsed={elapsed:.1f}s", flush=True)

    return predictions, no_match


def _label_df_from_gt() -> pd.DataFrame:
    gt = pd.read_csv(LABELS / "gt_tx_pairs.csv", dtype=str, keep_default_na=False)
    return gt.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})


def _eval_raw(
    predictions: list[dict[str, Any]],
    label_df: pd.DataFrame,
    gt_src: list[str],
    tx_to_src_flow: dict[str, str],
) -> dict[str, Any]:
    pred_df = pd.DataFrame(
        [{"srcTxHash": p["src_tx"], "dstTxHash": p["dst_tx"]} for p in predictions]
    )
    pr = pair_precision_recall_f1(pred_df, label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"}))

    covered_flows = {tx_to_src_flow[p["src_tx"]] for p in predictions if p["src_tx"] in tx_to_src_flow}

    return {
        "method": "connector",
        "condition": "native_features",
        "candidate_mode": "shared_pool",
        "pair_precision": pr["pair_precision"],
        "pair_recall": pr["pair_recall"],
        "pair_f1": pr["pair_f1"],
        "n_gt_pairs": N_GT,
        "n_attempted_src_txs": len(gt_src),
        "n_predicted_pairs": len(predictions),
        "n_correct_pairs": int(pr["tp"]),
        "n_false_positive": int(pr["fp"]),
        "n_false_negative_no_prediction": int(pr["fn"]),
        "tx_coverage": float(len(predictions) / N_GT),
        "src_flow_coverage": float(len(covered_flows) / N_SRC_FLOWS),
        "n_covered_src_flows": len(covered_flows),
        "candidate_pool_unique_dst_txs": 7296,
        "candidate_pool_file": "labels/candidate_bnb_universe_all_txs.csv",
        "closed_set_warning": True,
        "metric_unit": "tx_pair_exact_match",
        "flow_pair_projection_used": False,
    }


def _eval_top1_admissible(
    predictions: list[dict[str, Any]],
    label_df: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    tx_to_src_flow: dict[str, str],
) -> dict[str, Any]:
    truth = dict(
        zip(
            label_df["srcTxHash"].map(norm_addr),
            label_df["dstTxHash"].map(norm_addr),
        )
    )
    filtered: list[dict[str, str]] = []
    abstained = 0
    tx_viol = 0
    tx_eval = 0

    pred_by_src = {p["src_tx"]: p["dst_tx"] for p in predictions}

    for src_tx, gt_dst in truth.items():
        pred_dst = pred_by_src.get(src_tx)
        if not pred_dst:
            abstained += 1
            continue
        ts_s = eth_ts.get(src_tx)
        ts_d = bnb_ts.get(pred_dst)
        if ts_s is None or ts_d is None:
            abstained += 1
            continue
        delay = float(ts_d - ts_s)
        if delay < 0.0:
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
    pr = pair_precision_recall_f1(
        fdf,
        label_df.rename(columns={"srcTxHash": "srcTxhash", "dstTxHash": "dstTxhash"}),
    )
    covered_flows = {
        tx_to_src_flow[norm_addr(r["srcTxHash"])]
        for _, r in fdf.iterrows()
        if norm_addr(r["srcTxHash"]) in tx_to_src_flow
    }

    return {
        "method": "connector_top1_admissible_filter",
        "not_rc_uot_q_joint_time_admissible_filter": True,
        "not_positive_delay_top3_rescue": True,
        "filter_rule": "Keep Connector raw top-1 iff eth/bnb timestamps exist and delay_sec >= 0; no rerank/rescue/fallback",
        "delay_table_source": "same as admissible_decoding (_tx_timestamp_lookup on frozen ETH/BNB CSV)",
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
        "src_flow_coverage": float(len(covered_flows) / N_SRC_FLOWS),
        "n_covered_src_flows": len(covered_flows),
    }


def _write_report(raw_eval: dict[str, Any], adm_eval: dict[str, Any], audit: dict[str, Any], n_no_match: int) -> None:
    lines = [
        "# Connector Phase 1 report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Disclosure",
        "",
        "- **Connector original matcher core** (`WithdrawLocator` in `Connector-main/core/dst_chain.py`)",
        "- **Frozen closed-set universe** — not online BridgeSpider pipeline",
        "- **No DepositLocator** — labeled src txs fed as native Validation deposit features",
        "- **Candidate pool:** `candidate_bnb_universe_all_txs.csv` (7296 unique dst txs)",
        "",
        "## Field audit summary",
        "",
        f"- GT src txs in Connector sample.json: **{audit['gt_src_in_sample']}/{N_GT}**",
        f"- Candidate dst txs in BNB CSV: **{audit['candidate_dst_present_in_bnb_csv']}/{audit['candidate_pool_unique_dst_txs']}**",
        f"- ETH-BNB sample lacks `args.asset_d` (original `_match_asset_type` skip): documented",
        "",
        "## Raw top-1 (`native_features` + `shared_pool`)",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| pair precision | {raw_eval['pair_precision']:.6f} |",
        f"| pair recall | {raw_eval['pair_recall']:.6f} |",
        f"| pair F1 | {raw_eval['pair_f1']:.6f} |",
        f"| n_predicted | {raw_eval['n_predicted_pairs']} |",
        f"| n_no_match | {n_no_match} |",
        f"| tx_coverage | {raw_eval['tx_coverage']:.6f} |",
        f"| src_flow_coverage | {raw_eval['src_flow_coverage']:.6f} |",
        "",
        "## Diagnostic: `connector_top1_admissible_filter`",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| filtered precision | {adm_eval['filtered_precision']:.6f} |",
        f"| filtered recall | {adm_eval['filtered_recall']:.6f} |",
        f"| filtered F1 | {adm_eval['filtered_f1']:.6f} |",
        f"| abstention_rate | {adm_eval['abstention_rate']:.6f} |",
        f"| tx_CVR | {adm_eval['tx_CVR']:.6f} |",
        "",
        "## Limitations",
        "",
        "- **No top-k:** `search_withdraw()` returns top-1 only → top3 / RC-UOT-Q joint **N/A**",
        "- **Closed-set pool:** candidate tx hashes equal GT dst set (7296); not open-world discovery",
        "- **ABCTracer:** BLOCKED (no checkpoint) — not run in Phase 1",
        "",
        "## RC-UOT-Q reference (frozen, not recomputed)",
        "",
        "See `../rc_uot_q_frozen/rc_uot_q_reference_metrics.json` for headline RC-UOT-Q rows.",
        "",
    ]
    (OUT / "connector_phase1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(__file__)

    gt_src_df = pd.read_csv(LABELS / "gt_src_txs.csv", dtype=str)
    gt_src = [norm_addr(x) for x in gt_src_df["src_tx_hash"]]

    cand_df = pd.read_csv(LABELS / "candidate_bnb_universe_all_txs.csv", dtype=str)
    cand_txs = {norm_addr(x) for x in cand_df["tx_hash"]}

    sample_map = _sample_by_txhash()
    missing = [t for t in gt_src if t not in sample_map]
    if missing:
        raise RuntimeError(f"GT src txs missing from Connector sample.json: {len(missing)}")

    bnb_raw = pd.read_csv(BNB_CSV, dtype=str, low_memory=False)
    bnb_raw["hash"] = bnb_raw["hash"].map(norm_addr)
    bnb_filt = bnb_raw[bnb_raw["hash"].isin(cand_txs)].copy()
    dst_df = bnb_df_to_dst_txs(bnb_filt)

    eth_df = pd.read_csv(ETH_CSV, dtype=str, low_memory=False)
    eth_ts, bnb_ts, ts_missing = _tx_timestamp_lookup(eth_df, bnb_raw)

    audit = _field_audit(sample_map, gt_src, dst_df, cand_txs, eth_ts, bnb_ts)
    if audit["gt_src_missing_from_sample"] or audit["n_candidate_dst_missing_in_bnb_csv"]:
        audit["blocking_issues"].append("missing native or BNB CSV rows for evaluation universe")
    (OUT / "connector_input_field_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if audit["blocking_issues"]:
        raise RuntimeError(f"Field audit blocking issues: {audit['blocking_issues']}")

    tx_map = pd.read_csv(LABELS / "tx_to_flow_map_lao.csv", dtype=str)
    eth_map = tx_map[tx_map["chain"].str.upper() == "ETH"]
    tx_to_src_flow = dict(zip(eth_map["tx_hash"].map(norm_addr), eth_map["flow_id"]))

    WithdrawLocator = _load_connector_withdraw_locator()
    print("Running Connector WithdrawLocator (7296 src x shared_pool)...")
    predictions, no_match = _run_connector(WithdrawLocator, gt_src, sample_map, dst_df)

    pred_path = OUT / "pred_tx_pairs_connector_native_shared_pool_raw.csv"
    pd.DataFrame(predictions).to_csv(pred_path, index=False)
    pd.DataFrame(no_match).to_csv(OUT / "connector_no_match_src_txs.csv", index=False)

    label_df = _label_df_from_gt()
    raw_eval = _eval_raw(predictions, label_df, gt_src, tx_to_src_flow)
    (OUT / "connector_raw_eval.json").write_text(
        json.dumps(raw_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    adm_eval = _eval_top1_admissible(predictions, label_df, eth_ts, bnb_ts, tx_to_src_flow)
    (OUT / "connector_top1_admissible_eval.json").write_text(
        json.dumps(adm_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    cmd = f'python "{adapter_path}"'
    manifest = {
        "phase": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "connector_source_path": str(CONNECTOR_ROOT),
        "connector_core_file": str(CONNECTOR_DST_CHAIN.relative_to(REPO.parent)),
        "connector_core_sha256": _sha256(CONNECTOR_DST_CHAIN),
        "connector_core_modified": False,
        "adapter_script": str(adapter_path.relative_to(REPO)),
        "adapter_sha256": _sha256(adapter_path),
        "disclosure": {
            "mode": "Connector original WithdrawLocator matcher core over frozen closed-set universe",
            "online_bridge_spider": False,
            "deposit_locator_used": False,
            "connector_source_modified": False,
        },
        "candidate_pool_file": "labels/candidate_bnb_universe_all_txs.csv",
        "not_gt_dst_txs_as_pool": True,
        "closed_set_warning": True,
        "no_topk_warning": True,
        "connector_top3_status": "N/A",
        "joint_rc_uot_q_parity_status": "N/A",
        "abctracer_status": "BLOCKED",
        "abctracer_reason": "no official wgt.pth checkpoint",
        "input_sha256": {
            str(LABELS / "gt_tx_pairs.csv").replace("\\", "/"): _sha256(LABELS / "gt_tx_pairs.csv"),
            str(LABELS / "gt_src_txs.csv").replace("\\", "/"): _sha256(LABELS / "gt_src_txs.csv"),
            str(LABELS / "candidate_bnb_universe_all_txs.csv").replace("\\", "/"): _sha256(
                LABELS / "candidate_bnb_universe_all_txs.csv"
            ),
            str(CONNECTOR_SAMPLE.relative_to(REPO.parent)).replace("\\", "/"): _sha256(CONNECTOR_SAMPLE),
            str(ETH_CSV.relative_to(REPO)): _sha256(ETH_CSV),
            str(BNB_CSV.relative_to(REPO)): _sha256(BNB_CSV),
        },
        "command": cmd,
        "python": sys.version,
        "platform": platform.platform(),
        "n_predictions": len(predictions),
        "n_no_match": len(no_match),
    }
    (OUT / "connector_phase1_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _write_report(raw_eval, adm_eval, audit, len(no_match))

    # Update root manifest pointer
    root_manifest_path = BASE / "manifest.json"
    if root_manifest_path.is_file():
        root = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        root["phase1_connector"] = {
            "generated_at_utc": manifest["generated_at_utc"],
            "output_dir": "connector_phase1",
            "raw_pair_f1": raw_eval["pair_f1"],
            "top1_admissible_f1": adm_eval["filtered_f1"],
        }
        root_manifest_path.write_text(json.dumps(root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Phase 1 complete. raw F1={raw_eval['pair_f1']:.6f} n_pred={len(predictions)} n_no_match={len(no_match)}")


if __name__ == "__main__":
    main()
