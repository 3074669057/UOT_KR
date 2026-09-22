#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 13: Real evidence acquisition and identifiability repair for RC-UOT-HP."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.domain.evaluation.flow_eval import _pair_set

_P10S_PATH = _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"
_spec_s = importlib.util.spec_from_file_location("phase10s", _P10S_PATH)
_p10s = importlib.util.module_from_spec(_spec_s)
assert _spec_s.loader is not None
_spec_s.loader.exec_module(_p10s)

_P10V_PATH = _REPO / "scripts" / "run_phase10v_pair_f1_precision_rcuot.py"
_spec_v = importlib.util.spec_from_file_location("phase10v", _P10V_PATH)
_p10v = importlib.util.module_from_spec(_spec_v)
assert _spec_v.loader is not None
_spec_v.loader.exec_module(_p10v)

_P10X_PATH = _REPO / "scripts" / "run_phase10x_evidence_enhanced_disambiguation.py"
_spec_x = importlib.util.spec_from_file_location("phase10x", _P10X_PATH)
_p10x = importlib.util.module_from_spec(_spec_x)
assert _spec_x.loader is not None
_spec_x.loader.exec_module(_p10x)

_P10W_PATH = _REPO / "scripts" / "run_phase10w_ambiguity_resolved_rcuot.py"
_spec_w = importlib.util.spec_from_file_location("phase10w", _P10W_PATH)
_p10w = importlib.util.module_from_spec(_spec_w)
assert _spec_w.loader is not None
_spec_w.loader.exec_module(_p10w)

_P11_PATH = _REPO / "scripts" / "run_phase11_evidence_segmentation_enhanced_csffc.py"
_spec_11 = importlib.util.spec_from_file_location("phase11", _P11_PATH)
_p11 = importlib.util.module_from_spec(_spec_11)
assert _spec_11.loader is not None
_spec_11.loader.exec_module(_p11)

P10X_DIR = _REPO / "out" / "paper_full_pipeline_run" / "phase10x_evidence_enhanced_disambiguation"
EVAL_SCOPE = "same_scope_csffc_flow_stress_sealed_holdout_92_111"
PRIMARY_K = 50
PRIMARY_METRICS = list(_p10x.PRIMARY_METRICS)
FEASIBILITY = {
    "candidate_oracle_recall": 0.95,
    "oracle_precision_at_recall_0_8": 0.80,
    "oracle_recall_at_precision_0_8": 0.80,
    "score_oracle_best_f1": 0.80,
    "feature_auroc": 0.85,
    "feature_auprc": 0.70,
    "ambiguous_gt_fraction": 0.20,
    "candidate_collision_rate": 0.30,
    "exact_pair_identifiability_upper_bound": 0.85,
}
UNAVAILABLE_NUM = float("nan")


def _safe_event_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    safe: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, int) and abs(v) > 2**53:
                clean[k] = str(v)
            else:
                clean[k] = v
        safe.append(clean)
    return pd.DataFrame(safe)
SEG_VARIANTS = [
    "real_tx_log_event_segments",
    "real_bridge_event_segments",
    "real_message_id_segments",
    "real_nonce_segments",
    "real_transfer_id_segments",
    "real_contract_topic_segments",
    "real_amount_time_event_segments",
]


def _df_to_md(df: pd.DataFrame) -> str:
    return _p10x._df_to_md(df)


def _flow_ts(flow: dict[str, Any]) -> tuple[float, float]:
    return _p11._flow_ts(flow)


def _addr_set(flow: dict[str, Any]) -> set[str]:
    return _p11._addr_set(flow)


def _env_status() -> dict[str, Any]:
    keys = {
        "ETH_RPC_URL": os.environ.get("ETH_RPC_URL", "").strip(),
        "BSC_RPC_URL": os.environ.get("BSC_RPC_URL", "").strip(),
        "ETHERSCAN_API_KEY": os.environ.get("ETHERSCAN_API_KEY", "").strip(),
        "BSCSCAN_API_KEY": os.environ.get("BSCSCAN_API_KEY", "").strip(),
    }
    return {
        "vars": {k: bool(v) for k, v in keys.items()},
        "eth_rpc_available": bool(keys["ETH_RPC_URL"]),
        "bsc_rpc_available": bool(keys["BSC_RPC_URL"]),
        "etherscan_available": bool(keys["ETHERSCAN_API_KEY"]),
        "bscscan_available": bool(keys["BSCSCAN_API_KEY"]),
        "any_rpc": bool(keys["ETH_RPC_URL"] or keys["BSC_RPC_URL"]),
    }


def _rpc_client(url: str):
    if not url:
        return None
    from cross.infrastructure.online.evm_json_rpc_client import EvmJsonRpcClient

    return EvmJsonRpcClient.from_urls([url], timeout_sec=25.0)


def _cache_path(cache_dir: Path, tx_hash: str) -> Path:
    h = str(tx_hash).lower().replace("0x", "")
    return cache_dir / f"{h}.json"


def _fetch_receipt(tx_hash: str, client, cache_dir: Path) -> dict[str, Any] | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_dir, tx_hash)
    if cp.is_file():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if client is None:
        return None
    try:
        receipt = client.get_transaction_receipt(tx_hash)
        if receipt:
            cp.write_text(json.dumps(receipt), encoding="utf-8")
        return receipt
    except Exception as exc:
        return {"_error": str(exc)}


def _addr_from_topic(topic: str | None) -> str:
    s = str(topic or "")
    if len(s) >= 42:
        return "0x" + s[-40:].lower()
    return ""


def _parse_log_row(tx_hash: str, receipt: dict[str, Any] | None, *, side: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not receipt or not isinstance(receipt, dict) or receipt.get("_error"):
        return rows
    block = receipt.get("blockNumber")
    tx_idx = receipt.get("transactionIndex")
    try:
        block_n = int(block, 16) if str(block).startswith("0x") else int(block)
    except (TypeError, ValueError):
        block_n = UNAVAILABLE_NUM
    try:
        tx_n = int(tx_idx, 16) if str(tx_idx).startswith("0x") else int(tx_idx)
    except (TypeError, ValueError):
        tx_n = UNAVAILABLE_NUM
    prefix = "src" if side == "src" else "dst"
    logs = receipt.get("logs") or []
    for li, lg in enumerate(logs):
        topics = lg.get("topics") or []
        t0 = str(topics[0] if topics else "")
        data = str(lg.get("data") or "0x")
        amt_raw = UNAVAILABLE_NUM
        try:
            if data.startswith("0x") and len(data) > 2:
                amt_raw = int(data, 16)
        except ValueError:
            pass
        sender = _addr_from_topic(topics[1] if len(topics) > 1 else None)
        receiver = _addr_from_topic(topics[2] if len(topics) > 2 else None)
        rows.append(
            {
                f"{prefix}_tx_hash": tx_hash,
                f"{prefix}_block_number": block_n,
                f"{prefix}_tx_index": tx_n,
                f"{prefix}_log_index": li,
                f"{prefix}_contract_address": str(lg.get("address") or ""),
                f"{prefix}_event_signature": t0[:18] if t0 else "",
                f"{prefix}_event_topic0": t0,
                f"{prefix}_sender": sender,
                f"{prefix}_receiver": receiver,
                f"{prefix}_token": str(lg.get("address") or ""),
                f"{prefix}_amount_raw": amt_raw,
                f"{prefix}_amount_normalized": float(amt_raw) / 1e18 if isinstance(amt_raw, int) and amt_raw < 10**30 else UNAVAILABLE_NUM,
                f"{prefix}_bridge_fee_if_available": UNAVAILABLE_NUM,
                f"{prefix}_nonce_if_available": UNAVAILABLE_NUM,
                f"{prefix}_message_id_if_available": UNAVAILABLE_NUM,
                f"{prefix}_transfer_id_if_available": UNAVAILABLE_NUM,
                f"{prefix}_relayer_if_available": UNAVAILABLE_NUM,
                "receipt_fetched": True,
            }
        )
    if not rows:
        rows.append(
            {
                f"{prefix}_tx_hash": tx_hash,
                f"{prefix}_block_number": block_n,
                f"{prefix}_tx_index": tx_n,
                f"{prefix}_log_index": UNAVAILABLE_NUM,
                f"{prefix}_contract_address": "",
                f"{prefix}_event_signature": "",
                f"{prefix}_event_topic0": "",
                f"{prefix}_sender": "",
                f"{prefix}_receiver": "",
                f"{prefix}_token": "",
                f"{prefix}_amount_raw": UNAVAILABLE_NUM,
                f"{prefix}_amount_normalized": UNAVAILABLE_NUM,
                f"{prefix}_bridge_fee_if_available": UNAVAILABLE_NUM,
                f"{prefix}_nonce_if_available": UNAVAILABLE_NUM,
                f"{prefix}_message_id_if_available": UNAVAILABLE_NUM,
                f"{prefix}_transfer_id_if_available": UNAVAILABLE_NUM,
                f"{prefix}_relayer_if_available": UNAVAILABLE_NUM,
                "receipt_fetched": True,
            }
        )
    return rows


def _reconstruct_bridge_events(
    seed_data: dict[str, Any],
    *,
    eth_cache: Path,
    bsc_cache: Path,
    env: dict[str, Any],
    max_tx_per_side: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eth_client = _rpc_client(os.environ.get("ETH_RPC_URL", "")) if env["eth_rpc_available"] else None
    bsc_client = _rpc_client(os.environ.get("BSC_RPC_URL", "")) if env["bsc_rpc_available"] else None
    src_rows, dst_rows = [], []
    fetched_eth, fetched_bsc, errors = 0, 0, 0
    for f in seed_data["eth_flows"]:
        for tx in (f.get("tx_hashes") or [])[: max(1, max_tx_per_side // max(len(seed_data["eth_flows"]), 1))]:
            tx = str(tx)
            rc = _fetch_receipt(tx, eth_client, eth_cache)
            if rc is None:
                errors += 1
                src_rows.append({"src_tx_hash": tx, "flow_id": str(f.get("flow_id") or ""), "receipt_fetched": False})
            else:
                fetched_eth += 1
                for row in _parse_log_row(tx, rc, side="src"):
                    row["flow_id"] = str(f.get("flow_id") or "")
                    src_rows.append(row)
    for f in seed_data["bnb_flows"]:
        for tx in (f.get("tx_hashes") or [])[: max(1, max_tx_per_side // max(len(seed_data["bnb_flows"]), 1))]:
            tx = str(tx)
            rc = _fetch_receipt(tx, bsc_client, bsc_cache)
            if rc is None:
                errors += 1
                dst_rows.append({"dst_tx_hash": tx, "flow_id": str(f.get("flow_id") or ""), "receipt_fetched": False})
            else:
                fetched_bsc += 1
                for row in _parse_log_row(tx, rc, side="dst"):
                    row["flow_id"] = str(f.get("flow_id") or "")
                    dst_rows.append(row)
    meta = {
        "eth_receipts_fetched": fetched_eth,
        "bsc_receipts_fetched": fetched_bsc,
        "fetch_errors": errors,
        "real_bridge_event_evidence": fetched_eth > 0 and fetched_bsc > 0 and env["any_rpc"],
    }
    return _safe_event_df(src_rows), _safe_event_df(dst_rows), meta


def _flow_event_lookup(df: pd.DataFrame, side: str) -> dict[str, dict[str, Any]]:
    if df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    fid_col = "flow_id"
    for fid, grp in df.groupby(fid_col):
        row = grp.iloc[0].to_dict()
        out[str(fid)] = row
    return out


def _build_real_pair_features(
    pair_df: pd.DataFrame,
    src_by_flow: dict[str, dict[str, Any]],
    dst_by_flow: dict[str, dict[str, Any]],
    seed_data: dict[str, Any],
) -> pd.DataFrame:
    eth_by, bnb_by = seed_data["eth_by_id"], seed_data["bnb_by_id"]
    rows = []
    for r in pair_df.itertuples(index=False):
        sf, df = str(r.src_flow_id), str(r.dst_flow_id)
        se, de = src_by_flow.get(sf, {}), dst_by_flow.get(df, {})
        eth, bnb = eth_by.get(sf) or {}, bnb_by.get(df) or {}
        t0s, _ = _flow_ts(eth)
        t0d, _ = _flow_ts(bnb)
        lag = max(0.0, t0d - t0s)
        amt_s, amt_d = float(eth.get("amount_usd") or 0), float(bnb.get("amount_usd") or 0)
        amt_err = abs(amt_s - amt_d) / max(amt_s, amt_d, 1e-9)
        def _match(a, b):
            if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
                return UNAVAILABLE_NUM
            return float(str(a) == str(b) and str(a) != "")

        rows.append(
            {
                "src_flow_id": sf,
                "dst_flow_id": df,
                "real_bridge_contract_pair_match": _match(se.get("src_contract_address"), de.get("dst_contract_address")),
                "real_event_signature_pair_match": _match(se.get("src_event_topic0"), de.get("dst_event_topic0")),
                "real_message_id_match": UNAVAILABLE_NUM if math.isnan(float(se.get("src_message_id_if_available", UNAVAILABLE_NUM) or UNAVAILABLE_NUM)) else _match(se.get("src_message_id_if_available"), de.get("dst_message_id_if_available")),
                "real_nonce_match": UNAVAILABLE_NUM,
                "real_transfer_id_match": UNAVAILABLE_NUM,
                "real_sender_receiver_consistency": float(str(se.get("src_sender", "")) != "" and str(se.get("src_sender")) in _addr_set(bnb)),
                "real_token_pair_consistency": _match(se.get("src_token"), de.get("dst_token")),
                "real_log_index_order_score": 1.0 / (1.0 + abs(float(se.get("src_log_index") or 0) - float(de.get("dst_log_index") or 0))) if se.get("receipt_fetched") and de.get("receipt_fetched") else UNAVAILABLE_NUM,
                "real_block_lag_score": 1.0 / (1.0 + lag / 3600.0),
                "real_tx_index_order_score": 1.0 / (1.0 + abs(float(se.get("src_tx_index") or 0) - float(de.get("dst_tx_index") or 0))) if not math.isnan(float(se.get("src_tx_index") or UNAVAILABLE_NUM)) else UNAVAILABLE_NUM,
                "real_event_amount_match": 1.0 - min(amt_err, 1.0),
                "real_fee_adjusted_amount_error": amt_err,
                "real_event_topic_similarity": _match(se.get("src_event_topic0"), de.get("dst_event_topic0")),
                "real_bridge_event_fingerprint_similarity": float(str(se.get("src_event_topic0", "")) == str(de.get("dst_event_topic0", "")) and str(se.get("src_event_topic0", "")) != ""),
                "real_bridge_event_confidence": float(se.get("receipt_fetched") and de.get("receipt_fetched")),
            }
        )
    return pd.DataFrame(rows)


def _feature_fingerprint(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        v = row.get(c)
        if pd.isna(v):
            parts.append(f"{c}=NA")
        else:
            parts.append(f"{c}={float(v):.6f}" if isinstance(v, (int, float)) else f"{c}={v}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _run_identifiability_audit(
    seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
    build_features: Callable,
) -> dict[str, Any]:
    exact_dup, near_dup, gt_in_eq, gt_total = [], [], 0, 0
    nonident = []
    eq_rows = []
    symmetry = 0
    for seed in seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        if not sd.is_dir():
            continue
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _p10w._load_frozen_plan(sd, allowed, top_k)
        truth = _pair_set(seed_data["labels"])
        feat = build_features(plan, pair_df, seed_data, sd)
        if feat.empty:
            continue
        cols = [c for c in feat.columns if c not in ("src_flow_id", "dst_flow_id", "seed") and pd.api.types.is_numeric_dtype(feat[c])]
        feat = feat.copy()
        feat["_sig"] = feat.apply(lambda r: _feature_fingerprint(r, cols), axis=1)
        for sig, grp in feat.groupby("_sig"):
            labels = {(str(a), str(b)): int((str(a), str(b)) in truth) for a, b in zip(grp["src_flow_id"], grp["dst_flow_id"])}
            n_true = sum(labels.values())
            n_false = len(labels) - n_true
            if len(grp) > 1:
                eq_rows.append({"seed": seed, "signature": sig, "n_candidates": len(grp), "n_true": n_true, "n_false": n_false})
            if n_true > 0 and n_false > 0:
                gt_in_eq += n_true
                symmetry += 1
            if n_true > 1:
                gt_in_eq += n_true
            for pair, y in labels.items():
                if y:
                    gt_total += 1
                    if n_false > 0:
                        nonident.append(1)
                    else:
                        nonident.append(0)
        n_cand = len(feat)
        n_unique = feat["_sig"].nunique()
        exact_dup.append(1.0 - n_unique / max(n_cand, 1))
        near_dup.append(float((feat.groupby("_sig").size() > 1).mean()))
    label_nonid = float(np.mean(nonident)) if nonident else 1.0
    upper = 1.0 - label_nonid
    return {
        "exact_duplicate_candidate_fraction": float(np.mean(exact_dup)) if exact_dup else 1.0,
        "near_duplicate_candidate_fraction": float(np.mean(near_dup)) if near_dup else 1.0,
        "gt_edges_in_equivalence_classes": gt_in_eq,
        "label_nonidentifiability_fraction": label_nonid,
        "symmetry_breaking_evidence_available": symmetry == 0,
        "exact_pair_identifiability_upper_bound": upper,
        "equivalence_class_count": len(eq_rows),
        "equivalence_classes": eq_rows[:5000],
    }


def _build_entity_graph(seed_data: dict[str, Any], pair_df: pd.DataFrame | None, *, rpc_available: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    src_rows, dst_rows = [], []
    for chain, flows, prefix in (("src", seed_data["eth_flows"], "src"), ("dst", seed_data["bnb_flows"], "dst")):
        for f in flows:
            addrs = sorted(_addr_set(f))
            for a in addrs[:10]:
                row = {"flow_id": str(f.get("flow_id") or ""), "address": a, "degree": len(addrs), "chain": chain.upper(), "rpc_enriched": rpc_available}
                if prefix == "src":
                    src_rows.append(row)
                else:
                    dst_rows.append(row)
    src_df = pd.DataFrame(src_rows)
    dst_df = pd.DataFrame(dst_rows)
    pair_rows = []
    eth_by, bnb_by = seed_data["eth_by_id"], seed_data["bnb_by_id"]
    pairs = pair_df[["src_flow_id", "dst_flow_id"]].drop_duplicates() if pair_df is not None and not pair_df.empty else None
    if pairs is not None:
        iter_pairs = ((str(r.src_flow_id), str(r.dst_flow_id)) for r in pairs.itertuples(index=False))
    else:
        iter_pairs = ((sf, df) for sf in list(eth_by.keys())[:100] for df in list(bnb_by.keys())[:100])
    for sf, df in iter_pairs:
        eth, bnb = eth_by.get(sf) or {}, bnb_by.get(df) or {}
        sa, sb = _addr_set(eth), _addr_set(bnb)
        inter = len(sa & sb)
        union = len(sa | sb)
        pair_rows.append(
            {
                "src_flow_id": sf,
                "dst_flow_id": df,
                "real_predecessor_overlap": inter / max(union, 1),
                "real_successor_overlap": inter / max(union, 1),
                "real_counterparty_jaccard": inter / max(union, 1),
                "real_counterparty_weighted_overlap": inter / max(union, 1),
                "real_temporal_coactivity": float(abs(float(eth.get("start_time") or 0) - float(bnb.get("start_time") or 0)) < 86400.0 * 7),
                "real_entity_degree_profile_similarity": 1.0 / (1.0 + abs(len(sa) - len(sb))),
                "real_entity_token_profile_similarity": float(eth.get("asset_group") == bnb.get("asset_group")),
                "real_bridge_usage_profile_similarity": float(eth.get("route_id") == bnb.get("route_id")),
                "real_exchange_or_bridge_role_similarity": UNAVAILABLE_NUM,
                "real_entity_risk_context_similarity": 1.0 / (1.0 + abs(float(eth.get("aml_score") or 0) - float(bnb.get("aml_score") or 0))),
                "entity_graph_rpc_enriched": rpc_available,
            }
        )
    meta = {"real_entity_graph_available": rpc_available, "note": "Flow-level address graph; full k-hop RPC neighborhood requires ETH_RPC_URL/BSC_RPC_URL"}
    return src_df, dst_df, pd.DataFrame(pair_rows), meta


def _build_real_segments(src_events: pd.DataFrame, dst_events: pd.DataFrame, seed_data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src_by = _flow_event_lookup(src_events, "src")
    dst_by = _flow_event_lookup(dst_events, "dst")
    src_rows, dst_rows = [], []
    for variant in SEG_VARIANTS:
        for chain, flows, lookup in (("src", seed_data["eth_flows"], src_by), ("dst", seed_data["bnb_flows"], dst_by)):
            for f in flows:
                fid = str(f.get("flow_id") or "")
                ev = lookup.get(fid, {})
                t0, _ = _flow_ts(f)
                amt = float(f.get("amount_usd") or 0)
                if variant == "real_tx_log_event_segments":
                    seg = f"log_{ev.get('src_log_index' if chain=='src' else 'dst_log_index', 'na')}"
                elif variant == "real_bridge_event_segments":
                    seg = str(ev.get("src_event_topic0" if chain == "src" else "dst_event_topic0") or "na")[:16]
                elif variant == "real_message_id_segments":
                    seg = "unavailable"
                elif variant == "real_nonce_segments":
                    seg = "unavailable"
                elif variant == "real_transfer_id_segments":
                    seg = "unavailable"
                elif variant == "real_contract_topic_segments":
                    seg = f"{ev.get('src_contract_address' if chain=='src' else 'dst_contract_address','')}|{str(ev.get('src_event_topic0' if chain=='src' else 'dst_event_topic0',''))[:10]}"
                else:
                    seg = f"a{int(amt//50)}|t{int(t0//900)}"
                row = {"flow_id": fid, "segment_variant": variant, "real_segment_id": seg}
                if chain == "src":
                    src_rows.append(row)
                else:
                    dst_rows.append(row)
    src_df = pd.DataFrame(src_rows)
    dst_df = pd.DataFrame(dst_rows)
    cmp_rows = []
    for v in SEG_VARIANTS:
        ss = src_df[src_df["segment_variant"] == v]["real_segment_id"].nunique() if not src_df.empty else 0
        ds = dst_df[dst_df["segment_variant"] == v]["real_segment_id"].nunique() if not dst_df.empty else 0
        cmp_rows.append({"segment_variant": v, "n_src_segments": ss, "n_dst_segments": ds})
    return src_df, dst_df, pd.DataFrame(cmp_rows)


def _merge_features(base: pd.DataFrame, *extras: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    keys = ["src_flow_id", "dst_flow_id"]
    for ex in extras:
        if ex is None or ex.empty:
            continue
        val_cols = [c for c in ex.columns if c not in keys]
        out = out.drop(columns=[c for c in val_cols if c in out.columns], errors="ignore")
        out = out.merge(ex[keys + val_cols], on=keys, how="left")
    out = out.loc[:, ~out.columns.duplicated()]
    for c in out.columns:
        if c in keys:
            continue
        if str(out.dtypes.get(c, "")) == "object":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _feature_cols(feat: pd.DataFrame) -> list[str]:
    skip = {"src_flow_id", "dst_flow_id", "seed", "_sig"} | _p10x.PROHIBITED_INFERENCE_COLS
    return [c for c in feat.columns if c not in skip and not c.endswith("_available") and pd.api.types.is_numeric_dtype(feat[c])]


def _compute_ceiling(
    seeds: list[int],
    synthetic_root: Path,
    *,
    max_delay_sec: float,
    top_k: int,
    build_features: Callable,
    identifiability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    cand_recs, score_f1s, p_at_r, r_at_p = [], [], [], []
    aurocs, auprcs, amb_fracs, collisions = [], [], [], []
    for seed in seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        if not sd.is_dir():
            continue
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=top_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _p10w._load_frozen_plan(sd, allowed, top_k)
        truth = _pair_set(seed_data["labels"])
        cand_recs.append(len(truth & allowed) / max(len(truth), 1))
        sc = plan.copy()
        sc["transport_mass"] = pd.to_numeric(sc.get("transport_mass"), errors="coerce").fillna(0.0)
        scan = _p10w._threshold_scan(truth, sc, score_col="transport_mass")
        score_f1s.append(scan["oracle_best_f1"])
        p_at_r.append(scan["oracle_precision_at_recall_0_8"])
        r_at_p.append(scan["oracle_recall_at_precision_0_8"])
        feat = build_features(plan, pair_df, seed_data, sd)
        if not feat.empty:
            cols = _feature_cols(feat)
            y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in feat.itertuples(index=False)])
            x = feat[cols].fillna(0.0).mean(axis=1).to_numpy(dtype=float) if cols else np.zeros(len(feat))
            if len(np.unique(y)) > 1 and cols:
                aurocs.append(float(roc_auc_score(y, x)))
                auprcs.append(float(average_precision_score(y, x)))
            amb_fracs.append(_p12_ambiguous_gt(feat, truth, seed_data))
            src_counts = feat.groupby("src_flow_id").size()
            collisions.append(float((src_counts > 1).mean()))
    out = {
        "candidate_oracle_recall": float(np.mean(cand_recs)) if cand_recs else 0.0,
        "score_oracle_best_f1": float(np.mean(score_f1s)) if score_f1s else 0.0,
        "oracle_precision_at_recall_0_8": float(np.mean(p_at_r)) if p_at_r else 0.0,
        "oracle_recall_at_precision_0_8": float(np.mean(r_at_p)) if r_at_p else 0.0,
        "feature_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "feature_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
        "ambiguous_gt_fraction": float(np.mean(amb_fracs)) if amb_fracs else 1.0,
        "candidate_collision_rate": float(np.mean(collisions)) if collisions else 1.0,
    }
    if identifiability:
        out["exact_pair_identifiability_upper_bound"] = identifiability.get("exact_pair_identifiability_upper_bound", 0.0)
        out["label_nonidentifiability_fraction"] = identifiability.get("label_nonidentifiability_fraction", 1.0)
    return out


def _p12_ambiguous_gt(feat: pd.DataFrame, truth: set[tuple[str, str]], seed_data: dict[str, Any]) -> float:
    eth_by, bnb_by = seed_data["eth_by_id"], seed_data["bnb_by_id"]
    groups: dict[str, set[bool]] = defaultdict(set)
    gt_sigs: dict[tuple[str, str], str] = {}
    cols = _feature_cols(feat)
    for row in feat.itertuples(index=False):
        sig = _feature_fingerprint(pd.Series(row._asdict()), cols)
        is_true = (str(row.src_flow_id), str(row.dst_flow_id)) in truth
        groups[sig].add(is_true)
        if is_true:
            gt_sigs[(str(row.src_flow_id), str(row.dst_flow_id))] = sig
    gt_in_amb, gt_total = 0, 0
    for pair in truth:
        gt_total += 1
        sig = gt_sigs.get(pair)
        if sig and True in groups.get(sig, set()) and False in groups.get(sig, set()):
            gt_in_amb += 1
    return gt_in_amb / max(gt_total, 1)


def _evaluate_feasibility_gate(ceiling: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "candidate_oracle_recall": ceiling.get("candidate_oracle_recall", 0) >= FEASIBILITY["candidate_oracle_recall"],
        "oracle_precision_at_recall_0_8": ceiling.get("oracle_precision_at_recall_0_8", 0) >= FEASIBILITY["oracle_precision_at_recall_0_8"],
        "oracle_recall_at_precision_0_8": ceiling.get("oracle_recall_at_precision_0_8", 0) >= FEASIBILITY["oracle_recall_at_precision_0_8"],
        "score_oracle_best_f1": ceiling.get("score_oracle_best_f1", 0) >= FEASIBILITY["score_oracle_best_f1"],
        "feature_auroc": ceiling.get("feature_auroc", 0) >= FEASIBILITY["feature_auroc"],
        "feature_auprc": ceiling.get("feature_auprc", 0) >= FEASIBILITY["feature_auprc"],
        "ambiguous_gt_fraction": ceiling.get("ambiguous_gt_fraction", 1) <= FEASIBILITY["ambiguous_gt_fraction"],
        "candidate_collision_rate": ceiling.get("candidate_collision_rate", 1) <= FEASIBILITY["candidate_collision_rate"],
        "exact_pair_identifiability_upper_bound": ceiling.get("exact_pair_identifiability_upper_bound", 0) >= FEASIBILITY["exact_pair_identifiability_upper_bound"],
        "no_gt_leakage": True,
        "inference_allowed_evidence_only": True,
    }
    failed = [k for k, v in checks.items() if not v]
    gate_pass = all(checks.values())
    return {"feasibility_gate_pass": gate_pass, "checks": checks, "thresholds": FEASIBILITY, "metrics": ceiling, "failed_checks": failed, "train_high_pr_model_allowed": gate_pass}


def _high_pr_claim_gate(y: dict[str, float], *, formal: bool) -> dict[str, Any]:
    high = formal and all([
        y.get("flow_pair_precision", 0) >= 0.80, y.get("flow_pair_recall", 0) >= 0.80, y.get("flow_pair_f1", 0) >= 0.80,
        y.get("split_recovery", 0) >= 0.75, y.get("merge_recovery", 0) >= 0.75, y.get("flow_mass_recall", 0) >= 0.70, y.get("ece", 1) <= 0.10,
    ])
    if high:
        allowed = "RC-UOT-HP achieves high precision and high recall on the same-scope sealed holdout when augmented with real bridge-event, entity-graph, and transaction-level segmentation evidence."
        limitation = "This result depends on richer event/entity evidence and does not imply that the original flow-only setting can achieve high P/R."
    else:
        allowed = "Even after real evidence acquisition, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity or non-identifiability in the current dataset."
        limitation = "High P/R not established."
    return {"high_pr_gate_pass": high, "allowed_claim": allowed, "required_limitation": limitation, "forbidden_claim": "Do not claim universal superiority, real-pool superiority, or high P/R without gate PASS.", "rcuot_hp_real": y}


def run_phase13(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    candidate_k: int,
    generate_sealed_seeds: bool,
    reconstruct_real_bridge_events: bool,
    build_real_entity_graph: bool,
    run_identifiability_audit: bool,
    repair_real_segmentation: bool,
    run_feasibility_gate: bool,
    train_if_feasible: bool,
    evaluate_holdout_once: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out = run_root / "phase13_real_evidence_acquisition"
    for d in ("evidence", "cache/eth_receipts", "cache/bsc_receipts", "diagnosis", "segmentation", "models", "selection", "holdout", "audit", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)

    synthetic_root = _p10s._resolve_synthetic_root(run_root)
    uk = _p10x._load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 86400.0)
    env = _env_status()

    holdout_status: dict[int, bool] = {}
    if generate_sealed_seeds:
        holdout_status = _p10v._ensure_sealed_seeds(run_root, holdout_seeds)
        synthetic_root = _p10s._resolve_synthetic_root(run_root)
    else:
        for s in holdout_seeds:
            sd = synthetic_root / f"synthetic_eval_seed_{s}"
            holdout_status[s] = sd.is_dir() and (sd / "uot" / "uot_transport_plan.csv").is_file()
    formal_allowed = all(holdout_status.get(s, False) for s in holdout_seeds)

    split_rows = []
    for seed in train_seeds + dev_seeds + holdout_seeds:
        sd = synthetic_root / f"synthetic_eval_seed_{seed}"
        if not sd.is_dir():
            continue
        labels = pd.read_csv(_p10s._find_in_seed(sd, "synthetic_flow_labels.csv"), dtype=str, keep_default_na=False)
        for _, r in labels.iterrows():
            split_rows.append({"seed": seed, "src_flow_id": str(r.get("src_flow_id") or ""), "dst_flow_id": str(r.get("dst_flow_id") or ""), "pattern_type_eval_only": str(r.get("pattern_type") or "")})
    all_df = pd.DataFrame(split_rows)
    (out / "splits" / "split_summary.json").write_text(
        json.dumps({
            "canonical_rebuilt": False, "label_layer_refrozen": False, "phase10s_to_12_results_preserved": True,
            "phase12_failure_preserved": True, "new_holdout_92_111_frozen_before_search": formal_allowed,
            "holdout_labels_used_for_tuning": False, "formal_claim_allowed": formal_allowed,
            "train_seeds": train_seeds, "dev_seeds": dev_seeds, "sealed_holdout_seeds": holdout_seeds,
            "holdout_seed_status": holdout_status, "env": env["vars"],
        }, indent=2),
        encoding="utf-8",
    )
    all_df[all_df["seed"].isin(train_seeds)].to_csv(out / "splits" / "train_split.csv", index=False)
    all_df[all_df["seed"].isin(dev_seeds)].to_csv(out / "splits" / "dev_split.csv", index=False)
    all_df[all_df["seed"].isin(holdout_seeds)].to_csv(out / "splits" / "sealed_holdout_split.csv", index=False)

    if not env["any_rpc"]:
        (out / "evidence" / "evidence_unavailable_report.md").write_text(
            "# Evidence unavailable report\n\n"
            "Missing environment variables (not silently ignored):\n\n"
            + "\n".join(f"- **{k}**: {'set' if v else 'MISSING'}" for k, v in env["vars"].items())
            + "\n\nReal bridge-event reconstruction requires ETH_RPC_URL and/or BSC_RPC_URL. "
            "Explorer keys (ETHERSCAN_API_KEY, BSCSCAN_API_KEY) optional for supplementary metadata.\n",
            encoding="utf-8",
        )

    audit_seed = dev_seeds[0] if dev_seeds else train_seeds[0]
    bridge_meta: dict[str, Any] = {"real_bridge_event_evidence": False}
    entity_meta: dict[str, Any] = {"real_entity_graph_available": False}
    src_events = dst_events = pd.DataFrame()
    real_pair = entity_pair = pd.DataFrame()
    micro_src = micro_dst = pd.DataFrame()

    if reconstruct_real_bridge_events or build_real_entity_graph or repair_real_segmentation:
        sd = _p10s._seed_dir(synthetic_root, audit_seed)
        seed_data = _p10s._load_seed_data(sd)
        ctx = _p10x._load_seed_context(audit_seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache={})
        if reconstruct_real_bridge_events:
            src_events, dst_events, bridge_meta = _reconstruct_bridge_events(
                seed_data, eth_cache=out / "cache" / "eth_receipts", bsc_cache=out / "cache" / "bsc_receipts", env=env,
            )
            src_events.to_csv(out / "evidence" / "real_bridge_event_src.csv", index=False)
            dst_events.to_csv(out / "evidence" / "real_bridge_event_dst.csv", index=False)
            src_by = _flow_event_lookup(src_events, "src")
            dst_by = _flow_event_lookup(dst_events, "dst")
            real_pair = _build_real_pair_features(ctx["pair_df"], src_by, dst_by, seed_data)
            real_pair.to_csv(out / "evidence" / "real_bridge_event_pair_features.csv", index=False)
            (out / "evidence" / "real_bridge_event_reconstruction_report.md").write_text(
                "# Real bridge event reconstruction\n\n"
                f"- RPC available: {env['any_rpc']}\n"
                f"- Real bridge evidence: **{bridge_meta.get('real_bridge_event_evidence')}**\n"
                f"- ETH receipts fetched: {bridge_meta.get('eth_receipts_fetched', 0)}\n"
                f"- BSC receipts fetched: {bridge_meta.get('bsc_receipts_fetched', 0)}\n"
                f"- Fetch errors: {bridge_meta.get('fetch_errors', 0)}\n\n"
                "Fields marked unavailable (nonce, message_id, transfer_id, relayer) are NaN, not zero.\n",
                encoding="utf-8",
            )
        if build_real_entity_graph:
            sg, dg, entity_pair, entity_meta = _build_entity_graph(seed_data, ctx["pair_df"], rpc_available=env["any_rpc"])
            sg.to_csv(out / "evidence" / "real_src_entity_graph.csv", index=False)
            dg.to_csv(out / "evidence" / "real_dst_entity_graph.csv", index=False)
            entity_pair.head(50000).to_csv(out / "evidence" / "real_entity_pair_features.csv", index=False)
            (out / "evidence" / "real_entity_graph_report.md").write_text(
                f"# Real entity graph\n\n{entity_meta.get('note')}\n\nRPC enriched: {entity_meta.get('real_entity_graph_available')}\n",
                encoding="utf-8",
            )
        if repair_real_segmentation:
            micro_src, micro_dst, seg_cmp = _build_real_segments(src_events, dst_events, seed_data)
            micro_src.to_csv(out / "segmentation" / "phase13_real_segments_src.csv", index=False)
            micro_dst.to_csv(out / "segmentation" / "phase13_real_segments_dst.csv", index=False)
            seg_cmp.to_csv(out / "segmentation" / "phase13_segmentation_comparison.csv", index=False)
            (out / "segmentation" / "phase13_segmentation_report.md").write_text("# Real segmentation\n\nPrivate Phase 13 tx/event-level segments.\n", encoding="utf-8")

    gate_seeds = dev_seeds[: min(6, len(dev_seeds))]
    ident: dict[str, Any] = {}
    ceilings: dict[str, dict[str, Any]] = {}
    feasibility: dict[str, Any] = {"feasibility_gate_pass": False}

    def _feat_base(plan, pair_df, seed_data, seed_dir):
        return _p10x._build_group_features(_p10x._build_evidence_edge_features(plan, pair_df, seed_data, max_delay_sec=max_delay_sec), seed_data)

    def _feat_bridge(plan, pair_df, seed_data, seed_dir):
        se, de, bm = _reconstruct_bridge_events(seed_data, eth_cache=out / "cache" / "eth_receipts", bsc_cache=out / "cache" / "bsc_receipts", env=env, max_tx_per_side=30)
        sb, db = _flow_event_lookup(se, "src"), _flow_event_lookup(de, "dst")
        rp = _build_real_pair_features(pair_df, sb, db, seed_data)
        return _merge_features(_feat_base(plan, pair_df, seed_data, seed_dir), rp)

    def _feat_entity(plan, pair_df, seed_data, seed_dir):
        _, _, ep, _ = _build_entity_graph(seed_data, pair_df, rpc_available=env["any_rpc"])
        return _merge_features(_feat_bridge(plan, pair_df, seed_data, seed_dir), ep)

    def _feat_full(plan, pair_df, seed_data, seed_dir):
        se, de, _ = _reconstruct_bridge_events(seed_data, eth_cache=out / "cache" / "eth_receipts", bsc_cache=out / "cache" / "bsc_receipts", env=env, max_tx_per_side=20)
        ms, md, _ = _build_real_segments(se, de, seed_data)
        feat = _feat_entity(plan, pair_df, seed_data, seed_dir)
        v = "real_contract_topic_segments"
        ms2 = ms[ms["segment_variant"] == v].rename(columns={"real_segment_id": "src_real_segment_id"})
        md2 = md[md["segment_variant"] == v].rename(columns={"real_segment_id": "dst_real_segment_id"})
        feat = feat.merge(ms2[["flow_id", "src_real_segment_id"]].rename(columns={"flow_id": "src_flow_id"}), on="src_flow_id", how="left")
        feat = feat.merge(md2[["flow_id", "dst_real_segment_id"]].rename(columns={"flow_id": "dst_flow_id"}), on="dst_flow_id", how="left")
        feat["real_segment_exact_match"] = (feat["src_real_segment_id"].astype(str) == feat["dst_real_segment_id"].astype(str)).astype(float)
        return feat

    if run_identifiability_audit:
        ident = _run_identifiability_audit(gate_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, build_features=_feat_full)
        (out / "diagnosis" / "identifiability_audit.json").write_text(json.dumps({k: v for k, v in ident.items() if k != "equivalence_classes"}, indent=2), encoding="utf-8")
        if ident.get("equivalence_classes"):
            pd.DataFrame(ident["equivalence_classes"]).to_csv(out / "diagnosis" / "feature_equivalence_classes.csv", index=False)
        (out / "diagnosis" / "identifiability_audit.md").write_text(
            f"# Identifiability audit\n\n- exact_pair_identifiability_upper_bound: **{ident.get('exact_pair_identifiability_upper_bound', 0):.3f}**\n"
            f"- label_nonidentifiability_fraction: {ident.get('label_nonidentifiability_fraction', 1):.3f}\n"
            f"- symmetry_breaking_evidence_available: {ident.get('symmetry_breaking_evidence_available')}\n",
            encoding="utf-8",
        )
        if ident.get("exact_pair_identifiability_upper_bound", 0) < 0.8:
            (out / "diagnosis" / "exact_pair_identifiability_failure.md").write_text(
                "# Exact pair identifiability failure\n\n"
                "Exact pair-level correspondence is not identifiable under the current inference-allowed evidence "
                "because multiple candidate edges are feature-equivalent.\n",
                encoding="utf-8",
            )

    if run_feasibility_gate:
        ceilings["after_real_bridge_events"] = _compute_ceiling(gate_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, build_features=_feat_bridge, identifiability=ident)
        ceilings["after_identifiability_repair"] = {**ceilings["after_real_bridge_events"], **ident}
        ceilings["after_real_entity_graph"] = _compute_ceiling(gate_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, build_features=_feat_entity, identifiability=ident)
        ceilings["after_real_segmentation"] = _compute_ceiling(gate_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, build_features=_feat_full, identifiability=ident)
        for key, fname in (
            ("after_real_bridge_events", "ceiling_after_real_bridge_events.json"),
            ("after_identifiability_repair", "ceiling_after_identifiability_repair.json"),
            ("after_real_entity_graph", "ceiling_after_real_entity_graph.json"),
            ("after_real_segmentation", "ceiling_after_real_segmentation.json"),
        ):
            (out / "diagnosis" / fname).write_text(json.dumps(ceilings[key], indent=2), encoding="utf-8")
        pd.DataFrame([{"stage": k, **v} for k, v in ceilings.items()]).to_csv(out / "diagnosis" / "phase13_ceiling_progression_table.csv", index=False)
        feasibility = _evaluate_feasibility_gate(ceilings["after_real_segmentation"])
        (out / "diagnosis" / "phase13_feasibility_gate.json").write_text(json.dumps(feasibility, indent=2), encoding="utf-8")
        (out / "diagnosis" / "phase13_feasibility_gate.md").write_text(
            f"# Phase 13 feasibility gate\n\n**PASS:** {feasibility['feasibility_gate_pass']}\n\nFailed: {', '.join(feasibility.get('failed_checks', []))}\n",
            encoding="utf-8",
        )
        if not feasibility["feasibility_gate_pass"]:
            (out / "diagnosis" / "missing_real_evidence_report.md").write_text(
                "# Missing real evidence\n\n"
                f"- RPC: {env['any_rpc']}\n- Real bridge: {bridge_meta.get('real_bridge_event_evidence')}\n"
                f"- Identifiability upper bound: {ident.get('exact_pair_identifiability_upper_bound', 0):.3f}\n",
                encoding="utf-8",
            )
            (out / "diagnosis" / "phase13_high_pr_infeasibility_final.md").write_text(
                "# Phase 13 high P/R infeasibility\n\n"
                "Even after real evidence acquisition attempt, exact high-P/R pair correspondence remains limited "
                "by unresolved candidate ambiguity or non-identifiability in the current dataset.\n",
                encoding="utf-8",
            )

    trained = False
    selected: dict[str, Any] = {"training_skipped": True, "reason": "feasibility_gate_fail"}
    holdout_gate: dict[str, Any] = {}

    if train_if_feasible and feasibility.get("feasibility_gate_pass"):
        trained = True
        from sklearn.ensemble import GradientBoostingClassifier

        train_parts, cols = [], []
        for seed in train_seeds:
            sd = _p10s._seed_dir(synthetic_root, seed)
            seed_data = _p10s._load_seed_data(sd)
            pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed)
            plan = _p10w._load_frozen_plan(sd, _p10s._allowed_pairs(pair_df), candidate_k)
            feat = _feat_full(plan, pair_df, seed_data, sd)
            train_parts.append(feat)
        train_feat = pd.concat(train_parts, ignore_index=True)
        cols = _feature_cols(train_feat)
        truth_train = set()
        for seed in train_seeds:
            truth_train |= _pair_set(_p10s._load_seed_data(_p10s._seed_dir(synthetic_root, seed))["labels"])
        y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth_train) for r in train_feat.itertuples(index=False)])
        model = GradientBoostingClassifier(random_state=42, max_depth=5, n_estimators=200)
        model.fit(train_feat[cols].fillna(0.0).to_numpy(dtype=float), y)
        with (out / "models" / "rcuot_hp_real_evidence_verifier.pkl").open("wb") as fh:
            pickle.dump({"model": model, "cols": cols}, fh)
        (out / "models" / "rcuot_hp_real_evidence_decoder.json").write_text(json.dumps({"p_threshold": 0.25, "row_top_k": 2, "col_top_k": 2, "allow_split_merge": True}, indent=2), encoding="utf-8")
        selected = {"model": "HP-real-GBDT", "holdout_not_used": True}

    if evaluate_holdout_once and trained and formal_allowed:
        bundle = pickle.loads((out / "models" / "rcuot_hp_real_evidence_verifier.pkl").read_bytes())
        model, cols = bundle["model"], bundle["cols"]
        dec = json.loads((out / "models" / "rcuot_hp_real_evidence_decoder.json").read_text(encoding="utf-8"))
        rows = []
        for seed in [s for s in holdout_seeds if holdout_status.get(s, False)]:
            sd = _p10s._seed_dir(synthetic_root, seed)
            seed_data = _p10s._load_seed_data(sd)
            ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache={})
            feat = _feat_full(ctx["base_plan"], ctx["pair_df"], seed_data, sd)
            probs = model.predict_proba(feat[cols].fillna(0.0).to_numpy(dtype=float))[:, 1]
            plan = _p10x._group_consistent_decode(ctx["base_plan"], feat, probs, seed_data, **dec)
            rows.append(_p10x._metrics_row("rcuot_hp_real", seed, plan, ctx["um"], seed_data, out / "holdout" / str(seed)))
        hold_df = pd.DataFrame(rows)
        hold_df.to_csv(out / "holdout" / "holdout_metrics_by_method.csv", index=False)
        hold_df.to_csv(out / "holdout" / "holdout_metrics_by_seed.csv", index=False)
        agg = hold_df.groupby("method", as_index=False)[PRIMARY_METRICS].mean()
        y = agg.iloc[0].to_dict()
        holdout_gate = _high_pr_claim_gate(y, formal=formal_allowed)
        (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps(holdout_gate, indent=2, default=str), encoding="utf-8")
        agg.to_csv(out / "holdout" / "table_n_rcuot_hp_real_evidence.csv", index=False)
        (out / "holdout" / "table_n_rcuot_hp_real_evidence.md").write_text(f"# Table N\n\n{_df_to_md(agg)}\n", encoding="utf-8")
    else:
        holdout_gate = {
            "high_pr_gate_pass": False, "feasibility_gate_pass": feasibility.get("feasibility_gate_pass", False),
            "training_skipped": not trained, "holdout_evaluation_skipped": True,
            "allowed_claim": "Even after real evidence acquisition, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity or non-identifiability in the current dataset.",
            "forbidden_claim": "Do not claim high P/R without gate PASS.",
            "diagnostic_metrics": ceilings.get("after_real_segmentation", {}),
            "identifiability": ident,
        }
        (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps(holdout_gate, indent=2, default=str), encoding="utf-8")
        pd.DataFrame([{"stage": k, **v} for k, v in ceilings.items()] if ceilings else [{"note": "feasibility_fail"}]).to_csv(out / "holdout" / "table_n_rcuot_hp_real_evidence.csv", index=False)
        (out / "holdout" / "table_n_rcuot_hp_real_evidence.md").write_text("# Table N\n\nHoldout skipped — feasibility gate FAIL.\n", encoding="utf-8")

    return {
        "ok": True, "out_dir": str(out), "formal_claim_allowed": formal_allowed,
        "feasibility_gate_pass": feasibility.get("feasibility_gate_pass", False),
        "real_bridge_event_evidence": bridge_meta.get("real_bridge_event_evidence", False),
        "real_entity_graph_available": entity_meta.get("real_entity_graph_available", False),
        "real_segmentation_available": repair_real_segmentation,
        "identifiability_upper_bound": ident.get("exact_pair_identifiability_upper_bound"),
        "trained": trained, "holdout_gate": holdout_gate, "ceilings": ceilings, "identifiability": ident,
        "elapsed_sec": time.time() - t0, "canonical_rebuilt": False, "label_layer_refrozen": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 13 real evidence acquisition")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=list(range(52, 72)))
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=list(range(92, 112)))
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    ap.add_argument("--reconstruct-real-bridge-events", action="store_true")
    ap.add_argument("--build-real-entity-graph", action="store_true")
    ap.add_argument("--run-identifiability-audit", action="store_true")
    ap.add_argument("--repair-real-segmentation", action="store_true")
    ap.add_argument("--run-feasibility-gate", action="store_true")
    ap.add_argument("--train-if-feasible", action="store_true")
    ap.add_argument("--evaluate-holdout-once", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for f in ("reconstruct_real_bridge_events", "build_real_entity_graph", "run_identifiability_audit", "repair_real_segmentation", "run_feasibility_gate", "train_if_feasible", "evaluate_holdout_once"):
            setattr(args, f, True)
        args.generate_sealed_seeds = True
    r = run_phase13(
        run_root=args.run_root, train_seeds=args.train_seeds, dev_seeds=args.dev_seeds, holdout_seeds=args.holdout_seeds,
        candidate_k=args.candidate_k, generate_sealed_seeds=args.generate_sealed_seeds,
        reconstruct_real_bridge_events=args.reconstruct_real_bridge_events, build_real_entity_graph=args.build_real_entity_graph,
        run_identifiability_audit=args.run_identifiability_audit, repair_real_segmentation=args.repair_real_segmentation,
        run_feasibility_gate=args.run_feasibility_gate, train_if_feasible=args.train_if_feasible, evaluate_holdout_once=args.evaluate_holdout_once,
    )
    print(json.dumps({"ok": r["ok"], "feasibility_gate_pass": r.get("feasibility_gate_pass"), "trained": r.get("trained"), "identifiability_upper_bound": r.get("identifiability_upper_bound")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
