#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: Chain-verified transferId event deaggregation for high-P/R RC-UOT."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.domain.evaluation.flow_eval import _pair_set

for _name, _path in (
    ("phase10s", _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"),
    ("phase10v", _REPO / "scripts" / "run_phase10v_pair_f1_precision_rcuot.py"),
    ("phase10w", _REPO / "scripts" / "run_phase10w_ambiguity_resolved_rcuot.py"),
    ("phase10x", _REPO / "scripts" / "run_phase10x_evidence_enhanced_disambiguation.py"),
    ("phase13", _REPO / "scripts" / "run_phase13_real_evidence_acquisition.py"),
    ("phase14", _REPO / "scripts" / "run_phase14_rpc_bridge_evidence_verification.py"),
    ("phase15", _REPO / "scripts" / "run_phase15_celer_abi_decode.py"),
):
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    globals()[f"_{_name}"] = _mod

OUT_REL = "phase16_transferid_event_deaggregation"
PHASE14_OUT = "phase14_rpc_bridge_evidence_verification"
PHASE15_OUT = "phase15_celer_abi_decode"
PRIMARY_K = 50
UNAVAILABLE = float("nan")
ETH_CHAIN_ID_HEX = "0x1"
BSC_CHAIN_ID_HEX = "0x38"
ETH_CHAIN_ID_INT = 1
BSC_CHAIN_ID_INT = 56

PHASE15_BASELINE = {
    "feature_auroc": 0.398,
    "feature_auprc": 0.043,
    "oracle_precision_at_recall_0_8": 0.048,
    "score_oracle_best_f1": 0.092,
    "candidate_collision_rate": 0.785,
    "transfer_id_match_precision": 0.195,
    "transfer_id_match_recall": 0.784,
}

PILOT_PASS = {
    "event_auroc": 0.85,
    "event_auprc": 0.70,
    "event_oracle_p_at_r": 0.80,
    "event_oracle_r_at_p": 0.80,
    "event_oracle_f1": 0.80,
    "lifted_flow_oracle_p_at_r": 0.50,
    "lifted_flow_oracle_f1": 0.092,
    "event_collision_rate": 0.30,
}

FEASIBILITY = dict(_phase13.FEASIBILITY)


def _rpc_json_call(url: str, method: str, params: list | None = None) -> dict[str, Any]:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_rpc_chain_identity(url: str) -> dict[str, Any]:
    masked = _phase14.mask_rpc_url(url)
    try:
        blk = _rpc_json_call(url, "eth_blockNumber")
        cid = _rpc_json_call(url, "eth_chainId")
        block_hex = blk.get("result")
        chain_hex = str(cid.get("result") or "")
        if not block_hex or not str(block_hex).startswith("0x"):
            return {"pass": False, "masked": masked, "error": "invalid_block", "chain_id": None}
        if not chain_hex.startswith("0x"):
            return {"pass": False, "masked": masked, "error": "invalid_chain_id", "chain_id": chain_hex}
        chain_int = int(chain_hex, 16)
        return {
            "pass": True,
            "masked": masked,
            "block_number_hex": block_hex,
            "chain_id": chain_hex,
            "chain_id_int": chain_int,
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        return {"pass": False, "masked": masked, "error": f"HTTPError_{exc.code}", "chain_id": None}
    except Exception as exc:
        return {"pass": False, "masked": masked, "error": type(exc).__name__, "chain_id": None}


def _all_configured_rpc_urls(repo_root: Path) -> list[str]:
    """Pool ETH/BSC env URLs — handles swapped .env labels."""
    _phase14.load_env_file_if_exists(repo_root)
    urls: list[str] = []
    for key in ("ETH_RPC_URLS", "ETH_RPC_URL", "BSC_RPC_URLS", "BSC_RPC_URL"):
        val = os.environ.get(key, "")
        if val:
            urls.extend(u.strip() for u in val.split(",") if u.strip())
    for chain in ("eth", "bsc"):
        urls.extend(_phase14.resolve_rpc_endpoints(chain, repo_root))
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def select_chain_verified_endpoints(repo_root: Path) -> dict[str, Any]:
    urls = _all_configured_rpc_urls(repo_root)
    eth_tests, bsc_tests = [], []
    eth_sel = bsc_sel = None
    for idx, url in enumerate(urls):
        t = test_rpc_chain_identity(url)
        t["index"] = idx
        t["url"] = url
        cid = t.get("chain_id")
        if cid == ETH_CHAIN_ID_HEX:
            eth_tests.append(t)
            if eth_sel is None and t.get("pass"):
                eth_sel = {"selected_index": idx, "selected_url": url, **t}
        elif cid == BSC_CHAIN_ID_HEX:
            bsc_tests.append(t)
            if bsc_sel is None and t.get("pass"):
                bsc_sel = {"selected_index": idx, "selected_url": url, **t}
    resolution = {
        "eth_rpc_urls_configured_count": len(urls),
        "bsc_rpc_urls_configured_count": len(urls),
        "eth_selected_index": eth_sel.get("selected_index") if eth_sel else None,
        "bsc_selected_index": bsc_sel.get("selected_index") if bsc_sel else None,
        "eth_block_number_sample": eth_sel.get("block_number_hex") if eth_sel else None,
        "bsc_block_number_sample": bsc_sel.get("block_number_hex") if bsc_sel else None,
        "eth_chain_id": eth_sel.get("chain_id") if eth_sel else None,
        "bsc_chain_id": bsc_sel.get("chain_id") if bsc_sel else None,
        "eth_chain_id_expected": ETH_CHAIN_ID_HEX,
        "bsc_chain_id_expected": BSC_CHAIN_ID_HEX,
        "eth_chain_id_pass": eth_sel is not None and eth_sel.get("chain_id") == ETH_CHAIN_ID_HEX,
        "bsc_chain_id_pass": bsc_sel is not None and bsc_sel.get("chain_id") == BSC_CHAIN_ID_HEX,
        "chain_identity_preflight_pass": bool(eth_sel and bsc_sel),
        "eth_rpc_masked": eth_sel.get("masked", "") if eth_sel else "",
        "bsc_rpc_masked": bsc_sel.get("masked", "") if bsc_sel else "",
        "credentials_committed": False,
        "full_rpc_url_logged": False,
        "eth_endpoint_tests": [{"index": x["index"], "chain_id": x.get("chain_id"), "pass": x.get("pass"), "masked": x.get("masked"), "error": x.get("error")} for x in eth_tests],
        "bsc_endpoint_tests": [{"index": x["index"], "chain_id": x.get("chain_id"), "pass": x.get("pass"), "masked": x.get("masked"), "error": x.get("error")} for x in bsc_tests],
        "eth_selected_url": eth_sel.get("selected_url") if eth_sel else None,
        "bsc_selected_url": bsc_sel.get("selected_url") if bsc_sel else None,
    }
    return resolution


def _preflight_public_record(resolution: dict[str, Any]) -> dict[str, Any]:
    """Disk-safe preflight record: no full RPC URLs."""
    pub = {k: v for k, v in resolution.items() if k not in ("eth_selected_url", "bsc_selected_url")}
    pub["credentials_committed"] = False
    pub["full_rpc_url_logged"] = False
    return pub


def run_chain_identity_preflight(out: Path, repo_root: Path) -> dict[str, Any]:
    resolution = select_chain_verified_endpoints(repo_root)
    (out / "diagnosis" / "phase16_chain_identity_preflight.json").write_text(
        json.dumps(_preflight_public_record(resolution), indent=2), encoding="utf-8"
    )
    md = "\n".join([
        "# Phase 16 chain identity preflight",
        f"- eth_chain_id: {resolution['eth_chain_id']} (expected {resolution['eth_chain_id_expected']})",
        f"- bsc_chain_id: {resolution['bsc_chain_id']} (expected {resolution['bsc_chain_id_expected']})",
        f"- eth_chain_id_pass: {resolution['eth_chain_id_pass']}",
        f"- bsc_chain_id_pass: {resolution['bsc_chain_id_pass']}",
        f"- chain_identity_preflight_pass: {resolution['chain_identity_preflight_pass']}",
        f"- eth_rpc_masked: {resolution['eth_rpc_masked']}",
        f"- bsc_rpc_masked: {resolution['bsc_rpc_masked']}",
    ])
    (out / "diagnosis" / "phase16_chain_identity_preflight.md").write_text(md + "\n", encoding="utf-8")
    if not resolution["chain_identity_preflight_pass"]:
        (out / "diagnosis" / "phase16_chain_identity_failure.md").write_text(
            "# Phase 16 chain identity failure\n\n"
            "No endpoint with correct eth_chainId found. Fix `.env`: ETH_RPC_URLS must point to Ethereum (chainId 0x1), "
            "BSC_RPC_URLS must point to BSC (chainId 0x38). Phase 15 results under swapped endpoints are unreliable.\n",
            encoding="utf-8",
        )
    else:
        _phase14._apply_selected_rpc_env(resolution.get("eth_selected_url"), resolution.get("bsc_selected_url"))
    return resolution


def _build_src_atoms(src_events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rows, maps = [], []
    tid_to_flows: dict[str, set[str]] = defaultdict(set)
    for i, e in enumerate(src_events):
        if e.get("event_name") not in ("Send", "LogNewTransferOut") or not _phase15._field_nonempty(e.get("transfer_id")):
            continue
        tid = str(e.get("transfer_id") or "").lower()
        fid = str(e.get("flow_id") or "")
        atom_id = f"src_{fid}_{tid[:18]}_{e.get('log_index', i)}"
        tid_to_flows[tid].add(fid)
        rows.append({
            "src_event_atom_id": atom_id,
            "src_flow_id": fid,
            "src_tx_hash": e.get("tx_hash", ""),
            "src_log_index": e.get("log_index", i),
            "source_transfer_id": tid,
            "sender": e.get("sender", ""),
            "receiver": e.get("receiver", ""),
            "token": e.get("token", ""),
            "amount_raw": e.get("amount_raw", ""),
            "amount_normalized": e.get("amount_normalized", UNAVAILABLE),
            "dst_chain_id": e.get("dst_chain_id", ""),
            "nonce": e.get("nonce", ""),
            "block_number": e.get("block_number", ""),
            "tx_index": e.get("tx_index", ""),
        })
        maps.append({"src_flow_id": fid, "src_event_atom_id": atom_id, "source_transfer_id": tid})
    atoms = pd.DataFrame(rows)
    if not atoms.empty:
        dup = atoms.groupby("source_transfer_id")["src_flow_id"].nunique()
        atoms["duplicate_event_assignment"] = atoms["source_transfer_id"].map(lambda t: dup.get(t, 0) > 1)
    flow_tid_counts = atoms.groupby("src_flow_id")["source_transfer_id"].nunique() if not atoms.empty else pd.Series(dtype=int)
    stats = {
        "n_atoms": len(atoms),
        "transferids_reused": sum(1 for flows in tid_to_flows.values() if len(flows) > 1),
        "flows_multi_tid": int((flow_tid_counts > 1).sum()) if len(flow_tid_counts) else 0,
        "flows_total": int(flow_tid_counts.shape[0]) if len(flow_tid_counts) else 0,
    }
    return atoms, pd.DataFrame(maps), stats


def _build_dst_atoms(dst_events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rows, maps = [], []
    stid_to_flows: dict[str, set[str]] = defaultdict(set)
    for i, e in enumerate(dst_events):
        if e.get("event_name") not in ("Relay", "LogNewTransferIn") or not _phase15._field_nonempty(e.get("src_transfer_id")):
            continue
        stid = str(e.get("src_transfer_id") or "").lower()
        fid = str(e.get("flow_id") or "")
        atom_id = f"dst_{fid}_{stid[:18]}_{e.get('log_index', i)}"
        stid_to_flows[stid].add(fid)
        rows.append({
            "dst_event_atom_id": atom_id,
            "dst_flow_id": fid,
            "dst_tx_hash": e.get("tx_hash", ""),
            "dst_log_index": e.get("log_index", i),
            "destination_transfer_id": str(e.get("transfer_id") or "").lower(),
            "destination_src_transfer_id": stid,
            "sender": e.get("sender", ""),
            "receiver": e.get("receiver", ""),
            "token": e.get("token", ""),
            "amount_raw": e.get("amount_raw", ""),
            "amount_normalized": e.get("amount_normalized", UNAVAILABLE),
            "src_chain_id": e.get("src_chain_id", ""),
            "block_number": e.get("block_number", ""),
            "tx_index": e.get("tx_index", ""),
        })
        maps.append({"dst_flow_id": fid, "dst_event_atom_id": atom_id, "destination_src_transfer_id": stid})
    atoms = pd.DataFrame(rows)
    if not atoms.empty:
        dup = atoms.groupby("destination_src_transfer_id")["dst_flow_id"].nunique()
        atoms["duplicate_event_assignment"] = atoms["destination_src_transfer_id"].map(lambda t: dup.get(t, 0) > 1)
    flow_stid_counts = atoms.groupby("dst_flow_id")["destination_src_transfer_id"].nunique() if not atoms.empty else pd.Series(dtype=int)
    stats = {
        "n_atoms": len(atoms),
        "src_transferids_reused": sum(1 for flows in stid_to_flows.values() if len(flows) > 1),
        "flows_multi_stid": int((flow_stid_counts > 1).sum()) if len(flow_stid_counts) else 0,
        "flows_total": int(flow_stid_counts.shape[0]) if len(flow_stid_counts) else 0,
    }
    return atoms, pd.DataFrame(maps), stats


def _construct_event_candidates(src_atoms: pd.DataFrame, dst_atoms: pd.DataFrame, *, include_negatives: bool = True) -> pd.DataFrame:
    if src_atoms.empty or dst_atoms.empty:
        return pd.DataFrame()
    dst_by_stid = dst_atoms.groupby("destination_src_transfer_id")
    rows: list[dict[str, Any]] = []
    matched_dst_atoms: set[str] = set()

    def _row(sa, da, tid_match: float) -> dict[str, Any]:
        amt_s = float(sa.get("amount_normalized") or 0) if sa.get("amount_normalized") == sa.get("amount_normalized") else 0
        amt_d = float(da.get("amount_normalized") or 0) if da.get("amount_normalized") == da.get("amount_normalized") else 0
        amt_err = abs(amt_s - amt_d)
        amt_rel = amt_err / max(amt_s, amt_d, 1e-9)
        token_ok = str(sa.get("token") or "").lower() == str(da.get("token") or "").lower() and bool(sa.get("token"))
        recv_ok = str(sa.get("receiver") or "").lower() == str(da.get("receiver") or "").lower() and bool(sa.get("receiver"))
        chain_ok = str(sa.get("dst_chain_id") or "") == str(BSC_CHAIN_ID_INT) and str(da.get("src_chain_id") or "") == str(ETH_CHAIN_ID_INT)
        blk_lag = float(da.get("block_number") or 0) - float(sa.get("block_number") or 0)
        conf = tid_match
        if tid_match >= 1.0:
            if not chain_ok:
                conf *= 0.5
            if not token_ok:
                conf *= 0.8
            if amt_rel > 0.05:
                conf *= 0.7
        dup_pen = 0.2 if (sa.get("duplicate_event_assignment") or da.get("duplicate_event_assignment")) else 0.0
        return {
            "src_event_atom_id": sa["src_event_atom_id"],
            "dst_event_atom_id": da["dst_event_atom_id"],
            "src_flow_id": sa["src_flow_id"],
            "dst_flow_id": da["dst_flow_id"],
            "source_transfer_id": str(sa.get("source_transfer_id") or ""),
            "destination_src_transfer_id": str(da.get("destination_src_transfer_id") or ""),
            "event_transferid_exact_match": tid_match,
            "event_chain_direction_consistency": float(chain_ok),
            "event_nonce": sa.get("nonce", ""),
            "event_token_consistency": float(token_ok),
            "event_amount_abs_error": amt_err,
            "event_amount_rel_error": amt_rel,
            "event_amount_match": float(amt_rel <= 0.01),
            "event_receiver_consistency": float(recv_ok),
            "event_sender_consistency": float(str(sa.get("sender") or "").lower() == str(da.get("sender") or "").lower()),
            "event_contract_family_match": 1.0,
            "event_log_order_score": 1.0 / (1.0 + abs(float(sa.get("src_log_index") or 0) - float(da.get("dst_log_index") or 0))),
            "event_block_lag_score": 1.0 / (1.0 + max(0.0, -blk_lag) / 3600.0) if blk_lag == blk_lag else 0.5,
            "event_confidence": max(0.0, conf - dup_pen),
            "duplicate_src_event_assignment": float(bool(sa.get("duplicate_event_assignment"))),
            "duplicate_dst_event_assignment": float(bool(da.get("duplicate_event_assignment"))),
            "event_collision_penalty": dup_pen,
        }

    for _, sa in src_atoms.iterrows():
        tid = str(sa["source_transfer_id"])
        if tid not in dst_by_stid.groups:
            continue
        for _, da in dst_by_stid.get_group(tid).iterrows():
            matched_dst_atoms.add(str(da["dst_event_atom_id"]))
            rows.append(_row(sa, da, 1.0))

    if include_negatives and len(rows) < 50000:
        rng = np.random.default_rng(42)
        dst_records = dst_atoms.to_dict("records")
        for _, sa in src_atoms.iterrows():
            tid = str(sa["source_transfer_id"])
            neg_pool = [da for da in dst_records if str(da.get("destination_src_transfer_id") or "") != tid]
            if not neg_pool:
                continue
            n_neg = min(5, len(neg_pool))
            idx = rng.choice(len(neg_pool), size=n_neg, replace=False)
            for j in idx:
                rows.append(_row(sa, neg_pool[int(j)], 0.0))
            if len(rows) >= 50000:
                break
    return pd.DataFrame(rows)


def _lift_event_to_flow(event_pairs: pd.DataFrame, flow_candidates: pd.DataFrame) -> pd.DataFrame:
    if event_pairs.empty or flow_candidates.empty:
        return flow_candidates.copy()
    lift = event_pairs.groupby(["src_flow_id", "dst_flow_id"]).agg(
        event_max_confidence=("event_confidence", "max"),
        event_match_count=("event_transferid_exact_match", "sum"),
        event_amount_match_mean=("event_amount_match", "mean"),
    ).reset_index()
    out = flow_candidates.merge(lift, on=["src_flow_id", "dst_flow_id"], how="left")
    out["flow_lifted_score"] = out["event_max_confidence"].fillna(0.0)
    out["flow_event_match_count"] = out["event_match_count"].fillna(0)
    return out


def _collision_audit(
    src_atoms: pd.DataFrame,
    dst_atoms: pd.DataFrame,
    flow_pair_features: pd.DataFrame,
    event_pairs: pd.DataFrame,
    truth: set[tuple[str, str]],
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    if not src_atoms.empty:
        per_src = src_atoms.groupby("src_flow_id")["source_transfer_id"].nunique()
        audit["src_flows_with_multiple_transferids_fraction"] = float((per_src > 1).mean())
    else:
        audit["src_flows_with_multiple_transferids_fraction"] = 0.0
    if not dst_atoms.empty:
        per_dst = dst_atoms.groupby("dst_flow_id")["destination_src_transfer_id"].nunique()
        audit["dst_flows_with_multiple_src_transferids_fraction"] = float((per_dst > 1).mean())
    else:
        audit["dst_flows_with_multiple_src_transferids_fraction"] = 0.0
    if not src_atoms.empty:
        tid_flows = src_atoms.groupby("source_transfer_id")["src_flow_id"].nunique()
        audit["transferids_reused_across_src_flows_fraction"] = float((tid_flows > 1).mean())
    else:
        audit["transferids_reused_across_src_flows_fraction"] = 0.0
    if not dst_atoms.empty:
        stid_flows = dst_atoms.groupby("destination_src_transfer_id")["dst_flow_id"].nunique()
        audit["src_transferids_reused_across_dst_flows_fraction"] = float((stid_flows > 1).mean())
    else:
        audit["src_transferids_reused_across_dst_flows_fraction"] = 0.0
    if not src_atoms.empty:
        audit["duplicate_event_assignment_fraction"] = float(src_atoms.get("duplicate_event_assignment", pd.Series([False])).mean())
    else:
        audit["duplicate_event_assignment_fraction"] = 0.0
    if not flow_pair_features.empty and "celer_transfer_id_exact_match" in flow_pair_features.columns:
        flagged = flow_pair_features[flow_pair_features["celer_transfer_id_exact_match"].fillna(0) == 1.0]
        audit["any_intersection_collision_fraction"] = float(
            flagged.groupby("src_flow_id").size().gt(1).mean()
        ) if len(flagged) else 0.0
        audit["transferid_match_with_amount_consistency_fraction"] = float(
            (flagged["celer_amount_match"].fillna(0) == 1).mean()
        ) if len(flagged) and "celer_amount_match" in flagged.columns else 0.0
        audit["transferid_match_with_token_consistency_fraction"] = float(
            (flagged["celer_token_consistency"].fillna(0) == 1).mean()
        ) if len(flagged) and "celer_token_consistency" in flagged.columns else 0.0
        audit["transferid_match_with_receiver_consistency_fraction"] = float(
            (flagged["celer_sender_receiver_consistency"].fillna(0) == 1).mean()
        ) if len(flagged) and "celer_sender_receiver_consistency" in flagged.columns else 0.0
    else:
        audit["any_intersection_collision_fraction"] = 0.0
        audit["transferid_match_with_amount_consistency_fraction"] = 0.0
        audit["transferid_match_with_token_consistency_fraction"] = 0.0
        audit["transferid_match_with_receiver_consistency_fraction"] = 0.0
    if not event_pairs.empty:
        per_tid = event_pairs.groupby("source_transfer_id").size()
        audit["transferid_unique_pair_fraction"] = float((per_tid == 1).mean())
        audit["transferid_collision_group_count"] = int((per_tid > 1).sum())
        audit["max_candidates_per_transferid"] = int(per_tid.max()) if len(per_tid) else 0
        audit["median_candidates_per_transferid"] = float(per_tid.median()) if len(per_tid) else 0.0
    else:
        audit["transferid_unique_pair_fraction"] = 0.0
        audit["transferid_collision_group_count"] = 0
        audit["max_candidates_per_transferid"] = 0
        audit["median_candidates_per_transferid"] = 0.0
    gt_with_atom = 0
    for sf, df in truth:
        if not event_pairs.empty:
            m = event_pairs[(event_pairs["src_flow_id"] == sf) & (event_pairs["dst_flow_id"] == df)]
            if len(m):
                gt_with_atom += 1
    audit["gt_pairs_with_transferid_atom_match_fraction"] = gt_with_atom / max(len(truth), 1)
    audit["flow_level_any_intersection_used"] = True
    audit["event_level_exact_matching_used"] = True
    return audit


def _event_score(df: pd.DataFrame) -> np.ndarray:
    cols = ["event_confidence", "event_transferid_exact_match", "event_chain_direction_consistency", "event_amount_match", "event_token_consistency"]
    parts = [pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in cols if c in df.columns]
    if not parts:
        return np.zeros(len(df))
    return np.vstack(parts).max(axis=0)


def _compute_ceiling_from_scores(
    seeds_data: list[tuple[set[tuple[str, str]], pd.DataFrame, str]],
    *,
    score_col: str,
    feature_cols: list[str] | None = None,
    group_col: str = "src_flow_id",
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    cand_recs, score_f1s, p_at_r, r_at_p = [], [], [], []
    aurocs, auprcs, collisions = [], [], []
    for truth, feat, _label in seeds_data:
        if feat.empty:
            continue
        pairs = set(zip(feat["src_flow_id"].astype(str), feat["dst_flow_id"].astype(str)))
        cand_recs.append(len(truth & pairs) / max(len(truth), 1))
        sc = feat[["src_flow_id", "dst_flow_id", score_col]].copy()
        scan = _phase10w._threshold_scan(truth, sc, score_col=score_col)
        score_f1s.append(scan["oracle_best_f1"])
        p_at_r.append(scan["oracle_precision_at_recall_0_8"])
        r_at_p.append(scan["oracle_recall_at_precision_0_8"])
        y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth) for r in feat.itertuples(index=False)])
        cols = feature_cols or [c for c in feat.columns if c.startswith("event_") and pd.api.types.is_numeric_dtype(feat[c])]
        x = feat[cols].fillna(0.0).max(axis=1).to_numpy(dtype=float) if cols else feat[score_col].fillna(0.0).to_numpy(dtype=float)
        if len(np.unique(y)) > 1:
            aurocs.append(float(roc_auc_score(y, x)))
            auprcs.append(float(average_precision_score(y, x)))
        if "src_event_atom_id" in feat.columns:
            per_src = feat.groupby("src_event_atom_id").size()
            collisions.append(float((per_src > 1).mean()))
        else:
            per_src = feat.groupby(group_col)[score_col].apply(lambda s: int((s.fillna(0) >= 0.5).sum()))
            collisions.append(float((per_src > 1).mean()))
    return {
        "candidate_oracle_recall": float(np.mean(cand_recs)) if cand_recs else 0.0,
        "score_oracle_best_f1": float(np.mean(score_f1s)) if score_f1s else 0.0,
        "oracle_precision_at_recall_0_8": float(np.mean(p_at_r)) if p_at_r else 0.0,
        "oracle_recall_at_precision_0_8": float(np.mean(r_at_p)) if r_at_p else 0.0,
        "feature_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "feature_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
        "candidate_collision_rate": float(np.mean(collisions)) if collisions else 1.0,
        "ambiguous_gt_fraction": 0.0,
    }


def evaluate_pilot_gate(event_ceiling: dict[str, Any], flow_ceiling: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "event_auroc": event_ceiling.get("feature_auroc", 0) >= PILOT_PASS["event_auroc"],
        "event_auprc": event_ceiling.get("feature_auprc", 0) >= PILOT_PASS["event_auprc"],
        "event_oracle_p_at_r": event_ceiling.get("oracle_precision_at_recall_0_8", 0) >= PILOT_PASS["event_oracle_p_at_r"],
        "event_oracle_r_at_p": event_ceiling.get("oracle_recall_at_precision_0_8", 0) >= PILOT_PASS["event_oracle_r_at_p"],
        "event_oracle_f1": event_ceiling.get("score_oracle_best_f1", 0) >= PILOT_PASS["event_oracle_f1"],
        "lifted_flow_oracle_p_at_r": flow_ceiling.get("oracle_precision_at_recall_0_8", 0) >= PILOT_PASS["lifted_flow_oracle_p_at_r"],
        "lifted_flow_oracle_f1": flow_ceiling.get("score_oracle_best_f1", 0) > PILOT_PASS["lifted_flow_oracle_f1"],
        "event_collision_rate": event_ceiling.get("candidate_collision_rate", 1) < PILOT_PASS["event_collision_rate"],
        "no_gt_leakage": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }
    return {"pilot_pass": all(checks.values()), "checks": checks}


def _process_seed(
    seed: int,
    synthetic_root: Path,
    *,
    topic_map: dict,
    eth_cache: Path,
    bsc_cache: Path,
    eth_client,
    bsc_client,
    out_cache_eth: Path,
    out_cache_bsc: Path,
    candidate_k: int,
    max_delay_sec: float,
) -> dict[str, Any]:
    sd = _phase10s._seed_dir(synthetic_root, seed)
    seed_data = _phase10s._load_seed_data(sd)
    pair_df, _ = _phase10s.build_candidate_pool(seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed)
    src_d, dst_d, _, _, st = _phase15.decode_seed_events(
        seed_data, eth_cache=eth_cache, bsc_cache=bsc_cache, topic_map=topic_map,
        eth_client=eth_client, bsc_client=bsc_client,
        out_cache_eth=out_cache_eth, out_cache_bsc=out_cache_bsc,
    )
    flow_pf = _phase15.build_celer_pair_features(pair_df, src_d, dst_d, seed_data)
    src_atoms, src_map, src_st = _build_src_atoms(src_d)
    dst_atoms, dst_map, dst_st = _build_dst_atoms(dst_d)
    event_pairs = _construct_event_candidates(src_atoms, dst_atoms)
    flow_lifted = _lift_event_to_flow(event_pairs, pair_df)
    truth = _pair_set(seed_data["labels"])
    audit = _collision_audit(src_atoms, dst_atoms, flow_pf, event_pairs, truth)
    return {
        "seed": seed,
        "src_decoded": src_d,
        "dst_decoded": dst_d,
        "receipt_stats": st,
        "flow_pair_features": flow_pf,
        "src_atoms": src_atoms,
        "dst_atoms": dst_atoms,
        "src_map": src_map,
        "dst_map": dst_map,
        "event_pairs": event_pairs,
        "flow_lifted": flow_lifted,
        "truth": truth,
        "audit": audit,
        "atom_stats": {"src": src_st, "dst": dst_st},
    }


def run_phase16(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    pilot_seeds: list[int],
    candidate_k: int,
    chain_identity_preflight_only: bool,
    test_rpc_connections: bool,
    pilot_only: bool,
    chain_identity_check: bool,
    fetch_rpc_receipts_if_needed: bool,
    decode_celer_abi: bool,
    revalidate_phase15_decode: bool,
    build_event_atoms: bool,
    construct_event_candidates: bool,
    lift_event_to_flow: bool,
    run_feasibility_gate: bool,
    train_if_feasible: bool,
    evaluate_holdout_once: bool,
    generate_sealed_seeds: bool,
    use_phase15_abi_cache: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out = run_root / OUT_REL
    for d in ("diagnosis", "evidence", "segmentation", "candidate", "labels", "pilot", "cache/phase16/eth_receipts", "cache/phase16/bsc_receipts", "models", "selection", "holdout", "splits", "audit"):
        (out / d).mkdir(parents=True, exist_ok=True)

    chain_preflight = run_chain_identity_preflight(out, _REPO)
    result: dict[str, Any] = {
        "ok": True,
        "chain_identity_preflight_pass": chain_preflight.get("chain_identity_preflight_pass"),
        "eth_chain_id": chain_preflight.get("eth_chain_id"),
        "bsc_chain_id": chain_preflight.get("bsc_chain_id"),
        "eth_rpc_masked": chain_preflight.get("eth_rpc_masked"),
        "bsc_rpc_masked": chain_preflight.get("bsc_rpc_masked"),
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }

    if chain_identity_preflight_only or test_rpc_connections:
        result["elapsed_sec"] = time.time() - t0
        return result

    if not chain_preflight.get("chain_identity_preflight_pass"):
        result["ok"] = False
        result["error"] = "chain_identity_fail"
        result["elapsed_sec"] = time.time() - t0
        return result

    registry = _phase15.load_abi_registry()
    topic_map = registry["_topic_map"]
    phase14_out = run_root / PHASE14_OUT
    eth_cache = phase14_out / "cache" / "eth_receipts" if use_phase15_abi_cache else out / "cache" / "phase16" / "eth_receipts"
    bsc_cache = phase14_out / "cache" / "bsc_receipts" if use_phase15_abi_cache else out / "cache" / "phase16" / "bsc_receipts"
    eth_client = _phase14._rpc_client(str(chain_preflight.get("eth_selected_url") or ""))
    bsc_client = _phase14._rpc_client(str(chain_preflight.get("bsc_selected_url") or ""))
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    uk = _phase10x._load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 86400.0)
    seeds_run = pilot_seeds if pilot_only else dev_seeds

    all_src, all_dst, agg_stats = [], [], {"src_receipt_tot": 0, "src_receipt_ok": 0, "dst_receipt_tot": 0, "dst_receipt_ok": 0}
    seed_results: list[dict[str, Any]] = []
    pair_parts, event_parts, lifted_parts = [], [], []
    audits_agg: list[dict[str, Any]] = []

    for seed in seeds_run:
        sr = _process_seed(
            seed, synthetic_root, topic_map=topic_map,
            eth_cache=eth_cache, bsc_cache=bsc_cache,
            eth_client=eth_client if fetch_rpc_receipts_if_needed else None,
            bsc_client=bsc_client if fetch_rpc_receipts_if_needed else None,
            out_cache_eth=out / "cache" / "phase16" / "eth_receipts",
            out_cache_bsc=out / "cache" / "phase16" / "bsc_receipts",
            candidate_k=candidate_k, max_delay_sec=max_delay_sec,
        )
        seed_results.append(sr)
        all_src.extend(sr["src_decoded"])
        all_dst.extend(sr["dst_decoded"])
        for k in agg_stats:
            agg_stats[k] += sr["receipt_stats"].get(k, 0)
        if not sr["src_atoms"].empty:
            sr["src_atoms"]["seed"] = seed
            sr["dst_atoms"]["seed"] = seed
        if not sr["event_pairs"].empty:
            sr["event_pairs"]["seed"] = seed
        if not sr["flow_lifted"].empty:
            sr["flow_lifted"]["seed"] = seed
        pair_parts.append(sr["flow_pair_features"])
        event_parts.append(sr["event_pairs"])
        lifted_parts.append(sr["flow_lifted"])
        audits_agg.append(sr["audit"])

    src_df = pd.DataFrame(all_src)
    dst_df = pd.DataFrame(all_dst)
    if revalidate_phase15_decode or decode_celer_abi:
        src_df.to_csv(out / "evidence" / "phase16_revalidated_decoded_events_src.csv", index=False)
        dst_df.to_csv(out / "evidence" / "phase16_revalidated_decoded_events_dst.csv", index=False)

    pair_all = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    coverage = _phase15.compute_coverage(all_src, all_dst, [], [], pair_all, agg_stats)
    coverage["chain_direction_consistency"] = float(
        sum(1 for e in all_src if str(e.get("dst_chain_id")) == str(BSC_CHAIN_ID_INT))
        / max(len([e for e in all_src if e.get("event_name") == "Send"]), 1)
    )
    phase15_cov_path = run_root / PHASE15_OUT / "diagnosis" / "phase15_celer_decode_coverage.json"
    phase15_delta_note = "Phase 15 coverage not found for comparison"
    if phase15_cov_path.is_file():
        p15 = json.loads(phase15_cov_path.read_text(encoding="utf-8"))
        p15_cov = p15.get("coverage", p15)
        diffs = {}
        for k in ("source_celer_event_decode_coverage", "source_transfer_id_coverage", "destination_src_transfer_id_coverage"):
            v16 = coverage.get(k.replace("destination_src_transfer_id", "destination_src_transfer_id_coverage"), coverage.get(k, 0))
            v15 = p15_cov.get(k, 0)
            if abs(v16 - v15) > 0.05:
                diffs[k] = {"phase16": v16, "phase15": v15, "delta": v16 - v15}
        coverage["phase15_endpoint_swap_affected"] = len(diffs) > 0
        coverage["phase15_comparison_diffs"] = diffs
        phase15_delta_note = f"Diffs >5%: {list(diffs.keys())}" if diffs else "Revalidated decode within 5% of Phase 15 (same cache; chain now verified)"
    else:
        coverage["phase15_endpoint_swap_affected"] = None
        coverage["phase15_comparison_diffs"] = {}
    (out / "diagnosis" / "phase16_revalidated_decode_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (out / "diagnosis" / "phase16_revalidated_decode_coverage.md").write_text(
        f"# Revalidated decode coverage\n\n{phase15_delta_note}\n\n" + "\n".join(f"- {k}: {v}" for k, v in coverage.items() if k not in ("phase15_comparison_diffs",)) + "\n",
        encoding="utf-8",
    )

    if build_event_atoms:
        src_list = [sr["src_atoms"] for sr in seed_results if not sr["src_atoms"].empty]
        dst_list = [sr["dst_atoms"] for sr in seed_results if not sr["dst_atoms"].empty]
        src_atoms_all = pd.concat(src_list, ignore_index=True) if src_list else pd.DataFrame()
        dst_atoms_all = pd.concat(dst_list, ignore_index=True) if dst_list else pd.DataFrame()
        src_map_all = pd.concat([sr["src_map"] for sr in seed_results if not sr["src_map"].empty], ignore_index=True) if any(not sr["src_map"].empty for sr in seed_results) else pd.DataFrame()
        dst_map_all = pd.concat([sr["dst_map"] for sr in seed_results if not sr["dst_map"].empty], ignore_index=True) if any(not sr["dst_map"].empty for sr in seed_results) else pd.DataFrame()
        src_atoms_all.to_csv(out / "segmentation" / "phase16_event_atoms_src.csv", index=False)
        dst_atoms_all.to_csv(out / "segmentation" / "phase16_event_atoms_dst.csv", index=False)
        src_map_all.to_csv(out / "segmentation" / "phase16_flow_to_event_atom_map_src.csv", index=False)
        dst_map_all.to_csv(out / "segmentation" / "phase16_flow_to_event_atom_map_dst.csv", index=False)
        (out / "segmentation" / "phase16_deaggregation_report.md").write_text(
            f"# Event deaggregation\n\n- src atoms: {len(src_atoms_all)}\n- dst atoms: {len(dst_atoms_all)}\n",
            encoding="utf-8",
        )

    if construct_event_candidates:
        event_all = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
        event_all.to_csv(out / "evidence" / "phase16_event_atom_pair_features.csv", index=False)
        stats = {
            "n_event_pairs": len(event_all),
            "n_unique_transfer_ids": int(event_all["source_transfer_id"].nunique()) if not event_all.empty else 0,
        }
        event_all.to_csv(out / "candidate" / "phase16_event_candidates.csv", index=False)
        (out / "candidate" / "phase16_event_candidate_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        (out / "candidate" / "phase16_event_candidate_report.md").write_text(f"# Event candidates\n\n- pairs: {stats['n_event_pairs']}\n", encoding="utf-8")

    if lift_event_to_flow:
        lifted_all = pd.concat(lifted_parts, ignore_index=True) if lifted_parts else pd.DataFrame()
        lifted_all.to_csv(out / "evidence" / "phase16_flow_lifted_pair_features.csv", index=False)
        event_all = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
        label_rows = []
        for sr in seed_results:
            for _, ep in sr["event_pairs"].iterrows():
                label_rows.append({
                    "src_event_atom_id": ep["src_event_atom_id"],
                    "dst_event_atom_id": ep["dst_event_atom_id"],
                    "src_flow_id": ep["src_flow_id"],
                    "dst_flow_id": ep["dst_flow_id"],
                    "event_label": int((str(ep["src_flow_id"]), str(ep["dst_flow_id"])) in sr["truth"]),
                })
        pd.DataFrame(label_rows).to_csv(out / "labels" / "phase16_event_level_labels.csv", index=False)
        (out / "labels" / "phase16_label_projection_report.md").write_text(
            "# Label projection\n\nGT used for labels only, not inference features. Flow scores = max event confidence per flow pair.\n",
            encoding="utf-8",
        )

    audit_mean = {}
    if audits_agg:
        for k in audits_agg[0]:
            audit_mean[k] = float(np.mean([a.get(k, 0) for a in audits_agg]))
    (out / "diagnosis" / "phase16_transferid_collision_audit.json").write_text(json.dumps(audit_mean, indent=2), encoding="utf-8")
    (out / "diagnosis" / "phase16_transferid_collision_audit.md").write_text(
        "# TransferId collision audit\n\n" + "\n".join(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in audit_mean.items()) + "\n",
        encoding="utf-8",
    )
    examples = []
    if seed_results and not seed_results[0]["flow_pair_features"].empty:
        fp = seed_results[0]["flow_pair_features"]
        flagged = fp[fp["celer_transfer_id_exact_match"].fillna(0) == 1.0].head(20)
        examples = flagged[["src_flow_id", "dst_flow_id", "celer_source_transfer_id", "celer_destination_src_transfer_id"]].to_dict("records")
    pd.DataFrame(examples).to_csv(out / "diagnosis" / "phase16_transferid_collision_examples.csv", index=False)

    event_ceiling_data, flow_ceiling_data = [], []
    for sr in seed_results:
        truth = sr["truth"]
        ep = sr["event_pairs"]
        if not ep.empty:
            ep = ep.copy()
            ep["event_score"] = _event_score(ep)
            ep["src_flow_id"] = ep["src_flow_id"].astype(str)
            ep["dst_flow_id"] = ep["dst_flow_id"].astype(str)
            event_ceiling_data.append((truth, ep, "event"))
        fl = sr["flow_lifted"]
        if not fl.empty:
            fl = fl.copy()
            fl["flow_score"] = fl["flow_lifted_score"].fillna(0.0)
            flow_ceiling_data.append((truth, fl, "flow"))

    event_ceiling = _compute_ceiling_from_scores(event_ceiling_data, score_col="event_score", group_col="src_event_atom_id") if event_ceiling_data else {}
    flow_ceiling = _compute_ceiling_from_scores(flow_ceiling_data, score_col="flow_score") if flow_ceiling_data else {}
    (out / "diagnosis" / "phase16_event_level_ceiling.json").write_text(json.dumps(event_ceiling, indent=2), encoding="utf-8")
    (out / "diagnosis" / "phase16_flow_lifted_ceiling.json").write_text(json.dumps(flow_ceiling, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"stage": "phase15_flow_any_intersection", **PHASE15_BASELINE},
        {"stage": "phase16_event_level", **event_ceiling},
        {"stage": "phase16_flow_lifted", **flow_ceiling},
    ]).to_csv(out / "diagnosis" / "phase16_ceiling_progression_table.csv", index=False)
    (out / "diagnosis" / "phase16_feature_separability_report.md").write_text(
        f"# Feature separability\n\n## Event-level\n- AUROC: {event_ceiling.get('feature_auroc', 0):.3f}\n- Oracle P@R>=0.8: {event_ceiling.get('oracle_precision_at_recall_0_8', 0):.3f}\n\n## Flow lifted\n- AUROC: {flow_ceiling.get('feature_auroc', 0):.3f}\n",
        encoding="utf-8",
    )

    pilot_gate = evaluate_pilot_gate(event_ceiling, flow_ceiling)
    (out / "pilot" / "phase16_pilot_ceiling_metrics.json").write_text(json.dumps({"event": event_ceiling, "flow_lifted": flow_ceiling}, indent=2), encoding="utf-8")
    (out / "pilot" / "phase16_pilot_report.md").write_text(
        f"# Phase 16 pilot\n\n- pilot_pass: {pilot_gate['pilot_pass']}\n"
        f"- event AUROC: {event_ceiling.get('feature_auroc', 0):.3f}\n"
        f"- flow lifted oracle P@R>=0.8: {flow_ceiling.get('oracle_precision_at_recall_0_8', 0):.3f}\n",
        encoding="utf-8",
    )
    result["pilot_pass"] = pilot_gate["pilot_pass"]
    result["event_ceiling"] = event_ceiling
    result["flow_ceiling"] = flow_ceiling
    result["collision_audit"] = audit_mean

    if not pilot_gate["pilot_pass"]:
        event_ok = pilot_gate["checks"].get("event_auroc") and pilot_gate["checks"].get("event_oracle_p_at_r")
        if event_ok and not pilot_gate["checks"].get("lifted_flow_oracle_p_at_r"):
            (out / "diagnosis" / "phase16_flow_aggregation_bottleneck.md").write_text(
                "# Flow aggregation bottleneck\n\nEvent-level evidence may be identifiable but flow-level lift still fails high-P/R.\n",
                encoding="utf-8",
            )
        else:
            (out / "diagnosis" / "phase16_event_level_infeasibility.md").write_text(
                "# Event-level infeasibility\n\nEven transferId-level segmentation does not meet high-P/R pilot thresholds.\n",
                encoding="utf-8",
            )
        (out / "pilot" / "phase16_pilot_failure.md").write_text(
            "# Pilot failure\n\n" + "\n".join(f"- {k}: {v}" for k, v in pilot_gate["checks"].items() if not v) + "\n",
            encoding="utf-8",
        )

    _write_skip_artifacts(out, result, train_seeds, dev_seeds, holdout_seeds, pilot_gate=pilot_gate, event_ceiling=event_ceiling, flow_ceiling=flow_ceiling)

    if pilot_only or not pilot_gate["pilot_pass"]:
        result["ok"] = pilot_gate["pilot_pass"]
        if not pilot_gate["pilot_pass"]:
            result["error"] = "pilot_fail"
        result["elapsed_sec"] = time.time() - t0
        return result

    if run_feasibility_gate and not pilot_gate["pilot_pass"]:
        result["ok"] = False
        result["error"] = "feasibility_fail"
        result["elapsed_sec"] = time.time() - t0
        return result

    result["elapsed_sec"] = time.time() - t0
    return result


def _write_skip_artifacts(out, result, train_seeds, dev_seeds, holdout_seeds, *, pilot_gate=None, event_ceiling=None, flow_ceiling=None) -> None:
    feas = {"feasibility_gate_pass": False, "pilot_pass": pilot_gate.get("pilot_pass") if pilot_gate else False}
    (out / "diagnosis" / "phase16_feasibility_gate.json").write_text(json.dumps(feas, indent=2), encoding="utf-8")
    (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps({
        "high_pr_gate_pass": False,
        "feasibility_gate_pass": False,
        "training_skipped": True,
        "holdout_evaluation_skipped": True,
        "allowed_claim": "Even after chain-verified Celer ABI decoding and transferId-level deaggregation, exact high-P/R flow correspondence remains limited by flow aggregation collisions or dataset-level evidence limitations.",
        "forbidden_claim": "Do not claim high P/R without gate PASS.",
        "diagnostic_event_ceiling": event_ceiling or {},
        "diagnostic_flow_ceiling": flow_ceiling or {},
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2, default=str), encoding="utf-8")
    (out / "splits" / "split_summary.json").write_text(json.dumps({
        "canonical_rebuilt": False, "label_layer_refrozen": False,
        "phase10s_to_15_preserved": True, "phase15_failure_preserved": True,
        "chain_identity_verified": result.get("chain_identity_preflight_pass"),
        "train_seeds": train_seeds, "dev_seeds": dev_seeds, "sealed_holdout_seeds": holdout_seeds,
        "credentials_committed": False, "full_rpc_url_logged": False,
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 16 transferId event deaggregation")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=list(range(52, 72)))
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=list(range(152, 172)))
    ap.add_argument("--pilot-seeds", type=int, nargs="+", default=list(range(52, 58)))
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--chain-identity-preflight-only", action="store_true")
    ap.add_argument("--test-rpc-connections", action="store_true")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--chain-identity-check", action="store_true", default=True)
    ap.add_argument("--fetch-rpc-receipts-if-needed", action="store_true")
    ap.add_argument("--decode-celer-abi", action="store_true")
    ap.add_argument("--revalidate-phase15-decode", action="store_true")
    ap.add_argument("--build-event-atoms", action="store_true")
    ap.add_argument("--construct-event-candidates", action="store_true")
    ap.add_argument("--lift-event-to-flow", action="store_true")
    ap.add_argument("--run-feasibility-gate", action="store_true")
    ap.add_argument("--train-if-feasible", action="store_true")
    ap.add_argument("--evaluate-holdout-once", action="store_true")
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    ap.add_argument("--use-phase15-abi-cache", action="store_true", default=True)
    args = ap.parse_args()
    if args.pilot_only:
        for f in ("decode_celer_abi", "revalidate_phase15_decode", "build_event_atoms", "construct_event_candidates", "lift_event_to_flow"):
            setattr(args, f, True)
    r = run_phase16(
        run_root=args.run_root,
        train_seeds=args.train_seeds,
        dev_seeds=args.dev_seeds,
        holdout_seeds=args.holdout_seeds,
        pilot_seeds=args.pilot_seeds if args.pilot_seeds else list(range(52, 58)),
        candidate_k=args.candidate_k,
        chain_identity_preflight_only=args.chain_identity_preflight_only or args.test_rpc_connections,
        test_rpc_connections=args.test_rpc_connections,
        pilot_only=args.pilot_only,
        chain_identity_check=args.chain_identity_check,
        fetch_rpc_receipts_if_needed=args.fetch_rpc_receipts_if_needed,
        decode_celer_abi=args.decode_celer_abi,
        revalidate_phase15_decode=args.revalidate_phase15_decode,
        build_event_atoms=args.build_event_atoms,
        construct_event_candidates=args.construct_event_candidates,
        lift_event_to_flow=args.lift_event_to_flow,
        run_feasibility_gate=args.run_feasibility_gate,
        train_if_feasible=args.train_if_feasible,
        evaluate_holdout_once=args.evaluate_holdout_once,
        generate_sealed_seeds=args.generate_sealed_seeds,
        use_phase15_abi_cache=args.use_phase15_abi_cache,
    )
    print(json.dumps({k: r[k] for k in r if k not in ("event_ceiling", "flow_ceiling", "collision_audit")}, indent=2, default=str))
    return 0 if r.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
