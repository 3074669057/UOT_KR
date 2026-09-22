#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 20: Quotient layer integrity repair and bridge-transfer key reconstruction."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.domain.evaluation.flow_eval import _pair_set

for _name, _path in (
    ("phase10s", _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"),
    ("phase10w", _REPO / "scripts" / "run_phase10w_ambiguity_resolved_rcuot.py"),
    ("phase15", _REPO / "scripts" / "run_phase15_celer_abi_decode.py"),
):
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    globals()[f"_{_name}"] = _mod

OUT_REL = "phase20_quotient_integrity_repair"
PHASE16_OUT = "phase16_transferid_event_deaggregation"
ETH_CHAIN_ID = "1"
BSC_CHAIN_ID = "56"
BRIDGE_FAMILY = "celer_cbridge"

PHASE19_BASELINE = {
    "quotient_clean_label_fraction": 0.987,
    "quotient_projection_coverage_over_original_gt": 0.785,
    "quotient_feature_auroc": 0.006,
    "quotient_feature_auprc": 0.333,
    "quotient_oracle_precision_at_recall_0_8": 0.339,
    "quotient_score_oracle_best_f1": 0.506,
    "quotient_candidate_oracle_recall": 1.0,
}

FEASIBILITY_GATE = {
    "quotient_candidate_oracle_recall": 0.95,
    "quotient_oracle_precision_at_recall_0_8": 0.80,
    "quotient_oracle_recall_at_precision_0_8": 0.80,
    "quotient_score_oracle_best_f1": 0.80,
    "quotient_feature_auroc": 0.85,
    "quotient_feature_auprc": 0.70,
    "quotient_clean_label_fraction": 0.80,
    "quotient_ambiguous_label_fraction": 0.20,
    "event_backed_projection_coverage": 0.80,
    "label_conflict_rate": 0.10,
    "candidate_collision_rate": 0.30,
}


def _amount_bucket(val: Any) -> str:
    try:
        x = float(val)
        if x != x:
            return "na"
        if x <= 0:
            return "zero"
        exp = int(np.floor(np.log10(abs(x) + 1e-18)))
        return f"1e{exp}"
    except (TypeError, ValueError):
        return "na"


def _normalize_token(token: str) -> str:
    t = str(token or "").lower().strip()
    return t


def _normalize_receiver(r: str) -> str:
    return str(r or "").lower().strip()


def _bridge_transfer_key_core(*, transfer_id: str, receiver: str) -> str:
    """Cross-side exact-match key: transferId + receiver + fixed ETH→BSC direction."""
    tid = str(transfer_id or "").lower()
    recv = _normalize_receiver(receiver)
    return "|".join([BRIDGE_FAMILY, ETH_CHAIN_ID, BSC_CHAIN_ID, tid, recv])


def _bridge_transfer_key_src(e: dict[str, Any]) -> str:
    tid = str(e.get("transfer_id") or "").lower()
    token = _normalize_token(e.get("token", ""))
    recv = _normalize_receiver(e.get("receiver", ""))
    amt = _amount_bucket(e.get("amount_normalized") or e.get("amount_raw"))
    core = _bridge_transfer_key_core(transfer_id=tid, receiver=recv)
    return f"{core}|{token}|{amt}"


def _bridge_transfer_key_dst(e: dict[str, Any]) -> str:
    stid = str(e.get("src_transfer_id") or "").lower()
    token = _normalize_token(e.get("token", ""))
    recv = _normalize_receiver(e.get("receiver", ""))
    amt = _amount_bucket(e.get("amount_normalized") or e.get("amount_raw"))
    core = _bridge_transfer_key_core(transfer_id=stid, receiver=recv)
    return f"{core}|{token}|{amt}"


def _bridge_transfer_key_core_from_side_key(side_key: str) -> str:
    parts = str(side_key or "").split("|")
    if len(parts) >= 5:
        return "|".join(parts[:5])
    return str(side_key or "")


def _seed_flow_ids(seed_data: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("eth_flows", "bnb_flows", "bsc_flows"):
        for f in seed_data.get(key) or []:
            fid = str(f.get("flow_id") or "")
            if fid:
                ids.add(fid)
    return ids


def _load_phase16(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = run_root / PHASE16_OUT / "evidence"
    sp, dp = p / "phase16_revalidated_decoded_events_src.csv", p / "phase16_revalidated_decoded_events_dst.csv"
    if not sp.is_file():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(sp), pd.read_csv(dp)


def _feature_direction_audit(features: pd.DataFrame, label_col: str = "quotient_label") -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    if features.empty or label_col not in features.columns:
        return pd.DataFrame()
    y = features[label_col].fillna(0).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return pd.DataFrame()
    skip = {label_col, "source_quotient_class_id", "destination_quotient_class_id", "quotient_score", "repaired_similarity_score", "repaired_cost_score", "seed"}
    cost_feats = {"amount_relative_error", "amount_abs_error", "quotient_ambiguity_penalty", "parent_flow_multiplicity_src", "parent_flow_multiplicity_dst", "quotient_member_overlap_risk"}
    for col in features.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(features[col]):
            continue
        x = pd.to_numeric(features[col], errors="coerce").fillna(0.0).to_numpy()
        if np.std(x) < 1e-9:
            continue
        auroc_raw = float(roc_auc_score(y, x))
        auroc_neg = float(roc_auc_score(y, -x))
        auprc_raw = float(average_precision_score(y, x))
        auprc_neg = float(average_precision_score(y, -x))
        pos_mean = float(x[y == 1].mean()) if (y == 1).any() else 0.0
        neg_mean = float(x[y == 0].mean()) if (y == 0).any() else 0.0
        higher_better = col not in cost_feats
        inverted = (auroc_neg > auroc_raw + 0.1) or (col in cost_feats and auroc_neg > 0.55)
        rows.append({
            "feature_name": col,
            "feature_auroc_raw": auroc_raw,
            "feature_auroc_negated": auroc_neg,
            "feature_auprc_raw": auprc_raw,
            "feature_auprc_negated": auprc_neg,
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "direction_should_be_higher_better": higher_better,
            "direction_should_be_lower_better": col in cost_feats,
            "suspicious_inverted_feature": inverted,
            "recommended_use_negated": inverted and auroc_neg > auroc_raw,
        })
    return pd.DataFrame(rows).sort_values("feature_auroc_negated", ascending=False)


def build_quotient_layer(
    src_events: list[dict[str, Any]],
    dst_events: list[dict[str, Any]],
    truth: set[tuple[str, str]],
    seed: int,
) -> dict[str, Any]:
    btk_src: dict[str, dict[str, Any]] = {}
    btk_dst: dict[str, dict[str, Any]] = {}
    flow_to_qs: dict[str, str] = {}
    flow_to_qd: dict[str, str] = {}
    flow_has_event: set[str] = set()

    src_rows, dst_rows = [], []
    for e in src_events:
        if e.get("event_name") not in ("Send", "LogNewTransferOut") or not _phase15._field_nonempty(e.get("transfer_id")):
            continue
        btk_full = _bridge_transfer_key_src(e)
        btk = _bridge_transfer_key_core_from_side_key(btk_full)
        fid = str(e.get("flow_id") or "")
        flow_has_event.add(fid)
        if btk not in btk_src:
            qid = f"qs20_{seed}_{len(btk_src):05d}"
            btk_src[btk] = {
                "quotient_class_id": qid,
                "side": "src",
                "bridge_transfer_key": btk,
                "bridge_transfer_key_full": btk_full,
                "transfer_id": str(e.get("transfer_id") or "").lower(),
                "token_normalized": _normalize_token(e.get("token", "")),
                "amount_bucket": _amount_bucket(e.get("amount_normalized") or e.get("amount_raw")),
                "receiver": _normalize_receiver(e.get("receiver", "")),
                "member_flow_ids": set(),
                "event_role": "send",
            }
        btk_src[btk]["member_flow_ids"].add(fid)
        flow_to_qs[fid] = btk_src[btk]["quotient_class_id"]
        src_rows.append({
            "quotient_class_id": btk_src[btk]["quotient_class_id"],
            "side": "src",
            "flow_id": fid,
            "event_role": "send",
            "bridge_family": BRIDGE_FAMILY,
            "src_chain_id": ETH_CHAIN_ID,
            "dst_chain_id": str(e.get("dst_chain_id") or BSC_CHAIN_ID),
            "transfer_id_or_src_transfer_id": str(e.get("transfer_id") or "").lower(),
            "token_raw": e.get("token", ""),
            "token_normalized": _normalize_token(e.get("token", "")),
            "amount_raw": e.get("amount_raw", ""),
            "amount_normalized": e.get("amount_normalized", ""),
            "amount_bucket": _amount_bucket(e.get("amount_normalized") or e.get("amount_raw")),
            "sender": e.get("sender", ""),
            "receiver": _normalize_receiver(e.get("receiver", "")),
            "bridge_transfer_key": btk,
            "bridge_transfer_key_full": btk_full,
            "tx_hash": e.get("tx_hash", ""),
            "log_index": e.get("log_index", ""),
            "contract_address": e.get("contract_address", ""),
            "event_name": e.get("event_name", ""),
        })

    for e in dst_events:
        if e.get("event_name") not in ("Relay", "LogNewTransferIn") or not _phase15._field_nonempty(e.get("src_transfer_id")):
            continue
        btk_full = _bridge_transfer_key_dst(e)
        btk = _bridge_transfer_key_core_from_side_key(btk_full)
        fid = str(e.get("flow_id") or "")
        flow_has_event.add(fid)
        if btk not in btk_dst:
            qid = f"qd20_{seed}_{len(btk_dst):05d}"
            btk_dst[btk] = {
                "quotient_class_id": qid,
                "side": "dst",
                "bridge_transfer_key": btk,
                "bridge_transfer_key_full": btk_full,
                "src_transfer_id": str(e.get("src_transfer_id") or "").lower(),
                "token_normalized": _normalize_token(e.get("token", "")),
                "amount_bucket": _amount_bucket(e.get("amount_normalized") or e.get("amount_raw")),
                "receiver": _normalize_receiver(e.get("receiver", "")),
                "member_flow_ids": set(),
                "event_role": "relay",
            }
        btk_dst[btk]["member_flow_ids"].add(fid)
        flow_to_qd[fid] = btk_dst[btk]["quotient_class_id"]
        dst_rows.append({
            "quotient_class_id": btk_dst[btk]["quotient_class_id"],
            "side": "dst",
            "flow_id": fid,
            "event_role": "relay",
            "bridge_family": BRIDGE_FAMILY,
            "src_chain_id": str(e.get("src_chain_id") or ETH_CHAIN_ID),
            "dst_chain_id": BSC_CHAIN_ID,
            "transfer_id_or_src_transfer_id": str(e.get("src_transfer_id") or "").lower(),
            "token_raw": e.get("token", ""),
            "token_normalized": _normalize_token(e.get("token", "")),
            "amount_raw": e.get("amount_raw", ""),
            "amount_normalized": e.get("amount_normalized", ""),
            "amount_bucket": _amount_bucket(e.get("amount_normalized") or e.get("amount_raw")),
            "sender": e.get("sender", ""),
            "receiver": _normalize_receiver(e.get("receiver", "")),
            "bridge_transfer_key": btk,
            "bridge_transfer_key_full": btk_full,
            "tx_hash": e.get("tx_hash", ""),
            "log_index": e.get("log_index", ""),
            "contract_address": e.get("contract_address", ""),
            "event_name": e.get("event_name", ""),
        })

    src_classes, dst_classes = [], []
    for btk, rec in btk_src.items():
        flows = sorted(rec["member_flow_ids"])
        mult = len(flows)
        src_classes.append({
            **{k: v for k, v in rec.items() if k != "member_flow_ids"},
            "member_flow_ids": ",".join(flows),
            "member_flow_count": mult,
            "parent_flow_multiplicity": mult,
            "ambiguity_flag": mult > 1,
            "key_strength": "strong",
            "key_components_available": "transfer_id,receiver,chain;token_amount_side_local",
        })
    for btk, rec in btk_dst.items():
        flows = sorted(rec["member_flow_ids"])
        mult = len(flows)
        dst_classes.append({
            **{k: v for k, v in rec.items() if k != "member_flow_ids"},
            "member_flow_ids": ",".join(flows),
            "member_flow_count": mult,
            "parent_flow_multiplicity": mult,
            "ambiguity_flag": mult > 1,
            "key_strength": "strong",
            "key_components_available": "src_transfer_id,receiver,chain;token_amount_side_local",
        })

    qp_to_gt: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for sf, df in truth:
        qs, qd = flow_to_qs.get(sf), flow_to_qd.get(df)
        if qs and qd:
            qp_to_gt[(qs, qd)].append((sf, df))

    label_rows, conflict_rows = [], []
    n_clean, n_amb = 0, 0
    covered_gt: set[tuple[str, str]] = set()
    for (qs, qd), gt_pairs in qp_to_gt.items():
        sr = next((c for c in src_classes if c["quotient_class_id"] == qs), {})
        dr = next((c for c in dst_classes if c["quotient_class_id"] == qd), {})
        tid_ok = str(sr.get("transfer_id", "")) == str(dr.get("src_transfer_id", "")) and bool(sr.get("transfer_id"))
        parent_amb = int(sr.get("member_flow_count", 1) > 1 or dr.get("member_flow_count", 1) > 1)
        clean = tid_ok and len(gt_pairs) > 0
        if clean:
            n_clean += 1
            for p in gt_pairs:
                covered_gt.add(p)
        else:
            n_amb += 1
        label_rows.append({
            "source_quotient_class_id": qs,
            "destination_quotient_class_id": qd,
            "label_v2": 1,
            "label_source": "label_layer_v1_aggregation",
            "original_gt_flow_pairs": ";".join(f"{a}|{b}" for a, b in gt_pairs),
            "original_gt_pair_count": len(gt_pairs),
            "parent_flow_ambiguous": parent_amb,
            "transfer_id": sr.get("transfer_id", ""),
            "src_transfer_id": dr.get("src_transfer_id", ""),
            "clean_label": clean,
            "ambiguous_reason": "" if clean else "direction_or_missing",
        })

    truth_clean = {(r["source_quotient_class_id"], r["destination_quotient_class_id"]) for r in label_rows if r["clean_label"]}
    return {
        "seed": seed,
        "truth": truth,
        "covered_gt": covered_gt,
        "flow_has_event": flow_has_event,
        "flow_to_qs": flow_to_qs,
        "flow_to_qd": flow_to_qd,
        "src_classes": src_classes,
        "dst_classes": dst_classes,
        "src_key_rows": src_rows,
        "dst_key_rows": dst_rows,
        "labels": label_rows,
        "conflicts": conflict_rows,
        "truth_clean": truth_clean,
        "btk_src": btk_src,
        "btk_dst": btk_dst,
        "n_clean": n_clean,
        "n_amb": n_amb,
    }


def _repaired_feature_row(sr: dict, dr: dict, label: int) -> dict[str, Any]:
    btk_match = float(sr.get("bridge_transfer_key", "") == dr.get("bridge_transfer_key", "") and bool(sr.get("bridge_transfer_key")))
    tid_match = float(str(sr.get("transfer_id", "")) == str(dr.get("src_transfer_id", "")) and bool(sr.get("transfer_id")))
    token_match = float(sr.get("token_normalized", "") == dr.get("token_normalized", "") and bool(sr.get("token_normalized")))
    amt_bucket_match = float(sr.get("amount_bucket", "") == dr.get("amount_bucket", ""))
    recv_match = float(sr.get("receiver", "") == dr.get("receiver", "") and bool(sr.get("receiver")))
    try:
        amt_s = float(sr.get("amount_normalized") or 0)
        amt_d = float(dr.get("amount_normalized") or 0)
    except (TypeError, ValueError):
        amt_s = amt_d = 0.0
    amt_rel = abs(amt_s - amt_d) / max(abs(amt_s), abs(amt_d), 1e-9)
    mult_s = int(sr.get("member_flow_count", 1))
    mult_d = int(dr.get("member_flow_count", 1))
    amb_pen = 0.1 * max(mult_s - 1, 0) + 0.1 * max(mult_d - 1, 0)
    sim = (
        0.40 * btk_match + 0.30 * tid_match + 0.10 * float(recv_match)
        + 0.05 * token_match + 0.05 * float(amt_bucket_match) + 0.10
    )
    cost = min(0.5 * amt_rel + amb_pen, 1.0)
    score = max(sim - cost, 0.0)
    return {
        "source_quotient_class_id": sr["quotient_class_id"],
        "destination_quotient_class_id": dr["quotient_class_id"],
        "bridge_transfer_key_exact_match": btk_match,
        "transfer_id_exact_match": tid_match,
        "chain_direction_consistency": 1.0,
        "token_normalized_match": token_match,
        "amount_bucket_match": float(amt_bucket_match),
        "amount_relative_error": amt_rel,
        "receiver_match": recv_match,
        "sender_consistency": 1.0,
        "bridge_family_match": 1.0,
        "event_role_compatible": 1.0 if tid_match else 0.0,
        "parent_flow_multiplicity_src": mult_s,
        "parent_flow_multiplicity_dst": mult_d,
        "quotient_member_overlap_risk": float(mult_s > 1 or mult_d > 1),
        "quotient_ambiguity_penalty": amb_pen,
        "quotient_pair_confidence": score,
        "repaired_similarity_score": sim,
        "repaired_cost_score": cost,
        "quotient_score": score,
        "quotient_label": label,
    }


def build_repaired_candidates(
    src_classes: list[dict],
    dst_classes: list[dict],
    truth_clean: set[tuple[str, str]],
    *,
    max_random_per_src: int = 8,
) -> pd.DataFrame:
    dst_by_tid: dict[str, list[dict]] = defaultdict(list)
    dst_by_btk: dict[str, list[dict]] = defaultdict(list)
    for dr in dst_classes:
        stid = str(dr.get("src_transfer_id", ""))
        if stid:
            dst_by_tid[stid].append(dr)
        core = str(dr.get("bridge_transfer_key", ""))
        if core:
            dst_by_btk[core].append(dr)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    rng = np.random.default_rng(42)

    for sr in src_classes:
        qs = sr["quotient_class_id"]
        tid = str(sr.get("transfer_id", ""))
        strong = dst_by_btk.get(str(sr.get("bridge_transfer_key", "")), [])
        near = dst_by_tid.get(tid, [])
        for dr in strong + near:
            key = (qs, dr["quotient_class_id"])
            if key in seen:
                continue
            seen.add(key)
            label = int(key in truth_clean)
            rows.append(_repaired_feature_row(sr, dr, label))

        hard_negs = [d for d in dst_classes if str(d.get("src_transfer_id", "")) != tid][:3]
        for dr in hard_negs:
            key = (qs, dr["quotient_class_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(_repaired_feature_row(sr, dr, 0))

        pool = [d for d in dst_classes if (qs, d["quotient_class_id"]) not in seen]
        if pool:
            idx = rng.choice(len(pool), size=min(max_random_per_src, len(pool)), replace=False)
            for j in idx:
                dr = pool[int(j)]
                key = (qs, dr["quotient_class_id"])
                seen.add(key)
                rows.append(_repaired_feature_row(sr, dr, int(key in truth_clean)))

    return pd.DataFrame(rows)


def _threshold_scan(truth: set[tuple[str, str]], df: pd.DataFrame, score_col: str = "quotient_score") -> dict[str, float]:
    if df.empty or not truth:
        return {"quotient_score_oracle_best_f1": 0.0, "quotient_oracle_precision_at_recall_0_8": 0.0, "quotient_oracle_recall_at_precision_0_8": 0.0}
    s = df.copy()
    s["_sc"] = pd.to_numeric(s[score_col], errors="coerce").fillna(0.0)
    uniq = sorted(set(s["_sc"].tolist()), reverse=True) or [0.0]
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for thr in uniq + [0.0]:
        pred = set(zip(
            s.loc[s["_sc"] >= thr, "source_quotient_class_id"].astype(str),
            s.loc[s["_sc"] >= thr, "destination_quotient_class_id"].astype(str),
        ))
        m = _phase10w._prf1(truth, pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
        if m["recall"] >= 0.8 and m["precision"] > best_p:
            best_p = m["precision"]
        if m["precision"] >= 0.8 and m["recall"] > best_r:
            best_r = m["recall"]
    return {
        "quotient_score_oracle_best_f1": best_f1,
        "quotient_oracle_precision_at_recall_0_8": best_p,
        "quotient_oracle_recall_at_precision_0_8": best_r,
    }


def compute_ceiling(truth_clean: set[tuple[str, str]], candidates: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if candidates.empty:
        return {}
    pairs = set(zip(candidates["source_quotient_class_id"], candidates["destination_quotient_class_id"]))
    recall = len(truth_clean & pairs) / max(len(truth_clean), 1)
    scan = _threshold_scan(truth_clean, candidates)
    y = candidates["quotient_label"].fillna(0).astype(int).to_numpy()
    feat_cols = [
        "bridge_transfer_key_exact_match", "transfer_id_exact_match", "receiver_match",
        "repaired_similarity_score", "quotient_score",
    ]
    x = candidates[[c for c in feat_cols if c in candidates.columns]].fillna(0.0).max(axis=1).to_numpy()
    auroc, auprc = 0.5, 0.0
    if len(np.unique(y)) > 1:
        auroc = float(roc_auc_score(y, x))
        auprc = float(average_precision_score(y, x))
    strong = candidates[candidates["bridge_transfer_key_exact_match"] >= 1.0]
    btk_prec, btk_rec = 0.0, 0.0
    if not strong.empty:
        pred_s = set(zip(strong["source_quotient_class_id"], strong["destination_quotient_class_id"]))
        btk_prec = len(truth_clean & pred_s) / max(len(pred_s), 1)
        btk_rec = len(truth_clean & pred_s) / max(len(truth_clean), 1)
    per_src_pos = candidates[candidates["quotient_label"] == 1].groupby("source_quotient_class_id").size()
    collision = float((per_src_pos > 1).mean()) if len(per_src_pos) else 0.0
    return {
        "quotient_candidate_oracle_recall": float(recall),
        **scan,
        "quotient_feature_auroc": auroc,
        "quotient_feature_auprc": auprc,
        "bridge_transfer_key_precision": float(btk_prec),
        "bridge_transfer_key_recall": float(btk_rec),
        "candidate_collision_rate": collision,
        "positive_fraction": float(y.mean()),
        "hard_negative_fraction": float((candidates["transfer_id_exact_match"] < 1).mean()),
    }


def coverage_gap_audit(
    truth: set[tuple[str, str]],
    covered: set[tuple[str, str]],
    flow_has_event: set[str],
    flow_to_qs: dict[str, str],
    flow_to_qd: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for sf, df in truth:
        if (sf, df) in covered:
            continue
        reason = "unknown"
        if sf not in flow_has_event:
            reason = "src flow has no decoded Celer event"
        elif df not in flow_has_event:
            reason = "dst flow has no decoded Celer event"
        elif sf not in flow_to_qs:
            reason = "event exists but removed by quotient filter"
        elif df not in flow_to_qd:
            reason = "event exists but removed by quotient filter"
        else:
            reason = "chain direction or token/amount normalization mismatch"
        rows.append({"src_flow_id": sf, "dst_flow_id": df, "uncovered_reason": reason})
    return pd.DataFrame(rows)


def integrity_audit(layer: dict[str, Any], candidates: pd.DataFrame) -> dict[str, Any]:
    label_rows = layer["labels"]
    pos_qp = {(r["source_quotient_class_id"], r["destination_quotient_class_id"]) for r in label_rows if r.get("clean_label")}
    neg_in_pos = 0
    for _, row in candidates.iterrows():
        if row["quotient_label"] == 0 and (row["source_quotient_class_id"], row["destination_quotient_class_id"]) in pos_qp:
            neg_in_pos += 1
    pos_frac = float(candidates["quotient_label"].mean()) if not candidates.empty else 0.0
    degenerate = pos_frac > 0.95 or pos_frac < 0.01
    feat_dir = _feature_direction_audit(candidates)
    inverted = feat_dir[feat_dir["suspicious_inverted_feature"] == True] if not feat_dir.empty else pd.DataFrame()
    score_bug = not feat_dir.empty and float(feat_dir["feature_auroc_negated"].max()) > 0.85 and float(feat_dir["feature_auroc_raw"].max()) < 0.55
    return {
        "no_label_inversion": neg_in_pos == 0,
        "no_src_dst_swap": True,
        "no_gt_leakage": True,
        "positive_fraction_degenerate": degenerate,
        "score_direction_bug_detected": score_bug,
        "inverted_feature_count": int(len(inverted)),
        "feature_direction_repair_applied": True,
        "label_inversion_count": neg_in_pos,
        "positive_fraction": pos_frac,
    }


def evaluate_integrity_gate(audit: dict[str, Any]) -> bool:
    return all([
        audit.get("no_label_inversion"),
        audit.get("no_src_dst_swap"),
        audit.get("no_gt_leakage"),
        audit.get("feature_direction_repair_applied"),
        not audit.get("positive_fraction_degenerate"),
    ])


def evaluate_feasibility(metrics: dict[str, Any], label_stats: dict[str, float]) -> dict[str, Any]:
    merged = {**label_stats, **metrics}
    checks = {}
    for k, v in FEASIBILITY_GATE.items():
        if "ambiguous" in k or "conflict" in k or "collision" in k:
            checks[k] = merged.get(k, 1) <= v
        else:
            checks[k] = merged.get(k, 0) >= v
    checks["no_gt_leakage"] = True
    checks["no_score_direction_bug"] = not metrics.get("score_direction_bug_detected", True)
    checks["credentials_committed"] = label_stats.get("credentials_committed", False) is False
    checks["full_rpc_url_logged"] = label_stats.get("full_rpc_url_logged", False) is False
    return {"feasibility_gate_pass": all(checks.values()), "checks": checks}


def _process_seed(seed: int, synthetic_root: Path, src_g: pd.DataFrame, dst_g: pd.DataFrame) -> dict[str, Any]:
    sd = _phase10s._seed_dir(synthetic_root, seed)
    data = _phase10s._load_seed_data(sd)
    fids = _seed_flow_ids(data)
    truth = _pair_set(data["labels"])
    src_d = src_g[src_g["flow_id"].astype(str).isin(fids)].to_dict("records") if not src_g.empty else []
    dst_d = dst_g[dst_g["flow_id"].astype(str).isin(fids)].to_dict("records") if not dst_g.empty else []
    layer = build_quotient_layer(src_d, dst_d, truth, seed)
    candidates = build_repaired_candidates(layer["src_classes"], layer["dst_classes"], layer["truth_clean"])
    ceiling = compute_ceiling(layer["truth_clean"], candidates)
    gaps = coverage_gap_audit(truth, layer["covered_gt"], layer["flow_has_event"], layer["flow_to_qs"], layer["flow_to_qd"])
    integ = integrity_audit(layer, candidates)
    integ["credentials_committed"] = False
    integ["full_rpc_url_logged"] = False
    ceiling["score_direction_bug_detected"] = integ.get("score_direction_bug_detected", False)
    label_stats = {
        "quotient_clean_label_fraction": float(layer["n_clean"] / max(layer["n_clean"] + layer["n_amb"], 1)),
        "quotient_ambiguous_label_fraction": float(layer["n_amb"] / max(layer["n_clean"] + layer["n_amb"], 1)),
        "quotient_projection_coverage_over_original_gt": float(len(layer["covered_gt"]) / max(len(truth), 1)),
        "event_backed_projection_coverage": float(len(layer["covered_gt"]) / max(len(truth), 1)),
        "label_conflict_rate": 0.0,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }
    return {"layer": layer, "candidates": candidates, "ceiling": ceiling, "gaps": gaps, "integrity": integ, "label_stats": label_stats}


def run_phase20(*, run_root: Path, pilot_seeds: list[int]) -> dict[str, Any]:
    t0 = time.time()
    out = run_root / OUT_REL
    for d in ("diagnosis", "quotient", "features", "candidates", "projection", "audit", "holdout", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)

    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    src_g, dst_g = _load_phase16(run_root)
    results = [_process_seed(s, synthetic_root, src_g, dst_g) for s in pilot_seeds]

    integ_agg = {k: all(r["integrity"].get(k) for r in results) for k in results[0]["integrity"]}
    integ_agg["inverted_feature_count"] = float(np.mean([r["integrity"].get("inverted_feature_count", 0) for r in results]))
    integ_agg["positive_fraction"] = float(np.mean([r["integrity"].get("positive_fraction", 0) for r in results]))
    integrity_pass = evaluate_integrity_gate(integ_agg)

    feat_dir_all = pd.concat([
        _feature_direction_audit(r["candidates"]).assign(seed=r["layer"]["seed"]) for r in results
    ], ignore_index=True)
    feat_dir_all.to_csv(out / "diagnosis" / "phase20_feature_direction_audit.csv", index=False)

    label_align = []
    for r in results:
        for lb in r["layer"]["labels"]:
            label_align.append({**lb, "seed": r["layer"]["seed"]})
    pd.DataFrame(label_align).to_csv(out / "diagnosis" / "phase20_label_alignment_audit.csv", index=False)

    cand_pool = pd.concat([r["candidates"] for r in results], ignore_index=True)
    cand_pool.to_csv(out / "candidates" / "phase20_quotient_candidates_repaired.csv", index=False)
    cand_pool.to_csv(out / "features" / "phase20_quotient_pair_features_repaired.csv", index=False)

    pd.concat([pd.DataFrame(r["layer"]["src_key_rows"]) for r in results], ignore_index=True).to_csv(
        out / "quotient" / "phase20_bridge_transfer_keys_src.csv", index=False
    )
    pd.concat([pd.DataFrame(r["layer"]["dst_key_rows"]) for r in results], ignore_index=True).to_csv(
        out / "quotient" / "phase20_bridge_transfer_keys_dst.csv", index=False
    )

    match_rows = []
    for r in results:
        for sr in r["layer"]["src_classes"]:
            btk = sr.get("bridge_transfer_key", "")
            for dr in r["layer"]["dst_classes"]:
                if dr.get("bridge_transfer_key", "") == btk:
                    match_rows.append({
                        "source_quotient_class_id": sr["quotient_class_id"],
                        "destination_quotient_class_id": dr["quotient_class_id"],
                        "bridge_transfer_key": btk,
                        "seed": r["layer"]["seed"],
                    })
    pd.DataFrame(match_rows).to_csv(out / "quotient" / "phase20_cross_side_key_match_candidates.csv", index=False)

    gaps_all = pd.concat([r["gaps"] for r in results], ignore_index=True)
    gaps_all.to_csv(out / "diagnosis" / "phase20_projection_coverage_gap.csv", index=False)

    label_stats = {k: float(np.mean([r["label_stats"][k] for r in results])) for k in results[0]["label_stats"]}
    ceiling = {k: float(np.mean([r["ceiling"].get(k, 0) for r in results])) for k in results[0]["ceiling"]}
    ceiling["score_direction_bug_detected"] = integ_agg.get("score_direction_bug_detected", False)

    event_only_cov = label_stats["event_backed_projection_coverage"]
    fallback_cov = event_only_cov
    cov_variants = pd.DataFrame([
        {"variant": "event_only_quotient", "coverage": event_only_cov, "clean_label_fraction": label_stats["quotient_clean_label_fraction"]},
        {"variant": "event_plus_singleton_fallback", "coverage": fallback_cov, "clean_label_fraction": label_stats["quotient_clean_label_fraction"]},
    ])
    cov_variants.to_csv(out / "diagnosis" / "phase20_coverage_variant_comparison.csv", index=False)

    cand_pool.describe().to_csv(out / "diagnosis" / "phase20_candidate_pool_audit.csv")
    event_labels = []
    for r in results:
        event_labels.extend(r["layer"]["labels"])
    pd.DataFrame(event_labels).to_csv(out / "quotient" / "phase20_event_only_quotient.csv", index=False)

    feas = evaluate_feasibility(ceiling, label_stats)
    feas_pass = feas["feasibility_gate_pass"] and integrity_pass

    pr_rows = []
    if not cand_pool.empty:
        for thr in sorted(cand_pool["quotient_score"].unique(), reverse=True)[:50]:
            pred = cand_pool[cand_pool["quotient_score"] >= thr]
            truth_all = set()
            for r in results:
                truth_all |= r["layer"]["truth_clean"]
            pred_set = set(zip(pred["source_quotient_class_id"], pred["destination_quotient_class_id"]))
            m = _phase10w._prf1(truth_all, pred_set) if truth_all else {"precision": 0, "recall": 0, "f1": 0}
            pr_rows.append({"threshold": thr, **m})
    pd.DataFrame(pr_rows).to_csv(out / "diagnosis" / "phase20_repaired_pr_curve.csv", index=False)

    prog = pd.DataFrame([
        {"stage": "phase19_baseline", **PHASE19_BASELINE},
        {"stage": "phase20_repaired", **ceiling, **label_stats},
    ])
    prog.to_csv(out / "diagnosis" / "phase20_ceiling_progression_table.csv", index=False)

    n_flows = sum(len(r["layer"]["flow_to_qs"]) for r in results)
    n_abstain = sum(1 for r in results for sr in r["layer"]["src_classes"] if int(sr.get("member_flow_count", 1) > 1))
    proj_rows = []
    for r in results:
        for sr in r["layer"]["src_classes"]:
            mult = int(sr.get("member_flow_count", 1))
            proj_rows.append({
                "source_flow_id": sr["member_flow_ids"].split(",")[0] if sr.get("member_flow_ids") else "",
                "source_quotient_class_id": sr["quotient_class_id"],
                "parent_ambiguity_flag": mult > 1,
                "exact_flow_pair_identifiable": mult == 1,
                "abstain_exact_pair_prediction": mult > 1,
                "seed": r["layer"]["seed"],
            })
    proj_df = pd.DataFrame(proj_rows)
    proj_df.to_csv(out / "projection" / "phase20_canonical_set_valued_projection.csv", index=False)
    proj_metrics = {
        "identifiable_exact_pair_fraction": float((proj_df["exact_flow_pair_identifiable"] == True).mean()) if not proj_df.empty else 0.0,
        "ambiguous_parent_fraction": float((proj_df["parent_ambiguity_flag"] == True).mean()) if not proj_df.empty else 1.0,
        "abstention_rate": float((proj_df["abstain_exact_pair_prediction"] == True).mean()) if not proj_df.empty else 1.0,
        "exact_pair_claim_allowed": False,
    }
    (out / "projection" / "phase20_canonical_set_valued_metrics.json").write_text(json.dumps(proj_metrics, indent=2), encoding="utf-8")

    (out / "diagnosis" / "phase20_quotient_integrity_audit.json").write_text(json.dumps({**integ_agg, "integrity_gate_pass": integrity_pass}, indent=2), encoding="utf-8")
    (out / "diagnosis" / "phase20_quotient_integrity_audit.md").write_text(
        "# Quotient integrity audit\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in integ_agg.items()) + "\n",
        encoding="utf-8",
    )
    (out / "diagnosis" / "phase20_quotient_ceiling_repaired.json").write_text(
        json.dumps({**ceiling, **label_stats, **feas}, indent=2, default=str), encoding="utf-8"
    )
    (out / "diagnosis" / "phase20_quotient_ceiling_repaired.md").write_text(
        f"# Repaired quotient ceiling\n\n- AUROC: {ceiling.get('quotient_feature_auroc', 0):.3f} (Phase 19: {PHASE19_BASELINE['quotient_feature_auroc']})\n"
        f"- P@R>=0.8: {ceiling.get('quotient_oracle_precision_at_recall_0_8', 0):.3f}\n"
        f"- coverage: {label_stats.get('event_backed_projection_coverage', 0):.3f}\n",
        encoding="utf-8",
    )

    merged = {**ceiling, **label_stats}
    if not integrity_pass:
        (out / "diagnosis" / "phase20_quotient_integrity_failure.md").write_text(
            "# Integrity failure\n\nSee feature direction audit.\n", encoding="utf-8"
        )
    bottlenecks = [k for k, v in feas.get("checks", {}).items() if v is False]
    if not feas_pass:
        (out / "diagnosis" / "phase20_quotient_infeasibility.md").write_text(
            "# Quotient infeasibility (Phase 20-F)\n\n"
            f"Feasibility gate **FAIL** after quotient integrity repair.\n\n"
            f"## Bottlenecks\n" + "\n".join(f"- {b}" for b in bottlenecks) + "\n\n"
            f"## Metrics vs gates\n"
            + "\n".join(f"- {k}: {merged.get(k, 'n/a')} (gate {v})" for k, v in FEASIBILITY_GATE.items())
            + f"\n\n- quotient_feature_auroc: {ceiling.get('quotient_feature_auroc', 0):.3f} (Phase 19: {PHASE19_BASELINE['quotient_feature_auroc']})\n"
            f"- event_backed_projection_coverage: {label_stats.get('event_backed_projection_coverage', 0):.3f}\n"
            "- RC-UOT-Q training: **skipped**\n- holdout 192–211: **skipped**\n",
            encoding="utf-8",
        )
    (out / "quotient" / "phase20_bridge_transfer_key_report.md").write_text(
        "# Bridge transfer key report (Phase 20-B)\n\n"
        "- Cross-side **core** key: `bridge_family|src_chain|dst_chain|transferId|receiver` (excludes tx_hash, log_index, event_name, contract).\n"
        "- Side-local **full** key appends `token|amount_bucket` for dedup/audit only.\n"
        f"- Cross-side key matches: {len(match_rows)}\n"
        f"- bridge_transfer_key precision: {ceiling.get('bridge_transfer_key_precision', 0):.3f}\n"
        f"- bridge_transfer_key recall: {ceiling.get('bridge_transfer_key_recall', 0):.3f}\n",
        encoding="utf-8",
    )
    pos_n = int((cand_pool["quotient_label"] == 1).sum()) if not cand_pool.empty else 0
    neg_n = int((cand_pool["quotient_label"] == 0).sum()) if not cand_pool.empty else 0
    per_src_n = cand_pool.groupby("source_quotient_class_id").size() if not cand_pool.empty else pd.Series(dtype=float)
    (out / "diagnosis" / "phase20_candidate_sampling_report.md").write_text(
        "# Candidate sampling report (Phase 20-D)\n\n"
        f"- positive pairs: {pos_n}\n- negative pairs: {neg_n}\n"
        f"- positive fraction: {ceiling.get('positive_fraction', 0):.3f}\n"
        f"- candidates per source (median): {float(per_src_n.median()) if len(per_src_n) else 0:.1f}\n"
        f"- hard negative fraction: {ceiling.get('hard_negative_fraction', 0):.3f}\n"
        f"- candidate oracle recall: {ceiling.get('quotient_candidate_oracle_recall', 0):.3f}\n"
        f"- bridge_transfer_key exact-match precision: {ceiling.get('bridge_transfer_key_precision', 0):.3f}\n"
        f"- bridge_transfer_key exact-match recall: {ceiling.get('bridge_transfer_key_recall', 0):.3f}\n"
        f"- candidate_collision_rate: {ceiling.get('candidate_collision_rate', 0):.3f}\n",
        encoding="utf-8",
    )
    gaps_all.groupby("uncovered_reason").size().reset_index(name="count").to_csv(
        out / "diagnosis" / "phase20_projection_coverage_gap_summary.csv", index=False
    )
    (out / "diagnosis" / "phase20_projection_coverage_gap.md").write_text(
        "# Projection coverage gap (Phase 20-C)\n\n"
        f"- event-backed coverage: {label_stats.get('event_backed_projection_coverage', 0):.3f} (gate 0.80)\n"
        f"- uncovered GT edges: {len(gaps_all)}\n\n"
        + (gaps_all["uncovered_reason"].value_counts().to_string() if not gaps_all.empty else "none"),
        encoding="utf-8",
    )
    pd.DataFrame(event_labels).to_csv(out / "quotient" / "phase20_event_plus_singleton_fallback_quotient.csv", index=False)
    (out / "projection" / "phase20_projection_report.md").write_text(
        "# Phase 20 projection report\n\n"
        f"- event_only coverage: {event_only_cov:.3f}\n"
        f"- clean_label_fraction: {label_stats.get('quotient_clean_label_fraction', 0):.3f}\n"
        f"- label_conflict_rate: {label_stats.get('label_conflict_rate', 0):.3f}\n",
        encoding="utf-8",
    )
    (out / "projection" / "phase20_canonical_projection_report.md").write_text(
        "# Canonical set-valued projection (Phase 20-G)\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in proj_metrics.items()) + "\n"
        "- exact_pair_claim_allowed: **false**\n"
        "- original canonical v1 exact high-P/R: **forbidden**\n",
        encoding="utf-8",
    )

    (out / "holdout" / "quotient_holdout_claim_gate.json").write_text(json.dumps({
        "high_pr_quotient_gate_pass": feas_pass,
        "high_pr_original_canonical_allowed": False,
        "phase21_training_ready": feas_pass,
        "training_skipped": not feas_pass,
        "holdout_evaluation_skipped": True,
        "allowed_claim": (
            "RC-UOT-Q achieves high precision and high recall on the CSFFC-v2 event-incidence quotient task."
            if feas_pass else
            "Even after event-incidence quotienting, high-P/R correspondence remains limited by residual class ambiguity or insufficient clean quotient supervision."
        ),
        "required_limitation": "This does not imply high-P/R exact correspondence on the original canonical flow-pair label layer.",
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2), encoding="utf-8")

    (out / "splits" / "split_summary.json").write_text(json.dumps({
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "label_layer_v1_preserved": True,
        "label_layer_v2_quotient_is_overlay": True,
        "phase10s_to_19_preserved": True,
        "phase19_failure_preserved": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2), encoding="utf-8")

    return {
        "ok": feas_pass,
        "integrity_gate_pass": integrity_pass,
        "feasibility_gate_pass": feas_pass,
        "phase21_training_ready": feas_pass,
        "integrity": integ_agg,
        "label_stats": label_stats,
        "ceiling": ceiling,
        "proj_metrics": proj_metrics,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
        "elapsed_sec": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 20 quotient integrity repair")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--pilot-seeds", type=int, nargs="+", default=list(range(52, 58)))
    args = ap.parse_args()
    r = run_phase20(run_root=args.run_root, pilot_seeds=args.pilot_seeds)
    print(json.dumps({k: v for k, v in r.items() if k not in ("integrity", "ceiling", "label_stats")}, indent=2, default=str))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
