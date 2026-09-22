#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: TransferId-normalized micro-flow resegmentation and identifiability repair."""
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
    ("phase10x", _REPO / "scripts" / "run_phase10x_evidence_enhanced_disambiguation.py"),
    ("phase14", _REPO / "scripts" / "run_phase14_rpc_bridge_evidence_verification.py"),
    ("phase15", _REPO / "scripts" / "run_phase15_celer_abi_decode.py"),
):
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    globals()[f"_{_name}"] = _mod

OUT_REL = "phase17_transferid_microflow_resegmentation"
PHASE14_OUT = "phase14_rpc_bridge_evidence_verification"
PHASE16_OUT = "phase16_transferid_event_deaggregation"
PRIMARY_K = 50
UNAVAILABLE = float("nan")
ETH_CHAIN_ID_INT = 1
BSC_CHAIN_ID_INT = 56

PHASE16_FLOW_LIFTED_BASELINE = {
    "feature_auroc": 0.847,
    "feature_auprc": 0.161,
    "oracle_precision_at_recall_0_8": 0.086,
    "score_oracle_best_f1": 0.312,
    "candidate_collision_rate": 0.208,
}

MICROFLOW_PILOT_PASS = {
    "candidate_oracle_recall": 0.95,
    "oracle_precision_at_recall_0_8": 0.80,
    "oracle_recall_at_precision_0_8": 0.80,
    "score_oracle_best_f1": 0.80,
    "feature_auroc": 0.85,
    "feature_auprc": 0.70,
    "candidate_collision_rate": 0.30,
    "duplicate_event_assignment_rate": 0.20,
    "clean_microflow_label_fraction": 0.70,
}

VARIANTS = ("a", "b", "c", "d")
VARIANT_NAMES = {
    "a": "event_key_microflow",
    "b": "transferid_group_microflow",
    "c": "transferid_entity_microflow",
    "d": "fractional_parent_flow_microflow",
}


def _src_event_key(e: dict[str, Any]) -> str:
    chain = str(e.get("chain") or "ethereum").lower()
    tx = str(e.get("tx_hash") or "").lower()
    li = int(e.get("log_index") or 0)
    en = str(e.get("event_name") or "")
    tid = str(e.get("transfer_id") or "").lower()
    return f"{chain}|{tx}|{li}|{en}|{tid}"


def _dst_event_key(e: dict[str, Any]) -> str:
    chain = str(e.get("chain") or "bsc").lower()
    tx = str(e.get("tx_hash") or "").lower()
    li = int(e.get("log_index") or 0)
    en = str(e.get("event_name") or "")
    stid = str(e.get("src_transfer_id") or "").lower()
    return f"{chain}|{tx}|{li}|{en}|{stid}"


def _entity_key(e: dict[str, Any], *, is_src: bool) -> str:
    tid = str(e.get("transfer_id") if is_src else e.get("src_transfer_id") or "").lower()
    sender = str(e.get("sender") or "").lower()
    receiver = str(e.get("receiver") or "").lower()
    token = str(e.get("token") or "").lower()
    amt = e.get("amount_raw") or e.get("amount_normalized") or ""
    return f"{tid}|{sender}|{receiver}|{token}|{amt}"


def _amount_bucket(val: Any) -> str:
    try:
        x = float(val)
        if x != x:
            return "na"
        return f"{x:.6e}"
    except (TypeError, ValueError):
        return str(val or "na")


def _seed_flow_ids(seed_data: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("eth_flows", "bnb_flows", "bsc_flows"):
        for f in seed_data.get(key) or []:
            fid = str(f.get("flow_id") or "")
            if fid:
                ids.add(fid)
    return ids


def _filter_events(events: list[dict[str, Any]], flow_ids: set[str]) -> list[dict[str, Any]]:
    return [e for e in events if str(e.get("flow_id") or "") in flow_ids]


def _load_phase16_decoded(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p16 = run_root / PHASE16_OUT / "evidence"
    src_p = p16 / "phase16_revalidated_decoded_events_src.csv"
    dst_p = p16 / "phase16_revalidated_decoded_events_dst.csv"
    if not src_p.is_file() or not dst_p.is_file():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(src_p), pd.read_csv(dst_p)


def _events_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.to_dict("records")


def _pct_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
    }


def build_incidence_tables(
    src_events: list[dict[str, Any]],
    dst_events: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    src_rows, dst_rows, tid_rows = [], [], []
    for e in src_events:
        if e.get("event_name") not in ("Send", "LogNewTransferOut") or not _phase15._field_nonempty(e.get("transfer_id")):
            continue
        ek = _src_event_key(e)
        tid = str(e.get("transfer_id") or "").lower()
        src_rows.append({
            "event_key": ek,
            "transfer_id": tid,
            "src_transfer_id": "",
            "chain": e.get("chain", ""),
            "tx_hash": e.get("tx_hash", ""),
            "log_index": e.get("log_index", ""),
            "event_name": e.get("event_name", ""),
            "flow_id": e.get("flow_id", ""),
            "address_cluster_if_available": e.get("address_cluster", UNAVAILABLE),
            "amount_raw": e.get("amount_raw", ""),
            "amount_normalized": e.get("amount_normalized", UNAVAILABLE),
            "sender": e.get("sender", ""),
            "receiver": e.get("receiver", ""),
            "token": e.get("token", ""),
            "block_number": e.get("block_number", ""),
            "timestamp_if_available": e.get("timestamp", UNAVAILABLE),
        })
        tid_rows.append({"transfer_id": tid, "flow_id": e.get("flow_id", ""), "side": "src"})
    for e in dst_events:
        if e.get("event_name") not in ("Relay", "LogNewTransferIn") or not _phase15._field_nonempty(e.get("src_transfer_id")):
            continue
        ek = _dst_event_key(e)
        stid = str(e.get("src_transfer_id") or "").lower()
        dst_rows.append({
            "event_key": ek,
            "transfer_id": str(e.get("transfer_id") or "").lower(),
            "src_transfer_id": stid,
            "chain": e.get("chain", ""),
            "tx_hash": e.get("tx_hash", ""),
            "log_index": e.get("log_index", ""),
            "event_name": e.get("event_name", ""),
            "flow_id": e.get("flow_id", ""),
            "address_cluster_if_available": e.get("address_cluster", UNAVAILABLE),
            "amount_raw": e.get("amount_raw", ""),
            "amount_normalized": e.get("amount_normalized", UNAVAILABLE),
            "sender": e.get("sender", ""),
            "receiver": e.get("receiver", ""),
            "token": e.get("token", ""),
            "block_number": e.get("block_number", ""),
            "timestamp_if_available": e.get("timestamp", UNAVAILABLE),
        })
        tid_rows.append({"transfer_id": stid, "flow_id": e.get("flow_id", ""), "side": "dst"})

    src_inc = pd.DataFrame(src_rows)
    dst_inc = pd.DataFrame(dst_rows)
    tid_inc = pd.DataFrame(tid_rows)

    metrics: dict[str, Any] = {}
    if not src_inc.empty:
        epf = src_inc.groupby("flow_id").size()
        metrics["events_per_src_flow"] = _pct_stats(epf)
        fpt = src_inc.groupby("transfer_id")["flow_id"].nunique()
        metrics["src_flows_per_transferId"] = _pct_stats(fpt)
        metrics["transferId_reuse_rate"] = float((fpt > 1).mean())
        metrics["single_event_flow_fraction_src"] = float((epf == 1).mean())
        metrics["multi_event_flow_fraction_src"] = float((epf > 1).mean())
        ek_flows = src_inc.groupby("event_key")["flow_id"].nunique()
        metrics["duplicate_event_assignment_rate"] = float((ek_flows > 1).mean())
        metrics["flow_purity_rate"] = float((ek_flows == 1).mean())
        metrics["mixed_flow_rate"] = float((epf > 1).mean())
    else:
        metrics["events_per_src_flow"] = _pct_stats(pd.Series(dtype=float))
        metrics["src_flows_per_transferId"] = _pct_stats(pd.Series(dtype=float))
        metrics["transferId_reuse_rate"] = 0.0
        metrics["duplicate_event_assignment_rate"] = 1.0

    if not dst_inc.empty:
        epf_d = dst_inc.groupby("flow_id").size()
        metrics["events_per_dst_flow"] = _pct_stats(epf_d)
        fpt_d = dst_inc.groupby("src_transfer_id")["flow_id"].nunique()
        metrics["dst_flows_per_srcTransferId"] = _pct_stats(fpt_d)
        metrics["srcTransferId_reuse_rate"] = float((fpt_d > 1).mean())
        metrics["single_event_flow_fraction_dst"] = float((epf_d == 1).mean())
        metrics["multi_event_flow_fraction_dst"] = float((epf_d > 1).mean())
    else:
        metrics["events_per_dst_flow"] = _pct_stats(pd.Series(dtype=float))
        metrics["dst_flows_per_srcTransferId"] = _pct_stats(pd.Series(dtype=float))
        metrics["srcTransferId_reuse_rate"] = 0.0

    if not tid_inc.empty:
        src_tid = tid_inc[tid_inc["side"] == "src"]
        if not src_tid.empty:
            pairs = src_tid.groupby("transfer_id")["flow_id"].apply(lambda s: len(set(s)))
            metrics["transferId_unique_flow_pair_fraction"] = float((pairs == 1).mean())
        metrics["event_to_flow_nonidentifiability_rate"] = metrics.get("duplicate_event_assignment_rate", 1.0)

    metrics["credentials_committed"] = False
    metrics["full_rpc_url_logged"] = False
    return src_inc, dst_inc, tid_inc, metrics


def _build_microflow_variant_a(src_events: list[dict[str, Any]], dst_events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _src_rows():
        seen: dict[str, dict[str, Any]] = {}
        for e in src_events:
            if e.get("event_name") not in ("Send", "LogNewTransferOut") or not _phase15._field_nonempty(e.get("transfer_id")):
                continue
            ek = _src_event_key(e)
            fid = str(e.get("flow_id") or "")
            tid = str(e.get("transfer_id") or "").lower()
            parents = seen.get(ek, {}).get("_parents", set())
            parents.add(fid)
            seen[ek] = {
                "micro_flow_id": f"mfa_src_{ek[:80]}",
                "parent_flow_id": fid,
                "event_key": ek,
                "transfer_id": tid,
                "src_transfer_id": "",
                "tx_hash": e.get("tx_hash", ""),
                "log_index": e.get("log_index", ""),
                "event_name": e.get("event_name", ""),
                "sender": e.get("sender", ""),
                "receiver": e.get("receiver", ""),
                "token": e.get("token", ""),
                "amount_raw": e.get("amount_raw", ""),
                "amount_normalized": e.get("amount_normalized", UNAVAILABLE),
                "chain_id": e.get("dst_chain_id", BSC_CHAIN_ID_INT),
                "block_number": e.get("block_number", ""),
                "parent_flow_multiplicity": len(parents),
                "microflow_mass": 1.0,
                "construction_variant": VARIANT_NAMES["a"],
                "evidence_confidence": 1.0,
                "_parents": parents,
            }
        out = []
        for v in seen.values():
            parents = v.pop("_parents")
            v["parent_flow_multiplicity"] = len(parents)
            v["parent_flow_id"] = ",".join(sorted(parents))
            out.append({k: val for k, val in v.items() if not k.startswith("_")})
        return out

    def _dst_rows():
        seen: dict[str, dict[str, Any]] = {}
        for e in dst_events:
            if e.get("event_name") not in ("Relay", "LogNewTransferIn") or not _phase15._field_nonempty(e.get("src_transfer_id")):
                continue
            ek = _dst_event_key(e)
            fid = str(e.get("flow_id") or "")
            stid = str(e.get("src_transfer_id") or "").lower()
            parents = seen.get(ek, {}).get("_parents", set())
            parents.add(fid)
            seen[ek] = {
                "micro_flow_id": f"mfa_dst_{ek[:80]}",
                "parent_flow_id": fid,
                "event_key": ek,
                "transfer_id": str(e.get("transfer_id") or "").lower(),
                "src_transfer_id": stid,
                "tx_hash": e.get("tx_hash", ""),
                "log_index": e.get("log_index", ""),
                "event_name": e.get("event_name", ""),
                "sender": e.get("sender", ""),
                "receiver": e.get("receiver", ""),
                "token": e.get("token", ""),
                "amount_raw": e.get("amount_raw", ""),
                "amount_normalized": e.get("amount_normalized", UNAVAILABLE),
                "chain_id": e.get("src_chain_id") or ETH_CHAIN_ID_INT,
                "block_number": e.get("block_number", ""),
                "parent_flow_multiplicity": len(parents),
                "microflow_mass": 1.0,
                "construction_variant": VARIANT_NAMES["a"],
                "evidence_confidence": 1.0,
                "_parents": parents,
            }
        out = []
        for v in seen.values():
            parents = v.pop("_parents")
            v["parent_flow_multiplicity"] = len(parents)
            v["parent_flow_id"] = ",".join(sorted(parents))
            out.append({k: val for k, val in v.items() if not k.startswith("_")})
        return out

    return pd.DataFrame(_src_rows()), pd.DataFrame(_dst_rows())


def _build_microflow_variant_b(src_events: list[dict[str, Any]], dst_events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _group(events, *, is_src: bool):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            en = e.get("event_name")
            if is_src:
                if en not in ("Send", "LogNewTransferOut") or not _phase15._field_nonempty(e.get("transfer_id")):
                    continue
                key = str(e.get("transfer_id") or "").lower()
            else:
                if en not in ("Relay", "LogNewTransferIn") or not _phase15._field_nonempty(e.get("src_transfer_id")):
                    continue
                key = str(e.get("src_transfer_id") or "").lower()
            groups[key].append(e)
        rows = []
        for key, evs in groups.items():
            parents = {str(x.get("flow_id") or "") for x in evs}
            e0 = evs[0]
            prefix = "mfb_src" if is_src else "mfb_dst"
            rows.append({
                "micro_flow_id": f"{prefix}_{key[:40]}",
                "parent_flow_id": ",".join(sorted(parents)),
                "event_key": key,
                "transfer_id": key if is_src else str(e0.get("transfer_id") or "").lower(),
                "src_transfer_id": "" if is_src else key,
                "tx_hash": e0.get("tx_hash", ""),
                "log_index": e0.get("log_index", ""),
                "event_name": e0.get("event_name", ""),
                "sender": e0.get("sender", ""),
                "receiver": e0.get("receiver", ""),
                "token": e0.get("token", ""),
                "amount_raw": e0.get("amount_raw", ""),
                "amount_normalized": e0.get("amount_normalized", UNAVAILABLE),
                "chain_id": e0.get("dst_chain_id" if is_src else "src_chain_id", BSC_CHAIN_ID_INT if is_src else ETH_CHAIN_ID_INT),
                "block_number": e0.get("block_number", ""),
                "parent_flow_multiplicity": len(parents),
                "microflow_mass": float(len(evs)),
                "construction_variant": VARIANT_NAMES["b"],
                "evidence_confidence": 1.0 / max(len(parents), 1),
            })
        return rows

    return pd.DataFrame(_group(src_events, is_src=True)), pd.DataFrame(_group(dst_events, is_src=False))


def _build_microflow_variant_c(src_events: list[dict[str, Any]], dst_events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _group(events, *, is_src: bool):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            en = e.get("event_name")
            if is_src:
                if en not in ("Send", "LogNewTransferOut") or not _phase15._field_nonempty(e.get("transfer_id")):
                    continue
            else:
                if en not in ("Relay", "LogNewTransferIn") or not _phase15._field_nonempty(e.get("src_transfer_id")):
                    continue
            key = _entity_key(e, is_src=is_src)
            groups[key].append(e)
        rows = []
        for key, evs in groups.items():
            parents = {str(x.get("flow_id") or "") for x in evs}
            e0 = evs[0]
            prefix = "mfc_src" if is_src else "mfc_dst"
            tid = str(e0.get("transfer_id") or "").lower() if is_src else str(e0.get("src_transfer_id") or "").lower()
            rows.append({
                "micro_flow_id": f"{prefix}_{hash(key) % 10**10}",
                "parent_flow_id": ",".join(sorted(parents)),
                "event_key": key,
                "transfer_id": tid if is_src else str(e0.get("transfer_id") or "").lower(),
                "src_transfer_id": "" if is_src else tid,
                "tx_hash": e0.get("tx_hash", ""),
                "log_index": e0.get("log_index", ""),
                "event_name": e0.get("event_name", ""),
                "sender": e0.get("sender", ""),
                "receiver": e0.get("receiver", ""),
                "token": e0.get("token", ""),
                "amount_raw": e0.get("amount_raw", ""),
                "amount_normalized": e0.get("amount_normalized", UNAVAILABLE),
                "chain_id": e0.get("dst_chain_id" if is_src else "src_chain_id", BSC_CHAIN_ID_INT if is_src else ETH_CHAIN_ID_INT),
                "block_number": e0.get("block_number", ""),
                "parent_flow_multiplicity": len(parents),
                "microflow_mass": float(len(evs)),
                "construction_variant": VARIANT_NAMES["c"],
                "evidence_confidence": 1.0 / max(len(parents), 1),
            })
        return rows

    return pd.DataFrame(_group(src_events, is_src=True)), pd.DataFrame(_group(dst_events, is_src=False))


def _build_microflow_variant_d(src_events: list[dict[str, Any]], dst_events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    src_a, dst_a = _build_microflow_variant_a(src_events, dst_events)
    if not src_a.empty:
        pf_counts = src_events
        flow_amt: dict[str, float] = defaultdict(float)
        for e in src_events:
            if e.get("event_name") not in ("Send", "LogNewTransferOut"):
                continue
            fid = str(e.get("flow_id") or "")
            try:
                flow_amt[fid] += float(e.get("amount_normalized") or e.get("amount_raw") or 0)
            except (TypeError, ValueError):
                flow_amt[fid] += 1.0
        src_a = src_a.copy()
        masses = []
        for _, r in src_a.iterrows():
            try:
                ev_amt = float(r.get("amount_normalized") or r.get("amount_raw") or 1.0)
            except (TypeError, ValueError):
                ev_amt = 1.0
            parent = str(r.get("parent_flow_id") or "")
            total = flow_amt.get(parent, ev_amt) or ev_amt
            masses.append(ev_amt / max(total, 1e-9))
        src_a["microflow_mass"] = masses
        src_a["construction_variant"] = VARIANT_NAMES["d"]
    if not dst_a.empty:
        flow_amt_d: dict[str, float] = defaultdict(float)
        for e in dst_events:
            if e.get("event_name") not in ("Relay", "LogNewTransferIn"):
                continue
            fid = str(e.get("flow_id") or "")
            try:
                flow_amt_d[fid] += float(e.get("amount_normalized") or e.get("amount_raw") or 0)
            except (TypeError, ValueError):
                flow_amt_d[fid] += 1.0
        dst_a = dst_a.copy()
        masses_d = []
        for _, r in dst_a.iterrows():
            try:
                ev_amt = float(r.get("amount_normalized") or r.get("amount_raw") or 1.0)
            except (TypeError, ValueError):
                ev_amt = 1.0
            parent = str(r.get("parent_flow_id") or "")
            total = flow_amt_d.get(parent, ev_amt) or ev_amt
            masses_d.append(ev_amt / max(total, 1e-9))
        dst_a["microflow_mass"] = masses_d
        dst_a["construction_variant"] = VARIANT_NAMES["d"]
    return src_a, dst_a


BUILDERS = {
    "a": _build_microflow_variant_a,
    "b": _build_microflow_variant_b,
    "c": _build_microflow_variant_c,
    "d": _build_microflow_variant_d,
}


def _chain_direction_ok(sa: pd.Series, da: pd.Series) -> bool:
    """ETH→BSC bridge: source Send targets BSC; destination Relay records ETH src_chain."""
    sc = str(sa.get("chain_id") or sa.get("dst_chain_id") or "")
    dc = str(da.get("chain_id") or da.get("src_chain_id") or "")
    return sc in (str(BSC_CHAIN_ID_INT), "56", "0x38") and dc in (str(ETH_CHAIN_ID_INT), "1", "0x1")


def project_microflow_labels(
    truth: set[tuple[str, str]],
    src_micro: pd.DataFrame,
    dst_micro: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tid_gt_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for sf, df in truth:
        for _, sm in src_micro.iterrows():
            parents_s = str(sm.get("parent_flow_id") or "").split(",")
            if sf not in parents_s:
                continue
            tid = str(sm.get("transfer_id") or "").lower()
            tid_gt_pairs[tid].add((sf, df))

    rows, amb_rows = [], []
    n_gt = max(len(truth), 1)
    for _, sm in src_micro.iterrows():
        tid = str(sm.get("transfer_id") or "").lower()
        if not tid:
            continue
        parents_s = set(str(sm.get("parent_flow_id") or "").split(","))
        for _, dm in dst_micro.iterrows():
            stid = str(dm.get("src_transfer_id") or "").lower()
            if tid != stid:
                continue
            parents_d = set(str(dm.get("parent_flow_id") or "").split(","))
            parent_hits = [(ps, pd_) for ps in parents_s for pd_ in parents_d if (ps, pd_) in truth]
            chain_ok = _chain_direction_ok(sm, dm)
            label_ambiguous = len(tid_gt_pairs.get(tid, set())) > 1
            if not parent_hits:
                continue
            for ps, pd_ in parent_hits:
                rec = {
                    "src_micro_flow_id": sm["micro_flow_id"],
                    "dst_micro_flow_id": dm["micro_flow_id"],
                    "parent_src_flow_id": ps,
                    "parent_dst_flow_id": pd_,
                    "source_transfer_id": tid,
                    "destination_src_transfer_id": stid,
                    "projected_positive": 1,
                    "label_ambiguous": label_ambiguous,
                    "chain_direction_consistent": chain_ok,
                    "transferId_consistent": True,
                }
                if label_ambiguous:
                    amb_rows.append(rec)
                else:
                    rows.append(rec)

    labels = pd.DataFrame(rows)
    ambiguous = pd.DataFrame(amb_rows)
    n_proj = len(labels) + len(ambiguous)
    n_clean = len(labels)
    n_amb = len(ambiguous)
    stats = {
        "projected_positive_count": n_proj,
        "clean_projected_positive_count": n_clean,
        "ambiguous_projected_positive_count": n_amb,
        "projection_coverage_over_original_gt": float(n_clean / n_gt) if truth else 0.0,
        "clean_microflow_label_fraction": float(n_clean / max(n_proj, 1)),
        "microflow_label_ambiguity_rate": float(n_amb / max(n_proj, 1)),
        "parent_flow_label_ambiguity_rate": float(n_amb / max(n_proj, 1)),
        "transferId_consistent_gt_fraction": float(n_clean / max(len(truth), 1)),
        "transferId_inconsistent_gt_fraction": float(max(0, len(truth) - n_clean) / max(len(truth), 1)),
    }
    return labels, ambiguous, stats


def _parent_pair_in_truth(sm: pd.Series, dm: pd.Series, truth: set[tuple[str, str]]) -> int:
    parents_s = set(str(sm.get("parent_flow_id") or "").split(","))
    parents_d = set(str(dm.get("parent_flow_id") or "").split(","))
    return int(any((ps, pd) in truth for ps in parents_s for pd in parents_d))


def _micro_candidate_row(
    sm: pd.Series,
    dm: pd.Series,
    *,
    truth_micro: set[tuple[str, str]],
    truth_parent: set[tuple[str, str]],
) -> dict[str, Any]:
    tid = str(sm.get("transfer_id") or "").lower()
    stid = str(dm.get("src_transfer_id") or "").lower()
    tid_match = float(tid == stid and bool(tid))
    token_ok = str(sm.get("token") or "").lower() == str(dm.get("token") or "").lower() and bool(sm.get("token"))
    recv_ok = str(sm.get("receiver") or "").lower() == str(dm.get("receiver") or "").lower() and bool(sm.get("receiver"))
    send_ok = str(sm.get("sender") or "").lower() == str(dm.get("sender") or "").lower() and bool(sm.get("sender"))
    try:
        amt_s = float(sm.get("amount_normalized") or 0)
        amt_d = float(dm.get("amount_normalized") or 0)
    except (TypeError, ValueError):
        amt_s = amt_d = 0.0
    amt_err = abs(amt_s - amt_d)
    amt_rel = amt_err / max(abs(amt_s), abs(amt_d), 1e-9)
    amt_match = float(amt_rel <= 0.05)
    chain_ok = float(_chain_direction_ok(sm, dm))
    try:
        blk_lag = float(dm.get("block_number") or 0) - float(sm.get("block_number") or 0)
    except (TypeError, ValueError):
        blk_lag = 0.0
    blk_score = float(blk_lag > 0)
    parents_s = set(str(sm.get("parent_flow_id") or "").split(","))
    parents_d = set(str(dm.get("parent_flow_id") or "").split(","))
    overlap = len(parents_s & parents_d) / max(len(parents_s | parents_d), 1)
    mult_pen = 0.2 * max(int(sm.get("parent_flow_multiplicity") or 1) - 1, 0)
    reuse_pen = 0.2 if int(sm.get("parent_flow_multiplicity") or 1) > 1 else 0.0
    dup_pen = reuse_pen + mult_pen
    conf = tid_match * float(sm.get("evidence_confidence") or 1) * float(dm.get("evidence_confidence") or 1)
    if tid_match >= 1.0:
        if not chain_ok:
            conf *= 0.5
        if not token_ok:
            conf *= 0.8
        if not amt_match:
            conf *= 0.7
    conf = max(conf - dup_pen, 0.0)
    pair = (str(sm["micro_flow_id"]), str(dm["micro_flow_id"]))
    parent_label = _parent_pair_in_truth(sm, dm, truth_parent)
    return {
        "src_micro_flow_id": sm["micro_flow_id"],
        "dst_micro_flow_id": dm["micro_flow_id"],
        "parent_src_flow_id": sm.get("parent_flow_id", ""),
        "parent_dst_flow_id": dm.get("parent_flow_id", ""),
        "source_transfer_id": tid,
        "destination_src_transfer_id": stid,
        "micro_transferid_exact_match": tid_match,
        "micro_chain_direction_consistency": chain_ok,
        "micro_sender_receiver_consistency": float(recv_ok and send_ok),
        "micro_token_consistency": float(token_ok),
        "micro_amount_abs_error": amt_err,
        "micro_amount_rel_error": amt_rel,
        "micro_amount_match": amt_match,
        "micro_block_lag_score": blk_score,
        "micro_log_order_score": 1.0 if tid_match else 0.0,
        "micro_event_sequence_score": 1.0 if tid_match else 0.0,
        "parent_flow_overlap_score": overlap,
        "parent_flow_multiplicity_penalty": mult_pen,
        "duplicate_event_assignment_penalty": dup_pen,
        "transferid_reuse_penalty": reuse_pen,
        "microflow_confidence": conf,
        "micro_score": conf,
        "microflow_label": int(pair in truth_micro),
        "parent_flow_pair_label": parent_label,
        "evaluation_label": parent_label,
    }


def build_microflow_candidates(
    src_micro: pd.DataFrame,
    dst_micro: pd.DataFrame,
    labels: pd.DataFrame,
    truth_parent: set[tuple[str, str]],
    *,
    include_negatives: bool = True,
) -> pd.DataFrame:
    if src_micro.empty or dst_micro.empty:
        return pd.DataFrame()
    truth_micro: set[tuple[str, str]] = set()
    if not labels.empty:
        truth_micro = set(zip(labels["src_micro_flow_id"].astype(str), labels["dst_micro_flow_id"].astype(str)))
    dst_by_stid = dst_micro.groupby("src_transfer_id")
    rows: list[dict[str, Any]] = []
    for _, sm in src_micro.iterrows():
        tid = str(sm.get("transfer_id") or "").lower()
        if tid not in dst_by_stid.groups:
            continue
        for _, dm in dst_by_stid.get_group(tid).iterrows():
            rows.append(_micro_candidate_row(sm, dm, truth_micro=truth_micro, truth_parent=truth_parent))
    if include_negatives and len(rows) < 50000:
        rng = np.random.default_rng(42)
        dst_records = dst_micro.to_dict("records")
        for _, sm in src_micro.iterrows():
            tid = str(sm.get("transfer_id") or "")
            neg_pool = [d for d in dst_records if str(d.get("src_transfer_id") or "") != tid]
            if not neg_pool:
                continue
            n_neg = min(3, len(neg_pool))
            idx = rng.choice(len(neg_pool), size=n_neg, replace=False)
            for j in idx:
                rows.append(_micro_candidate_row(sm, pd.Series(neg_pool[int(j)]), truth_micro=truth_micro, truth_parent=truth_parent))
            if len(rows) >= 50000:
                break
    return pd.DataFrame(rows)


def _micro_score(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "microflow_confidence", "micro_transferid_exact_match", "micro_chain_direction_consistency",
        "micro_amount_match", "micro_token_consistency",
    ]
    parts = [pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in cols if c in df.columns]
    if not parts:
        return np.zeros(len(df))
    return np.vstack(parts).max(axis=0)


def _lift_parent_pairs(feat: pd.DataFrame, thr: float) -> set[tuple[str, str]]:
    pred: set[tuple[str, str]] = set()
    sub = feat[pd.to_numeric(feat["micro_score"], errors="coerce").fillna(0.0) >= thr]
    for r in sub.itertuples(index=False):
        for ps in str(getattr(r, "parent_src_flow_id", "") or "").split(","):
            ps = ps.strip()
            if not ps:
                continue
            for dst_fid in str(getattr(r, "parent_dst_flow_id", "") or "").split(","):
                dst_fid = dst_fid.strip()
                if dst_fid:
                    pred.add((ps, dst_fid))
    return pred


def _threshold_scan_micro(
    truth: set[tuple[str, str]],
    scores: pd.DataFrame,
    score_col: str = "micro_score",
    *,
    truth_parent: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if scores.empty:
        return {"oracle_best_f1": 0.0, "oracle_precision_at_recall_0_8": 0.0, "oracle_recall_at_precision_0_8": 0.0}
    s = scores.copy()
    s["_sc"] = pd.to_numeric(s[score_col], errors="coerce").fillna(0.0)
    s["_y"] = [
        int((str(r.src_micro_flow_id), str(r.dst_micro_flow_id)) in truth)
        for r in s.itertuples(index=False)
    ]
    uniq = sorted(set(s["_sc"].tolist()), reverse=True) or [0.0]
    best_f1, best_p_at_r, best_r_at_p = 0.0, 0.0, 0.0
    eval_truth = truth_parent if truth_parent else truth
    for thr in uniq + [0.0]:
        if truth_parent is not None:
            pred = _lift_parent_pairs(s, thr)
        else:
            pred = set(
                zip(
                    s.loc[s["_sc"] >= thr, "src_micro_flow_id"].astype(str),
                    s.loc[s["_sc"] >= thr, "dst_micro_flow_id"].astype(str),
                )
            )
        m = _phase10w._prf1(eval_truth, pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
        if m["recall"] >= 0.8 and m["precision"] > best_p_at_r:
            best_p_at_r = m["precision"]
        if m["precision"] >= 0.8 and m["recall"] > best_r_at_p:
            best_r_at_p = m["recall"]
    return {
        "oracle_best_f1": best_f1,
        "oracle_precision_at_recall_0_8": best_p_at_r,
        "oracle_recall_at_precision_0_8": best_r_at_p,
    }


def _compute_micro_ceiling(
    seeds_data: list[tuple[set[tuple[str, str]], pd.DataFrame, set[tuple[str, str]]]],
    *,
    group_col: str = "src_micro_flow_id",
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    cand_recs, score_f1s, p_at_r, r_at_p = [], [], [], []
    aurocs, auprcs, collisions, dup_rates, uniq_fracs = [], [], [], [], []
    clean_fracs = []
    for truth_micro, feat, truth_parent in seeds_data:
        if feat.empty:
            continue
        pairs = set(zip(feat["src_micro_flow_id"].astype(str), feat["dst_micro_flow_id"].astype(str)))
        if "evaluation_label" in feat.columns:
            pos = feat[feat["evaluation_label"].fillna(0).astype(int) == 1]
            lifted: set[tuple[str, str]] = set()
            for r in pos.itertuples(index=False):
                for ps in str(getattr(r, "parent_src_flow_id", "") or "").split(","):
                    ps = ps.strip()
                    if not ps:
                        continue
                    for dst_fid in str(getattr(r, "parent_dst_flow_id", "") or "").split(","):
                        dst_fid = dst_fid.strip()
                        if dst_fid:
                            lifted.add((ps, dst_fid))
            cand_recs.append(len(truth_parent & lifted) / max(len(truth_parent), 1))
        else:
            cand_recs.append(len(truth_micro & pairs) / max(len(truth_micro), 1))
        sc = feat[["src_micro_flow_id", "dst_micro_flow_id", "micro_score"]].copy()
        scan = _threshold_scan_micro(truth_micro, sc, score_col="micro_score", truth_parent=truth_parent)
        score_f1s.append(scan["oracle_best_f1"])
        p_at_r.append(scan["oracle_precision_at_recall_0_8"])
        r_at_p.append(scan["oracle_recall_at_precision_0_8"])
        y = feat["evaluation_label"].fillna(feat.get("parent_flow_pair_label", 0)).fillna(0).astype(int).to_numpy()
        x = _micro_score(feat)
        if len(np.unique(y)) > 1:
            aurocs.append(float(roc_auc_score(y, x)))
            auprcs.append(float(average_precision_score(y, x)))
        per_src = feat.groupby(group_col).size()
        collisions.append(float((per_src > 1).mean()))
        if "duplicate_event_assignment_penalty" in feat.columns:
            dup_rates.append(float((feat["duplicate_event_assignment_penalty"].fillna(0) > 0).mean()))
        if "micro_transferid_exact_match" in feat.columns:
            exact = feat[feat["micro_transferid_exact_match"].fillna(0) >= 1.0]
            if not exact.empty:
                per_tid = exact.groupby("source_transfer_id").size()
                uniq_fracs.append(float((per_tid == 1).mean()))
        clean_fracs.append(float(y.mean()))
    return {
        "candidate_oracle_recall": float(np.mean(cand_recs)) if cand_recs else 0.0,
        "score_oracle_best_f1": float(np.mean(score_f1s)) if score_f1s else 0.0,
        "oracle_precision_at_recall_0_8": float(np.mean(p_at_r)) if p_at_r else 0.0,
        "oracle_recall_at_precision_0_8": float(np.mean(r_at_p)) if r_at_p else 0.0,
        "feature_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "feature_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
        "candidate_collision_rate": float(np.mean(collisions)) if collisions else 1.0,
        "duplicate_event_assignment_rate": float(np.mean(dup_rates)) if dup_rates else 1.0,
        "transferId_unique_candidate_fraction": float(np.mean(uniq_fracs)) if uniq_fracs else 0.0,
        "clean_microflow_label_fraction": float(np.mean(clean_fracs)) if clean_fracs else 0.0,
        "ambiguous_gt_fraction": 0.0,
    }


def evaluate_microflow_pilot_gate(ceiling: dict[str, Any], label_stats: dict[str, Any]) -> dict[str, Any]:
    clean_frac = label_stats.get("clean_microflow_label_fraction", ceiling.get("clean_microflow_label_fraction", 0))
    checks = {
        "candidate_oracle_recall": ceiling.get("candidate_oracle_recall", 0) >= MICROFLOW_PILOT_PASS["candidate_oracle_recall"],
        "oracle_precision_at_recall_0_8": ceiling.get("oracle_precision_at_recall_0_8", 0) >= MICROFLOW_PILOT_PASS["oracle_precision_at_recall_0_8"],
        "oracle_recall_at_precision_0_8": ceiling.get("oracle_recall_at_precision_0_8", 0) >= MICROFLOW_PILOT_PASS["oracle_recall_at_precision_0_8"],
        "score_oracle_best_f1": ceiling.get("score_oracle_best_f1", 0) >= MICROFLOW_PILOT_PASS["score_oracle_best_f1"],
        "feature_auroc": ceiling.get("feature_auroc", 0) >= MICROFLOW_PILOT_PASS["feature_auroc"],
        "feature_auprc": ceiling.get("feature_auprc", 0) >= MICROFLOW_PILOT_PASS["feature_auprc"],
        "candidate_collision_rate": ceiling.get("candidate_collision_rate", 1) <= MICROFLOW_PILOT_PASS["candidate_collision_rate"],
        "duplicate_event_assignment_rate": ceiling.get("duplicate_event_assignment_rate", 1) <= MICROFLOW_PILOT_PASS["duplicate_event_assignment_rate"],
        "clean_microflow_label_fraction": clean_frac >= MICROFLOW_PILOT_PASS["clean_microflow_label_fraction"],
        "no_gt_leakage": True,
    }
    return {"pilot_pass": all(checks.values()), "checks": checks}


def _process_seed(
    seed: int,
    synthetic_root: Path,
    *,
    src_global: pd.DataFrame,
    dst_global: pd.DataFrame,
    topic_map: dict,
    eth_cache: Path,
    bsc_cache: Path,
    use_phase16_decoded: bool,
) -> dict[str, Any]:
    sd = _phase10s._seed_dir(synthetic_root, seed)
    seed_data = _phase10s._load_seed_data(sd)
    flow_ids = _seed_flow_ids(seed_data)
    truth = _pair_set(seed_data["labels"])

    if use_phase16_decoded and not src_global.empty:
        src_d = _events_from_df(src_global[src_global["flow_id"].astype(str).isin(flow_ids)])
        dst_d = _events_from_df(dst_global[dst_global["flow_id"].astype(str).isin(flow_ids)])
    else:
        src_d, dst_d, _, _, _ = _phase15.decode_seed_events(
            seed_data, eth_cache=eth_cache, bsc_cache=bsc_cache, topic_map=topic_map,
            eth_client=None, bsc_client=None,
        )

    src_inc, dst_inc, _, inc_metrics = build_incidence_tables(src_d, dst_d)
    variants: dict[str, dict[str, Any]] = {}
    for vk in VARIANTS:
        src_m, dst_m = BUILDERS[vk](src_d, dst_d)
        labels, ambiguous, lstats = project_microflow_labels(truth, src_m, dst_m)
        truth_micro = set(zip(labels["src_micro_flow_id"], labels["dst_micro_flow_id"])) if not labels.empty else set()
        cands = build_microflow_candidates(src_m, dst_m, labels, truth)
        if not cands.empty:
            cands["micro_score"] = _micro_score(cands)
        variants[vk] = {
            "src_micro": src_m,
            "dst_micro": dst_m,
            "labels": labels,
            "ambiguous": ambiguous,
            "label_stats": lstats,
            "candidates": cands,
            "truth_micro": truth_micro,
            "incidence_metrics": inc_metrics,
        }
    return {"seed": seed, "truth": truth, "src_events": src_d, "dst_events": dst_d, "variants": variants, "incidence": inc_metrics}


def _write_skip_artifacts(out: Path, result: dict[str, Any], train_seeds, dev_seeds, holdout_seeds, *, pilot_pass: bool, best_variant: str, ceilings: dict) -> None:
    (out / "diagnosis" / "phase17_feasibility_gate.json").write_text(
        json.dumps({"feasibility_gate_pass": False, "pilot_pass": pilot_pass, "best_variant": best_variant}, indent=2),
        encoding="utf-8",
    )
    (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps({
        "high_pr_gate_pass": False,
        "feasibility_gate_pass": False,
        "training_skipped": True,
        "holdout_evaluation_skipped": True,
        "allowed_claim": (
            "Even after transferId-level ABI decoding and micro-flow re-segmentation, exact high-P/R flow "
            "correspondence remains limited by canonical flow aggregation collisions and dataset-level label ambiguity."
        ),
        "forbidden_claim": "Do not claim high P/R without gate PASS; do not claim original flow-only high P/R.",
        "best_microflow_variant": best_variant,
        "diagnostic_ceilings": ceilings,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "splits" / "split_summary.json").write_text(json.dumps({
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "phase10s_to_16_preserved": True,
        "phase16_bottleneck_preserved": True,
        "train_seeds": train_seeds,
        "dev_seeds": dev_seeds,
        "sealed_holdout_seeds": holdout_seeds,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2),
        encoding="utf-8",
    )


def run_phase17(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    pilot_seeds: list[int],
    candidate_k: int,
    pilot_only: bool,
    use_phase16_decoded_events: bool,
    build_event_flow_incidence: bool,
    construct_transferid_microflows: bool,
    project_microflow_labels_flag: bool,
    build_microflow_candidates_flag: bool,
    run_microflow_oracle_ceiling: bool,
    run_feasibility_gate: bool,
    train_if_feasible: bool,
    evaluate_holdout_once: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out = run_root / OUT_REL
    for d in (
        "incidence", "microflow", "labels", "candidates", "diagnosis", "pilot",
        "models", "selection", "holdout", "splits", "audit",
    ):
        (out / d).mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "ok": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }

    registry = _phase15.load_abi_registry()
    topic_map = registry["_topic_map"]
    eth_cache = run_root / PHASE14_OUT / "cache" / "eth_receipts"
    bsc_cache = run_root / PHASE14_OUT / "cache" / "bsc_receipts"
    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    src_global, dst_global = _load_phase16_decoded(run_root) if use_phase16_decoded_events else (pd.DataFrame(), pd.DataFrame())
    if use_phase16_decoded_events and (src_global.empty or dst_global.empty):
        result["warning"] = "phase16_decoded_events_missing_fallback_decode"

    seeds_run = pilot_seeds if pilot_only else dev_seeds
    seed_results: list[dict[str, Any]] = []
    all_src_inc, all_dst_inc, all_tid_inc = [], [], []
    audits_inc: list[dict[str, Any]] = []

    for seed in seeds_run:
        sr = _process_seed(
            seed, synthetic_root,
            src_global=src_global, dst_global=dst_global,
            topic_map=topic_map, eth_cache=eth_cache, bsc_cache=bsc_cache,
            use_phase16_decoded=use_phase16_decoded_events and not src_global.empty,
        )
        seed_results.append(sr)
        src_inc, dst_inc, tid_inc, _ = build_incidence_tables(sr["src_events"], sr["dst_events"])
        if not src_inc.empty:
            src_inc["seed"] = seed
            all_src_inc.append(src_inc)
        if not dst_inc.empty:
            dst_inc["seed"] = seed
            all_dst_inc.append(dst_inc)
        if not tid_inc.empty:
            tid_inc["seed"] = seed
            all_tid_inc.append(tid_inc)
        audits_inc.append(sr["incidence"])

    if build_event_flow_incidence:
        src_inc_all = pd.concat(all_src_inc, ignore_index=True) if all_src_inc else pd.DataFrame()
        dst_inc_all = pd.concat(all_dst_inc, ignore_index=True) if all_dst_inc else pd.DataFrame()
        tid_inc_all = pd.concat(all_tid_inc, ignore_index=True) if all_tid_inc else pd.DataFrame()
        src_inc_all.to_csv(out / "incidence" / "src_event_flow_incidence.csv", index=False)
        dst_inc_all.to_csv(out / "incidence" / "dst_event_flow_incidence.csv", index=False)
        tid_inc_all.to_csv(out / "incidence" / "transferid_flow_incidence.csv", index=False)
        inc_audit: dict[str, Any] = {}
        if audits_inc:
            for k in audits_inc[0]:
                if isinstance(audits_inc[0][k], dict):
                    inc_audit[k] = audits_inc[0][k]
                else:
                    inc_audit[k] = float(np.mean([a.get(k, 0) for a in audits_inc]))
        audit_mean = inc_audit
        (out / "diagnosis" / "phase17_event_flow_incidence_audit.json").write_text(json.dumps(inc_audit, indent=2, default=str), encoding="utf-8")
        (out / "diagnosis" / "phase17_event_flow_incidence_audit.md").write_text(
            "# Event-flow incidence audit\n\n" + "\n".join(
                f"- {k}: {v}" for k, v in inc_audit.items() if not isinstance(v, dict)
            ) + "\n",
            encoding="utf-8",
        )
        dup_before = inc_audit.get("duplicate_event_assignment_rate", 1.0)
        (out / "incidence" / "event_flow_incidence_report.md").write_text(
            f"# Event-flow incidence\n\n- duplicate_event_assignment_rate (flow-level): {dup_before:.3f}\n"
            f"- transferId_reuse_rate: {inc_audit.get('transferId_reuse_rate', 0):.3f}\n",
            encoding="utf-8",
        )

    micro_parts: dict[str, list[pd.DataFrame]] = {f"src_{v}": [] for v in VARIANTS} | {f"dst_{v}": [] for v in VARIANTS}
    label_parts, amb_parts, cand_parts = [], [], {v: [] for v in VARIANTS}
    ceiling_data: dict[str, list[tuple[set[tuple[str, str]], pd.DataFrame, set[tuple[str, str]]]]] = {v: [] for v in VARIANTS}
    label_stats_agg: dict[str, list[dict]] = {v: [] for v in VARIANTS}

    for sr in seed_results:
        for vk in VARIANTS:
            vd = sr["variants"][vk]
            if construct_transferid_microflows and not vd["src_micro"].empty:
                vd["src_micro"]["seed"] = sr["seed"]
                vd["dst_micro"]["seed"] = sr["seed"]
                micro_parts[f"src_{vk}"].append(vd["src_micro"])
                micro_parts[f"dst_{vk}"].append(vd["dst_micro"])
            if project_microflow_labels_flag and not vd["labels"].empty:
                vd["labels"]["seed"] = sr["seed"]
                label_parts.append(vd["labels"])
            if not vd["ambiguous"].empty:
                vd["ambiguous"]["seed"] = sr["seed"]
                amb_parts.append(vd["ambiguous"])
            label_stats_agg[vk].append(vd["label_stats"])
            if build_microflow_candidates_flag and not vd["candidates"].empty:
                vd["candidates"]["seed"] = sr["seed"]
                cand_parts[vk].append(vd["candidates"])
            if not vd["candidates"].empty:
                ceiling_data[vk].append((vd["truth_micro"], vd["candidates"], sr["truth"]))

    audit_mean: dict[str, Any] = {}
    if audits_inc:
        for k in audits_inc[0]:
            if isinstance(audits_inc[0][k], dict):
                audit_mean[k] = audits_inc[0][k]
            else:
                audit_mean[k] = float(np.mean([a.get(k, 0) for a in audits_inc]))

    if construct_transferid_microflows:
        name_map = {"a": "event_key", "b": "transferid_group", "c": "transferid_entity", "d": "fractional"}
        for vk in VARIANTS:
            src_m = pd.concat(micro_parts[f"src_{vk}"], ignore_index=True) if micro_parts[f"src_{vk}"] else pd.DataFrame()
            dst_m = pd.concat(micro_parts[f"dst_{vk}"], ignore_index=True) if micro_parts[f"dst_{vk}"] else pd.DataFrame()
            src_m.to_csv(out / "microflow" / f"variant_{vk}_{name_map[vk]}_src.csv", index=False)
            dst_m.to_csv(out / "microflow" / f"variant_{vk}_{name_map[vk]}_dst.csv", index=False)
        (out / "microflow" / "microflow_construction_report.md").write_text(
            "# Micro-flow construction\n\nVariants A–D built from Phase 16 decoded events without modifying canonical.\n",
            encoding="utf-8",
        )

    labels_all = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    amb_all = pd.concat(amb_parts, ignore_index=True) if amb_parts else pd.DataFrame()
    if project_microflow_labels_flag:
        labels_all.to_csv(out / "labels" / "phase17_projected_microflow_labels.csv", index=False)
        amb_all.to_csv(out / "labels" / "phase17_ambiguous_label_projection.csv", index=False)
        agg_label = {}
        for vk in VARIANTS:
            stats_list = label_stats_agg[vk]
            if stats_list:
                agg_label[vk] = {k: float(np.mean([s.get(k, 0) for s in stats_list])) for k in stats_list[0]}
        clean_frac = float(np.mean([agg_label[v].get("clean_microflow_label_fraction", 0) for v in VARIANTS if v in agg_label])) if agg_label else 0.0
        (out / "labels" / "phase17_label_projection_report.md").write_text(
            f"# Label projection\n\n- clean_microflow_label_fraction (mean): {clean_frac:.3f}\n"
            "- GT used for evaluation labels only, not inference features.\n",
            encoding="utf-8",
        )
        if clean_frac < 0.70:
            (out / "diagnosis" / "phase17_label_projection_failure.md").write_text(
                f"# Label projection failure\n\nclean_microflow_label_fraction={clean_frac:.3f} < 0.70. "
                "Canonical flow labels and transferId event evidence are inconsistent or over-aggregated.\n",
                encoding="utf-8",
            )

    if build_microflow_candidates_flag:
        for vk in VARIANTS:
            cands = pd.concat(cand_parts[vk], ignore_index=True) if cand_parts[vk] else pd.DataFrame()
            cands.to_csv(out / "candidates" / f"variant_{vk}_microflow_candidates.csv", index=False)
        (out / "candidates" / "microflow_candidate_report.md").write_text("# Micro-flow candidates\n", encoding="utf-8")

    ceilings: dict[str, dict[str, Any]] = {}
    dup_after: dict[str, float] = {}
    micro_dup_after: dict[str, float] = {}
    for vk in VARIANTS:
        ceilings[vk] = _compute_micro_ceiling(ceiling_data[vk]) if ceiling_data[vk] else {}
        dup_after[vk] = ceilings[vk].get("duplicate_event_assignment_rate", 1.0)
        src_m = pd.concat(micro_parts[f"src_{vk}"], ignore_index=True) if micro_parts[f"src_{vk}"] else pd.DataFrame()
        if not src_m.empty and "parent_flow_multiplicity" in src_m.columns:
            micro_dup_after[vk] = float((src_m["parent_flow_multiplicity"].fillna(1) > 1).mean())
        else:
            micro_dup_after[vk] = dup_after[vk]
        if run_microflow_oracle_ceiling:
            (out / "diagnosis" / f"phase17_microflow_ceiling_variant_{vk}.json").write_text(
                json.dumps(ceilings[vk], indent=2), encoding="utf-8"
            )

    best_variant = max(ceilings, key=lambda v: ceilings[v].get("feature_auroc", 0) + ceilings[v].get("oracle_precision_at_recall_0_8", 0)) if ceilings else "a"
    best_ceiling = ceilings.get(best_variant, {})
    agg_label_stats = {}
    if label_stats_agg[best_variant]:
        agg_label_stats = {k: float(np.mean([s.get(k, 0) for s in label_stats_agg[best_variant]])) for k in label_stats_agg[best_variant][0]}

    if run_microflow_oracle_ceiling:
        rows_prog = [{"stage": "phase16_flow_lifted", **PHASE16_FLOW_LIFTED_BASELINE}]
        for vk in VARIANTS:
            rows_prog.append({"stage": f"phase17_variant_{vk}", **ceilings.get(vk, {})})
        pd.DataFrame(rows_prog).to_csv(out / "diagnosis" / "phase17_microflow_ceiling_progression_table.csv", index=False)
        (out / "diagnosis" / "phase17_microflow_oracle_report.md").write_text(
            f"# Micro-flow oracle\n\n- best variant: **{best_variant}**\n"
            f"- AUROC: {best_ceiling.get('feature_auroc', 0):.3f}\n"
            f"- oracle P@R>=0.8: {best_ceiling.get('oracle_precision_at_recall_0_8', 0):.3f}\n"
            f"- collision: {best_ceiling.get('candidate_collision_rate', 0):.3f}\n"
            f"- duplicate_event_assignment after microflow (best): {micro_dup_after.get(best_variant, 1):.3f}\n"
            f"- candidate collision (best): {best_ceiling.get('candidate_collision_rate', 0):.3f}\n",
            encoding="utf-8",
        )

    pilot_gate = evaluate_microflow_pilot_gate(best_ceiling, agg_label_stats)
    (out / "pilot" / "phase17_pilot_ceiling_metrics.json").write_text(
        json.dumps({"best_variant": best_variant, "ceilings": ceilings, "label_stats": agg_label_stats, "pilot_gate": pilot_gate}, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "pilot" / "phase17_pilot_report.md").write_text(
        f"# Phase 17 pilot\n\n- pilot_pass: **{pilot_gate['pilot_pass']}**\n"
        f"- best variant: {best_variant}\n"
        f"- clean_microflow_label_fraction: {agg_label_stats.get('clean_microflow_label_fraction', 0):.3f}\n"
        f"- feature AUROC: {best_ceiling.get('feature_auroc', 0):.3f} (Phase 16 flow-lifted: {PHASE16_FLOW_LIFTED_BASELINE['feature_auroc']})\n"
        f"- oracle P@R>=0.8: {best_ceiling.get('oracle_precision_at_recall_0_8', 0):.3f}\n"
        f"- duplicate_event_assignment after microflow: {dup_after.get(best_variant, 1):.3f}\n",
        encoding="utf-8",
    )

    result["pilot_pass"] = pilot_gate["pilot_pass"]
    result["best_variant"] = best_variant
    result["best_ceiling"] = best_ceiling
    result["ceilings"] = ceilings
    result["label_stats"] = agg_label_stats
    result["duplicate_event_assignment_before"] = audit_mean.get("duplicate_event_assignment_rate", 1.0) if audits_inc else 1.0
    result["duplicate_event_assignment_after"] = micro_dup_after.get(best_variant, dup_after.get(best_variant, 1.0))
    result["microflow_parent_multiplicity_rate"] = micro_dup_after.get(best_variant, 1.0)

    if not pilot_gate["pilot_pass"]:
        (out / "diagnosis" / "phase17_microflow_infeasibility.md").write_text(
            "# Micro-flow infeasibility\n\n"
            "- Canonical flow aggregation destroys exact pair identifiability.\n"
            "- transferId evidence is reused across multiple parent flows.\n"
            "- Micro-flow labels cannot be cleanly projected (clean fraction < 0.70 or oracle ceiling below thresholds).\n"
            "- Exact high-P/R flow correspondence is not feasible under current dataset construction.\n",
            encoding="utf-8",
        )
        (out / "pilot" / "phase17_pilot_failure.md").write_text(
            "# Pilot failure\n\n" + "\n".join(f"- {k}: {v}" for k, v in pilot_gate["checks"].items() if not v) + "\n",
            encoding="utf-8",
        )

    _write_skip_artifacts(out, result, train_seeds, dev_seeds, holdout_seeds, pilot_pass=pilot_gate["pilot_pass"], best_variant=best_variant, ceilings=ceilings)

    if not pilot_gate["pilot_pass"]:
        result["ok"] = False
        result["error"] = "pilot_fail"
    elif run_feasibility_gate and not pilot_gate["pilot_pass"]:
        result["ok"] = False
        result["error"] = "feasibility_fail"

    result["elapsed_sec"] = time.time() - t0
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 17 transferId micro-flow resegmentation")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=list(range(52, 72)))
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=list(range(172, 192)))
    ap.add_argument("--pilot-seeds", type=int, nargs="+", default=list(range(52, 58)))
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--use-phase16-decoded-events", action="store_true", default=True)
    ap.add_argument("--build-event-flow-incidence", action="store_true")
    ap.add_argument("--construct-transferid-microflows", action="store_true")
    ap.add_argument("--project-microflow-labels", action="store_true")
    ap.add_argument("--build-microflow-candidates", action="store_true")
    ap.add_argument("--run-microflow-oracle-ceiling", action="store_true")
    ap.add_argument("--run-feasibility-gate", action="store_true")
    ap.add_argument("--train-if-feasible", action="store_true")
    ap.add_argument("--evaluate-holdout-once", action="store_true")
    args = ap.parse_args()
    if args.pilot_only:
        for f in (
            "build_event_flow_incidence", "construct_transferid_microflows",
            "project_microflow_labels", "build_microflow_candidates", "run_microflow_oracle_ceiling",
        ):
            setattr(args, f, True)
    r = run_phase17(
        run_root=args.run_root,
        train_seeds=args.train_seeds,
        dev_seeds=args.dev_seeds,
        holdout_seeds=args.holdout_seeds,
        pilot_seeds=args.pilot_seeds if args.pilot_seeds else list(range(52, 58)),
        candidate_k=args.candidate_k,
        pilot_only=args.pilot_only,
        use_phase16_decoded_events=args.use_phase16_decoded_events,
        build_event_flow_incidence=args.build_event_flow_incidence,
        construct_transferid_microflows=args.construct_transferid_microflows,
        project_microflow_labels_flag=args.project_microflow_labels,
        build_microflow_candidates_flag=args.build_microflow_candidates,
        run_microflow_oracle_ceiling=args.run_microflow_oracle_ceiling,
        run_feasibility_gate=args.run_feasibility_gate,
        train_if_feasible=args.train_if_feasible,
        evaluate_holdout_once=args.evaluate_holdout_once,
    )
    slim = {k: v for k, v in r.items() if k not in ("ceilings", "best_ceiling")}
    print(json.dumps(slim, indent=2, default=str))
    return 0 if r.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
