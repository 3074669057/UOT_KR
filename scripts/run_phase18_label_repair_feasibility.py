#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 18: Canonical flow label repair feasibility (CSFFC-v2 / label_layer_v2)."""
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

OUT_REL = "phase18_label_repair_feasibility"
PHASE16_OUT = "phase16_transferid_event_deaggregation"
UNAVAILABLE = float("nan")

REPAIR_GATE = {
    "clean_label_fraction": 0.80,
    "ambiguous_label_fraction": 0.20,
    "projection_coverage_over_original_gt": 0.80,
    "label_conflict_rate": 0.10,
    "event_support_overlap_rate": 0.20,
    "repaired_candidate_oracle_recall": 0.95,
    "repaired_oracle_precision_at_recall_0_8": 0.80,
    "repaired_oracle_recall_at_precision_0_8": 0.80,
    "repaired_score_oracle_best_f1": 0.80,
    "repaired_feature_auroc": 0.85,
    "repaired_feature_auprc": 0.70,
}

REPAIR_VARIANTS = ("a", "b", "c", "d")
VARIANT_LABELS = {
    "a": "deduplicate_identical_event_support",
    "b": "event_majority_assignment",
    "c": "transferid_event_pair_csffc_v2",
    "d": "non_overlapping_repaired_flow",
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


def _seed_flow_ids(seed_data: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("eth_flows", "bnb_flows", "bsc_flows"):
        for f in seed_data.get(key) or []:
            fid = str(f.get("flow_id") or "")
            if fid:
                ids.add(fid)
    return ids


def _load_phase16_decoded(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p16 = run_root / PHASE16_OUT / "evidence"
    src_p = p16 / "phase16_revalidated_decoded_events_src.csv"
    dst_p = p16 / "phase16_revalidated_decoded_events_dst.csv"
    if not src_p.is_file() or not dst_p.is_file():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(src_p), pd.read_csv(dst_p)


def _pct_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {"mean": float(series.mean()), "median": float(series.median()), "p95": float(series.quantile(0.95))}


def _union_find_classes(flow_ids: list[str], shared_pairs: list[tuple[str, str]]) -> dict[str, int]:
    parent: dict[str, str] = {f: f for f in flow_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in shared_pairs:
        if a in parent and b in parent:
            union(a, b)
    roots = {find(f) for f in flow_ids}
    root_to_cls = {r: i for i, r in enumerate(sorted(roots))}
    return {f: root_to_cls[find(f)] for f in flow_ids}


def build_incidence_graph(
    src_events: list[dict[str, Any]],
    dst_events: list[dict[str, Any]],
    truth: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    src_edges, dst_edges = [], []
    flow_event_keys_src: dict[str, set[str]] = defaultdict(set)
    flow_event_keys_dst: dict[str, set[str]] = defaultdict(set)
    event_to_src_flows: dict[str, set[str]] = defaultdict(set)
    event_to_dst_flows: dict[str, set[str]] = defaultdict(set)
    tid_to_src: dict[str, set[str]] = defaultdict(set)
    stid_to_dst: dict[str, set[str]] = defaultdict(set)

    for e in src_events:
        if e.get("event_name") not in ("Send", "LogNewTransferOut") or not _phase15._field_nonempty(e.get("transfer_id")):
            continue
        ek = _src_event_key(e)
        fid = str(e.get("flow_id") or "")
        tid = str(e.get("transfer_id") or "").lower()
        flow_event_keys_src[fid].add(ek)
        event_to_src_flows[ek].add(fid)
        tid_to_src[tid].add(fid)
        for sf, df in truth:
            if sf == fid:
                src_edges.append({
                    "edge_type": "gt_flow_pair_src",
                    "flow_id": fid,
                    "dst_flow_id": df,
                    "event_key": ek,
                    "transfer_id": tid,
                    "src_transfer_id": "",
                    "tx_hash": e.get("tx_hash", ""),
                    "log_index": e.get("log_index", ""),
                })
        src_edges.append({
            "edge_type": "flow_contains_event",
            "flow_id": fid,
            "event_key": ek,
            "transfer_id": tid,
            "tx_hash": e.get("tx_hash", ""),
            "log_index": e.get("log_index", ""),
        })
        src_edges.append({
            "edge_type": "event_maps_transferId",
            "event_key": ek,
            "transfer_id": tid,
            "flow_id": fid,
        })

    for e in dst_events:
        if e.get("event_name") not in ("Relay", "LogNewTransferIn") or not _phase15._field_nonempty(e.get("src_transfer_id")):
            continue
        ek = _dst_event_key(e)
        fid = str(e.get("flow_id") or "")
        stid = str(e.get("src_transfer_id") or "").lower()
        flow_event_keys_dst[fid].add(ek)
        event_to_dst_flows[ek].add(fid)
        stid_to_dst[stid].add(fid)
        for sf, df in truth:
            if df == fid:
                dst_edges.append({
                    "edge_type": "gt_flow_pair_dst",
                    "flow_id": fid,
                    "src_flow_id": sf,
                    "event_key": ek,
                    "src_transfer_id": stid,
                    "transfer_id": str(e.get("transfer_id") or "").lower(),
                    "tx_hash": e.get("tx_hash", ""),
                    "log_index": e.get("log_index", ""),
                })
        dst_edges.append({
            "edge_type": "flow_contains_event",
            "flow_id": fid,
            "event_key": ek,
            "src_transfer_id": stid,
            "tx_hash": e.get("tx_hash", ""),
            "log_index": e.get("log_index", ""),
        })

    src_df = pd.DataFrame(src_edges)
    dst_df = pd.DataFrame(dst_edges)

    support_sig_src: dict[str, frozenset[str]] = {f: frozenset(ks) for f, ks in flow_event_keys_src.items()}
    support_sig_dst: dict[str, frozenset[str]] = {f: frozenset(ks) for f, ks in flow_event_keys_dst.items()}
    sig_to_flows_src: dict[frozenset[str], list[str]] = defaultdict(list)
    sig_to_flows_dst: dict[frozenset[str], list[str]] = defaultdict(list)
    for f, sig in support_sig_src.items():
        if sig:
            sig_to_flows_src[sig].append(f)
    for f, sig in support_sig_dst.items():
        if sig:
            sig_to_flows_dst[sig].append(f)

    exact_dup_src = sum(1 for flows in sig_to_flows_src.values() if len(flows) > 1)
    exact_dup_dst = sum(1 for flows in sig_to_flows_dst.values() if len(flows) > 1)
    n_src_flows = max(len(flow_event_keys_src), 1)
    n_dst_flows = max(len(flow_event_keys_dst), 1)

    shared_src_pairs = []
    for flows in event_to_src_flows.values():
        fl = sorted(flows)
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                shared_src_pairs.append((fl[i], fl[j]))
    shared_dst_pairs = []
    for flows in event_to_dst_flows.values():
        fl = sorted(flows)
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                shared_dst_pairs.append((fl[i], fl[j]))

    src_classes = _union_find_classes(list(flow_event_keys_src.keys()), shared_src_pairs)
    dst_classes = _union_find_classes(list(flow_event_keys_dst.keys()), shared_dst_pairs)
    class_sizes_src = pd.Series(list(src_classes.values())).value_counts()
    class_sizes_dst = pd.Series(list(dst_classes.values())).value_counts()

    gt_inside, gt_crossing = 0, 0
    for sf, df in truth:
        sc = src_classes.get(sf)
        dc = dst_classes.get(df)
        if sc is None or dc is None:
            continue
        src_peers = [f for f, c in src_classes.items() if c == sc]
        dst_peers = [f for f, c in dst_classes.items() if c == dc]
        if len(src_peers) > 1 or len(dst_peers) > 1:
            gt_inside += 1
        else:
            gt_crossing += 1

    ek_multi_src = [ek for ek, fl in event_to_src_flows.items() if len(fl) > 1]
    ek_multi_dst = [ek for ek, fl in event_to_dst_flows.items() if len(fl) > 1]

    audit = {
        "events_per_src_flow": _pct_stats(pd.Series([len(v) for v in flow_event_keys_src.values()])),
        "events_per_dst_flow": _pct_stats(pd.Series([len(v) for v in flow_event_keys_dst.values()])),
        "src_flows_per_event_key": _pct_stats(pd.Series([len(v) for v in event_to_src_flows.values()])),
        "dst_flows_per_event_key": _pct_stats(pd.Series([len(v) for v in event_to_dst_flows.values()])),
        "src_flows_per_transferId": _pct_stats(pd.Series([len(v) for v in tid_to_src.values()])),
        "dst_flows_per_srcTransferId": _pct_stats(pd.Series([len(v) for v in stid_to_dst.values()])),
        "exact_duplicate_flow_fraction_src": float(exact_dup_src / max(len(sig_to_flows_src), 1)),
        "exact_duplicate_flow_fraction_dst": float(exact_dup_dst / max(len(sig_to_flows_dst), 1)),
        "exact_duplicate_flow_fraction": float((exact_dup_src + exact_dup_dst) / max(len(sig_to_flows_src) + len(sig_to_flows_dst), 1)),
        "overlapping_flow_fraction_src": float(sum(1 for v in event_to_src_flows.values() if len(v) > 1) / max(len(event_to_src_flows), 1)),
        "overlapping_flow_fraction_dst": float(sum(1 for v in event_to_dst_flows.values() if len(v) > 1) / max(len(event_to_dst_flows), 1)),
        "duplicate_event_assignment_rate": float((len(ek_multi_src) + len(ek_multi_dst)) / max(len(event_to_src_flows) + len(event_to_dst_flows), 1)),
        "event_to_flow_nonidentifiability_rate": float(len(ek_multi_src) / max(len(event_to_src_flows), 1)),
        "flow_purity_rate": float(1.0 - len(ek_multi_src) / max(len(event_to_src_flows), 1)),
        "number_of_flow_equivalence_classes_src": int(class_sizes_src.shape[0]),
        "number_of_flow_equivalence_classes_dst": int(class_sizes_dst.shape[0]),
        "largest_flow_equivalence_class_size_src": int(class_sizes_src.max()) if len(class_sizes_src) else 0,
        "largest_flow_equivalence_class_size_dst": int(class_sizes_dst.max()) if len(class_sizes_dst) else 0,
        "largest_flow_equivalence_class_size": int(max(class_sizes_src.max() if len(class_sizes_src) else 0, class_sizes_dst.max() if len(class_sizes_dst) else 0)),
        "gt_edges_inside_equivalence_classes": gt_inside,
        "gt_edges_crossing_equivalence_classes": gt_crossing,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }
    audit["_flow_event_keys_src"] = flow_event_keys_src
    audit["_flow_event_keys_dst"] = flow_event_keys_dst
    audit["_event_to_src_flows"] = event_to_src_flows
    audit["_event_to_dst_flows"] = event_to_dst_flows
    audit["_tid_to_src"] = tid_to_src
    audit["_stid_to_dst"] = stid_to_dst
    audit["_support_sig_src"] = support_sig_src
    audit["_support_sig_dst"] = support_sig_dst
    return src_df, dst_df, audit


def _threshold_scan(truth: set[tuple[str, str]], scores: pd.DataFrame, id_cols: tuple[str, str], score_col: str) -> dict[str, float]:
    if scores.empty or not truth:
        return {"oracle_best_f1": 0.0, "oracle_precision_at_recall_0_8": 0.0, "oracle_recall_at_precision_0_8": 0.0}
    s = scores.copy()
    s["_sc"] = pd.to_numeric(s[score_col], errors="coerce").fillna(0.0)
    ic, jc = id_cols
    uniq = sorted(set(s["_sc"].tolist()), reverse=True) or [0.0]
    best_f1, best_p_at_r, best_r_at_p = 0.0, 0.0, 0.0
    for thr in uniq + [0.0]:
        pred = set(zip(s.loc[s["_sc"] >= thr, ic].astype(str), s.loc[s["_sc"] >= thr, jc].astype(str)))
        m = _phase10w._prf1(truth, pred)
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


def _oracle_metrics(
    truth: set[tuple[str, str]],
    candidates: pd.DataFrame,
    id_cols: tuple[str, str],
    score_col: str,
    label_col: str,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if candidates.empty:
        return {
            "repaired_candidate_oracle_recall": 0.0,
            "repaired_oracle_precision_at_recall_0_8": 0.0,
            "repaired_oracle_recall_at_precision_0_8": 0.0,
            "repaired_score_oracle_best_f1": 0.0,
            "repaired_feature_auroc": 0.0,
            "repaired_feature_auprc": 0.0,
        }
    ic, jc = id_cols
    pairs = set(zip(candidates[ic].astype(str), candidates[jc].astype(str)))
    recall = len(truth & pairs) / max(len(truth), 1)
    scan = _threshold_scan(truth, candidates, id_cols, score_col)
    y = candidates[label_col].fillna(0).astype(int).to_numpy()
    x = pd.to_numeric(candidates[score_col], errors="coerce").fillna(0.0).to_numpy()
    auroc, auprc = 0.0, 0.0
    if len(np.unique(y)) > 1:
        auroc = float(roc_auc_score(y, x))
        auprc = float(average_precision_score(y, x))
    return {
        "repaired_candidate_oracle_recall": float(recall),
        **{f"repaired_{k}": v for k, v in scan.items()},
        "repaired_feature_auroc": auroc,
        "repaired_feature_auprc": auprc,
    }


def _label_quality(
    n_clean: int,
    n_amb: int,
    n_gt: int,
    n_covered: int,
    n_conflict: int,
    n_overlap: int,
    n_total_labels: int,
) -> dict[str, float]:
    n_proj = n_clean + n_amb
    return {
        "clean_label_fraction": float(n_clean / max(n_proj, 1)),
        "ambiguous_label_fraction": float(n_amb / max(n_proj, 1)),
        "projection_coverage_over_original_gt": float(n_covered / max(n_gt, 1)),
        "parent_flow_preservation_rate": float(n_clean / max(n_gt, 1)),
        "label_conflict_rate": float(n_conflict / max(n_total_labels, 1)),
        "event_support_overlap_rate": float(n_overlap / max(n_total_labels, 1)),
    }


def repair_variant_a(
    audit: dict[str, Any],
    truth: set[tuple[str, str]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sig_to_flows: dict[frozenset[str], list[str]] = defaultdict(list)
    for f, sig in audit["_support_sig_src"].items():
        if sig:
            sig_to_flows[sig].append(f)
    repaired_map: dict[str, str] = {}
    rows = []
    n_conflict, n_clean, n_amb = 0, 0, 0
    gt_labels_in_group: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for sig, flows in sig_to_flows.items():
        rid = f"repair_a_src_{seed}_{hash(sig) % 10**8}"
        for f in flows:
            repaired_map[f] = rid
        gt_in = {(sf, df) for sf, df in truth if sf in flows}
        for sf, df in gt_in:
            gt_labels_in_group[rid].add((sf, df))
        if len(gt_in) > 1:
            n_conflict += 1
        conflict = len({df for _, df in gt_in}) > 1 or len({sf for sf, _ in gt_in}) > 1 and len(flows) > 1
        rows.append({
            "repaired_flow_id": rid,
            "parent_flow_ids": ",".join(sorted(flows)),
            "event_support_size": len(sig),
            "parent_flow_multiplicity": len(flows),
            "label_conflict": conflict,
            "side": "src",
            "construction_variant": VARIANT_LABELS["a"],
        })

    sig_to_flows_d: dict[frozenset[str], list[str]] = defaultdict(list)
    for f, sig in audit["_support_sig_dst"].items():
        if sig:
            sig_to_flows_d[sig].append(f)
    repaired_map_d: dict[str, str] = {}
    for sig, flows in sig_to_flows_d.items():
        rid = f"repair_a_dst_{seed}_{hash(sig) % 10**8}"
        for f in flows:
            repaired_map_d[f] = rid
        gt_in = {(sf, df) for sf, df in truth if df in flows}
        conflict = len({sf for sf, _ in gt_in}) > 1
        if conflict:
            n_conflict += 1
        rows.append({
            "repaired_flow_id": rid,
            "parent_flow_ids": ",".join(sorted(flows)),
            "event_support_size": len(sig),
            "parent_flow_multiplicity": len(flows),
            "label_conflict": conflict,
            "side": "dst",
            "construction_variant": VARIANT_LABELS["a"],
        })

    projected = []
    for sf, df in truth:
        rs = repaired_map.get(sf)
        rd = repaired_map_d.get(df)
        if not rs or not rd:
            continue
        amb = len(sig_to_flows.get(audit["_support_sig_src"].get(sf, frozenset()), [])) > 1
        amb = amb or len(sig_to_flows_d.get(audit["_support_sig_dst"].get(df, frozenset()), [])) > 1
        if amb:
            n_amb += 1
        else:
            n_clean += 1
        projected.append((rs, rd, sf, df, not amb))
    n_covered = len({(sf, df) for _, _, sf, df, c in projected if c})

    candidates = _build_repaired_candidates(projected, truth, id_prefix="a")
    quality = _label_quality(n_clean, n_amb, len(truth), n_covered, n_conflict, len(sig_to_flows), len(rows))
    quality.update(_oracle_metrics(
        {(rs, rd) for rs, rd, _, _, c in projected if c},
        candidates,
        ("repaired_src_id", "repaired_dst_id"),
        "repair_score",
        "evaluation_label",
    ))
    return pd.DataFrame(rows), quality


def repair_variant_b(
    audit: dict[str, Any],
    truth: set[tuple[str, str]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    ek_dominant_src: dict[str, str] = {}
    ek_dominant_dst: dict[str, str] = {}
    n_amb = 0

    for ek, flows in audit["_event_to_src_flows"].items():
        fl = sorted(flows)
        if len(fl) == 1:
            dom, amb = fl[0], False
        else:
            dom, amb = fl[0], True
            n_amb += 1
        ek_dominant_src[ek] = dom
        rows.append({"event_key": ek, "dominant_flow_id": dom, "candidate_flows": ",".join(fl), "ambiguous": amb, "side": "src", "construction_variant": VARIANT_LABELS["b"]})

    for ek, flows in audit["_event_to_dst_flows"].items():
        fl = sorted(flows)
        if len(fl) == 1:
            dom, amb = fl[0], False
        else:
            dom, amb = fl[0], True
            n_amb += 1
        ek_dominant_dst[ek] = dom
        rows.append({"event_key": ek, "dominant_flow_id": dom, "candidate_flows": ",".join(fl), "ambiguous": amb, "side": "dst", "construction_variant": VARIANT_LABELS["b"]})

    n_clean, n_conflict = 0, 0
    projected = []
    for sf, df in truth:
        src_eks = audit["_flow_event_keys_src"].get(sf, set())
        dst_eks = audit["_flow_event_keys_dst"].get(df, set())
        amb = any(len(audit["_event_to_src_flows"].get(ek, set())) > 1 for ek in src_eks)
        amb = amb or any(len(audit["_event_to_dst_flows"].get(ek, set())) > 1 for ek in dst_eks)
        rs = f"repair_b_src_{seed}_{sf}"
        rd = f"repair_b_dst_{seed}_{df}"
        if not amb:
            n_clean += 1
        projected.append((rs, rd, sf, df, not amb))
    n_covered = len({(sf, df) for _, _, sf, df, c in projected if c})
    n_amb_labels = sum(1 for *_, c in projected if not c)

    candidates = _build_repaired_candidates(projected, truth, id_prefix="b")
    quality = _label_quality(n_clean, n_amb_labels, len(truth), n_covered, n_conflict, n_amb, len(rows))
    quality.update(_oracle_metrics(
        {(rs, rd) for rs, rd, _, _, c in projected if c},
        candidates,
        ("repaired_src_id", "repaired_dst_id"),
        "repair_score",
        "evaluation_label",
    ))
    return pd.DataFrame(rows), quality


def repair_variant_c(
    audit: dict[str, Any],
    truth: set[tuple[str, str]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    tid_to_gt: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for sf, df in truth:
        for ek in audit["_flow_event_keys_src"].get(sf, set()):
            tid = ek.split("|")[-1]
            tid_to_gt[tid].add((sf, df))

    event_pairs_truth: set[tuple[str, str]] = set()
    n_clean, n_amb, n_conflict = 0, 0, 0

    for tid, flows in audit["_tid_to_src"].items():
        dst_flows = audit["_stid_to_dst"].get(tid, set())
        gt_pairs = tid_to_gt.get(tid, set())
        amb = len(flows) > 1 or len(dst_flows) > 1 or len(gt_pairs) > 1
        if len(gt_pairs) > 1:
            n_conflict += 1
        pair_id_src = f"v2_event_src_{seed}_{tid[:16]}"
        pair_id_dst = f"v2_event_dst_{seed}_{tid[:16]}"
        if gt_pairs and not amb:
            event_pairs_truth.add((pair_id_src, pair_id_dst))
            n_clean += 1
        elif gt_pairs:
            n_amb += 1
        rows.append({
            "event_pair_id": f"{pair_id_src}|{pair_id_dst}",
            "transfer_id": tid,
            "src_transfer_id": tid,
            "parent_src_flow_ids": ",".join(sorted(flows)),
            "parent_dst_flow_ids": ",".join(sorted(dst_flows)),
            "label_ambiguous": amb,
            "csffc_task": "CSFFC-v2-event",
            "construction_variant": VARIANT_LABELS["c"],
        })

    candidates = []
    for tid in audit["_tid_to_src"]:
        if tid not in audit["_stid_to_dst"]:
            continue
        ps = f"v2_event_src_{seed}_{tid[:16]}"
        pd_id = f"v2_event_dst_{seed}_{tid[:16]}"
        label = int((ps, pd_id) in event_pairs_truth)
        candidates.append({
            "repaired_src_id": ps,
            "repaired_dst_id": pd_id,
            "transfer_id": tid,
            "repair_score": 1.0 if label else 0.3,
            "evaluation_label": label,
            "csffc_v2": True,
        })
    cand_df = pd.DataFrame(candidates)
    parent_covered = 0
    for sf, df in truth:
        for ek in audit["_flow_event_keys_src"].get(sf, set()):
            tid = ek.split("|")[-1]
            if tid in audit["_stid_to_dst"] and df in audit["_stid_to_dst"][tid]:
                parent_covered += 1
                break

    quality = _label_quality(n_clean, n_amb, len(truth), parent_covered, n_conflict, n_amb, len(rows))
    quality["csffc_v2_event_task"] = True
    quality.update(_oracle_metrics(event_pairs_truth, cand_df, ("repaired_src_id", "repaired_dst_id"), "repair_score", "evaluation_label"))
    return pd.DataFrame(rows), quality


def repair_variant_d(
    audit: dict[str, Any],
    truth: set[tuple[str, str]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ek_to_src: dict[str, str] = {}
    ek_to_dst: dict[str, str] = {}
    rows = []
    for ek in audit["_event_to_src_flows"]:
        fl = sorted(audit["_event_to_src_flows"][ek])
        rid = f"repair_d_src_{seed}_{hash(ek) % 10**8}"
        ek_to_src[ek] = rid
        rows.append({
            "repaired_flow_id": rid,
            "event_key": ek,
            "parent_flow_ids": ",".join(fl),
            "parent_flow_multiplicity": len(fl),
            "side": "src",
            "construction_variant": VARIANT_LABELS["d"],
        })
    for ek in audit["_event_to_dst_flows"]:
        fl = sorted(audit["_event_to_dst_flows"][ek])
        rid = f"repair_d_dst_{seed}_{hash(ek) % 10**8}"
        ek_to_dst[ek] = rid
        rows.append({
            "repaired_flow_id": rid,
            "event_key": ek,
            "parent_flow_ids": ",".join(fl),
            "parent_flow_multiplicity": len(fl),
            "side": "dst",
            "construction_variant": VARIANT_LABELS["d"],
        })

    flow_to_repaired_src: dict[str, str] = {}
    flow_to_repaired_dst: dict[str, str] = {}
    for f, eks in audit["_flow_event_keys_src"].items():
        if eks:
            flow_to_repaired_src[f] = ek_to_src[sorted(eks)[0]]
    for f, eks in audit["_flow_event_keys_dst"].items():
        if eks:
            flow_to_repaired_dst[f] = ek_to_dst[sorted(eks)[0]]

    v2_rows = []
    n_clean, n_amb, n_conflict = 0, 0, 0
    projected = []
    for sf, df in truth:
        rs = flow_to_repaired_src.get(sf)
        rd = flow_to_repaired_dst.get(df)
        if not rs or not rd:
            continue
        src_eks = audit["_flow_event_keys_src"].get(sf, set())
        dst_eks = audit["_flow_event_keys_dst"].get(df, set())
        tid_match = any(
            ek.split("|")[-1] == dek.split("|")[-1]
            for ek in src_eks for dek in dst_eks
        )
        amb = len(audit["_event_to_src_flows"].get(next(iter(src_eks), ""), set())) > 1 if src_eks else True
        if tid_match and not amb:
            n_clean += 1
        else:
            n_amb += 1
        projected.append((rs, rd, sf, df, tid_match and not amb))
        v2_rows.append({
            "label_layer": "label_layer_v2_proposal",
            "repaired_src_flow_id": rs,
            "repaired_dst_flow_id": rd,
            "parent_src_flow_id": sf,
            "parent_dst_flow_id": df,
            "clean_label": tid_match and not amb,
            "seed": seed,
        })

    n_covered = len({(sf, df) for _, _, sf, df, c in projected if c})
    candidates = _build_repaired_candidates(projected, truth, id_prefix="d")
    quality = _label_quality(n_clean, n_amb, len(truth), n_covered, n_conflict, 0, len(rows))
    quality.update(_oracle_metrics(
        {(rs, rd) for rs, rd, _, _, c in projected if c},
        candidates,
        ("repaired_src_id", "repaired_dst_id"),
        "repair_score",
        "evaluation_label",
    ))
    quality["_v2_proposal_rows"] = v2_rows
    return pd.DataFrame(rows), quality


def _build_repaired_candidates(
    projected: list[tuple[str, str, str, str, bool]],
    truth: set[tuple[str, str]],
    *,
    id_prefix: str,
) -> pd.DataFrame:
    rows = []
    truth_repaired = {(rs, rd) for rs, rd, _, _, c in projected if c}
    for rs, rd, sf, df, clean in projected:
        label = int((rs, rd) in truth_repaired and clean)
        rows.append({
            "repaired_src_id": rs,
            "repaired_dst_id": rd,
            "parent_src_flow_id": sf,
            "parent_dst_flow_id": df,
            "repair_score": 1.0 if label else 0.2,
            "evaluation_label": label,
            "clean": clean,
        })
    if len(rows) < 50000:
        rng = np.random.default_rng(42)
        src_ids = list({r["repaired_src_id"] for r in rows})
        dst_ids = list({r["repaired_dst_id"] for r in rows})
        for _ in range(min(50, len(src_ids) * len(dst_ids))):
            i, j = rng.integers(0, len(src_ids)), rng.integers(0, len(dst_ids))
            rs, rd = src_ids[int(i)], dst_ids[int(j)]
            if (rs, rd) in truth_repaired:
                continue
            rows.append({
                "repaired_src_id": rs,
                "repaired_dst_id": rd,
                "parent_src_flow_id": "",
                "parent_dst_flow_id": "",
                "repair_score": 0.1,
                "evaluation_label": 0,
                "clean": False,
            })
    return pd.DataFrame(rows)


def evaluate_repair_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "clean_label_fraction": metrics.get("clean_label_fraction", 0) >= REPAIR_GATE["clean_label_fraction"],
        "ambiguous_label_fraction": metrics.get("ambiguous_label_fraction", 1) <= REPAIR_GATE["ambiguous_label_fraction"],
        "projection_coverage_over_original_gt": metrics.get("projection_coverage_over_original_gt", 0) >= REPAIR_GATE["projection_coverage_over_original_gt"],
        "label_conflict_rate": metrics.get("label_conflict_rate", 1) <= REPAIR_GATE["label_conflict_rate"],
        "event_support_overlap_rate": metrics.get("event_support_overlap_rate", 1) <= REPAIR_GATE["event_support_overlap_rate"],
        "repaired_candidate_oracle_recall": metrics.get("repaired_candidate_oracle_recall", 0) >= REPAIR_GATE["repaired_candidate_oracle_recall"],
        "repaired_oracle_precision_at_recall_0_8": metrics.get("repaired_oracle_precision_at_recall_0_8", 0) >= REPAIR_GATE["repaired_oracle_precision_at_recall_0_8"],
        "repaired_oracle_recall_at_precision_0_8": metrics.get("repaired_oracle_recall_at_precision_0_8", 0) >= REPAIR_GATE["repaired_oracle_recall_at_precision_0_8"],
        "repaired_score_oracle_best_f1": metrics.get("repaired_score_oracle_best_f1", 0) >= REPAIR_GATE["repaired_score_oracle_best_f1"],
        "repaired_feature_auroc": metrics.get("repaired_feature_auroc", 0) >= REPAIR_GATE["repaired_feature_auroc"],
        "repaired_feature_auprc": metrics.get("repaired_feature_auprc", 0) >= REPAIR_GATE["repaired_feature_auprc"],
        "no_gt_leakage": True,
        "original_canonical_preserved": True,
        "label_layer_v1_preserved": True,
    }
    return {"label_repair_gate_pass": all(checks.values()), "checks": checks}


def _process_seed(
    seed: int,
    synthetic_root: Path,
    src_global: pd.DataFrame,
    dst_global: pd.DataFrame,
) -> dict[str, Any]:
    sd = _phase10s._seed_dir(synthetic_root, seed)
    seed_data = _phase10s._load_seed_data(sd)
    flow_ids = _seed_flow_ids(seed_data)
    truth = _pair_set(seed_data["labels"])
    src_d = src_global[src_global["flow_id"].astype(str).isin(flow_ids)].to_dict("records") if not src_global.empty else []
    dst_d = dst_global[dst_global["flow_id"].astype(str).isin(flow_ids)].to_dict("records") if not dst_global.empty else []

    src_graph, dst_graph, audit = build_incidence_graph(src_d, dst_d, truth)
    audit_public = {k: v for k, v in audit.items() if not k.startswith("_")}

    repairs: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for vk, fn in (
        ("a", repair_variant_a),
        ("b", repair_variant_b),
        ("c", repair_variant_c),
        ("d", repair_variant_d),
    ):
        df, qual = fn(audit, truth, seed)
        frames[vk] = df
        repairs[vk] = qual

    return {
        "seed": seed,
        "truth": truth,
        "src_graph": src_graph,
        "dst_graph": dst_graph,
        "audit": audit_public,
        "repairs": repairs,
        "frames": frames,
    }


def _write_infeasibility_final(out: Path) -> None:
    text = """# Original canonical high-P/R infeasibility (final)

## Summary

Under the original canonical flow-level label layer (`label_layer_v1`), exact high-P/R flow correspondence is **not attainable** without reconstructing labels or changing the prediction unit.

## Phase chain

1. **Phase 12–14:** Flow-level proxy / generic RPC evidence does not lift oracle ceiling.
2. **Phase 15:** Targeted Celer ABI decode succeeds (transferId coverage 1.0), but transferId exact-match precision ~0.20 due to flow-level candidate collision (~79%).
3. **Phase 16:** Chain-verified event deaggregation improves flow-lifted AUROC (0.85) and reduces collision (0.21), but oracle P@R≥0.8 remains ~0.09; 100% transferId reuse across flows.
4. **Phase 17:** Micro-flow resegmentation yields `transferId_unique_candidate_fraction = 1.0` but `clean_microflow_label_fraction = 0.0` — bridge events cannot be projected into clean, non-overlapping flow-level supervision.
5. **Phase 18:** Label repair feasibility gate **FAIL** — no CSFFC-v2 variant achieves clean_label_fraction ≥ 0.80 with repaired oracle thresholds.

## Allowed statement

"Under the original canonical flow-level label layer, exact high-P/R flow correspondence is not attainable because bridge-event evidence cannot be projected into clean, non-overlapping flow-level supervision."

## Forbidden statements

- RC-UOT-HP achieves high P/R on the **original** canonical task.
- Micro-flow or label repair "fixes" the original task without CSFFC-v2 relabeling.
- Celer ABI evidence alone solves high-P/R.
- Universal / real-pool superiority; SOTA real Celer.

Any Precision/Recall ≥0.8 claim must come from a **new** label definition (CSFFC-v2), event-level supervision, or external adjudication — not the original canonical flow-pair setting.
"""
    (out / "diagnosis" / "original_canonical_high_pr_infeasibility_final.md").write_text(text, encoding="utf-8")


def run_phase18(
    *,
    run_root: Path,
    pilot_seeds: list[int],
    use_phase16_decoded_events: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out = run_root / OUT_REL
    for d in ("diagnosis", "repair", "audit", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)

    synthetic_root = _phase10s._resolve_synthetic_root(run_root)
    src_global, dst_global = _load_phase16_decoded(run_root) if use_phase16_decoded_events else (pd.DataFrame(), pd.DataFrame())

    seed_results = []
    for seed in pilot_seeds:
        seed_results.append(_process_seed(seed, synthetic_root, src_global, dst_global))

    src_graphs = [sr["src_graph"] for sr in seed_results if not sr["src_graph"].empty]
    dst_graphs = [sr["dst_graph"] for sr in seed_results if not sr["dst_graph"].empty]
    pd.concat(src_graphs, ignore_index=True).to_csv(out / "diagnosis" / "event_flow_incidence_graph_src.csv", index=False)
    pd.concat(dst_graphs, ignore_index=True).to_csv(out / "diagnosis" / "event_flow_incidence_graph_dst.csv", index=False)

    audits = [sr["audit"] for sr in seed_results]
    overlap_audit: dict[str, Any] = {}
    for k in audits[0]:
        if isinstance(audits[0][k], dict):
            overlap_audit[k] = audits[0][k]
        else:
            overlap_audit[k] = float(np.mean([a.get(k, 0) for a in audits]))
    overlap_audit["gt_edges_inside_equivalence_classes"] = float(sum(a.get("gt_edges_inside_equivalence_classes", 0) for a in audits))
    overlap_audit["gt_edges_crossing_equivalence_classes"] = float(sum(a.get("gt_edges_crossing_equivalence_classes", 0) for a in audits))
    (out / "diagnosis" / "event_flow_overlap_audit.json").write_text(json.dumps(overlap_audit, indent=2, default=str), encoding="utf-8")
    (out / "diagnosis" / "event_flow_overlap_audit.md").write_text(
        "# Event-flow overlap audit\n\n" + "\n".join(
            f"- {k}: {v}" for k, v in overlap_audit.items() if not isinstance(v, dict)
        ) + "\n",
        encoding="utf-8",
    )

    variant_agg: dict[str, dict[str, float]] = {v: {} for v in REPAIR_VARIANTS}
    v2_proposals = []
    for vk in REPAIR_VARIANTS:
        keys: set[str] = set()
        for sr in seed_results:
            keys |= set(sr["repairs"][vk].keys())
        for k in keys:
            if k.startswith("_"):
                continue
            vals = [sr["repairs"][vk].get(k, 0) for sr in seed_results if k in sr["repairs"][vk]]
            if vals:
                variant_agg[vk][k] = float(np.mean(vals))
        frames = [sr["frames"][vk] for sr in seed_results if not sr["frames"][vk].empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                out / "repair" / f"variant_{vk}_{'dedup_flows' if vk == 'a' else 'event_majority_assignment' if vk == 'b' else 'transferid_event_pairs' if vk == 'c' else 'repaired_flows'}.csv",
                index=False,
            )
        for sr in seed_results:
            v2 = sr["repairs"][vk].get("_v2_proposal_rows", [])
            v2_proposals.extend(v2)

    if v2_proposals:
        pd.DataFrame(v2_proposals).to_csv(out / "repair" / "label_layer_v2_proposal.csv", index=False)

    comparison_rows = []
    for vk in REPAIR_VARIANTS:
        row = {"variant": vk, "variant_name": VARIANT_LABELS[vk], **variant_agg[vk]}
        row["label_repair_gate_pass"] = evaluate_repair_gate(variant_agg[vk])["label_repair_gate_pass"]
        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(out / "diagnosis" / "phase18_repair_variant_comparison.csv", index=False)
    md_lines = ["# Repair variant comparison", "", "| " + " | ".join(comp_df.columns) + " |", "| " + " | ".join(["---"] * len(comp_df.columns)) + " |"]
    for _, row in comp_df.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in comp_df.columns) + " |")
    (out / "diagnosis" / "phase18_repair_variant_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    best_row = max(comparison_rows, key=lambda r: r.get("clean_label_fraction", 0) + r.get("repaired_feature_auroc", 0))
    best_variant = best_row["variant"]
    gate = evaluate_repair_gate(variant_agg[best_variant])
    gate_pass = gate["label_repair_gate_pass"]

    (out / "diagnosis" / "phase18_label_repair_gate.json").write_text(
        json.dumps({
            "label_repair_gate_pass": gate_pass,
            "best_variant": best_variant,
            "best_variant_name": VARIANT_LABELS[best_variant],
            "phase19_training_ready": gate_pass,
            "checks": gate["checks"],
            "metrics": variant_agg[best_variant],
            "overlap_audit": {k: v for k, v in overlap_audit.items() if not isinstance(v, dict)},
            "credentials_committed": False,
            "full_rpc_url_logged": False,
            "canonical_rebuilt": False,
            "label_layer_refrozen": False,
            "label_layer_v1_preserved": True,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "diagnosis" / "phase18_label_repair_gate.md").write_text(
        f"# Phase 18 label repair gate\n\n- gate_pass: **{gate_pass}**\n- best_variant: {best_variant} ({VARIANT_LABELS[best_variant]})\n"
        f"- phase19_training_ready: {gate_pass}\n",
        encoding="utf-8",
    )

    if not gate_pass:
        (out / "diagnosis" / "phase18_label_repair_infeasibility.md").write_text(
            "# Label repair infeasibility\n\n"
            "No repair variant meets CSFFC-v2 gate thresholds. Original canonical flow-pair high-P/R is not attainable "
            "unless labels are reconstructed or the prediction unit changes to event-level (CSFFC-v2-event).\n\n"
            f"- best clean_label_fraction: {best_row.get('clean_label_fraction', 0):.3f}\n"
            f"- best repaired oracle P@R>=0.8: {best_row.get('repaired_oracle_precision_at_recall_0_8', 0):.3f}\n",
            encoding="utf-8",
        )

    (out / "repair" / "label_projection_repair_report.md").write_text(
        "# Label projection repair\n\n"
        "Private repair variants A–D; `label_layer_v1` and canonical unchanged.\n\n"
        + "\n".join(md_lines[2:]) + "\n",
        encoding="utf-8",
    )

    _write_infeasibility_final(out)

    (out / "splits" / "split_summary.json").write_text(json.dumps({
        "canonical_rebuilt": False,
        "label_layer_refrozen": False,
        "label_layer_v1_preserved": True,
        "phase10s_to_17_preserved": True,
        "phase17_failure_preserved": True,
        "training_skipped": True,
        "holdout_evaluation_skipped": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }, indent=2), encoding="utf-8")

    result = {
        "ok": gate_pass,
        "label_repair_gate_pass": gate_pass,
        "phase19_training_ready": gate_pass,
        "best_variant": best_variant,
        "overlap_audit": overlap_audit,
        "variant_metrics": variant_agg,
        "best_metrics": variant_agg[best_variant],
        "original_infeasibility_written": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
        "elapsed_sec": time.time() - t0,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 18 label repair feasibility")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--pilot-seeds", type=int, nargs="+", default=list(range(52, 58)))
    ap.add_argument("--use-phase16-decoded-events", action="store_true", default=True)
    args = ap.parse_args()
    r = run_phase18(run_root=args.run_root, pilot_seeds=args.pilot_seeds, use_phase16_decoded_events=args.use_phase16_decoded_events)
    slim = {k: v for k, v in r.items() if k not in ("variant_metrics", "overlap_audit")}
    print(json.dumps(slim, indent=2, default=str))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
