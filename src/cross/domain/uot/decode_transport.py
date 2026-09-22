"""Decode UOT transport matrix into human-readable correspondences and tx-level pairs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow as _pick_dst_tx_in_flow_impl
from cross.shared.normalize import norm_addr
from cross.shared.transfers import parse_transfer_value
from cross.utils.safe_cast import safe_float


def decode_correspondence(
    p: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    threshold: float = 0.01,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    p = np.asarray(p, dtype=float)
    for i, s in enumerate(source_flows):
        row_sum = float(p[i].sum()) if p.shape[1] else 0.0
        for j, t in enumerate(target_flows):
            mass = float(p[i, j])
            if mass >= threshold:
                results.append(
                    {
                        "source_flow": s.get("flow_id"),
                        "target_flow": t.get("flow_id"),
                        "transport_mass": mass,
                        "source_share": mass / (row_sum + 1e-12),
                        "source_amount_usd": s.get("amount_usd"),
                        "target_amount_usd": t.get("amount_usd"),
                        "source_chain": s.get("chain"),
                        "target_chain": t.get("chain"),
                    }
                )
    return results


def compute_unmatched_source_mass(p: np.ndarray, source_mass: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    source_mass = np.asarray(source_mass, dtype=float).ravel()
    row_mass = p.sum(axis=1)
    return source_mass - row_mass


def compute_unmatched_target_mass(p: np.ndarray, target_mass: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    target_mass = np.asarray(target_mass, dtype=float).ravel()
    col_mass = p.sum(axis=0)
    return target_mass - col_mass


def row_entropies(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalized row entropy (bits), per source flow."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(p.shape[0], dtype=float)
    for i in range(p.shape[0]):
        row = p[i] + eps
        row = row / (row.sum() + eps)
        out[i] = float(-np.sum(row * np.log(row + eps)))
    return out


def _pick_dst_tx_in_flow(
    src_tx: str,
    src_ts: float,
    src_human_amt: float,
    dst_flow: dict[str, Any],
    dst_norm: pd.DataFrame,
) -> str:
    """Pick best BNB tx hash inside dst_flow for this src tx (legacy policy)."""
    dst, _, _ = _pick_dst_tx_in_flow_impl(src_tx, src_ts, src_human_amt, dst_flow, dst_norm, policy="legacy")
    return dst


def _tx_to_flow_index(source_flows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            out[norm_addr(str(txh))] = i
    return out


def derive_top1_tx_pairs(
    p: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Row-argmax over flows, then pick concrete dst tx per src tx. Returns ``(mapping, per_src_meta)``."""
    p = np.asarray(p, dtype=float)
    mapping: dict[str, str] = {}
    meta: dict[str, dict[str, Any]] = {}

    tx_to_i = _tx_to_flow_index(source_flows)
    ents = row_entropies(p) if p.size else np.array([])

    for _, r in src_all.iterrows():
        txh = norm_addr(r.get("txhash", ""))
        if not txh:
            continue
        s_ts = safe_float(r.get("timestamp"), 0.0)
        s_amt = safe_float(r.get("args.amount"), 0.0)

        if not source_flows or not target_flows or p.size == 0:
            mapping[txh] = ""
            meta[txh] = {
                "flow_i": -1,
                "flow_j": -1,
                "transport_mass": 0.0,
                "source_share": 0.0,
                "target_share": 0.0,
                "row_entropy": 0.0,
            }
            continue

        i = int(tx_to_i.get(txh, -1))
        if i < 0 or i >= p.shape[0]:
            mapping[txh] = ""
            meta[txh] = {
                "flow_i": -1,
                "flow_j": -1,
                "transport_mass": 0.0,
                "source_share": 0.0,
                "target_share": 0.0,
                "row_entropy": 0.0,
            }
            continue

        row = p[i]
        j = int(np.argmax(row)) if row.size else -1
        mass = float(row[j]) if j >= 0 and j < row.size else 0.0
        row_sum = float(row.sum()) + 1e-12
        share = mass / row_sum
        col_sum = float(p[:, j].sum()) + 1e-12 if j >= 0 and j < p.shape[1] else 1e-12
        tgt_share = mass / col_sum
        dst_f = target_flows[j] if 0 <= j < len(target_flows) else {}
        dst_pick = _pick_dst_tx_in_flow(txh, s_ts, s_amt, dst_f, dst_norm)
        mapping[txh] = dst_pick
        meta[txh] = {
            "flow_i": i,
            "flow_j": j,
            "transport_mass": mass,
            "source_share": share,
            "target_share": tgt_share,
            "row_entropy": float(ents[i]) if i < len(ents) else 0.0,
            "source_flow_id": source_flows[i].get("flow_id") if i < len(source_flows) else "",
            "target_flow_id": dst_f.get("flow_id") if dst_f else "",
        }
    return mapping, meta


def enrich_decoded_rows_for_csv(
    decoded: list[dict[str, Any]],
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add ``delay_sec`` and ``relation_type`` for paper-style CSV export."""
    from collections import Counter

    fid_s = {str(f.get("flow_id")): f for f in eth_flows}
    fid_t = {str(f.get("flow_id")): f for f in bnb_flows}
    sc = Counter(str(d.get("source_flow")) for d in decoded if d.get("source_flow"))
    tc = Counter(str(d.get("target_flow")) for d in decoded if d.get("target_flow"))
    out: list[dict[str, Any]] = []
    for d in decoded:
        row = dict(d)
        sfid = str(d.get("source_flow") or "")
        tfid = str(d.get("target_flow") or "")
        s = fid_s.get(sfid) or {}
        t = fid_t.get(tfid) or {}
        if s and t:
            row["delay_sec"] = float(t.get("start_time", 0.0)) - float(s.get("end_time", 0.0))
        else:
            row["delay_sec"] = float("nan")
        if sc.get(sfid, 0) > 1 and tc.get(tfid, 0) > 1:
            row["relation_type"] = "many_to_many"
        elif sc.get(sfid, 0) > 1:
            row["relation_type"] = "one_to_many"
        elif tc.get(tfid, 0) > 1:
            row["relation_type"] = "many_to_one"
        else:
            row["relation_type"] = "one_to_one"
        out.append(row)
    return out


def build_transport_plan_export_rows(
    p: np.ndarray,
    c: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    *,
    mass_threshold: float = 1e-9,
    weak_share_threshold: float = 0.12,
    cost_decomp: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per positive-mass cell: transport mass, shares, decomposed costs, ranks, causal flags."""
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    if p.size == 0:
        return []
    active_r = (p > mass_threshold).sum(axis=1)
    active_c = (p > mass_threshold).sum(axis=0)

    row_argmax = {int(i): int(np.argmax(p[i])) for i in range(p.shape[0])}

    def _decomp_cell(name: str, i: int, j: int) -> float:
        if not cost_decomp:
            return 0.0
        arr = cost_decomp.get(name)
        if not isinstance(arr, np.ndarray) or arr.shape != p.shape:
            return 0.0
        return float(arr[i, j])

    def _rank_in_row(i: int, j: int) -> int:
        order = np.argsort(-p[i])
        return int(np.where(order == j)[0][0]) + 1

    def _rank_in_col(i: int, j: int) -> int:
        order = np.argsort(-p[:, j])
        return int(np.where(order == i)[0][0]) + 1

    rows: list[dict[str, Any]] = []
    for i in range(p.shape[0]):
        row_sum = float(p[i].sum()) + 1e-12
        for j in range(p.shape[1]):
            mass = float(p[i, j])
            if mass <= mass_threshold:
                continue
            col_sum = float(p[:, j].sum()) + 1e-12
            src_share = mass / row_sum
            tgt_share = mass / col_sum
            ar = int(active_r[i])
            ac = int(active_c[j])
            if ar <= 1 and ac <= 1:
                struct = "one_to_one"
            elif ar > 1 and ac <= 1:
                struct = "one_to_many"
            elif ar <= 1 and ac > 1:
                struct = "many_to_one"
            else:
                struct = "many_to_many"
            if src_share < weak_share_threshold or tgt_share < weak_share_threshold:
                mtype = "weak_match"
            else:
                mtype = struct
            sid = str(source_flows[i].get("flow_id", "") if i < len(source_flows) else i)
            tid = str(target_flows[j].get("flow_id", "") if j < len(target_flows) else j)
            tot = float(c[i, j]) if c.shape == p.shape else float("nan")
            feas = cost_decomp.get("feasible_flag") if cost_decomp else None
            is_causal = True
            if isinstance(feas, np.ndarray) and feas.shape == p.shape:
                is_causal = bool(feas[i, j])
            is_pred = bool(j == row_argmax.get(i, -1))
            rows.append(
                {
                    "src_flow_id": sid,
                    "dst_flow_id": tid,
                    "transport_mass": mass,
                    "source_share": src_share,
                    "target_share": tgt_share,
                    "total_cost": tot,
                    "cost_total": tot,
                    "cost_amount": _decomp_cell("amount_cost", i, j),
                    "cost_time": _decomp_cell("time_cost", i, j),
                    "cost_route": _decomp_cell("route_cost", i, j),
                    "cost_risk": _decomp_cell("risk_cost", i, j),
                    "cost_graph": _decomp_cell("graph_cost", i, j),
                    "cost_evidence": _decomp_cell("evidence_cost", i, j),
                    "rank_for_source": _rank_in_row(i, j),
                    "rank_for_target": _rank_in_col(i, j),
                    "is_causal_valid": is_causal,
                    "is_predicted_match": is_pred,
                    "match_type": mtype,
                }
            )
    return rows


def build_uot_summary_dict(
    p: np.ndarray,
    *,
    uot_solver_meta: dict[str, Any] | None,
    path_b_opts: dict[str, Any],
    split_merge: dict[str, Any],
    transport_rows: list[dict[str, Any]],
    unmatched_source_mass: list[float] | None = None,
    unmatched_target_mass: list[float] | None = None,
) -> dict[str, Any]:
    p = np.asarray(p, dtype=float)
    sm = uot_solver_meta if isinstance(uot_solver_meta, dict) else {}
    weak_n = sum(1 for r in transport_rows if str(r.get("match_type", "")) == "weak_match")
    a = np.asarray(sm.get("source_mass_risk_weighted") or sm.get("source_mass_original") or [], dtype=float)
    b = np.asarray(sm.get("target_mass_evidence_weighted") or sm.get("target_mass_original") or [], dtype=float)
    tot_src = float(a.sum()) if a.size else float(p.shape[0] or 0)
    tot_tgt = float(b.sum()) if b.size else float(p.shape[1] or 0)
    transported = float(p.sum()) if p.size else 0.0
    um_s = float(sum(float(x) for x in (unmatched_source_mass or [])))
    um_t = float(sum(float(x) for x in (unmatched_target_mass or [])))
    return {
        "num_source_flows": int(p.shape[0]) if p.size else 0,
        "num_target_flows": int(p.shape[1]) if p.size else 0,
        "total_source_mass": tot_src,
        "total_target_mass": tot_tgt,
        "transported_mass": transported,
        "unmatched_source_mass": um_s,
        "unmatched_target_mass": um_t,
        "split_count": int(split_merge.get("sources_with_multiple_targets", 0)),
        "merge_count": int(split_merge.get("targets_with_multiple_sources", 0)),
        "weak_match_count": int(weak_n),
        "solver_backend": str(sm.get("actual_backend") or path_b_opts.get("uot_backend") or ""),
        "reg": path_b_opts.get("uot_reg"),
        "reg_m": path_b_opts.get("uot_reg_m"),
        "lambda_risk": sm.get("lambda_risk"),
    }


def compute_split_merge_summary(p: np.ndarray, *, edge_threshold: float = 1e-9) -> dict[str, Any]:
    """Counts active transport edges per source/target flow (for split/merge summary JSON)."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return {
            "n_source_flows": 0,
            "n_target_flows": 0,
            "sources_with_multiple_targets": 0,
            "targets_with_multiple_sources": 0,
        }
    active_r = (p > edge_threshold).sum(axis=1)
    active_c = (p > edge_threshold).sum(axis=0)
    return {
        "n_source_flows": int(p.shape[0]),
        "n_target_flows": int(p.shape[1]),
        "sources_with_multiple_targets": int((active_r > 1).sum()),
        "targets_with_multiple_sources": int((active_c > 1).sum()),
        "sources_with_single_target": int((active_r == 1).sum()),
        "targets_with_single_source": int((active_c == 1).sum()),
        "isolated_sources": int((active_r == 0).sum()),
        "isolated_targets": int((active_c == 0).sum()),
    }
