"""Tractable BNB column pool for standalone flow UOT (no arbitrary head truncation of flows)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _addr_overlap_bonus(eth: dict[str, Any], bnb: dict[str, Any]) -> float:
    sa = set(eth.get("address_set") or [])
    sb = set(bnb.get("address_set") or [])
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return float(inter / max(uni, 1))


def _heuristic_cost(eth: dict[str, Any], bnb: dict[str, Any], *, max_delay_sec: float) -> float:
    """Lower is better; amount / time / route / address (amount is ranking only, not a hard gate)."""
    s_usd = max(_f(eth.get("amount_usd")), 0.0)
    t_usd = max(_f(bnb.get("amount_usd")), 0.0)
    amount_err = abs(s_usd - t_usd) / max(s_usd, t_usd, 1e-12)
    amount_err = min(amount_err, 1.0)

    se = _f(eth.get("end_time"))
    ds = _f(bnb.get("start_time"))
    delay = ds - se
    max_d = max(float(max_delay_sec), 1.0)
    if delay < 0:
        time_c = 1.0 + min(abs(delay) / max_d, 1.0)
    else:
        time_c = min(delay / max_d, 1.0)

    ag_s = str(eth.get("asset_group") or "")
    ag_t = str(bnb.get("asset_group") or "")
    route_c = 0.0 if ag_s and ag_t and ag_s == ag_t else 0.35
    rid_s = str(eth.get("route_id") or "")
    rid_t = str(bnb.get("route_id") or "")
    if rid_s and rid_t and rid_s == rid_t:
        route_c = min(route_c, 0.08)

    sym_s = str(eth.get("token_symbol") or "").upper()
    sym_t = str(bnb.get("token_symbol") or "").upper()
    if sym_s and sym_t and sym_s == sym_t:
        route_c = min(route_c, 0.1)

    base = 0.45 * amount_err + 0.35 * time_c + 0.20 * route_c
    return max(0.0, base - 0.22 * _addr_overlap_bonus(eth, bnb))


def _window_js_widened(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    *,
    max_delay_sec: float,
    max_scan: int,
) -> list[int]:
    """Time window only (soft filter); wider post/pre padding than legacy 120s."""
    se = _f(eth.get("end_time"))
    pad_pre = max(900.0, min(10_800.0, float(max_delay_sec) * 0.12))
    lo = se - pad_pre
    hi = se + float(max_delay_sec) * 1.38
    js: list[int] = []
    for j, t in enumerate(bnb_flows):
        ds = _f(t.get("start_time"))
        if ds < lo or ds > hi:
            continue
        js.append(j)
    if len(js) > max_scan:
        ae = _f(eth.get("amount_usd"))
        js = sorted(js, key=lambda j: abs(_f(bnb_flows[j].get("amount_usd")) - ae))[:max_scan]
    return js


def _route_asset_candidate_indices(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    *,
    by_ag: dict[str, list[int]],
    by_rid: dict[str, list[int]],
    cap: int,
) -> set[int]:
    """Strong recall: all BNB flows sharing asset_group or route_id with ETH (amount only caps size)."""
    ag = str(eth.get("asset_group") or "").strip()
    rid = str(eth.get("route_id") or "").strip()
    u: set[int] = set()
    if ag:
        u.update(by_ag.get(ag, ()))
    if rid:
        u.update(by_rid.get(rid, ()))
    if len(u) <= cap:
        return u
    ae = _f(eth.get("amount_usd"))
    ranked = sorted(u, key=lambda j: abs(_f(bnb_flows[j].get("amount_usd")) - ae))[:cap]
    return set(ranked)


def _bnb_indices_by_route_and_asset(bnb_flows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    by_ag: dict[str, list[int]] = defaultdict(list)
    by_rid: dict[str, list[int]] = defaultdict(list)
    for j, t in enumerate(bnb_flows):
        ag = str(t.get("asset_group") or "").strip()
        if ag:
            by_ag[ag].append(j)
        rid = str(t.get("route_id") or "").strip()
        if rid:
            by_rid[rid].append(j)
    return by_ag, by_rid


def _k_for_eth(eth: dict[str, Any], top_k_default: int, m: int, adaptive_top_k_by_asset_group: dict[str, int] | None) -> int:
    k0 = max(1, min(int(top_k_default), m))
    if not adaptive_top_k_by_asset_group:
        return k0
    ag = str(eth.get("asset_group") or "").strip()
    return min(max(k0, int(adaptive_top_k_by_asset_group.get(ag, k0))), m)


def _route_preserving_indices(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    *,
    route_preserving_m: int,
) -> list[int]:
    if route_preserving_m <= 0:
        return []
    rid = str(eth.get("route_id") or "").strip()
    if not rid:
        return []
    se = _f(eth.get("end_time"))
    pairs: list[tuple[float, int]] = []
    for j, bf in enumerate(bnb_flows):
        if str(bf.get("route_id") or "").strip() != rid:
            continue
        ds = _f(bf.get("start_time"))
        pairs.append((abs(ds - se), j))
    pairs.sort(key=lambda x: x[0])
    return [j for _, j in pairs[:route_preserving_m]]


def _in_widened_time_window(eth: dict[str, Any], bnb: dict[str, Any], *, max_delay_sec: float) -> bool:
    se = _f(eth.get("end_time"))
    pad_pre = max(900.0, min(10_800.0, float(max_delay_sec) * 0.12))
    lo = se - pad_pre
    hi = se + float(max_delay_sec) * 1.38
    ds = _f(bnb.get("start_time"))
    return lo <= ds <= hi


def _sole_anchor(per_i: list[list[int]], u_set: set[int], j: int) -> bool:
    for ranked in per_i:
        ins = [x for x in ranked if x in u_set]
        if len(ins) == 1 and ins[0] == j:
            return True
    return False


def _initial_global_dst_pool(
    dst_score: dict[int, float],
    per_i: list[list[int]],
    bnb_flows: list[dict[str, Any]],
    max_u: int,
    *,
    pool_strategy: str,
    min_pool_per_asset: int,
) -> set[int]:
    ps = str(pool_strategy or "default").strip().lower()
    if ps != "stratified_global_pool" or min_pool_per_asset <= 0:
        u_list = sorted(dst_score.keys(), key=lambda j: -dst_score[j])[:max_u]
        return set(u_list)
    cand_union: set[int] = set()
    for ranked in per_i:
        cand_union.update(ranked)
    by_ag: dict[str, list[int]] = defaultdict(list)
    for j in cand_union:
        ag = str(bnb_flows[j].get("asset_group") or "").strip() or "_empty_"
        by_ag[ag].append(j)
    n_g = max(len(by_ag), 1)
    quota = min(int(min_pool_per_asset), max(1, max_u // n_g))
    u: set[int] = set()
    for ag, js in sorted(by_ag.items(), key=lambda kv: -max((dst_score.get(x, 0.0) for x in kv[1]), default=0.0)):
        js_sorted = sorted(js, key=lambda j: -dst_score.get(j, 0.0))
        for j in js_sorted[: min(quota, len(js_sorted))]:
            if len(u) >= max_u:
                break
            u.add(j)
        if len(u) >= max_u:
            break
    if len(u) < max_u:
        rest = [j for j in cand_union if j not in u]
        rest.sort(key=lambda j: -dst_score.get(j, 0.0))
        for j in rest:
            if len(u) >= max_u:
                break
            u.add(j)
    return u


def select_bnb_subgraph_for_flow_uot(
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    *,
    top_k_per_src: int,
    max_delay_sec: float,
    max_matrix_cells: int,
    pool_strategy: str = "default",
    min_pool_per_asset: int = 0,
    route_preserving_m: int = 0,
    adaptive_top_k_by_asset_group: dict[str, int] | None = None,
    oracle_dst_flow_indices: Iterable[int] | None = None,
    oracle_diagnostic_exceed_budget: bool = False,
    pool_debug: dict[str, Any] | None = None,
    oracle_return_mid_u: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(bnb_subflows, meta)`` using either full BNB columns or a capped global pool.

    When ``n * m`` exceeds ``max_matrix_cells``, we keep **all** ETH rows and choose a global
    set ``U`` of at most ``floor(max_matrix_cells / n)`` BNB column indices. Per-ETH candidates
    are the union of (widened) time-window hits and **all** same-``asset_group`` / same-``route_id``
    BNB flows (hard recall); candidates are ranked by heuristic (amount proximity is **not** a hard filter).

    ``pool_strategy``:
      - ``default``: legacy global pool ranked by ETH-amount-weighted per-src ranks.
      - ``stratified_global_pool``: reserve ``min_pool_per_asset`` dst per ``asset_group`` (quota-capped)
        before filling the rest by global ``dst_score``.

    ``oracle_dst_flow_indices`` (diagnostic): force these BNB column indices into ``U``. They are
    **never** removed by budget trimming. If ``oracle_diagnostic_exceed_budget`` is true, the final
    pool may exceed ``max_matrix_cells`` so that all protected indices remain (diagnostic upper bound
    only; not for fair benchmarking).

    ``oracle_return_mid_u``: when true, ``meta["oracle_u_after_while"]`` lists column indices after
    the first budget-while loop (for audit CSVs).
    """
    n, m = len(eth_flows), len(bnb_flows)
    if n == 0 or m == 0:
        return [], {"mode": "empty", "n_eth": n, "n_bnb_original": m, "n_bnb_active": 0}

    budget = max(1, int(max_matrix_cells))
    if n * m <= budget:
        return list(bnb_flows), {
            "mode": "full_matrix",
            "n_eth": n,
            "n_bnb_original": m,
            "n_bnb_active": m,
            "matrix_cells": n * m,
            "top_k_per_src": int(top_k_per_src),
            "max_matrix_cells": budget,
            "pool_strategy": str(pool_strategy or "default"),
        }

    max_u = max(1, min(m, budget // max(n, 1)))
    max_scan = min(m, 8000)
    hard_cap = 4000
    universe_cap = 9000

    by_ag, by_rid = _bnb_indices_by_route_and_asset(bnb_flows)

    per_i: list[list[int]] = []
    per_src_k_req: list[int] = []
    for eth in eth_flows:
        k_req_i = _k_for_eth(eth, top_k_per_src, m, adaptive_top_k_by_asset_group)
        per_src_k_req.append(int(k_req_i))
        js_time = _window_js_widened(eth, bnb_flows, max_delay_sec=max_delay_sec, max_scan=max_scan)
        hard = _route_asset_candidate_indices(eth, bnb_flows, by_ag=by_ag, by_rid=by_rid, cap=hard_cap)
        cand = set(js_time) | set(hard)
        for j in _route_preserving_indices(eth, bnb_flows, route_preserving_m=route_preserving_m):
            cand.add(j)
        if not cand:
            cand = set(range(min(m, max_scan)))
        cand_list = list(cand)
        if len(cand_list) > universe_cap:
            cand_list = sorted(
                cand_list,
                key=lambda j: _heuristic_cost(eth, bnb_flows[j], max_delay_sec=max_delay_sec),
            )[:universe_cap]
        ranked = sorted(
            cand_list,
            key=lambda j: _heuristic_cost(eth, bnb_flows[j], max_delay_sec=max_delay_sec),
        )[: min(k_req_i, len(cand_list))]
        per_i.append(ranked)

    dst_score: dict[int, float] = defaultdict(float)
    for i, ranked in enumerate(per_i):
        w_i = max(_f(eth_flows[i].get("amount_usd")), 1e-12)
        for rnk, j in enumerate(ranked):
            dst_score[j] += w_i / (1.0 + float(rnk))

    protected: set[int] = set(int(x) for x in (oracle_dst_flow_indices or ()) if 0 <= int(x) < m)

    u_set = _initial_global_dst_pool(
        dst_score,
        per_i,
        bnb_flows,
        max_u,
        pool_strategy=str(pool_strategy or "default"),
        min_pool_per_asset=int(min_pool_per_asset or 0),
    )
    u_set |= protected

    for i, ranked in enumerate(per_i):
        if any(j in u_set for j in ranked):
            continue
        j0 = ranked[0]
        u_set.add(j0)
        dst_score[j0] += max(_f(eth_flows[i].get("amount_usd")), 1e-12)

    while len(u_set) > max_u:
        removable = [j for j in u_set if j not in protected and not _sole_anchor(per_i, u_set, j)]
        if not removable:
            break
        j_drop = min(removable, key=lambda j: dst_score.get(j, 0.0))
        u_set.discard(j_drop)

    u_after_while = set(u_set)
    active_sorted = sorted(u_set)
    oracle_cells_exceed_budget = False
    if oracle_diagnostic_exceed_budget and protected:
        active = active_sorted
        oracle_cells_exceed_budget = bool(n * len(active) > budget)
    elif len(active_sorted) > max_u:
        keep: set[int] = set(protected) & u_set
        rest = sorted(u_set - keep, key=lambda j: -dst_score.get(j, 0.0))
        for j in rest:
            if len(keep) >= max_u:
                break
            keep.add(j)
        active = sorted(keep)
    else:
        active = active_sorted
    u_set = set(active)

    def _has_hit(ranked: list[int]) -> bool:
        return any(j in u_set for j in ranked)

    for i, ranked in enumerate(per_i):
        if _has_hit(ranked):
            continue
        j0 = ranked[0]
        if j0 in u_set:
            continue
        u_set.add(j0)
        while len(u_set) > max_u:
            removable = [j for j in u_set if j not in protected and j != j0 and not _sole_anchor(per_i, u_set, j)]
            if not removable:
                break
            u_set.discard(min(removable, key=lambda j: dst_score.get(j, 0.0)))
    active = sorted(u_set)

    covered = 0
    total = 0
    for ranked in per_i:
        total += 1
        if any(j in active for j in ranked):
            covered += 1
    per_src_in_pool = float(covered / max(total, 1))

    bnb_sub = [bnb_flows[j] for j in active]
    meta: dict[str, Any] = {
        "mode": "global_dst_pool",
        "n_eth": n,
        "n_bnb_original": m,
        "n_bnb_active": len(active),
        "matrix_cells": n * len(active),
        "top_k_per_src": int(top_k_per_src),
        "k_requested": int(max(per_src_k_req) if per_src_k_req else min(int(top_k_per_src), m)),
        "per_src_k_req_max": int(max(per_src_k_req) if per_src_k_req else 0),
        "max_matrix_cells": budget,
        "max_unique_dst": int(max_u),
        "per_src_topk_hits_active_pool": float(per_src_in_pool),
        "active_dst_global_indices": active,
        "pool_strategy": str(pool_strategy or "default"),
        "min_pool_per_asset": int(min_pool_per_asset or 0),
        "route_preserving_m": int(route_preserving_m or 0),
        "oracle_dst_indices_count": int(len(protected)),
        "oracle_diagnostic_exceed_budget": bool(oracle_diagnostic_exceed_budget),
        "oracle_cells_exceed_budget": bool(oracle_cells_exceed_budget),
        "oracle_protected_count": int(len(protected)),
        "oracle_final_active_count": int(len(active)),
        "oracle_recall_upper_bound": 1.0 if (oracle_diagnostic_exceed_budget and protected) else None,
        "candidate_policy_note": (
            "time_window_widened; asset_group+route_id union as hard recall; "
            "amount_proximity_for_sorting_only"
        ),
    }
    if pool_debug is not None:
        pool_debug.clear()
        pool_debug.update(
            {
                "per_src_topk": [list(x) for x in per_i],
                "per_src_k_req": list(per_src_k_req),
                "top_k_per_src_arg": int(top_k_per_src),
                "active_dst_global_indices": list(active),
                "active_index_set": set(active),
                "dst_score": dict(dst_score),
                "max_u": int(max_u),
                "max_delay_sec": float(max_delay_sec),
                "max_scan": int(max_scan),
                "hard_cap": int(hard_cap),
                "universe_cap": int(universe_cap),
                "route_preserving_m": int(route_preserving_m or 0),
                "bnb_flows": bnb_flows,
                "eth_flows": eth_flows,
                "pool_strategy": str(pool_strategy or "default"),
            }
        )
    return bnb_sub, meta


def enrich_transport_graph_meta(meta: dict[str, Any], *, n_eth_full: int, n_bnb_full: int) -> dict[str, Any]:
    """Canonical keys for paper / eval JSON (also written to ``uot_diagnostics.json``)."""
    out = dict(meta)
    mode = str(out.get("mode") or "")
    if mode == "full_matrix":
        out["subgraph_mode"] = "full_matrix"
    elif mode == "global_dst_pool":
        out["subgraph_mode"] = "global_dst_pool"
    else:
        out["subgraph_mode"] = mode or "unknown"
    out["num_src_flows"] = int(out.get("n_eth_full", out.get("n_eth", n_eth_full)))
    out["num_dst_flows_original"] = int(out.get("n_bnb_full", n_bnb_full))
    out["num_dst_flows_selected"] = int(out.get("n_bnb_active", 0))
    out["matrix_cells"] = int(out.get("matrix_cells", out["num_src_flows"] * max(out["num_dst_flows_selected"], 1)))
    return out


def candidate_dst_recall_from_flow_labels(
    flow_labels_csv: Path | None,
    active_dst_flow_ids: set[str],
    *,
    min_label_confidence: float,
    bnb_segment_flow_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Weak-label dst coverage in the active BNB transport pool.

    Returns **unique-dst** and **edge-level** recalls (denominator restricted to dst that exist in
    full BNB segment export when ``bnb_segment_flow_ids`` is provided).

    Keys:
    - ``candidate_dst_recall`` / ``candidate_dst_recall_unique_dst``: unique label dst in pool /
      unique label dst present in segment file (preferred headline metric).
    - ``candidate_dst_recall_edge_level``: label edges whose dst is in pool / edges whose dst is in segments.
    - ``truth_dst_flows``: unique dst in labels (legacy count).
    - ``truth_dst_unique_in_segments``: unique dst in labels that appear in ``bnb_segment_flow_ids``.
    - ``truth_dst_in_subgraph``: unique dst in pool (intersection with labels, same as hit unique).
    """
    if not flow_labels_csv or not flow_labels_csv.is_file() or not active_dst_flow_ids:
        return {
            "truth_dst_flows": 0,
            "truth_dst_unique_in_segments": 0,
            "truth_label_edges": 0,
            "truth_label_edges_in_segments": 0,
            "truth_dst_in_subgraph": 0,
            "truth_edges_hit": 0,
            "candidate_dst_recall": 1.0,
            "candidate_dst_recall_unique_dst": 1.0,
            "candidate_dst_recall_edge_level": 1.0,
            "candidate_dst_recall_legacy_all_labels_denom": 1.0,
        }
    df = pd.read_csv(flow_labels_csv, dtype=str, keep_default_na=False)
    if df.empty or "dst_flow_id" not in df.columns:
        return {
            "truth_dst_flows": 0,
            "truth_dst_unique_in_segments": 0,
            "truth_label_edges": 0,
            "truth_label_edges_in_segments": 0,
            "truth_dst_in_subgraph": 0,
            "truth_edges_hit": 0,
            "candidate_dst_recall": 1.0,
            "candidate_dst_recall_unique_dst": 1.0,
            "candidate_dst_recall_edge_level": 1.0,
            "candidate_dst_recall_legacy_all_labels_denom": 1.0,
        }
    df["_lc"] = pd.to_numeric(df.get("label_confidence"), errors="coerce").fillna(0.0)
    sub = df[df["_lc"] >= float(min_label_confidence)]
    dsts_all = {str(x).strip() for x in sub["dst_flow_id"].tolist() if str(x).strip()}
    seg = bnb_segment_flow_ids
    if seg is not None:
        dsts_seg = {d for d in dsts_all if d in seg}
    else:
        dsts_seg = set(dsts_all)
    truth_n_all = len(dsts_all)
    truth_n_seg = len(dsts_seg)
    hit_unique_seg = sum(1 for d in dsts_seg if d in active_dst_flow_ids)
    hit_unique_all = sum(1 for d in dsts_all if d in active_dst_flow_ids)

    edges_total = int(len(sub))
    edges_in_seg = 0
    edges_hit = 0
    if "dst_flow_id" in sub.columns:
        for _, r in sub.iterrows():
            did = str(r.get("dst_flow_id") or "").strip()
            if not did:
                continue
            in_seg = seg is None or did in seg
            if in_seg:
                edges_in_seg += 1
                if did in active_dst_flow_ids:
                    edges_hit += 1

    rec_u = float(hit_unique_seg / max(truth_n_seg, 1))
    rec_e = float(edges_hit / max(edges_in_seg, 1)) if edges_in_seg else 1.0
    legacy = float(hit_unique_all / max(truth_n_all, 1))
    return {
        "truth_dst_flows": int(truth_n_all),
        "truth_dst_unique_in_segments": int(truth_n_seg),
        "truth_label_edges": int(edges_total),
        "truth_label_edges_in_segments": int(edges_in_seg),
        "truth_dst_in_subgraph": int(hit_unique_seg),
        "truth_edges_hit": int(edges_hit),
        "candidate_dst_recall": rec_u,
        "candidate_dst_recall_unique_dst": rec_u,
        "candidate_dst_recall_edge_level": rec_e,
        "candidate_dst_recall_legacy_all_labels_denom": legacy,
    }


def _build_sorted_candidates_one_eth(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    *,
    by_ag: dict[str, list[int]],
    by_rid: dict[str, list[int]],
    max_delay_sec: float,
    max_scan: int,
    hard_cap: int,
    universe_cap: int,
    k_req: int,
    route_preserving_m: int,
) -> tuple[list[int], list[int]]:
    m = len(bnb_flows)
    js_time = _window_js_widened(eth, bnb_flows, max_delay_sec=max_delay_sec, max_scan=max_scan)
    hard = _route_asset_candidate_indices(eth, bnb_flows, by_ag=by_ag, by_rid=by_rid, cap=hard_cap)
    cand: set[int] = set(js_time) | set(hard)
    for j in _route_preserving_indices(eth, bnb_flows, route_preserving_m=route_preserving_m):
        cand.add(j)
    if not cand:
        cand = set(range(min(m, max_scan)))
    cand_list = list(cand)
    if len(cand_list) > universe_cap:
        cand_list = sorted(
            cand_list,
            key=lambda j: _heuristic_cost(eth, bnb_flows[j], max_delay_sec=max_delay_sec),
        )[:universe_cap]
    ranked_full = sorted(
        cand_list,
        key=lambda j: _heuristic_cost(eth, bnb_flows[j], max_delay_sec=max_delay_sec),
    )
    topk = ranked_full[: min(k_req, len(ranked_full))]
    return ranked_full, topk


def _raw_candidate_membership(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    j: int,
    *,
    by_ag: dict[str, list[int]],
    by_rid: dict[str, list[int]],
    max_delay_sec: float,
    max_scan_loose: int,
    hard_cap_loose: int,
    route_preserving_m: int,
) -> bool:
    js_time = _window_js_widened(eth, bnb_flows, max_delay_sec=max_delay_sec, max_scan=max_scan_loose)
    hard = _route_asset_candidate_indices(eth, bnb_flows, by_ag=by_ag, by_rid=by_rid, cap=hard_cap_loose)
    cand: set[int] = set(js_time) | set(hard)
    for rj in _route_preserving_indices(eth, bnb_flows, route_preserving_m=route_preserving_m):
        cand.add(rj)
    return j in cand


def _classify_miss_reason(
    eth: dict[str, Any],
    bnb_j: dict[str, Any],
    j: int,
    *,
    full_sorted: list[int],
    topk: list[int],
    k_req: int,
    active_index_set: set[int],
    dst_score: dict[int, float],
    max_delay_sec: float,
    by_ag: dict[str, list[int]],
    by_rid: dict[str, list[int]],
    bnb_flows: list[dict[str, Any]],
    max_scan: int,
    hard_cap: int,
    universe_cap: int,
    route_preserving_m: int,
) -> str:
    if j < 0 or j >= len(bnb_flows):
        return "dst_not_found_in_segments"
    in_active = j in active_index_set
    if in_active:
        return "unknown"
    in_full = j in set(full_sorted)
    in_topk = j in set(topk)
    if not in_full:
        m = len(bnb_flows)
        in_raw = _raw_candidate_membership(
            eth,
            bnb_flows,
            j,
            by_ag=by_ag,
            by_rid=by_rid,
            max_delay_sec=max_delay_sec,
            max_scan_loose=min(m, 50_000),
            hard_cap_loose=min(m, 50_000),
            route_preserving_m=route_preserving_m,
        )
        if not in_raw:
            ag_m = str(eth.get("asset_group") or "").strip() == str(bnb_j.get("asset_group") or "").strip()
            rid_m = (
                str(eth.get("route_id") or "").strip()
                and str(eth.get("route_id") or "").strip() == str(bnb_j.get("route_id") or "").strip()
            )
            if _in_widened_time_window(eth, bnb_j, max_delay_sec=max_delay_sec) and (ag_m or rid_m):
                return "time_window_too_narrow"
            if not ag_m and not rid_m:
                return "route_or_asset_group_not_selected"
            return "time_window_too_narrow"
        return "amount_rank_too_low"
    rk = full_sorted.index(j) + 1
    if rk > k_req:
        return "dst_exists_but_rank_beyond_top_k"
    if rk <= min(5, k_req) and _addr_overlap_bonus(eth, bnb_j) <= 0.0 and not in_active:
        return "address_overlap_missing"
    if not in_active:
        return "dst_dropped_by_global_pool_budget"
    return "unknown"


def _empty_missed_diag() -> dict[str, Any]:
    return {
        "true_dst_flow_id": "",
        "src_asset_group": "",
        "dst_asset_group": "",
        "src_route_id": "",
        "dst_route_id": "",
        "src_start_time": "",
        "src_end_time": "",
        "dst_start_time": "",
        "dst_end_time": "",
        "delay_sec": "",
        "src_amount_usd": "",
        "dst_amount_usd": "",
        "amount_error": "",
        "address_overlap": "",
        "rank_before_global_pool": "",
        "rank_after_global_pool": "",
        "rank_in_per_src_heuristic": "",
        "rank_in_per_src_topk": "",
        "selected_by_per_src_topk": "",
        "selected_by_global_pool": "",
        "miss_reason": "",
    }


def _rank_in_sorted_desc(score_map: dict[int, float], j: int) -> int:
    keys = sorted(score_map.keys(), key=lambda x: -score_map.get(x, 0.0))
    if j not in score_map:
        return len(keys) + 1
    return keys.index(j) + 1


def _rank_in_active_pool(dst_score: dict[int, float], active: set[int], j: int) -> int:
    keys = sorted(active, key=lambda x: -dst_score.get(x, 0.0))
    if j not in active:
        return len(keys) + 1
    return keys.index(j) + 1


def write_missed_true_dst_reason_summary(missed_csv: Path, out_summary_csv: Path) -> None:
    if not missed_csv.is_file():
        return
    df = pd.read_csv(missed_csv, dtype=str, keep_default_na=False)
    if df.empty or "miss_reason" not in df.columns:
        return
    num_cols = [c for c in ("delay_sec", "amount_error", "rank_before_global_pool", "rank_after_global_pool") if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _top_join(subdf: pd.DataFrame, col: str, n: int = 5) -> str:
        if col not in subdf.columns:
            return ""
        vc = subdf[col].astype(str).str.strip()
        vc = vc[vc != ""]
        if vc.empty:
            return ""
        return "|".join(f"{a}:{b}" for a, b in vc.value_counts().head(n).items())

    rows_out: list[dict[str, Any]] = []
    for reason, g in df.groupby("miss_reason"):
        n = len(g)
        row: dict[str, Any] = {
            "reason": str(reason),
            "count": int(n),
            "ratio": float(n / max(len(df), 1)),
        }
        if "delay_sec" in g.columns:
            row["median_delay_sec"] = float(g["delay_sec"].median()) if g["delay_sec"].notna().any() else ""
            row["p90_delay_sec"] = float(g["delay_sec"].quantile(0.9)) if g["delay_sec"].notna().any() else ""
        else:
            row["median_delay_sec"] = ""
            row["p90_delay_sec"] = ""
        row["top_asset_groups"] = _top_join(g, "dst_asset_group") if "dst_asset_group" in g.columns else ""
        row["top_route_ids"] = _top_join(g, "dst_route_id") if "dst_route_id" in g.columns else ""
        if "amount_error" in g.columns:
            row["median_amount_error"] = float(g["amount_error"].median()) if g["amount_error"].notna().any() else ""
            row["p90_amount_error"] = float(g["amount_error"].quantile(0.9)) if g["amount_error"].notna().any() else ""
        else:
            row["median_amount_error"] = ""
            row["p90_amount_error"] = ""
        rows_out.append(row)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).sort_values("count", ascending=False).to_csv(out_summary_csv, index=False)


def write_missed_true_dst_debug(
    flow_labels_csv: Path,
    active_dst_flow_ids: set[str],
    eth_by_flow_id: dict[str, dict[str, Any]],
    bnb_by_flow_id: dict[str, dict[str, Any]],
    out_csv: Path,
    *,
    min_label_confidence: float,
    pool_debug: dict[str, Any] | None = None,
) -> int:
    """Oracle-style: label dst flows absent from the UOT BNB column pool (for tuning recall).

    When ``pool_debug`` is populated by :func:`select_bnb_subgraph_for_flow_uot`, rows include
    per-edge diagnostics (ranks, delays, coarse miss_reason buckets).
    """
    df = pd.read_csv(flow_labels_csv, dtype=str, keep_default_na=False)
    if df.empty or "dst_flow_id" not in df.columns:
        return 0
    df["_lc"] = pd.to_numeric(df.get("label_confidence"), errors="coerce").fillna(0.0)
    sub = df[df["_lc"] >= float(min_label_confidence)]

    eth_list: list[dict[str, Any]] | None = None
    bnb_list: list[dict[str, Any]] | None = None
    eth_idx: dict[str, int] = {}
    bnb_idx: dict[str, int] = {}
    by_ag: dict[str, list[int]] | None = None
    by_rid: dict[str, list[int]] | None = None
    cache: dict[int, tuple[list[int], list[int], int]] = {}
    active_index_set: set[int] = set()
    dst_score: dict[int, float] = {}
    max_delay_sec = 21_600.0
    max_scan = 8000
    hard_cap = 4000
    universe_cap = 9000
    route_preserving_m = 0

    if pool_debug:
        eth_list = pool_debug.get("eth_flows")  # type: ignore[assignment]
        bnb_list = pool_debug.get("bnb_flows")  # type: ignore[assignment]
        if isinstance(eth_list, list) and isinstance(bnb_list, list):
            eth_idx = {str(f.get("flow_id") or "").strip(): i for i, f in enumerate(eth_list) if str(f.get("flow_id") or "").strip()}
            bnb_idx = {str(f.get("flow_id") or "").strip(): j for j, f in enumerate(bnb_list) if str(f.get("flow_id") or "").strip()}
            ais = pool_debug.get("active_index_set")
            if isinstance(ais, set):
                active_index_set = ais
            ds = pool_debug.get("dst_score")
            if isinstance(ds, dict):
                dst_score = {int(k): float(v) for k, v in ds.items()}
            max_delay_sec = float(pool_debug.get("max_delay_sec") or max_delay_sec)
            max_scan = int(pool_debug.get("max_scan") or max_scan)
            hard_cap = int(pool_debug.get("hard_cap") or hard_cap)
            universe_cap = int(pool_debug.get("universe_cap") or universe_cap)
            route_preserving_m = int(pool_debug.get("route_preserving_m") or 0)
            by_ag, by_rid = _bnb_indices_by_route_and_asset(bnb_list)

    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        did = str(r.get("dst_flow_id") or "").strip()
        sid = str(r.get("src_flow_id") or "").strip()
        if not did or did in active_dst_flow_ids:
            continue
        eth = eth_by_flow_id.get(sid) or {}
        bnb = bnb_by_flow_id.get(did) or {}
        base = {
            "dst_flow_id": did,
            "src_flow_id": sid,
            "label_confidence": float(r.get("_lc") or 0.0),
            "pattern_type": str(r.get("pattern_type") or ""),
            "dst_in_candidate_pool": False,
            "eth_asset_group": str(eth.get("asset_group") or ""),
            "eth_route_id": str(eth.get("route_id") or ""),
            "dst_asset_group": str(bnb.get("asset_group") or ""),
            "dst_route_id": str(bnb.get("route_id") or ""),
            "dst_amount_usd": _f(bnb.get("amount_usd")),
            "eth_amount_usd": _f(eth.get("amount_usd")),
        }
        if (
            not pool_debug
            or not isinstance(eth_list, list)
            or not isinstance(bnb_list, list)
            or by_ag is None
            or by_rid is None
        ):
            rows.append(base)
            continue
        i = eth_idx.get(sid, -1)
        j = bnb_idx.get(did, -1)
        if i < 0:
            base.update(_empty_missed_diag())
            base["miss_reason"] = "dst_not_found_in_segments"
            rows.append(base)
            continue
        eth_i = eth_list[i]
        prk = pool_debug.get("per_src_k_req")
        if isinstance(prk, list) and i < len(prk):
            k_req = int(prk[i])
        else:
            k_req = int(pool_debug.get("top_k_per_src_arg") or 200)
        if i not in cache:
            fs, tk = _build_sorted_candidates_one_eth(
                eth_i,
                bnb_list,
                by_ag=by_ag,
                by_rid=by_rid,
                max_delay_sec=max_delay_sec,
                max_scan=max_scan,
                hard_cap=hard_cap,
                universe_cap=universe_cap,
                k_req=k_req,
                route_preserving_m=route_preserving_m,
            )
            cache[i] = (fs, tk, k_req)
        full_sorted, topk, k_eff = cache[i]
        if j < 0:
            base.update(_empty_missed_diag())
            base["miss_reason"] = "dst_not_found_in_segments"
            rows.append(base)
            continue
        bnb_j = bnb_list[j]
        se = _f(eth_i.get("end_time"))
        ds = _f(bnb_j.get("start_time"))
        delay = ds - se
        sa = max(_f(eth_i.get("amount_usd")), 1e-12)
        da = max(_f(bnb_j.get("amount_usd")), 1e-12)
        amt_err = abs(sa - da) / max(sa, 1e-12)
        ov = _addr_overlap_bonus(eth_i, bnb_j)
        rk_full = full_sorted.index(j) + 1 if j in set(full_sorted) else 999_999
        rk_top = topk.index(j) + 1 if j in set(topk) else ""
        rk_bg = _rank_in_sorted_desc(dst_score, j) if dst_score else 0
        rk_ag = _rank_in_active_pool(dst_score, active_index_set, j) if dst_score else 0
        miss_reason = _classify_miss_reason(
            eth_i,
            bnb_j,
            j,
            full_sorted=full_sorted,
            topk=topk,
            k_req=k_eff,
            active_index_set=active_index_set,
            dst_score=dst_score,
            max_delay_sec=max_delay_sec,
            by_ag=by_ag,
            by_rid=by_rid,
            bnb_flows=bnb_list,
            max_scan=max_scan,
            hard_cap=hard_cap,
            universe_cap=universe_cap,
            route_preserving_m=route_preserving_m,
        )
        base.update(
            {
                "true_dst_flow_id": did,
                "src_asset_group": str(eth_i.get("asset_group") or ""),
                "dst_asset_group": str(bnb_j.get("asset_group") or ""),
                "src_route_id": str(eth_i.get("route_id") or ""),
                "dst_route_id": str(bnb_j.get("route_id") or ""),
                "src_start_time": _f(eth_i.get("start_time")),
                "src_end_time": se,
                "dst_start_time": ds,
                "dst_end_time": _f(bnb_j.get("end_time")),
                "delay_sec": delay,
                "src_amount_usd": sa,
                "dst_amount_usd": da,
                "amount_error": amt_err,
                "address_overlap": ov,
                "rank_before_global_pool": int(rk_bg),
                "rank_after_global_pool": int(rk_ag),
                "rank_in_per_src_heuristic": int(rk_full) if rk_full < 999_999 else "",
                "rank_in_per_src_topk": rk_top,
                "selected_by_per_src_topk": j in set(topk),
                "selected_by_global_pool": j in active_index_set,
                "miss_reason": miss_reason,
            }
        )
        rows.append(base)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    reason_csv = out_csv.parent / "missed_true_dst_reason_summary.csv"
    write_missed_true_dst_reason_summary(out_csv, reason_csv)
    return len(rows)
