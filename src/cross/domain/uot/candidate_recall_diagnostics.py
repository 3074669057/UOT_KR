"""Diagnostic-only candidate pool analysis (oracle ranks, @k recall, pool sweeps; no RC-UOT training)."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import locate_output_file
from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv
from cross.domain.uot.flow_uot_candidate_subgraph import (
    _addr_overlap_bonus,
    _f,
    _heuristic_cost,
    candidate_dst_recall_from_flow_labels,
    select_bnb_subgraph_for_flow_uot,
)

logger = logging.getLogger(__name__)

LOCK_FILENAME = "candidate_recall_diagnostics.lock"
ORACLE_LOG_EVERY = 100


def _flush_logs() -> None:
    for lg in (logging.getLogger(), logger):
        for h in getattr(lg, "handlers", []) or []:
            try:
                h.flush()
            except Exception:
                pass


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            k = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
            if h:
                k.CloseHandle(h)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_diagnostics_lock(out_root: Path) -> Path:
    """Create lock file; raise if another live process holds it."""
    exp = Path(out_root) / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    lock_p = exp / LOCK_FILENAME
    if lock_p.is_file():
        try:
            raw = lock_p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            old_pid = int(raw[0]) if raw else -1
        except (ValueError, OSError):
            old_pid = -1
        if old_pid > 0 and _pid_running(old_pid):
            raise RuntimeError("candidate recall diagnostics already running for this output directory")
        try:
            lock_p.unlink()
        except OSError as e:
            raise RuntimeError(f"Cannot clear stale diagnostics lock {lock_p}: {e}") from e
    lock_p.write_text(f"{os.getpid()}\n{time.time()}\n", encoding="utf-8")
    _flush_logs()
    return lock_p


def _release_lock(lock_p: Path | None) -> None:
    if lock_p is None:
        return
    try:
        if lock_p.is_file():
            lock_p.unlink()
    except OSError:
        pass


def _atomic_replace(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(tmp), str(final))


def output_subpath_eval(out_root: Path, name: str) -> Path:
    return out_root / "eval" / name


def _flow_index_maps(
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    ei = {str(f.get("flow_id") or "").strip(): i for i, f in enumerate(eth_flows) if str(f.get("flow_id") or "").strip()}
    bj = {str(f.get("flow_id") or "").strip(): j for j, f in enumerate(bnb_flows) if str(f.get("flow_id") or "").strip()}
    return ei, bj


def _flow_ids_from_segment_csv(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "flow_id" not in df.columns:
        return set()
    return {str(x).strip() for x in df["flow_id"].tolist() if str(x).strip()}


def write_flow_id_consistency_report(out_root: Path, lab_p: Path) -> Path:
    out_root = Path(out_root)
    df = pd.read_csv(lab_p, dtype=str, keep_default_na=False)
    dsts = {str(x).strip() for x in df.get("dst_flow_id", pd.Series(dtype=str)).tolist() if str(x).strip()}
    n_dst = len(dsts)

    paths = {
        "labels_bnb": out_root / "labels" / "flow_segments_bnb.csv",
        "label_layer_v1_bnb": out_root / "label_layer_v1" / "flow_segments_bnb.csv",
        "uot_bnb": out_root / "uot" / "flow_segments_bnb.csv",
    }
    seg_ids = {k: _flow_ids_from_segment_csv(p) for k, p in paths.items()}
    n_seg = {k: len(v) for k, v in seg_ids.items()}

    def ratio(found: int) -> float:
        return float(found / max(n_dst, 1))

    found_counts = {k: sum(1 for d in dsts if d in ids) for k, ids in seg_ids.items()}
    ratios = {f"label_dst_found_in_{k}_bnb_segments_ratio": ratio(found_counts[k]) for k in seg_ids}

    missing_examples: list[str] = []
    lab_ids = seg_ids["labels_bnb"]
    for d in list(dsts)[:80]:
        if d not in lab_ids:
            missing_examples.append(d)

    rep = {
        "label_dst_unique_count": int(n_dst),
        "bnb_segment_flow_id_unique_count": {k: int(n_seg[k]) for k in n_seg},
        **ratios,
        "examples_missing_dst_flow_ids": missing_examples,
    }
    outp = out_root / "experiments" / "flow_id_consistency_report.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    tmp.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    _atomic_replace(tmp, outp)
    return outp


def _oracle_dst_indices_from_labels(
    flow_labels_csv: Path,
    bnb_idx: dict[str, int],
    *,
    min_label_confidence: float,
) -> set[int]:
    df = pd.read_csv(flow_labels_csv, dtype=str, keep_default_na=False)
    df["_lc"] = pd.to_numeric(df.get("label_confidence"), errors="coerce").fillna(0.0)
    sub = df[df["_lc"] >= float(min_label_confidence)]
    out: set[int] = set()
    for did in sub["dst_flow_id"].astype(str).str.strip().unique():
        if not did:
            continue
        j = bnb_idx.get(did)
        if j is not None:
            out.add(int(j))
    return out


def write_oracle_forcing_debug_and_summary(
    out_root: Path,
    lab_p: Path,
    bnb_flows: list[dict[str, Any]],
    bj: dict[str, int],
    oracle_idx: set[int],
    meta: dict[str, Any],
    active_indices: list[int],
    *,
    min_label_confidence: float,
    rec_metrics: dict[str, Any],
) -> tuple[Path, Path]:
    """Per label edge: oracle injection / mid-pool / final pool audit (F strategy)."""
    out_root = Path(out_root)
    mid_u = set(int(x) for x in (meta.get("oracle_u_after_while") or active_indices))
    active_idx_set = set(int(x) for x in active_indices)
    active_flow_ids = {str(bnb_flows[j].get("flow_id") or "").strip() for j in active_idx_set if 0 <= j < len(bnb_flows)}

    df = pd.read_csv(lab_p, dtype=str, keep_default_na=False)
    df["_lc"] = pd.to_numeric(df.get("label_confidence"), errors="coerce").fillna(0.0)
    sub = df[df["_lc"] >= float(min_label_confidence)]

    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        sid = str(r.get("src_flow_id") or "").strip()
        did = str(r.get("dst_flow_id") or "").strip()
        if not sid or not did:
            continue
        j = bj.get(did)
        in_seg = j is not None
        inj = j is not None and j in oracle_idx
        in_mid = j is not None and j in mid_u
        in_final_idx = j is not None and j in active_idx_set
        counted = did in active_flow_ids

        if not in_seg:
            reason = "dst_flow_id_not_found_in_bnb_segments"
        elif not inj:
            reason = "dst_flow_id_found_but_index_not_injected"
        elif not in_mid:
            reason = "injected_but_removed_by_budget_trim"
        elif not in_final_idx:
            reason = "injected_but_not_in_final_active_cols"
        elif not counted:
            reason = "recall_metric_failed_to_count"
        else:
            reason = "ok"

        rows.append(
            {
                "label_src_flow_id": sid,
                "label_dst_flow_id": did,
                "dst_flow_id_found_in_bnb_segments": bool(in_seg),
                "dst_flow_index": int(j) if j is not None else "",
                "oracle_index_injected": inj,
                "oracle_index_present_after_pool_trim": in_mid,
                "oracle_index_present_in_final_active_cols": in_final_idx,
                "candidate_dst_recall_counted": bool(counted),
                "failure_reason": reason,
            }
        )

    dbg_p = out_root / "experiments" / "oracle_forcing_debug.csv"
    dbg_p.parent.mkdir(parents=True, exist_ok=True)
    dbg_tmp = dbg_p.with_suffix(dbg_p.suffix + ".tmp")
    logger.info("writing %s", dbg_tmp.name)
    pd.DataFrame(rows).to_csv(dbg_tmp, index=False)
    logger.info("renamed %s -> %s", dbg_tmp.name, dbg_p.name)
    _atomic_replace(dbg_tmp, dbg_p)
    _flush_logs()

    fc = Counter(str(r["failure_reason"]) for r in rows)
    summary = {
        "num_label_edges": int(len(rows)),
        "num_unique_label_dst": int(len({str(r["label_dst_flow_id"]) for r in rows})),
        "num_label_dst_found_in_bnb_segments": int(sum(1 for r in rows if r["dst_flow_id_found_in_bnb_segments"])),
        "num_oracle_indices_injected": int(len(oracle_idx)),
        "num_oracle_indices_present_after_pool_trim": int(sum(1 for j in oracle_idx if j in mid_u)),
        "num_oracle_indices_present_in_final_active_cols": int(sum(1 for j in oracle_idx if j in active_idx_set)),
        "candidate_dst_recall_under_oracle": float(rec_metrics.get("candidate_dst_recall_unique_dst") or 0.0),
        "candidate_dst_recall_edge_level_under_oracle": float(rec_metrics.get("candidate_dst_recall_edge_level") or 0.0),
        "failure_reason_counts": dict(fc),
    }
    sum_p = out_root / "experiments" / "oracle_forcing_summary.json"
    sum_tmp = sum_p.with_suffix(sum_p.suffix + ".tmp")
    logger.info("writing %s", sum_tmp.name)
    sum_tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("renamed %s -> %s", sum_tmp.name, sum_p.name)
    _atomic_replace(sum_tmp, sum_p)
    _flush_logs()
    return dbg_p, sum_p


def _oracle_rank_maps_for_src(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    *,
    max_delay_sec: float,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[int, int], int]:
    m = len(bnb_flows)
    se = _f(eth.get("end_time"))
    sa = max(_f(eth.get("amount_usd")), 1e-12)
    ag_s = str(eth.get("asset_group") or "").strip()
    rid_s = str(eth.get("route_id") or "").strip()

    hc = np.empty(m, dtype=float)
    tdel = np.empty(m, dtype=float)
    aerr = np.empty(m, dtype=float)
    addv = np.empty(m, dtype=float)
    for j, bf in enumerate(bnb_flows):
        hc[j] = float(_heuristic_cost(eth, bf, max_delay_sec=max_delay_sec))
        tdel[j] = abs(float(_f(bf.get("start_time"))) - se)
        da = max(float(_f(bf.get("amount_usd"))), 1e-12)
        aerr[j] = abs(sa - da) / max(sa, 1e-12)
        addv[j] = -float(_addr_overlap_bonus(eth, bf))

    def ranks_from_scores(scores: np.ndarray) -> dict[int, int]:
        order = np.argsort(scores, kind="mergesort")
        return {int(order[idx]): idx + 1 for idx in range(m)}

    r_c = ranks_from_scores(hc)
    r_t = ranks_from_scores(tdel)
    r_a = ranks_from_scores(aerr)
    r_d = ranks_from_scores(addv)

    route_js = [
        j
        for j in range(m)
        if rid_s
        and str(bnb_flows[j].get("route_id") or "").strip() == rid_s
        and (str(bnb_flows[j].get("asset_group") or "").strip() == ag_s or not ag_s)
    ]
    if route_js:
        rsub = np.array([hc[j] for j in route_js], dtype=float)
        order_local = np.argsort(rsub, kind="mergesort")
        r_route = {}
        for rk, pos in enumerate(order_local, start=1):
            j = route_js[int(pos)]
            r_route[int(j)] = rk
    else:
        r_route = {}
    return r_c, r_t, r_a, r_d, r_route, m


def _oracle_row_from_maps(
    eth: dict[str, Any],
    bnb_flows: list[dict[str, Any]],
    j_star: int,
    *,
    maps: tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[int, int], int],
    active_pool: set[int],
    max_delay_sec: float,
) -> dict[str, Any]:
    r_c, r_t, r_a, r_d, r_route, m = maps
    se = _f(eth.get("end_time"))
    ag_s = str(eth.get("asset_group") or "").strip()
    rid_s = str(eth.get("route_id") or "").strip()
    b_star = bnb_flows[j_star]
    ag_m = ag_s == str(b_star.get("asset_group") or "").strip() and bool(ag_s)
    rid_m = bool(rid_s) and rid_s == str(b_star.get("route_id") or "").strip()
    delay_sec = _f(b_star.get("start_time")) - se
    sa = max(_f(eth.get("amount_usd")), 1e-12)
    da = max(_f(b_star.get("amount_usd")), 1e-12)
    amount_error = abs(sa - da) / max(sa, 1e-12)
    ov = _addr_overlap_bonus(eth, b_star)
    r_comb = r_c.get(j_star, m + 1)
    r_time = r_t.get(j_star, m + 1)
    r_amt = r_a.get(j_star, m + 1)
    r_addr = r_d.get(j_star, m + 1)
    r_rt = r_route.get(j_star, m + 1)

    def in_top(k: int, r: int) -> bool:
        return r <= k

    ks = (50, 100, 200, 500, 1000)
    return {
        "true_dst_combined_rank": int(r_comb),
        "true_dst_time_rank": int(r_time),
        "true_dst_amount_rank": int(r_amt),
        "true_dst_route_rank": int(r_rt),
        "true_dst_address_rank": int(r_addr),
        **{f"true_dst_in_top_{k}": in_top(k, r_comb) for k in ks},
        "true_dst_in_global_pool": bool(j_star in active_pool),
        "delay_sec": float(delay_sec),
        "amount_error": float(amount_error),
        "asset_group_match": bool(ag_m),
        "route_id_match": bool(rid_m),
        "address_overlap": float(ov),
    }


def _label_subframe_for_diagnostics(lab_p: Path, min_label_confidence: float) -> pd.DataFrame:
    df = pd.read_csv(lab_p, dtype=str, keep_default_na=False)
    df["_lc"] = pd.to_numeric(df.get("label_confidence"), errors="coerce").fillna(0.0)
    return df[df["_lc"] >= float(min_label_confidence)]


def _sampled_src_flow_ids(sub: pd.DataFrame, ei: dict[str, int], max_sources: int | None) -> set[str] | None:
    if max_sources is None or max_sources <= 0:
        return None
    sids = sorted({str(x).strip() for x in sub.get("src_flow_id", pd.Series(dtype=str)).tolist() if str(x).strip()})
    allowed = set(sids[: int(max_sources)])
    return allowed


def write_candidate_oracle_rank_debug(
    out_root: Path,
    *,
    min_label_confidence: float,
    max_delay_sec: float,
    top_k_per_src: int,
    max_matrix_cells: int,
    max_sources: int | None = None,
    eth_flows: list[dict[str, Any]] | None = None,
    bnb_flows: list[dict[str, Any]] | None = None,
) -> Path | None:
    out_root = Path(out_root)
    eth_p = locate_output_file(out_root, "flow_segments_eth.csv")
    bnb_p = locate_output_file(out_root, "flow_segments_bnb.csv")
    lab_p = locate_output_file(out_root, "flow_labels.csv")
    loaded_here = False
    if eth_flows is None:
        logger.info("loaded flow_segments_eth from %s", eth_p)
        eth_flows = flows_from_segment_export_csv(eth_p, chain="ETH")
        loaded_here = True
    if bnb_flows is None:
        logger.info("loaded flow_segments_bnb from %s", bnb_p)
        bnb_flows = flows_from_segment_export_csv(bnb_p, chain="BNB")
        loaded_here = True
    if loaded_here:
        logger.info("loaded src flow count=%d dst flow count=%d", len(eth_flows), len(bnb_flows))
    ei, bj = _flow_index_maps(eth_flows, bnb_flows)
    logger.info("starting oracle rank diagnostics")
    _flush_logs()
    pool_dbg: dict[str, Any] = {}
    _, meta = select_bnb_subgraph_for_flow_uot(
        eth_flows,
        bnb_flows,
        top_k_per_src=int(top_k_per_src),
        max_delay_sec=float(max_delay_sec),
        max_matrix_cells=int(max_matrix_cells),
        pool_debug=pool_dbg,
    )
    active_idx = set(int(x) for x in (meta.get("active_dst_global_indices") or []))

    sub = _label_subframe_for_diagnostics(lab_p, min_label_confidence)
    logger.info("loaded label edges count=%d (oracle pass)", int(len(sub)))
    src_allow = _sampled_src_flow_ids(sub, ei, max_sources)

    edges_by_i: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    for _, r in sub.iterrows():
        sid = str(r.get("src_flow_id") or "").strip()
        did = str(r.get("dst_flow_id") or "").strip()
        if not sid or not did:
            continue
        if src_allow is not None and sid not in src_allow:
            continue
        i = ei.get(sid)
        j = bj.get(did)
        if i is None or j is None:
            edges_by_i[-1].append((sid, did, -1))
        else:
            edges_by_i[i].append((sid, did, j))

    pos_keys = sorted(k for k in edges_by_i if k >= 0)
    total_src_flows = len(pos_keys)
    logger.info("oracle rank: labelled source flows in scope=%d", total_src_flows)

    rows: list[dict[str, Any]] = []
    map_cache: dict[int, tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[int, int], int]] = {}
    processed = 0
    for i in pos_keys:
        triples = edges_by_i[i]
        eth = eth_flows[i]
        if i not in map_cache:
            map_cache[i] = _oracle_rank_maps_for_src(eth, bnb_flows, max_delay_sec=max_delay_sec)
        maps = map_cache[i]
        for sid, did, j in triples:
            o = _oracle_row_from_maps(eth, bnb_flows, j, maps=maps, active_pool=active_idx, max_delay_sec=max_delay_sec)
            rows.append({"src_flow_id": sid, "true_dst_flow_id": did, **o})
        processed += 1
        if processed % ORACLE_LOG_EVERY == 0 or processed == total_src_flows:
            logger.info("oracle rank progress: processed %d / %d source flows", processed, total_src_flows)
            _flush_logs()

    for sid, did, _ in edges_by_i.get(-1, []):
        rows.append({"src_flow_id": sid, "true_dst_flow_id": did, "true_dst_combined_rank": "", "note": "missing_flow_index"})

    outp = output_subpath_eval(out_root, "candidate_oracle_rank_debug.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    logger.info("writing %s", tmp.name)
    pd.DataFrame(rows).to_csv(tmp, index=False)
    logger.info("renamed %s -> %s", tmp.name, outp.name)
    _atomic_replace(tmp, outp)
    _flush_logs()
    logger.info("finished oracle rank diagnostics")
    _flush_logs()
    return outp


def write_candidate_recall_at_k_from_oracle_csv(oracle_csv: Path, out_csv: Path) -> None:
    if not oracle_csv.is_file():
        return
    df = pd.read_csv(oracle_csv, dtype=str, keep_default_na=False)
    if df.empty or "true_dst_combined_rank" not in df.columns:
        return
    df["_r"] = pd.to_numeric(df["true_dst_combined_rank"], errors="coerce")
    df["_rt"] = pd.to_numeric(df.get("true_dst_time_rank"), errors="coerce")
    df["_ra"] = pd.to_numeric(df.get("true_dst_amount_rank"), errors="coerce")
    df["_rr"] = pd.to_numeric(df.get("true_dst_route_rank"), errors="coerce")
    n = int(df["_r"].notna().sum())
    if n <= 0:
        return
    ks = [50, 100, 200, 500, 1000]

    def frac(col: str, k: int) -> float:
        s = df[col].dropna()
        if s.empty:
            return 0.0
        return float((s <= k).sum() / max(len(s), 1))

    rows: list[dict[str, Any]] = []
    for k in ks:
        ag_sub = df[df["asset_group_match"].astype(str).str.lower().isin(("true", "1"))] if "asset_group_match" in df.columns else df
        ag_rec = float((ag_sub["_r"] <= k).mean()) if len(ag_sub) and ag_sub["_r"].notna().any() else 0.0
        rows.append(
            {
                "k": k,
                "candidate_recall_at_k": frac("_r", k),
                "asset_group_recall_at_k": ag_rec,
                "route_recall_at_k": float((df["_rr"] <= k).mean()) if df["_rr"].notna().any() else 0.0,
                "time_window_recall_at_k": frac("_rt", k),
                "amount_rank_recall_at_k": frac("_ra", k),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    logger.info("writing %s", tmp.name)
    pd.DataFrame(rows).to_csv(tmp, index=False)
    logger.info("renamed %s -> %s", tmp.name, out_csv.name)
    _atomic_replace(tmp, out_csv)
    _flush_logs()


def run_candidate_pool_sweep(
    out_root: Path,
    *,
    min_label_confidence: float,
    max_delay_sec: float,
    uot_flow_dst_top_k_default: int,
    fast_mode: bool,
    max_sources: int | None,
    eth_flows: list[dict[str, Any]] | None = None,
    bnb_flows: list[dict[str, Any]] | None = None,
    only_strategies: list[str] | None = None,
) -> Path:
    """Writes ``experiments/candidate_pool_sweep.csv`` (recall metrics; no full UOT solve)."""
    t0 = time.perf_counter()
    out_root = Path(out_root)
    lock_p: Path | None = None
    if only_strategies:
        lock_p = _acquire_diagnostics_lock(out_root)
    try:
        return _run_candidate_pool_sweep_impl(
            out_root,
            min_label_confidence=min_label_confidence,
            max_delay_sec=max_delay_sec,
            uot_flow_dst_top_k_default=uot_flow_dst_top_k_default,
            fast_mode=fast_mode,
            max_sources=max_sources,
            eth_flows=eth_flows,
            bnb_flows=bnb_flows,
            only_strategies=only_strategies,
            t0=t0,
        )
    finally:
        if only_strategies:
            _release_lock(lock_p)


def _run_candidate_pool_sweep_impl(
    out_root: Path,
    *,
    min_label_confidence: float,
    max_delay_sec: float,
    uot_flow_dst_top_k_default: int,
    fast_mode: bool,
    max_sources: int | None,
    eth_flows: list[dict[str, Any]] | None,
    bnb_flows: list[dict[str, Any]] | None,
    only_strategies: list[str] | None,
    t0: float,
) -> Path:
    eth_p = locate_output_file(out_root, "flow_segments_eth.csv")
    bnb_p = locate_output_file(out_root, "flow_segments_bnb.csv")
    lab_p = locate_output_file(out_root, "flow_labels.csv")
    if eth_flows is None:
        eth_flows = flows_from_segment_export_csv(eth_p, chain="ETH")
    if bnb_flows is None:
        bnb_flows = flows_from_segment_export_csv(bnb_p, chain="BNB")
    _, bj = _flow_index_maps(eth_flows, bnb_flows)
    n = len(eth_flows)
    bnb_seg_ids = {str(f.get("flow_id") or "").strip() for f in bnb_flows if str(f.get("flow_id") or "").strip()}

    logger.info("starting candidate_pool_sweep")
    _flush_logs()

    oracle_csv = output_subpath_eval(out_root, "candidate_oracle_rank_debug.csv")
    recall_k_csv = output_subpath_eval(out_root, "candidate_recall_at_k.csv")
    rk_df = pd.DataFrame()

    if not fast_mode:
        write_candidate_oracle_rank_debug(
            out_root,
            min_label_confidence=min_label_confidence,
            max_delay_sec=max_delay_sec,
            top_k_per_src=int(uot_flow_dst_top_k_default),
            max_matrix_cells=6_000_000,
            max_sources=max_sources,
            eth_flows=eth_flows,
            bnb_flows=bnb_flows,
        )
        write_candidate_recall_at_k_from_oracle_csv(oracle_csv, recall_k_csv)
        rk_df = pd.read_csv(recall_k_csv, dtype=str, keep_default_na=False) if recall_k_csv.is_file() else pd.DataFrame()
    else:
        logger.info("fast mode: skipping full oracle rank CSV and candidate_recall_at_k.csv")

    def recall_at(k: int, col_prefix: str = "candidate_recall_at_k") -> str:
        if fast_mode or rk_df.empty or "k" not in rk_df.columns:
            return ""
        row = rk_df[rk_df["k"].astype(str) == str(k)]
        if row.empty:
            return ""
        c = col_prefix if col_prefix in row.columns else "candidate_recall_at_k"
        if c not in row.columns:
            return ""
        return str(row.iloc[0][c])

    oracle_idx = _oracle_dst_indices_from_labels(lab_p, bj, min_label_confidence=min_label_confidence)

    specs: list[tuple[str, str, int, int, int, int, frozenset[int] | None, bool, bool]] = [
        ("A_current", "default", int(uot_flow_dst_top_k_default), 6_000_000, 0, 0, None, False, False),
        ("B_larger_topk", "default", 500, 6_000_000, 0, 0, None, False, False),
        ("C_larger_budget", "default", int(uot_flow_dst_top_k_default), 12_000_000, 0, 0, None, False, False),
        ("D_stratified_pool", "stratified_global_pool", int(uot_flow_dst_top_k_default), 6_000_000, 100, 0, None, False, False),
        ("E_stratified_larger", "stratified_global_pool", 500, 12_000_000, 100, 0, None, False, False),
        ("G_larger_budget_18M", "default", int(uot_flow_dst_top_k_default), 18_000_000, 0, 0, None, False, False),
        ("F_oracle_upper_bound", "default", int(uot_flow_dst_top_k_default), 6_000_000, 0, 0, frozenset(oracle_idx), True, True),
    ]

    if only_strategies:
        allow = {str(x).strip() for x in only_strategies if str(x).strip()}
        specs = [s for s in specs if s[0] in allow]
        if not specs:
            raise ValueError(f"no strategies matched only_strategies={only_strategies!r}")

    rows_out: list[dict[str, Any]] = []
    for name, strategy, tk, budget, min_pool, route_m, oracle_fz, oracle_exceed, ret_mid in specs:
        logger.info("sweep strategy %s started", name)
        t1 = time.perf_counter()
        oracle_arg = set(oracle_fz) if oracle_fz is not None else None
        bnb_sub, meta = select_bnb_subgraph_for_flow_uot(
            eth_flows,
            bnb_flows,
            top_k_per_src=int(tk),
            max_delay_sec=float(max_delay_sec),
            max_matrix_cells=int(budget),
            pool_strategy=strategy,
            min_pool_per_asset=int(min_pool),
            route_preserving_m=int(route_m),
            oracle_dst_flow_indices=oracle_arg,
            oracle_diagnostic_exceed_budget=bool(oracle_exceed),
            oracle_return_mid_u=bool(ret_mid),
            pool_debug=None,
        )
        active_ids = {str(f.get("flow_id") or "").strip() for f in bnb_sub if str(f.get("flow_id") or "").strip()}
        rec = candidate_dst_recall_from_flow_labels(
            lab_p,
            active_ids,
            min_label_confidence=float(min_label_confidence),
            bnb_segment_flow_ids=bnb_seg_ids,
        )
        cells = int(meta.get("matrix_cells") or (n * len(bnb_sub)))
        est = max(0.05, 2e-6 * float(cells))

        row_base: dict[str, Any] = {
            "strategy": name,
            "pool_strategy": strategy,
            "flow_dst_top_k": int(tk),
            "flow_max_matrix_cells": int(budget),
            "min_pool_per_asset": int(min_pool),
            "route_preserving_m": int(route_m),
            "oracle_forced_dst": oracle_arg is not None and len(oracle_arg) > 0,
            "oracle_diagnostic_exceed_budget": bool(oracle_exceed),
            "bnb_active": int(len(bnb_sub)),
            "matrix_cells": int(cells),
            "candidate_dst_recall": float(rec.get("candidate_dst_recall") or 0.0),
            "candidate_dst_recall_unique_dst": float(rec.get("candidate_dst_recall_unique_dst") or 0.0),
            "candidate_dst_recall_edge_level": float(rec.get("candidate_dst_recall_edge_level") or 0.0),
            "candidate_dst_recall_legacy_all_labels_denom": float(rec.get("candidate_dst_recall_legacy_all_labels_denom") or 0.0),
            "candidate_recall_at_50": recall_at(50),
            "candidate_recall_at_100": recall_at(100),
            "candidate_recall_at_200": recall_at(200),
            "candidate_recall_at_500": recall_at(500),
            "candidate_recall_at_1000": recall_at(1000),
            "estimated_runtime_sec": round(est, 3),
            "flow_pair_f1": "",
            "flow_mass_recall": "",
            "oracle_cells_exceed_budget": meta.get("oracle_cells_exceed_budget", ""),
            "oracle_protected_count": meta.get("oracle_protected_count", ""),
            "oracle_final_active_count": meta.get("oracle_final_active_count", ""),
            "oracle_recall_upper_bound": meta.get("oracle_recall_upper_bound", ""),
            "sweep_row_seconds": round(time.perf_counter() - t1, 3),
        }
        rows_out.append(row_base)

        if name == "F_oracle_upper_bound" and oracle_arg is not None:
            write_oracle_forcing_debug_and_summary(
                out_root,
                lab_p,
                bnb_flows,
                bj,
                set(oracle_arg),
                meta,
                list(meta.get("active_dst_global_indices") or []),
                min_label_confidence=min_label_confidence,
                rec_metrics=rec,
            )
        logger.info("sweep strategy %s finished (%.2fs)", name, time.perf_counter() - t1)
        _flush_logs()

    df_new = pd.DataFrame(rows_out)
    df_new["total_diag_seconds"] = round(time.perf_counter() - t0, 3)
    outp = out_root / "experiments" / "candidate_pool_sweep.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    logger.info("writing %s", tmp.name)
    if only_strategies and outp.is_file():
        allow_names = {s[0] for s in specs}
        old = pd.read_csv(outp, dtype=str, keep_default_na=False)
        if "strategy" in old.columns:
            old = old[~old["strategy"].astype(str).isin(allow_names)]
        df_out = pd.concat([old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(tmp, index=False)
    logger.info("renamed %s -> %s", tmp.name, outp.name)
    _atomic_replace(tmp, outp)
    _flush_logs()
    return outp


def run_candidate_recall_diagnostics(
    out_root: Path,
    *,
    min_label_confidence: float,
    max_delay_sec: float,
    top_k: int,
    fast_mode: bool = False,
    max_sources: int | None = None,
) -> dict[str, str]:
    """Orchestrate diagnostics with lock, progress logs, and optional fast mode."""
    out_root = Path(out_root)
    lock_p: Path | None = None
    try:
        lock_p = _acquire_diagnostics_lock(out_root)

        lab_p = locate_output_file(out_root, "flow_labels.csv")
        logger.info("loaded flow_labels from %s", lab_p)
        if not lab_p.is_file():
            raise FileNotFoundError(f"flow_labels.csv missing: {lab_p}")

        eth_p = locate_output_file(out_root, "flow_segments_eth.csv")
        bnb_p = locate_output_file(out_root, "flow_segments_bnb.csv")
        logger.info("loaded flow_segments_eth from %s", eth_p)
        eth_flows = flows_from_segment_export_csv(eth_p, chain="ETH")
        logger.info("loaded flow_segments_bnb from %s", bnb_p)
        bnb_flows = flows_from_segment_export_csv(bnb_p, chain="BNB")

        sub = _label_subframe_for_diagnostics(lab_p, min_label_confidence)
        n_edges = int(len(sub))
        u_src = {str(x).strip() for x in sub.get("src_flow_id", pd.Series(dtype=str)).tolist() if str(x).strip()}
        u_dst = {str(x).strip() for x in sub.get("dst_flow_id", pd.Series(dtype=str)).tolist() if str(x).strip()}
        logger.info("loaded label edges count=%d", n_edges)
        n_eth = len(eth_flows)
        n_bnb = len(bnb_flows)
        logger.info("loaded src flow count=%d dst flow count=%d", n_eth, n_bnb)

        m_dst = max(1, n_bnb)
        n_src_eff = min(len(u_src), max_sources) if max_sources else len(u_src)
        approx_ops = (
            int(n_src_eff * m_dst * 5 + 6 * max(1, n_eth) * m_dst * 0.0001) if not fast_mode else int(6 * max(1, n_eth) * m_dst * 0.0001)
        )
        logger.info(
            "candidate_recall_diagnostics workload: num_src_flows=%d num_dst_flows=%d num_label_edges=%d "
            "num_unique_label_src=%d num_unique_label_dst=%d expected_operations_approx=%d fast_mode=%s max_sources=%s",
            n_eth,
            n_bnb,
            n_edges,
            len(u_src),
            len(u_dst),
            approx_ops,
            fast_mode,
            max_sources if max_sources is not None else "all",
        )
        _flush_logs()

        logger.info("starting flow_id_consistency_report")
        fid_rep = write_flow_id_consistency_report(out_root, lab_p)
        logger.info("finished flow_id_consistency_report -> %s", fid_rep)
        _flush_logs()

        o3 = run_candidate_pool_sweep(
            out_root,
            min_label_confidence=min_label_confidence,
            max_delay_sec=max_delay_sec,
            uot_flow_dst_top_k_default=int(top_k),
            fast_mode=bool(fast_mode),
            max_sources=max_sources,
            eth_flows=eth_flows,
            bnb_flows=bnb_flows,
        )
        logger.info("finished candidate_pool_sweep -> %s", o3)
        _flush_logs()

        o1p = output_subpath_eval(out_root, "candidate_oracle_rank_debug.csv")
        o2p = output_subpath_eval(out_root, "candidate_recall_at_k.csv")
        o4 = str(out_root / "experiments" / "oracle_forcing_debug.csv")
        o5 = str(out_root / "experiments" / "oracle_forcing_summary.json")

        logger.info("finished candidate_recall_diagnostics")
        _flush_logs()
        return {
            "flow_id_consistency_report": str(fid_rep),
            "candidate_oracle_rank_debug": str(o1p) if o1p.is_file() else "",
            "candidate_recall_at_k": str(o2p) if o2p.is_file() else "",
            "candidate_pool_sweep": str(o3),
            "oracle_forcing_debug": o4,
            "oracle_forcing_summary": o5,
        }
    finally:
        _release_lock(lock_p)
