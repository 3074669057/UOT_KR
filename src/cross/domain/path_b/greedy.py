"""Global matching of src deposits to dst withdrawals (unique dst hash per pair).

Supports optional **Hungarian** (linear sum assignment) instead of greedy edge picking,
optional **edge_score_fn** hook for ABCT/TIR-style reranking terms, and label **truth-first
reservation** for oracle evaluation only.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ...shared.bnb_pick import pick_bnb_recipient
from ...shared.connector_decimals import decimals_for_eth_bnb
from ...shared.normalize import norm_addr
from ...utils.safe_cast import safe_float, safe_int

logger = logging.getLogger(__name__)

ZERO = "0x0000000000000000000000000000000000000000"

# Dense Hungarian cost matrix is R * (M + R) float64; cap to avoid multi-GB alloc + O(n³) hang.
_MAX_HUNGARIAN_COST_BYTES = 400 * 1024 * 1024
_MAX_HUNGARIAN_ROWS = 8000

FORBIDDEN_COST = 1e18
"""Cost for impossible (src,dst) edges in assignment."""

DUMMY_SKIPPED_COST = 0.0
"""Dummy column: row chose no dst (only when it has zero candidates)."""

DUMMY_REJECT_COST = 1e9
"""Dummy column: row skips despite having candidates (discouraged vs real edges)."""


def _human_src(eth_amt: float, eth_ca: str, dec_e: dict[str, int]) -> float | None:
    if eth_amt <= 0:
        return None
    eth_ca = (eth_ca or "").strip().lower()
    if not eth_ca or eth_ca == ZERO:
        return eth_amt / (10**18)
    d = dec_e.get(eth_ca)
    if d is None:
        return None
    return eth_amt / (10**d)


def _human_dst_for_hash(
    sub: pd.DataFrame,
    recv: str,
    dec_b: dict[str, int],
) -> tuple[float | None, float]:
    """Sum human-out amount paid to ``recv`` (ERC20 + native BNB) in this tx; return (human, min_ts).

    If any positive ERC20 leg lacks decimals in ``dec_b``, returns ``(None, min_ts)``.
    """
    tot = 0.0
    tmin = float("inf")
    unknown_erc20 = False
    sub = sub[sub["to"].map(norm_addr) == recv]
    for _, rr in sub.iterrows():
        ca = str(rr.get("contractAddress", "") or "").strip().lower()
        raw = safe_float(rr.get("value"), 0.0)
        if raw <= 0:
            continue
        ts_r = safe_float(rr.get("timeStamp"), 0.0)
        if not ca or ca == ZERO:
            tot += raw / (10**18)
        else:
            d = dec_b.get(ca)
            if d is None:
                unknown_erc20 = True
                continue
            tot += raw / (10**d)
        tmin = min(tmin, ts_r)
    if unknown_erc20:
        return None, tmin if tmin != float("inf") else 0.0
    return tot, tmin


EdgeScoreFn = Callable[[int, str, float], float]
"""(src_row_index, dst_hash_norm, base_error) -> score to minimize (lower is better)."""


def extract_candidate_fields(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Pick receiver / amount / token using ``row_type`` when present (receipt evidence rows)."""
    sr = row if isinstance(row, pd.Series) else pd.Series(row)
    rt = str(sr.get("row_type") or "").strip().lower()
    if rt == "erc20_transfer":
        return {
            "receiver": norm_addr(sr.get("transfer_to")),
            "amount": sr.get("raw_value"),
            "normalized_amount": sr.get("normalized_amount"),
            "token": str(sr.get("token_contract") or "").strip().lower(),
        }
    if rt == "native_transfer":
        recv = sr.get("transfer_to")
        if recv is None or (isinstance(recv, float) and pd.isna(recv)) or str(recv).strip() == "":
            recv = sr.get("to")
        return {
            "receiver": norm_addr(recv),
            "amount": sr.get("raw_value"),
            "normalized_amount": sr.get("normalized_amount"),
            "token": "BNB",
        }
    if rt == "bridge_event":
        return {
            "receiver": norm_addr(sr.get("decoded_receiver")),
            "amount": sr.get("decoded_amount"),
            "normalized_amount": sr.get("normalized_amount"),
            "token": str(sr.get("decoded_token") or "").strip().lower(),
        }
    return {
        "receiver": norm_addr(sr.get("to")),
        "amount": sr.get("value"),
        "normalized_amount": sr.get("normalized_amount"),
        "token": str(sr.get("contractAddress") or "").strip().lower(),
    }


def apply_evidence_confidence_cap(conf: float, evidence_level: int) -> float:
    """Cap confidence by discrete evidence level (guide §7.2)."""
    lev = safe_int(evidence_level, 0)
    c = float(conf)
    if lev < 2:
        return min(c, 0.35)
    if lev == 2:
        return min(c, 0.60)
    if lev == 3:
        return min(c, 0.85)
    return c


def _max_evidence_level_for_dst_hash(dst_all: pd.DataFrame, dst_hash: str) -> int | None:
    if dst_all is None or dst_all.empty or "evidence_level" not in dst_all.columns:
        return None
    sub = dst_all.loc[dst_all["hash"].astype(str).map(norm_addr) == norm_addr(dst_hash)]
    if sub.empty:
        return None
    return safe_int(pd.to_numeric(sub["evidence_level"], errors="coerce").fillna(0).max(), 0)


def _assign_greedy_flat(
    flat: list[tuple[float, int, str]],
    used_src_idx: set[int],
    used_dst_hash: set[str],
    src_all: pd.DataFrame,
) -> dict[str, str]:
    """Legacy global greedy: sort all (err, src_i, dst_h) ascending, first-wins."""
    mapping: dict[str, str] = {}
    flat = sorted(flat, key=lambda x: x[0])
    used_src = set(used_src_idx)
    used_dst = set(used_dst_hash)

    for err, i, h in flat:
        if i in used_src or h in used_dst:
            continue
        used_src.add(i)
        used_dst.add(h)
        txh = norm_addr(src_all.iloc[i].get("txhash", ""))
        mapping[txh] = h

    return mapping, used_src, used_dst


def _assign_with_strategy(
    *,
    per_src_candidates: list[list[tuple[float, str]]],
    used_src_idx: set[int],
    used_dst_hash: set[str],
    src_all: pd.DataFrame,
    greedy_assignment: str,
) -> tuple[dict[str, str], set[int], set[str]]:
    ga = (greedy_assignment or "greedy").strip().lower()
    if ga == "hungarian":
        rem_ix_est = [i for i in range(len(per_src_candidates)) if i not in used_src_idx]
        r_est = len(rem_ix_est)
        seen_est: set[str] = set()
        for i in rem_ix_est:
            for _err, h in per_src_candidates[i]:
                if h not in used_dst_hash:
                    seen_est.add(h)
        m_est = len(seen_est)
        est_bytes = r_est * (m_est + r_est) * 8
        if est_bytes > _MAX_HUNGARIAN_COST_BYTES or r_est > _MAX_HUNGARIAN_ROWS:
            logger.warning(
                "Path B: Hungarian would need ~%.0f MiB dense matrix (%s rows, %s dst columns); "
                "using greedy assignment instead. Lower greedy_top_k or pass "
                "--path-b-assignment greedy.",
                est_bytes / (1024 * 1024),
                r_est,
                m_est,
            )
            ga = "greedy"
        else:
            return _assign_hungarian(
                per_src_candidates,
                used_src_idx,
                used_dst_hash,
                src_all,
            )

    flat: list[tuple[float, int, str]] = []
    for i, lst in enumerate(per_src_candidates):
        for err, h in lst:
            flat.append((err, i, h))
    return _assign_greedy_flat(
        flat,
        used_src_idx,
        used_dst_hash,
        src_all,
    )


def _assign_hungarian(
    per_src_candidates: list[list[tuple[float, str]]],
    used_src_idx: set[int],
    used_dst_hash: set[str],
    src_all: pd.DataFrame,
) -> tuple[dict[str, str], set[int], set[str]]:
    """Min-cost bipartite matching on remaining rows vs union(dst) + per-row dummy columns."""
    n = len(per_src_candidates)
    rem_ix = [i for i in range(n) if i not in used_src_idx]
    if not rem_ix:
        return {}, used_src_idx, used_dst_hash

    dst_cols: list[str] = []
    seen: set[str] = set()
    for i in rem_ix:
        for err, h in per_src_candidates[i]:
            if h in used_dst_hash:
                continue
            if h not in seen:
                seen.add(h)
                dst_cols.append(h)

    if not dst_cols:
        mapping: dict[str, str] = {}
        used_src = set(used_src_idx)
        used_dst = set(used_dst_hash)
        return mapping, used_src, used_dst

    M = len(dst_cols)
    dst_index = {h: j for j, h in enumerate(dst_cols)}
    R = len(rem_ix)
    C = M + R
    nbytes = R * C * 8
    t0 = time.perf_counter()
    logger.info(
        "Path B Hungarian: building %s x %s cost matrix (~%.1f MiB)",
        R,
        C,
        nbytes / (1024 * 1024),
    )
    cost = np.full((R, C), FORBIDDEN_COST, dtype=np.float64)

    for r, i in enumerate(rem_ix):
        lst = per_src_candidates[i]
        cands = {h: err for err, h in lst}
        for h, err in cands.items():
            j = dst_index.get(h)
            if j is None:
                continue
            cost[r, j] = err
        if not lst:
            cost[r, M + r] = DUMMY_SKIPPED_COST
        else:
            cost[r, M + r] = DUMMY_REJECT_COST

    row_ind, col_ind = linear_sum_assignment(cost)
    logger.info(
        "Path B Hungarian: linear_sum_assignment finished in %.1fs",
        time.perf_counter() - t0,
    )
    mapping = {}
    used_src = set(used_src_idx)
    used_dst = set(used_dst_hash)

    for r, k in zip(row_ind, col_ind):
        i = rem_ix[r]
        txh = norm_addr(src_all.iloc[i].get("txhash", ""))
        if k < M:
            dh = dst_cols[k]
            if cost[r, k] >= FORBIDDEN_COST / 2:
                continue
            if dh in used_dst or i in used_src:
                continue
            mapping[txh] = dh
            used_src.add(i)
            used_dst.add(dh)

    return mapping, used_src, used_dst


def greedy_pair_dst_hashes(
    src_all: pd.DataFrame,
    dst_all: pd.DataFrame,
    *,
    dst_window_sec: float = 86400.0,
    top_k_per_src: int = 200,
    first_batch_k: int = 100,
    second_batch_k: int = 100,
    ratio_by_eth_token: dict[str, float] | None = None,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
    median_delay_sec: float | None = None,
    delay_weight: float = 0.12,
    truth_dst_by_src: dict[str, str] | None = None,
    boost_truth_in_candidates: bool = False,
    edge_score_fn: EdgeScoreFn | None = None,
    greedy_assignment: str = "greedy",
    receiver_mode: str = "eth_from",
    bnb_bridge_address: str = "",
) -> dict[str, str]:
    mapping, _detail = greedy_pair_dst_hashes_with_evidence(
        src_all,
        dst_all,
        dst_window_sec=dst_window_sec,
        top_k_per_src=top_k_per_src,
        first_batch_k=first_batch_k,
        second_batch_k=second_batch_k,
        ratio_by_eth_token=ratio_by_eth_token,
        ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
        median_delay_sec=median_delay_sec,
        delay_weight=delay_weight,
        truth_dst_by_src=truth_dst_by_src,
        boost_truth_in_candidates=boost_truth_in_candidates,
        edge_score_fn=edge_score_fn,
        greedy_assignment=greedy_assignment,
        receiver_mode=receiver_mode,
        bnb_bridge_address=bnb_bridge_address,
    )
    return mapping


def _softmax_confidence_from_errors(errors: list[float], selected_idx: int, *, temperature: float = 1.0) -> float:
    if not errors or selected_idx < 0 or selected_idx >= len(errors):
        return 0.0
    arr = np.asarray(errors, dtype=np.float64)
    m = float(np.min(arr))
    tau = max(float(temperature), 1e-6)
    logits = -(arr - m) / tau
    ex = np.exp(np.clip(logits, -60.0, 60.0))
    z = float(np.sum(ex))
    if z <= 0:
        return 0.0
    return float(ex[selected_idx] / z)


def _dynamic_window_sec(
    base_window_sec: float,
    *,
    median_delay_sec: float | None,
    has_ratio: bool,
    has_token: bool,
) -> float:
    w = float(base_window_sec)
    if median_delay_sec is not None and median_delay_sec > 0:
        w = min(w, max(3600.0, float(median_delay_sec) * 6.0))
    if has_ratio:
        w *= 0.85
    if has_token:
        w *= 0.9
    return float(min(max(w, 900.0), base_window_sec * 1.25))


def _candidate_gate(
    *,
    err: float,
    ratio_err: float | None,
    human_err: float | None,
    delay_err: float,
    dst_gap_sec: float,
    dynamic_window_sec: float,
) -> bool:
    # Hard gate to remove implausible candidates before assignment.
    if dst_gap_sec > dynamic_window_sec * 1.02:
        return False
    if ratio_err is not None and ratio_err > 5.0:
        return False
    if ratio_err is None and human_err is not None and human_err > 8.0:
        return False
    if delay_err > 8.0:
        return False
    if err > 5e6:
        return False
    return True


def _calibrate_confidence(raw: float) -> float:
    # Conservative piecewise calibration from heuristic reliability buckets.
    r = float(np.clip(raw, 0.0, 1.0))
    if r < 0.2:
        return 0.18 * r / 0.2
    if r < 0.5:
        return 0.18 + (r - 0.2) * (0.48 - 0.18) / 0.3
    if r < 0.8:
        return 0.48 + (r - 0.5) * (0.76 - 0.48) / 0.3
    return min(0.93, 0.76 + (r - 0.8) * (0.93 - 0.76) / 0.2)


def _address_novelty_error(receiver_hits: int) -> float:
    """Higher when receiver address is rarely seen in dst set."""
    h = max(int(receiver_hits), 0)
    # hits=0 -> 1.0, hits=1 -> 0.5, hits=3 -> 0.25, ...
    return float(1.0 / (1.0 + h))


def _amount_scale_error(raw_tot: float, expected_raw: float) -> float:
    """Log-scale amount deviation; robust to very large raw units."""
    if raw_tot <= 0 or expected_raw <= 0:
        return 1.0
    return float(abs(np.log1p(raw_tot) - np.log1p(expected_raw)))


def _uncertainty_band(selected_err: float, second_err: float | None, candidate_count: int) -> tuple[str, float]:
    if candidate_count <= 1:
        return "high", 1.0
    if second_err is None:
        return "high", 1.0
    gap = max(float(second_err) - float(selected_err), 0.0)
    rel = gap / max(abs(float(selected_err)), 1e-6)
    if rel >= 1.0:
        return "low", float(min(rel, 5.0))
    if rel >= 0.3:
        return "medium", float(rel)
    return "high", float(rel)


def _narrative_template(info: dict[str, Any]) -> dict[str, Any]:
    selected = str(info.get("selected_dstTxHash") or "")
    conf = float(info.get("selected_confidence_calibrated") or 0.0)
    ub = str(info.get("uncertainty_band") or "")
    topk = info.get("topk_candidates") or []
    main_evidence: list[str] = []
    for c in topk[:2]:
        eb = c.get("evidence_breakdown") or {}
        ratio_err = eb.get("amount_ratio_error")
        delay_err = eb.get("delay_error")
        if ratio_err is not None and float(ratio_err) < 0.3:
            main_evidence.append("amount_alignment_good")
        if delay_err is not None and float(delay_err) < 0.4:
            main_evidence.append("time_delay_alignment_good")
    if not main_evidence and selected:
        main_evidence.append("best_overall_error_rank")
    counter = [str(x) for x in (info.get("counter_evidence") or []) if str(x)]
    review_next = "review_runner_up_candidate" if ub in ("high", "medium") else "routine_sampling_review"
    return {
        "what_happened": f"ETH src matched to BNB dst {selected}" if selected else "No confident BNB dst matched",
        "why_we_believe": sorted(set(main_evidence))[:3],
        "counter_evidence": counter[:3],
        "risk_propagation": "source_risk_to_destination_candidate",
        "review_next": review_next,
        "confidence_calibrated": conf,
        "uncertainty_band": ub,
    }


def greedy_pair_dst_hashes_with_evidence(
    src_all: pd.DataFrame,
    dst_all: pd.DataFrame,
    *,
    dst_window_sec: float = 86400.0,
    top_k_per_src: int = 200,
    first_batch_k: int = 100,
    second_batch_k: int = 100,
    ratio_by_eth_token: dict[str, float] | None = None,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
    median_delay_sec: float | None = None,
    delay_weight: float = 0.12,
    truth_dst_by_src: dict[str, str] | None = None,
    boost_truth_in_candidates: bool = False,
    edge_score_fn: EdgeScoreFn | None = None,
    greedy_assignment: str = "greedy",
    receiver_mode: str = "eth_from",
    bnb_bridge_address: str = "",
    confidence_temperature: float = 0.45,
    confidence_top_n: int = 50,
    address_novelty_weight: float = 0.08,
    amount_scale_weight: float = 0.06,
    min_evidence_level_for_confirmed: int = 4,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Return mapping ``src_txhash`` -> ``dst_txhash``.

    ``edge_score_fn`` can blend ABCT/TIR logits or extra penalties: it receives the row
    index, normalized dst hash, and the heuristic **base** error; return a scalar to **minimize**
    (same convention as internal errors).

    ``greedy_assignment``: ``"greedy"`` (sort all edges globally) or ``"hungarian"`` (min-cost
    matching on remaining rows after truth reservation).

    When ``boost_truth_in_candidates`` is True and ``truth_dst_by_src`` is set, performs
    chronological **truth-first reservation** (label eval only).

    ``receiver_mode``: ``eth_from`` filters BNB rows with ``to == ETH depositor`` (legacy).
    ``bnb_pick_per_candidate`` keeps all dst txs in the time window and, for each candidate
    ``hash``, uses :func:`pick_bnb_recipient` on that tx's rows to resolve the BNB payout
    address (Path A alignment). Requires ``bnb_bridge_address`` (Celer bridge on BNB).
    Also returns per-src explanation bundle with top-k candidates, selected rank,
    confidence (softmax over negative candidate error), and feature breakdown.
    """
    want_eth = {
        str(x).strip().lower()
        for x in src_all["args.asset_s"].tolist()
        if str(x).strip() and str(x).lower() not in ("nan", "")
    }
    want_bnb = {
        str(x).strip().lower()
        for x in dst_all["contractAddress"].tolist()
        if str(x).strip() and str(x).lower() not in ("nan", "", ZERO)
    }
    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb)

    dst_ts = pd.to_numeric(dst_all["timeStamp"], errors="coerce").fillna(0)
    dst_all = dst_all.copy()
    dst_recv_hits = dst_all["to"].map(norm_addr).value_counts().to_dict()

    mapping: dict[str, str] = {}
    used_src_idx: set[int] = set()
    used_dst_hash: set[str] = set()

    if boost_truth_in_candidates and truth_dst_by_src:
        chronological = sorted(
            range(len(src_all)),
            key=lambda i: safe_float(src_all.iloc[i].get("timestamp"), 0.0),
        )
        for i in chronological:
            row = src_all.iloc[i]
            txh = norm_addr(row.get("txhash", ""))
            ts = safe_float(row.get("timestamp"), 0.0)
            want = norm_addr(truth_dst_by_src.get(txh, ""))
            if not want:
                continue
            mask = (dst_ts > ts) & (dst_ts <= ts + dst_window_sec)
            sl = dst_all.loc[mask].copy()
            if sl.empty:
                sl = dst_all.copy()
            present = {norm_addr(h) for h in sl["hash"].astype(str)}
            if want in present and want not in used_dst_hash:
                mapping[txh] = want
                used_dst_hash.add(want)
                used_src_idx.add(i)

    recv_mode = (receiver_mode or "eth_from").strip().lower()
    bridge = norm_addr(bnb_bridge_address)

    max_total_k = max(1, int(top_k_per_src))
    phase1_k = max(1, min(int(first_batch_k), max_total_k))
    phase2_k = max(0, min(int(second_batch_k), max_total_k - phase1_k))
    total_k = phase1_k + phase2_k

    per_src_candidates: list[list[tuple[float, str]]] = []
    per_src_candidate_detail: list[list[dict[str, Any]]] = []

    for pos in range(len(src_all)):
        row = src_all.iloc[pos]
        eth_from = norm_addr(row.get("args.receiver", ""))
        recv = eth_from
        ts = safe_float(row.get("timestamp"), 0.0)
        eth_amt = safe_float(row.get("args.amount"), 0.0)
        eth_ca = str(row.get("args.asset_s", "") or "").strip().lower()
        asset_d = str(row.get("args.asset_d", "") or "").strip().lower()
        hs = _human_src(eth_amt, eth_ca, dec_e)

        mask = (dst_ts > ts) & (dst_ts <= ts + dst_window_sec)
        sl = dst_all.loc[mask].copy()
        if sl.empty:
            sl = dst_all.copy()

        if recv_mode != "bnb_pick_per_candidate":
            sl = sl[sl["to"].map(norm_addr) == recv]

        if sl.empty:
            per_src_candidates.append([])
            per_src_candidate_detail.append([])
            continue

        scored: list[tuple[float, str]] = []
        detailed: list[dict[str, Any]] = []
        ratio = None
        if ratio_by_eth_bnb_pair and eth_ca and asset_d:
            ratio = ratio_by_eth_bnb_pair.get((eth_ca, norm_addr(asset_d)))
        if ratio is None and ratio_by_eth_token and eth_ca:
            ratio = ratio_by_eth_token.get(eth_ca)
        dwin = _dynamic_window_sec(
            dst_window_sec,
            median_delay_sec=median_delay_sec,
            has_ratio=ratio is not None,
            has_token=bool(eth_ca and eth_ca != ZERO),
        )

        for h_raw, g in sl.groupby(sl["hash"].astype(str)):
            h = norm_addr(h_raw)
            recv_eff = recv
            recv_res_status = "resolved"
            recv_penalty = ""
            if recv_mode == "bnb_pick_per_candidate":
                recv_pick, _, _ = pick_bnb_recipient(g, eth_from, bridge)
                if not recv_pick:
                    # Soft-weak: unresolved pick — keep hash for flow-level UOT; use first ``to`` for tx-baseline err.
                    recv_res_status = "pick_unresolved_fallback"
                    recv_penalty = "pick_unresolved"
                    recv_eff = norm_addr(str(g.iloc[0].get("to", "") or "")) if not g.empty else ""
                    if not recv_eff:
                        continue
                else:
                    recv_eff = norm_addr(recv_pick)

            hd, _tmin = _human_dst_for_hash(g, recv_eff, dec_b)
            sub = g[g["to"].map(norm_addr) == recv_eff]
            raw_tot = safe_float(pd.to_numeric(sub["value"], errors="coerce").fillna(0).sum(), 0.0)
            tmin_dst = safe_float(pd.to_numeric(g["timeStamp"], errors="coerce").fillna(0).min(), 0.0)

            ratio_err = None
            human_err = None
            delay_err = 0.0
            amount_scale_err = 1.0
            if ratio is not None and eth_amt > 0 and raw_tot > 0:
                exp_raw = eth_amt * ratio
                ratio_err = abs(raw_tot - exp_raw) / max(exp_raw, 1e-30)
                amount_scale_err = _amount_scale_error(raw_tot, exp_raw)
                err = float(ratio_err)
            elif eth_amt > 0 and raw_tot > 0:
                if eth_ca and eth_ca != ZERO and eth_ca not in dec_e:
                    ratio_err = None
                    err = 1e6 + 2.0
                else:
                    # No ratio_map: compare raw wei units (native / known-decimal ERC20 legs).
                    ratio_err = abs(raw_tot - eth_amt) / max(eth_amt, 1e-30)
                    amount_scale_err = _amount_scale_error(raw_tot, eth_amt)
                    err = float(ratio_err)
            elif hs is not None and hs > 0 and hd is not None and hd > 0:
                human_err = abs(hd - hs) / max(hs, 1e-30)
                amount_scale_err = _amount_scale_error(hd, hs)
                err = float(human_err)
            else:
                err = 1e6 + (1.0 if (hd is not None and hd > 0) else 1e3)

            if median_delay_sec is not None and median_delay_sec > 0:
                gap = max(tmin_dst - ts, 0)
                terr = abs(gap - median_delay_sec) / max(median_delay_sec, 60.0)
                delay_err = float(terr)
                err = err + delay_weight * terr
            gap_sec = max(tmin_dst - ts, 0.0)
            if not _candidate_gate(
                err=float(err),
                ratio_err=ratio_err,
                human_err=human_err,
                delay_err=float(delay_err),
                dst_gap_sec=float(gap_sec),
                dynamic_window_sec=float(dwin),
            ):
                continue

            if edge_score_fn is not None:
                err = float(edge_score_fn(pos, h, float(err)))
            novelty_hits = int(dst_recv_hits.get(recv_eff, 0))
            novelty_err = _address_novelty_error(novelty_hits)
            err = float(
                err
                + float(address_novelty_weight) * novelty_err
                + float(amount_scale_weight) * float(amount_scale_err)
            )

            scored.append((err, h))
            detailed.append(
                {
                    "dstTxHash": h,
                    "base_error": float(err),
                    "amount_ratio_error": None if ratio_err is None else float(ratio_err),
                    "amount_human_error": None if human_err is None else float(human_err),
                    "delay_error": float(delay_err),
                    "eth_amount_raw": float(eth_amt),
                    "bnb_amount_raw": float(raw_tot),
                    "src_timestamp": float(ts),
                    "dst_timestamp": float(tmin_dst),
                    "resolved_receiver": str(recv_eff),
                    "receiver_resolution_status": recv_res_status,
                    "receipt_penalty_reason": recv_penalty,
                    "address_novelty_error": float(novelty_err),
                    "amount_scale_error": float(amount_scale_err),
                    "receiver_hit_count": int(novelty_hits),
                    "dst_gap_sec": float(gap_sec),
                    "dynamic_window_sec": float(dwin),
                    "evidence_breakdown": {
                        "amount_ratio_error": None if ratio_err is None else float(ratio_err),
                        "amount_human_error": None if human_err is None else float(human_err),
                        "delay_error": float(delay_err),
                        "address_novelty_error": float(novelty_err),
                        "amount_scale_error": float(amount_scale_err),
                    },
                }
            )

        scored.sort(key=lambda x: x[0])
        keep_hashes = [h for _e, h in scored[:total_k]]
        detail_by_hash = {str(d["dstTxHash"]): d for d in detailed}
        kept_detail = [detail_by_hash[h] for h in keep_hashes if h in detail_by_hash]
        per_src_candidates.append(scored[:total_k])
        per_src_candidate_detail.append(kept_detail)

        if (pos + 1) % 1000 == 0:
            logger.info(
                "Path B greedy: built candidates for %s/%s source rows",
                pos + 1,
                len(src_all),
            )

    phase1_candidates = [lst[:phase1_k] for lst in per_src_candidates]
    extra_m1, used_src_idx, used_dst_hash = _assign_with_strategy(
        per_src_candidates=phase1_candidates,
        used_src_idx=used_src_idx,
        used_dst_hash=used_dst_hash,
        src_all=src_all,
        greedy_assignment=greedy_assignment,
    )
    mapping.update(extra_m1)

    phase2_enabled = phase2_k > 0
    extra_m2: dict[str, str] = {}
    if phase2_enabled:
        phase2_candidates = [lst[:total_k] for lst in per_src_candidates]
        extra_m2, used_src_idx, used_dst_hash = _assign_with_strategy(
            per_src_candidates=phase2_candidates,
            used_src_idx=used_src_idx,
            used_dst_hash=used_dst_hash,
            src_all=src_all,
            greedy_assignment=greedy_assignment,
        )
        mapping.update(extra_m2)
    logger.info(
        "Path B candidate phases: phase1_k=%d phase2_k=%d phase1_hits=%d phase2_hits=%d unmatched=%d",
        phase1_k,
        phase2_k,
        len(extra_m1),
        len(extra_m2),
        max(0, len(src_all) - len(extra_m1) - len(extra_m2)),
    )

    for i in range(len(src_all)):
        txh = norm_addr(src_all.iloc[i].get("txhash", ""))
        if mapping.get(txh):
            continue
        lst = per_src_candidates[i]
        mapping[txh] = lst[0][1] if lst else ""

    evidence_by_src: dict[str, dict[str, Any]] = {}
    for i in range(len(src_all)):
        txh = norm_addr(src_all.iloc[i].get("txhash", ""))
        selected = mapping.get(txh, "")
        details = per_src_candidate_detail[i] if i < len(per_src_candidate_detail) else []
        selected_rank = -1
        errs: list[float] = []
        topk: list[dict[str, Any]] = []
        gate_reject_counts = {
            "time_window": 0,
            "ratio_error": 0,
            "human_error": 0,
            "delay_error": 0,
            "overall_error": 0,
        }
        for j, d in enumerate(details):
            e = float(d.get("base_error", 1e9))
            errs.append(e)
            conf_j = _softmax_confidence_from_errors(
                [float(x.get("base_error", 1e9)) for x in details],
                j,
                temperature=float(confidence_temperature),
            )
            item = dict(d)
            item["rank"] = j + 1
            item["candidate_confidence"] = float(conf_j)
            topk.append(item)
            if norm_addr(d.get("dstTxHash", "")) == norm_addr(selected):
                selected_rank = j
        eff_n = max(1, min(int(confidence_top_n), len(errs)))
        eff_errs = errs[:eff_n]
        eff_selected_rank = selected_rank if 0 <= selected_rank < eff_n else -1
        selected_conf = _softmax_confidence_from_errors(
            eff_errs,
            eff_selected_rank,
            temperature=float(confidence_temperature),
        )
        selected_conf_cal = _calibrate_confidence(selected_conf)
        lev_max = _max_evidence_level_for_dst_hash(dst_all, selected) if selected else None
        if lev_max is not None:
            selected_conf_cal = apply_evidence_confidence_cap(selected_conf_cal, lev_max)
            selected_conf = apply_evidence_confidence_cap(float(selected_conf), lev_max)
        competition_gap = 0.0
        second_err = None
        sel_err = None
        if eff_selected_rank >= 0 and eff_selected_rank < len(eff_errs):
            sel_err = eff_errs[eff_selected_rank]
            sorted_err = sorted(eff_errs)
            if len(sorted_err) > 1:
                second_err = sorted_err[1]
                competition_gap = float(second_err - sorted_err[0])
        ub, uscore = _uncertainty_band(
            float(sel_err if sel_err is not None else 1e9),
            None if second_err is None else float(second_err),
            len(topk),
        )
        counter_evidence: list[str] = []
        if selected_rank <= 1 and len(topk) > 1:
            counter_evidence.append("top2_competition_close")
        if second_err is not None and sel_err is not None and (second_err - sel_err) < 0.05 * max(abs(sel_err), 1.0):
            counter_evidence.append("small_error_gap_to_runnerup")
        if not topk:
            counter_evidence.append("no_candidate_in_window")
        phase1_hashes = {h for _err, h in (per_src_candidates[i][:phase1_k] if i < len(per_src_candidates) else [])}
        phase2_hashes = {h for _err, h in (per_src_candidates[i][:total_k] if i < len(per_src_candidates) else [])}
        if selected:
            if selected in phase1_hashes:
                candidate_phase = "phase1"
            elif selected in phase2_hashes:
                candidate_phase = "phase2"
            else:
                candidate_phase = "fallback"
        else:
            candidate_phase = "none"
        sel_rs = ""
        for d in details:
            if norm_addr(str(d.get("dstTxHash", ""))) == norm_addr(str(selected or "")):
                sel_rs = str(d.get("receiver_resolution_status") or "resolved")
                break
        weak_ev = False
        conf_eligible = True
        if lev_max is not None:
            weak_ev = lev_max < 2 or lev_max < int(min_evidence_level_for_confirmed)
            conf_eligible = lev_max >= int(min_evidence_level_for_confirmed)
        evidence_by_src[txh] = {
            "selected_dstTxHash": selected,
            "selected_rank": int(selected_rank + 1) if selected_rank >= 0 else 0,
            "selected_confidence": float(selected_conf),
            "selected_confidence_calibrated": float(selected_conf_cal),
            "uncertainty_band": ub,
            "uncertainty_score": float(uscore),
            "competition_gap": float(competition_gap),
            "counter_evidence": counter_evidence,
            "candidate_count": len(topk),
            "candidate_phase": candidate_phase,
            "candidate_phase1_count": int(min(phase1_k, len(topk))),
            "candidate_phase2_count": int(max(0, min(total_k, len(topk)) - min(phase1_k, len(topk)))),
            "confidence_temperature": float(confidence_temperature),
            "confidence_candidate_pool_size": int(eff_n),
            "best_error": None if sel_err is None else float(sel_err),
            "second_error": None if second_err is None else float(second_err),
            "relative_gap": None
            if (sel_err is None or second_err is None)
            else float(max(second_err - sel_err, 0.0) / max(abs(sel_err), 1e-9)),
            "gate_reject_counts": gate_reject_counts,
            "topk_candidates": topk,
            "evidence_level_max": lev_max,
            "weak_evidence": weak_ev,
            "confirmed_eligible": conf_eligible,
            "min_evidence_level_for_confirmed": int(min_evidence_level_for_confirmed),
            "selected_receiver_resolution_status": sel_rs or ("resolved" if selected else ""),
        }
        evidence_by_src[txh]["narrative"] = _narrative_template(evidence_by_src[txh])
        if ub == "high":
            evidence_by_src[txh]["uncertainty_reason"] = "close_competition_or_sparse_candidates"
        elif ub == "medium":
            evidence_by_src[txh]["uncertainty_reason"] = "moderate_error_gap"
        else:
            evidence_by_src[txh]["uncertainty_reason"] = "clear_winner_candidate"
    return mapping, evidence_by_src


def dataframe_from_greedy_map(
    src_all: pd.DataFrame,
    mapping: dict[str, str],
    evidence_by_src: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = []
    for i in range(len(src_all)):
        txh = norm_addr(src_all.iloc[i].get("txhash", ""))
        rec = {
            "srcnet": "ETH",
            "srcTxHash": txh,
            "dstnet": "BNB",
            "dstTxHash": mapping.get(txh, ""),
        }
        if evidence_by_src:
            info = evidence_by_src.get(txh) or {}
            rec["candidateRank"] = safe_int(info.get("selected_rank"), 0)
            rec["matchConfidence"] = float(info.get("selected_confidence") or 0.0)
            rec["matchConfidenceCalibrated"] = float(info.get("selected_confidence_calibrated") or 0.0)
            rec["uncertaintyBand"] = str(info.get("uncertainty_band") or "")
            rec["uncertaintyReason"] = str(info.get("uncertainty_reason") or "")
            topk = info.get("topk_candidates") or []
            rec["topKDstTxHash"] = ";".join(
                [str(x.get("dstTxHash", "")) for x in topk[:3] if str(x.get("dstTxHash", ""))]
            )
            rec["topKScore"] = ";".join(
                [f"{float(x.get('base_error', 0.0)):.6g}" for x in topk[:3]]
            )
            rec["counterEvidence"] = ";".join([str(x) for x in (info.get("counter_evidence") or [])[:3]])
        rows.append(rec)
    return pd.DataFrame(rows)


def split_path_b_pairs_by_confidence(
    pairs: pd.DataFrame,
    evidence_by_src: dict[str, dict[str, Any]] | None,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split Path B rows into accepted pairs (conf >= threshold) vs low-confidence best candidates."""
    empty_low = pd.DataFrame(
        columns=[
            "srcTxHash",
            "best_dstTxHash",
            "matchConfidenceCalibrated",
            "confidence_threshold",
            "match_status",
        ]
    )
    if pairs is None or pairs.empty:
        return pd.DataFrame(), empty_low.copy(), {
            "path_b_total_sources": 0,
            "accepted_pairs": 0,
            "low_confidence_best_candidates": 0,
            "unmatched_sources": 0,
            "confidence_threshold": float(threshold),
        }

    ev = evidence_by_src if isinstance(evidence_by_src, dict) else {}
    accepted_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    n_accepted = 0
    n_low = 0
    n_no_dst = 0

    for _, row in pairs.iterrows():
        src = norm_addr(str(row.get("srcTxHash", "") or ""))
        dst = norm_addr(str(row.get("dstTxHash", "") or ""))
        info = ev.get(src) or {}
        raw_c = info.get("selected_confidence_calibrated")
        if raw_c is not None:
            conf = float(raw_c)
        else:
            conf = float(row.get("matchConfidenceCalibrated", 0.0) or 0.0)
        has_dst = bool(dst)
        ok = has_dst and conf >= float(threshold)
        sel_rs = str((info.get("selected_receiver_resolution_status") or "")).strip()
        if sel_rs == "pick_unresolved_fallback":
            ok = False

        if ok:
            n_accepted += 1
            rec = row.to_dict()
            rec["match_status"] = "accepted"
            rec["confidence_accept_threshold"] = float(threshold)
            accepted_rows.append(rec)
        elif has_dst:
            n_low += 1
            low_rows.append(
                {
                    "srcTxHash": src,
                    "best_dstTxHash": dst,
                    "matchConfidenceCalibrated": conf,
                    "confidence_threshold": float(threshold),
                    "match_status": "below_confidence_threshold",
                }
            )
        else:
            n_no_dst += 1

    if not accepted_rows:
        accepted_df = pairs.iloc[0:0].copy()
        accepted_df["match_status"] = pd.Series(dtype=str)
        accepted_df["confidence_accept_threshold"] = pd.Series(dtype=float)
    else:
        accepted_df = pd.DataFrame(accepted_rows)
    low_df = pd.DataFrame(low_rows) if low_rows else empty_low.copy()
    n_total = int(len(pairs))
    summary = {
        "path_b_total_sources": n_total,
        "accepted_pairs": int(n_accepted),
        "low_confidence_best_candidates": int(n_low),
        "unmatched_sources": int(n_total - n_accepted),
        "unmatched_no_dst": int(n_no_dst),
        "confidence_threshold": float(threshold),
    }
    return accepted_df, low_df, summary
