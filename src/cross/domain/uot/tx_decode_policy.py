"""Tx-level decoding policies within a matched target flow (no label oracle in production)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cross.shared.normalize import norm_addr
from cross.shared.transfers import parse_transfer_value
from cross.utils.safe_cast import safe_float

TX_DECODE_POLICIES: frozenset[str] = frozenset(
    {
        "legacy",
        "amount_nearest_positive_delay",
        "choose_earliest_positive_delay_tx",
        "choose_nearest_positive_delay_tx",
        "choose_amount_nearest_positive_delay_tx",
        "choose_latest_tx",
        "choose_label_oracle_if_in_flow",
    }
)

DEFAULT_TX_DECODE_POLICY = "legacy"

# Alias for production CLI
PRODUCTION_TX_DECODE_POLICY = "amount_nearest_positive_delay"


def _dst_flow_tx_rows(dst_flow: dict[str, Any], dst_norm: pd.DataFrame) -> list[dict[str, Any]]:
    hashes = [norm_addr(h) for h in (dst_flow.get("tx_hashes") or []) if str(h).strip()]
    if not hashes:
        return []
    if dst_norm is None or dst_norm.empty or "hash" not in dst_norm.columns:
        return [{"hash": h, "timeStamp": 0.0, "value": 0.0} for h in hashes]
    sub = dst_norm[dst_norm["hash"].astype(str).map(norm_addr).isin(set(hashes))]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for h in hashes:
        if h in seen:
            continue
        part = sub[sub["hash"].astype(str).map(norm_addr) == h]
        if part.empty:
            rows.append({"hash": h, "timeStamp": 0.0, "value": 0.0})
        else:
            ts = safe_float(pd.to_numeric(part["timeStamp"], errors="coerce").fillna(0).min(), 0.0)
            vals = part["value"].map(parse_transfer_value).astype(float)
            rows.append({"hash": h, "timeStamp": ts, "value": float(vals.sum())})
        seen.add(h)
    return rows


def pick_dst_tx_in_flow(
    src_tx: str,
    src_ts: float,
    src_human_amt: float,
    dst_flow: dict[str, Any],
    dst_norm: pd.DataFrame,
    *,
    policy: str = DEFAULT_TX_DECODE_POLICY,
    gt_dst_tx: str | None = None,
) -> tuple[str, str, bool]:
    """Return (dst_tx_hash, effective_policy, used_positive_delay_fallback)."""
    policy = str(policy or DEFAULT_TX_DECODE_POLICY).strip()
    rows = _dst_flow_tx_rows(dst_flow, dst_norm)
    if not rows:
        return "", policy, False
    if len(rows) == 1:
        return rows[0]["hash"], "single_tx_in_flow", False

    gt_n = norm_addr(gt_dst_tx) if gt_dst_tx else ""

    if policy == "choose_label_oracle_if_in_flow" and gt_n:
        for r in rows:
            if r["hash"] == gt_n:
                return gt_n, policy, False

    positive = [r for r in rows if float(r["timeStamp"]) >= float(src_ts)]
    used_fallback = False

    if policy in ("amount_nearest_positive_delay", "choose_amount_nearest_positive_delay_tx"):
        pool = positive if positive else rows
        used_fallback = not bool(positive)
        best = min(
            pool,
            key=lambda r: (
                abs(float(r["value"]) - float(src_human_amt)) / max(float(src_human_amt), float(r["value"]), 1e-9),
                float(r["timeStamp"]) - float(src_ts),
            ),
        )
        return best["hash"], policy, used_fallback

    if policy == "choose_earliest_positive_delay_tx":
        pool = positive if positive else rows
        used_fallback = not bool(positive)
        best = min(pool, key=lambda r: (float(r["timeStamp"]), r["hash"]))
        return best["hash"], policy, used_fallback

    if policy == "choose_nearest_positive_delay_tx":
        pool = positive if positive else rows
        used_fallback = not bool(positive)
        best = min(pool, key=lambda r: (float(r["timeStamp"]) - float(src_ts), r["hash"]))
        return best["hash"], policy, used_fallback

    if policy == "choose_latest_tx":
        best = max(rows, key=lambda r: (float(r["timeStamp"]), r["hash"]))
        return best["hash"], policy, False

    # legacy: amount nearest with large penalty for ts < src_ts
    best_h = ""
    best_score = float("inf")
    for r in rows:
        ts = float(r["timeStamp"])
        human = float(r["value"])
        delay_pen = 0.0 if ts >= float(src_ts) else 1e6
        score = delay_pen + abs(human - float(src_human_amt)) / max(float(src_human_amt), human, 1e-9) + abs(ts - float(src_ts)) / 86400.0
        if score < best_score:
            best_score = score
            best_h = r["hash"]
    return best_h or rows[0]["hash"], "legacy", False


def infer_legacy_selection_rule(
    src_ts: float,
    src_human_amt: float,
    dst_flow: dict[str, Any],
    dst_norm: pd.DataFrame,
    picked: str,
) -> str:
    """Heuristic label for which rule legacy pick resembles."""
    rows = _dst_flow_tx_rows(dst_flow, dst_norm)
    if not rows:
        return "unknown"
    if len(rows) == 1:
        return "single_tx_in_flow"
    picked = norm_addr(picked)
    positive = [r for r in rows if float(r["timeStamp"]) >= float(src_ts)]

    if positive:
        by_amt = min(positive, key=lambda r: abs(float(r["value"]) - float(src_human_amt)))
        if by_amt["hash"] == picked:
            return "amount_nearest_among_positive_delay"
        by_earliest = min(positive, key=lambda r: float(r["timeStamp"]))
        if by_earliest["hash"] == picked:
            return "earliest_tx"
        by_nearest_delay = min(positive, key=lambda r: float(r["timeStamp"]) - float(src_ts))
        if by_nearest_delay["hash"] == picked:
            return "nearest_positive_delay"
    else:
        by_amt = min(rows, key=lambda r: abs(float(r["value"]) - float(src_human_amt)))
        if by_amt["hash"] == picked:
            return "amount_nearest_no_positive_delay_candidate"
        by_earliest = min(rows, key=lambda r: float(r["timeStamp"]))
        if by_earliest["hash"] == picked:
            return "earliest_tx"
        by_latest = max(rows, key=lambda r: float(r["timeStamp"]))
        if by_latest["hash"] == picked:
            return "latest_tx"

    first_h = rows[0]["hash"]
    if picked == first_h:
        return "first_tx"
    return "unknown"


def _tx_to_flow_index(source_flows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            out[norm_addr(str(txh))] = i
    return out


def derive_tx_pairs(
    p: np.ndarray,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    *,
    policy: str = DEFAULT_TX_DECODE_POLICY,
    label_truth: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Row-argmax flow match, then pick dst tx per src tx under ``policy``."""
    from cross.domain.uot.decode_transport import row_entropies

    p = np.asarray(p, dtype=float)
    mapping: dict[str, str] = {}
    meta: dict[str, dict[str, Any]] = {}
    tx_to_i = _tx_to_flow_index(source_flows)
    ents = row_entropies(p) if p.size else np.array([])
    truth = label_truth or {}

    for _, r in src_all.iterrows():
        txh = norm_addr(r.get("txhash", r.get("txHash", "")))
        if not txh:
            continue
        s_ts = safe_float(r.get("timestamp"), 0.0)
        s_amt = safe_float(r.get("args.amount"), 0.0)

        empty_meta = {
            "flow_i": -1,
            "flow_j": -1,
            "transport_mass": 0.0,
            "source_share": 0.0,
            "target_share": 0.0,
            "row_entropy": 0.0,
            "tx_decode_policy": policy,
            "tx_decode_fallback": False,
        }
        if not source_flows or not target_flows or p.size == 0:
            mapping[txh] = ""
            meta[txh] = dict(empty_meta)
            continue

        i = int(tx_to_i.get(txh, -1))
        if i < 0 or i >= p.shape[0]:
            mapping[txh] = ""
            meta[txh] = dict(empty_meta)
            continue

        row = p[i]
        j = int(np.argmax(row)) if row.size else -1
        mass = float(row[j]) if j >= 0 and j < row.size else 0.0
        row_sum = float(row.sum()) + 1e-12
        share = mass / row_sum
        col_sum = float(p[:, j].sum()) + 1e-12 if j >= 0 and j < p.shape[1] else 1e-12
        tgt_share = mass / col_sum
        dst_f = target_flows[j] if 0 <= j < len(target_flows) else {}
        gt_dst = truth.get(txh)
        dst_pick, eff_policy, used_fallback = pick_dst_tx_in_flow(
            txh, s_ts, s_amt, dst_f, dst_norm, policy=policy, gt_dst_tx=gt_dst
        )
        mapping[txh] = dst_pick
        meta[txh] = {
            **empty_meta,
            "flow_i": i,
            "flow_j": j,
            "transport_mass": mass,
            "source_share": share,
            "target_share": tgt_share,
            "row_entropy": float(ents[i]) if i < len(ents) else 0.0,
            "source_flow_id": source_flows[i].get("flow_id") if i < len(source_flows) else "",
            "target_flow_id": dst_f.get("flow_id") if dst_f else "",
            "tx_decode_policy": eff_policy,
            "tx_decode_fallback": used_fallback,
        }
    return mapping, meta
