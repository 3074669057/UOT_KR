"""RC paper flow segments: token + route + 30m window + address overlap (not tx-level UOT)."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pandas as pd

from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float, safe_int

if TYPE_CHECKING:
    from cross.domain.path_b.route_registry import RouteRegistry

ZERO = "0x0000000000000000000000000000000000000000"


def _safe_log_index(v: Any) -> int:
    return safe_int(v, -1)


def _row_addrs_eth(r: pd.Series, bridge: str) -> set[str]:
    out: set[str] = set()
    for k in ("from", "to", "args.receiver"):
        a = norm_addr(str(r.get(k, "") or ""))
        if a:
            out.add(a)
    br = norm_addr(bridge or "")
    if br:
        out.add(br)
    return out


def _row_addrs_bnb(r: pd.Series, bridge: str) -> set[str]:
    out: set[str] = set()
    for k in ("from", "to"):
        a = norm_addr(str(r.get(k, "") or ""))
        if a:
            out.add(a)
    br = norm_addr(bridge or "")
    if br:
        out.add(br)
    return out


def _eth_route_for_token(rr: "RouteRegistry | None", token: str) -> tuple[str, str]:
    if not rr or not token or token == ZERO:
        return "", "unknown"
    tok = norm_addr(token)
    for route in getattr(rr, "resolved_routes", []) or []:
        if norm_addr(route.src) == tok:
            return str(route.route_id), str(route.route_type or "")
    return "", "unknown"


def _bnb_route_for_contract(rr: "RouteRegistry | None", ca: str) -> tuple[str, str]:
    if not rr or not ca or ca == ZERO:
        return "", "unknown"
    c = norm_addr(ca)
    for route in getattr(rr, "resolved_routes", []) or []:
        if norm_addr(route.dst) == c:
            return str(route.route_id), str(route.route_type or "")
    return "", "unknown"


def _merge_span_ok(times: list[float], max_span_sec: float) -> bool:
    if len(times) <= 1:
        return True
    return (max(times) - min(times)) <= float(max_span_sec)


def _overlap(a: set[str], b: set[str]) -> bool:
    return bool(a and b and (a & b))


def build_rc_eth_flow_segments(
    src_all: pd.DataFrame,
    *,
    route_registry: "RouteRegistry | None" = None,
    bridge_addr: str = "",
    max_time_gap_sec: float = 1800.0,
    bridge_label: str = "celer",
) -> list[dict[str, Any]]:
    """Cluster AML src txs: same ``token_contract`` + ``route_id``, <=30m span, address overlap."""
    if src_all is None or src_all.empty:
        return []
    bridge_n = norm_addr(bridge_addr or "")
    rows: list[dict[str, Any]] = []
    for _, r in src_all.iterrows():
        txh = norm_addr(str(r.get("txhash", "") or ""))
        if not txh:
            continue
        ts = safe_float(r.get("timestamp"), 0.0)
        tok = str(r.get("args.asset_s", "") or "").strip().lower() or ""
        tok_n = norm_addr(tok) if tok else ""
        rid, rtype = _eth_route_for_token(route_registry, tok_n)
        rec = norm_addr(str(r.get("args.receiver", "") or ""))
        rows.append(
            {
                "txhash": txh,
                "timestamp": ts,
                "token_contract": tok_n,
                "route_id": rid,
                "route_type": rtype,
                "receiver": rec,
                "addrs": _row_addrs_eth(r, bridge_n),
                "raw_row": r,
            }
        )
    rows.sort(key=lambda x: (x["token_contract"], x["route_id"], x["timestamp"]))

    segments: list[dict[str, Any]] = []
    for row in rows:
        placed = False
        for seg in segments:
            if seg["token_contract"] != row["token_contract"] or seg["route_id"] != row["route_id"]:
                continue
            cand_times = list(seg["_times"]) + [row["timestamp"]]
            if not _merge_span_ok(cand_times, max_time_gap_sec):
                continue
            if not _overlap(seg["_addrs"], row["addrs"]):
                continue
            seg["tx_hashes"].append(row["txhash"])
            seg["_times"].append(row["timestamp"])
            seg["_addrs"] |= row["addrs"]
            placed = True
            break
        if not placed:
            fid = f"eth_flow_{uuid.uuid4().hex[:12]}"
            segments.append(
                {
                    "flow_id": fid,
                    "chain": "ETH",
                    "source_type": "rc_segment",
                    "tx_hashes": [row["txhash"]],
                    "_times": [row["timestamp"]],
                    "_addrs": set(row["addrs"]),
                    "token_contract": row["token_contract"],
                    "route_id": row["route_id"],
                    "route_type": row["route_type"],
                    "bridge": bridge_label,
                    "address_set": set(),
                }
            )

    out: list[dict[str, Any]] = []
    for seg in segments:
        ts_vals = [t for t in seg["_times"] if t > 0]
        t0 = min(ts_vals) if ts_vals else 0.0
        t1 = max(ts_vals) if ts_vals else 0.0
        txs = sorted(set(seg["tx_hashes"]))
        addrs = sorted({x for x in seg["_addrs"] if x})
        out.append(
            {
                "flow_id": seg["flow_id"],
                "chain": "ETH",
                "source_type": seg.get("source_type", "rc_segment"),
                "tx_hashes": txs,
                "address_set": addrs,
                "token_contract": seg["token_contract"],
                "route_id": seg["route_id"],
                "route_type": seg.get("route_type", ""),
                "asset_group": seg["token_contract"] or "native_eth",
                "start_time": t0,
                "end_time": t1 if t1 >= t0 else t0,
                "token_symbol": (seg["token_contract"][:12] + "..") if len(seg["token_contract"] or "") > 14 else (seg["token_contract"] or "ETH"),
                "bridge": seg["bridge"],
                "amount_usd": 0.0,
                "amount_token": 0.0,
                "aml_score": 0.0,
                "aml_risk_level": "",
                "aml_rule_hits": "",
                "graph_embedding": None,
                "risk_features": {},
                "tx_count": len(txs),
                "address_count": max(1, len(addrs)),
            }
        )
    return out


def build_rc_bnb_flow_segments(
    dst_norm: pd.DataFrame,
    *,
    route_registry: "RouteRegistry | None" = None,
    bridge_addr: str = "",
    max_time_gap_sec: float = 1800.0,
    bridge_label: str = "celer",
) -> list[dict[str, Any]]:
    """Cluster BNB transfer rows: same ``token_contract`` + ``route_id``, <=30m span, address overlap."""
    if dst_norm is None or dst_norm.empty:
        return []
    bridge_n = norm_addr(bridge_addr or "")
    rows: list[dict[str, Any]] = []
    for _, r in dst_norm.iterrows():
        h = norm_addr(str(r.get("hash", "") or ""))
        if not h:
            continue
        ts = safe_float(r.get("timeStamp"), 0.0)
        ca = str(r.get("contractAddress", "") or "").strip().lower() or ""
        ca_n = norm_addr(ca) if ca else ""
        rid, rtype = _bnb_route_for_contract(route_registry, ca_n)
        ev = r.get("evidence_level", "")
        ev_s = str(ev) if ev is not None else ""
        rows.append(
            {
                "txhash": h,
                "timestamp": ts,
                "token_contract": ca_n,
                "route_id": rid,
                "route_type": rtype,
                "receiver": norm_addr(str(r.get("to", "") or "")),
                "addrs": _row_addrs_bnb(r, bridge_n),
                "evidence_level_str": ev_s,
                "log_index": _safe_log_index(r.get("log_index")),
                "bridge_contract_hit": bool(r.get("bridge_contract_hit", False)),
            }
        )
    rows.sort(key=lambda x: (x["token_contract"], x["route_id"], x["timestamp"], x["txhash"]))

    segments: list[dict[str, Any]] = []
    for row in rows:
        placed = False
        for seg in segments:
            if seg["token_contract"] != row["token_contract"] or seg["route_id"] != row["route_id"]:
                continue
            cand_times = list(seg["_times"]) + [row["timestamp"]]
            if not _merge_span_ok(cand_times, max_time_gap_sec):
                continue
            if not _overlap(seg["_addrs"], row["addrs"]):
                continue
            seg["tx_hashes"].append(row["txhash"])
            seg["_times"].append(row["timestamp"])
            seg["_addrs"] |= row["addrs"]
            if row["evidence_level_str"]:
                seg["_ev_levels"].append(row["evidence_level_str"])
            seg.setdefault("_bridge_hits", []).append(bool(row.get("bridge_contract_hit")))
            placed = True
            break
        if not placed:
            fid = f"bnb_flow_{uuid.uuid4().hex[:12]}"
            segments.append(
                {
                    "flow_id": fid,
                    "chain": "BNB",
                    "source_type": "rc_segment",
                    "tx_hashes": [row["txhash"]],
                    "_times": [row["timestamp"]],
                    "_addrs": set(row["addrs"]),
                    "token_contract": row["token_contract"],
                    "route_id": row["route_id"],
                    "route_type": row["route_type"],
                    "bridge": bridge_label,
                    "_ev_levels": [row["evidence_level_str"]] if row["evidence_level_str"] else [],
                    "_bridge_hits": [bool(row.get("bridge_contract_hit"))],
                }
            )

    out: list[dict[str, Any]] = []
    for seg in segments:
        ts_vals = [t for t in seg["_times"] if t > 0]
        t0 = min(ts_vals) if ts_vals else 0.0
        t1 = max(ts_vals) if ts_vals else 0.0
        txs = sorted(set(seg["tx_hashes"]))
        addrs = sorted({x for x in seg["_addrs"] if x})
        evs = sorted({e for e in seg.get("_ev_levels", []) if e})
        bhits = seg.get("_bridge_hits") or []
        bridge_hit = any(bool(x) for x in bhits)
        out.append(
            {
                "flow_id": seg["flow_id"],
                "chain": "BNB",
                "source_type": seg.get("source_type", "rc_segment"),
                "tx_hashes": txs,
                "address_set": addrs,
                "token_contract": seg["token_contract"],
                "route_id": seg["route_id"],
                "route_type": seg.get("route_type", ""),
                "asset_group": seg["token_contract"] or "native_bnb",
                "start_time": t0,
                "end_time": t1 if t1 >= t0 else t0,
                "token_symbol": (seg["token_contract"][:12] + "..") if len(seg["token_contract"] or "") > 14 else (seg["token_contract"] or "BNB"),
                "bridge": seg["bridge"],
                "amount_usd": 0.0,
                "amount_token": 0.0,
                "aml_score": 0.0,
                "evidence_levels": ",".join(evs) if evs else "token_transfer_log",
                "evidence_quality_score": 0.75,
                "graph_embedding": None,
                "risk_features": {},
                "tx_count": len(txs),
                "address_count": max(1, len(addrs)),
                "bridge_contract_hit": bridge_hit,
            }
        )
    return out


__all__ = ["build_rc_bnb_flow_segments", "build_rc_eth_flow_segments"]
