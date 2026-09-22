"""Aggregate raw transactions into flow segments for UOT."""
from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

import pandas as pd

from cross.shared.normalize import norm_addr
from cross.shared.transfers import parse_transfer_value
from cross.utils.safe_cast import safe_float, safe_int

ZERO = "0x0000000000000000000000000000000000000000"


def _token_symbol_from_ca(ca: str) -> str:
    c = (ca or "").strip().lower()
    if not c or c == ZERO:
        return "ETH"
    return c[:10] + ".." if len(c) > 12 else c


def _time_bucket(ts: float, bucket_sec: int) -> int:
    t = safe_float(ts, 0.0)
    if t <= 0:
        return 0
    return int(t // max(safe_int(bucket_sec, 1), 1))


def _new_flow(
    *,
    flow_id: str,
    chain: str,
    tx_hashes: list[str],
    address_set: set[str],
    start_time: float,
    end_time: float,
    token_symbol: str,
    bridge: str,
) -> dict[str, Any]:
    return {
        "flow_id": flow_id,
        "chain": chain,
        "tx_hashes": tx_hashes,
        "address_set": sorted(address_set),
        "amount_usd": 0.0,
        "amount_token": 0.0,
        "token_symbol": token_symbol,
        "start_time": start_time,
        "end_time": end_time,
        "bridge": bridge,
        "aml_score": 0.0,
        "address_count": len(address_set),
        "tx_count": len(tx_hashes),
        "graph_embedding": None,
        "risk_features": {},
        "candidate_tags": [],
    }


def build_eth_flow_segments(
    src_all: pd.DataFrame,
    *,
    flow_mode: str = "segment",
    time_bucket_sec: int = 600,
    bridge: str = "celer",
) -> list[dict[str, Any]]:
    """Build ETH-side flow segments from Path B ``src_all`` rows (one row per src tx)."""
    mode = (flow_mode or "segment").strip().lower()
    if src_all is None or src_all.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in src_all.iterrows():
        txh = norm_addr(r.get("txhash", ""))
        if not txh:
            continue
        recv = norm_addr(r.get("args.receiver", ""))
        ts = safe_float(r.get("timestamp"), 0.0)
        asset = str(r.get("args.asset_s", "") or "").strip().lower()
        rows.append(
            {
                "txhash": txh,
                "receiver": recv,
                "timestamp": ts,
                "asset_s": asset,
                "token_symbol": _token_symbol_from_ca(asset),
                "time_bucket": _time_bucket(ts, time_bucket_sec),
                "raw_row": r,
            }
        )

    if mode == "tx":
        out: list[dict[str, Any]] = []
        for row in rows:
            fid = f"eth_flow_{uuid.uuid4().hex[:12]}"
            addr = {row["receiver"]} if row["receiver"] else set()
            f = _new_flow(
                flow_id=fid,
                chain="ETH",
                tx_hashes=[row["txhash"]],
                address_set=addr or {""},
                start_time=row["timestamp"] or 0.0,
                end_time=row["timestamp"] or 0.0,
                token_symbol=row["token_symbol"],
                bridge=bridge,
            )
            f["address_set"] = sorted({x for x in f["address_set"] if x})
            f["address_count"] = len(f["address_set"])
            out.append(f)
        return out

    # segment: (receiver, time_bucket, token_symbol)
    buckets: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["receiver"], row["time_bucket"], row["token_symbol"])
        buckets[key].append(row)

    out = []
    for (_recv, _tb, tok), members in buckets.items():
        fid = f"eth_flow_{uuid.uuid4().hex[:12]}"
        txs = sorted({m["txhash"] for m in members})
        addrs: set[str] = set()
        for m in members:
            if m["receiver"]:
                addrs.add(m["receiver"])
        ts_vals = [m["timestamp"] for m in members if m["timestamp"] > 0]
        t0 = min(ts_vals) if ts_vals else 0.0
        t1 = max(ts_vals) if ts_vals else 0.0
        f = _new_flow(
            flow_id=fid,
            chain="ETH",
            tx_hashes=txs,
            address_set=addrs or {""},
            start_time=t0,
            end_time=t1 if t1 >= t0 else t0,
            token_symbol=tok,
            bridge=bridge,
        )
        f["address_set"] = sorted({x for x in f["address_set"] if x})
        f["address_count"] = max(1, len(f["address_set"]))
        f["tx_count"] = len(txs)
        out.append(f)
    return out


def build_bnb_flow_segments(
    dst_norm: pd.DataFrame,
    *,
    flow_mode: str = "segment",
    time_bucket_sec: int = 600,
    bridge: str = "celer",
) -> list[dict[str, Any]]:
    """Build BNB-side flow segments from normalized BNB transfer rows (``bnb_df_to_dst_txs``)."""
    mode = (flow_mode or "segment").strip().lower()
    if dst_norm is None or dst_norm.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in dst_norm.iterrows():
        h = norm_addr(r.get("hash", ""))
        if not h:
            continue
        to_a = norm_addr(r.get("to", ""))
        ts = safe_float(r.get("timeStamp"), 0.0)
        ca = str(r.get("contractAddress", "") or "").strip().lower()
        rows.append(
            {
                "txhash": h,
                "receiver": to_a,
                "timestamp": ts,
                "contract": ca,
                "token_symbol": _token_symbol_from_ca(ca),
                "time_bucket": _time_bucket(ts, time_bucket_sec),
            }
        )

    if mode == "tx":
        out: list[dict[str, Any]] = []
        for row in rows:
            fid = f"bnb_flow_{uuid.uuid4().hex[:12]}"
            f = _new_flow(
                flow_id=fid,
                chain="BNB",
                tx_hashes=[row["txhash"]],
                address_set={row["receiver"]} if row["receiver"] else set(),
                start_time=row["timestamp"] or 0.0,
                end_time=row["timestamp"] or 0.0,
                token_symbol=row["token_symbol"],
                bridge=bridge,
            )
            f["address_set"] = sorted({x for x in f["address_set"] if x})
            f["address_count"] = max(1, len(f["address_set"]))
            out.append(f)
        return out

    buckets: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["receiver"], row["time_bucket"], row["token_symbol"])
        buckets[key].append(row)

    out = []
    for (_recv, _tb, tok), members in buckets.items():
        fid = f"bnb_flow_{uuid.uuid4().hex[:12]}"
        txs = sorted({m["txhash"] for m in members})
        addrs = {m["receiver"] for m in members if m["receiver"]}
        ts_vals = [m["timestamp"] for m in members if m["timestamp"] > 0]
        t0 = min(ts_vals) if ts_vals else 0.0
        t1 = max(ts_vals) if ts_vals else 0.0
        f = _new_flow(
            flow_id=fid,
            chain="BNB",
            tx_hashes=txs,
            address_set=addrs or {""},
            start_time=t0,
            end_time=t1 if t1 >= t0 else t0,
            token_symbol=tok,
            bridge=bridge,
        )
        f["address_set"] = sorted({x for x in f["address_set"] if x})
        f["address_count"] = max(1, len(f["address_set"]))
        f["tx_count"] = len(txs)
        out.append(f)
    return out


def attach_eth_amounts_from_src_all(flows: list[dict[str, Any]], src_all: pd.DataFrame) -> None:
    """Fill per-tx raw amounts on flows for downstream ``enrich_flow_segments`` (human amounts)."""
    by_tx: dict[str, dict] = {}
    for _, r in src_all.iterrows():
        h = norm_addr(r.get("txhash", ""))
        if h:
            by_tx[h] = {
                "args.amount": safe_float(r.get("args.amount"), 0.0),
                "args.asset_s": str(r.get("args.asset_s", "") or "").strip().lower(),
                "timestamp": safe_float(r.get("timestamp"), 0.0),
            }
    for f in flows:
        f["_per_tx"] = []
        for txh in f.get("tx_hashes") or []:
            f["_per_tx"].append({"txhash": txh, **(by_tx.get(txh) or {"args.amount": 0.0, "args.asset_s": "", "timestamp": 0.0})})


def attach_bnb_amounts_from_dst(flows: list[dict[str, Any]], dst_norm: pd.DataFrame) -> None:
    """Attach per-tx value + contract from dst_norm for human-amount conversion."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    ev_col = "evidence_level" if dst_norm is not None and "evidence_level" in dst_norm.columns else None
    for _, r in dst_norm.iterrows():
        h = norm_addr(r.get("hash", ""))
        if not h:
            continue
        ev_raw = r.get("evidence_level", "")
        ev_str = str(ev_raw) if isinstance(ev_raw, str) else ""
        row = {
            "value": float(parse_transfer_value(r.get("value", 0))),
            "contractAddress": str(r.get("contractAddress", "") or "").strip().lower(),
            "timeStamp": safe_float(r.get("timeStamp"), 0.0),
            "to": norm_addr(r.get("to", "")),
            "evidence_level_str": ev_str or str(r.get("evidence_level", "") or ""),
        }
        if ev_col:
            row["evidence_level"] = safe_int(r.get("evidence_level"), 0)
        by_hash[h].append(row)
    for f in flows:
        f["_per_tx"] = []
        for txh in f.get("tx_hashes") or []:
            parts = by_hash.get(txh, [])
            if not parts:
                f["_per_tx"].append({"txhash": txh, "value": 0.0, "contractAddress": "", "timeStamp": 0.0})
            else:
                # aggregate same-hash lines into one pseudo-row for amount sum
                tot = sum(p["value"] for p in parts)
                ca = next((p["contractAddress"] for p in parts if p["contractAddress"]), "")
                ts = min(p["timeStamp"] for p in parts)
                to_a = parts[0].get("to", "")
                pt = {"txhash": txh, "value": tot, "contractAddress": ca, "timeStamp": ts, "to": to_a}
                if ev_col:
                    pt["evidence_level"] = max(safe_int(p.get("evidence_level"), 0) for p in parts)
                evs = sorted({str(p.get("evidence_level_str") or "").strip() for p in parts if str(p.get("evidence_level_str") or "").strip()})
                if evs:
                    pt["evidence_level_str"] = ",".join(evs)
                f["_per_tx"].append(pt)
