"""Load flow-segment CSV exports into RC-UOT internal flow dicts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cross.utils.safe_cast import safe_float, safe_int


def _split_pipe(s: str) -> list[str]:
    t = (s or "").strip()
    if not t:
        return []
    return [x.strip() for x in t.split("|") if x.strip()]


def flows_from_segment_export_csv(path: Path, *, chain: str) -> list[dict[str, Any]]:
    """Parse ``flow_segments_{eth|bnb}.csv`` produced by the Celer label pipeline or Path B export."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        txs = _split_pipe(str(r.get("tx_hashes") or ""))
        addrs = _split_pipe(str(r.get("address_set") or ""))
        usd = safe_float(r.get("usd_amount_sum"), 0.0)
        if usd <= 0:
            usd = safe_float(r.get("amount_usd"), 0.0)
        aml_mean = safe_float(r.get("aml_score_mean"), 0.0)
        aml_max = safe_float(r.get("aml_score_max"), 0.0)
        if aml_mean <= 0 and aml_max > 0:
            aml_mean = aml_max
        ev_mean = safe_float(r.get("evidence_quality_mean"), 0.65)
        tok_sym = str(r.get("token_symbols") or r.get("asset_group") or "unknown")
        route_type = f"same_asset_bridge:{str(r.get('asset_group') or '')}"
        ev_level = int(min(4, max(1, round(ev_mean * 4))))
        out.append(
            {
                "flow_id": str(r.get("flow_id") or ""),
                "chain": str(r.get("chain") or chain).upper(),
                "tx_hashes": txs,
                "address_set": addrs,
                "amount_usd": float(usd),
                "start_time": safe_float(r.get("start_time"), 0.0),
                "end_time": safe_float(r.get("end_time"), 0.0),
                "token_symbol": tok_sym[:120],
                "route_type": route_type[:200],
                "route_id": str(r.get("route_id") or ""),
                "asset_group": str(r.get("asset_group") or ""),
                "aml_score": float(aml_mean / 100.0) if aml_mean > 1.0 else float(aml_mean),
                "aml_risk_score_raw": float(aml_mean) if aml_mean > 1.0 else float(aml_mean * 100.0),
                "evidence_quality_score": float(min(1.0, max(0.05, ev_mean))),
                "evidence_level": ev_level,
                "evidence_levels": str(r.get("evidence_quality_mean") or ""),
                "price_snapshot_ok": True,
                "graph_embedding": None,
                "tx_count": safe_int(r.get("tx_count"), len(txs)),
                "raw_amount_sum": safe_float(r.get("raw_amount_sum"), 0.0),
                "human_amount_sum": safe_float(r.get("human_amount_sum"), 0.0),
                "bridge": str(r.get("bridge") or "celer"),
            }
        )
    return out
