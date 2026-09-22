"""Minimal Celer CSV → ABCT-compatible ``Tx`` JSON (query/target IR interchange shape).

Uses the same structural idea as common cross-chain tracer datasets (``input.func``, ``input.param``,
``event_logs``). Does **not** fetch full RPC receipts — only fields present in ``cun`` / ``qu``
exports. For a full IR graph, extend with log decoding or external API.

See also: ``scripts/export_abct_tx_sample.py`` for a one-off export.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cross.shared.normalize import norm_addr


def _kv_event(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Single synthetic Event carrying token transfer rows as flattened params."""
    param: dict[str, Any] = {"rows": len(rows)}
    for i, r in enumerate(rows[:32]):
        param[f"t{i}_contract"] = str(r.get("contractAddress", "") or "")
        param[f"t{i}_from"] = str(r.get("from", "") or "")
        param[f"t{i}_to"] = str(r.get("to", "") or "")
        param[f"t{i}_value"] = str(r.get("value", "") or "")
    return {"event": name, "param": param}


def eth_row_to_abct_tx(row: pd.Series, *, bridge: str = "Celer", network: str = "eth") -> dict[str, Any]:
    """One ETH ``cun`` row → minimal ``Tx`` dict."""
    h = norm_addr(row.get("hash", ""))
    ca = str(row.get("contractAddress", "") or "").strip().lower()
    return {
        "hash": h,
        "input": {
            "func": "Transfer",
            "param": {
                "bridge": bridge,
                "network": network.lower(),
                "txhash": h,
                "contractAddress": ca,
                "from": norm_addr(row.get("from", "")),
                "to": norm_addr(row.get("to", "")),
                "value": str(row.get("value", "") or ""),
                "timeStamp": str(row.get("timeStamp", "") or ""),
            },
        },
        "event_logs": [
            _kv_event(
                "TokenTransfer",
                [
                    {
                        "contractAddress": ca,
                        "from": norm_addr(row.get("from", "")),
                        "to": norm_addr(row.get("to", "")),
                        "value": str(row.get("value", "") or ""),
                    }
                ],
            )
        ],
    }


def bnb_hash_group_to_abct_tx(sub: pd.DataFrame, *, bridge: str = "Celer", network: str = "bnb") -> dict[str, Any]:
    """All ``qu`` rows for one ``hash`` → one ``Tx`` with multiple synthetic log rows."""
    if sub.empty:
        return {"hash": "", "input": {"func": "UnknownEvent", "param": {}}, "event_logs": []}
    sub = sub.copy()
    h = norm_addr(sub.iloc[0].get("hash", ""))
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "contractAddress": str(r.get("contractAddress", "") or "").strip().lower(),
                "from": norm_addr(r.get("from", "")),
                "to": norm_addr(r.get("to", "")),
                "value": str(r.get("value", "") or ""),
            }
        )
    ts = sub["timeStamp"].iloc[0] if "timeStamp" in sub.columns else ""
    return {
        "hash": h,
        "input": {
            "func": "TransferBatch",
            "param": {
                "bridge": bridge,
                "network": network.lower(),
                "txhash": h,
                "timeStamp": str(ts),
                "token_row_count": len(rows),
            },
        },
        "event_logs": [_kv_event("TokenTransfers", rows)],
    }


def index_bnb_by_hash(bnb_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group BNB export by transaction hash (lowercase)."""
    out: dict[str, pd.DataFrame] = {}
    for h, g in bnb_df.groupby(bnb_df["hash"].astype(str).str.lower()):
        out[norm_addr(h)] = g
    return out
