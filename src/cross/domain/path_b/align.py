"""Align label rows to ETH ``src_txs`` rows (one row per label line)."""
from __future__ import annotations

import pandas as pd

from ...shared.transfers import parse_transfer_value
from ...shared.normalize import norm_addr
from ..path_a.pairing import eth_row_for_src


def src_txs_aligned_to_labels(
    eth_df: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    token_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One ``src_txs`` row per label line, same ETH row choice as Path A; optional ``args.asset_d``."""
    rows: list[dict] = []
    for _, lab in labels.iterrows():
        h = norm_addr(lab.get("srcTxhash", ""))
        er = eth_row_for_src(eth_df, h)
        if er is None:
            rows.append(
                {
                    "txhash": h,
                    "args.receiver": "",
                    "args.amount": 0.0,
                    "args.asset_s": "",
                    "args.asset_d": "",
                    "args.srcChain": "ETH",
                    "args.dstChain": "BNB",
                    "timestamp": 0.0,
                }
            )
            continue
        eth_ca = str(er.get("contractAddress", "") or "").strip().lower()
        asset_d = ""
        if token_map and eth_ca:
            asset_d = token_map.get(eth_ca, "") or ""
        rows.append(
            {
                "txhash": h,
                "args.receiver": norm_addr(er.get("from", "")),
                "args.amount": parse_transfer_value(er.get("value", "0")),
                "args.asset_s": eth_ca,
                "args.asset_d": asset_d,
                "args.srcChain": "ETH",
                "args.dstChain": "BNB",
                "timestamp": float(str(er.get("timeStamp", "0") or "0")),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["args.amount"] = out["args.amount"].astype("float64")
    return out
