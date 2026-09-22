"""ETH/BNB CSV rows → normalized ``src_txs`` / ``dst_txs`` frames for matching."""
from __future__ import annotations

import pandas as pd

from cross.utils.safe_cast import safe_float

from .normalize import norm_addr


def parse_transfer_value(cell: object) -> float:
    """Parse CSV ``value`` (decimal or 0x hex) to float."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return 0.0
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return 0.0
    if s.startswith("0x") or s.startswith("0X"):
        try:
            return float(int(s, 16))
        except ValueError:
            return 0.0
    try:
        return safe_float(s, 0.0)
    except Exception:
        return 0.0


def eth_df_to_src_txs(eth_df: pd.DataFrame) -> pd.DataFrame:
    """Build ``src_txs`` frame from cun export (one row per tx hash).

    ``Celer_ETH_cun`` may list several internal transfer lines per ``hash``; we keep one
    row per hash preferring a non-empty token ``contractAddress`` (same idea as Path A).
    """
    rows = []
    by_h: dict[str, dict] = {}
    for _, r in eth_df.iterrows():
        h = norm_addr(r.get("hash", ""))
        if not h:
            continue
        ca = str(r.get("contractAddress", "") or "").strip().lower()
        row = {
            "txhash": h,
            "args.receiver": norm_addr(r.get("from", "")),
            "args.amount": parse_transfer_value(r.get("value", "0")),
            "args.asset_s": ca,
            "args.srcChain": "ETH",
            "args.dstChain": "BNB",
            "timestamp": safe_float(r.get("timeStamp"), 0.0),
        }
        prev = by_h.get(h)
        if prev is None:
            by_h[h] = row
            continue
        prev_empty = not prev["args.asset_s"] or prev["args.asset_s"] == "0x0000000000000000000000000000000000000000"
        cur_nonempty = ca and ca != "0x0000000000000000000000000000000000000000"
        if prev_empty and cur_nonempty:
            by_h[h] = row

    rows = list(by_h.values())
    out = pd.DataFrame(rows)
    if not out.empty:
        out["args.amount"] = out["args.amount"].astype("float64")
    return out


def bnb_df_to_dst_txs(bnb_df: pd.DataFrame) -> pd.DataFrame:
    out = bnb_df.copy()
    for c in list(out.columns):
        if str(c).startswith("Unnamed"):
            out = out.drop(columns=[c], errors="ignore")
    out["hash"] = out["hash"].map(norm_addr)
    out["contractAddress"] = out["contractAddress"].fillna("").astype(str).str.lower()
    out["timeStamp"] = pd.to_numeric(out["timeStamp"], errors="coerce").fillna(0)
    out["value"] = out["value"].map(parse_transfer_value).astype("float64")
    out["from"] = out["from"].map(norm_addr)
    out["to"] = out["to"].map(norm_addr)
    out["Net"] = "BNB"
    return out
