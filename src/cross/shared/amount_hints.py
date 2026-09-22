"""Per ETH-token median amount ratio and bridge delay hints from labeled pairs."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from cross.shared.bnb_pick import pick_bnb_recipient
from cross.domain.path_a.pairing import eth_row_for_src
from cross.shared.load_csv import bridge_address_from_eth_df
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float


def build_median_amount_ratio_by_eth_token(eth_df, bnb_df, labels) -> dict[str, float]:
    bridge = bridge_address_from_eth_df(eth_df)
    buckets: dict[str, list[float]] = defaultdict(list)
    for _, lab in labels.iterrows():
        src_h = norm_addr(lab.get("srcTxhash", ""))
        dst_h = norm_addr(lab.get("dstTxhash", ""))
        er = eth_row_for_src(eth_df, src_h)
        if er is None:
            continue
        eth_ca = str(er.get("contractAddress", "") or "").strip().lower()
        if not eth_ca or eth_ca == "0x0000000000000000000000000000000000000000":
            continue
        try:
            eth_amt = float(str(er.get("value", "0") or "0"))
        except ValueError:
            continue
        if eth_amt <= 0:
            continue
        dst_slice = bnb_df[bnb_df["hash"] == dst_h]
        if dst_slice.empty:
            continue
        user_a = norm_addr(er.get("from", ""))
        addr_b, _, _ = pick_bnb_recipient(dst_slice, user_a, bridge)
        if not addr_b:
            continue
        bnb_val = None
        for _, row in dst_slice.iterrows():
            if norm_addr(row.get("to")) != addr_b:
                continue
            ca = str(row.get("contractAddress", "") or "").strip().lower()
            if not ca or ca == "0x0000000000000000000000000000000000000000":
                continue
            try:
                raw = row.get("value", "0")
                v = float(int(raw, 16)) if isinstance(raw, str) and raw.startswith("0x") else float(raw)
                if v > 0:
                    bnb_val = v
                    break
            except (ValueError, TypeError):
                continue
        if bnb_val is not None and eth_amt > 0:
            buckets[eth_ca].append(bnb_val / eth_amt)
    out: dict[str, float] = {}
    for eth_ca, vals in buckets.items():
        if not vals:
            continue
        s = sorted(vals)
        mid = len(s) // 2
        out[eth_ca] = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    return out


def median_bridge_delay_seconds(eth_df, bnb_df, labels) -> float:
    delays: list[float] = []
    bnb_df = bnb_df.copy()
    bnb_df["hash"] = bnb_df["hash"].astype(str).str.strip().str.lower()
    for _, lab in labels.iterrows():
        src_h = str(lab.get("srcTxhash", "")).strip().lower()
        dst_h = str(lab.get("dstTxhash", "")).strip().lower()
        if not src_h.startswith("0x"):
            src_h = "0x" + src_h
        if not dst_h.startswith("0x"):
            dst_h = "0x" + dst_h
        er = eth_row_for_src(eth_df, src_h)
        if er is None:
            continue
        sl = bnb_df[bnb_df["hash"] == dst_h]
        if sl.empty:
            continue
        ts_src = safe_float(er.get("timeStamp"), 0.0)
        ts_dst = safe_float(sl["timeStamp"].iloc[0], 0.0)
        if ts_dst > ts_src:
            delays.append(ts_dst - ts_src)
    if not delays:
        return 120.0
    s = sorted(delays)
    mid = len(s) // 2
    return float(s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid]))
