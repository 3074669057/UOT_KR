"""Aggregate transaction-level Path B predictions to ETH address ↔ BNB address counts."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ...shared.bnb_pick import pick_bnb_recipient
from ...shared.load_csv import bridge_address_from_eth_df
from ...shared.normalize import norm_addr
from ..path_a.pairing import eth_row_for_src


def aggregate_address_pairs_from_path_b(
    pairs: pd.DataFrame,
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count predicted (address_a, address_b) pairs from Path B tx-level output.

    ``address_a`` is ETH depositor ``from`` for the src tx; ``address_b`` comes from
    :func:`pick_bnb_recipient` on the predicted BNB dst tx slice (same rule as Path A).
    Rows with missing dst hash or unresolved BNB pick are skipped.
    """
    bridge = bridge_address_from_eth_df(eth_df)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, r in pairs.iterrows():
        src_h = norm_addr(r.get("srcTxHash", ""))
        dst_h = norm_addr(str(r.get("dstTxHash", "") or ""))
        if not src_h or not dst_h:
            continue
        er = eth_row_for_src(eth_df, src_h)
        if er is None:
            continue
        address_a = norm_addr(er.get("from", ""))
        if not address_a:
            continue
        hn = bnb_df["hash"].astype(str).str.strip().str.lower()
        sl = bnb_df.loc[hn == dst_h]
        addr_b, reason, _ = pick_bnb_recipient(sl, address_a, bridge)
        if not addr_b or reason == "unresolved":
            continue
        counts[(address_a, addr_b)] += 1

    rows = [
        {"address_a": a, "address_b": b, "pair_count": c, "confidence_note": "tx_level_pair_count"}
        for (a, b), c in sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]
    return pd.DataFrame(rows)
