"""Infer ETH ERC20 -> BNB token contract map from labeled pairs."""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from cross.shared.bnb_pick import pick_bnb_recipient
from cross.domain.path_a.pairing import eth_row_for_src
from cross.shared.load_csv import bridge_address_from_eth_df
from cross.shared.normalize import norm_addr


def build_eth_bnb_token_contract_map(
    eth_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, str]:
    bridge = bridge_address_from_eth_df(eth_df)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _, lab in labels.iterrows():
        src_h = norm_addr(lab.get("srcTxhash", ""))
        dst_h = norm_addr(lab.get("dstTxhash", ""))
        er = eth_row_for_src(eth_df, src_h)
        if er is None:
            continue
        eth_ca = str(er.get("contractAddress", "") or "").strip().lower()
        if not eth_ca or eth_ca == "0x0000000000000000000000000000000000000000":
            continue
        dst_slice = bnb_df[bnb_df["hash"] == dst_h]
        if dst_slice.empty:
            continue
        user_a = norm_addr(er.get("from", ""))
        addr_b, _, _ = pick_bnb_recipient(dst_slice, user_a, bridge)
        if not addr_b:
            continue
        bnb_ca = ""
        for _, row in dst_slice.iterrows():
            if norm_addr(row.get("to")) != addr_b:
                continue
            ca = str(row.get("contractAddress", "") or "").strip().lower()
            if ca and ca != "0x0000000000000000000000000000000000000000":
                bnb_ca = ca
                break
        if bnb_ca:
            counts[eth_ca][bnb_ca] += 1
    return {eth: ctr.most_common(1)[0][0] for eth, ctr in counts.items() if ctr}
