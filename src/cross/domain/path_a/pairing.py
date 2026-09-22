"""Path A: join labeled pairs with ETH/BNB CSV rows and extract (a, b)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cross.shared.bnb_pick import pick_bnb_recipient
from cross.shared.load_csv import bridge_address_from_eth_df, load_eth_cun, load_label_csv
from cross.shared.normalize import norm_addr

logger = logging.getLogger(__name__)

_PAIR_LABEL_COLUMNS = [
    "srcTxHash",
    "dstTxHash",
    "address_a",
    "address_b",
    "bnb_pick_reason",
    "bnb_candidate_count",
    "eth_extract_note",
    "hit_eth_csv",
    "hit_bnb_csv",
    "complete_ab",
]


def eth_row_for_src(eth_df: pd.DataFrame, src_tx: str) -> pd.Series | None:
    h = norm_addr(src_tx)
    sub = eth_df[eth_df["hash"] == h]
    if sub.empty:
        return None
    ca = sub["contractAddress"].fillna("").astype(str).str.lower()
    tok = sub[(ca.ne("")) & (ca.ne("0x0000000000000000000000000000000000000000"))]
    pick = tok.iloc[0] if not tok.empty else sub.iloc[0]
    return pick


def run_path_a(
    label_path: Path,
    eth_path: Path,
    bnb_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    labels = load_label_csv(label_path)
    eth_df = load_eth_cun(eth_path)
    bridge = bridge_address_from_eth_df(eth_df)
    if not bridge:
        logger.warning("No bridge Address column in ETH CSV; BNB pick may be less reliable")

    rows_out = []
    missing_eth = 0
    missing_bnb = 0
    unresolved_b = 0

    bnb_by_hash = {h: g for h, g in bnb_df.groupby("hash")}

    for _, lab in labels.iterrows():
        src_tx = str(lab.get("srcTxhash", "")).strip().lower()
        dst_tx = str(lab.get("dstTxhash", "")).strip().lower()
        if not src_tx.startswith("0x"):
            src_tx = "0x" + src_tx
        if not dst_tx.startswith("0x"):
            dst_tx = "0x" + dst_tx

        eth_row = eth_row_for_src(eth_df, src_tx)
        if eth_row is None:
            missing_eth += 1
            rows_out.append(
                {
                    "srcTxHash": src_tx,
                    "dstTxHash": dst_tx,
                    "address_a": "",
                    "address_b": "",
                    "bnb_pick_reason": "",
                    "bnb_candidate_count": 0,
                    "eth_extract_note": "missing_eth_row",
                    "hit_eth_csv": False,
                    "hit_bnb_csv": dst_tx in bnb_by_hash,
                    "complete_ab": False,
                }
            )
            continue

        address_a = norm_addr(eth_row.get("from", ""))
        note = "ok"
        if norm_addr(eth_row.get("to", "")) != bridge and bridge:
            note = "eth_to_differs_from_bridge_column"

        dst_slice = bnb_df[bnb_df["hash"] == dst_tx]
        if dst_slice.empty:
            missing_bnb += 1
            b_hit = False
            addr_b = ""
            reason = "missing_bnb_rows"
            cand = 0
        else:
            b_hit = True
            addr_b, reason, cand = pick_bnb_recipient(dst_slice, address_a, bridge)
            if not addr_b or reason == "unresolved":
                unresolved_b += 1

        rows_out.append(
            {
                "srcTxHash": src_tx,
                "dstTxHash": dst_tx,
                "address_a": address_a,
                "address_b": addr_b,
                "bnb_pick_reason": reason,
                "bnb_candidate_count": cand,
                "eth_extract_note": note,
                "hit_eth_csv": True,
                "hit_bnb_csv": b_hit,
                "complete_ab": bool(address_a and addr_b),
            }
        )

    out_df = pd.DataFrame(rows_out)
    if out_df.empty:
        out_df = pd.DataFrame(columns=_PAIR_LABEL_COLUMNS)
    stats = {
        "label_rows": len(labels),
        "pairs_written": len(out_df),
        "complete_ab": int(out_df["complete_ab"].sum()),
        "missing_eth": missing_eth,
        "missing_bnb": missing_bnb,
        "unresolved_bnb_pick": unresolved_b,
        "bridge_address_used": bridge,
    }
    return out_df, stats
