"""Chunked / per-src ``WithdrawLocator`` + amount-based fallback (legacy Path B path)."""
from __future__ import annotations

import pandas as pd

from ...shared.connector_decimals import decimals_for_eth_bnb
from ...shared.transfers import bnb_df_to_dst_txs, eth_df_to_src_txs
from ...shared.normalize import norm_addr
from ...utils.safe_cast import safe_float
from ..locator.withdraw_locator import WithdrawLocator

from .constants import ZERO_ADDR


def fallback_dst_hash(
    src_row: pd.Series,
    dst_slice: pd.DataFrame,
    *,
    ratio_by_eth_token: dict[str, float] | None = None,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
    decimals_eth: dict[str, int] | None = None,
    decimals_bnb: dict[str, int] | None = None,
) -> str:
    """Pick BNB withdrawal hash: receiver + time window, rank by human amount error.

    Falls back to median raw ratio or earliest ``timeStamp`` when decimals are unavailable.
    """
    if dst_slice.empty:
        return ""
    recv = norm_addr(src_row.get("args.receiver", ""))
    ts = safe_float(src_row.get("timestamp"), 0.0)
    eth_amt = safe_float(src_row.get("args.amount"), 0.0)
    eth_ca = str(src_row.get("args.asset_s", "") or "").strip().lower()
    asset_d = str(src_row.get("args.asset_d", "") or "").strip().lower()
    ratio = None
    if ratio_by_eth_bnb_pair and eth_ca and asset_d:
        ratio = ratio_by_eth_bnb_pair.get((eth_ca, norm_addr(asset_d)))
    if ratio is None and ratio_by_eth_token and eth_ca:
        ratio = ratio_by_eth_token.get(eth_ca)

    df = dst_slice.copy()
    df["_ts"] = pd.to_numeric(df["timeStamp"], errors="coerce").fillna(0)
    df = df[df["_ts"] > ts]
    df = df[df["to"].map(norm_addr) == recv]
    if df.empty:
        return ""

    dec_e = decimals_eth or {}
    dec_b = decimals_bnb or {}

    def human_src_amt() -> float | None:
        if eth_amt <= 0:
            return None
        if not eth_ca or eth_ca == ZERO_ADDR:
            return eth_amt / (10**18)
        d = dec_e.get(eth_ca)
        if d is None:
            return None
        return eth_amt / (10**d)

    hs = human_src_amt()
    use_human = bool(hs is not None and hs > 0)

    best_hash = ""
    best_score = float("inf")
    best_tmin = float("inf")

    for h_raw, g in df.groupby(df["hash"].astype(str)):
        sub = g[g["to"].map(norm_addr) == recv]
        if sub.empty:
            continue
        tmin = safe_float(pd.to_numeric(sub["_ts"], errors="coerce").fillna(0).min(), 0.0)

        hd = 0.0
        raw_tot = 0.0
        unknown_dec = False
        for _, rr in sub.iterrows():
            ca = str(rr.get("contractAddress", "") or "").strip().lower()
            raw = safe_float(rr.get("value"), 0.0)
            if raw <= 0:
                continue
            raw_tot += raw
            if not ca or ca == ZERO_ADDR:
                hd += raw / (10**18)
                continue
            d = dec_b.get(ca)
            if d is None:
                unknown_dec = True
                continue
            hd += raw / (10**d)

        if unknown_dec:
            hd = 0.0

        if use_human and hd > 0 and not unknown_dec:
            score = abs(hd - hs) / max(float(hs), 1e-30)
        elif ratio is not None and eth_amt > 0 and raw_tot > 0:
            score = abs(raw_tot / eth_amt - ratio)
        elif eth_amt > 0 and raw_tot > 0 and eth_ca and eth_ca != ZERO_ADDR and eth_ca not in dec_e:
            score = 1e9
        else:
            score = tmin - ts

        pick = score < best_score - 1e-15 or (abs(score - best_score) <= 1e-15 and tmin < best_tmin)
        if pick:
            best_score = score
            best_tmin = tmin
            best_hash = norm_addr(h_raw)

    if not best_hash:
        idx = df["_ts"].idxmin()
        best_hash = norm_addr(df.loc[idx, "hash"])
    return best_hash


def _record_from_row(txh: str, dst_h: str) -> dict:
    return {
        "srcnet": "ETH",
        "srcTxHash": norm_addr(txh),
        "dstnet": "BNB",
        "dstTxHash": norm_addr(dst_h) if dst_h else "",
    }


def run_withdraw_locator_per_src(
    src_all: pd.DataFrame,
    dst_all: pd.DataFrame,
    *,
    dst_window_sec: float = 14400.0,
    use_fallback: bool = True,
    ratio_by_eth_token: dict[str, float] | None = None,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """One WithdrawLocator pass per source tx and a tight BNB time slice."""
    want_eth = {
        str(x).strip().lower()
        for x in src_all["args.asset_s"].tolist()
        if str(x).strip() and str(x).lower() not in ("nan", "")
    }
    want_bnb = {
        str(x).strip().lower()
        for x in dst_all["contractAddress"].tolist()
        if str(x).strip() and str(x).lower() not in ("nan", "", ZERO_ADDR)
    }
    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb)

    dst_ts = pd.to_numeric(dst_all["timeStamp"], errors="coerce").fillna(0)
    records: list[dict] = []

    for i in range(len(src_all)):
        row_df = src_all.iloc[[i]].copy()
        ts = safe_float(row_df["timestamp"].iloc[0], 0.0)
        lo = ts
        hi = ts + dst_window_sec
        mask = (dst_ts > lo) & (dst_ts <= hi)
        dst_slice = dst_all.loc[mask].copy()
        if dst_slice.empty:
            dst_slice = dst_all.copy()

        loc = WithdrawLocator(row_df, dst_slice)
        rec = loc.search_withdraw()
        got = ""
        if rec:
            got = str(rec[0].get("dstTxHash") or "").strip()
        if (not got or got.lower() == "nan") and use_fallback:
            got = fallback_dst_hash(
                row_df.iloc[0],
                dst_slice,
                ratio_by_eth_token=ratio_by_eth_token,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
                decimals_eth=dec_e,
                decimals_bnb=dec_b,
            )
            if got:
                records.append(
                    _record_from_row(row_df.iloc[0]["txhash"], got),
                )
                continue
        if rec:
            records.append(rec[0])
        else:
            txh = norm_addr(row_df.iloc[0].get("txhash", ""))
            records.append(_record_from_row(txh, ""))

    return pd.DataFrame(records)


def run_withdraw_locator_chunked(
    eth_or_src_df: pd.DataFrame,
    bnb_df: pd.DataFrame,
    chunk_size: int = 80,
    time_pad: float = 2400.0,
    *,
    src_preformatted: bool = False,
) -> pd.DataFrame:
    """Chunked runner: cross join each chunk with a BNB slice filtered by chunk time span + pad."""
    if src_preformatted:
        src_all = eth_or_src_df.copy()
    else:
        src_all = eth_df_to_src_txs(eth_or_src_df)
    dst_all = bnb_df_to_dst_txs(bnb_df)
    dst_all_ts = pd.to_numeric(dst_all["timeStamp"], errors="coerce").fillna(0)

    records = []
    for start in range(0, len(src_all), chunk_size):
        chunk = src_all.iloc[start : start + chunk_size].copy()
        t0 = chunk["timestamp"].min()
        t1 = chunk["timestamp"].max() + time_pad
        mask = (dst_all_ts >= t0) & (dst_all_ts <= t1)
        dst_slice = dst_all.loc[mask].copy()
        if dst_slice.empty:
            dst_slice = dst_all.copy()

        loc = WithdrawLocator(chunk, dst_slice)
        rec = loc.search_withdraw()
        records.extend(rec)

    return pd.DataFrame(records)


def apply_fallback_to_pairs(
    pairs: pd.DataFrame,
    src_all: pd.DataFrame,
    dst_all: pd.DataFrame,
    *,
    dst_window_sec: float = 14400.0,
    ratio_by_eth_token: dict[str, float] | None = None,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Fill empty ``dstTxHash`` using :func:`fallback_dst_hash` (same window as Path B)."""
    out = pairs.copy()
    if out.empty or "srcTxHash" not in out.columns:
        return out

    want_eth = {
        str(x).strip().lower()
        for x in src_all["args.asset_s"].tolist()
        if str(x).strip() and str(x).lower() not in ("nan", "")
    }
    want_bnb = {
        str(x).strip().lower()
        for x in dst_all["contractAddress"].tolist()
        if str(x).strip() and str(x).lower() not in ("nan", "", ZERO_ADDR)
    }
    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb)

    dst_ts = pd.to_numeric(dst_all["timeStamp"], errors="coerce").fillna(0)
    src_all = src_all.copy()
    src_all["_txh"] = src_all["txhash"].map(norm_addr)

    for i in out.index:
        got = str(out.loc[i, "dstTxHash"] or "").strip().lower()
        if got and got not in ("nan", "none"):
            continue
        txh = norm_addr(out.loc[i, "srcTxHash"])
        sub = src_all[src_all["_txh"] == txh]
        if sub.empty:
            continue
        src_row = sub.iloc[0]
        ts = safe_float(src_row.get("timestamp"), 0.0)
        mask = (dst_ts > ts) & (dst_ts <= ts + dst_window_sec)
        dst_slice = dst_all.loc[mask].copy()
        if dst_slice.empty:
            dst_slice = dst_all.copy()
        fb = fallback_dst_hash(
            src_row,
            dst_slice,
            ratio_by_eth_token=ratio_by_eth_token,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            decimals_eth=dec_e,
            decimals_bnb=dec_b,
        )
        if fb:
            out.loc[i, "dstTxHash"] = fb
            if "dstnet" in out.columns:
                out.loc[i, "dstnet"] = "BNB"
            if "srcnet" in out.columns:
                out.loc[i, "srcnet"] = "ETH"

    return out
