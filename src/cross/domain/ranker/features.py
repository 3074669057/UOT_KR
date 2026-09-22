"""Tabular features for a single (src_row_index, dst_tx_hash) edge."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross.shared.bnb_pick import pick_bnb_recipient
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float
from cross.domain.path_b.greedy import ZERO, _human_dst_for_hash, _human_src

FEATURE_DIM = 9


def compute_base_error_and_features(
    pos: int,
    dst_hash: str,
    *,
    src_all: pd.DataFrame,
    dst_all: pd.DataFrame,
    dst_ts: pd.Series,
    dst_window_sec: float,
    dec_e: dict[str, int],
    dec_b: dict[str, int],
    ratio_by_eth_token: dict[str, float] | None,
    median_delay_sec: float | None,
    delay_weight: float,
    receiver_mode: str,
    bnb_bridge_address: str,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
) -> tuple[float, np.ndarray]:
    row = src_all.iloc[pos]
    eth_from = norm_addr(row.get("args.receiver", ""))
    recv_legacy = eth_from
    ts = safe_float(row.get("timestamp"), 0.0)
    eth_amt = safe_float(row.get("args.amount"), 0.0)
    eth_ca = str(row.get("args.asset_s", "") or "").strip().lower()
    asset_d = str(row.get("args.asset_d", "") or "").strip().lower()
    hs = _human_src(eth_amt, eth_ca, dec_e)
    h = norm_addr(dst_hash)
    mask = (dst_ts > ts) & (dst_ts <= ts + dst_window_sec)
    sl = dst_all.loc[mask]
    g = sl[sl["hash"].astype(str).map(norm_addr) == h]
    if g.empty:
        z = np.zeros(FEATURE_DIM, dtype=np.float32); z[8] = 1e6
        return 1e9, z
    bridge = norm_addr(bnb_bridge_address)
    recv_eff = recv_legacy
    rm = (receiver_mode or "eth_from").strip().lower()
    if rm == "bnb_pick_per_candidate":
        recv_pick, _, _ = pick_bnb_recipient(g, eth_from, bridge)
        if not recv_pick:
            z = np.zeros(FEATURE_DIM, dtype=np.float32); z[8] = 1e6
            return 1e8, z
        recv_eff = norm_addr(recv_pick)
    else:
        g = g[g["to"].map(norm_addr) == recv_eff]
        if g.empty:
            z = np.zeros(FEATURE_DIM, dtype=np.float32); z[8] = 1e6
            return 1e9, z
    hd, _ = _human_dst_for_hash(g, recv_eff, dec_b)
    sub = g[g["to"].map(norm_addr) == recv_eff]
    raw_tot = safe_float(pd.to_numeric(sub["value"], errors="coerce").fillna(0).sum(), 0.0)
    tmin_dst = safe_float(pd.to_numeric(g["timeStamp"], errors="coerce").fillna(0).min(), 0.0)
    ratio = None
    if ratio_by_eth_bnb_pair and eth_ca and asset_d:
        ratio = ratio_by_eth_bnb_pair.get((eth_ca, norm_addr(asset_d)))
    if ratio is None and ratio_by_eth_token and eth_ca:
        ratio = ratio_by_eth_token.get(eth_ca)
    if ratio is not None and eth_amt > 0 and raw_tot > 0:
        exp_raw = eth_amt * ratio
        err = abs(raw_tot - exp_raw) / max(exp_raw, 1e-30)
        ratio_err = err
    elif eth_amt > 0 and raw_tot > 0:
        if eth_ca and eth_ca != ZERO and eth_ca not in dec_e:
            err = 1e6 + 2.0; ratio_err = 0.0
        else:
            ratio_err = abs(raw_tot - eth_amt) / max(eth_amt, 1e-30); err = float(ratio_err)
    elif hs is not None and hs > 0 and hd is not None and hd > 0:
        err = abs(hd - hs) / max(hs, 1e-30); ratio_err = 0.0
    else:
        err = 1e6 + (1.0 if (hd is not None and hd > 0) else 1e3); ratio_err = 0.0
    human_rel = (
        abs(hd - hs) / max(hs, 1e-30)
        if (hs is not None and hs > 0 and hd is not None and hd > 0)
        else 0.0
    )
    terr = 0.0
    if median_delay_sec is not None and median_delay_sec > 0:
        gap = max(tmin_dst - ts, 0)
        terr = abs(gap - median_delay_sec) / max(median_delay_sec, 60.0)
        err = err + delay_weight * terr
    dt_norm = (tmin_dst - ts) / max(dst_window_sec, 1.0)
    feat = np.array([
        np.log1p(max(eth_amt, 0.0)),
        np.log1p(max(raw_tot, 0.0)),
        min(ratio_err, 50.0),
        min(human_rel, 50.0),
        terr,
        float(np.clip(dt_norm, -2.0, 5.0)),
        float(ratio is not None),
        float(hs is not None and hs > 0 and hd > 0),
        min(err, 1e6),
    ], dtype=np.float32)
    return float(err), feat


__all__ = ["FEATURE_DIM", "compute_base_error_and_features"]
