"""Fixed-length AML features from a single ETH ``cun`` row (no BNB / no RPC)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cross.shared.transfers import parse_transfer_value
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float

FEATURE_VERSION = "v1"
FEATURE_DIM = 24
ZERO = "0x0000000000000000000000000000000000000000"


def _bucket_index(addr: str, n: int = 16) -> int:
    s = (addr or "").strip().lower()
    if not s or s == "nan":
        return 0
    h = 0
    for ch in s[-16:]:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h % n)


def aml_vector_from_eth_row(row: pd.Series, *, bridge_addr: str = "") -> np.ndarray:
    v = float(parse_transfer_value(row.get("value", "0")))
    lv = math.log1p(max(v, 0.0)) / 50.0
    ts = safe_float(row.get("timeStamp"), 0.0)
    ts_n = ts / 1.0e9
    sec_day = ts % 86400.0 if ts > 0 else 0.0
    ang = 2.0 * math.pi * (sec_day / 86400.0)
    h_sin, h_cos = math.sin(ang), math.cos(ang)
    ca = str(row.get("contractAddress", "") or "").strip().lower()
    is_zero_ca = 1.0 if (not ca or ca == ZERO or ca == "nan") else 0.0
    bi = _bucket_index(ca, 16)
    buckets = np.zeros(16, dtype=np.float32)
    buckets[bi] = 1.0
    fr = norm_addr(str(row.get("from", "") or ""))
    has_from = 1.0 if fr else 0.0
    br = norm_addr(bridge_addr or "")
    to_a = norm_addr(str(row.get("to", "") or ""))
    to_bridge = 1.0 if br and to_a == br else 0.0
    gas_used = safe_float(row.get("gasUsed", row.get("gas", "0")), 0.0)
    gas_used_n = math.log1p(max(gas_used, 0.0)) / 20.0
    gas_price = safe_float(row.get("gasPrice", "0"), 0.0)
    gp_n = math.log1p(max(gas_price, 0.0)) / 25.0
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    out[0], out[1], out[2], out[3] = float(lv), float(ts_n), float(h_sin), float(h_cos)
    out[4:20] = buckets
    out[20], out[21], out[22], out[23] = float(is_zero_ca), float(has_from), float(to_bridge), float(gas_used_n + gp_n) * 0.5
    return out


def eth_representative_row_by_hash(eth_df: pd.DataFrame) -> dict[str, pd.Series]:
    by_h: dict[str, pd.Series] = {}
    for _, r in eth_df.iterrows():
        h = norm_addr(r.get("hash", ""))
        if not h:
            continue
        ca = str(r.get("contractAddress", "") or "").strip().lower()
        prev = by_h.get(h)
        if prev is None:
            by_h[h] = r
            continue
        prev_ca = str(prev.get("contractAddress", "") or "").strip().lower()
        prev_empty = not prev_ca or prev_ca == ZERO
        cur_nonempty = bool(ca) and ca != ZERO
        if prev_empty and cur_nonempty:
            by_h[h] = r
    return by_h


__all__ = ["FEATURE_DIM", "FEATURE_VERSION", "aml_vector_from_eth_row", "eth_representative_row_by_hash"]
