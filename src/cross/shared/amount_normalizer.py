"""Human vs raw token amounts and cross-chain raw ratio (Decimal)."""
from __future__ import annotations

from decimal import Decimal
from typing import Union

Num = Union[int, float, str, Decimal]


def _d(x: Num) -> Decimal:
    return Decimal(str(x))


def raw_to_human(raw_amount: Num, decimals: int) -> Decimal:
    return _d(raw_amount) / (Decimal(10) ** int(decimals))


def human_to_raw(human_amount: Num, decimals: int) -> Decimal:
    return _d(human_amount) * (Decimal(10) ** int(decimals))


def expected_raw_ratio(src_decimals: int, dst_decimals: int, human_ratio: Num = Decimal("1")) -> Decimal:
    """bnb_raw/eth_raw when both sides follow expected_human_ratio in human units."""
    return (Decimal(10) ** (int(dst_decimals) - int(src_decimals))) * _d(human_ratio)


def amount_error_human(src_human: Num, dst_human: Num) -> Decimal:
    sh, dh = _d(src_human), _d(dst_human)
    denom = max(abs(sh), abs(dh), Decimal("1e-12"))
    return abs(sh - dh) / denom


def amount_error_usd(src_usd: Num, dst_usd: Num) -> Decimal:
    su, du = _d(src_usd), _d(dst_usd)
    denom = max(abs(su), abs(du), Decimal("1e-12"))
    return abs(su - du) / denom


def dynamic_raw_ratio_from_prices(
    *,
    src_decimals: int,
    dst_decimals: int,
    src_price_usd: Num,
    dst_price_usd: Num,
) -> Decimal:
    """raw_dst/raw_src for value transfer when USD prices are per 1 human token."""
    pu, pv = _d(src_price_usd), _d(dst_price_usd)
    if pu <= 0 or pv <= 0:
        return Decimal(0)
    human_ratio = pu / pv
    return expected_raw_ratio(src_decimals, dst_decimals, human_ratio)


__all__ = [
    "raw_to_human",
    "human_to_raw",
    "expected_raw_ratio",
    "amount_error_human",
    "amount_error_usd",
    "dynamic_raw_ratio_from_prices",
]
