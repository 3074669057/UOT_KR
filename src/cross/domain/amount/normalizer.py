"""Human/raw conversions and cross-chain amount error metrics (domain facade)."""
from __future__ import annotations

from decimal import Decimal
from typing import Union

from cross.domain.token.decimals_registry import DecimalsLookup, DomainDecimalsRegistry

from cross.shared import amount_normalizer as _core

Num = Union[int, float, str, Decimal]


def raw_to_human(raw_amount: Num, decimals: int | None) -> Decimal | None:
    if decimals is None:
        return None
    return _core.raw_to_human(raw_amount, int(decimals))


def human_to_raw(human_amount: Num, decimals: int | None) -> Decimal | None:
    if decimals is None:
        return None
    return _core.human_to_raw(human_amount, int(decimals))


def compute_expected_raw_ratio(
    src_decimals: int | None,
    dst_decimals: int | None,
    human_ratio: Num = Decimal("1"),
) -> Decimal | None:
    if src_decimals is None or dst_decimals is None:
        return None
    return _core.expected_raw_ratio(int(src_decimals), int(dst_decimals), human_ratio)


def amount_error_human(src_human: Num, dst_human: Num) -> Decimal:
    return _core.amount_error_human(src_human, dst_human)


def amount_error_usd(src_usd: Num, dst_usd: Num) -> Decimal:
    return _core.amount_error_usd(src_usd, dst_usd)


class AmountNormalizer:
    """Resolves decimals via :class:`DomainDecimalsRegistry` so unknown tokens do not silently match."""

    def __init__(self, registry: DomainDecimalsRegistry | None = None) -> None:
        self._reg = registry or DomainDecimalsRegistry()

    def raw_to_human(self, chain: str, token: str, raw_amount: Num) -> tuple[Decimal | None, DecimalsLookup]:
        lu = self._reg.lookup(chain, token)
        if lu.decimals is None:
            return None, lu
        return _core.raw_to_human(raw_amount, lu.decimals), lu

    def human_to_raw(self, chain: str, token: str, human_amount: Num) -> tuple[Decimal | None, DecimalsLookup]:
        lu = self._reg.lookup(chain, token)
        if lu.decimals is None:
            return None, lu
        return _core.human_to_raw(human_amount, lu.decimals), lu


__all__ = [
    "AmountNormalizer",
    "amount_error_human",
    "amount_error_usd",
    "compute_expected_raw_ratio",
    "human_to_raw",
    "raw_to_human",
]
