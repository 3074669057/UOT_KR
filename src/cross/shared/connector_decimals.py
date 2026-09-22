"""Load ERC20 decimals from bundled data/Token/*.csv (via :class:`DecimalsRegistry`)."""
from __future__ import annotations

from cross.shared.decimals_registry import DecimalsRegistry


def decimals_for_eth_bnb(eth_tokens: set[str], bnb_tokens: set[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Return per-address decimals maps for addresses **found** in CSV (missing keys omitted)."""
    reg = DecimalsRegistry()
    reg.ensure_many("eth", set(eth_tokens))
    reg.ensure_many("bsc", set(bnb_tokens))
    return reg.subset_dict_eth(set(eth_tokens)), reg.subset_dict_bsc(set(bnb_tokens))
