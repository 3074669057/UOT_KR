"""ERC20/BEP20 decimals from CSV registries; never silently defaults to 18 for unknown tokens."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cross.config.paths import token_data_dir


def _load_subset_from_csv(csv_path: Path, wanted: set[str]) -> dict[str, int]:
    if not wanted:
        return {}
    need = {str(w).strip().lower() for w in wanted if str(w).strip()}
    need.discard("")
    need.discard("nan")
    if not need:
        return {}
    out: dict[str, int] = {}
    if not csv_path.is_file():
        return {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=["address", "decimal"],
        chunksize=80000,
        dtype={"address": str},
        low_memory=False,
    ):
        chunk["address"] = chunk["address"].str.lower()
        hit = chunk[chunk["address"].isin(need)]
        for addr, dec in zip(hit["address"], hit["decimal"]):
            out[addr] = int(dec)
        if need.issubset(out.keys()):
            break
    return out


class DecimalsRegistry:
    """Load decimals from ETH/BSC CSV tables; missing entries return ``None`` (no silent 18)."""

    def __init__(
        self,
        eth_csv_path: Path | None = None,
        bsc_csv_path: Path | None = None,
    ) -> None:
        td = token_data_dir()
        self._eth_csv = Path(eth_csv_path) if eth_csv_path is not None else td / "ERC20.csv"
        self._bsc_csv = Path(bsc_csv_path) if bsc_csv_path is not None else td / "BERC20.csv"
        self._eth_cache: dict[str, int] = {}
        self._bsc_cache: dict[str, int] = {}

    def get_decimals(self, chain: str, address: str) -> int | None:
        addr = str(address or "").strip().lower()
        if not addr or addr == "0x0000000000000000000000000000000000000000":
            return None
        c = (chain or "").strip().lower()
        if c in ("eth", "ethereum"):
            if addr in self._eth_cache:
                return self._eth_cache[addr]
            got = _load_subset_from_csv(self._eth_csv, {addr})
            self._eth_cache.update(got)
            return self._eth_cache.get(addr)
        if c in ("bsc", "bnb", "binance", "bep20"):
            if addr in self._bsc_cache:
                return self._bsc_cache[addr]
            got = _load_subset_from_csv(self._bsc_csv, {addr})
            self._bsc_cache.update(got)
            return self._bsc_cache.get(addr)
        raise ValueError(f"unsupported chain for decimals: {chain!r}")

    def ensure_many(self, chain: str, addresses: set[str] | None) -> None:
        """Batch-load decimals for many addresses (one CSV scan per missing subset)."""
        if not addresses:
            return
        c = (chain or "").strip().lower()
        if c in ("eth", "ethereum"):
            cache = self._eth_cache
            csv_p = self._eth_csv
        elif c in ("bsc", "bnb", "binance", "bep20"):
            cache = self._bsc_cache
            csv_p = self._bsc_csv
        else:
            raise ValueError(f"unsupported chain for decimals: {chain!r}")
        need = {str(a).strip().lower() for a in addresses if str(a).strip()}
        need.discard("")
        need.discard("nan")
        need -= set(cache.keys())
        if not need:
            return
        cache.update(_load_subset_from_csv(csv_p, need))

    def subset_dict_eth(self, tokens: set[str]) -> dict[str, int]:
        self.ensure_many("eth", tokens)
        out: dict[str, int] = {}
        for t in tokens:
            k = str(t).strip().lower()
            v = self._eth_cache.get(k)
            if v is not None:
                out[k] = v
        return out

    def subset_dict_bsc(self, tokens: set[str]) -> dict[str, int]:
        self.ensure_many("bsc", tokens)
        out: dict[str, int] = {}
        for t in tokens:
            k = str(t).strip().lower()
            v = self._bsc_cache.get(k)
            if v is not None:
                out[k] = v
        return out


__all__ = ["DecimalsRegistry"]
