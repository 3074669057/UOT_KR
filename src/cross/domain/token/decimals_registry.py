"""Load ERC20 / BEP20 decimals from CSV; unknown tokens never default to 18."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cross.config.paths import token_data_dir
from cross.shared.decimals_registry import DecimalsRegistry as _SharedDecimalsRegistry


class DecimalsStatus(str, Enum):
    known = "known"
    unknown = "unknown"
    low_confidence = "low_confidence"


@dataclass(frozen=True)
class DecimalsLookup:
    """Result of a decimals query (explicit status, no silent 18)."""

    decimals: int | None
    status: DecimalsStatus

    @property
    def is_known(self) -> bool:
        return self.decimals is not None and self.status == DecimalsStatus.known


class DomainDecimalsRegistry:
    """Domain-facing decimals table: same CSV semantics as ``cross.shared.decimals_registry``."""

    def __init__(
        self,
        eth_csv_path: Path | None = None,
        bsc_csv_path: Path | None = None,
    ) -> None:
        td = token_data_dir()
        self._inner = _SharedDecimalsRegistry(
            eth_csv_path=Path(eth_csv_path) if eth_csv_path is not None else None,
            bsc_csv_path=Path(bsc_csv_path) if bsc_csv_path is not None else None,
        )

    def lookup(self, chain: str, address: str) -> DecimalsLookup:
        v = self._inner.get_decimals(chain, address)
        if v is None:
            return DecimalsLookup(decimals=None, status=DecimalsStatus.unknown)
        return DecimalsLookup(decimals=int(v), status=DecimalsStatus.known)

    def get_decimals(self, chain: str, address: str) -> int | None:
        """Backward-compatible ``int | None`` for solver code paths."""
        return self._inner.get_decimals(chain, address)

    def ensure_many(self, chain: str, addresses: set[str] | None) -> None:
        self._inner.ensure_many(chain, addresses)


__all__ = ["DecimalsLookup", "DecimalsStatus", "DomainDecimalsRegistry"]
