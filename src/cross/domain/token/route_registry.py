"""Curated ETH→BSC token routes (separate from decimals CSVs)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from cross.shared.normalize import norm_addr

logger = logging.getLogger(__name__)

ZERO = "0x0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class TokenRoute:
    route_id: str
    src_chain: str
    src_token: str
    dst_chain: str
    dst_token: str
    route_type: str
    value_ratio_mode: str
    expected_human_ratio: Decimal | None
    confidence: float
    requires_price_snapshot: bool


def _dec(v: Any) -> Decimal | None:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


def load_token_routes(path: Path | None) -> list[TokenRoute]:
    """Load routes from ``token_routes.eth_bsc.json``-shaped file (``routes`` array)."""
    if path is None or not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("failed to read token routes: %s", path)
        return []
    if isinstance(raw, dict) and "routes" in raw:
        rows = raw["routes"]
    elif isinstance(raw, list):
        rows = raw
    else:
        return []
    out: list[TokenRoute] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        rid = str(row.get("route_id") or f"route_{i}")
        src = norm_addr(str(row.get("src") or ""))
        dst = norm_addr(str(row.get("dst") or ""))
        if not src or src == ZERO or not dst or dst == ZERO:
            continue
        out.append(
            TokenRoute(
                route_id=rid,
                src_chain=str(row.get("src_chain") or "eth").lower(),
                src_token=src,
                dst_chain=str(row.get("dst_chain") or "bsc").lower(),
                dst_token=dst,
                route_type=str(row.get("route_type") or ""),
                value_ratio_mode=str(row.get("value_ratio_mode") or "").lower(),
                expected_human_ratio=_dec(row.get("expected_human_ratio")),
                confidence=float(row.get("confidence", 1.0) or 1.0),
                requires_price_snapshot=bool(row.get("requires_price_snapshot")),
            )
        )
    return out


__all__ = ["TokenRoute", "load_token_routes"]
