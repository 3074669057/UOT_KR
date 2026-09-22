"""Load curated ETH→BSC routes; join DecimalsRegistry; compute expected bnb_raw/eth_raw ratios."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from cross.shared.amount_normalizer import dynamic_raw_ratio_from_prices, expected_raw_ratio
from cross.shared.decimals_registry import DecimalsRegistry
from cross.shared.normalize import norm_addr

logger = logging.getLogger(__name__)

ZERO = "0x0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class ResolvedRoute:
    route_id: str
    src: str
    dst: str
    route_type: str
    value_ratio_mode: str
    expected_human_ratio: Decimal | None
    src_decimals: int
    dst_decimals: int
    expected_raw_ratio: float | None
    requires_price_snapshot: bool


def _parse_human_ratio(v: Any) -> Decimal | None:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


class RouteRegistry:
    def __init__(
        self,
        routes: list[dict[str, Any]],
        *,
        decimals: DecimalsRegistry,
        invalid: list[str],
        resolved: list[ResolvedRoute],
    ) -> None:
        self._decimals = decimals
        self.invalid_messages = invalid
        self.resolved_routes = resolved
        self._by_src: dict[str, list[ResolvedRoute]] = {}
        for r in resolved:
            self._by_src.setdefault(r.src, []).append(r)

    @classmethod
    def from_json_obj(
        cls,
        raw: Any,
        *,
        decimals_registry: DecimalsRegistry | None = None,
    ) -> RouteRegistry | None:
        if raw is None:
            return None
        if isinstance(raw, dict) and "routes" in raw:
            routes = raw["routes"]
        elif isinstance(raw, list):
            routes = raw
        else:
            raise ValueError("token routes JSON must be a list or {\"routes\": [...]}")
        if not isinstance(routes, list):
            raise ValueError("routes must be a JSON array")
        dec = decimals_registry or DecimalsRegistry()
        return cls._from_route_list(routes, decimals=dec)

    @classmethod
    def _from_route_list(
        cls,
        routes: list[Any],
        *,
        decimals: DecimalsRegistry,
    ) -> RouteRegistry:
        invalid: list[str] = []
        resolved: list[ResolvedRoute] = []
        for i, row in enumerate(routes):
            if not isinstance(row, dict):
                invalid.append(f"route[{i}]: not an object")
                continue
            rid = str(row.get("route_id") or f"route_{i}")
            src = norm_addr(str(row.get("src") or ""))
            dst = norm_addr(str(row.get("dst") or ""))
            if not src or src == ZERO or not dst or dst == ZERO:
                invalid.append(f"{rid}: missing src or dst")
                continue
            src_chain = str(row.get("src_chain") or "eth").lower()
            dst_chain = str(row.get("dst_chain") or "bsc").lower()
            if src_chain not in ("eth", "ethereum") or dst_chain not in ("bsc", "bnb"):
                invalid.append(f"{rid}: only eth→bsc routes supported (got {src_chain}->{dst_chain})")
                continue
            sd = decimals.get_decimals("eth", src)
            dd = decimals.get_decimals("bsc", dst)
            if sd is None:
                invalid.append(f"{rid}: missing ETH decimals for {src}")
                continue
            if dd is None:
                invalid.append(f"{rid}: missing BSC decimals for {dst}")
                continue
            vmode = str(row.get("value_ratio_mode") or "").strip().lower()
            rh = _parse_human_ratio(row.get("expected_human_ratio"))
            req_price = bool(row.get("requires_price_snapshot"))
            rtype = str(row.get("route_type") or "")
            exp_ratio_f: float | None = None
            if vmode == "dynamic_price" or req_price:
                exp_ratio_f = None
            else:
                hr = rh if rh is not None else Decimal(1)
                exp_ratio_f = float(expected_raw_ratio(sd, dd, hr))
            resolved.append(
                ResolvedRoute(
                    route_id=rid,
                    src=src,
                    dst=dst,
                    route_type=rtype,
                    value_ratio_mode=vmode,
                    expected_human_ratio=rh,
                    src_decimals=sd,
                    dst_decimals=dd,
                    expected_raw_ratio=exp_ratio_f,
                    requires_price_snapshot=req_price or vmode == "dynamic_price",
                )
            )
        return cls(routes, decimals=decimals, invalid=invalid, resolved=resolved)

    @classmethod
    def load(
        cls,
        path: Path | None,
        *,
        decimals_registry: DecimalsRegistry | None = None,
    ) -> RouteRegistry | None:
        if path is None:
            return None
        p = Path(path)
        if not p.is_file():
            logger.warning("token routes file not found: %s", p)
            return None
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_json_obj(raw, decimals_registry=decimals_registry)

    def routes_for_src(self, eth_token: str) -> list[ResolvedRoute]:
        return list(self._by_src.get(norm_addr(eth_token), []))

    def ratio_by_eth_bnb_pair(
        self,
        price_snapshot_usd: dict[str, float] | None,
    ) -> dict[tuple[str, str], float]:
        """(eth_token_lower, bsc_token_lower) -> bnb_raw/eth_raw."""
        out: dict[tuple[str, str], float] = {}
        snap = price_snapshot_usd or {}
        snap_l = {norm_addr(k): float(v) for k, v in snap.items() if v and float(v) > 0}
        for r in self.resolved_routes:
            key = (r.src, r.dst)
            if r.requires_price_snapshot:
                pu, pv = snap_l.get(r.src), snap_l.get(r.dst)
                if pu is None or pv is None:
                    continue
                rr = dynamic_raw_ratio_from_prices(
                    src_decimals=r.src_decimals,
                    dst_decimals=r.dst_decimals,
                    src_price_usd=pu,
                    dst_price_usd=pv,
                )
                out[key] = float(rr)
            elif r.expected_raw_ratio is not None:
                out[key] = float(r.expected_raw_ratio)
        return out

    def token_map_eth_to_bnb(self) -> dict[str, str]:
        """Prefer first static route per ETH src when multiple dst (e.g. WETH peg before WBNB)."""
        m: dict[str, str] = {}
        for r in self.resolved_routes:
            if r.requires_price_snapshot:
                continue
            if r.src not in m:
                m[r.src] = r.dst
        return m

    def ratio_by_eth_token_fallback(self) -> dict[str, float]:
        """Single ratio per ETH src when exactly one static dst route exists (legacy shape)."""
        static_by_src: dict[str, list[float]] = {}
        for r in self.resolved_routes:
            if r.requires_price_snapshot or r.expected_raw_ratio is None:
                continue
            static_by_src.setdefault(r.src, []).append(float(r.expected_raw_ratio))
        out: dict[str, float] = {}
        for src, vals in static_by_src.items():
            if len(vals) == 1:
                out[src] = vals[0]
        return out


__all__ = ["ResolvedRoute", "RouteRegistry"]
