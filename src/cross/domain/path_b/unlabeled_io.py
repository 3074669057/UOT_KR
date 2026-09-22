"""Unlabeled Path B: bridge-deposit filter, optional JSON priors for ratio/delay/token map."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ...shared.transfers import eth_df_to_src_txs
from ...shared.normalize import norm_addr

from .route_registry import RouteRegistry

logger = logging.getLogger(__name__)

ZERO = "0x0000000000000000000000000000000000000000"

DEFAULT_UNLABELED_PRIORS: dict[str, Any] = {
    "median_bridge_delay_sec": 3600.0,
    "default_ratio": None,
    "ratio_by_eth_token": {},
    "token_map_eth_to_bnb": {},
    "price_snapshot_usd": {},
}


def _merge_legacy_compat_export(user: dict[str, Any], out: dict[str, Any]) -> None:
    """Fill ratio/token map from ``legacy_compat_export`` when top-level keys are absent."""
    leg = user.get("legacy_compat_export")
    if not isinstance(leg, dict):
        return
    if isinstance(leg.get("ratio_by_eth_token"), dict):
        base = dict(out.get("ratio_by_eth_token") or {})
        for k, v in leg["ratio_by_eth_token"].items():
            kk = str(k).strip().lower()
            if kk in base:
                continue
            try:
                base[kk] = float(v)
            except (TypeError, ValueError):
                continue
        out["ratio_by_eth_token"] = base
    if isinstance(leg.get("token_map_eth_to_bnb"), dict):
        base_tm = dict(out.get("token_map_eth_to_bnb") or {})
        for k, v in leg["token_map_eth_to_bnb"].items():
            ek = norm_addr(str(k))
            vk = norm_addr(str(v))
            if ek and vk and ek not in base_tm:
                base_tm[ek] = vk
        out["token_map_eth_to_bnb"] = base_tm
    if "default_ratio" in leg and "default_ratio" not in user:
        out["default_ratio"] = leg.get("default_ratio")


def load_unlabeled_priors_json(path: Path | None) -> dict[str, Any]:
    """Merge ``DEFAULT_UNLABELED_PRIORS`` with optional JSON file.

    JSON keys (all optional):

    - ``median_bridge_delay_sec`` (float): prior bridge delay in seconds.
    - ``default_ratio`` (float | null): if set, applied to every ETH token contract in
      ``src_txs`` that has no entry in ``ratio_by_eth_token`` / computed ratio_map.
    - ``ratio_by_eth_token``: map ``eth_contract_address -> float`` (BNB raw / ETH raw).
    - ``token_map_eth_to_bnb``: map ``eth_token_contract -> bnb_token_contract``.
    - ``price_snapshot_usd``: optional map ``token_contract_lower -> usd_price`` for ``dynamic_price`` routes.
    - ``legacy_compat_export``: optional nested object with ``ratio_by_eth_token`` / ``token_map_eth_to_bnb``
      merged when top-level maps omit those entries (v2 priors file compatibility).
    """
    out = dict(DEFAULT_UNLABELED_PRIORS)
    if path is None:
        return out
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"unlabeled priors file not found: {p}")
    with open(p, encoding="utf-8") as f:
        user = json.load(f)
    if not isinstance(user, dict):
        raise ValueError("unlabeled priors JSON must be an object")
    out.update(user)
    _merge_legacy_compat_export(user, out)
    return out


def overlay_ratios_from_route_registry(
    ratio_map: dict[str, float],
    route_registry: RouteRegistry | None,
    *,
    price_snapshot_usd: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Apply static route ratios (per-ETH-src) and return pair map for (eth,bsc) lookups.

    Route-derived flat ratios **override** existing ``ratio_map`` entries for the same
    ETH contract (metadata-aware priors take precedence over label medians / file order).
    """
    snap: dict[str, float] | None = None
    if isinstance(price_snapshot_usd, dict) and price_snapshot_usd:
        snap = {}
        for k, v in price_snapshot_usd.items():
            a = norm_addr(str(k))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if a and fv > 0:
                snap[a] = fv
    pair: dict[tuple[str, str], float] = {}
    if route_registry is None:
        return dict(ratio_map), pair
    pair = route_registry.ratio_by_eth_bnb_pair(snap)
    out = dict(ratio_map)
    for k, v in route_registry.ratio_by_eth_token_fallback().items():
        out[k] = float(v)
    return out, pair


def merge_token_map_with_route_defaults(
    priors_map: dict[str, str],
    route_registry: RouteRegistry | None,
) -> dict[str, str]:
    """Routes provide defaults; explicit ``priors_map`` entries win."""
    out: dict[str, str] = dict(priors_map)
    if route_registry is None:
        return out
    for k, v in route_registry.token_map_eth_to_bnb().items():
        out.setdefault(k, v)
    return out


def filter_eth_bridge_deposits(eth_df: pd.DataFrame, bridge_address: str) -> pd.DataFrame:
    """Keep rows whose ``to`` equals the Celer bridge (deposit-side), if identifiable.

    If ``bridge_address`` is empty or ``to`` column is missing, returns ``eth_df`` unchanged
    and logs a warning.
    """
    if not bridge_address:
        logger.warning(
            "unlabeled: no bridge Address in ETH CSV; using full ETH export (may include non-deposit txs)",
        )
        return eth_df
    if "to" not in eth_df.columns:
        logger.warning("unlabeled: ETH CSV has no 'to' column; using full ETH export")
        return eth_df
    br = norm_addr(bridge_address)
    to_n = eth_df["to"].map(norm_addr)
    sel = eth_df.loc[to_n == br].copy()
    if sel.empty:
        logger.warning(
            "unlabeled: no rows with to==bridge; falling back to full ETH export",
        )
        return eth_df
    return sel


def enrich_src_txs_token_map(src_all: pd.DataFrame, token_map: dict[str, str] | None) -> pd.DataFrame:
    """Add ``args.asset_d`` from ETH→BNB token map (same role as label-driven align)."""
    out = src_all.copy()
    if token_map:
        ds: list[str] = []
        for _, row in out.iterrows():
            eth_ca = str(row.get("args.asset_s") or "").strip().lower()
            ds.append(token_map.get(eth_ca, "") if eth_ca and eth_ca != ZERO else "")
        out["args.asset_d"] = ds
    else:
        out["args.asset_d"] = ""
    return out


def merge_ratio_priors(
    computed: dict[str, float],
    priors: dict[str, Any],
    src_all: pd.DataFrame,
) -> dict[str, float]:
    """Overlay JSON ``ratio_by_eth_token``, then fill missing tokens with ``default_ratio``."""
    out = dict(computed)
    raw_rt = priors.get("ratio_by_eth_token") or {}
    if isinstance(raw_rt, dict):
        for k, v in raw_rt.items():
            key = str(k).strip().lower()
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                continue
    dr = priors.get("default_ratio")
    if dr is None:
        return out
    try:
        dr_f = float(dr)
    except (TypeError, ValueError):
        return out
    for _, row in src_all.iterrows():
        ca = str(row.get("args.asset_s") or "").strip().lower()
        if ca and ca != ZERO and ca not in out:
            out[ca] = dr_f
    return out


def normalize_token_map_eth_to_bnb(raw: dict[str, Any] | None) -> dict[str, str]:
    if not raw or not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if v is None or str(v).strip() == "":
            continue
        ek = norm_addr(str(k))
        vk = norm_addr(str(v))
        if ek and vk:
            out[ek] = vk
    return out


def build_src_txs_unlabeled(eth_df_filtered: pd.DataFrame) -> pd.DataFrame:
    """One row per ETH tx hash from filtered deposits (Connector-style columns)."""
    return eth_df_to_src_txs(eth_df_filtered)
