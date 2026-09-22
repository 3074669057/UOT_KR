"""Unified runtime config loader (defaults + local override)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cross.config.paths import DEFAULT_CONFIG_PATH, DEFAULT_LOCAL_CONFIG_PATH


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config file must be JSON object: {path}")
    return obj


def load_runtime_config(
    *,
    defaults_path: Path | None = None,
    local_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = _load_json(defaults_path or DEFAULT_CONFIG_PATH)
    local = _load_json(local_path or DEFAULT_LOCAL_CONFIG_PATH)
    merged = _deep_merge(defaults, local)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    return merged


def _non_empty_rpc_urls(nr: dict[str, Any]) -> list[str]:
    raw = nr.get("rpc_urls") or nr.get("rpc_url")
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        return [str(u).strip() for u in raw if str(u).strip()]
    return []


def validate_nodereal_config(cfg: dict[str, Any]) -> None:
    nr = (cfg or {}).get("nodereal") or {}
    if not nr:
        return
    keys = nr.get("api_keys") or []
    key_ok = isinstance(keys, list) and len(keys) >= 1
    url_ok = len(_non_empty_rpc_urls(nr)) >= 1
    if nr.get("enabled") and not key_ok and not url_ok:
        raise ValueError("nodereal.enabled=true requires nodereal.api_keys or nodereal.rpc_urls")


def validate_ethereum_config(cfg: dict[str, Any]) -> None:
    eth = (cfg or {}).get("ethereum") or {}
    if not eth:
        return
    keys = eth.get("api_keys") or []
    key_ok = isinstance(keys, list) and len(keys) >= 1
    url_ok = len(_non_empty_rpc_urls(eth)) >= 1
    if eth.get("enabled") and not key_ok and not url_ok:
        raise ValueError("ethereum.enabled=true requires ethereum.api_keys or ethereum.rpc_urls")
