from __future__ import annotations

from pathlib import Path

from .loader import load_runtime_config, validate_ethereum_config, validate_nodereal_config


def load_and_validate_config(defaults_path: Path, local_path: Path) -> dict:
    cfg = load_runtime_config(defaults_path=defaults_path, local_path=local_path)
    validate_nodereal_config(cfg)
    validate_ethereum_config(cfg)
    return cfg
