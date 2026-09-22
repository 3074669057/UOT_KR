"""Project roots and default IO paths (self-contained; no sibling ``Connector`` repo)."""
from __future__ import annotations

import os
from pathlib import Path

# src/cross/config/paths.py -> repo root is parents[3]
CROSS_ROOT = Path(__file__).resolve().parents[3]


def token_data_dir() -> Path:
    """Directory with ``ERC20.csv`` / ``BERC20.csv`` / ``PERC20.csv`` (address, decimal).

    Override with env ``CROSS_TOKEN_DIR`` to use an external copy of the token tables.
    """
    override = (os.environ.get("CROSS_TOKEN_DIR") or "").strip()
    if override:
        return Path(override)
    return CROSS_ROOT / "data" / "Token"


DEFAULT_ETH_CSV = CROSS_ROOT / "in" / "Celer_ETH_cun.csv"
DEFAULT_LABEL_DIR = CROSS_ROOT / "label"
DEFAULT_LABEL_CSV = DEFAULT_LABEL_DIR / "celer_label.csv"
DEFAULT_CELER_TX_LABELS_CSV = CROSS_ROOT / "label" / "tx" / "celer_label.csv"
DEFAULT_EMPTY_LABEL_CSV = DEFAULT_LABEL_DIR / "empty.csv"
DEFAULT_OUT_DIR = CROSS_ROOT / "out"
DEFAULT_RANKER_CHECKPOINT = DEFAULT_OUT_DIR / "ranker_mlp.pt"
DEFAULT_CONFIG_DIR = CROSS_ROOT / "config"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "defaults.json"
DEFAULT_LOCAL_CONFIG_PATH = DEFAULT_CONFIG_DIR / "local.json"
