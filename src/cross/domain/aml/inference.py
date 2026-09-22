"""AML inference helpers: model-based and rule-based src filtering."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cross.shared.load_csv import bridge_address_from_eth_df
from cross.shared.normalize import norm_addr

from .features import FEATURE_DIM, aml_vector_from_eth_row, eth_representative_row_by_hash
from .model import AmlMLP
from .rules import apply_rules_to_src


def apply_aml_filter_to_src_all(
    src_all: pd.DataFrame,
    eth_df: pd.DataFrame,
    *,
    aml_mode: str = "rules",
    aml_keep_levels: tuple[str, ...] = ("medium", "high"),
    aml_medium_threshold: float = 40.0,
    aml_high_threshold: float = 70.0,
    aml_checkpoint: Path | None = None,
    aml_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Apply AML pre-filter; same behavior as Path B ``_run_path_b_core`` AML block.

    Returns ``(filtered_src_all, aml_meta)``. When mode is invalid or checkpoint missing,
    passes through ``src_all`` and records why in meta (warnings logged by callers if needed).
    """
    aml_meta: dict = {"aml_filter_applied": False}
    mode = (aml_mode or "rules").strip().lower()
    if mode == "model":
        if aml_checkpoint is not None:
            ck = Path(aml_checkpoint)
            if ck.is_file():
                return filter_src_by_aml_risk(src_all, eth_df, ck, aml_threshold)
            logger = logging.getLogger(__name__)
            logger.warning("AML checkpoint not found (%s); skipping AML filter", ck)
            return src_all, {
                "aml_filter_applied": False,
                "aml_mode": "model",
                "aml_note": f"checkpoint_missing:{ck}",
            }
        logger = logging.getLogger(__name__)
        logger.warning("AML mode=model but aml_checkpoint is empty; skipping AML filter")
        return src_all, {"aml_filter_applied": False, "aml_mode": "model", "aml_note": "no_checkpoint"}
    if mode == "rules":
        # Align with pipeline online AML gate: `_fetch_bnb_dataset_online` keeps rows with
        # aml_risk_score >= aml_threshold*100. Without this OR, txs scored in [25,40) stay
        # level "low" and Path B would drop them while BNB data was fetched for them.
        score_floor = float(aml_threshold) * 100.0
        return filter_src_by_aml_rules(
            src_all,
            eth_df,
            medium_threshold=aml_medium_threshold,
            high_threshold=aml_high_threshold,
            keep_levels=aml_keep_levels,
            score_floor=score_floor,
        )
    if mode == "off":
        return src_all, {"aml_filter_applied": False, "aml_mode": "off"}
    logger = logging.getLogger(__name__)
    logger.warning("Unknown aml_mode=%s; skipping AML filter", mode)
    return src_all, {"aml_filter_applied": False, "aml_mode": mode, "aml_note": "unknown_mode"}


def load_aml_bundle(path: Path) -> tuple[AmlMLP, dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"AML checkpoint not found: {p}")
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(p, map_location="cpu")
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise ValueError(f"Invalid AML checkpoint (expected dict with state_dict): {p}")
    meta = blob.get("meta") or {}
    in_dim = int(meta.get("in_dim", meta.get("feature_dim", FEATURE_DIM)))
    hid = int(meta.get("hid", 64))
    if in_dim != FEATURE_DIM:
        raise ValueError(f"AML checkpoint in_dim={in_dim} != code FEATURE_DIM={FEATURE_DIM}")
    model = AmlMLP(in_dim, hid=hid)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta


def aml_probabilities_for_txhashes(
    txhashes: list[str],
    eth_df: pd.DataFrame,
    model: AmlMLP,
    *,
    bridge_addr: str,
    device: torch.device | None = None,
) -> np.ndarray:
    dev = device or torch.device("cpu")
    model = model.to(dev)
    by_h = eth_representative_row_by_hash(eth_df)
    rows: list[np.ndarray] = []
    for h in txhashes:
        row = by_h.get(norm_addr(h))
        rows.append(np.zeros(FEATURE_DIM, dtype=np.float32) if row is None else aml_vector_from_eth_row(row, bridge_addr=bridge_addr))
    x = torch.from_numpy(np.stack(rows, axis=0).astype(np.float32)).to(dev)
    with torch.no_grad():
        logits = model(x)
        return torch.sigmoid(logits).cpu().numpy()


def aml_scores_for_src_all(src_all: pd.DataFrame, eth_df: pd.DataFrame, checkpoint: Path) -> np.ndarray:
    bridge = bridge_address_from_eth_df(eth_df)
    model, _meta = load_aml_bundle(checkpoint)
    if "txhash" not in src_all.columns:
        raise ValueError("src_all must contain column 'txhash'")
    hashes = [norm_addr(x) for x in src_all["txhash"].astype(str).tolist()]
    return aml_probabilities_for_txhashes(hashes, eth_df, model, bridge_addr=bridge)


def filter_src_by_aml_risk(
    src_all: pd.DataFrame,
    eth_df: pd.DataFrame,
    checkpoint: Path,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    if src_all.empty:
        return src_all, {
            "aml_filter_applied": True, "aml_threshold": float(threshold), "aml_kept_rows": 0, "aml_dropped_rows": 0,
            "aml_checkpoint": str(Path(checkpoint).resolve()), "aml_note": "src_all was empty before AML filter",
        }
    scores = aml_scores_for_src_all(src_all, eth_df, checkpoint)
    mask = scores >= float(threshold)
    out = src_all.loc[mask].reset_index(drop=True)
    meta = {
        "aml_filter_applied": True, "aml_mode": "model", "aml_threshold": float(threshold),
        "aml_kept_rows": int(mask.sum()), "aml_dropped_rows": int((~mask).sum()),
        "aml_checkpoint": str(Path(checkpoint).resolve()),
    }
    return out, meta


def filter_src_by_aml_rules(
    src_all: pd.DataFrame,
    eth_df: pd.DataFrame,
    *,
    medium_threshold: float = 40.0,
    high_threshold: float = 70.0,
    keep_levels: tuple[str, ...] = ("medium", "high"),
    score_floor: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    bridge = bridge_address_from_eth_df(eth_df)
    return apply_rules_to_src(
        src_all,
        eth_df,
        bridge_addr=bridge,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
        keep_levels=keep_levels,
        score_floor=score_floor,
    )


__all__ = [
    "apply_aml_filter_to_src_all",
    "load_aml_bundle",
    "aml_probabilities_for_txhashes",
    "aml_scores_for_src_all",
    "filter_src_by_aml_risk",
    "filter_src_by_aml_rules",
]
