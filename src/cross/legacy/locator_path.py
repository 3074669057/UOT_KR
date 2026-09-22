"""Backward-compatible re-exports for locator-style Path B imports."""
from __future__ import annotations

from cross.domain.evaluation.compare import compare_to_label
from cross.domain.path_b.align import src_txs_aligned_to_labels
from cross.domain.path_b.env import configure_path_b_connector_env
from cross.domain.path_b.legacy_connector import (
    apply_fallback_to_pairs,
    fallback_dst_hash,
    run_withdraw_locator_chunked,
    run_withdraw_locator_per_src,
)
from cross.domain.path_b.runner import run_path_b, write_path_b_outputs
from cross.shared.transfers import bnb_df_to_dst_txs, eth_df_to_src_txs, parse_transfer_value

_parse_transfer_value = parse_transfer_value

__all__ = [
    "apply_fallback_to_pairs",
    "bnb_df_to_dst_txs",
    "compare_to_label",
    "configure_path_b_connector_env",
    "eth_df_to_src_txs",
    "fallback_dst_hash",
    "parse_transfer_value",
    "run_path_b",
    "run_withdraw_locator_chunked",
    "run_withdraw_locator_per_src",
    "src_txs_aligned_to_labels",
    "write_path_b_outputs",
]
