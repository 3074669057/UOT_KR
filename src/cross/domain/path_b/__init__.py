"""Path B: greedy assignment, WithdrawLocator legacy path, and orchestration."""

from .align import src_txs_aligned_to_labels
from .constants import ZERO_ADDR
from .env import configure_path_b_connector_env
from .legacy_connector import (
    apply_fallback_to_pairs,
    fallback_dst_hash,
    run_withdraw_locator_chunked,
    run_withdraw_locator_per_src,
)
from .runner import run_path_b, write_path_b_outputs

__all__ = [
    "ZERO_ADDR",
    "apply_fallback_to_pairs",
    "configure_path_b_connector_env",
    "fallback_dst_hash",
    "run_path_b",
    "run_withdraw_locator_chunked",
    "run_withdraw_locator_per_src",
    "src_txs_aligned_to_labels",
    "write_path_b_outputs",
]
