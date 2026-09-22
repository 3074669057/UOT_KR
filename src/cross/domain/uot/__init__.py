"""Unbalanced optimal transport for cross-chain flow correspondence."""

from .cost_matrix import build_cost_matrix, build_cost_matrix_decomposed
from .decode_transport import (
    build_transport_plan_export_rows,
    build_uot_summary_dict,
    compute_unmatched_source_mass,
    compute_unmatched_target_mass,
    decode_correspondence,
    derive_top1_tx_pairs,
    row_entropies,
)
from .uot_solver import solve_uot

__all__ = [
    "build_cost_matrix",
    "build_cost_matrix_decomposed",
    "build_transport_plan_export_rows",
    "build_uot_summary_dict",
    "solve_uot",
    "decode_correspondence",
    "compute_unmatched_source_mass",
    "compute_unmatched_target_mass",
    "derive_top1_tx_pairs",
    "row_entropies",
]
