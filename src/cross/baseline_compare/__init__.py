"""Baseline comparison utilities (Connector vs RC-UOT-Q ablation packages)."""

from cross.baseline_compare.bridge_semantic_masking import (
    BridgeSemanticField,
    MaskSpec,
    get_mask_spec,
    iter_b_subset_mask_specs,
    iter_main_mask_specs,
    write_mask_specs,
)

__all__ = [
    "BridgeSemanticField",
    "MaskSpec",
    "get_mask_spec",
    "iter_b_subset_mask_specs",
    "iter_main_mask_specs",
    "write_mask_specs",
]
