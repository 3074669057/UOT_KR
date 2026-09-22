"""Constants for v3b high-F1 supplementary experiments."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_V3B = REPO_ROOT / "out" / "baseline_compare" / "bridge_semantic_ablation_v3b_high_f1"
OUT_V3 = REPO_ROOT / "out" / "baseline_compare" / "bridge_semantic_ablation_v3"
UOT_BASE = REPO_ROOT / "out" / "uot_delay_fixed_production"
LABELS = REPO_ROOT / "out" / "baseline_compare" / "labels"
ETH_CSV = REPO_ROOT / "in" / "Celer_ETH_cun.csv"
BNB_CSV = REPO_ROOT / "label" / "tx" / "Celer_BNB_qu.csv"

HASH_SEED = "bridge_semantic_ablation_v3b_high_f1"
DEV_RATIO = 0.3
TEST_RATIO = 0.7

STRICT_NO_ALL_MASK = "no_all_bridge_semantics"
CHAIN_OBSERVABLES_MASK = "no_bridge_metadata_chain_observables_retained"

TX_CVR_MAX = 0.01

TOP_K_GRID = [1, 2, 3, 5, 10]
TRANSPORT_MASS_THRESHOLD_GRID = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]
CONFIDENCE_THRESHOLD_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MARGIN_THRESHOLD_GRID = [0.0, 0.02, 0.05, 0.1, 0.2]
COVERAGE_THRESHOLD_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
TIME_ADMISSIBLE_WINDOW_SEC_GRID = [3600, 7200, 21600, 43200]
ABSTENTION_POLICY_GRID = [
    "none",
    "joint_time",
    "confidence",
    "joint_time_confidence",
    "topk_rescue",
]

HIGH_CONF_COVERAGE_FLOORS = [0.3, 0.5, 0.7]

FROZEN_PARAMS = {
    "top_k": 1,
    "transport_mass_threshold": 0.0,
    "confidence_threshold": 0.0,
    "margin_threshold": 0.0,
    "coverage_threshold": 0.0,
    "time_admissible_window_sec": 43200,
    "abstention_policy": "joint_time",
    "flow_pair_aggregation_rule": "argmax_mass",
}
