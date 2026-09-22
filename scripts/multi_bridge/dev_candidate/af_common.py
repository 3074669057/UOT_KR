"""Amount-free candidate development — shared machinery.

The candidate definition is FROZEN by the hash-locked NEXT_CANDIDATE_SPEC.md (sole source
of truth; do not re-derive). Exact weights (from the spec):
  PRIMARY amount-free renormalized: kept frozen absolute weights
  {time 0.25, route 0.15, risk 0.15, evidence 0.05, novelty 0.05} (sum 0.65),
  renormalized to sum 1 -> each weight divided by 0.65.
  ABLATION unrenormalized: the same absolute weights, NOT renormalized.
Amount removal is PAIRWISE ONLY: the amount-derived transport marginals
(a = risk-weighted USD, b = evidence-weighted USD) are preserved unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

AF = REPO / "out" / "multi_bridge_expansion" / "amount_free_candidate_dev"
CTD = REPO / "out" / "multi_bridge_expansion" / "cost_transport_diagnosis"

BRIDGES = ("Celer", "Multi", "Poly")
DEV_SEEDS = (201, 202, 203, 204, 205)
HOLDOUT_SEEDS = (301, 302, 303, 304, 305)  # NEVER generated / never read this round

# Frozen absolute weights of the components KEPT by the candidate (from the locked spec;
# amount (0.40 incl. merged graph 0.05) is REMOVED from the pairwise cost only).
KEPT_ABS_WEIGHTS = {"time": 0.25, "route": 0.15, "risk": 0.15, "evidence": 0.05,
                    "novelty": 0.05}
COMPONENT_OF = {"time": "time_cost", "route": "route_cost", "risk": "risk_cost",
                "evidence": "evidence_cost", "novelty": "address_novelty_cost"}
RENORM_DENOM = float(sum(KEPT_ABS_WEIGHTS.values()))  # 0.65, per the spec


def primary_weights() -> dict[str, float]:
    """PRIMARY amount-free renormalized weights (exact fractions, spec-locked)."""
    return {k: v / RENORM_DENOM for k, v in KEPT_ABS_WEIGHTS.items()}


def ablation_weights() -> dict[str, float]:
    """ABLATION unrenormalized weights (exact LOCO condition, spec-locked)."""
    return dict(KEPT_ABS_WEIGHTS)


def build_amount_free_costs(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the PRIMARY and ABLATION pairwise cost matrices from the FROZEN components.

    Returns {'primary': C_p, 'ablation': C_a}. The FULL frozen cost is C_effective
    (bridge_prior_bonus is verified to be zero on these synthetic cells).
    """
    primary = np.zeros_like(components["time_cost"], dtype=float)
    ablation = np.zeros_like(components["time_cost"], dtype=float)
    pw = primary_weights()
    aw = ablation_weights()
    for name, w in pw.items():
        primary += w * components[COMPONENT_OF[name]]
    for name, w in aw.items():
        ablation += w * components[COMPONENT_OF[name]]
    return {"primary": primary, "ablation": ablation}


def load_dev_cell(bridge: str, seed: int) -> dict[str, Any]:
    """Reuse the previous round's dev cell (same templates/features/GT; frozen params)."""
    from baseline_mechanism.common import tpl_maps, truth_structure
    from diag.ctd_common import load_dev_cell as _ldc

    cell = _ldc(bridge, seed)
    cell["truth"] = truth_structure(cell["labels"])
    return cell


def cost_d4_edges(C: np.ndarray, sids: list[str], tids: list[str], k: int = 5) -> list[tuple[str, str]]:
    from diag.ctd_common import _rank_asc
    rr = _rank_asc(C, "row")
    cr = _rank_asc(C, "col")
    edges = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                edges.append((sids[i], tids[j]))
    return edges
