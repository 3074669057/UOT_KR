"""Reliability features for RC-UOT-v2.1. Label-free, inference-time available."""
from __future__ import annotations
import numpy as np
from typing import Any
from dataclasses import dataclass, field

@dataclass
class ReliabilityFeatures:
    """Pre-registered reliability feature set for a single flow (source or target)."""
    # Evidence completeness
    evidence_quality_score: float = 0.5          # raw score from flow metadata [0,1]
    evidence_level_present: float = 0.0           # boolean: has evidence_level field
    evidence_levels_count: float = 0.0            # normalized count of evidence indicators

    # Candidate ambiguity (source-side: across dst candidates; target-side: across src candidates)
    candidate_count: float = 0.0                  # number of admissible candidates
    candidate_count_norm: float = 0.0             # normalized by max possible
    cost_margin_best_vs_second: float = 0.0       # gap to second-best candidate
    cost_entropy_normalized: float = 0.0          # entropy of inverse-cost distribution

    # Amount consistency
    amount_usd: float = 0.0                       # raw USD amount (for scale context)
    amount_log_norm: float = 0.0                  # log-normalized amount

    # Temporal
    time_window_width_sec: float = 0.0            # width of flow time window
    has_exact_timestamp: float = 0.0              # boolean: precise timestamp available

    # Observation multiplicity
    tx_count: float = 0.0                         # number of transactions in flow
    tx_count_norm: float = 0.0                    # normalized
    address_count_norm: float = 0.0               # normalized unique addresses

    # Route/evidence support
    bridge_set_size: float = 0.0                  # number of bridge contracts hit
    has_bridge_hit: float = 0.0                   # boolean: any bridge contract match
    aml_risk_norm: float = 0.0                    # normalized AML risk (lower = more reliable)

    # Missingness
    missing_field_ratio: float = 0.0              # fraction of expected fields missing

    # All feature vector (for calibrator)
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(18, dtype=float))

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.evidence_quality_score, self.evidence_level_present,
            self.evidence_levels_count,
            self.candidate_count_norm, self.cost_margin_best_vs_second,
            self.cost_entropy_normalized,
            self.amount_log_norm, self.time_window_width_sec,
            self.has_exact_timestamp, self.tx_count_norm,
            self.address_count_norm, self.bridge_set_size,
            self.has_bridge_hit, self.aml_risk_norm,
            self.missing_field_ratio,
        ], dtype=float)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            'evidence_quality_score', 'evidence_level_present', 'evidence_levels_count',
            'candidate_count_norm', 'cost_margin_best_vs_second', 'cost_entropy_normalized',
            'amount_log_norm', 'time_window_width_sec', 'has_exact_timestamp',
            'tx_count_norm', 'address_count_norm', 'bridge_set_size',
            'has_bridge_hit', 'aml_risk_norm', 'missing_field_ratio',
        ]


def extract_reliability_features_from_flow(
    flow: dict[str, Any],
    candidate_costs: np.ndarray | None = None,
    causal_mask_row: np.ndarray | None = None,
) -> ReliabilityFeatures:
    """Extract label-free reliability features from a single flow segment.

    Args:
        flow: Flow segment dict from load_flow_segments
        candidate_costs: 1D array of costs to candidates (for ambiguity features)
        causal_mask_row: 1D boolean array of causally admissible candidates

    Returns:
        ReliabilityFeatures dataclass
    """
    f = ReliabilityFeatures()

    # Evidence completeness
    eqs = flow.get('evidence_quality_score')
    if eqs is not None and not (isinstance(eqs, float) and (eqs != eqs)):
        try:
            f.evidence_quality_score = float(np.clip(float(eqs), 0, 1))
        except (ValueError, TypeError):
            f.evidence_quality_score = 0.5

    el = flow.get('evidence_level')
    f.evidence_level_present = 1.0 if el is not None else 0.0

    els = flow.get('evidence_levels')
    if isinstance(els, (list, tuple)):
        f.evidence_levels_count = min(len(els) / 5.0, 1.0)
    elif els is not None:
        f.evidence_levels_count = 0.2

    # Candidate ambiguity
    if candidate_costs is not None and len(candidate_costs) > 0:
        if causal_mask_row is not None:
            admissible = candidate_costs[causal_mask_row]
        else:
            admissible = candidate_costs[candidate_costs < 1e11]

        n_ads = len(admissible)
        f.candidate_count = float(n_ads)
        f.candidate_count_norm = min(n_ads / 100.0, 1.0)

        if n_ads >= 2:
            sorted_costs = np.sort(admissible)
            margin = (sorted_costs[1] - sorted_costs[0]) / max(abs(sorted_costs[0]), 1.0)
            f.cost_margin_best_vs_second = float(np.clip(margin, 0, 10))

        if n_ads >= 1:
            inv_costs = 1.0 / (admissible + 1e-6)
            probs = inv_costs / (inv_costs.sum() + 1e-12)
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            max_entropy = np.log(max(n_ads, 2))
            f.cost_entropy_normalized = float(entropy / max(max_entropy, 1e-12))

    # Amount
    amt = flow.get('amount_usd') or flow.get('human_amount_sum') or flow.get('raw_amount_sum')
    if amt is not None:
        try:
            amt_f = float(amt)
            if amt_f == amt_f:  # not NaN
                f.amount_usd = amt_f
                f.amount_log_norm = float(np.clip(np.log1p(abs(amt_f)) / 20.0, 0, 1))
        except (ValueError, TypeError):
            pass

    # Temporal
    st = flow.get('start_time'); et = flow.get('end_time')
    try:
        st_f = float(st) if st is not None else None
        et_f = float(et) if et is not None else None
    except (ValueError, TypeError):
        st_f = None; et_f = None
    if st_f is not None and et_f is not None:
        f.time_window_width_sec = float(np.clip(abs(et_f - st_f) / 86400.0, 0, 30))
        f.has_exact_timestamp = 1.0
    elif st is not None or et is not None:
        f.time_window_width_sec = 1.0
        f.has_exact_timestamp = 0.5

    # Observation multiplicity
    tc = flow.get('tx_count')
    if tc is not None:
        f.tx_count = float(tc)
        f.tx_count_norm = min(float(tc) / 50.0, 1.0)

    ac = flow.get('address_count')
    if ac is not None:
        f.address_count_norm = min(float(ac) / 20.0, 1.0)

    # Route support
    bs = flow.get('bridge_set')
    if isinstance(bs, (list, tuple)):
        f.bridge_set_size = min(len(bs) / 5.0, 1.0)
    f.has_bridge_hit = 1.0 if flow.get('bridge_contract_hit') else 0.0

    # AML risk (lower = more reliable)
    aml = flow.get('aml_risk_score') or flow.get('aml_score_mean')
    if aml is not None:
        f.aml_risk_norm = 1.0 - float(np.clip(float(aml), 0, 1))

    # Missingness
    expected_fields = ['evidence_quality_score', 'start_time', 'end_time', 'amount_usd', 'tx_count', 'address_count']
    missing = sum(1 for ef in expected_fields if flow.get(ef) is None)
    f.missing_field_ratio = missing / len(expected_fields)

    f.feature_vector = f.to_vector()
    return f


def compute_reliability_q(
    features: ReliabilityFeatures,
    construction: str = 'Q1',
    betas: np.ndarray | None = None,
) -> float:
    """Compute q_s or q_t from reliability features.

    Args:
        features: ReliabilityFeatures for the flow
        construction: 'Q0' (uniform), 'Q1' (rule-based), 'Q2' (calibrated)
        betas: Coefficient array for Q1/Q2. If None, uses defaults.

    Returns:
        q value in [0, 1]
    """
    if construction == 'Q0':
        return 1.0

    x = features.to_vector()

    if construction == 'Q1':
        # Rule-based with frozen monotonic coefficients
        if betas is None:
            betas = np.array([
                1.5,   # evidence_quality_score (positive: more evidence -> higher q)
                0.3,   # evidence_level_present
                0.2,   # evidence_levels_count
                -0.8,  # candidate_count_norm (negative: more candidates -> lower q)
                0.5,   # cost_margin_best_vs_second (positive: clear best -> higher q)
                -0.6,  # cost_entropy_normalized (negative: high entropy -> lower q)
                0.0,   # amount_log_norm (neutral)
                -0.2,  # time_window_width_sec (negative: wide window -> lower q)
                0.3,   # has_exact_timestamp
                0.1,   # tx_count_norm (weak positive: more txs -> more evidence)
                0.1,   # address_count_norm
                0.2,   # bridge_set_size
                0.3,   # has_bridge_hit
                -0.3,  # aml_risk_norm (already inverted: lower AML -> higher -> positive beta)
                -0.5,  # missing_field_ratio (negative: more missing -> lower q)
                0.0,   # tx_count (reserved)
            ])
        # Ensure monotonicity constraints
        betas = np.array(betas)
        betas[3] = -abs(betas[3])   # candidate_count: negative
        betas[5] = -abs(betas[5])   # entropy: negative
        betas[7] = -abs(betas[7])   # time_window: negative
        betas[13] = -abs(betas[13]) # aml: already reversed, but ensure negative
        betas[14] = -abs(betas[14]) # missingness: negative

        logit = betas[0] * 0.5 + np.dot(betas[1:], x)  # bias term + features
        q = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(q, 0.01, 0.99))

    if construction == 'Q2':
        # Placeholder for calibrated (isotonic regression)
        # For now, falls back to Q1
        return compute_reliability_q(features, 'Q1', betas)

    return 1.0


def compute_all_q(
    source_flows: list[dict],
    target_flows: list[dict],
    cost_matrix: np.ndarray,
    causal_mask: np.ndarray,
    construction: str = 'Q1',
    betas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute q_s and q_t for all flows.

    Args:
        source_flows: List of source flow dicts
        target_flows: List of target flow dicts
        cost_matrix: (n_src, n_dst) cost matrix
        causal_mask: (n_src, n_dst) causal admissibility mask
        construction: Q construction method
        betas: Optional coefficients

    Returns:
        (q_s, q_t) arrays
    """
    n_src = len(source_flows)
    n_dst = len(target_flows)

    q_s = np.zeros(n_src, dtype=float)
    q_t = np.zeros(n_dst, dtype=float)

    for i in range(n_src):
        feat = extract_reliability_features_from_flow(
            source_flows[i],
            candidate_costs=cost_matrix[i, :] if cost_matrix is not None else None,
            causal_mask_row=causal_mask[i, :] if causal_mask is not None else None,
        )
        q_s[i] = compute_reliability_q(feat, construction, betas)

    for j in range(n_dst):
        feat = extract_reliability_features_from_flow(
            target_flows[j],
            candidate_costs=cost_matrix[:, j] if cost_matrix is not None else None,
            causal_mask_row=causal_mask[:, j] if causal_mask is not None else None,
        )
        q_t[j] = compute_reliability_q(feat, construction, betas)

    return q_s, q_t