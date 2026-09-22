"""
Standalone RC-UOT-Q style matcher (rule/config-based, no training required).

This implementation is for triage inference only and is NOT identical to the
full Phase 25/29.1 training pipeline. High-confidence outputs require coverage gates.
"""
from __future__ import annotations

from typing import Any

from cross_aml.schemas import CoverageDecision, Flow, MatchResult, PairFeatures
from cross_aml.utils import clip01


def score_pair(
    src: Flow,
    dst: Flow,
    features: PairFeatures,
    coverage: CoverageDecision,
    config: dict[str, Any] | None = None,
) -> MatchResult:
    mcfg = (config or {}).get("matching") or {}
    conf_thr = float(mcfg.get("confidence_threshold", 0.7))
    precision_oriented = bool(mcfg.get("precision_oriented", True))

    # Rule-based composite (inspired by phase21 corrected_quotient_decision_score)
    match_score = clip01(
        0.30 * features.transfer_key_match
        + 0.25 * features.bridge_event_score
        + 0.20 * features.amount_consistency
        + 0.15 * features.time_causality_score
        + 0.05 * features.token_consistency
        + 0.05 * features.address_overlap_score
        - 0.10 * features.group_ambiguity_score
        - 0.05 * features.mass_diffusion_score
    )

    precision_score = clip01(
        0.45 * features.transfer_key_match
        + 0.25 * features.bridge_event_score
        + 0.15 * features.amount_consistency
        - 0.15 * features.group_ambiguity_score
    ) if precision_oriented else match_score

    confidence = clip01(match_score * (1.0 if coverage.high_confidence_allowed else 0.5))

    decision = _decision_from_scores(
        match_score,
        precision_score,
        confidence,
        coverage,
        conf_thr,
        float(mcfg.get("abstention_threshold", 0.5)),
    )

    return MatchResult(
        source_flow_id=src.flow_id,
        destination_flow_id=dst.flow_id,
        decision=decision,
        confidence=confidence,
        coverage_tier=coverage.tier,
        evidence_score=coverage.evidence_score,
        match_score=match_score,
        precision_oriented_score=precision_score,
        features=features.as_dict(),
        evidence_refs=sorted(set(src.evidence_refs + dst.evidence_refs)),
        explanation=[],
        limitations=[
            "Standalone tool scorer; not Phase 29.1 trained model.",
            "Coverage-qualified inference only; no full-scope claim.",
        ],
    )


def _decision_from_scores(
    match_score: float,
    precision_score: float,
    confidence: float,
    coverage: CoverageDecision,
    conf_thr: float,
    abstention_thr: float,
) -> str:
    if coverage.tier == "Uncovered" or not coverage.covered:
        return "abstain"
    if coverage.tier == "C":
        if match_score >= abstention_thr:
            return "diagnostic_candidate"
        return "abstain"
    if not coverage.high_confidence_allowed:
        return "low_confidence_candidate"
    if match_score >= conf_thr and precision_score >= conf_thr:
        return "match"
    if match_score < abstention_thr:
        return "non_match"
    if match_score >= abstention_thr:
        return "low_confidence_candidate"
    return "abstain"
