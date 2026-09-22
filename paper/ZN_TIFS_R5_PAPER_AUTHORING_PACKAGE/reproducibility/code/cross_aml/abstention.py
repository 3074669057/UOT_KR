"""Abstention rules when evidence is insufficient."""
from __future__ import annotations

from typing import Any

from cross_aml.schemas import CoverageDecision, MatchResult, PairFeatures


def apply_abstention(
    result: MatchResult,
    coverage: CoverageDecision,
    features: PairFeatures,
    config: dict[str, Any] | None = None,
) -> MatchResult:
    tool = (config or {}).get("tool") or {}
    if not tool.get("abstain_when_uncovered", True):
        return result

    reasons: list[str] = []
    if coverage.tier == "Uncovered":
        result.decision = "abstain"
        reasons.append("uncovered: insufficient event-backed evidence")
    if coverage.evidence_score < float((config or {}).get("coverage", {}).get("min_evidence_score", 0.6)):
        if result.decision == "match":
            result.decision = "abstain"
        reasons.append("evidence_score below minimum")
    if features.amount_consistency < 0.3:
        result.decision = "abstain"
        reasons.append("amount consistency too weak")
    if features.time_causality_score <= 0:
        if result.decision in ("match", "low_confidence_candidate"):
            result.decision = "abstain"
        reasons.append("time causality violated")
    if features.group_ambiguity_score > 0.5:
        if result.decision == "match":
            result.decision = "low_confidence_candidate"
        reasons.append("high group ambiguity (split/merge)")
    if not coverage.high_confidence_allowed and result.decision == "match":
        result.decision = "low_confidence_candidate"
        reasons.append("high confidence not allowed for this coverage tier")

    result.explanation.extend(reasons)
    return result
