"""Coverage qualification for evidence-backed inference."""
from __future__ import annotations

from typing import Any

from cross_aml.schemas import CoverageDecision, Flow, PairFeatures
from cross_aml.quotient_builder import evidence_tier_for_pair
from cross_aml.utils import clip01


def qualify_coverage(
    candidate_pair: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> CoverageDecision:
    """
    Determine whether a candidate pair is covered for high-confidence matching.

    Tier A/B: may enter high-confidence RC-UOT-Q matching (if other gates pass).
    Tier C: diagnostic only by default.
    Uncovered: must abstain when abstain_when_uncovered is true.
    """
    cfg = (config or {}).get("coverage") or {}
    min_ev = float(cfg.get("min_evidence_score", 0.6))
    min_amt = float(cfg.get("min_amount_consistency", 0.7))
    max_delay_h = float(cfg.get("max_time_delay_hours", 48))
    require_token = bool(cfg.get("require_token_consistency", True))

    src: Flow = candidate_pair["source_flow"]
    dst: Flow = candidate_pair["destination_flow"]
    feats: PairFeatures = candidate_pair["features"]

    tier = evidence_tier_for_pair(src, dst, feats.evidence_score, feats.transfer_key_match)
    reasons: list[str] = []
    missing: list[str] = []

    if feats.bridge_event_score < 0.5:
        missing.append("bridge_event")
    if feats.transfer_key_match < 0.85:
        missing.append("transfer_key_match")
    if feats.amount_consistency < min_amt:
        missing.append("amount_consistency")
        reasons.append(f"amount_consistency {feats.amount_consistency:.3f} < {min_amt}")
    if require_token and feats.token_consistency < 0.5:
        missing.append("token_consistency")
    if feats.time_causality_score <= 0 and max_delay_h > 0:
        missing.append("time_causality")
        reasons.append("destination not causally after source within window")

    if tier == "Uncovered" or feats.evidence_score < min_ev:
        tier = "Uncovered"
        covered = False
        high_conf = False
    elif tier == "C":
        covered = True
        high_conf = False
        reasons.append("Tier C: diagnostic evidence only")
    else:
        covered = True
        high_conf = tier in ("A", "B") and feats.evidence_score >= min_ev

    return CoverageDecision(
        covered=covered,
        tier=tier,  # type: ignore
        evidence_score=clip01(feats.evidence_score),
        reasons=reasons,
        missing_evidence=missing,
        high_confidence_allowed=high_conf,
    )
