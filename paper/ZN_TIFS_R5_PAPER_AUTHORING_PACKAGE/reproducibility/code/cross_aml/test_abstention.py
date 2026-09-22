import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cross_aml.abstention import apply_abstention
from cross_aml.coverage import qualify_coverage
from cross_aml.feature_builder import build_pair_features
from cross_aml.rcuotq_matcher import score_pair
from cross_aml.schemas import CoverageDecision, Flow, PairFeatures, MatchResult


def _flow(fid: str, chain: str) -> Flow:
    return Flow(
        flow_id=fid,
        chain=chain,
        tx_hashes=["0x" + "a" * 64],
        addresses=["0xabc"],
        token_addresses=[],
        token_symbols=["USDT"],
        total_amount_raw=100,
        amount_normalized_optional=100.0,
        start_time=100,
        end_time=200,
        direction="outbound",
        risk_seed_flag=True,
    )


def test_abstention_on_uncovered():
    src, dst = _flow("s", "ethereum"), _flow("d", "bsc")
    feats = PairFeatures(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.1)
    cov = CoverageDecision(False, "Uncovered", 0.1, [], ["bridge_event"], False)
    res = score_pair(src, dst, feats, cov, {})
    res = apply_abstention(res, cov, feats, {"tool": {"abstain_when_uncovered": True}})
    assert res.decision == "abstain"
