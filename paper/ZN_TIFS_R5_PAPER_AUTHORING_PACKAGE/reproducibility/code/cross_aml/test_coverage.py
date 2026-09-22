import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cross_aml.coverage import qualify_coverage
from cross_aml.feature_builder import build_pair_features
from cross_aml.flow_builder import build_flows_from_events
from cross_aml.event_fetcher import EventFetcher
from cross_aml.mock_data import MOCK_DST_TX, MOCK_SRC_TX


def test_coverage_abstains_weak_pair():
    fe = EventFetcher({}, dry_run=True)
    src_ev = fe.fetch_events_for_tx("ethereum", MOCK_SRC_TX)
    dst_ev = fe.fetch_events_for_tx("bsc", MOCK_DST_TX)
    src_f = build_flows_from_events("ethereum", src_ev)[0]
    # Artificial weak dst flow with no events
    from cross_aml.schemas import Flow
    weak_dst = Flow(
        flow_id="weak",
        chain="bsc",
        tx_hashes=[MOCK_DST_TX],
        addresses=["0x0"],
        token_addresses=[],
        token_symbols=["UNKNOWN"],
        total_amount_raw=1,
        amount_normalized_optional=1.0,
        start_time=0,
        end_time=0,
        direction="inbound",
        risk_seed_flag=False,
    )
    feats = build_pair_features(src_f, weak_dst, src_ev, [])
    cov = qualify_coverage(
        {"source_flow": src_f, "destination_flow": weak_dst, "features": feats},
        {"coverage": {"min_evidence_score": 0.99, "min_amount_consistency": 0.99}},
    )
    assert cov.tier == "Uncovered" or not cov.high_confidence_allowed
