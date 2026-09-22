"""Deterministic stratified semi-synthetic seed sampling."""
from __future__ import annotations

import random

import pandas as pd

from cross.domain.evaluation.semi_synthetic_flows import pick_semi_synthetic_seeds


def _sample_fl(n: int = 200) -> pd.DataFrame:
    rng = __import__("numpy").random.default_rng(0)
    return pd.DataFrame(
        {
            "pattern_type": ["one_to_one"] * n,
            "src_amount_usd": rng.lognormal(0, 1, n),
            "median_delay_sec": rng.integers(60, 3600, n),
            "label_confidence": [1.0] * n,
            "src_flow_id": [f"eth_{i}" for i in range(n)],
            "dst_flow_id": [f"bnb_{i}" for i in range(n)],
        }
    )


def test_stratified_pick_same_seed_same_ids():
    fl = _sample_fl()
    r1 = random.Random(42)
    s1, _ = pick_semi_synthetic_seeds(fl, rng=r1, max_seeds=48)
    r2 = random.Random(42)
    s2, _ = pick_semi_synthetic_seeds(fl, rng=r2, max_seeds=48)
    ids1 = [str(x["src_flow_id"]) for x in s1]
    ids2 = [str(x["src_flow_id"]) for x in s2]
    assert ids1 == ids2
    assert len(ids1) == 48


def test_stratified_pick_different_seed_different_ids():
    fl = _sample_fl()
    s1, _ = pick_semi_synthetic_seeds(fl, rng=random.Random(42), max_seeds=48)
    s2, _ = pick_semi_synthetic_seeds(fl, rng=random.Random(43), max_seeds=48)
    ids1 = {str(x["src_flow_id"]) for x in s1}
    ids2 = {str(x["src_flow_id"]) for x in s2}
    assert ids1 != ids2
