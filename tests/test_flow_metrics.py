from __future__ import annotations

import numpy as np
import pandas as pd

from cross.domain.evaluation.flow_metrics import (
    pair_precision_recall_f1,
    unmatched_mass_detection_summary,
)


def test_pair_f1_perfect():
    pred = pd.DataFrame(
        [
            {"srcTxHash": "0xs1", "dstTxHash": "0xd1", "matchConfidenceCalibrated": 0.9},
        ]
    )
    lab = pd.DataFrame([{"srcTxhash": "0xs1", "dstTxhash": "0xd1"}])
    m = pair_precision_recall_f1(pred, lab)
    assert m["pair_f1"] == 1.0
    assert m["tp"] == 1


def test_pair_f1_wrong():
    pred = pd.DataFrame([{"srcTxHash": "0xs1", "dstTxHash": "0xbad", "matchConfidenceCalibrated": 0.5}])
    lab = pd.DataFrame([{"srcTxhash": "0xs1", "dstTxhash": "0xd1"}])
    m = pair_precision_recall_f1(pred, lab)
    assert m["tp"] == 0
    assert m["fp"] == 1


def test_unmatched_mass_summary():
    u = unmatched_mass_detection_summary(np.array([0.1, 0.2]), np.array([0.05, 0.15]))
    assert "mean_unmatched_source" in u
