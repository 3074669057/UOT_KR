"""Tests for compute_ece."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.compute_ece import compute_ece


class TestComputeECE:

    def test_perfect_calibration(self):
        """Scores match bin accuracy -> ECE near 0."""
        np.random.seed(42)
        n = 1000
        scores = np.random.uniform(0, 1, n)
        correct = (np.random.uniform(0, 1, n) < scores).astype(int)
        ece10, df10 = compute_ece(scores, correct, 10)
        ece15, df15 = compute_ece(scores, correct, 15)
        assert ece10 < 0.10, f"ECE@10={ece10:.4f} should be small"
        assert ece15 < 0.10, f"ECE@15={ece15:.4f} should be small"
        assert len(df10) == 10
        assert len(df15) == 15

    def test_overconfident(self):
        n = 1000
        scores = np.full(n, 0.95)
        correct = np.zeros(n, dtype=int)
        correct[:300] = 1
        ece, df = compute_ece(scores, correct, 10)
        assert ece > 0.40, f"ECE={ece:.4f} should be large for overconfident"

    def test_score_1_enters_last_bin(self):
        scores = np.array([1.0, 1.0, 0.0, 0.0])
        correct = np.array([1, 0, 1, 0])
        ece, df = compute_ece(scores, correct, 10)
        last_bin = df[df["bin"] == 9].iloc[0]
        assert last_bin["n"] == 2

    def test_empty_bin(self):
        scores = np.array([0.05, 0.15, 0.25])
        correct = np.array([0, 1, 1])
        ece, df = compute_ece(scores, correct, 5)
        empty_bins = df[df["n"] == 0]
        assert len(empty_bins) > 0
        for _, row in empty_bins.iterrows():
            assert row["ece_contrib"] == 0.0

    def test_m10_and_m15(self):
        scores = np.random.uniform(0, 1, 200)
        correct = (np.random.uniform(0, 1, 200) < scores).astype(int)
        for m in [10, 15]:
            ece, df = compute_ece(scores, correct, m)
            assert isinstance(ece, float)
            assert len(df) == m

    def test_score_out_of_range_raises(self):
        scores = np.array([0.5, 1.2, 0.3])
        correct = np.array([1, 0, 1])
        with pytest.raises(ValueError):
            compute_ece(scores, correct, 10)

    def test_negative_score_raises(self):
        scores = np.array([0.5, -0.1, 0.3])
        correct = np.array([1, 0, 1])
        with pytest.raises(ValueError):
            compute_ece(scores, correct, 10)

    def test_non_binary_correct_raises(self):
        scores = np.array([0.5, 0.6])
        correct = np.array([0, 2])
        with pytest.raises(ValueError):
            compute_ece(scores, correct, 10)

    def test_empty_input(self):
        ece, df = compute_ece(np.array([]), np.array([]), 10)
        assert ece == 0.0
        assert df.empty

    def test_nan_handling(self):
        scores = np.array([0.5, np.nan, 0.6, 0.7])
        correct = np.array([1, 0, np.nan, 1])
        with pytest.warns(UserWarning):
            ece, df = compute_ece(scores, correct, 10)
        assert df["n"].sum() == 2

    def test_all_nan(self):
        scores = np.array([np.nan, np.nan])
        correct = np.array([0, 1])
        with pytest.warns(UserWarning):
            ece, df = compute_ece(scores, correct, 10)
        assert ece == 0.0
        assert df.empty


class TestCSVInput:

    def test_csv_input_auto_infer(self, tmp_path):
        df = pd.DataFrame({
            "matchConfidence": [0.7, 0.8, 0.9],
            "correct": [1, 0, 1],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        from scripts.compute_ece import _read_input, _infer_columns
        loaded = _read_input(str(csv_path))
        sc, cc = _infer_columns(loaded)
        assert sc == "matchConfidence"
        assert cc == "correct"

    def test_jsonl_input(self, tmp_path):
        records = [{"score": 0.5, "hit": 1}, {"score": 0.9, "hit": 0}]
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        from scripts.compute_ece import _read_input
        loaded = _read_input(str(jsonl_path))
        assert len(loaded) == 2

    def test_y_true_auto_infer(self, tmp_path):
        df = pd.DataFrame({
            "score": [0.5, 0.6, 0.7],
            "y_true": [0, 1, 1],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        from scripts.compute_ece import _read_input, _infer_columns
        loaded = _read_input(str(csv_path))
        sc, cc = _infer_columns(loaded)
        assert sc == "score"
        assert cc == "y_true"

