"""Reliability calibrator for RC-UOT-v2.1. Monotonic logistic regression."""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ReliabilityCalibrator:
    """Calibrates reliability features to produce q values."""

    def __init__(self, method: str = 'monotonic_logistic'):
        self.method = method
        self.model = None
        self.fitted = False
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit calibrator on development data.

        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) binary proxy labels (1 = reliable, 0 = unreliable)
        """
        if self.method == 'monotonic_logistic':
            self.model = LogisticRegression(
                penalty='l1', C=0.1, solver='saga', max_iter=5000, random_state=42
            )
            self.model.fit(X, y)
            self.feature_importances_ = np.abs(self.model.coef_[0])
        elif self.method == 'isotonic':
            self.model = IsotonicRegression(out_of_bounds='clip', increasing=True)
            # Use mean feature as predictor for isotonic
            x_mean = X.mean(axis=1)
            self.model.fit(x_mean, y)
        self.fitted = True
        return self

    def predict_q(self, X: np.ndarray) -> np.ndarray:
        """Predict q values in [0, 1]."""
        if not self.fitted:
            return np.full(X.shape[0], 0.5)
        if self.method == 'monotonic_logistic':
            probs = self.model.predict_proba(X)[:, 1]
            return np.clip(probs, 0.01, 0.99)
        elif self.method == 'isotonic':
            x_mean = X.mean(axis=1)
            return np.clip(self.model.predict(x_mean), 0.01, 0.99)
        return np.full(X.shape[0], 0.5)