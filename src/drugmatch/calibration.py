"""Small, serializable probability calibrators fitted on validation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    """Calibrate an already fitted classifier without touching the test set."""

    method: str = "platt"

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=int).reshape(-1)
        if len(np.unique(labels)) < 2:
            raise ValueError("Calibration requires both classes")
        if self.method == "platt":
            self.model_ = LogisticRegression(solver="lbfgs").fit(probabilities.reshape(-1, 1), labels)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip").fit(probabilities, labels)
        else:
            raise ValueError("Calibration method must be 'platt' or 'isotonic'")
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Calibrator has not been fitted")
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        if self.method == "platt":
            return self.model_.predict_proba(values.reshape(-1, 1))[:, 1]
        return np.asarray(self.model_.predict(values), dtype=float)
