"""Transparent split-conformal uncertainty for regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitConformalInterval:
    """Symmetric split-conformal prediction intervals."""

    coverage: float = 0.90

    def fit(self, true_values: np.ndarray, predictions: np.ndarray) -> SplitConformalInterval:
        if not 0 < self.coverage < 1:
            raise ValueError("Coverage must be between zero and one")
        residuals = np.abs(
            np.asarray(true_values, dtype=float) - np.asarray(predictions, dtype=float)
        )
        if residuals.size < 5:
            raise ValueError("At least five calibration residuals are required")
        quantile = min(1.0, np.ceil((residuals.size + 1) * self.coverage) / residuals.size)
        self.radius_ = float(np.quantile(residuals, quantile, method="higher"))
        return self

    def predict(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "radius_"):
            raise RuntimeError("Conformal interval has not been fitted")
        prediction = np.asarray(predictions, dtype=float)
        return prediction - self.radius_, prediction + self.radius_

    def empirical_coverage(self, true_values: np.ndarray, predictions: np.ndarray) -> float:
        lower, upper = self.predict(predictions)
        true = np.asarray(true_values, dtype=float)
        return float(np.mean((true >= lower) & (true <= upper)))
