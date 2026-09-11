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
        if np.asarray(true_values).shape != np.asarray(predictions).shape:
            raise ValueError("Calibration arrays must have the same shape")
        if residuals.ndim != 1 or not np.isfinite(residuals).all():
            raise ValueError("Calibration residuals must be finite and one-dimensional")
        self.calibration_residuals_ = residuals.copy()
        # Exact one-based finite-sample order statistic. The extra infinity is
        # essential when n cannot support the requested coverage.
        rank = int(np.ceil((residuals.size + 1) * self.coverage))
        self.radius_ = (
            float(np.partition(residuals, rank - 1)[rank - 1])
            if rank <= residuals.size
            else float("inf")
        )
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
