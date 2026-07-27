"""Simple PCA-distance out-of-distribution warnings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCADistanceOOD:
    """Flag samples far from the transformed training distribution."""

    n_components: int = 8
    caution_quantile: float = 0.95
    ood_quantile: float = 0.99

    def fit(self, X: np.ndarray) -> PCADistanceOOD:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or len(values) < 10:
            raise ValueError(
                "OOD fitting requires a two-dimensional matrix with at least ten samples"
            )
        self.scaler_ = StandardScaler().fit(values)
        standardized = self.scaler_.transform(values)
        components = min(
            self.n_components, standardized.shape[1], max(1, standardized.shape[0] - 1)
        )
        self.pca_ = PCA(n_components=components, random_state=0).fit(standardized)
        embedding = self.pca_.transform(standardized)
        self.center_ = embedding.mean(axis=0)
        distances = np.linalg.norm(embedding - self.center_, axis=1)
        self.caution_threshold_ = float(np.quantile(distances, self.caution_quantile))
        self.ood_threshold_ = float(np.quantile(distances, self.ood_quantile))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pca_"):
            raise RuntimeError("OOD detector has not been fitted")
        embedding = self.pca_.transform(self.scaler_.transform(np.asarray(X, dtype=float)))
        return np.linalg.norm(embedding - self.center_, axis=1)

    def label(self, X: np.ndarray) -> list[str]:
        labels: list[str] = []
        for distance in self.score(X):
            if distance >= self.ood_threshold_:
                labels.append("out-of-distribution")
            elif distance >= self.caution_threshold_:
                labels.append("caution")
            else:
                labels.append("in-distribution")
        return labels
