"""Stable prediction API for serialized DrugMatch bundles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from drugmatch.explain import local_explanation


@dataclass(frozen=True)
class PredictionResult:
    drug: str
    predicted_auc: float
    interval_lower: float
    interval_upper: float
    predicted_class: str
    response_zone: str
    model_agreement: str
    sensitivity_probability: float
    confidence: str
    ood_status: str
    top_drivers: list[dict[str, float | str]]
    feature_coverage: float
    missing_feature_count: int
    data_release: str
    model_status: str
    evidence_summary: str
    disclaimer: str


class DrugMatchPredictor:
    """Load a trained drug-specific bundle and make validated predictions."""

    def __init__(self, bundle: dict):
        self.bundle = bundle

    @classmethod
    def load(cls, path: str | Path) -> DrugMatchPredictor:
        return cls(joblib.load(path))

    @property
    def expected_features(self) -> list[str]:
        return list(self.bundle["feature_columns"])

    def _frame(self, features: pd.DataFrame | pd.Series | Mapping[str, object]) -> pd.DataFrame:
        if isinstance(features, pd.Series):
            frame = features.to_frame().T
        elif isinstance(features, pd.DataFrame):
            frame = features.copy()
        else:
            frame = pd.DataFrame([dict(features)])
        missing = [column for column in self.expected_features if column not in frame.columns]
        if missing:
            # Numerical features may be absent for user-uploaded samples and are imputed by the pipeline.
            for column in missing:
                frame[column] = np.nan if column != "meta::lineage" else "unknown"
        return frame[self.expected_features]

    def predict(
        self, features: pd.DataFrame | pd.Series | Mapping[str, object]
    ) -> PredictionResult:
        if isinstance(features, pd.Series):
            raw_frame = features.to_frame().T
        elif isinstance(features, pd.DataFrame):
            raw_frame = features.copy()
        else:
            raw_frame = pd.DataFrame([dict(features)])
        numerical_expected = [
            column for column in self.expected_features if column != "meta::lineage"
        ]
        present = [
            column
            for column in numerical_expected
            if column in raw_frame.columns and raw_frame[column].notna().any()
        ]
        feature_coverage = len(present) / max(len(numerical_expected), 1)
        missing_feature_count = len(numerical_expected) - len(present)
        frame = self._frame(raw_frame)
        if len(frame) != 1:
            raise ValueError("The public prediction API currently accepts one sample at a time")
        regression = self.bundle["regression"]
        classification = self.bundle["classification"]
        predicted_auc = float(regression.predict(frame)[0])
        response_bounds = self.bundle.get("response_bounds", {"minimum": 0.0, "maximum": 1.0})
        minimum_response = float(response_bounds.get("minimum", 0.0))
        maximum_response = float(response_bounds.get("maximum", 1.0))
        predicted_auc = float(np.clip(predicted_auc, minimum_response, maximum_response))
        lower, upper = self.bundle["conformal"].predict(np.array([predicted_auc]))
        lower = np.clip(lower, minimum_response, maximum_response)
        upper = np.clip(upper, minimum_response, maximum_response)
        raw_probability = classification.predict_proba(frame)[:, 1]
        probability = float(self.bundle["calibrator"].predict(raw_probability)[0])
        transformed = regression.named_steps["preprocess"].transform(frame)
        ood_status = self.bundle["ood"].label(transformed)[0]
        confidence = "high"
        if ood_status == "caution":
            confidence = "moderate"
        elif ood_status == "out-of-distribution":
            confidence = "low"
        if 0.4 <= probability <= 0.6:
            confidence = "low"
        if feature_coverage < 0.50:
            confidence = "low"
        elif feature_coverage < 0.80 and confidence == "high":
            confidence = "moderate"
        drivers = local_explanation(classification, frame, top_n=8).to_dict(orient="records")
        predicted_class = (
            "sensitive"
            if probability >= float(self.bundle.get("decision_threshold", 0.5))
            else "resistant"
        )
        class_thresholds = self.bundle.get("class_thresholds", {})
        sensitive_max = float(class_thresholds.get("sensitive_auc_max", minimum_response))
        resistant_min = float(class_thresholds.get("resistant_auc_min", maximum_response))
        if predicted_auc <= sensitive_max:
            response_zone = "sensitive"
        elif predicted_auc >= resistant_min:
            response_zone = "resistant"
        else:
            response_zone = "intermediate"
        if response_zone == "intermediate":
            model_agreement = "indeterminate"
            if confidence == "high":
                confidence = "moderate"
        elif response_zone == predicted_class:
            model_agreement = "concordant"
        else:
            model_agreement = "discordant"
            confidence = "low"
        return PredictionResult(
            drug=self.bundle["drug"],
            predicted_auc=predicted_auc,
            interval_lower=float(lower[0]),
            interval_upper=float(upper[0]),
            predicted_class=predicted_class,
            response_zone=response_zone,
            model_agreement=model_agreement,
            sensitivity_probability=probability,
            confidence=confidence,
            ood_status=ood_status,
            top_drivers=drivers,
            feature_coverage=float(feature_coverage),
            missing_feature_count=int(missing_feature_count),
            data_release=self.bundle["data_release"],
            model_status=str(self.bundle.get("validation_status", "unreviewed")),
            evidence_summary=str(
                self.bundle.get("evidence_summary", "Validation review not yet attached.")
            ),
            disclaimer=self.bundle["disclaimer"],
        )
