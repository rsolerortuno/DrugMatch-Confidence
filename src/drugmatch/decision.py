"""Predeclared, auditable abstention policy for preclinical model screening."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    action: str
    confidence: str
    reasons: tuple[str, ...]


def screening_decision(
    *,
    probability: float,
    threshold: float,
    interval_lower: float,
    interval_upper: float,
    sensitive_max: float,
    resistant_min: float,
    feature_coverage: float,
    ood_status: str,
    model_status: str,
    calibration_protocol: str,
    margin: float = 0.10,
) -> Decision:
    """Abstain unless a reviewed model and both endpoints support one response tail.

    Confidence is a policy label, never a probability of correctness. Parameters
    are fixed before evaluation; external labels must not tune this policy.
    """
    reasons = []
    values = [
        probability,
        threshold,
        interval_lower,
        interval_upper,
        sensitive_max,
        resistant_min,
        feature_coverage,
        margin,
    ]
    if (
        not all(math.isfinite(v) for v in values)
        or interval_lower > interval_upper
        or sensitive_max >= resistant_min
        or not 0 <= probability <= 1
        or not 0 <= threshold <= 1
        or not 0 <= feature_coverage <= 1
        or margin < 0
    ):
        return Decision("abstain", "low", ("invalid_or_unbounded_uncertainty",))
    if model_status not in {"research_candidate", "validated_demo"}:
        reasons.append("model_evidence_not_sufficient")
    if calibration_protocol != "independent_calibration_v2":
        reasons.append("independent_calibration_not_verified")
    if feature_coverage < 0.80:
        reasons.append("insufficient_feature_coverage")
    if ood_status != "in-distribution":
        reasons.append("distribution_shift_or_unknown")
    if abs(probability - threshold) < margin:
        reasons.append("probability_near_decision_threshold")
    proposed = "sensitive" if probability >= threshold else "resistant"
    supported = (proposed == "sensitive" and interval_upper <= sensitive_max) or (
        proposed == "resistant" and interval_lower >= resistant_min
    )
    if not supported:
        reasons.append("interval_does_not_support_class")
    return Decision(
        "abstain" if reasons else proposed, "low" if reasons else "moderate", tuple(reasons)
    )
