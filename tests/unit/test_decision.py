import numpy as np
import pytest

from drugmatch.decision import screening_decision
from drugmatch.uncertainty import SplitConformalInterval

BASE = dict(
    probability=0.9,
    threshold=0.5,
    interval_lower=0.1,
    interval_upper=0.2,
    sensitive_max=0.25,
    resistant_min=0.75,
    feature_coverage=1.0,
    ood_status="in-distribution",
    model_status="research_candidate",
    calibration_protocol="independent_calibration_v2",
)


def test_supported_research_class_is_not_high_confidence():
    result = screening_decision(**BASE)
    assert result.action == "sensitive"
    assert result.confidence == "moderate"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"model_status": "insufficient"}, "model_evidence_not_sufficient"),
        ({"calibration_protocol": "legacy"}, "independent_calibration_not_verified"),
        ({"interval_upper": 0.8}, "interval_does_not_support_class"),
        ({"feature_coverage": 0.7}, "insufficient_feature_coverage"),
        ({"ood_status": "caution"}, "distribution_shift_or_unknown"),
        ({"threshold": 0.85}, "probability_near_decision_threshold"),
        ({"probability": float("nan")}, "invalid_or_unbounded_uncertainty"),
    ],
)
def test_independent_abstention_guards(change, reason):
    result = screening_decision(**(BASE | change))
    assert result.action == "abstain"
    assert reason in result.reasons


def test_exact_finite_sample_order_statistic():
    interval = SplitConformalInterval(0.8).fit(np.arange(1, 11.0), np.zeros(10))
    assert interval.radius_ == 9


def test_small_calibration_sample_has_unbounded_interval():
    interval = SplitConformalInterval(0.9).fit(np.arange(5.0), np.zeros(5))
    assert np.isinf(interval.radius_)


@pytest.mark.parametrize("true,pred", [([1] * 5, [0] * 6), ([np.nan] * 5, [0] * 5)])
def test_invalid_calibration_inputs_fail(true, pred):
    with pytest.raises(ValueError):
        SplitConformalInterval().fit(np.array(true), np.array(pred))
