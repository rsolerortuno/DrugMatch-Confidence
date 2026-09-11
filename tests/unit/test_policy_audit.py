from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from drugmatch.policy_audit import (
    boundary_tie_summary,
    candidate_feasibility,
    maximum_nominal_coverage,
)


def test_radius_wider_than_class_gap_can_still_support_a_tail():
    # A wide middle band comparison is not an impossibility theorem: the
    # response tails are half-lines, and this prediction lies well inside one.
    frame = pd.DataFrame(
        [
            dict(
                predicted_auc=0.1,
                interval_lower=-0.3,
                interval_upper=0.5,
                sensitive_auc_max=0.6,
                resistant_auc_min=0.8,
                sensitivity_probability=0.9,
            )
        ]
    )
    result = candidate_feasibility(frame).iloc[0]
    assert result.radius_90 > result.class_band_width
    assert result.technical_support_90


def test_same_90_percent_radius_does_not_identify_first_support_coverage():
    a = np.array([0.1] * 9 + [0.9])
    b = np.array([0.8] * 9 + [0.9])
    assert np.sort(a)[9] == np.sort(b)[9]  # exact 90% rank for n=10
    assert maximum_nominal_coverage(a, 0.2) == pytest.approx(9 / 11)
    assert maximum_nominal_coverage(b, 0.2) == 0


def test_perfect_calibration_can_support_nonzero_coverage():
    assert maximum_nominal_coverage(np.zeros(20), 0.1) == pytest.approx(20 / 21)


def test_tie_expectation_matches_all_boundary_combinations():
    frame = pd.DataFrame(
        dict(
            family=["lineage"] * 5,
            model_id=list("abcde"),
            predicted_auc=[0.1, 0.2, 0.2, 0.2, 0.9],
            true_auc=[0.1, 0.1, 0.8, 0.1, 0.9],
            sensitive_auc_max=[0.25] * 5,
        )
    )
    row = boundary_tie_summary(frame, 3)
    possible = [1 + sum(x) for x in combinations([1, 0, 1], 2)]
    assert row["hits_tie_min"] == min(possible)
    assert row["hits_tie_max"] == max(possible)
    assert row["hits_tie_expected"] == pytest.approx(np.mean(possible))
