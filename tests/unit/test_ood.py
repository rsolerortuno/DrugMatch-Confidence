import numpy as np

from drugmatch.ood import PCADistanceOOD


def test_ood_flags_extreme_sample() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, (100, 5))
    detector = PCADistanceOOD(n_components=3).fit(train)
    assert detector.label(np.array([[20, 20, 20, 20, 20]]))[0] == "out-of-distribution"
