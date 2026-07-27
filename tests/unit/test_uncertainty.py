import numpy as np

from drugmatch.uncertainty import SplitConformalInterval


def test_conformal_returns_ordered_bounds() -> None:
    true = np.arange(20, dtype=float)
    pred = true + 0.5
    interval = SplitConformalInterval().fit(true, pred)
    lower, upper = interval.predict(np.array([1.0, 2.0]))
    assert np.all(lower < upper)
