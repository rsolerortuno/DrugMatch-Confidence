import numpy as np

from drugmatch.calibration import ProbabilityCalibrator


def test_platt_calibrator_outputs_probabilities() -> None:
    probabilities = np.linspace(0.05, 0.95, 40)
    labels = (probabilities > 0.5).astype(int)
    calibrator = ProbabilityCalibrator().fit(probabilities, labels)
    output = calibrator.predict(np.array([0.2, 0.8]))
    assert np.all((output >= 0) & (output <= 1))
