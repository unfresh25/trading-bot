import numpy as np
import pandas as pd

from hybrid_trader.calibration import select_probability_calibrator


def test_calibration_selection_uses_validation_labels_and_normalizes_probabilities():
    index = pd.date_range("2024-01-01", periods=60, freq="15min", tz="UTC")
    raw = pd.DataFrame(
        {
            -1: np.tile([0.6, 0.2, 0.1], 20),
            0: np.tile([0.3, 0.6, 0.2], 20),
            1: np.tile([0.1, 0.2, 0.7], 20),
        },
        index=index,
    )
    labels = pd.Series(np.tile([-1, 0, 1], 20), index=index)

    selection = select_probability_calibrator(raw, labels)
    calibrated = selection.calibrator.transform(raw)

    assert selection.method in {"identity", "sigmoid", "isotonic"}
    assert set(selection.validation_log_loss) == {"identity", "sigmoid", "isotonic"}
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert ((calibrated >= 0) & (calibrated <= 1)).all().all()
