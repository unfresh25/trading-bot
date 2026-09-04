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


def test_cross_validated_selection_rejects_isotonic_on_already_calibrated_data():
    # On well-calibrated data, isotonic should NOT win: it overfits its own fit set.
    # In-sample selection (the previous bug) would have preferred it.
    rng = np.random.default_rng(0)
    n = 600
    logits = rng.normal(0, 1, (n, 3))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = pd.Series([rng.choice([-1, 0, 1], p=p) for p in proba])
    raw = pd.DataFrame(proba, columns=[-1, 0, 1])

    selection = select_probability_calibrator(raw, labels)

    assert selection.method != "isotonic"
    # The reported scores are out-of-fold, so isotonic's overfit shows as a worse score.
    assert selection.validation_log_loss["isotonic"] >= selection.validation_log_loss[selection.method]


def test_small_window_falls_back_without_crashing():
    raw = pd.DataFrame({-1: [0.6, 0.3], 0: [0.3, 0.4], 1: [0.1, 0.3]})
    labels = pd.Series([-1, 1])
    selection = select_probability_calibrator(raw, labels)
    assert selection.method in {"identity", "sigmoid", "isotonic"}
