import pandas as pd

from hybrid_trader.walk_forward import calendar_folds


def test_folds_are_ordered_and_disjoint():
    index = pd.date_range("2024-01-01", "2025-02-28", freq="6h", tz="UTC")
    folds = calendar_folds(index)
    assert folds
    for fold in folds:
        assert fold.train.max() < fold.calibration.min() < fold.test.min()
        assert not fold.train.intersection(fold.calibration).size
        assert not fold.calibration.intersection(fold.test).size


def test_folds_purge_labels_and_embargo_each_evaluation_period():
    index = pd.date_range("2024-01-01", "2025-02-28", freq="15min", tz="UTC")
    purge = pd.Timedelta(minutes=75)
    embargo = pd.Timedelta(minutes=15)

    fold = calendar_folds(index, purge=purge, embargo=embargo)[0]

    assert fold.calibration.min() - fold.train.max() > purge
    assert fold.test.min() - fold.calibration.max() > purge
    assert fold.calibration.min().minute == 15
    assert fold.test.min().minute == 15
