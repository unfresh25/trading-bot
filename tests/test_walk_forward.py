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

