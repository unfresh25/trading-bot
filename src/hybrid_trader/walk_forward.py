from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train: pd.DatetimeIndex
    calibration: pd.DatetimeIndex
    test: pd.DatetimeIndex


def calendar_folds(
    index: pd.DatetimeIndex,
    train_months: int = 6,
    calibration_months: int = 1,
    test_months: int = 1,
) -> list[WalkForwardFold]:
    if not index.is_monotonic_increasing:
        raise ValueError("Index must be increasing")
    start = index.min().tz_localize(None).to_period("M").to_timestamp()
    last = index.max().tz_localize(None)
    folds: list[WalkForwardFold] = []
    cursor = start + pd.DateOffset(months=train_months)
    while cursor + pd.DateOffset(months=calibration_months + test_months) <= last + pd.offsets.MonthBegin(1):
        train_start = cursor - pd.DateOffset(months=train_months)
        calibration_end = cursor + pd.DateOffset(months=calibration_months)
        test_end = calibration_end + pd.DateOffset(months=test_months)
        naive = index.tz_localize(None)
        folds.append(
            WalkForwardFold(
                index[(naive >= train_start) & (naive < cursor)],
                index[(naive >= cursor) & (naive < calibration_end)],
                index[(naive >= calibration_end) & (naive < test_end)],
            )
        )
        cursor += pd.DateOffset(months=test_months)
    return [fold for fold in folds if len(fold.train) and len(fold.calibration) and len(fold.test)]

