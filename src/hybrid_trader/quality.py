from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from .data import validate_ohlcv
from .market_data import AssetClass


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    gap_count: int
    missing_bars: int | None
    zero_volume_rate: float


def assess_quality(
    bars: pd.DataFrame, asset_class: AssetClass, expected_interval: timedelta
) -> DataQualityReport:
    validated = validate_ohlcv(bars)
    if validated.empty:
        raise ValueError("Cannot assess an empty dataset")
    intervals = validated.index.to_series().diff().dropna()
    gaps = intervals[intervals > expected_interval]
    missing_bars = None
    if asset_class is AssetClass.CRYPTO:
        seconds = gaps.dt.total_seconds()
        missing_bars = int((seconds / expected_interval.total_seconds() - 1).sum())
    return DataQualityReport(
        rows=len(validated),
        start=validated.index[0],
        end=validated.index[-1],
        gap_count=len(gaps),
        missing_bars=missing_bars,
        zero_volume_rate=float((validated.volume == 0).mean()),
    )
