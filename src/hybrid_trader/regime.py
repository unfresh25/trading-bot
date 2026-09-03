from __future__ import annotations

import pandas as pd


def deterministic_regime(features: pd.DataFrame) -> pd.DataFrame:
    trend = features["distance_ema_20"]
    vol = features["volatility_20"]
    vol_cutoff = vol.rolling(60, min_periods=20).median()
    trending = trend.abs() > features["atr_14_norm"]
    high_vol = vol > vol_cutoff
    labels = pd.Series("sideways_low_vol", index=features.index, dtype="string")
    labels[trending & ~high_vol] = "trend_low_vol"
    labels[trending & high_vol] = "trend_high_vol"
    labels[~trending & high_vol] = "sideways_high_vol"
    return pd.DataFrame(
        {"regime": labels, "trend_score": trend, "volatility_score": vol}, index=features.index
    )

