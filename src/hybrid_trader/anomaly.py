from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826  # scales median-absolute-deviation to a normal-consistent sigma


def robust_anomaly_score(
    features: pd.DataFrame,
    columns: tuple[str, ...] = ("return_1", "volatility_20", "relative_volume_20"),
    window: int = 250,
) -> pd.Series:
    """Rolling robust z-score magnitude across a few features, computed causally.

    For each column the robust z-score is ``(x - rolling median) / (MAD * 1.4826)`` using
    only the trailing ``window`` (shifted by one bar so the current value never enters its
    own baseline). The per-bar anomaly score is the max absolute robust z across columns,
    so a flash crash, a volatility spike, or an abnormal volume surge all raise it. NaN
    during warm-up. No future information is used.
    """
    scores = []
    for column in columns:
        series = features[column]
        baseline = series.shift(1)
        median = baseline.rolling(window, min_periods=window // 2).median()
        mad = (baseline - median).abs().rolling(window, min_periods=window // 2).median()
        sigma = (mad * MAD_TO_SIGMA).replace(0, np.nan)
        scores.append(((series - median) / sigma).abs())
    return pd.concat(scores, axis=1).max(axis=1).rename("anomaly_score")
