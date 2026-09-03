from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig, TargetConfig


def build_targets(
    bars: pd.DataFrame, costs: CostConfig, config: TargetConfig
) -> pd.DataFrame:
    """Label at close[t], assuming entry at open[t+1] and exit open[t+1+horizon]."""
    entry = bars["open"].shift(-1)
    exit_price = bars["open"].shift(-(config.horizon_bars + 1))
    future_return = exit_price / entry - 1
    past_vol = np.log(bars["close"]).diff().rolling(config.volatility_window).std(ddof=0)
    threshold = pd.concat(
        [
            pd.Series(config.minimum_threshold, index=bars.index),
            past_vol * np.sqrt(config.horizon_bars) * config.volatility_multiple,
            pd.Series(costs.round_trip_rate * config.cost_safety_multiple, index=bars.index),
        ],
        axis=1,
    ).max(axis=1)
    label = pd.Series(0, index=bars.index, dtype="int8")
    label[future_return > threshold] = 1
    label[future_return < -threshold] = -1
    label[future_return.isna()] = pd.NA
    return pd.DataFrame(
        {"future_return": future_return, "threshold": threshold, "label": label},
        index=bars.index,
    )

