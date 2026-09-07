from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Market-implied expectation series from FRED. Their daily change around a macro release is
# a genuine, point-in-time surprise: the market re-prices these instruments when a print
# beats or misses what it had discounted, without needing an analyst survey.
#   DGS2   - 2y Treasury yield: moves on rate/FOMC surprises (the price IS the expectation)
#   T10YIE - 10y breakeven inflation: moves on inflation surprises
#   T5YIFR - 5y5y forward breakeven: cleaner inflation-expectations gauge
MARKET_SERIES: tuple[str, ...] = ("DGS2", "T10YIE", "T5YIFR")


def load_market_series(
    series: tuple[str, ...] = MARKET_SERIES,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch daily market-implied expectation series from FRED (free key required).

    These are market prices, not revised statistics, so the observation date is the
    availability date — no vintage handling needed. Returns a daily frame indexed by date.
    """
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise ValueError("FRED_API_KEY is required (free key at fredaccount.stlouisfed.org)")
    fred = Fred(api_key=key)
    columns = {name: fred.get_series(name) for name in series}
    frame = pd.DataFrame(columns)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def market_surprise_features(
    market: pd.DataFrame,
    index: pd.DatetimeIndex,
    vol_window: int = 60,
) -> pd.DataFrame:
    """Point-in-time market-surprise features aligned to a bar index, without look-ahead.

    For each market series the surprise is its daily change standardised by the trailing
    ``vol_window`` volatility of daily changes (known only up to the prior day). A day's
    change is available at that day's close, so it is stamped to the next UTC day-open and
    forward-filled onto the bar index — a bar at t only ever sees changes already closed
    before t. Returns one column per series plus a combined ``market_surprise`` (mean
    absolute standardised change) and a signed ``rate_surprise`` / ``infl_surprise`` split.
    """
    daily = market.copy()
    daily.index = pd.to_datetime(daily.index, utc=True)
    changes = daily.diff()
    vol = changes.abs().rolling(vol_window, min_periods=vol_window // 2).mean().shift(1)
    standardised = (changes / vol.replace(0, np.nan)).clip(-5, 5)
    # Available at close of day d -> usable from the next day.
    standardised.index = standardised.index + pd.Timedelta(days=1)
    aligned = standardised.reindex(standardised.index.union(index)).ffill().reindex(index)

    out = pd.DataFrame(index=index)
    for name in market.columns:
        out[f"surprise_{name}"] = aligned[name]
    out["rate_surprise"] = out.get("surprise_DGS2", 0.0)
    infl_cols = [c for c in ("surprise_T10YIE", "surprise_T5YIFR") if c in out]
    out["infl_surprise"] = out[infl_cols].mean(axis=1) if infl_cols else 0.0
    out["market_surprise"] = out[[f"surprise_{c}" for c in market.columns]].abs().mean(axis=1)
    return out
