from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Revised series: their first-release value differs from the final one, so ALFRED vintage
# matters (using the revised value would be look-ahead). Each is a release SPY/BTC react to.
REVISED_SERIES: tuple[str, ...] = (
    "CPIAUCSL",  # CPI, all items
    "CPILFESL",  # core CPI
    "PAYEMS",  # nonfarm payrolls
    "UNRATE",  # unemployment rate
    "FEDFUNDS",  # fed funds rate (monthly average)
)
# Daily market rates: not revised, so the observation date IS the availability date. These
# have thousands of vintages that exceed ALFRED's all-releases cap, so fetch them plainly.
UNREVISED_SERIES: tuple[str, ...] = (
    "DGS2",  # 2y treasury
    "DGS10",  # 10y treasury
)
MACRO_SERIES: tuple[str, ...] = REVISED_SERIES + UNREVISED_SERIES


@dataclass(frozen=True)
class MacroReleaseSeries:
    """A macro indicator as first published, with the timestamp it became public.

    ``as_of`` is the ALFRED ``realtime_start`` — the first date the value was available —
    so a backtest at time t may use only rows whose ``as_of <= t``. ``value`` is the value
    as first released (no later revisions), which is exactly what a trader saw then.
    """

    name: str
    frame: pd.DataFrame  # columns: as_of (UTC), reference_date, value


def load_macro_releases(
    series: tuple[str, ...] = MACRO_SERIES,
    api_key: str | None = None,
) -> dict[str, MacroReleaseSeries]:
    """Fetch first-release macro data from ALFRED (free FRED API key required).

    Uses ``get_series_first_release`` semantics: for each observation the value as first
    published and the date it was published, so nothing is used before it existed.
    Requires ``FRED_API_KEY`` in the environment (or passed in). Kept as an adapter so the
    rest of the pipeline never talks to FRED directly.
    """
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise ValueError("FRED_API_KEY is required (free key at fredaccount.stlouisfed.org)")
    fred = Fred(api_key=key)
    out: dict[str, MacroReleaseSeries] = {}
    for name in series:
        if name in UNREVISED_SERIES:
            out[name] = _load_unrevised(fred, name)
        else:
            out[name] = _load_first_release(fred, name)
    return out


def _load_first_release(fred, name: str) -> MacroReleaseSeries:
    # all_releases carries realtime_start per observation; first release = earliest.
    releases = fred.get_series_all_releases(name)
    first = (
        releases.sort_values("realtime_start")
        .groupby("date", as_index=False)
        .first()
        .rename(columns={"date": "reference_date", "realtime_start": "as_of"})
    )
    first["as_of"] = pd.to_datetime(first["as_of"], utc=True)
    first["reference_date"] = pd.to_datetime(first["reference_date"])
    return MacroReleaseSeries(name, first[["as_of", "reference_date", "value"]])


def _load_unrevised(fred, name: str) -> MacroReleaseSeries:
    # Daily market rates are not revised: the value is known at that day's close, so the
    # availability timestamp is the observation date + 1 day (conservative).
    series = fred.get_series(name).dropna()
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(series.index, utc=True) + pd.Timedelta(days=1),
            "reference_date": pd.to_datetime(series.index),
            "value": series.to_numpy(dtype=float),
        }
    )
    return MacroReleaseSeries(name, frame.reset_index(drop=True))


def load_cached_releases(path: str) -> dict[str, MacroReleaseSeries]:
    """Read a Parquet cache written by macro_cli back into MacroReleaseSeries objects."""
    combined = pd.read_parquet(path)
    combined["as_of"] = pd.to_datetime(combined["as_of"], utc=True)
    combined["reference_date"] = pd.to_datetime(combined["reference_date"])
    out: dict[str, MacroReleaseSeries] = {}
    for name, group in combined.groupby("series"):
        out[str(name)] = MacroReleaseSeries(
            str(name), group[["as_of", "reference_date", "value"]].reset_index(drop=True)
        )
    return out


def macro_surprise_features(
    releases: dict[str, MacroReleaseSeries],
    index: pd.DatetimeIndex,
    lookback: int = 12,
) -> pd.DataFrame:
    """Point-in-time macro-surprise features aligned to a bar index, without look-ahead.

    Without a paid consensus feed, the surprise for each release is proxied as its z-score
    against the trailing ``lookback`` first-release values known at publication time — i.e.
    how far the print landed from its own recent trend. Each surprise is stamped to its
    ``as_of`` publication timestamp and forward-filled onto the bar index, so a bar at t
    only ever sees releases already public at t. Returns one column per series plus a
    combined ``macro_surprise`` (mean absolute surprise across series).
    """
    columns: dict[str, pd.Series] = {}
    for name, release in releases.items():
        frame = release.frame.sort_values("as_of").copy()
        values = frame["value"].astype(float)
        mean = values.rolling(lookback, min_periods=lookback // 2).mean().shift(1)
        std = values.rolling(lookback, min_periods=lookback // 2).std(ddof=0).shift(1)
        surprise = ((values - mean) / std.replace(0, np.nan)).to_numpy()
        stamped = pd.Series(surprise, index=pd.DatetimeIndex(frame["as_of"]))
        stamped = stamped[~stamped.index.duplicated(keep="last")].sort_index()
        columns[f"surprise_{name}"] = stamped.reindex(
            stamped.index.union(index)
        ).ffill().reindex(index)
    features = pd.DataFrame(columns, index=index)
    features["macro_surprise"] = features.abs().mean(axis=1)
    return features
