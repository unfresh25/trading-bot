from __future__ import annotations

import io
import zipfile

import httpx
import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"

# Binance publishes derivatives market data as monthly/daily ZIPs of CSV. These are
# recorded market facts (funding paid, open interest snapshot), immutable by nature — the
# funding for a past date is never recalculated, so there is no point-in-time risk.


def _download_csv(url: str, client: httpx.Client) -> pd.DataFrame | None:
    response = client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            return pd.read_csv(handle)


def load_funding_rate(
    symbol: str,
    months: list[str],
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Funding-rate history for a perp, one row per 8h settlement.

    ``months`` are 'YYYY-MM' strings. Returns a frame indexed by the funding calc time
    (UTC) with a ``funding_rate`` column. Missing months are skipped.
    """
    owned = client or httpx.Client(timeout=60)
    try:
        frames = []
        for month in months:
            url = f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
            csv = _download_csv(url, owned)
            if csv is not None and len(csv):
                frames.append(csv)
        if not frames:
            return pd.DataFrame(columns=["funding_rate"], index=pd.DatetimeIndex([], tz="UTC"))
        combined = pd.concat(frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["calc_time"], unit="ms", utc=True)
        return (
            combined.set_index("time")[["last_funding_rate"]]
            .rename(columns={"last_funding_rate": "funding_rate"})
            .sort_index()
        )
    finally:
        if client is None:
            owned.close()


def load_metrics(
    symbol: str,
    days: list[str],
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Open-interest and long/short-ratio history at 5-minute granularity.

    ``days`` are 'YYYY-MM-DD' strings. Returns a frame indexed by snapshot time (UTC) with
    open interest and taker/top-trader long-short ratios. Missing days are skipped.
    """
    owned = client or httpx.Client(timeout=60)
    try:
        frames = []
        for day in days:
            url = f"{BASE}/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
            csv = _download_csv(url, owned)
            if csv is not None and len(csv):
                frames.append(csv)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["create_time"], utc=True)
        cols = {
            "sum_open_interest": "open_interest",
            "sum_toptrader_long_short_ratio": "toptrader_ls_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ls_ratio",
        }
        return combined.set_index("time")[list(cols)].rename(columns=cols).sort_index()
    finally:
        if client is None:
            owned.close()


def derivatives_features(
    funding: pd.DataFrame,
    metrics: pd.DataFrame,
    index: pd.DatetimeIndex,
    z_window: int = 96,
) -> pd.DataFrame:
    """Point-in-time BTC-derivatives features aligned to a bar index, without look-ahead.

    Builds features that OHLCV does not contain — leveraged positioning and its carry cost:
    funding level, funding z-score (crowding of the long side), open-interest change, the
    taker long/short ratio, and funding*OI-change (a crowding-plus-buildup proxy). Every
    input is forward-filled from its own timestamp onto the bar index and then shifted one
    bar, so a bar at t only sees derivatives data recorded strictly before t. ``z_window``
    is in bars (96 * 15min = 24h).
    """
    out = pd.DataFrame(index=index)

    if len(funding):
        fr = funding["funding_rate"].reindex(funding.index.union(index)).ffill().reindex(index)
        fr_mean = fr.rolling(z_window, min_periods=z_window // 2).mean()
        fr_std = fr.rolling(z_window, min_periods=z_window // 2).std(ddof=0).replace(0, np.nan)
        out["funding_rate"] = fr
        out["funding_z"] = (fr - fr_mean) / fr_std
    else:
        out["funding_rate"] = 0.0
        out["funding_z"] = 0.0

    if len(metrics):
        oi = metrics["open_interest"].reindex(metrics.index.union(index)).ffill().reindex(index)
        out["oi_change"] = oi.pct_change(z_window)
        out["taker_ls_ratio"] = (
            metrics["taker_ls_ratio"].reindex(metrics.index.union(index)).ffill().reindex(index)
        )
    else:
        out["oi_change"] = 0.0
        out["taker_ls_ratio"] = 1.0

    out["crowding"] = out["funding_z"].fillna(0.0) * out["oi_change"].fillna(0.0)
    # Shift one bar: features are only usable from the bar after they were recorded.
    return out.shift(1).replace([np.inf, -np.inf], np.nan)
