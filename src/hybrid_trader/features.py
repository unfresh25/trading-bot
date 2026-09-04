from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import validate_ohlcv
from .regime_models import hmm_regime_probabilities
from .volatility_models import conditional_volatility


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _rolling_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def build_features(raw: pd.DataFrame, feature_config: FeatureConfig | None = None) -> pd.DataFrame:
    """Build features using only the current and previous closed bars.

    28 baseline features by default. When ``feature_config.use_garch`` is set, two
    look-ahead-free GARCH volatility features are appended so their contribution can be
    measured by ablation against the baseline.
    """
    bars = validate_ohlcv(raw)
    close, high, low, volume = bars.close, bars.high, bars.low, bars.volume
    simple = close.pct_change()
    logret = np.log(close).diff()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
    mean20, std20 = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    typical = (high + low + close) / 3
    vwap20 = (typical * volume).rolling(20).sum() / volume.rolling(20).sum()
    rolling_peak = close.rolling(60).max()

    out = pd.DataFrame(index=bars.index)
    for period in (1, 5, 20, 60):
        out[f"return_{period}"] = close.pct_change(period)
    out["volatility_20"] = logret.rolling(20).std(ddof=0) * np.sqrt(20)
    out["volatility_ewma"] = logret.ewm(span=20, adjust=False).std()
    out["atr_14_norm"] = atr / close
    out["momentum_acceleration"] = simple.rolling(5).mean().diff(5)
    out["rsi_14"] = _rsi(close) / 100
    out["macd_norm"] = macd / close
    out["macd_hist_slope"] = macd_hist.diff() / close
    out["distance_sma_20"] = close / mean20 - 1
    out["distance_ema_20"] = close / close.ewm(span=20, adjust=False).mean() - 1
    out["price_zscore_20"] = (close - mean20) / std20
    out["bollinger_width"] = 4 * std20 / mean20
    out["bollinger_position"] = (close - (mean20 - 2 * std20)) / (4 * std20)
    out["relative_volume_20"] = volume / volume.rolling(20).mean()
    out["volume_change"] = volume.pct_change()
    out["distance_vwap_20"] = close / vwap20 - 1
    out["trend_slope_20"] = np.log(close).rolling(20).apply(_rolling_slope, raw=True)
    out["drawdown_60"] = close / rolling_peak - 1
    out["return_skew_20"] = logret.rolling(20).skew()
    out["return_kurtosis_20"] = logret.rolling(20).kurt()
    out["return_autocorr_20"] = logret.rolling(20).corr(logret.shift(1))
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    if feature_config is not None and feature_config.use_garch:
        garch_vol = conditional_volatility(
            close,
            model=feature_config.garch_model,
            window=feature_config.garch_window,
            refit_every=feature_config.garch_refit_every,
        )
        out["garch_vol_norm"] = garch_vol
        out["garch_vol_ratio"] = garch_vol / out["volatility_ewma"]
    if feature_config is not None and feature_config.use_hmm_regime:
        regime = hmm_regime_probabilities(
            out,
            n_states=feature_config.hmm_states,
            window=feature_config.hmm_window,
            refit_every=feature_config.hmm_refit_every,
        )
        out = out.join(regime)
    return out.replace([np.inf, -np.inf], np.nan)
