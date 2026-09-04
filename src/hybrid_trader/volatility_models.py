from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

RETURN_SCALE = 100.0  # arch fits are numerically better on percent returns


def conditional_volatility(
    close: pd.Series,
    model: str = "GARCH",
    window: int = 750,
    refit_every: int = 50,
    min_train: int = 250,
) -> pd.Series:
    """One-step-ahead GARCH/EGARCH conditional volatility, computed without look-ahead.

    For each bar ``t`` the value is the volatility that a model trained only on returns
    up to ``t-1`` forecasts for ``t``. The model is refit every ``refit_every`` bars on a
    trailing ``window`` of returns (a rolling refit — refitting every bar over tens of
    thousands of bars is needlessly expensive and does not change the forecast much), and
    its parameters are rolled forward between refits by filtering the latest returns.

    Returns a series aligned to ``close.index`` in the same (fractional) units as the log
    returns, with NaN before enough history exists. Never uses a return at or after ``t``.
    """
    from arch import arch_model

    log_returns = np.log(close).diff() * RETURN_SCALE
    values = log_returns.to_numpy()
    n = len(values)
    forecast = np.full(n, np.nan)

    # Refit on a trailing window at every refit_every-th bar; each fit yields the
    # one-step-ahead forecast for that bar, held until the next refit. The window ends
    # at t (exclusive), so the forecast for bar t never sees the return at t.
    for t in range(min_train, n, refit_every):
        history = values[max(0, t - window) : t]
        history = history[~np.isnan(history)]
        if len(history) < min_train:
            continue
        fitted = _fit(arch_model, history, model)
        if fitted is None:
            continue
        forecast[t] = _forecast_next(fitted)
    series = pd.Series(forecast / RETURN_SCALE, index=close.index, name="garch_volatility")
    # Hold each forecast forward until the next refit (a step function of past-only info).
    return series.ffill()


def _fit(arch_model, history: np.ndarray, model: str):
    builder = (
        {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}
        if model.upper() == "EGARCH"
        else {"vol": "GARCH", "p": 1, "q": 1}
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return arch_model(history, mean="Zero", dist="normal", **builder).fit(disp="off")
    except (ValueError, np.linalg.LinAlgError):
        return None


def _forecast_next(fitted) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variance = fitted.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        return float(np.sqrt(variance))
    except (ValueError, IndexError):
        return float(np.nan)
