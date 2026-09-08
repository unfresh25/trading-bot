from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostConfig
from .costs import simulate_long_round_trip


@dataclass(frozen=True)
class OverlayResult:
    net_return: float
    max_drawdown: float
    sharpe: float
    exposure_mean: float  # average fraction invested (1.0 = always long)
    switches: int  # number of enter/exit transitions (each costs a round trip)


def defensive_overlay(
    bars: pd.DataFrame,
    risk_signal: pd.Series,
    risk_threshold: float,
    costs: CostConfig,
    annualization_days: int = 252,
    cooldown_bars: int = 26,
) -> OverlayResult:
    """Buy & hold, but step out of the market while a risk signal is elevated.

    ``risk_signal`` is a per-bar, point-in-time risk score (higher = riskier). Exposure
    for bar t is decided from the signal at t-1 (shifted), so the overlay never acts on
    information from the bar it trades — no look-ahead. When the signal exceeds
    ``risk_threshold`` the position goes flat and stays flat for at least ``cooldown_bars``
    bars (hysteresis), preventing the churn that intrabar threshold-crossings would cause;
    it re-enters only once the signal has been calm for the whole cooldown. A round-trip
    cost is charged on every switch. Compare against the same bars held passively
    (``risk_threshold=inf``) to see if stepping out helps rather than just churns.
    """
    close = bars["close"].astype(float)
    bar_return = close.pct_change().fillna(0.0)
    elevated = (risk_signal.reindex(close.index).shift(1) > risk_threshold).fillna(False).to_numpy()

    # Flat while risk is elevated, and for cooldown_bars after it last was (hysteresis).
    invested = np.ones(len(close))
    flat_until = -1
    for i in range(len(close)):
        if elevated[i]:
            flat_until = i + cooldown_bars
        invested[i] = 0.0 if i <= flat_until else 1.0
    invested = pd.Series(invested, index=close.index)

    gross = 1.0 + invested * bar_return
    equity = gross.cumprod()

    switches = int((invested.diff().abs() > 0).sum())
    cost_per_switch = _round_trip_cost_fraction(costs)
    equity *= (1.0 - cost_per_switch) ** switches  # apply switching costs to final equity

    daily = _daily_returns(equity, close.index)
    return OverlayResult(
        net_return=float(equity.iloc[-1] - 1.0),
        max_drawdown=float(-(equity / equity.cummax() - 1.0).min()),
        sharpe=_sharpe(daily, annualization_days),
        exposure_mean=float(invested.mean()),
        switches=switches,
    )


def _round_trip_cost_fraction(costs: CostConfig) -> float:
    fill = simulate_long_round_trip(100.0, 100.0, 1.0, costs)
    return float(-fill.net_pnl / 100.0)  # loss as a fraction of notional at constant price


def _daily_returns(equity: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    daily_equity = equity.groupby(index.normalize()).last()
    return daily_equity.pct_change().dropna()


def _sharpe(returns: pd.Series, annualization_days: int) -> float:
    deviation = returns.std(ddof=0)
    if deviation == 0 or not np.isfinite(deviation):
        return 0.0
    return float(np.sqrt(annualization_days) * returns.mean() / deviation)
