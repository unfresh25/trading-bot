from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance_metrics(
    trades: pd.DataFrame,
    initial_equity: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    annualization_days: int,
) -> dict[str, float]:
    if trades.empty:
        return _empty_metrics()
    pnl = trades.net_pnl.astype(float)
    equity = pd.concat([pd.Series([initial_equity]), initial_equity + pnl.cumsum()], ignore_index=True)
    returns_by_day = daily_returns(trades, initial_equity, start, end, include_flat=False)
    losses = -pnl[pnl < 0].sum()
    wins = pnl[pnl > 0].sum()
    average_equity = float(equity.mean())
    notional = ((trades.entry_fill + trades.exit_fill) * trades.quantity).sum()
    return {
        "net_return": float(equity.iloc[-1] / initial_equity - 1),
        "max_drawdown": float(-(equity / equity.cummax() - 1).min()),
        "trades": float(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins / losses) if losses else math.inf,
        "expectancy": float(pnl.mean()),
        "costs": float(trades.cost.sum()),
        "sharpe": _sharpe(returns_by_day, annualization_days),
        "sortino": _sortino(returns_by_day, annualization_days),
        "turnover": float(notional / average_equity),
        "max_consecutive_losses": float(_maximum_losing_streak(pnl)),
    }


def daily_returns(
    trades: pd.DataFrame,
    initial_equity: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_flat: bool = True,
) -> pd.Series:
    """Return the daily portfolio return series.

    With ``include_flat=False`` only days that actually closed a trade are kept, so
    risk-adjusted ratios (Sharpe/Sortino) are not diluted by zero-return calendar days
    (weekends for SPY, any idle day). The compounded net return is invariant to those
    zeros, so dropping them does not change ``net_return`` or the bootstrap centre.
    ``include_flat=True`` preserves the full calendar series for callers that need it.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    daily_pnl = trades.groupby(trades.exit_time.dt.normalize()).net_pnl.sum().sort_index()
    if include_flat:
        dates = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
        daily_pnl = daily_pnl.reindex(dates, fill_value=0)
    opening_equity = initial_equity + daily_pnl.cumsum().shift(fill_value=0)
    return daily_pnl / opening_equity


def _sharpe(returns: pd.Series, annualization_days: int) -> float:
    deviation = returns.std(ddof=0)
    if deviation == 0:
        return 0.0
    return float(np.sqrt(annualization_days) * returns.mean() / deviation)


def _sortino(returns: pd.Series, annualization_days: int) -> float:
    downside = returns[returns < 0]
    deviation = downside.std(ddof=0)
    if not np.isfinite(deviation) or deviation == 0:
        return 0.0
    return float(np.sqrt(annualization_days) * returns.mean() / deviation)


def _maximum_losing_streak(pnl: pd.Series) -> int:
    maximum = current = 0
    for value in pnl:
        current = current + 1 if value < 0 else 0
        maximum = max(maximum, current)
    return maximum


def _empty_metrics() -> dict[str, float]:
    return {
        key: 0.0
        for key in (
            "net_return",
            "max_drawdown",
            "trades",
            "win_rate",
            "profit_factor",
            "expectancy",
            "costs",
            "sharpe",
            "sortino",
            "turnover",
            "max_consecutive_losses",
        )
    }


def trade_breakdown(trades: pd.DataFrame) -> dict[str, dict]:
    if trades.empty:
        return {"by_fold": {}, "by_year": {}, "by_regime": {}}
    return {
        "by_fold": _summaries(trades, trades.fold),
        "by_year": _summaries(trades, trades.exit_time.dt.year),
        "by_regime": _summaries(trades, trades.regime),
    }


def _summaries(trades: pd.DataFrame, groups) -> dict[str, dict[str, float]]:
    result = {}
    for name, group in trades.groupby(groups):
        result[str(name)] = {
            "trades": float(len(group)),
            "net_pnl": float(group.net_pnl.sum()),
            "win_rate": float((group.net_pnl > 0).mean()),
        }
    return result
