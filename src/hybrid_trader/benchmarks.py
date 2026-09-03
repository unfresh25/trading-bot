from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import CostConfig
from .costs import simulate_long_round_trip
from .data import validate_ohlcv


@dataclass(frozen=True)
class BenchmarkResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    net_return: float
    net_pnl: float
    cost: float


@dataclass(frozen=True)
class StrategyBenchmarkResult:
    net_return: float
    trades: int
    cost: float


def buy_and_hold(bars: pd.DataFrame, capital: float, costs: CostConfig) -> BenchmarkResult:
    if bars.empty:
        raise ValueError("Cannot benchmark an empty dataset")
    validated = validate_ohlcv(bars)
    entry_mid = float(validated.open.iloc[0])
    exit_mid = float(validated.close.iloc[-1])
    quantity = capital / entry_mid
    fill = simulate_long_round_trip(entry_mid, exit_mid, quantity, costs)
    gross_pnl = (exit_mid - entry_mid) * quantity
    cost = gross_pnl - fill.net_pnl
    return BenchmarkResult(
        entry_time=validated.index[0],
        exit_time=validated.index[-1],
        net_return=fill.net_pnl / capital,
        net_pnl=fill.net_pnl,
        cost=cost,
    )


def simple_strategy_benchmarks(
    bars: pd.DataFrame, capital: float, costs: CostConfig
) -> dict[str, StrategyBenchmarkResult]:
    validated = validate_ohlcv(bars)
    mean = validated.close.rolling(20).mean()
    deviation = validated.close.rolling(20).std(ddof=0)
    zscore = (validated.close - mean) / deviation
    return {
        "momentum_sma_20": _simulate_rule(
            validated, validated.close > mean, validated.close <= mean, capital, costs
        ),
        "mean_reversion_zscore_20": _simulate_rule(
            validated, zscore < -1, zscore >= 0, capital, costs
        ),
    }


def _simulate_rule(
    bars: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    initial_capital: float,
    costs: CostConfig,
) -> StrategyBenchmarkResult:
    equity = initial_capital
    entry_mid: float | None = None
    total_cost = 0.0
    trades = 0
    for position in range(len(bars) - 1):
        if entry_mid is None and bool(entries.iloc[position]):
            entry_mid = float(bars.open.iloc[position + 1])
            continue
        if entry_mid is None or not bool(exits.iloc[position]):
            continue
        equity, trade_cost = _close_trade(equity, entry_mid, float(bars.open.iloc[position + 1]), costs)
        total_cost += trade_cost
        trades += 1
        entry_mid = None
    if entry_mid is not None:
        equity, trade_cost = _close_trade(equity, entry_mid, float(bars.close.iloc[-1]), costs)
        total_cost += trade_cost
        trades += 1
    return StrategyBenchmarkResult(equity / initial_capital - 1, trades, total_cost)


def _close_trade(
    equity: float, entry_mid: float, exit_mid: float, costs: CostConfig
) -> tuple[float, float]:
    quantity = equity / entry_mid
    fill = simulate_long_round_trip(entry_mid, exit_mid, quantity, costs)
    gross_pnl = (exit_mid - entry_mid) * quantity
    return equity + fill.net_pnl, gross_pnl - fill.net_pnl
