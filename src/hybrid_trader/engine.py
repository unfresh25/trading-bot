from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AppConfig
from .costs import simulate_long_round_trip
from .features import build_features
from .model import LogisticBaseline
from .targets import build_targets
from .walk_forward import calendar_folds


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, float]


def _metrics(trades: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    if trades.empty:
        return {"net_return": 0.0, "max_drawdown": 0.0, "trades": 0.0, "win_rate": 0.0}
    equity = initial_equity + trades["net_pnl"].cumsum()
    drawdown = equity / equity.cummax() - 1
    wins = trades.loc[trades.net_pnl > 0, "net_pnl"].sum()
    losses = -trades.loc[trades.net_pnl < 0, "net_pnl"].sum()
    return {
        "net_return": float(equity.iloc[-1] / initial_equity - 1),
        "max_drawdown": float(-drawdown.min()),
        "trades": float(len(trades)),
        "win_rate": float((trades.net_pnl > 0).mean()),
        "profit_factor": float(wins / losses) if losses else float("inf"),
        "expectancy": float(trades.net_pnl.mean()),
        "costs": float(trades.cost.sum()),
    }


def run_walk_forward(bars: pd.DataFrame, config: AppConfig | None = None) -> BacktestResult:
    """Research backtest. Every signal at close[t] fills no earlier than open[t+1]."""
    config = config or AppConfig()
    features = build_features(bars)
    targets = build_targets(bars, config.costs, config.target)
    usable = features.join(targets).dropna()
    feature_columns = list(features.columns)
    folds = calendar_folds(usable.index)
    if not folds:
        raise ValueError("At least eight calendar months of usable data are required")

    predictions: list[pd.DataFrame] = []
    for fold_id, fold in enumerate(folds):
        train = usable.loc[fold.train]
        test = usable.loc[fold.test]
        if train.label.nunique() < 2 or test.empty:
            continue
        model = LogisticBaseline()
        model.fit(train[feature_columns], train.label.astype(int))
        probability = model.predict_proba(test[feature_columns])
        output = test[["future_return", "threshold", "label"]].copy()
        output["probability_up"] = probability.get(1, 0.0)
        output["fold"] = fold_id
        predictions.append(output)
    if not predictions:
        raise ValueError("No valid walk-forward folds contained enough target classes")
    predicted = pd.concat(predictions).sort_index()

    equity = config.risk.initial_equity
    peak = equity
    trade_rows: list[dict[str, float | pd.Timestamp]] = []
    next_available = bars.index.min()
    for timestamp, row in predicted.iterrows():
        if timestamp < next_available or row.probability_up < config.strategy.minimum_up_probability:
            continue
        if equity / peak - 1 <= -config.risk.max_drawdown:
            break
        position = bars.index.get_loc(timestamp)
        exit_position = position + config.target.horizon_bars + 1
        if exit_position >= len(bars):
            continue
        entry_mid = float(bars.open.iloc[position + 1])
        exit_mid = float(bars.open.iloc[exit_position])
        atr = float((bars.high - bars.low).rolling(14).mean().iloc[position])
        if not np.isfinite(atr) or atr <= 0:
            continue
        stop_distance = config.risk.atr_stop_multiple * atr
        quantity = min(
            equity * config.risk.risk_fraction / stop_distance,
            equity * config.risk.max_position_fraction / entry_mid,
        )
        fill = simulate_long_round_trip(entry_mid, exit_mid, quantity, config.costs)
        equity += fill.net_pnl
        peak = max(peak, equity)
        next_available = bars.index[exit_position]
        trade_rows.append(
            {
                "signal_time": timestamp,
                "entry_time": bars.index[position + 1],
                "exit_time": bars.index[exit_position],
                "quantity": quantity,
                "entry_fill": fill.entry_fill,
                "exit_fill": fill.exit_fill,
                "cost": (entry_mid - fill.entry_fill + fill.exit_fill - exit_mid) * -quantity
                + fill.commission,
                "net_pnl": fill.net_pnl,
                "equity": equity,
            }
        )
    trades = pd.DataFrame(trade_rows)
    return BacktestResult(trades, predicted, _metrics(trades, config.risk.initial_equity))
