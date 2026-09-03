from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import AppConfig
from .costs import simulate_long_round_trip
from .decision_policy import SignalContext, evaluate_signal
from .diagnostics import prediction_diagnostics
from .features import build_features
from .model import LogisticBaseline
from .performance import performance_metrics, trade_breakdown
from .regime import deterministic_regime
from .targets import build_targets
from .walk_forward import calendar_folds


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, float]
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)


def run_walk_forward(
    bars: pd.DataFrame,
    config: AppConfig | None = None,
    annualization_days: int = 252,
    reject_zero_volume: bool = False,
) -> BacktestResult:
    """Run a close[t] decision process whose earliest fill is open[t+1]."""
    settings = config or AppConfig()
    features = build_features(bars)
    usable = _research_frame(bars, features, settings)
    predicted = _walk_forward_predictions(usable, list(features.columns), reject_zero_volume)
    trades, decisions = _simulate_decisions(bars, predicted, settings, reject_zero_volume)
    metrics = performance_metrics(
        trades,
        settings.risk.initial_equity,
        predicted.index.min(),
        predicted.index.max(),
        annualization_days,
    )
    diagnostics = prediction_diagnostics(predicted)
    diagnostics["trade_breakdown"] = trade_breakdown(trades)
    diagnostics["decision_counts"] = decisions.reason.value_counts().to_dict()
    return BacktestResult(trades, predicted, metrics, decisions, diagnostics)


def _research_frame(
    bars: pd.DataFrame, features: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    targets = build_targets(bars, config.costs, config.target)
    regimes = deterministic_regime(features)
    market = pd.DataFrame({"volume": bars.volume}, index=bars.index)
    return features.join(targets).join(regimes).join(market).dropna()


def _walk_forward_predictions(
    usable: pd.DataFrame, feature_columns: list[str], reject_zero_volume: bool
) -> pd.DataFrame:
    folds = calendar_folds(usable.index)
    if not folds:
        raise ValueError("At least eight calendar months of usable data are required")
    predictions = []
    for fold_id, fold in enumerate(folds):
        train = usable.loc[fold.train]
        if reject_zero_volume:
            train = train[train.volume > 0]
        test = usable.loc[fold.test]
        prediction = _predict_fold(train, test, feature_columns, fold_id)
        if prediction is not None:
            predictions.append(prediction)
    if not predictions:
        raise ValueError("No valid walk-forward folds contained enough target classes")
    return pd.concat(predictions).sort_index()


def _predict_fold(
    train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], fold_id: int
) -> pd.DataFrame | None:
    if train.label.nunique() < 2 or test.empty:
        return None
    model = LogisticBaseline()
    model.fit(train[feature_columns], train.label.astype(int))
    probability = model.predict_proba(test[feature_columns])
    output = test[["future_return", "threshold", "label", "regime", "volume"]].copy()
    output["probability_down"] = probability.get(-1, 0.0)
    output["probability_neutral"] = probability.get(0, 0.0)
    output["probability_up"] = probability.get(1, 0.0)
    output["fold"] = fold_id
    return output


def _simulate_decisions(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    config: AppConfig,
    reject_zero_volume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = peak = config.risk.initial_equity
    unavailable_until = cooldown_until = -1
    trades: list[dict] = []
    decisions: list[dict] = []
    atr = (bars.high - bars.low).rolling(14).mean()
    for timestamp, row in predictions.iterrows():
        position = bars.index.get_loc(timestamp)
        evaluation = evaluate_signal(_signal_context(row, reject_zero_volume), config.strategy, config.costs)
        reason = evaluation.reason
        if reason != "zero_volume":
            state_rejection = _state_rejection(
                position, unavailable_until, cooldown_until, equity, peak, config
            )
            reason = state_rejection or reason
        decision = "BUY" if reason == "approved" else "NO_TRADE"
        if decision == "BUY" and (not np.isfinite(atr.iloc[position]) or atr.iloc[position] <= 0):
            decision, reason = "NO_TRADE", "invalid_atr"
        decisions.append(_decision_row(timestamp, row, decision, reason, evaluation.expected_edge))
        if decision == "NO_TRADE":
            continue
        trade = _execute_trade(bars, timestamp, row, position, equity, float(atr.iloc[position]), config)
        if trade is None:
            decisions[-1].update(decision="NO_TRADE", reason="insufficient_future_bars")
            continue
        trades.append(trade)
        equity = float(trade["equity"])
        peak = max(peak, equity)
        unavailable_until = int(trade["exit_position"])
        cooldown_until = unavailable_until + config.strategy.cooldown_bars
    trade_frame = pd.DataFrame(trades).drop(columns="exit_position", errors="ignore")
    return trade_frame, pd.DataFrame(decisions).set_index("timestamp")


def _signal_context(row: pd.Series, reject_zero_volume: bool) -> SignalContext:
    return SignalContext(
        probability_up=float(row.probability_up),
        probability_down=float(row.probability_down),
        threshold=float(row.threshold),
        volume=float(row.volume),
        reject_zero_volume=reject_zero_volume,
    )


def _state_rejection(position, unavailable_until, cooldown_until, equity, peak, config):
    if equity / peak - 1 <= -config.risk.max_drawdown:
        return "drawdown_limit"
    if position < unavailable_until:
        return "position_open"
    if position < cooldown_until:
        return "cooldown"
    return None


def _decision_row(timestamp, row, decision, reason, expected_edge) -> dict:
    return {
        "timestamp": timestamp,
        "decision": decision,
        "reason": reason,
        "probability_up": float(row.probability_up),
        "probability_down": float(row.probability_down),
        "expected_edge": expected_edge,
        "volume": float(row.volume),
        "fold": int(row.fold),
        "regime": row.regime,
    }


def _execute_trade(bars, timestamp, row, position, equity, atr, config) -> dict | None:
    exit_position = position + config.target.horizon_bars + 1
    if exit_position >= len(bars):
        return None
    entry_mid = float(bars.open.iloc[position + 1])
    exit_mid = float(bars.open.iloc[exit_position])
    stop_distance = config.risk.atr_stop_multiple * atr
    quantity = min(
        equity * config.risk.risk_fraction / stop_distance,
        equity * config.risk.max_position_fraction / entry_mid,
    )
    fill = simulate_long_round_trip(entry_mid, exit_mid, quantity, config.costs)
    return {
        "signal_time": timestamp,
        "entry_time": bars.index[position + 1],
        "exit_time": bars.index[exit_position],
        "exit_position": exit_position,
        "quantity": quantity,
        "entry_fill": fill.entry_fill,
        "exit_fill": fill.exit_fill,
        "cost": (exit_mid - entry_mid) * quantity - fill.net_pnl,
        "net_pnl": fill.net_pnl,
        "equity": equity + fill.net_pnl,
        "fold": int(row.fold),
        "regime": row.regime,
    }
