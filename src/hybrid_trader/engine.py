from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import select_probability_calibrator
from .config import AppConfig
from .costs import simulate_long_round_trip
from .decision_policy import SignalContext, evaluate_signal
from .diagnostics import prediction_diagnostics
from .features import build_features
from .model import LogisticBaseline
from .performance import daily_returns, performance_metrics, trade_breakdown
from .regime import deterministic_regime
from .targets import build_targets
from .uncertainty import block_bootstrap_net_return
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
    predicted = _walk_forward_predictions(
        usable, list(features.columns), reject_zero_volume, settings
    )
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
    if not predicted.empty:
        returns_by_day = daily_returns(
            trades,
            settings.risk.initial_equity,
            predicted.index.min(),
            predicted.index.max(),
        )
        diagnostics["net_return_block_bootstrap_95"] = block_bootstrap_net_return(
            returns_by_day
        )
        diagnostics["calibration_by_fold"] = _calibration_diagnostics(predicted)
    return BacktestResult(trades, predicted, metrics, decisions, diagnostics)


def _research_frame(
    bars: pd.DataFrame, features: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    targets = build_targets(bars, config.costs, config.target)
    regimes = deterministic_regime(features)
    market = pd.DataFrame({"volume": bars.volume}, index=bars.index)
    return features.join(targets).join(regimes).join(market).dropna()


def _walk_forward_predictions(
    usable: pd.DataFrame,
    feature_columns: list[str],
    reject_zero_volume: bool,
    config: AppConfig,
) -> pd.DataFrame:
    folds = calendar_folds(
        usable.index,
        purge=pd.Timedelta(config.target.label_span),
        embargo=pd.Timedelta(config.target.bar_interval),
    )
    if not folds:
        raise ValueError("At least eight calendar months of usable data are required")
    predictions = []
    for fold_id, fold in enumerate(folds):
        train = usable.loc[fold.train]
        if reject_zero_volume:
            train = train[train.volume > 0]
        calibration = usable.loc[fold.calibration]
        if reject_zero_volume:
            calibration = calibration[calibration.volume > 0]
        test = usable.loc[fold.test]
        prediction = _predict_fold(train, calibration, test, feature_columns, fold_id)
        if prediction is not None:
            predictions.append(prediction)
    if not predictions:
        raise ValueError("No valid walk-forward folds contained enough target classes")
    return pd.concat(predictions).sort_index()


def _predict_fold(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    fold_id: int,
) -> pd.DataFrame | None:
    if train.label.nunique() < 2 or calibration.empty or test.empty:
        return None
    model = LogisticBaseline()
    model.fit(train[feature_columns], train.label.astype(int))
    calibration_probability = model.predict_proba(calibration[feature_columns])
    selection = select_probability_calibrator(
        calibration_probability, calibration.label.astype(int)
    )
    raw_probability = model.predict_proba(test[feature_columns])
    probability = selection.calibrator.transform(raw_probability)
    output = test[["future_return", "threshold", "label", "regime", "volume"]].copy()
    output["raw_probability_down"] = raw_probability.get(-1, 0.0)
    output["raw_probability_neutral"] = raw_probability.get(0, 0.0)
    output["raw_probability_up"] = raw_probability.get(1, 0.0)
    output["probability_down"] = probability.get(-1, 0.0)
    output["probability_neutral"] = probability.get(0, 0.0)
    output["probability_up"] = probability.get(1, 0.0)
    output["fold"] = fold_id
    output["calibration_method"] = selection.method
    for method, score in selection.validation_log_loss.items():
        output[f"calibration_log_loss_{method}"] = score
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
    entry_time = timestamp + pd.Timedelta(config.target.bar_interval)
    exit_time = timestamp + pd.Timedelta(config.target.label_span)
    locations = bars.index.get_indexer([entry_time, exit_time])
    if (locations < 0).any():
        return None
    entry_position, exit_position = map(int, locations)
    entry_mid = float(bars.open.iloc[entry_position])
    stop_distance = config.risk.atr_stop_multiple * atr
    stop_price = entry_mid - stop_distance
    quantity = min(
        equity * config.risk.risk_fraction / stop_distance,
        equity * config.risk.max_position_fraction / entry_mid,
    )
    if config.risk.apply_intrabar_stop:
        exit_mid, resolved_exit, exited_on_stop = _resolve_exit(
            bars, entry_position, exit_position, stop_price
        )
    else:
        exit_mid, resolved_exit, exited_on_stop = (
            float(bars.open.iloc[exit_position]),
            exit_position,
            False,
        )
    fill = simulate_long_round_trip(entry_mid, exit_mid, quantity, config.costs)
    return {
        "signal_time": timestamp,
        "entry_time": entry_time,
        "exit_time": bars.index[resolved_exit],
        "exit_position": resolved_exit,
        "quantity": quantity,
        "entry_fill": fill.entry_fill,
        "exit_fill": fill.exit_fill,
        "exited_on_stop": exited_on_stop,
        "cost": (exit_mid - entry_mid) * quantity - fill.net_pnl,
        "net_pnl": fill.net_pnl,
        "equity": equity + fill.net_pnl,
        "fold": int(row.fold),
        "regime": row.regime,
    }


def _resolve_exit(
    bars: pd.DataFrame, entry_position: int, exit_position: int, stop_price: float
) -> tuple[float, int, bool]:
    """Return (exit_price, exit_position, exited_on_stop) honouring an intrabar stop.

    Bars held during the position are scanned in order; the first whose low pierces the
    stop closes the trade at ``stop_price`` (a conservative fill), bounding the loss to
    the risk the position was sized against. Otherwise the planned time-based exit at
    ``open[exit_position]`` stands.
    """
    held = bars.iloc[entry_position:exit_position]
    breached = held.index[held.low <= stop_price]
    if len(breached):
        stop_bar = int(bars.index.get_loc(breached[0]))
        return stop_price, stop_bar, True
    return float(bars.open.iloc[exit_position]), exit_position, False


def _calibration_diagnostics(predictions: pd.DataFrame) -> dict[str, dict]:
    result = {}
    for fold, group in predictions.groupby("fold"):
        result[str(fold)] = {
            "method": str(group.calibration_method.iloc[0]),
            "validation_log_loss": {
                method: float(group[f"calibration_log_loss_{method}"].iloc[0])
                for method in ("identity", "sigmoid", "isotonic")
            },
        }
    return result
