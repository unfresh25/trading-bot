from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import pandas as pd

from .application import prepare_market_bars
from .config import AppConfig, FeatureConfig
from .contracts import ProbabilisticModel
from .engine import run_walk_forward
from .market_profiles import profile_for
from .model import LogisticBaseline, XGBoostBaseline


@dataclass(frozen=True)
class StopVariant:
    """A named risk configuration to evaluate against a baseline."""

    name: str
    atr_stop_multiple: float
    apply_intrabar_stop: bool


DEFAULT_STOP_GRID: tuple[StopVariant, ...] = (
    StopVariant("stop_2atr", 2.0, True),
    StopVariant("stop_3atr", 3.0, True),
    StopVariant("stop_5atr", 5.0, True),
    StopVariant("no_stop_size_2atr", 2.0, False),
)


def stop_ablation(
    symbol: str,
    bars: pd.DataFrame,
    variants: tuple[StopVariant, ...] = DEFAULT_STOP_GRID,
    config: AppConfig | None = None,
) -> list[dict]:
    """Run the walk-forward backtest once per stop variant on the same locked data.

    Returns one row per variant with headline metrics and the number of stop exits, so
    the effect of the stop policy is comparable across variants and markets. Signals are
    identical across variants; only sizing and exit differ.
    """
    profile = profile_for(symbol)
    base = replace(config or AppConfig(), costs=profile.costs)
    prepared = prepare_market_bars(symbol, bars)
    rows: list[dict] = []
    for variant in variants:
        settings = replace(
            base,
            risk=replace(
                base.risk,
                atr_stop_multiple=variant.atr_stop_multiple,
                apply_intrabar_stop=variant.apply_intrabar_stop,
            ),
        )
        result = run_walk_forward(
            prepared,
            settings,
            annualization_days=profile.annualization_days,
            reject_zero_volume=profile.reject_zero_volume,
        )
        rows.append(
            {
                "variant": variant.name,
                "atr_stop_multiple": variant.atr_stop_multiple,
                "apply_intrabar_stop": variant.apply_intrabar_stop,
                **_metrics_row(result),
            }
        )
    return rows


MODEL_FACTORIES: dict[str, Callable[[], ProbabilisticModel]] = {
    "logistic": LogisticBaseline,
    "xgboost": XGBoostBaseline,
}


def model_ablation(
    symbol: str,
    bars: pd.DataFrame,
    models: tuple[str, ...] = ("logistic", "xgboost"),
    config: AppConfig | None = None,
) -> list[dict]:
    """Run the walk-forward once per model on the same locked data and market profile.

    Compares each model over identical periods, costs and stop policy, so the only
    difference is the learner. Returns one row per model with headline metrics.
    """
    profile = profile_for(symbol)
    settings = replace(
        config or AppConfig(),
        costs=profile.costs,
        risk=replace(
            (config or AppConfig()).risk,
            atr_stop_multiple=profile.atr_stop_multiple,
            apply_intrabar_stop=profile.apply_intrabar_stop,
        ),
    )
    prepared = prepare_market_bars(symbol, bars)
    rows: list[dict] = []
    for name in models:
        result = run_walk_forward(
            prepared,
            settings,
            annualization_days=profile.annualization_days,
            reject_zero_volume=profile.reject_zero_volume,
            model_factory=MODEL_FACTORIES[name],
        )
        rows.append({"model": name, **_metrics_row(result)})
    return rows


def feature_ablation(
    symbol: str,
    bars: pd.DataFrame,
    variants: tuple[tuple[str, FeatureConfig], ...] = (
        ("baseline", FeatureConfig()),
        ("garch", FeatureConfig(use_garch=True, garch_model="GARCH")),
        ("egarch", FeatureConfig(use_garch=True, garch_model="EGARCH")),
        ("hmm", FeatureConfig(use_hmm_regime=True)),
    ),
    config: AppConfig | None = None,
) -> list[dict]:
    """Compare feature sets (baseline vs GARCH/EGARCH) over identical periods and policy.

    Reports calibration quality (Brier, log loss) alongside returns so a feature can be
    judged by the V1 criterion: it earns its place only if it improves out-of-sample
    calibration or return without widening the confidence interval into worse territory.
    """
    profile = profile_for(symbol)
    base = config or AppConfig()
    prepared = prepare_market_bars(symbol, bars)
    rows: list[dict] = []
    for label, feature_config in variants:
        settings = replace(
            base,
            costs=profile.costs,
            risk=replace(
                base.risk,
                atr_stop_multiple=profile.atr_stop_multiple,
                apply_intrabar_stop=profile.apply_intrabar_stop,
            ),
            features=feature_config,
        )
        result = run_walk_forward(
            prepared,
            settings,
            annualization_days=profile.annualization_days,
            reject_zero_volume=profile.reject_zero_volume,
        )
        rows.append({"features": label, **_metrics_row(result)})
    return rows


def _metrics_row(result) -> dict:
    trades = result.trades
    ci = result.diagnostics.get("net_return_block_bootstrap_95", {})
    probability = result.diagnostics.get("probability", {})
    return {
        "trades": len(trades),
        "stops": (
            int(trades["exited_on_stop"].sum())
            if "exited_on_stop" in trades.columns
            else 0
        ),
        "net_return": result.metrics["net_return"],
        "max_drawdown": result.metrics["max_drawdown"],
        "sharpe": result.metrics["sharpe"],
        "sortino": result.metrics["sortino"],
        "profit_factor": result.metrics["profit_factor"],
        "brier_up": probability.get("brier_up"),
        "log_loss": probability.get("log_loss"),
        "ci_lower": ci.get("lower"),
        "ci_upper": ci.get("upper"),
    }
