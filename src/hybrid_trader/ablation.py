from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .application import prepare_market_bars
from .config import AppConfig
from .engine import run_walk_forward
from .market_profiles import profile_for


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
        trades = result.trades
        ci = result.diagnostics.get("net_return_block_bootstrap_95", {})
        rows.append(
            {
                "variant": variant.name,
                "atr_stop_multiple": variant.atr_stop_multiple,
                "apply_intrabar_stop": variant.apply_intrabar_stop,
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
                "ci_lower": ci.get("lower"),
                "ci_upper": ci.get("upper"),
            }
        )
    return rows
