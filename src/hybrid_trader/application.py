from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import pandas as pd

from .config import AppConfig
from .engine import BacktestResult, run_walk_forward
from .market_data import AssetClass
from .market_profiles import profile_for
from .sessions import regular_nyse_session


class BacktestRunner(Protocol):
    def run(self, symbol: str, bars: pd.DataFrame) -> BacktestResult: ...


class WalkForwardBacktestRunner:
    """Application service adapting the research engine to delivery mechanisms."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()

    def run(self, symbol: str, bars: pd.DataFrame) -> BacktestResult:
        profile = profile_for(symbol)
        prepared = prepare_market_bars(symbol, bars)
        settings = replace(self._config, costs=profile.costs)
        return run_walk_forward(
            prepared,
            settings,
            annualization_days=profile.annualization_days,
            reject_zero_volume=profile.reject_zero_volume,
        )


def prepare_market_bars(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    profile = profile_for(symbol)
    if profile.asset_class is AssetClass.STOCK:
        return regular_nyse_session(bars)
    return bars.copy()
