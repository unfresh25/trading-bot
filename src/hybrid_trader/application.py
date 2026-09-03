from __future__ import annotations

from typing import Protocol

import pandas as pd

from .config import AppConfig
from .engine import BacktestResult, run_walk_forward


class BacktestRunner(Protocol):
    def run(self, symbol: str, bars: pd.DataFrame) -> BacktestResult: ...


class WalkForwardBacktestRunner:
    """Application service adapting the research engine to delivery mechanisms."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()

    def run(self, symbol: str, bars: pd.DataFrame) -> BacktestResult:
        del symbol  # The V0 portfolio evaluates each supplied instrument independently.
        return run_walk_forward(bars, self._config)

