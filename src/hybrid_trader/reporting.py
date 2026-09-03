from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .application import BacktestRunner
from .benchmarks import buy_and_hold, simple_strategy_benchmarks
from .config import AppConfig


def compare_markets(
    datasets: dict[str, pd.DataFrame],
    runner: BacktestRunner,
    config: AppConfig | None = None,
) -> dict[str, dict]:
    settings = config or AppConfig()
    report: dict[str, dict] = {}
    for symbol, bars in datasets.items():
        strategy = runner.run(symbol, bars)
        benchmark = buy_and_hold(bars, settings.risk.initial_equity, settings.costs)
        simple = simple_strategy_benchmarks(bars, settings.risk.initial_equity, settings.costs)
        report[symbol] = {
            "observations": len(bars),
            "strategy": strategy.metrics,
            "buy_and_hold": asdict(benchmark),
            "simple_strategies": {name: asdict(result) for name, result in simple.items()},
        }
    return report
