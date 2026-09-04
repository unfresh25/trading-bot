from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .application import BacktestRunner, prepare_market_bars
from .artifacts import BacktestArtifactRepository
from .benchmarks import buy_and_hold, simple_strategy_benchmarks
from .config import AppConfig
from .market_profiles import profile_for
from .quality import assess_quality
from .serialization import dump_json

FIFTEEN_MINUTES = timedelta(minutes=15)


def compare_markets(
    datasets: dict[str, pd.DataFrame],
    runner: BacktestRunner,
    config: AppConfig | None = None,
    artifact_repository: BacktestArtifactRepository | None = None,
) -> dict[str, dict]:
    settings = config or AppConfig()
    report: dict[str, dict] = {}
    for symbol, bars in datasets.items():
        profile = profile_for(symbol)
        prepared = prepare_market_bars(symbol, bars)
        strategy = runner.run(symbol, prepared)
        benchmark = buy_and_hold(prepared, settings.risk.initial_equity, profile.costs)
        simple = simple_strategy_benchmarks(prepared, settings.risk.initial_equity, profile.costs)
        quality = assess_quality(bars, profile.asset_class, FIFTEEN_MINUTES)
        report[symbol] = {
            "raw_observations": len(bars),
            "usable_observations": len(prepared),
            "cost_assumptions": asdict(profile.costs),
            "data_quality": asdict(quality),
            "zero_volume_policy": _zero_volume_policy(profile.reject_zero_volume),
            "strategy": strategy.metrics,
            "diagnostics": strategy.diagnostics,
            "buy_and_hold": asdict(benchmark),
            "simple_strategies": {name: asdict(result) for name, result in simple.items()},
        }
        if artifact_repository is not None:
            paths = artifact_repository.save(symbol, strategy, config=settings)
            report[symbol]["audit_artifacts"] = {
                name: str(path) for name, path in asdict(paths).items()
            }
    return report


def _zero_volume_policy(reject: bool) -> str:
    if reject:
        return "exclude_training_and_veto_execution"
    return "not_applicable"


def write_report(path: str | Path, report: dict) -> Path:
    return dump_json(report, path)
