import json

import pandas as pd

from hybrid_trader.engine import BacktestResult
from hybrid_trader.reporting import compare_markets, write_report


class FixedRunner:
    def run(self, symbol, bars):
        return BacktestResult(
            trades=pd.DataFrame([{"net_pnl": 1.0}]),
            predictions=bars.iloc[:3],
            metrics={"net_return": 0.02 if symbol == "SPY" else 0.03},
        )


def test_comparison_reports_strategy_and_buy_hold_per_market(bars):
    report = compare_markets({"SPY": bars, "BTC/USD": bars}, FixedRunner())

    assert set(report) == {"SPY", "BTC/USD"}
    assert report["SPY"]["strategy"]["net_return"] == 0.02
    assert report["BTC/USD"]["strategy"]["net_return"] == 0.03
    assert "net_return" in report["SPY"]["buy_and_hold"]
    assert set(report["SPY"]["simple_strategies"]) == {
        "momentum_sma_20",
        "mean_reversion_zscore_20",
    }
    assert report["SPY"]["raw_observations"] == len(bars)
    assert report["SPY"]["usable_observations"] < len(bars)
    assert report["BTC/USD"]["usable_observations"] == len(bars)
    assert report["BTC/USD"]["zero_volume_policy"] == "exclude_training_and_veto_execution"
    assert "zero_volume_sensitivity" not in report["BTC/USD"]


def test_report_is_persisted_as_json_artifact(tmp_path):
    path = tmp_path / "report.json"

    write_report(path, {"SPY": {"net_return": 0.1}})

    assert json.loads(path.read_text(encoding="utf-8"))["SPY"]["net_return"] == 0.1
