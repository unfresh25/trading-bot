import pandas as pd

from hybrid_trader.engine import BacktestResult
from hybrid_trader.reporting import compare_markets


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
    assert report["SPY"]["observations"] == len(bars)
