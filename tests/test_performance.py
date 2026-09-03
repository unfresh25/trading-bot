import pandas as pd
import pytest

from hybrid_trader.performance import performance_metrics


def test_performance_reports_risk_turnover_and_losing_streak():
    trades = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(
                ["2025-01-02T20:00:00Z", "2025-01-03T20:00:00Z", "2025-01-06T20:00:00Z"]
            ),
            "net_pnl": [-1.0, -2.0, 4.0],
            "entry_fill": [100.0, 100.0, 100.0],
            "exit_fill": [99.0, 98.0, 104.0],
            "quantity": [1.0, 1.0, 1.0],
            "cost": [0.2, 0.2, 0.2],
        }
    )

    metrics = performance_metrics(
        trades,
        initial_equity=500,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-07", tz="UTC"),
        annualization_days=252,
    )

    assert metrics["max_consecutive_losses"] == 2
    assert metrics["turnover"] > 0
    assert "sharpe" in metrics
    assert "sortino" in metrics
    assert metrics["net_return"] == pytest.approx(0.002)
