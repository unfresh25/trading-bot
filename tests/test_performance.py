import numpy as np
import pandas as pd
import pytest

from hybrid_trader.performance import daily_returns, performance_metrics


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


def _sparse_trades() -> pd.DataFrame:
    return pd.DataFrame(
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


def test_daily_returns_active_days_are_not_diluted_by_calendar_zeros():
    trades = _sparse_trades()
    start, end = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-07", tz="UTC")
    calendar = daily_returns(trades, 500, start, end, include_flat=True)
    active = daily_returns(trades, 500, start, end, include_flat=False)

    # Only trade-closing days survive, so weekend/idle zeros cannot shrink the deviation.
    assert (active != 0).all()
    assert len(active) == 3
    assert len(calendar) == 7
    # The compounded net return is invariant to the dropped zero days.
    assert np.prod(1 + active) == pytest.approx(np.prod(1 + calendar))
    # Removing the diluting zeros raises the Sharpe magnitude toward the honest value.
    sharpe = lambda r: np.sqrt(252) * r.mean() / r.std(ddof=0)
    assert abs(sharpe(active)) > abs(sharpe(calendar))


def test_metrics_use_undiluted_active_day_sharpe():
    trades = _sparse_trades()
    metrics = performance_metrics(
        trades,
        initial_equity=500,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-07", tz="UTC"),
        annualization_days=252,
    )
    active = daily_returns(
        trades, 500, pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-07", tz="UTC"), include_flat=False,
    )
    expected = float(np.sqrt(252) * active.mean() / active.std(ddof=0))
    assert metrics["sharpe"] == pytest.approx(expected)
