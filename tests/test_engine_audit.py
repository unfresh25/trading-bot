import numpy as np
import pandas as pd

from hybrid_trader.config import AppConfig
from hybrid_trader.engine import run_walk_forward


def test_every_out_of_sample_prediction_has_an_auditable_decision():
    rng = np.random.default_rng(22)
    index = pd.date_range("2024-01-01", "2025-02-01", freq="6h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )

    result = run_walk_forward(bars, AppConfig(), annualization_days=365)

    assert len(result.decisions) == len(result.predictions)
    assert set(result.decisions.decision) <= {"BUY", "NO_TRADE"}
    assert result.diagnostics["class_distribution"]
    assert "calibration_up" in result.diagnostics


def test_zero_volume_predictions_are_never_executed():
    rng = np.random.default_rng(23)
    index = pd.date_range("2024-01-01", "2025-02-01", freq="6h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    bars.loc[bars.index[::10], "volume"] = 0

    result = run_walk_forward(
        bars, AppConfig(), annualization_days=365, reject_zero_volume=True
    )

    zero_volume = result.decisions[result.decisions.volume == 0]
    assert not zero_volume.empty
    assert (zero_volume.decision == "NO_TRADE").all()
    assert (zero_volume.reason == "zero_volume").all()
