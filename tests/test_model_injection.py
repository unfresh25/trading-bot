import numpy as np
import pandas as pd
import pytest

from hybrid_trader.config import AppConfig
from hybrid_trader.engine import run_walk_forward
from hybrid_trader.model import LogisticBaseline, XGBoostBaseline

# XGBoost is an optional dependency (needs system libomp); skip cleanly if unavailable.
xgboost = pytest.importorskip("xgboost")


def _bars():
    rng = np.random.default_rng(4)
    index = pd.date_range("2024-01-01", "2024-11-20", freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0035, len(index))))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": op, "high": np.maximum(op, close) * 1.003,
         "low": np.minimum(op, close) * 0.997, "close": close, "volume": 1000.0},
        index=index,
    )


def test_xgboost_baseline_matches_probabilistic_model_interface():
    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    labels = pd.Series(rng.choice([-1, 0, 1], size=300))
    model = XGBoostBaseline()
    assert model.version == "xgboost-v1"
    model.fit(features, labels)
    proba = model.predict_proba(features)
    # Columns are the original (-1, 0, 1) classes, rows sum to 1.
    assert list(proba.columns) == [-1, 0, 1]
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert len(proba) == len(features)


def test_engine_default_model_is_logistic_unchanged():
    # Injecting LogisticBaseline explicitly must equal the default (no behaviour change).
    bars = _bars()
    default = run_walk_forward(bars, AppConfig(), annualization_days=365)
    explicit = run_walk_forward(
        bars, AppConfig(), annualization_days=365, model_factory=LogisticBaseline
    )
    assert default.metrics["net_return"] == pytest.approx(explicit.metrics["net_return"])
    assert len(default.trades) == len(explicit.trades)


def test_engine_accepts_injected_xgboost_without_engine_changes():
    bars = _bars()
    result = run_walk_forward(
        bars, AppConfig(), annualization_days=365, model_factory=XGBoostBaseline
    )
    # The injected model drives predictions and produces a valid, distinct run.
    assert not result.predictions.empty
    assert "net_return" in result.metrics
