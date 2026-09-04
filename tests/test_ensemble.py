import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from hybrid_trader.config import AppConfig
from hybrid_trader.engine import run_walk_forward
from hybrid_trader.model import EnsembleModel, LogisticBaseline


def test_ensemble_averages_to_a_valid_distribution():
    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    labels = pd.Series(rng.choice([-1, 0, 1], size=300))
    model = EnsembleModel()
    model.fit(features, labels)
    proba = model.predict_proba(features)
    assert list(proba.columns) == [-1, 0, 1]
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert model.version.startswith("ensemble-")


def test_ensemble_of_one_matches_that_model():
    rng = np.random.default_rng(2)
    features = pd.DataFrame(rng.normal(size=(200, 4)), columns=[f"f{i}" for i in range(4)])
    labels = pd.Series(rng.choice([-1, 0, 1], size=200))
    solo = LogisticBaseline()
    solo.fit(features, labels)
    ensemble = EnsembleModel(factories=(LogisticBaseline,))
    ensemble.fit(features, labels)
    pd.testing.assert_frame_equal(
        solo.predict_proba(features).reindex(columns=[-1, 0, 1], fill_value=0.0),
        ensemble.predict_proba(features),
        check_like=True,
    )


def test_engine_accepts_injected_ensemble():
    rng = np.random.default_rng(4)
    index = pd.date_range("2024-01-01", "2024-11-20", freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0035, len(index))))
    op = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {"open": op, "high": np.maximum(op, close) * 1.003,
         "low": np.minimum(op, close) * 0.997, "close": close, "volume": 1000.0},
        index=index,
    )
    result = run_walk_forward(
        bars, AppConfig(), annualization_days=365, model_factory=EnsembleModel
    )
    assert not result.predictions.empty
    assert "net_return" in result.metrics
