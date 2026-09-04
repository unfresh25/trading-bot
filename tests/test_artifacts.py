import json

import pandas as pd

from hybrid_trader.artifacts import BacktestArtifactRepository
from hybrid_trader.engine import BacktestResult


def test_backtest_artifacts_persist_complete_audit_tables(tmp_path):
    index = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
    result = BacktestResult(
        trades=pd.DataFrame({"signal_time": index[:1], "net_pnl": [1.0]}),
        predictions=pd.DataFrame({"probability_up": [0.6, 0.7]}, index=index),
        metrics={"net_return": 0.01},
        decisions=pd.DataFrame({"decision": ["BUY", "NO_TRADE"]}, index=index),
    )

    paths = BacktestArtifactRepository(tmp_path).save("BTC/USD", result)

    assert len(pd.read_parquet(paths.predictions)) == 2
    assert len(pd.read_parquet(paths.decisions)) == 2
    assert len(pd.read_parquet(paths.trades)) == 1
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["symbol"] == "BTC/USD"
    assert all(item["sha256"] for item in manifest["files"].values())


def _result(symbol_seed: int = 0):
    index = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
    return BacktestResult(
        trades=pd.DataFrame({"signal_time": index[:1], "net_pnl": [1.0 + symbol_seed]}),
        predictions=pd.DataFrame({"probability_up": [0.6, 0.7]}, index=index),
        metrics={"net_return": 0.01},
        decisions=pd.DataFrame({"decision": ["BUY", "NO_TRADE"]}, index=index),
    )


def test_similar_symbols_do_not_collide(tmp_path):
    repo = BacktestArtifactRepository(tmp_path)
    a = repo.save("BTC/USD", _result(1))
    b = repo.save("BTC-USD", _result(2))
    # Distinct symbols must land in distinct directories and not overwrite each other.
    assert a.manifest.parent.parent != b.manifest.parent.parent
    assert json.loads(a.manifest.read_text())["symbol"] == "BTC/USD"
    assert json.loads(b.manifest.read_text())["symbol"] == "BTC-USD"


def test_reruns_are_versioned_immutably(tmp_path):
    repo = BacktestArtifactRepository(tmp_path)
    first = repo.save("SPY", _result(1))
    changed = repo.save("SPY", _result(2))  # different trades -> different content hash
    assert first.manifest.parent != changed.manifest.parent  # separate version dirs
    # The first run's artifacts still exist untouched.
    assert first.manifest.exists()


def test_manifest_carries_reproducibility_fields(tmp_path):
    from hybrid_trader.config import AppConfig

    paths = BacktestArtifactRepository(tmp_path).save(
        "SPY", _result(1), config=AppConfig(), input_sha256="abc123"
    )
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["input_sha256"] == "abc123"
    assert manifest["bootstrap_seed"] == 42
    assert manifest["config"]["risk"]["atr_stop_multiple"] == 2.0
    assert manifest["period"]["start"] and manifest["period"]["end"]
    assert "git_commit" in manifest  # None outside a git repo is acceptable
