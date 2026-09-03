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
