from datetime import UTC, datetime

import duckdb

from hybrid_trader.audit import DecisionEvent, DuckDBDecisionStore
from hybrid_trader.contracts import Decision


def test_decision_event_round_trip(tmp_path):
    database = tmp_path / "audit.duckdb"
    store = DuckDBDecisionStore(database)
    event_id = store.append(
        DecisionEvent(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            symbol="SPY",
            decision=Decision.NO_TRADE,
            reason="low_model_confidence",
            market_data_version="fixture-v1",
            feature_set_version="features-v1",
            model_version="logistic-v1",
            payload={"probability_up": 0.51},
        )
    )
    with duckdb.connect(str(database)) as connection:
        row = connection.execute(
            "SELECT event_id, symbol, decision, reason FROM decisions"
        ).fetchone()
    assert row == (event_id, "SPY", "NO_TRADE", "low_model_confidence")
