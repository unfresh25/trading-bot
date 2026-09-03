from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from .contracts import Decision


@dataclass(frozen=True)
class DecisionEvent:
    timestamp: datetime
    symbol: str
    decision: Decision
    reason: str
    market_data_version: str
    feature_set_version: str
    model_version: str
    payload: dict[str, Any]
    event_id: str = ""

    def normalized(self) -> dict[str, Any]:
        row = asdict(self)
        row["event_id"] = self.event_id or str(uuid4())
        row["decision"] = self.decision.value
        row["payload"] = json.dumps(self.payload, sort_keys=True, default=str)
        return row


class DuckDBDecisionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with duckdb.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS decisions (
                event_id VARCHAR PRIMARY KEY, timestamp TIMESTAMPTZ, symbol VARCHAR,
                decision VARCHAR, reason VARCHAR, market_data_version VARCHAR,
                feature_set_version VARCHAR, model_version VARCHAR, payload JSON)"""
            )

    def append(self, event: DecisionEvent) -> str:
        row = event.normalized()
        with duckdb.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row["event_id"],
                    row["timestamp"],
                    row["symbol"],
                    row["decision"],
                    row["reason"],
                    row["market_data_version"],
                    row["feature_set_version"],
                    row["model_version"],
                    row["payload"],
                ],
            )
        return row["event_id"]
