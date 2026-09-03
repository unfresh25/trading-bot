from fastapi.testclient import TestClient

from hybrid_trader.api import create_app
from hybrid_trader.engine import BacktestResult


class SuccessfulBacktestRunner:
    def run(self, symbol, bars):
        assert symbol == "SPY"
        assert str(bars.index.tz) == "UTC"
        return BacktestResult(
            trades=bars.iloc[:0],
            predictions=bars.iloc[:2],
            metrics={"net_return": 0.01, "trades": 0.0},
        )


def valid_request():
    return {
        "symbol": "SPY",
        "bars": [
            {
                "timestamp": "2025-01-01T14:30:00Z",
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "volume": 1000,
            },
            {
                "timestamp": "2025-01-01T14:45:00Z",
                "open": 101,
                "high": 103,
                "low": 100,
                "close": 102,
                "volume": 1100,
            }
        ],
    }


def test_health_resource_reports_service_is_ready():
    client = TestClient(create_app(SuccessfulBacktestRunner()))

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "live_trading": False}


def test_creating_backtest_returns_resource_summary():
    client = TestClient(create_app(SuccessfulBacktestRunner()))

    response = client.post("/api/v1/backtests", json=valid_request())

    assert response.status_code == 201
    body = response.json()
    assert body["symbol"] == "SPY"
    assert body["metrics"]["net_return"] == 0.01
    assert body["prediction_count"] == 2
    assert body["trade_count"] == 0
    assert body["id"]


def test_invalid_bar_returns_stable_error_shape():
    client = TestClient(create_app(SuccessfulBacktestRunner()))
    request = valid_request()
    request["bars"][0]["high"] = 98

    response = client.post("/api/v1/backtests", json=request)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["message"] == "Request validation failed"
    assert body["path"] == "/api/v1/backtests"
    assert body["details"]
