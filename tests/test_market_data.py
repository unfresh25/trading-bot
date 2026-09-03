from datetime import UTC, datetime

import httpx
import pandas as pd

from hybrid_trader.market_data import (
    AlpacaCredentials,
    AlpacaMarketDataSource,
    AssetClass,
    CsvMarketDataSource,
    MarketDataQuery,
)


def test_alpaca_stock_source_follows_pagination_and_normalizes_bars():
    requests = []

    def handler(request):
        requests.append(request)
        page = request.url.params.get("page_token")
        bars = [{"t": "2025-01-02T14:30:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 10}]
        if page:
            bars = [{"t": "2025-01-02T14:45:00Z", "o": 101, "h": 103, "l": 100, "c": 102, "v": 11}]
        return httpx.Response(
            200,
            json={"bars": {"SPY": bars}, "next_page_token": None if page else "next"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = AlpacaMarketDataSource(AlpacaCredentials("key", "secret"), client)
    query = MarketDataQuery(
        symbol="SPY",
        asset_class=AssetClass.STOCK,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 2, 1, tzinfo=UTC),
    )

    result = source.fetch(query)

    assert len(result) == 2
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert requests[0].url.path == "/v2/stocks/bars"
    assert requests[1].url.params["page_token"] == "next"
    assert requests[0].headers["APCA-API-KEY-ID"] == "key"


def test_alpaca_crypto_source_uses_crypto_resource():
    def handler(request):
        assert request.url.path == "/v1beta3/crypto/us/bars"
        return httpx.Response(200, json={"bars": {"BTC/USD": []}, "next_page_token": None})

    source = AlpacaMarketDataSource(
        AlpacaCredentials("key", "secret"),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    query = MarketDataQuery(
        symbol="BTC/USD",
        asset_class=AssetClass.CRYPTO,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 2, 1, tzinfo=UTC),
    )

    result = source.fetch(query)

    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)


def test_csv_source_normalizes_timestamp_column(tmp_path, bars):
    path = tmp_path / "bars.csv"
    bars.rename_axis("timestamp").to_csv(path)

    result = CsvMarketDataSource(path).fetch()

    pd.testing.assert_frame_equal(result, bars, check_freq=False)
