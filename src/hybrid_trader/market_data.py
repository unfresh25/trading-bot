from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import httpx
import pandas as pd

from .data import REQUIRED_COLUMNS, validate_ohlcv


class AssetClass(StrEnum):
    STOCK = "stock"
    CRYPTO = "crypto"


@dataclass(frozen=True)
class AlpacaCredentials:
    key_id: str
    secret_key: str

    @classmethod
    def from_environment(cls) -> AlpacaCredentials:
        key_id = os.environ.get("APCA_API_KEY_ID", "")
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
        if not key_id or not secret_key:
            raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY are required")
        return cls(key_id, secret_key)


@dataclass(frozen=True)
class MarketDataQuery:
    symbol: str
    asset_class: AssetClass
    start: datetime
    end: datetime
    timeframe: str = "15Min"

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Query timestamps must include a timezone")
        if self.start >= self.end:
            raise ValueError("Query start must be before end")


class AlpacaMarketDataSource:
    _BASE_URL = "https://data.alpaca.markets"

    def __init__(self, credentials: AlpacaCredentials, client: httpx.Client | None = None) -> None:
        self._credentials = credentials
        self._client = client or httpx.Client(base_url=self._BASE_URL, timeout=30)

    def fetch(self, query: MarketDataQuery) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            payload = self._request_page(query, page_token)
            records.extend(payload.get("bars", {}).get(query.symbol, []))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
            if page_token in seen_tokens:
                raise RuntimeError("Alpaca returned a repeated pagination token")
            seen_tokens.add(page_token)
        return _normalize_alpaca_bars(records)

    def _request_page(self, query: MarketDataQuery, page_token: str | None) -> dict:
        path = "/v2/stocks/bars"
        params = _query_parameters(query, page_token)
        if query.asset_class is AssetClass.CRYPTO:
            path = "/v1beta3/crypto/us/bars"
            params.pop("feed")
        response = self._client.get(self._BASE_URL + path, params=params, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._credentials.key_id,
            "APCA-API-SECRET-KEY": self._credentials.secret_key,
        }


class CsvMarketDataSource:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def fetch(self) -> pd.DataFrame:
        frame = pd.read_csv(self._path, parse_dates=["timestamp"]).set_index("timestamp")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame.index.name = None
        return validate_ohlcv(frame)


def _query_parameters(query: MarketDataQuery, page_token: str | None) -> dict[str, object]:
    params: dict[str, object] = {
        "symbols": query.symbol,
        "timeframe": query.timeframe,
        "start": query.start.isoformat(),
        "end": query.end.isoformat(),
        "limit": 10_000,
        "sort": "asc",
        "feed": "iex",
    }
    if page_token:
        params["page_token"] = page_token
    return params


def _normalize_alpaca_bars(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=REQUIRED_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame.from_records(records).rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame.index.name = None
    return validate_ohlcv(frame)
