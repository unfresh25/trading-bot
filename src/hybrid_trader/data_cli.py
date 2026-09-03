from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from .datasets import DatasetIdentity, ParquetDatasetRepository
from .market_data import (
    AlpacaCredentials,
    AlpacaMarketDataSource,
    AssetClass,
    CsvMarketDataSource,
    MarketDataQuery,
)
from .quality import assess_quality


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire and version OHLCV datasets")
    subparsers = parser.add_subparsers(dest="source", required=True)
    alpaca = subparsers.add_parser("alpaca")
    alpaca.add_argument("symbol")
    alpaca.add_argument("asset_class", choices=[item.value for item in AssetClass])
    alpaca.add_argument("--start", required=True)
    alpaca.add_argument("--end", required=True)
    alpaca.add_argument("--output", type=Path, default=Path("data/processed"))
    csv = subparsers.add_parser("csv")
    csv.add_argument("path", type=Path)
    csv.add_argument("symbol")
    csv.add_argument("asset_class", choices=[item.value for item in AssetClass])
    csv.add_argument("--output", type=Path, default=Path("data/processed"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    asset_class = AssetClass(args.asset_class)
    if args.source == "alpaca":
        query = MarketDataQuery(
            symbol=args.symbol,
            asset_class=asset_class,
            start=datetime.fromisoformat(args.start),
            end=datetime.fromisoformat(args.end),
        )
        bars = AlpacaMarketDataSource(AlpacaCredentials.from_environment()).fetch(query)
        provider = "alpaca"
    else:
        bars = CsvMarketDataSource(args.path).fetch()
        provider = "csv"
    artifact = ParquetDatasetRepository(args.output).save(
        DatasetIdentity(args.symbol, provider, "15Min"), bars
    )
    quality = assess_quality(bars, asset_class, expected_interval=timedelta(minutes=15))
    output = {
        "data_path": str(artifact.data_path),
        "metadata_path": str(artifact.metadata_path),
        "sha256": artifact.sha256,
        "quality": asdict(quality),
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
