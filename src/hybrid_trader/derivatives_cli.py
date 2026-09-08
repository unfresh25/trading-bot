from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .derivatives_data import load_funding_rate, load_metrics


def _month_range(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m") for d in pd.date_range(start, end, freq="MS")]


def _day_range(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def main() -> None:
    """Cache Binance BTC derivatives (funding + metrics) to Parquet for offline backtests.

    These are recorded market facts (immutable by nature), so no vintage handling is needed.
    """
    parser = argparse.ArgumentParser(description="Cache Binance derivatives data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=Path("data/derivatives"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    funding = load_funding_rate(args.symbol, _month_range(args.start, args.end))
    funding.to_parquet(args.output / f"{args.symbol}-funding.parquet")
    metrics = load_metrics(args.symbol, _day_range(args.start, args.end))
    metrics.to_parquet(args.output / f"{args.symbol}-metrics.parquet")
    print(f"funding: {len(funding)} rows, metrics: {len(metrics)} rows -> {args.output}")


if __name__ == "__main__":
    main()
