from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .engine import run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research-only V0 backtest")
    parser.add_argument("csv", type=Path, help="CSV with timestamp, open, high, low, close, volume")
    args = parser.parse_args()
    bars = pd.read_csv(args.csv, parse_dates=["timestamp"]).set_index("timestamp")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    result = run_walk_forward(bars)
    print(pd.Series(result.metrics).to_string())


if __name__ == "__main__":
    main()

