from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .application import WalkForwardBacktestRunner
from .reporting import compare_markets, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare walk-forward results by market")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="SYMBOL=PARQUET",
        help="May be supplied multiple times",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-report.json"))
    args = parser.parse_args()
    datasets: dict[str, pd.DataFrame] = {}
    for specification in args.dataset:
        symbol, separator, path = specification.partition("=")
        if not separator:
            parser.error("Each --dataset must use SYMBOL=PARQUET")
        datasets[symbol] = pd.read_parquet(Path(path))
    report = compare_markets(datasets, WalkForwardBacktestRunner())
    destination = write_report(args.output, report)
    print(f"Report written to {destination}")
    for symbol, result in report.items():
        print(f"{symbol}: {result['strategy']}")


if __name__ == "__main__":
    main()
