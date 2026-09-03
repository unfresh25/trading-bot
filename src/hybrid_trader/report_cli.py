from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .application import WalkForwardBacktestRunner
from .reporting import compare_markets


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare walk-forward results by market")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="SYMBOL=PARQUET",
        help="May be supplied multiple times",
    )
    args = parser.parse_args()
    datasets: dict[str, pd.DataFrame] = {}
    for specification in args.dataset:
        symbol, separator, path = specification.partition("=")
        if not separator:
            parser.error("Each --dataset must use SYMBOL=PARQUET")
        datasets[symbol] = pd.read_parquet(Path(path))
    report = compare_markets(datasets, WalkForwardBacktestRunner())
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

