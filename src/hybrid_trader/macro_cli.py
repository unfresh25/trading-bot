from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .macro_data import MACRO_SERIES, load_macro_releases


def main() -> None:
    """Download point-in-time (first-release) macro data from ALFRED to a Parquet cache.

    Requires a free FRED API key in FRED_API_KEY. The cache stores one row per (series,
    release) with the publication timestamp, so backtests read it offline and reproducibly.
    """
    parser = argparse.ArgumentParser(description="Cache ALFRED first-release macro data")
    parser.add_argument("--output", type=Path, default=Path("data/macro/releases.parquet"))
    parser.add_argument("--series", nargs="*", default=list(MACRO_SERIES))
    args = parser.parse_args()

    releases = load_macro_releases(tuple(args.series))
    frames = []
    for name, series in releases.items():
        frame = series.frame.copy()
        frame.insert(0, "series", name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"Wrote {len(combined)} releases across {len(releases)} series -> {args.output}")


if __name__ == "__main__":
    main()
