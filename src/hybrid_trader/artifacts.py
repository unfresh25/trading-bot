from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .engine import BacktestResult
from .serialization import dump_json


@dataclass(frozen=True)
class ArtifactPaths:
    predictions: Path
    decisions: Path
    trades: Path
    manifest: Path


class BacktestArtifactRepository:
    """Filesystem adapter for an immutable, inspectable backtest audit trail."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(self, symbol: str, result: BacktestResult) -> ArtifactPaths:
        destination = self._root / re.sub(r"[^a-z0-9]+", "-", symbol.lower()).strip("-")
        destination.mkdir(parents=True, exist_ok=True)
        paths = ArtifactPaths(
            destination / "predictions.parquet",
            destination / "decisions.parquet",
            destination / "trades.parquet",
            destination / "manifest.json",
        )
        result.predictions.rename_axis("timestamp").reset_index().to_parquet(
            paths.predictions, index=False
        )
        result.decisions.rename_axis("timestamp").reset_index().to_parquet(
            paths.decisions, index=False
        )
        result.trades.to_parquet(paths.trades, index=False)
        files = {
            "predictions": _file_record(paths.predictions, len(result.predictions)),
            "decisions": _file_record(paths.decisions, len(result.decisions)),
            "trades": _file_record(paths.trades, len(result.trades)),
        }
        dump_json(
            {"symbol": symbol, "metrics": result.metrics, "files": files},
            paths.manifest,
        )
        return paths


def _file_record(path: Path, rows: int) -> dict[str, str | int]:
    return {"path": str(path), "rows": rows, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
