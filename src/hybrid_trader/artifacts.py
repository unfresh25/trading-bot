from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .engine import BacktestResult
from .serialization import dump_json


@dataclass(frozen=True)
class ArtifactPaths:
    predictions: Path
    decisions: Path
    trades: Path
    manifest: Path


def _symbol_slug(symbol: str) -> str:
    """Filesystem-safe, collision-free slug for a market symbol.

    A readable stem (alphanumerics case-folded, separators to '_') plus a short hash of
    the exact original symbol. The hash guarantees that symbols differing only in their
    separators — 'BTC/USD' vs 'BTC-USD' — never share a directory. The manifest also
    records the untransformed symbol as the source of truth.
    """
    readable = "".join(c if c.isalnum() else "_" for c in symbol.lower())
    suffix = hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{suffix}"


class BacktestArtifactRepository:
    """Filesystem adapter for an immutable, inspectable backtest audit trail.

    Each run is written under an immutable ``<slug>/<content-hash>`` directory so a
    re-run never overwrites a previous audit trail, mirroring the dataset repository.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(
        self,
        symbol: str,
        result: BacktestResult,
        config: AppConfig | None = None,
        input_sha256: str | None = None,
    ) -> ArtifactPaths:
        predictions = result.predictions.rename_axis("timestamp").reset_index()
        decisions = result.decisions.rename_axis("timestamp").reset_index()
        trades = result.trades

        version = _run_version(symbol, result)
        destination = self._root / _symbol_slug(symbol) / version
        destination.mkdir(parents=True, exist_ok=True)
        paths = ArtifactPaths(
            destination / "predictions.parquet",
            destination / "decisions.parquet",
            destination / "trades.parquet",
            destination / "manifest.json",
        )
        predictions.to_parquet(paths.predictions, index=False)
        decisions.to_parquet(paths.decisions, index=False)
        trades.to_parquet(paths.trades, index=False)
        files = {
            "predictions": _file_record(paths.predictions, len(result.predictions)),
            "decisions": _file_record(paths.decisions, len(result.decisions)),
            "trades": _file_record(paths.trades, len(result.trades)),
        }
        dump_json(_manifest(symbol, version, result, files, config, input_sha256), paths.manifest)
        return paths


def _manifest(
    symbol: str,
    version: str,
    result: BacktestResult,
    files: dict[str, Any],
    config: AppConfig | None,
    input_sha256: str | None,
) -> dict[str, Any]:
    predictions = result.predictions
    period = None
    if not predictions.empty:
        period = {
            "start": str(predictions.index.min()),
            "end": str(predictions.index.max()),
        }
    return {
        "symbol": symbol,
        "version": version,
        "git_commit": _git_commit(),
        "bootstrap_seed": 42,
        "period": period,
        "input_sha256": input_sha256,
        "config": _config_snapshot(config),
        "metrics": result.metrics,
        "files": files,
    }


def _config_snapshot(config: AppConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "costs": asdict(config.costs),
        "target": asdict(config.target),
        "risk": asdict(config.risk),
        "strategy": asdict(config.strategy),
    }


def _run_version(symbol: str, result: BacktestResult) -> str:
    """Deterministic content hash of the run, so identical runs land in the same dir."""
    digest = hashlib.sha256()
    digest.update(symbol.encode("utf-8"))
    for frame in (result.predictions, result.trades, result.decisions):
        digest.update(pd_hash(frame))
    return digest.hexdigest()[:16]


def pd_hash(frame: Any) -> bytes:
    from pandas.util import hash_pandas_object

    if frame.empty:
        return b"empty"
    return hash_pandas_object(frame, index=True).values.tobytes()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _file_record(path: Path, rows: int) -> dict[str, str | int]:
    return {"path": str(path), "rows": rows, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
