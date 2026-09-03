from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import validate_ohlcv


@dataclass(frozen=True)
class DatasetIdentity:
    symbol: str
    provider: str
    timeframe: str

    @property
    def slug(self) -> str:
        return self.symbol.replace("/", "-").lower()


@dataclass(frozen=True)
class DatasetArtifact:
    data_path: Path
    metadata_path: Path
    sha256: str


class ParquetDatasetRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(self, identity: DatasetIdentity, bars: pd.DataFrame) -> DatasetArtifact:
        validated = validate_ohlcv(bars)
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        directory = self._root / identity.slug / version
        directory.mkdir(parents=True, exist_ok=False)
        data_path = directory / "bars.parquet"
        metadata_path = directory / "metadata.json"
        validated.to_parquet(data_path)
        digest = _sha256(data_path)
        metadata = {
            **asdict(identity),
            "rows": len(validated),
            "start": validated.index.min().isoformat(),
            "end": validated.index.max().isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "sha256": digest,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return DatasetArtifact(data_path, metadata_path, digest)

    def load(self, artifact: DatasetArtifact) -> pd.DataFrame:
        if _sha256(artifact.data_path) != artifact.sha256:
            raise ValueError("Dataset hash does not match its artifact")
        return validate_ohlcv(pd.read_parquet(artifact.data_path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
