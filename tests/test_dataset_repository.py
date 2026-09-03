import json

import pandas as pd

from hybrid_trader.datasets import DatasetIdentity, ParquetDatasetRepository


def test_parquet_dataset_is_versioned_with_auditable_metadata(tmp_path, bars):
    repository = ParquetDatasetRepository(tmp_path)
    identity = DatasetIdentity(symbol="BTC/USD", provider="fixture", timeframe="15Min")

    artifact = repository.save(identity, bars)

    assert artifact.data_path.exists()
    assert artifact.metadata_path.exists()
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["symbol"] == "BTC/USD"
    assert metadata["rows"] == len(bars)
    assert metadata["sha256"] == artifact.sha256
    pd.testing.assert_frame_equal(repository.load(artifact), bars, check_freq=False)

