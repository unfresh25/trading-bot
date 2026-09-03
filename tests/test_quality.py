from datetime import timedelta

from hybrid_trader.market_data import AssetClass
from hybrid_trader.quality import assess_quality


def test_crypto_quality_counts_missing_continuous_bars(bars):
    with_gap = bars.drop(bars.index[100:102])

    report = assess_quality(with_gap, AssetClass.CRYPTO, timedelta(minutes=15))

    assert report.rows == len(with_gap)
    assert report.missing_bars == 2
    assert report.gap_count == 1
    assert report.zero_volume_rate == 0


def test_stock_quality_does_not_call_market_closures_missing_bars(bars):
    report = assess_quality(bars.iloc[[0, 1, -1]], AssetClass.STOCK, timedelta(minutes=15))

    assert report.missing_bars is None
    assert report.gap_count == 1
