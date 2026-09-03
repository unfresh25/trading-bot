from hybrid_trader.market_data import AssetClass
from hybrid_trader.market_profiles import profile_for


def test_spy_and_btc_use_independent_market_assumptions():
    spy = profile_for("SPY")
    bitcoin = profile_for("BTC/USD")

    assert spy.asset_class is AssetClass.STOCK
    assert spy.annualization_days == 252
    assert bitcoin.asset_class is AssetClass.CRYPTO
    assert bitcoin.annualization_days == 365
    assert bitcoin.costs.round_trip_rate > spy.costs.round_trip_rate
    assert not spy.reject_zero_volume
    assert bitcoin.reject_zero_volume

