import pandas as pd

from hybrid_trader.uncertainty import block_bootstrap_net_return


def test_block_bootstrap_is_deterministic_and_returns_ordered_interval():
    returns = pd.Series([0.01, -0.005, 0.002, 0.0, 0.004] * 10)

    first = block_bootstrap_net_return(returns, samples=200, block_size=5, seed=7)
    second = block_bootstrap_net_return(returns, samples=200, block_size=5, seed=7)

    assert first == second
    assert first["lower"] <= first["median"] <= first["upper"]
    assert first["samples"] == 200
