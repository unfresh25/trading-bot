from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range("2025-01-01", periods=500, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.001, 0.004, len(index))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": rng.lognormal(8, 0.4, len(index)),
        },
        index=index,
    )

