from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_net_return(
    returns: pd.Series,
    samples: int = 1_000,
    block_size: int = 5,
    seed: int = 42,
) -> dict[str, float | int]:
    values = returns.dropna().to_numpy(dtype=float)
    if not len(values):
        return {"lower": 0.0, "median": 0.0, "upper": 0.0, "samples": samples}
    if samples <= 0 or block_size <= 0:
        raise ValueError("samples and block_size must be positive")
    rng = np.random.default_rng(seed)
    starts = np.arange(len(values))
    estimates = np.empty(samples)
    blocks_needed = int(np.ceil(len(values) / block_size))
    for sample in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        draw = np.concatenate(
            [values[(start + np.arange(block_size)) % len(values)] for start in chosen]
        )[: len(values)]
        estimates[sample] = np.prod(1 + draw) - 1
    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
        "samples": samples,
    }
