from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

STATE_NAMES = ("bear", "sideways", "bull")


def hmm_regime_probabilities(
    features: pd.DataFrame,
    columns: tuple[str, ...] = ("return_1", "volatility_20"),
    n_states: int = 3,
    window: int = 750,
    refit_every: int = 50,
    min_train: int = 250,
) -> pd.DataFrame:
    """Causal HMM state probabilities (bear/sideways/bull) with no look-ahead.

    A Gaussian HMM is refit on a trailing ``window`` every ``refit_every`` bars. For each
    refit the probabilities are read only at the **last** observation of the window — the
    filtered state distribution given data up to that bar — and held forward until the
    next refit. Using only the terminal point avoids the look-ahead that the smoothed
    (forward-backward) posterior would introduce for interior points, which see future
    observations within the window. States are relabelled by ascending mean return so the
    ``bear``/``sideways``/``bull`` columns stay stable across refits (a raw HMM assigns
    arbitrary state indices per fit).

    Returns a frame aligned to ``features.index`` with one column per state name, NaN
    before enough history exists.
    """
    from hmmlearn.hmm import GaussianHMM

    data = features.loc[:, list(columns)].to_numpy(dtype=float)
    n = len(data)
    out = np.full((n, n_states), np.nan)
    for end in range(min_train, n, refit_every):
        train = data[max(0, end - window) : end + 1]  # inclusive of bar `end`
        if np.isnan(train).any() or len(train) < min_train:
            continue
        model = _fit(GaussianHMM, train, n_states)
        if model is None:
            continue
        order = np.argsort(model.means_[:, 0])  # ascending mean return -> bear..bull
        terminal = model.predict_proba(train)[-1, order]  # filtered dist at bar `end`
        out[end] = terminal
    series = pd.DataFrame(
        out, index=features.index, columns=[f"hmm_{name}" for name in STATE_NAMES]
    )
    return series.ffill()


def _fit(GaussianHMM, train: np.ndarray, n_states: int):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=50,
                random_state=42,
            )
            model.fit(train)
        return model
    except (ValueError, np.linalg.LinAlgError):
        return None
