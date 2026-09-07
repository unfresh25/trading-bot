import numpy as np
import pandas as pd

from hybrid_trader.agents.macro_agent import macro_agent, macro_agent_scores
from hybrid_trader.agents.schemas import Direction
from hybrid_trader.config import CostConfig, StrategyConfig
from hybrid_trader.decision_policy import SignalContext, evaluate_signal
from hybrid_trader.macro_data import MacroReleaseSeries, macro_surprise_features


def _cpi_releases():
    dates = pd.date_range("2021-01-01", "2024-01-01", freq="ME")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(dates, utc=True) + pd.Timedelta(days=10),
            "reference_date": dates,
            "value": 250 + np.cumsum(rng.normal(0.5, 0.3, len(dates))),
        }
    )
    return {"CPIAUCSL": MacroReleaseSeries("CPIAUCSL", frame)}


def test_surprise_features_are_point_in_time():
    releases = _cpi_releases()
    index = pd.date_range("2021-06-01", "2023-12-01", freq="15min", tz="UTC")
    feats = macro_surprise_features(releases, index)
    first_publication = releases["CPIAUCSL"].frame["as_of"].min()
    # No surprise value exists on bars before the first release was public.
    before = feats.loc[feats.index < first_publication, "surprise_CPIAUCSL"]
    assert before.isna().all()
    assert feats.loc[feats.index >= first_publication, "surprise_CPIAUCSL"].notna().any()


def test_surprise_features_do_not_leak_the_future():
    releases = _cpi_releases()
    index = pd.date_range("2021-06-01", "2022-06-01", freq="15min", tz="UTC")
    full = macro_surprise_features(releases, index)
    # Truncating future releases must not change any already-computed past value.
    cutoff = pd.Timestamp("2022-01-01", tz="UTC")
    truncated_releases = {
        k: MacroReleaseSeries(k, v.frame[v.frame["as_of"] <= cutoff])
        for k, v in releases.items()
    }
    truncated = macro_surprise_features(truncated_releases, index)
    common = full.index[full.index <= cutoff]
    pd.testing.assert_series_equal(
        full.loc[common, "surprise_CPIAUCSL"].dropna(),
        truncated.loc[common, "surprise_CPIAUCSL"].dropna(),
    )


def test_macro_agent_reads_hot_inflation_as_bearish():
    # A large positive CPI surprise (hot inflation) is risk-off -> bearish.
    row = pd.Series({"surprise_CPIAUCSL": 2.5})
    signal = macro_agent(row)
    assert signal.direction is Direction.BEARISH
    assert signal.directional_score < 0
    assert signal.evidence


def test_macro_agent_abstains_without_data():
    assert macro_agent(pd.Series({"surprise_CPIAUCSL": np.nan})).abstain is True
    assert macro_agent(pd.Series({})).abstain is True


def test_macro_scores_align_and_stay_in_range():
    feats = pd.DataFrame({"surprise_CPIAUCSL": np.linspace(-3, 3, 20)})
    scores = macro_agent_scores(feats)
    assert len(scores) == 20
    assert (scores.abs() <= 1.0).all()


def test_macro_disagreement_vetoes_a_long():
    strategy = StrategyConfig(minimum_up_probability=0.5)
    costs = CostConfig()
    ok = SignalContext(0.7, 0.1, 0.01, 1000.0, False, macro_score=0.2, macro_veto_below=-0.3)
    veto = SignalContext(0.7, 0.1, 0.01, 1000.0, False, macro_score=-0.8, macro_veto_below=-0.3)
    assert evaluate_signal(ok, strategy, costs).reason == "approved"
    assert evaluate_signal(veto, strategy, costs).reason == "macro_disagreement"
