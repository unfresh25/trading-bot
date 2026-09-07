from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import Direction, Evidence, Horizon, RiskLevel, TechnicalSignal

# Sign of each surprise's effect on risk assets. A rising-rates or rising-inflation
# surprise is a headwind (risk-off, bearish); a falling one is a tailwind. The market-
# surprise inputs (rate_surprise, infl_surprise) are the daily standardised move in the
# market's own expectation — a genuine surprise — and dominate. The release-based columns
# remain for backward compatibility with the ALFRED-only path.
RISK_SIGN = {
    "rate_surprise": -1.0,  # 2y yield jumps -> hawkish surprise -> risk-off
    "infl_surprise": -1.0,  # breakeven jumps -> inflation surprise -> risk-off
    "surprise_CPIAUCSL": -1.0,
    "surprise_CPILFESL": -1.0,
    "surprise_PAYEMS": -1.0,
    "surprise_UNRATE": +1.0,
    "surprise_FEDFUNDS": -1.0,
}


def macro_agent(row: pd.Series, as_of: pd.Timestamp | None = None) -> TechnicalSignal:
    """Deterministic macro read of the current point-in-time surprise features.

    Rule-based (no LLM), reproducible. Each recent macro surprise contributes a signed,
    magnitude-capped vote toward risk-on (bullish) or risk-off (bearish); the net vote
    sets direction and confidence. Abstains when no surprise is available (warm-up) or
    when hot and soft prints cancel. Reuses the strict TechnicalSignal schema so the macro
    agent slots into the same decision path as the technical agent.
    """
    available = [(c, row.get(c)) for c in RISK_SIGN if c in row]
    finite = [(c, v) for c, v in available if v is not None and np.isfinite(v)]
    if not finite:
        return _abstain(as_of)

    votes: list[tuple[float, Evidence]] = []
    for column, value in finite:
        weight = float(np.clip(RISK_SIGN[column] * value * 0.25, -0.35, 0.35))
        votes.append((
            weight,
            Evidence(indicator=column, value=float(value), interpretation="point-in-time macro surprise z"),
        ))

    net = float(np.clip(sum(w for w, _ in votes), -1.0, 1.0))
    strength = float(np.clip(sum(abs(w) for w, _ in votes), 0.0, 1.0))
    evidence = [item for _, item in votes]

    if abs(net) < 0.15:
        return _abstain(as_of, evidence)

    direction = Direction.BULLISH if net > 0 else Direction.BEARISH
    return TechnicalSignal(
        direction=direction,
        confidence=abs(net),
        trend_strength=strength,
        risk=RiskLevel.MEDIUM,
        horizon=Horizon.MEDIUM_TERM,
        abstain=False,
        evidence=evidence,
        as_of=as_of.to_pydatetime() if as_of is not None else None,
    )


def macro_agent_scores(features: pd.DataFrame) -> pd.Series:
    """Per-bar signed macro directional score in [-1, 1] (0 when abstaining)."""
    scores = [macro_agent(row).directional_score for _, row in features.iterrows()]
    return pd.Series(scores, index=features.index, name="macro_score")


def _abstain(as_of: pd.Timestamp | None, evidence: list[Evidence] | None = None) -> TechnicalSignal:
    return TechnicalSignal(
        direction=Direction.NEUTRAL,
        confidence=0.0,
        trend_strength=0.0,
        risk=RiskLevel.MEDIUM,
        horizon=Horizon.MEDIUM_TERM,
        abstain=True,
        evidence=evidence or [],
        as_of=as_of.to_pydatetime() if as_of is not None else None,
    )
