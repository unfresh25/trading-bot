from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import Direction, Evidence, Horizon, RiskLevel, TechnicalSignal

# Feature columns the agent reads. Kept explicit so a missing input triggers abstention
# rather than a silent wrong verdict.
REQUIRED = ("distance_ema_20", "trend_slope_20", "macd_norm", "rsi_14", "atr_14_norm")


def technical_agent(row: pd.Series, as_of: pd.Timestamp | None = None) -> TechnicalSignal:
    """Deterministic technical read of one bar's indicators into a strict signal.

    Rule-based (no LLM), so the backtest stays fully reproducible. Each rule contributes
    a signed vote and a piece of traceable evidence; the net vote sets the direction and
    confidence. The agent abstains when any required indicator is missing or when the
    bullish and bearish evidence roughly cancel, so a genuinely ambiguous bar adds nothing
    to the decision instead of injecting noise.
    """
    if any(c not in row or not np.isfinite(row.get(c, np.nan)) for c in REQUIRED):
        return _abstain(as_of)

    votes: list[tuple[float, Evidence]] = []
    _trend_votes(row, votes)
    _momentum_votes(row, votes)

    net = float(np.clip(sum(weight for weight, _ in votes), -1.0, 1.0))
    strength = float(np.clip(sum(abs(weight) for weight, _ in votes), 0.0, 1.0))
    evidence = [item for _, item in votes]

    if abs(net) < 0.15:  # bullish and bearish evidence roughly cancel
        return _abstain(as_of, evidence)

    direction = Direction.BULLISH if net > 0 else Direction.BEARISH
    return TechnicalSignal(
        direction=direction,
        confidence=abs(net),
        trend_strength=strength,
        risk=_risk(row),
        horizon=Horizon.SHORT_TERM,
        abstain=False,
        evidence=evidence,
        as_of=as_of.to_pydatetime() if as_of is not None else None,
    )


def technical_agent_scores(features: pd.DataFrame) -> pd.Series:
    """Signed directional score in [-1, 1] from the technical agent, per bar.

    A convenience batch wrapper for the backtest: applies the deterministic agent to each
    row and returns its ``directional_score`` (0 when the agent abstains). Causal — each
    bar reads only its own indicators.
    """
    scores = [technical_agent(row).directional_score for _, row in features.iterrows()]
    return pd.Series(scores, index=features.index, name="agent_score")


def _trend_votes(row: pd.Series, votes: list) -> None:
    ema = float(row["distance_ema_20"])
    slope = float(row["trend_slope_20"])
    macd = float(row["macd_norm"])
    votes.append((
        np.clip(ema * 40, -0.4, 0.4),
        Evidence(indicator="distance_ema_20", value=ema, interpretation="price vs EMA20"),
    ))
    votes.append((
        np.clip(slope * 300, -0.3, 0.3),
        Evidence(indicator="trend_slope_20", value=slope, interpretation="log-price regression slope"),
    ))
    votes.append((
        np.clip(macd * 80, -0.3, 0.3),
        Evidence(indicator="macd_norm", value=macd, interpretation="MACD normalised by price"),
    ))


def _momentum_votes(row: pd.Series, votes: list) -> None:
    rsi = float(row["rsi_14"])  # already scaled to [0, 1] in features
    # Mean-reversion read: oversold (<0.3) leans bullish, overbought (>0.7) bearish.
    weight = np.clip((0.5 - rsi) * 1.2, -0.4, 0.4)
    votes.append((
        weight,
        Evidence(indicator="rsi_14", value=rsi, interpretation="RSI mean-reversion tilt"),
    ))


def _risk(row: pd.Series) -> RiskLevel:
    atr = float(row["atr_14_norm"])
    if atr > 0.02:
        return RiskLevel.HIGH
    if atr > 0.008:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _abstain(as_of: pd.Timestamp | None, evidence: list[Evidence] | None = None) -> TechnicalSignal:
    return TechnicalSignal(
        direction=Direction.NEUTRAL,
        confidence=0.0,
        trend_strength=0.0,
        risk=RiskLevel.MEDIUM,
        horizon=Horizon.SHORT_TERM,
        abstain=True,
        evidence=evidence or [],
        as_of=as_of.to_pydatetime() if as_of is not None else None,
    )
