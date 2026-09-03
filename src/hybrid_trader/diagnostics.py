from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASS_NAMES = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
PROBABILITY_COLUMNS = ["probability_down", "probability_neutral", "probability_up"]


def prediction_diagnostics(predictions: pd.DataFrame) -> dict:
    return {
        "class_distribution": _class_distribution(predictions.label),
        "probability": _probability_scores(predictions),
        "calibration_up": _calibration_curve(predictions),
        "by_fold": _grouped_scores(predictions, predictions.fold),
        "by_year": _grouped_scores(predictions, predictions.index.year),
        "by_regime": _grouped_scores(predictions, predictions.regime),
    }


def _class_distribution(labels: pd.Series) -> dict[str, float]:
    proportions = labels.value_counts(normalize=True)
    return {name: float(proportions.get(label, 0)) for label, name in CLASS_NAMES.items()}


def _probability_scores(predictions: pd.DataFrame) -> dict[str, float]:
    probability = _normalized_probabilities(predictions)
    actual_up = (predictions.label == 1).astype(float)
    class_rates = np.array([(predictions.label == label).mean() for label in (-1, 0, 1)])
    baseline = np.tile(class_rates, (len(predictions), 1))
    brier = float(np.mean((probability[:, 2] - actual_up) ** 2))
    baseline_brier = float(np.mean((baseline[:, 2] - actual_up) ** 2))
    return {
        "brier_up": brier,
        "log_loss": float(log_loss(predictions.label, probability, labels=[-1, 0, 1])),
        "baseline_brier_up": baseline_brier,
        "baseline_log_loss": float(
            log_loss(predictions.label, baseline, labels=[-1, 0, 1])
        ),
        "brier_skill_score": float(1 - brier / baseline_brier) if baseline_brier else 0.0,
    }


def _normalized_probabilities(predictions: pd.DataFrame) -> np.ndarray:
    probability = predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    probability = np.clip(probability, 1e-15, 1 - 1e-15)
    return probability / probability.sum(axis=1, keepdims=True)


def _calibration_curve(predictions: pd.DataFrame, bins: int = 10) -> list[dict[str, float | int]]:
    predicted = predictions.probability_up.to_numpy(dtype=float)
    actual = (predictions.label == 1).to_numpy(dtype=float)
    assignments = np.minimum((predicted * bins).astype(int), bins - 1)
    result = []
    for bin_number in range(bins):
        selected = assignments == bin_number
        if not selected.any():
            continue
        result.append(
            {
                "mean_probability": float(predicted[selected].mean()),
                "observed_frequency": float(actual[selected].mean()),
                "count": int(selected.sum()),
            }
        )
    return result


def _grouped_scores(predictions: pd.DataFrame, groups) -> dict[str, dict[str, float]]:
    output = {}
    for name, positions in predictions.groupby(groups).groups.items():
        subset = predictions.loc[positions]
        output[str(name)] = {"count": float(len(subset)), **_probability_scores(subset)}
    return output
