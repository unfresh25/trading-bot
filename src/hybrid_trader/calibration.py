from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

CLASSES = (-1, 0, 1)
EPSILON = 1e-9


class ProbabilityCalibrator(Protocol):
    def transform(self, probabilities: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class CalibrationSelection:
    method: str
    calibrator: ProbabilityCalibrator
    validation_log_loss: dict[str, float]


class IdentityCalibrator:
    def transform(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        return _normalized(probabilities)


class OneVsRestCalibrator:
    def __init__(self, method: str) -> None:
        self.method = method
        self._models: dict[int, object | None] = {}

    def fit(self, probabilities: pd.DataFrame, labels: pd.Series) -> None:
        aligned = _class_frame(probabilities)
        for class_id in CLASSES:
            target = (labels == class_id).astype(int)
            if target.nunique() < 2:
                self._models[class_id] = None
            elif self.method == "sigmoid":
                predictor = _logit(aligned[class_id]).to_numpy().reshape(-1, 1)
                self._models[class_id] = LogisticRegression().fit(predictor, target)
            else:
                self._models[class_id] = IsotonicRegression(out_of_bounds="clip").fit(
                    aligned[class_id], target
                )

    def transform(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        aligned = _class_frame(probabilities)
        result = pd.DataFrame(index=aligned.index)
        for class_id in CLASSES:
            model = self._models[class_id]
            if model is None:
                result[class_id] = aligned[class_id]
            elif self.method == "sigmoid":
                predictor = _logit(aligned[class_id]).to_numpy().reshape(-1, 1)
                result[class_id] = model.predict_proba(predictor)[:, 1]
            else:
                result[class_id] = model.predict(aligned[class_id])
        return _normalized(result)


METHODS = ("identity", "sigmoid", "isotonic")


def _build_calibrator(method: str) -> ProbabilityCalibrator:
    return IdentityCalibrator() if method == "identity" else OneVsRestCalibrator(method)


def _fitted_calibrator(
    method: str, probabilities: pd.DataFrame, labels: pd.Series
) -> ProbabilityCalibrator:
    calibrator = _build_calibrator(method)
    if isinstance(calibrator, OneVsRestCalibrator):
        calibrator.fit(probabilities, labels)
    return calibrator


def select_probability_calibrator(
    probabilities: pd.DataFrame, labels: pd.Series, folds: int = 3
) -> CalibrationSelection:
    """Choose the calibration method by out-of-fold log loss, then refit on all data.

    Scoring each method on the same set used to fit it favours the most flexible one
    (isotonic overfits its own selection set). Instead the method is chosen by K-fold
    cross-validation *inside* the calibration window, so ``validation_log_loss`` is an
    honest out-of-sample estimate. The winner is then refit on the full window to avoid
    wasting data. Falls back to in-sample scoring when the window is too small or too
    class-imbalanced to stratify.
    """
    scores = _cross_validated_log_loss(probabilities, labels, folds)
    if scores is None:
        scores = {
            method: float(
                log_loss(
                    labels,
                    _fitted_calibrator(method, probabilities, labels).transform(probabilities),
                    labels=CLASSES,
                )
            )
            for method in METHODS
        }
    selected = min(scores, key=scores.get)
    return CalibrationSelection(
        selected, _fitted_calibrator(selected, probabilities, labels), scores
    )


def _cross_validated_log_loss(
    probabilities: pd.DataFrame, labels: pd.Series, folds: int
) -> dict[str, float] | None:
    from sklearn.model_selection import StratifiedKFold

    counts = labels.value_counts()
    if len(labels) < folds * len(CLASSES) or counts.min() < folds:
        return None
    splitter = StratifiedKFold(n_splits=folds, shuffle=False)
    reset_probabilities = probabilities.reset_index(drop=True)
    reset_labels = labels.reset_index(drop=True)
    totals = {method: 0.0 for method in METHODS}
    for train_idx, test_idx in splitter.split(reset_probabilities, reset_labels):
        fit_prob, fit_labels = reset_probabilities.iloc[train_idx], reset_labels.iloc[train_idx]
        eval_prob, eval_labels = reset_probabilities.iloc[test_idx], reset_labels.iloc[test_idx]
        for method in METHODS:
            calibrated = _fitted_calibrator(method, fit_prob, fit_labels).transform(eval_prob)
            totals[method] += float(log_loss(eval_labels, calibrated, labels=CLASSES))
    return {method: totals[method] / folds for method in METHODS}


def _class_frame(probabilities: pd.DataFrame) -> pd.DataFrame:
    return probabilities.reindex(columns=CLASSES, fill_value=0.0).astype(float)


def _normalized(probabilities: pd.DataFrame) -> pd.DataFrame:
    clipped = _class_frame(probabilities).clip(EPSILON, 1.0)
    return clipped.div(clipped.sum(axis=1), axis=0)


def _logit(probability: pd.Series) -> pd.Series:
    clipped = probability.clip(EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))
