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


def select_probability_calibrator(
    probabilities: pd.DataFrame, labels: pd.Series
) -> CalibrationSelection:
    candidates: dict[str, ProbabilityCalibrator] = {"identity": IdentityCalibrator()}
    for method in ("sigmoid", "isotonic"):
        calibrator = OneVsRestCalibrator(method)
        calibrator.fit(probabilities, labels)
        candidates[method] = calibrator
    scores = {
        name: float(log_loss(labels, candidate.transform(probabilities), labels=CLASSES))
        for name, candidate in candidates.items()
    }
    selected = min(scores, key=scores.get)
    return CalibrationSelection(selected, candidates[selected], scores)


def _class_frame(probabilities: pd.DataFrame) -> pd.DataFrame:
    return probabilities.reindex(columns=CLASSES, fill_value=0.0).astype(float)


def _normalized(probabilities: pd.DataFrame) -> pd.DataFrame:
    clipped = _class_frame(probabilities).clip(EPSILON, 1.0)
    return clipped.div(clipped.sum(axis=1), axis=0)


def _logit(probability: pd.Series) -> pd.Series:
    clipped = probability.clip(EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))
