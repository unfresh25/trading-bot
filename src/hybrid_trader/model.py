from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticBaseline:
    def __init__(self) -> None:
        self._model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1_000, class_weight="balanced")),
            ]
        )

    @property
    def version(self) -> str:
        return "logistic-v1"

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self._model.fit(features, labels)

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        probabilities = self._model.predict_proba(features)
        classes = self._model.named_steps["model"].classes_
        return pd.DataFrame(probabilities, index=features.index, columns=classes)


class XGBoostBaseline:
    """Gradient-boosted alternative with the same interface as LogisticBaseline.

    XGBoost multiclass requires contiguous 0-indexed labels, so the (-1, 0, 1) target
    classes are mapped to (0, 1, 2) for training and the returned probability columns
    are mapped back, keeping the rest of the pipeline (calibration, diagnostics)
    identical regardless of which model is injected. Only classes present in the
    training fold appear as columns, mirroring LogisticBaseline behaviour.
    """

    def __init__(self) -> None:
        from xgboost import XGBClassifier

        self._model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._classes: np.ndarray | None = None

    @property
    def version(self) -> str:
        return "xgboost-v1"

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self._classes = np.array(sorted(labels.unique()))
        encoded = labels.map({cls: index for index, cls in enumerate(self._classes)})
        self._model.fit(features, encoded)

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        probabilities = self._model.predict_proba(features)
        return pd.DataFrame(probabilities, index=features.index, columns=self._classes)



class EnsembleModel:
    """Averages the calibrated class probabilities of several ProbabilisticModels.

    Combines diverse learners (e.g. logistic + XGBoost) by fitting each on the same data
    and averaging their per-class probabilities, reindexed to the union of classes so a
    model that never saw a class contributes zero to it. Implements the same interface,
    so it is injected through model_factory without any engine change.
    """

    def __init__(self, factories: tuple[type, ...] = (LogisticBaseline, XGBoostBaseline)) -> None:
        self._models = [factory() for factory in factories]

    @property
    def version(self) -> str:
        return "ensemble-" + "+".join(m.version for m in self._models)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        for model in self._models:
            model.fit(features, labels)

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        frames = [m.predict_proba(features) for m in self._models]
        classes = sorted({c for frame in frames for c in frame.columns})
        aligned = [frame.reindex(columns=classes, fill_value=0.0) for frame in frames]
        averaged = sum(aligned) / len(aligned)
        return averaged.div(averaged.sum(axis=1), axis=0)
