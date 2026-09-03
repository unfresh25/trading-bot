from __future__ import annotations

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

