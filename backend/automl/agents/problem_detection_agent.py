"""Problem-type detection agent.

Heuristic decision tree that returns the most likely problem type given the
target column and dataset characteristics.
"""
from __future__ import annotations

import pandas as pd

from ..core.schema import ProblemType


class ProblemDetectionAgent:
    name = "problem_detection"

    def detect(self, df: pd.DataFrame, target: str | None) -> ProblemType:
        if not target or target not in df.columns:
            return self._unsupervised(df)
        return self._supervised(df[target])

    def _supervised(self, series: pd.Series) -> ProblemType:
        n_unique = series.nunique(dropna=True)
        n_rows = max(len(series), 1)
        if pd.api.types.is_numeric_dtype(series) and n_unique > 20 and n_unique / n_rows > 0.05:
            return ProblemType.REGRESSION
        if pd.api.types.is_bool_dtype(series) or n_unique <= 20:
            return ProblemType.CLASSIFICATION
        return ProblemType.REGRESSION

    def _unsupervised(self, df: pd.DataFrame) -> ProblemType:
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if datetime_cols:
            return ProblemType.TIME_SERIES
        if df.shape[0] >= 50:
            return ProblemType.CLUSTERING
        return ProblemType.CLUSTERING
