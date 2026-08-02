"""Target Recommendation Agent.

Given a freshly loaded dataset, suggests the most likely target column and the
associated problem type. Returns a ranked list of suggestions with explanations
so the UI can either auto-pick or let the user confirm.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.schema import ProblemType


@dataclass(frozen=True)
class TargetSuggestion:
    column: str
    problem_type: ProblemType
    score: float
    reasons: list[str]


_NAME_HINTS = {
    "target", "label", "y", "class", "output", "response", "outcome",
    "sentiment", "category", "churn", "fraud", "spam", "risk", "score",
}


def recommend_target(df: pd.DataFrame, user_target: str | None = None) -> list[TargetSuggestion]:
    """Return ranked target suggestions for `df`."""
    if user_target and user_target in df.columns:
        problem = _detect_problem_type(df[user_target])
        return [
            TargetSuggestion(
                column=user_target,
                problem_type=problem,
                score=1.0,
                reasons=["Colonne cible sélectionnée par l'utilisateur."],
            )
        ]

    suggestions: list[TargetSuggestion] = []
    n_rows = max(len(df), 1)
    columns = list(df.columns)

    for idx, col in enumerate(columns):
        series = df[col]
        score = 0.0
        reasons: list[str] = []

        name_l = str(col).lower()
        for hint in _NAME_HINTS:
            if hint in name_l:
                score += 0.35
                reasons.append(f"Nom de colonne évocateur (« {hint} »).")
                break

        if idx == len(columns) - 1:
            score += 0.10
            reasons.append("Colonne la plus à droite (souvent la cible).")
        elif idx >= len(columns) - 3:
            score += 0.04

        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n_rows if n_rows else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series)
        n_missing = int(series.isna().sum())
        missing_ratio = n_missing / n_rows if n_rows else 0.0

        if n_unique <= 15 and not is_numeric:
            score += 0.30
            reasons.append(f"Variable catégorielle à {n_unique} modalités — compatible classification.")
        elif n_unique <= 20 and is_numeric:
            score += 0.20
            reasons.append(f"Variable numérique à {n_unique} valeurs distinctes — classification possible.")
        elif is_numeric and unique_ratio > 0.05:
            score += 0.15
            reasons.append("Variable numérique continue — compatible régression.")

        if 1 < n_unique <= 5:
            score += 0.05
            reasons.append("Faible cardinalité — typique d'une cible.")

        if missing_ratio > 0.30:
            score -= 0.20
            reasons.append(f"Taux de valeurs manquantes élevé ({missing_ratio*100:.1f}%) — peu probable comme cible.")
        if missing_ratio > 0.60:
            score = -1.0

        if score > 0:
            suggestions.append(TargetSuggestion(col, _detect_problem_type(series), round(score, 3), reasons))

    suggestions.sort(key=lambda s: s.score, reverse=True)
    return suggestions[:5]


def _detect_problem_type(series: pd.Series) -> ProblemType:
    n_unique = series.nunique(dropna=True)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if is_numeric and (n_unique > 20 or n_unique / max(len(series), 1) > 0.05):
        return ProblemType.REGRESSION
    return ProblemType.CLASSIFICATION
