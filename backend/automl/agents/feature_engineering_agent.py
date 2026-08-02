"""Feature engineering agent: produces interaction / polynomial features
based on the most important numeric columns detected by correlation analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineeringAgent:
    name = "feature_engineering"

    def propose(self, df: pd.DataFrame, target: str | None, top_k: int = 3) -> list[str]:
        proposals: list[str] = []
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return proposals

        corr = numeric.corr().abs()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(mask).stack().sort_values(ascending=False)
        if not pairs.empty:
            top = pairs.head(top_k)
            for (a, b), value in top.items():
                if value >= 0.6:
                    proposals.append(
                        f"Interaction entre {a} et {b} (|corr|={value:.2f}): envisager une feature multiplicative."
                    )

        if numeric.shape[1] <= 5 and numeric.shape[0] >= 200:
            proposals.append("PolynomialFeatures(degree=2, interaction_only=True) recommandé.")
        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            proposals.append("Binning supervisé de la cible en quantiles pour la classification dérivée.")

        if not proposals:
            proposals.append("Pas de feature engineering supplémentaire nécessaire.")
        return proposals
