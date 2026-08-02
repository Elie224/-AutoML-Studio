"""Natural-language Q&A agent over the dataset and pipeline outputs.

This is a deterministic, rule-based agent that answers common questions by
inspecting the latest EDAResult / ModelResult. It is designed to be replaced
later by an LLM-backed agent without changing the public API.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


class QAAgent:
    name = "qa"

    def __init__(self, df: pd.DataFrame, target: str | None = None, eda=None, leaderboard=None):
        self.df = df
        self.target = target
        self.eda = eda
        self.leaderboard = leaderboard or []

    def ask(self, question: str) -> dict[str, Any]:
        q = question.lower().strip()
        if not q:
            return {"answer": "Posez une question sur le dataset ou les modèles."}

        if any(k in q for k in ["combien de lignes", "taille du dataset", "nombre d'observations"]):
            return {"answer": f"Le dataset contient {len(self.df)} lignes et {self.df.shape[1]} colonnes."}

        if any(k in q for k in ["combien de colonnes", "variables", "features"]):
            return {"answer": f"{self.df.shape[1]} variables sont présentes, dont {len(self.df.select_dtypes(include='number').columns)} numériques."}

        if "manquante" in q or "nan" in q:
            counts = self.df.isna().sum()
            counts = counts[counts > 0]
            if counts.empty:
                return {"answer": "Aucune valeur manquante détectée."}
            top = counts.sort_values(ascending=False).head(5).to_dict()
            txt = ", ".join(f"{c}={v}" for c, v in top.items())
            return {"answer": f"Valeurs manquantes : {txt}."}

        if "doublon" in q:
            n = int(self.df.duplicated().sum())
            return {"answer": f"{n} doublons exacts détectés."}

        if "corr" in q or "corrél" in q:
            numeric = self.df.select_dtypes(include="number")
            if numeric.shape[1] < 2:
                return {"answer": "Pas assez de colonnes numériques pour calculer des corrélations."}
            corr = numeric.corr().abs()
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            top = corr.where(mask).stack().sort_values(ascending=False).head(3)
            txt = ", ".join(f"{a}-{b} ({v:.2f})" for (a, b), v in top.items())
            return {"answer": f"Corrélations les plus fortes : {txt}."}

        if any(k in q for k in ["meilleur modèle", "best model", "leaderboard", "comparer"]):
            if not self.leaderboard:
                return {"answer": "Aucun modèle entraîné pour le moment."}
            top = self.leaderboard[0]
            return {"answer": f"Modèle leader : {top.name} avec {top.metrics}."}

        if "cible" in q or "target" in q:
            if not self.target or self.target not in self.df.columns:
                return {"answer": "Aucune colonne cible n'a été définie."}
            series = self.df[self.target]
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
                return {"answer": f"Cible '{self.target}' : moyenne={series.mean():.2f}, médiane={series.median():.2f}, std={series.std():.2f}."}
            counts = series.value_counts().head(5)
            txt = ", ".join(f"{k} ({v})" for k, v in counts.items())
            return {"answer": f"Cible '{self.target}' (top 5) : {txt}."}

        if "outlier" in q:
            if self.eda and self.eda.outliers:
                top = list(self.eda.outliers.items())[:5]
                txt = ", ".join(f"{c} ({v['count']})" for c, v in top)
                return {"answer": f"Outliers détectés sur : {txt}."}
            return {"answer": "Pas d'outliers notables identifiés par l'EDA."}

        if "supprimer" in q and "colonne" in q:
            if self.eda and self.eda.missing_values.get("missing_per_column"):
                worst = sorted(self.eda.missing_values["missing_per_column"].items(), key=lambda x: x[1]["count"], reverse=True)
                if worst and worst[0][1]["ratio"] > 0.4:
                    col = worst[0][0]
                    return {"answer": f"Recommandation : supprimer la colonne '{col}' (>40% manquant)."}
            return {"answer": "Aucune colonne n'a un taux de valeurs manquantes alarmant d'après l'EDA."}

        if "influence" in q or "important" in q:
            if self.eda and hasattr(self.eda, "feature_importance"):
                top = list(self.eda.feature_importance.items())[:5]
                txt = ", ".join(f"{k} ({v:.3f})" for k, v in top)
                return {"answer": f"Variables les plus influentes (top 5) : {txt}."}
            return {"answer": "L'importance des variables est disponible après entraînement via SHAP."}

        return {"answer": "Je n'ai pas encore d'interprétation automatique pour cette question. Reformulez ou entraînez d'abord un modèle."}
