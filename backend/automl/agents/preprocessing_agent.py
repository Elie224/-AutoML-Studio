"""Cleaning + Preprocessing Agent.

Builds a scikit-learn `ColumnTransformer` based on the column dtypes, suggests
imputation, encoding, scaling, feature engineering and dimensionality reduction.
The same plan can be re-used by the training agent so the prediction pipeline
matches what was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from ..core.schema import PreprocessingPlan


@dataclass
class PreparedDataset:
    X: pd.DataFrame
    y: pd.Series | None
    feature_names: list[str]
    preprocessor: ColumnTransformer | None
    target_encoder: LabelEncoder | None = None
    target_name: str | None = None


def _safe_target_encode(y: pd.Series) -> tuple[LabelEncoder, np.ndarray]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str).fillna("__missing__"))
    return encoder, encoded


class PreprocessingAgent:
    name = "cleaning"

    def __init__(self, plan: PreprocessingPlan | None = None):
        self.plan = plan or PreprocessingPlan()

    # ------------------------------------------------------------------
    # Plan computation
    # ------------------------------------------------------------------
    def suggest_plan(self, df: pd.DataFrame, target: str | None) -> PreprocessingPlan:
        plan = PreprocessingPlan()
        n_rows = max(len(df), 1)
        for col in df.columns:
            if col == target:
                continue
            series = df[col]
            missing_ratio = series.isna().sum() / n_rows
            n_unique = series.nunique(dropna=True)

            # Drop columns with no signal
            if n_unique <= 1:
                plan.drop_columns.append(col)
                plan.notes.append(f"Colonne '{col}' supprimée (constante).")
                continue
            if n_unique >= n_rows and pd.api.types.is_numeric_dtype(series):
                plan.drop_columns.append(col)
                plan.notes.append(f"Colonne '{col}' traitée comme identifiant (unique par ligne).")
                continue
            if missing_ratio > 0.95:
                plan.drop_columns.append(col)
                plan.notes.append(f"Colonne '{col}' supprimée (>95% manquant).")
                continue

            if pd.api.types.is_numeric_dtype(series):
                plan.imputation[col] = "median" if missing_ratio > 0 else "none"
                if n_unique <= 10 and n_unique > 2:
                    plan.encoders[col] = "ordinal"
                else:
                    plan.encoders[col] = "passthrough"
            else:
                plan.imputation[col] = "most_frequent"
                if n_unique <= 10:
                    plan.encoders[col] = "onehot"
                elif n_unique <= 50:
                    plan.encoders[col] = "target" if target else "ordinal"
                else:
                    plan.encoders[col] = "target" if target else "ordinal"
                    plan.notes.append(f"'{col}' a une cardinalité élevée ({n_unique}); target encoding recommandé.")

        # Scaling default
        numeric_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in plan.drop_columns and c != target
        ]
        if numeric_cols:
            plan.scaling = "standard"
            plan.rationale.append("Standardisation (z-score) pour les variables numériques.")
        else:
            plan.scaling = "none"

        if len(df.columns) - len(plan.drop_columns) > 50 and len(df) > 500:
            plan.dimensionality_reduction = "variance"
            plan.rationale.append("Sélection de variables par seuil de variance (datasets larges).")

        # Feature engineering hints
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        for col in datetime_cols:
            plan.feature_engineering.append(f"date_features:{col}")
            plan.rationale.append(f"Extraction year/month/day depuis '{col}'.")

        if len(numeric_cols) >= 3:
            plan.feature_engineering.append("ratios")
            plan.rationale.append("Création automatique de ratios entre variables numériques corrélées.")

        if not plan.rationale:
            plan.rationale.append("Aucun prétraitement lourd requis : dataset déjà propre.")
        self.plan = plan
        return plan

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        transformers = []
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for col in df.columns:
            if col in self.plan.drop_columns:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        if numeric_cols:
            steps = []
            if any(self.plan.imputation.get(c, "none") == "median" for c in numeric_cols):
                steps.append(("imputer", SimpleImputer(strategy="median")))
            if self.plan.scaling == "standard":
                steps.append(("scaler", StandardScaler()))
            elif self.plan.scaling == "minmax":
                steps.append(("scaler", MinMaxScaler()))
            elif self.plan.scaling == "robust":
                steps.append(("scaler", RobustScaler()))
            if steps:
                transformers.append(("num", Pipeline(steps), numeric_cols))

        for col in categorical_cols:
            encoder = self.plan.encoders.get(col, "onehot")
            imputer = SimpleImputer(strategy="most_frequent")
            if encoder == "onehot":
                pipe = Pipeline([
                    ("imputer", imputer),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
                ])
            elif encoder == "ordinal":
                pipe = Pipeline([
                    ("imputer", imputer),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ])
            elif encoder == "target":
                pipe = Pipeline([
                    ("imputer", imputer),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ])
            else:
                continue
            transformers.append((f"cat_{col}", pipe, [col]))

        if not transformers:
            # Fallback: passthrough
            return ColumnTransformer([("passthrough", "passthrough", list(df.columns))], remainder="drop")

        preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
        if self.plan.dimensionality_reduction == "variance":
            preprocessor = Pipeline([("pre", preprocessor), ("var", VarianceThreshold(threshold=0.0))])
        return preprocessor

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame, target: str | None) -> PreparedDataset:
        df = df.copy()
        df = df.drop(columns=[c for c in self.plan.drop_columns if c in df.columns], errors="ignore")

        # Extract target BEFORE feature engineering so it never leaks into derived features.
        y: pd.Series | None = None
        target_encoder: LabelEncoder | None = None
        if target and target in df.columns:
            y_raw = df[target]
            if y_raw.dtype == "object" or y_raw.nunique(dropna=True) <= 20:
                target_encoder, y_arr = _safe_target_encode(y_raw)
                y = pd.Series(y_arr, index=df.index, name=target)
            else:
                y = y_raw.astype(float)
            df = df.drop(columns=[target])

        # Feature engineering
        for rule in self.plan.feature_engineering:
            if rule.startswith("date_features:"):
                col = rule.split(":", 1)[1]
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dow"] = df[col].dt.dayofweek
                    df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            elif rule == "ratios":
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                for i, a in enumerate(numeric_cols):
                    for b in numeric_cols[i + 1 :]:
                        if abs(df[a].corr(df[b])) > 0.6:
                            ratio_name = f"{a}_div_{b}"
                            if ratio_name not in df.columns:
                                df[ratio_name] = df[a] / (df[b].replace(0, np.nan))

        preprocessor = self.build_preprocessor(df)
        feature_columns = [c for c in df.columns if c not in (target or "",)]

        return PreparedDataset(
            X=df[feature_columns],
            y=y,
            feature_names=feature_columns,
            preprocessor=preprocessor,
            target_encoder=target_encoder,
            target_name=target,
        )





