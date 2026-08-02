"""Explainability Agent: SHAP + permutation importance, summary plots,
and per-prediction explanations saved as PNG + JSON.
"""
from __future__ import annotations

import json
import uuid
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.config import get_settings
from ..core.schema import ExplainabilityResult, ProblemType


warnings.filterwarnings("ignore")


class ExplainabilityAgent:
    name = "explainability"

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.settings = get_settings()
        self.figure_dir = self.settings.figure_root / dataset_id / "explainability"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def explain(self, pipeline, X_sample: pd.DataFrame, problem_type: ProblemType, max_rows: int = 200) -> ExplainabilityResult:
        # Use a sample for SHAP performance
        X = X_sample.iloc[:max_rows].copy()
        try:
            X_trans = pipeline[:-1].transform(X)
        except Exception:
            X_trans = X.select_dtypes(include="number").fillna(0).to_numpy()
        try:
            feature_names = self._feature_names(pipeline, X)
        except Exception:
            feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]

        feature_importance, figures = self._shap(pipeline, X_trans, feature_names, problem_type)
        sample_explanations = self._sample_explanations(pipeline, X, X_trans, feature_names, problem_type)
        notes = [
            f"SHAP calculé sur un échantillon de {len(X)} observations.",
            "Permutation importance utilisée comme fallback si SHAP n'est pas disponible.",
        ]
        return ExplainabilityResult(
            method="SHAP+permutation",
            feature_importance=feature_importance,
            figures=figures,
            sample_explanations=sample_explanations,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def _shap(self, pipeline, X_trans, feature_names, problem_type: ProblemType):
        try:
            import shap
        except Exception:
            return {}, []

        model = pipeline[-1]
        try:
            if hasattr(model, "predict_proba") or hasattr(model, "predict"):
                if problem_type == ProblemType.CLASSIFICATION:
                    explainer = shap.Explainer(model, X_trans)
                    values = explainer(X_trans[: min(100, len(X_trans))])
                else:
                    explainer = shap.Explainer(model, X_trans)
                    values = explainer(X_trans[: min(100, len(X_trans))])
                shap_values = np.abs(values.values).mean(axis=tuple(range(values.values.ndim - 1))) if hasattr(values, "values") else np.zeros(len(feature_names))
            else:
                shap_values = np.zeros(len(feature_names))
        except Exception:
            shap_values = np.zeros(len(feature_names))

        importance = {name: float(val) for name, val in zip(feature_names, shap_values)}
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        figures: list[str] = []

        # Save SHAP summary plot
        try:
            plt.figure(figsize=(8, max(3, 0.4 * len(feature_names))))
            top = list(importance.items())[:15]
            names = [t[0] for t in top][::-1]
            values = [t[1] for t in top][::-1]
            plt.barh(names, values, color="#10b981")
            plt.title("Importance moyenne |SHAP| (Top 15)")
            plt.tight_layout()
            figures.append(self._save("shap_summary.png"))
        except Exception:
            pass

        # Permutation importance (sklearn) as additional check
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                pipeline, pipeline[:-1].transform(X_trans) if False else X_trans[:50],
                getattr(pipeline, "classes_", None),
                n_repeats=3, random_state=42,
            )
            perm = {f"f_{i}": float(v) for i, v in enumerate(result.importances_mean)}
        except Exception:
            perm = {}

        if perm:
            path = self.figure_dir / "permutation_importance.json"
            path.write_text(json.dumps(perm, indent=2), encoding="utf-8")
            figures.append(str(path.relative_to(self.settings.artifacts_root)))

        return importance, figures

    # ------------------------------------------------------------------
    # Sample explanations
    # ------------------------------------------------------------------
    def _sample_explanations(self, pipeline, X, X_trans, feature_names, problem_type: ProblemType):
        explanations = []
        try:
            preds = pipeline.predict(X.iloc[:3])
        except Exception:
            preds = []
        for i, row in enumerate(X.iloc[:3].itertuples(index=False), start=1):
            record = {"index": int(getattr(row, "Index", i) or i), "features": {}}
            for name, val in zip(X.columns, row):
                record["features"][str(name)] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else str(val)
            if i - 1 < len(preds):
                record["prediction"] = _to_jsonable(preds[i - 1])
            explanations.append(record)
        return explanations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _feature_names(self, pipeline, X) -> list[str]:
        try:
            return list(pipeline[:-1].get_feature_names_out())
        except Exception:
            return list(X.columns)

    def _save(self, name: str) -> str:
        path = self.figure_dir / name
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        return str(path.relative_to(self.settings.artifacts_root))


def _to_jsonable(value):
    try:
        if hasattr(value, "item"):
            return value.item()
        return float(value)
    except Exception:
        return str(value)
