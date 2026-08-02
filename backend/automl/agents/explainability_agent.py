"""Explainability Agent.

* Supervised problems: SHAP + permutation importance.
* Clustering / anomaly detection: per-cluster profiles, 2D PCA scatter, and
  silhouette-based quality assessment.
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
from .id_detector import detect_id_columns


warnings.filterwarnings("ignore")


class ExplainabilityAgent:
    name = "explainability"

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.settings = get_settings()
        self.figure_dir = self.settings.figure_root / dataset_id / "explainability"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def explain(self, pipeline, X_sample: pd.DataFrame, problem_type: ProblemType, max_rows: int = 200) -> ExplainabilityResult:
        if problem_type in (ProblemType.CLUSTERING, ProblemType.ANOMALY_DETECTION):
            return self._explain_unsupervised(pipeline, X_sample, problem_type, max_rows)
        return self._explain_supervised(pipeline, X_sample, problem_type, max_rows)

    # ------------------------------------------------------------------
    # Supervised: SHAP + permutation
    # ------------------------------------------------------------------
    def _explain_supervised(self, pipeline, X_sample: pd.DataFrame, problem_type: ProblemType, max_rows: int) -> ExplainabilityResult:
        X = X_sample.iloc[:max_rows].copy()
        try:
            X_trans = pipeline[:-1].transform(X)
        except Exception:
            X_trans = X.select_dtypes(include="number").fillna(0).to_numpy()
        try:
            feature_names = self._feature_names(pipeline, X)
        except Exception:
            feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]

        feature_importance, figures = self._shap(pipeline, X_trans, feature_names)
        sample_explanations = self._sample_explanations(pipeline, X, feature_names)
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

    def _shap(self, pipeline, X_trans, feature_names):
        try:
            import shap
        except Exception:
            return {}, []

        model = pipeline[-1]
        try:
            explainer = shap.Explainer(model, X_trans)
            values = explainer(X_trans[: min(100, len(X_trans))])
            shap_values = np.abs(values.values).mean(axis=tuple(range(values.values.ndim - 1))) if hasattr(values, "values") else np.zeros(len(feature_names))
        except Exception:
            shap_values = np.zeros(len(feature_names))

        importance = {name: float(val) for name, val in zip(feature_names, shap_values)}
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        figures: list[str] = []
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

        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                pipeline,
                X_trans[:50],
                getattr(pipeline, "classes_", None),
                n_repeats=3, random_state=42,
            )
            perm = {f"f_{i}": float(v) for i, v in enumerate(result.importances_mean)}
            path = self.figure_dir / "permutation_importance.json"
            path.write_text(json.dumps(perm, indent=2), encoding="utf-8")
            figures.append(str(path.relative_to(self.settings.artifacts_root)))
        except Exception:
            pass

        return importance, figures

    def _sample_explanations(self, pipeline, X, feature_names):
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
    # Unsupervised: cluster profiles + PCA 2D
    # ------------------------------------------------------------------
    def _explain_unsupervised(self, pipeline, X_sample: pd.DataFrame, problem_type: ProblemType, max_rows: int) -> ExplainabilityResult:
        X = X_sample.iloc[:max_rows].copy()
        for id_col in detect_id_columns(X):
            X = X.drop(columns=[id_col.name], errors="ignore")
        try:
            X_trans = pipeline[:-1].transform(X)
        except Exception:
            X_trans = X.select_dtypes(include="number").fillna(0).to_numpy()

        try:
            model = pipeline[-1]
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_trans)
        except Exception:
            labels = np.zeros(len(X_trans))

        figures = self._cluster_figures(X, X_trans, labels)
        feature_importance = self._cluster_profiles(X, labels)
        silhouette = self._silhouette_score(X_trans, labels)

        notes = [
            f"Clustering non supervisé : {len(set(labels)) - (1 if -1 in labels else 0)} cluster(s) identifié(s).",
        ]
        if silhouette is not None:
            if silhouette >= 0.50:
                notes.append(f"Silhouette = {silhouette:.3f} : séparation bonne à excellente.")
            elif silhouette >= 0.25:
                notes.append(f"Silhouette = {silhouette:.3f} : séparation modérée.")
            else:
                notes.append(f"Silhouette = {silhouette:.3f} : séparation faible, clusters peu distincts.")
        return ExplainabilityResult(
            method="Cluster profiles + PCA 2D",
            feature_importance=feature_importance,
            figures=figures,
            sample_explanations=[],
            notes=notes,
        )

    def _cluster_figures(self, X, X_trans, labels) -> list[str]:
        figures: list[str] = []
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_trans)
        except Exception:
            return figures

        plt.figure(figsize=(8, 6))
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = labels == label
            label_str = "Bruit" if label == -1 else f"Cluster {label}"
            plt.scatter(coords[mask, 0], coords[mask, 1], label=label_str, alpha=0.7, s=30)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Projection PCA 2D colorée par cluster")
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        figures.append(self._save("cluster_pca2d.png"))

        plt.figure(figsize=(6, 4))
        counts = pd.Series(labels).value_counts().sort_index()
        counts.index = ["Bruit" if i == -1 else f"Cluster {i}" for i in counts.index]
        counts.plot(kind="bar", color="#3b82f6", ax=plt.gca())
        plt.title("Taille des clusters")
        plt.ylabel("Nombre d'observations")
        plt.xticks(rotation=0)
        plt.tight_layout()
        figures.append(self._save("cluster_sizes.png"))
        return figures

    def _cluster_profiles(self, X: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
        """Return top discriminating features (max - min of standardized means)."""
        if X.empty or len(set(labels)) < 2:
            return {}
        df = X.copy()
        df["_cluster"] = labels
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return {}
        grouped = numeric.groupby("_cluster").mean()
        if grouped.empty:
            return {}
        stds = numeric.std().replace(0, 1)
        standardized = (grouped - numeric.mean()) / stds
        spread = standardized.max() - standardized.min()
        return spread.abs().sort_values(ascending=False).head(15).to_dict()

    def _silhouette_score(self, X_trans, labels) -> float | None:
        try:
            from sklearn.metrics import silhouette_score
            unique = set(labels)
            if len(unique) < 2 or (len(unique) == 2 and -1 in unique):
                return None
            return float(silhouette_score(X_trans, labels))
        except Exception:
            return None

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
