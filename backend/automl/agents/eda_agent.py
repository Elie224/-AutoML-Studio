"""EDA Agent: produces statistics, missing values, correlations, distributions,
outliers, class balance, temporal and anomaly summaries, and an HTML/PNG report.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.config import get_settings
from ..core.schema import EDAResult


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _df_to_records(df: pd.DataFrame) -> dict:
    return json.loads(df.to_json(orient="columns", default_handler=str))


class EDAAgent:
    """Compute the full EDA summary and generate figures."""

    name = "eda"

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.settings = get_settings()
        self.figure_dir = self.settings.figure_root / dataset_id
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, target: str | None = None) -> EDAResult:
        result = EDAResult(
            summary=self._summary(df, target),
            descriptive_stats=self._descriptive(df),
            missing_values=self._missing(df),
            duplicates=int(df.duplicated().sum()),
            correlations=self._correlations(df),
            outliers=self._outliers(df),
            class_balance=self._class_balance(df, target),
            categorical_summary=self._categorical(df),
            temporal_summary=self._temporal(df),
            anomaly_summary=self._anomalies(df),
            figures=[],
            insights=[],
        )
        result.figures = self._figures(df, target)
        result.insights = self._insights(result)
        return result

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------
    def _summary(self, df: pd.DataFrame, target: str | None) -> dict:
        return {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
            "target": target,
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "n_numeric": int(df.select_dtypes(include="number").shape[1]),
            "n_categorical": int(df.select_dtypes(include=["object", "category"]).shape[1]),
            "n_datetime": int(sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)),
        }

    def _descriptive(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return {"numeric": {}, "categorical": {}}
        desc = numeric.describe().transpose()
        desc_records = desc.to_dict(orient="index")
        return {"numeric": {k: {kk: _jsonable(vv) for kk, vv in v.items()} for k, v in desc_records.items()}}

    def _missing(self, df: pd.DataFrame) -> dict:
        counts = df.isna().sum()
        total = max(len(df), 1)
        return {
            "total_missing": int(counts.sum()),
            "missing_per_column": {
                str(col): {"count": int(cnt), "ratio": round(cnt / total, 4)}
                for col, cnt in counts.items()
                if cnt > 0
            },
            "complete_rows": int(df.dropna().shape[0]),
        }

    def _correlations(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return {"matrix": {}, "top_pairs": []}
        corr = numeric.corr(numeric_only=True)
        matrix = {
            str(row): {str(col): _jsonable(corr.loc[row, col]) for col in corr.columns}
            for row in corr.index
        }
        # Top absolute correlations excluding self-correlation
        pairs = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                val = corr.loc[a, b]
                if pd.isna(val):
                    continue
                pairs.append({"a": str(a), "b": str(b), "value": _jsonable(val)})
        pairs.sort(key=lambda x: abs(x["value"]), reverse=True)
        return {"matrix": matrix, "top_pairs": pairs[:10]}

    def _outliers(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return {}
        out: dict[str, dict] = {}
        for col in numeric.columns:
            series = numeric[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (numeric[col] < lower) | (numeric[col] > upper)
            n_out = int(mask.sum())
            if n_out > 0:
                out[str(col)] = {
                    "count": n_out,
                    "ratio": round(n_out / max(len(numeric), 1), 4),
                    "lower_bound": _jsonable(lower),
                    "upper_bound": _jsonable(upper),
                }
        return out

    def _class_balance(self, df: pd.DataFrame, target: str | None) -> dict | None:
        if not target or target not in df.columns:
            return None
        counts = df[target].value_counts(dropna=False)
        total = max(counts.sum(), 1)
        values = counts.tolist()
        ratio = max(values) / total if values else 1.0
        return {
            "target": target,
            "classes": {str(k): int(v) for k, v in counts.items()},
            "n_classes": int(len(counts)),
            "imbalance_ratio": round(float(ratio), 4),
            "is_imbalanced": bool(ratio > 5 and len(counts) > 1),
        }

    def _categorical(self, df: pd.DataFrame) -> dict:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        summary: dict[str, dict] = {}
        for col in cat_cols:
            counts = df[col].value_counts(dropna=True).head(10)
            summary[str(col)] = {
                "n_unique": int(df[col].nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in counts.items()},
                "cardinality": "high" if df[col].nunique(dropna=True) > 50 else "low",
            }
        return summary

    def _temporal(self, df: pd.DataFrame) -> dict | None:
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not datetime_cols:
            return None
        out: dict[str, dict] = {}
        for col in datetime_cols:
            series = df[col].dropna().sort_values()
            if series.empty:
                continue
            diffs = series.diff().dt.total_seconds().dropna()
            median_step = float(diffs.median()) if not diffs.empty else None
            out[col] = {
                "min": series.min().isoformat(),
                "max": series.max().isoformat(),
                "range_s": (series.max() - series.min()).total_seconds(),
                "median_step_s": median_step,
                "is_regular": bool(median_step is not None and diffs.std() < abs(median_step or 1) * 0.1),
            }
        return out

    def _anomalies(self, df: pd.DataFrame) -> dict:
        from sklearn.ensemble import IsolationForest

        numeric = df.select_dtypes(include="number").fillna(0)
        if numeric.shape[0] < 20 or numeric.shape[1] == 0:
            return {"available": False, "n_anomalies": 0, "ratio": 0.0}
        model = IsolationForest(contamination=0.05, random_state=42)
        preds = model.fit_predict(numeric)
        n_anom = int((preds == -1).sum())
        return {
            "available": True,
            "n_anomalies": n_anom,
            "ratio": round(n_anom / max(len(df), 1), 4),
            "method": "IsolationForest(contamination=0.05)",
        }

    # ------------------------------------------------------------------
    # Figures and insights
    # ------------------------------------------------------------------
    def _figures(self, df: pd.DataFrame, target: str | None) -> list[str]:
        figures: list[str] = []
        sns.set_theme(style="whitegrid")

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            # Missing values bar
            missing = df.isna().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(missing))))
                missing.sort_values().plot(kind="barh", ax=ax, color="#3b82f6")
                ax.set_title("Valeurs manquantes par colonne")
                ax.set_xlabel("Nombre de valeurs manquantes")
                fig.tight_layout()
                figures.append(self._save(fig, "missing_values.png"))

            # Distributions for up to 12 numeric columns
            cols = numeric.columns[:12]
            n = len(cols)
            rows = max(1, (n + 3) // 4)
            fig, axes = plt.subplots(rows, 4, figsize=(4 * 4, 3 * rows))
            axes = axes.flatten() if n > 1 else [axes]
            for i, col in enumerate(cols):
                ax = axes[i]
                series = numeric[col].dropna()
                sns.histplot(series, kde=True, ax=ax, color="#22c55e")
                ax.set_title(str(col))
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
            fig.suptitle("Distributions (numériques)")
            fig.tight_layout()
            figures.append(self._save(fig, "distributions.png"))

            # Correlation heatmap
            if numeric.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(min(12, 0.7 * numeric.shape[1] + 4), min(10, 0.7 * numeric.shape[1] + 4)))
                sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax, annot=False)
                ax.set_title("Matrice de corrélation")
                fig.tight_layout()
                figures.append(self._save(fig, "correlation.png"))

        # Class balance
        if target and target in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            counts = df[target].value_counts(dropna=False).head(20)
            sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, palette="viridis")
            ax.set_title(f"Distribution de la cible: {target}")
            ax.set_xlabel("Count")
            fig.tight_layout()
            figures.append(self._save(fig, "target_distribution.png"))

        # Outliers boxplot
        if not numeric.empty:
            cols = numeric.columns[:10]
            fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(cols)), 4))
            numeric[cols].boxplot(ax=ax)
            ax.set_title("Boxplots (détection d'outliers)")
            fig.tight_layout()
            figures.append(self._save(fig, "boxplots.png"))

        return figures

    def _save(self, fig, name: str) -> str:
        path = self.figure_dir / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.settings.artifacts_root))

    def _insights(self, result: EDAResult) -> list[str]:
        insights: list[str] = []
        if result.missing_values["total_missing"] > 0:
            top = sorted(result.missing_values["missing_per_column"].items(), key=lambda x: x[1]["count"], reverse=True)[:3]
            cols = ", ".join(f"{c} ({v['ratio']*100:.1f}%)" for c, v in top)
            insights.append(f"Valeurs manquantes détectées sur: {cols}.")
        if result.duplicates > 0:
            insights.append(f"{result.duplicates} doublons exacts identifiés dans le jeu de données.")
        if result.class_balance and result.class_balance.get("is_imbalanced"):
            insights.append("La cible est déséquilibrée : envisagez SMOTE, class_weight ou stratified sampling.")
        if result.outliers:
            top_out = sorted(result.outliers.items(), key=lambda x: x[1]["count"], reverse=True)[:3]
            cols = ", ".join(c for c, _ in top_out)
            insights.append(f"Outliers détectés (méthode IQR) sur: {cols}.")
        if result.anomaly_summary.get("available") and result.anomaly_summary.get("ratio", 0) > 0.1:
            insights.append(f"{result.anomaly_summary['n_anomalies']} anomalies potentielles (Isolation Forest).")
        if not insights:
            insights.append("Le jeu de données semble propre : aucune anomalie majeure détectée.")
        return insights
