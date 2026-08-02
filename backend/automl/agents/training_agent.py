"""Training + optimization agent.

* Trains baseline candidates
* Optimizes top candidates with Optuna
* Returns a leaderboard of `ModelResult` and the best model
"""
from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ..core.config import get_settings
from ..core.schema import ModelResult, ProblemType
from .model_selection_agent import ModelSpec, get_models

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrainingArtifacts:
    best_model: ModelResult
    leaderboard: list[ModelResult]
    leaderboard_path: Path
    best_pipeline_path: Path
    best_model_path: Path


class TrainingAgent:
    name = "training"

    def __init__(self, problem_type: ProblemType, dataset_id: str, random_state: int = 42):
        self.problem_type = problem_type
        self.dataset_id = dataset_id
        self.random_state = random_state
        self.settings = get_settings()
        self.model_dir = self.settings.model_root / dataset_id
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_all(
        self,
        preprocessor,
        X: Any,
        y: Any | None = None,
        cv_folds: int = 5,
        optuna_trials: int = 15,
    ) -> TrainingArtifacts:
        models = get_models(self.problem_type, self.random_state)
        leaderboard: list[ModelResult] = []
        best_score = -np.inf
        best_result: ModelResult | None = None

        if y is not None and self.problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y if self.problem_type == ProblemType.CLASSIFICATION else None
            )
        else:
            X_train = X_valid = X
            y_train = y_valid = None

        for spec in models:
            try:
                result = self._train_one(spec, preprocessor, X_train, y_train, X_valid, y_valid, cv_folds, optuna_trials)
            except Exception as exc:  # pragma: no cover
                leaderboard.append(_error_result(spec, str(exc)))
                continue
            leaderboard.append(result)
            if result.metrics.get(self.primary_metric(), -np.inf) > best_score:
                best_score = result.metrics[self.primary_metric()]
                best_result = result

        if best_result is None:
            raise RuntimeError("Aucun modèle n'a pu être entraîné. Vérifiez les données et les hyperparamètres.")

        leaderboard.sort(key=lambda m: m.metrics.get(self.primary_metric(), -np.inf), reverse=True)
        leaderboard_path = self._save_leaderboard(leaderboard)
        return TrainingArtifacts(
            best_model=best_result,
            leaderboard=leaderboard,
            leaderboard_path=leaderboard_path,
            best_pipeline_path=Path(best_result.pipeline_path),
            best_model_path=Path(best_result.artifact_path),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _train_one(
        self,
        spec: ModelSpec,
        preprocessor,
        X_train,
        y_train,
        X_valid,
        y_valid,
        cv_folds: int,
        optuna_trials: int,
    ) -> ModelResult:
        pipeline = Pipeline([("pre", preprocessor), ("model", spec.factory())])

        start = time.perf_counter()

        if self.problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state) if self.problem_type == ProblemType.CLASSIFICATION else KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=self.primary_metric(), n_jobs=1).mean()
            best_params, score = self._optuna_search(spec, pipeline, X_train, y_train, cv_folds, optuna_trials)
            pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})
            pipeline.fit(X_train, y_train)
            metrics = self._supervised_metrics(pipeline, X_valid, y_valid, cv_score)
        else:
            pipeline.fit(X_train)
            metrics = self._unsupervised_metrics(pipeline, X_train)
            best_params = {}
            score = metrics.get(self.primary_metric(), 0.0)

        elapsed = time.perf_counter() - start
        artifact_path, pipeline_path = self._persist(spec, pipeline)
        return ModelResult(
            name=spec.name,
            problem_type=self.problem_type,
            metrics={k: float(v) for k, v in metrics.items()},
            params=best_params or {},
            artifact_path=str(artifact_path.relative_to(self.settings.artifacts_root)),
            pipeline_path=str(pipeline_path.relative_to(self.settings.artifacts_root)),
            training_time_s=round(elapsed, 2),
        )

    def _optuna_search(self, spec: ModelSpec, pipeline: Pipeline, X, y, cv_folds: int, n_trials: int):
        if not spec.space or n_trials <= 0:
            return {}, 0.0
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state) if self.problem_type == ProblemType.CLASSIFICATION else KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scoring = self.primary_metric()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        for trial in [None] * n_trials:
            params = {k: _sample(study, trial, k, v) for k, v in spec.space.items()}
            try:
                pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})
                score = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1).mean()
                study.enqueue_trial(params, user_attrs={"score": score})
            except Exception:
                study.enqueue_trial(params, user_attrs={"score": -np.inf})
        study.optimize(lambda t: t.user_attrs.get("score", -np.inf), n_trials=1)
        return study.best_trial.params if study.best_trial else {}, study.best_value or 0.0

    def _supervised_metrics(self, pipeline: Pipeline, X, y, cv_score: float) -> dict[str, float]:
        preds = pipeline.predict(X)
        if self.problem_type == ProblemType.CLASSIFICATION:
            return {
                "accuracy": float(accuracy_score(y, preds)),
                "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
                "cv_score": float(cv_score),
            }
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        return {
            "mae": float(mean_absolute_error(y, preds)),
            "rmse": rmse,
            "r2": float(r2_score(y, preds)),
            "cv_score": float(cv_score),
        }

    def _unsupervised_metrics(self, pipeline: Pipeline, X) -> dict[str, float]:
        try:
            transformed = pipeline[:-1].transform(X)
        except Exception:
            transformed = np.asarray(X)
        if hasattr(pipeline[-1], "labels_"):
            labels = pipeline[-1].labels_
        elif hasattr(pipeline[-1], "predict"):
            labels = pipeline[-1].predict(transformed)
        else:
            labels = np.zeros(len(transformed))
        try:
            sil = float(silhouette_score(transformed, labels)) if len(set(labels)) > 1 else 0.0
        except Exception:
            sil = 0.0
        return {"silhouette": sil, "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0))}

    def primary_metric(self) -> str:
        if self.problem_type == ProblemType.CLASSIFICATION:
            return "f1_weighted"
        if self.problem_type == ProblemType.REGRESSION:
            return "r2"
        return "silhouette"

    def _persist(self, spec: ModelSpec, pipeline: Pipeline) -> tuple[Path, Path]:
        model_id = uuid.uuid4().hex[:8]
        artifact = self.model_dir / f"{spec.name}_{model_id}.joblib"
        pipeline_path = self.model_dir / f"{spec.name}_{model_id}_pipeline.joblib"
        joblib.dump(pipeline[-1], artifact)
        joblib.dump(pipeline, pipeline_path)
        return artifact, pipeline_path

    def _save_leaderboard(self, leaderboard: list[ModelResult]) -> Path:
        import json

        path = self.model_dir / "leaderboard.json"
        path.write_text(json.dumps([m.to_dict() for m in leaderboard], indent=2, default=str), encoding="utf-8")
        return path


def _sample(study, trial, key, values):
    if isinstance(values, list):
        return study.sampler._categorical_sampler_for_trial(study, trial, key, values) if trial else values[0]
    return values


def _error_result(spec: ModelSpec, message: str) -> ModelResult:
    return ModelResult(
        name=spec.name,
        problem_type=spec.problem_type,
        metrics={"error": 0.0},
        params={"error": message},
        artifact_path="",
        pipeline_path="",
        training_time_s=0.0,
    )
