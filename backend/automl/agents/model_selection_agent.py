"""Model registry: each problem type exposes a curated list of estimators with
default hyperparameters and search spaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM, SVC

from ..core.schema import ProblemType


@dataclass
class ModelSpec:
    name: str
    problem_type: ProblemType
    factory: callable
    space: dict[str, Any] = field(default_factory=dict)
    supports_proba: bool = False


def classification_models(random_state: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="LogisticRegression",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: LogisticRegression(max_iter=1000, random_state=random_state),
            space={"C": [0.1, 1.0, 5.0], "solver": ["lbfgs", "liblinear"]},
            supports_proba=True,
        ),
        ModelSpec(
            name="RandomForest",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: RandomForestClassifier(random_state=random_state, n_estimators=200),
            space={"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
            supports_proba=True,
        ),
        ModelSpec(
            name="XGBoost",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: _try_xgb_classifier(random_state),
            space={"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            supports_proba=True,
        ),
        ModelSpec(
            name="LightGBM",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: _try_lgbm_classifier(random_state),
            space={"n_estimators": [200, 400], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
            supports_proba=True,
        ),
        ModelSpec(
            name="CatBoost",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: _try_catboost_classifier(random_state),
            space={"iterations": [200, 400], "depth": [4, 8], "learning_rate": [0.05, 0.1]},
            supports_proba=True,
        ),
        ModelSpec(
            name="KNN",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: KNeighborsClassifier(),
            space={"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]},
        ),
        ModelSpec(
            name="SVM",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: SVC(probability=True, random_state=random_state),
            space={"C": [0.5, 1.0, 5.0], "kernel": ["rbf", "linear"]},
            supports_proba=True,
        ),
        ModelSpec(
            name="MLP",
            problem_type=ProblemType.CLASSIFICATION,
            factory=lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
            space={"hidden_layer_sizes": [(64,), (64, 32)], "alpha": [1e-4, 1e-3]},
            supports_proba=True,
        ),
    ]


def regression_models(random_state: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="LinearRegression",
            problem_type=ProblemType.REGRESSION,
            factory=lambda: LinearRegression(),
            space={},
        ),
        ModelSpec(
            name="RandomForest",
            problem_type=ProblemType.REGRESSION,
            factory=lambda: RandomForestRegressor(random_state=random_state, n_estimators=200),
            space={"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        ModelSpec(
            name="XGBoost",
            problem_type=ProblemType.REGRESSION,
            factory=lambda: _try_xgb_regressor(random_state),
            space={"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        ),
        ModelSpec(
            name="LightGBM",
            problem_type=ProblemType.REGRESSION,
            factory=lambda: _try_lgbm_regressor(random_state),
            space={"n_estimators": [200, 400], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
        ),
        ModelSpec(
            name="CatBoost",
            problem_type=ProblemType.REGRESSION,
            factory=lambda: _try_catboost_regressor(random_state),
            space={"iterations": [200, 400], "depth": [4, 8], "learning_rate": [0.05, 0.1]},
        ),
        ModelSpec(
            name="GradientBoosting",
            problem_type=ProblemType.REGRESSION,
            factory=lambda: GradientBoostingRegressor(random_state=random_state),
            space={"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        ),
    ]


def clustering_models(random_state: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="KMeans",
            problem_type=ProblemType.CLUSTERING,
            factory=lambda: KMeans(n_clusters=4, random_state=random_state, n_init=10),
            space={"n_clusters": [3, 4, 5, 8]},
        ),
        ModelSpec(
            name="DBSCAN",
            problem_type=ProblemType.CLUSTERING,
            factory=lambda: DBSCAN(eps=0.5, min_samples=5),
            space={"eps": [0.3, 0.5, 0.8], "min_samples": [3, 5, 10]},
        ),
        ModelSpec(
            name="Agglomerative",
            problem_type=ProblemType.CLUSTERING,
            factory=lambda: AgglomerativeClustering(n_clusters=4),
            space={"n_clusters": [3, 4, 5]},
        ),
    ]


def anomaly_models(random_state: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="IsolationForest",
            problem_type=ProblemType.ANOMALY_DETECTION,
            factory=lambda: IsolationForest(random_state=random_state, contamination=0.05),
            space={"contamination": [0.05, 0.1]},
        ),
        ModelSpec(
            name="OneClassSVM",
            problem_type=ProblemType.ANOMALY_DETECTION,
            factory=lambda: OneClassSVM(nu=0.05),
            space={"nu": [0.05, 0.1]},
        ),
        ModelSpec(
            name="LocalOutlierFactor",
            problem_type=ProblemType.ANOMALY_DETECTION,
            factory=lambda: LocalOutlierFactor(contamination=0.05),
            space={"n_neighbors": [10, 20]},
        ),
    ]


def get_models(problem_type: ProblemType, random_state: int = 42) -> list[ModelSpec]:
    if problem_type == ProblemType.CLASSIFICATION:
        return classification_models(random_state)
    if problem_type == ProblemType.REGRESSION:
        return regression_models(random_state)
    if problem_type == ProblemType.CLUSTERING:
        return clustering_models(random_state)
    if problem_type == ProblemType.ANOMALY_DETECTION:
        return anomaly_models(random_state)
    raise ValueError(f"Unsupported problem type: {problem_type}")


# Lazy imports for optional libraries
def _try_xgb_classifier(random_state: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        verbosity=0,
    )


def _try_xgb_regressor(random_state: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        tree_method="hist",
        verbosity=0,
    )


def _try_lgbm_classifier(random_state: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=random_state,
        verbosity=-1,
    )


def _try_lgbm_regressor(random_state: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        random_state=random_state,
        verbosity=-1,
    )


def _try_catboost_classifier(random_state: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=False, random_seed=random_state)


def _try_catboost_regressor(random_state: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=False, random_seed=random_state)
