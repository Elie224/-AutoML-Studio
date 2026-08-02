"""Typed data structures shared across agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"


@dataclass
class DatasetSummary:
    dataset_id: str
    name: str
    rows: int
    columns: int
    size_bytes: int
    columns_info: list[dict[str, Any]]
    target_column: str | None = None
    problem_type: ProblemType | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.problem_type is not None:
            data["problem_type"] = self.problem_type.value
        return data


@dataclass
class EDAResult:
    summary: dict[str, Any]
    descriptive_stats: dict[str, Any]
    missing_values: dict[str, Any]
    duplicates: int
    correlations: dict[str, Any]
    outliers: dict[str, Any]
    class_balance: dict[str, Any] | None
    categorical_summary: dict[str, Any]
    temporal_summary: dict[str, Any] | None
    anomaly_summary: dict[str, Any]
    figures: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    quality_score: dict[str, Any] = field(default_factory=dict)
    id_columns: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class PreprocessingPlan:
    drop_columns: list[str] = field(default_factory=list)
    imputation: dict[str, str] = field(default_factory=dict)
    encoders: dict[str, str] = field(default_factory=dict)
    scaling: str = "standard"
    feature_engineering: list[str] = field(default_factory=list)
    dimensionality_reduction: str | None = None
    notes: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetric:
    name: str
    value: float


@dataclass
class ModelResult:
    name: str
    problem_type: ProblemType
    metrics: dict[str, float]
    params: dict[str, Any]
    artifact_path: str
    pipeline_path: str
    training_time_s: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["problem_type"] = self.problem_type.value
        return data


@dataclass
class ExplainabilityResult:
    method: str
    feature_importance: dict[str, float]
    figures: list[str] = field(default_factory=list)
    sample_explanations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    dataset: DatasetSummary
    eda: EDAResult | None
    preprocessing: PreprocessingPlan
    problem_type: ProblemType
    leaderboard: list[ModelResult]
    best_model: ModelResult
    explainability: ExplainabilityResult | None
    artifacts: dict[str, str]
    target_suggestions: list[dict[str, Any]] = field(default_factory=list)
    report_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "eda": self.eda.to_dict() if self.eda else None,
            "preprocessing": self.preprocessing.to_dict(),
            "problem_type": self.problem_type.value,
            "leaderboard": [m.to_dict() for m in self.leaderboard],
            "best_model": self.best_model.to_dict(),
            "explainability": self.explainability.to_dict() if self.explainability else None,
            "artifacts": self.artifacts,
            "target_suggestions": self.target_suggestions,
            "report_paths": self.report_paths,
        }




