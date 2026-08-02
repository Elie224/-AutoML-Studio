"""Runtime configuration for AutoML Studio.

The platform is intentionally framework-agnostic so the same code runs inside
FastAPI, Streamlit or a notebook. Settings are resolved with `get_settings`
which caches the result for the lifetime of the process.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    project_root: Path = PROJECT_ROOT
    data_root: Path = DATA_ROOT
    artifacts_root: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    upload_root: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "uploads")
    model_root: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "models")
    report_root: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "reports")
    figure_root: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "figures")
    notebook_root: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "notebooks")
    sample_root: Path = field(default_factory=lambda: DATA_ROOT / "samples")
    mlflow_tracking_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "file:./artifacts/mlruns"))
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    optuna_trials: int = 25
    max_rows_for_shap: int = 500
    api_host: str = field(default_factory=lambda: os.environ.get("AUTOML_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.environ.get("AUTOML_PORT", "8000")))

    def ensure_dirs(self) -> "Settings":
        for path in (
            self.artifacts_root,
            self.upload_root,
            self.model_root,
            self.report_root,
            self.figure_root,
            self.notebook_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings().ensure_dirs()
