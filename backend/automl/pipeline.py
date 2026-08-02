"""AutoMLPipeline orchestrator: glue between agents and the public API."""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .agents.eda_agent import EDAAgent
from .agents.explainability_agent import ExplainabilityAgent
from .agents.feature_engineering_agent import FeatureEngineeringAgent
from .agents.preprocessing_agent import PreprocessingAgent
from .agents.problem_detection_agent import ProblemDetectionAgent
from .agents.qa_agent import QAAgent
from .agents.reporting_agent import ReportingAgent
from .agents.training_agent import TrainingAgent
from .core.config import get_settings
from .core.io import (
    dataframe_summary,
    load_csv,
    save_upload,
    store_metadata,
)
from .core.schema import (
    DatasetSummary,
    EDAResult,
    PipelineResult,
    ProblemType,
)



class AutoMLPipeline:
    """High-level orchestrator used by the API and the Streamlit UI."""

    def __init__(
        self,
        source: str | Path | bytes,
        target: str | None = None,
        name: str | None = None,
        optuna_trials: int = 15,
        cv_folds: int = 5,
        run_explainability: bool = True,
    ):
        self.settings = get_settings()
        self.target = target
        self.optuna_trials = optuna_trials
        self.cv_folds = cv_folds
        self.run_explainability = run_explainability

        if isinstance(source, (str, Path)):
            source = Path(source)
            if not source.exists():
                raise FileNotFoundError(source)
            dataset_id = uuid.uuid4().hex[:12]
            saved = save_upload(source, source.name)
            self.df = load_csv(saved)
            self.dataset_id = dataset_id
            self.dataset_name = name or source.stem
        else:
            dataset_id = uuid.uuid4().hex[:12]
            saved = save_upload(source, name or "upload.csv")
            self.df = load_csv(saved)
            self.dataset_id = dataset_id
            self.dataset_name = (name or "upload").rsplit(".", 1)[0]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        summary = dataframe_summary(self.df, self.dataset_id, self.dataset_name, target=self.target)
        store_metadata(summary)

        eda_agent = EDAAgent(self.dataset_id)
        eda = eda_agent.run(self.df, target=self.target)

        problem_agent = ProblemDetectionAgent()
        problem_type = problem_agent.detect(self.df, self.target)
        summary.problem_type = problem_type
        store_metadata(summary)

        preprocessing_agent = PreprocessingAgent()
        preprocessing_plan = preprocessing_agent.suggest_plan(self.df, self.target)
        prepared = preprocessing_agent.prepare(self.df, self.target)

        # Training
        trainer = TrainingAgent(problem_type, self.dataset_id)
        training = trainer.train_all(
            prepared.preprocessor,
            prepared.X,
            prepared.y,
            cv_folds=self.cv_folds,
            optuna_trials=self.optuna_trials,
        )

        # Explainability
        explainability = None
        if self.run_explainability:
            try:
                best_pipeline = joblib.load(self.settings.artifacts_root / training.best_model.pipeline_path)
                explainability_agent = ExplainabilityAgent(self.dataset_id)
                explainability = explainability_agent.explain(
                    best_pipeline,
                    prepared.X,
                    problem_type,
                    max_rows=self.settings.max_rows_for_shap,
                )
            except Exception as exc:
                explainability = None
                print(f"[explainability] skipped: {exc}")

        # Reporting
        result = PipelineResult(
            dataset=summary,
            eda=eda,
            preprocessing=preprocessing_plan,
            problem_type=problem_type,
            leaderboard=training.leaderboard,
            best_model=training.best_model,
            explainability=explainability,
            artifacts={
                "leaderboard": str(training.leaderboard_path.relative_to(self.settings.artifacts_root)),
                "pipeline": training.best_model.pipeline_path,
                "model": training.best_model.artifact_path,
            },
        )

        reporter = ReportingAgent(self.dataset_id)
        html_path = reporter.write_html(result)
        pdf_path = reporter.write_pdf(result)
        notebook_path = reporter.write_notebook(result)
        exec_path = reporter.write_executive(result)
        result.report_paths = {
            "html": str(html_path.relative_to(self.settings.artifacts_root)),
            "pdf": str(pdf_path.relative_to(self.settings.artifacts_root)) if pdf_path else "",
            "notebook": str(notebook_path.relative_to(self.settings.artifacts_root)),
            "executive": str(exec_path.relative_to(self.settings.artifacts_root)),
        }

        # Export ONNX (best-effort)
        try:
            from .agents.deployment_agent import DeploymentAgent

            deploy = DeploymentAgent(self.dataset_id)
            result.artifacts["onnx"] = str(deploy.export_onnx(training.best_model.pipeline_path))
        except Exception as exc:
            print(f"[deployment] ONNX export skipped: {exc}")

        # Save full result
        result_path = self.settings.artifacts_root / "reports" / self.dataset_id / "pipeline_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def ask(self, question: str, eda: EDAResult | None = None, leaderboard=None) -> dict[str, Any]:
        qa = QAAgent(self.df, target=self.target, eda=eda, leaderboard=leaderboard or [])
        return qa.ask(question)





