"""Smoke tests for AutoML Studio. Run with: pytest backend/tests/ -v"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automl.agents.eda_agent import EDAAgent
from automl.agents.preprocessing_agent import PreprocessingAgent
from automl.agents.problem_detection_agent import ProblemDetectionAgent
from automl.core.io import load_csv, save_upload
from automl.core.schema import ProblemType


SAMPLE = Path(__file__).resolve().parents[2] / "data" / "samples" / "titanic.csv"


@pytest.fixture(scope="module")
def titanic_df() -> pd.DataFrame:
    if not SAMPLE.exists():
        pytest.skip("Titanic sample missing; run data/generate_samples.py first")
    return load_csv(SAMPLE)


def test_load_csv(titanic_df):
    assert titanic_df.shape[0] > 100
    assert "Survived" in titanic_df.columns


def test_problem_detection(titanic_df):
    agent = ProblemDetectionAgent()
    ptype = agent.detect(titanic_df, "Survived")
    assert ptype == ProblemType.CLASSIFICATION


def test_preprocessing_plan(titanic_df):
    agent = PreprocessingAgent()
    plan = agent.suggest_plan(titanic_df, "Survived")
    assert "Fare" in plan.imputation
    assert any(v in {"median", "most_frequent", "none"} for v in plan.imputation.values())


def test_eda_runs(titanic_df, tmp_path: Path):
    agent = EDAAgent("pytest_eda")
    result = agent.run(titanic_df, target="Survived")
    assert result.summary["shape"][0] == titanic_df.shape[0]
    assert result.duplicates >= 0
    assert isinstance(result.insights, list) and len(result.insights) > 0
    # At least one figure is generated
    assert any("missing_values" in f or "distributions" in f or "correlation" in f for f in result.figures)
