"""FastAPI surface for AutoML Studio."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..agents.eda_agent import EDAAgent
from ..agents.preprocessing_agent import PreprocessingAgent
from ..agents.qa_agent import QAAgent
from ..core.config import get_settings
from ..core.io import dataframe_summary, load_csv, store_metadata
from ..core.schema import ProblemType
from ..pipeline import AutoMLPipeline


app = FastAPI(
    title="AutoML Studio API",
    description="Plateforme AutoML de bout en bout : import, EDA, preprocessing, entraînement, explicabilité, déploiement.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "AutoML Studio",
        "version": "0.1.0",
        "endpoints": [
            "/health",
            "/upload",
            "/datasets/{dataset_id}",
            "/eda/{dataset_id}",
            "/run",
            "/qa",
            "/predict/{dataset_id}",
            "/report/{dataset_id}/{kind}",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {"status": "ok", "artifacts": str(settings.artifacts_root)}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), target: str | None = Form(None)) -> dict[str, Any]:
    payload = await file.read()
    settings = get_settings()
    dataset_id = uuid.uuid4().hex[:12]
    target_dir = settings.upload_root / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.csv").name
    saved_path = target_dir / safe_name
    saved_path.write_bytes(payload)

    try:
        df = load_csv(saved_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read file: {exc}") from exc

    summary = dataframe_summary(df, dataset_id, Path(file.filename or "upload").stem, target=target)
    store_metadata(summary)
    return {"dataset_id": dataset_id, "summary": summary.to_dict()}


@app.get("/datasets/{dataset_id}")
def dataset_info(dataset_id: str) -> dict[str, Any]:
    settings = get_settings()
    path = settings.upload_root / dataset_id / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/eda/{dataset_id}")
def eda(dataset_id: str, target: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    summary_path = settings.upload_root / dataset_id / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    data_files = list((settings.upload_root / dataset_id).glob("*.csv")) + list((settings.upload_root / dataset_id).glob("*.parquet"))
    if not data_files:
        raise HTTPException(status_code=400, detail="Dataset file missing")
    df = load_csv(data_files[0])
    result = EDAAgent(dataset_id).run(df, target=target)
    return result.to_dict()


class RunRequest(BaseModel):
    dataset_id: str
    target: str | None = None
    optuna_trials: int = 15
    cv_folds: int = 5
    run_explainability: bool = True


@app.post("/run")
def run_pipeline(req: RunRequest) -> dict[str, Any]:
    settings = get_settings()
    upload_dir = settings.upload_root / req.dataset_id
    files = list(upload_dir.glob("*.csv")) + list(upload_dir.glob("*.parquet")) + list(upload_dir.glob("*.xlsx"))
    if not files:
        raise HTTPException(status_code=404, detail="Dataset not found")
    pipeline = AutoMLPipeline(
        source=files[0],
        target=req.target,
        name=files[0].stem,
        optuna_trials=req.optuna_trials,
        cv_folds=req.cv_folds,
        run_explainability=req.run_explainability,
    )
    pipeline.dataset_id = req.dataset_id
    result = pipeline.run()
    return result.to_dict()


class QARequest(BaseModel):
    dataset_id: str
    question: str


@app.post("/qa")
def qa(req: QARequest) -> dict[str, Any]:
    settings = get_settings()
    files = list((settings.upload_root / req.dataset_id).glob("*.csv")) + list((settings.upload_root / req.dataset_id).glob("*.parquet"))
    if not files:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = load_csv(files[0])
    summary_path = settings.upload_root / req.dataset_id / "summary.json"
    target = None
    if summary_path.exists():
        target = json.loads(summary_path.read_text(encoding="utf-8")).get("target_column")
    qa_agent = QAAgent(df, target=target)
    return qa_agent.ask(req.question)


class PredictRequest(BaseModel):
    features: dict


@app.post("/predict/{dataset_id}")
def predict(dataset_id: str, req: PredictRequest) -> dict[str, Any]:
    settings = get_settings()
    model_dir = settings.model_root / dataset_id
    pipelines = sorted(model_dir.glob("*_pipeline.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pipelines:
        raise HTTPException(status_code=404, detail="No trained model")
    pipeline = joblib.load(pipelines[0])
    df = pd.DataFrame([req.features])
    try:
        prediction = pipeline.predict(df).tolist()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"prediction": prediction, "model": pipelines[0].name}


@app.get("/report/{dataset_id}/{kind}")
def report(dataset_id: str, kind: str):
    settings = get_settings()
    report_dir = settings.report_root / dataset_id
    candidates = {
        "html": report_dir / "report.html",
        "pdf": report_dir / "report.pdf",
        "executive": report_dir / "executive_summary.md",
        "notebook": settings.notebook_root / dataset_id / "reproduce.ipynb",
    }
    target = candidates.get(kind)
    if target is None or not target.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(target)


@app.get("/artifact")
def artifact(path: str):
    settings = get_settings()
    full = settings.artifacts_root / path
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(full)
