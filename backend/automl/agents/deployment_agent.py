"""Deployment Agent: export the trained pipeline to pickle + ONNX + FastAPI stub."""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import joblib

from ..core.config import get_settings


class DeploymentAgent:
    name = "deployment"

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.settings = get_settings()
        self.deploy_dir = self.settings.artifacts_root / "deploy" / dataset_id
        self.deploy_dir.mkdir(parents=True, exist_ok=True)

    def export_pickle(self, pipeline_path: str | Path) -> Path:
        pipeline_path = self.settings.artifacts_root / pipeline_path
        target = self.deploy_dir / "model.pkl"
        shutil.copy2(pipeline_path, target)
        return target

    def export_onnx(self, pipeline_path: str | Path) -> str:
        """Best-effort ONNX export. Falls back to a placeholder note if the
        model type is not supported by skl2onnx."""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except Exception as exc:
            note = self.deploy_dir / "onnx_note.txt"
            note.write_text(f"skl2onnx unavailable: {exc}\n", encoding="utf-8")
            return str(note.relative_to(self.settings.artifacts_root))

        pipeline = joblib.load(self.settings.artifacts_root / pipeline_path)
        try:
            n_features = pipeline[:-1].transform(pipeline.feature_names_in_.reshape(1, -1) if hasattr(pipeline, "feature_names_in_") else None).shape[1]
        except Exception:
            try:
                pre = pipeline[:-1]
                sample = pipeline.feature_names_in_.reshape(1, -1)
                n_features = pre.transform(sample).shape[1]
            except Exception:
                n_features = 10
        try:
            onnx_model = convert_sklearn(pipeline, initial_types=[("input", FloatTensorType([None, n_features]))])
            target = self.deploy_dir / "model.onnx"
            target.write_bytes(onnx_model.SerializeToString())
            return str(target.relative_to(self.settings.artifacts_root))
        except Exception as exc:
            note = self.deploy_dir / "onnx_note.txt"
            note.write_text(f"ONNX conversion failed: {exc}\n", encoding="utf-8")
            return str(note.relative_to(self.settings.artifacts_root))

    def write_fastapi_stub(self, feature_names: list[str]) -> Path:
        body = textwrap.dedent(
            f"""\
            from fastapi import FastAPI
            from pydantic import BaseModel
            import joblib
            import pandas as pd

            app = FastAPI(title="AutoML Studio model")
            model = joblib.load("model.pkl")

            class Record(BaseModel):
                features: dict

            @app.get("/health")
            def health():
                return {{"status": "ok"}}

            @app.post("/predict")
            def predict(record: Record):
                df = pd.DataFrame([record.features])[{feature_names!r}]
                return {{"prediction": model.predict(df).tolist()}}
            """
        )
        path = self.deploy_dir / "app.py"
        path.write_text(body, encoding="utf-8")
        return path

    def write_dockerfile(self) -> Path:
        body = textwrap.dedent(
            """\
            FROM python:3.11-slim
            WORKDIR /app
            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt
            COPY model.pkl .
            COPY app.py .
            EXPOSE 8000
            CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
            """
        )
        path = self.deploy_dir / "Dockerfile"
        path.write_text(body, encoding="utf-8")
        return path
