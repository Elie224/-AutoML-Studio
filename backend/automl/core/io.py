"""Dataset I/O: ingestion from CSV / Excel / Parquet / SQL and lightweight persistence."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from .config import get_settings
from .schema import DatasetSummary, ProblemType


SUPPORTED_READERS = {
    ".csv": lambda p: pd.read_csv(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".xlsx": lambda p: pd.read_excel(p),
    ".xls": lambda p: pd.read_excel(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".json": lambda p: pd.read_json(p),
    ".jsonl": lambda p: pd.read_json(p, lines=True),
}


def infer_columns(df: pd.DataFrame) -> list[dict]:
    """Infer column dtypes with extended metadata used by downstream agents."""
    info: list[dict] = []
    for col in df.columns:
        series = df[col]
        kind = "categorical" if series.dtype == "object" or str(series.dtype).startswith("category") else "numeric"
        try:
            is_dt = pd.api.types.is_datetime64_any_dtype(series)
        except Exception:
            is_dt = False
        if is_dt:
            kind = "datetime"
        info.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "kind": kind,
                "n_unique": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
                "example": _safe_example(series),
            }
        )
    return info


def _safe_example(series: pd.Series) -> object:
    try:
        value = series.dropna().iloc[0] if series.dropna().size else None
    except Exception:
        value = None
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    reader = SUPPORTED_READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    return reader(path)


def load_sql(query: str, connection_string: str) -> pd.DataFrame:
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    return df


def load_api_json(payload: list[dict] | dict) -> pd.DataFrame:
    """Load data from an API JSON payload (records or single object)."""
    if isinstance(payload, dict):
        payload = [payload]
    return pd.DataFrame(payload)


def save_upload(payload: bytes | Path, original_name: str) -> Path:
    """Persist an uploaded file under a unique id and return the saved path."""
    settings = get_settings()
    dataset_id = uuid.uuid4().hex[:12]
    target_dir = settings.upload_root / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(original_name).name or "upload.csv"
    target_path = target_dir / safe_name
    if isinstance(payload, Path):
        shutil.copy2(payload, target_path)
    else:
        target_path.write_bytes(payload)
    return target_path


def dataframe_summary(
    df: pd.DataFrame,
    dataset_id: str,
    name: str,
    target: str | None = None,
) -> DatasetSummary:
    return DatasetSummary(
        dataset_id=dataset_id,
        name=name,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        size_bytes=int(df.memory_usage(deep=True).sum()),
        columns_info=infer_columns(df),
        target_column=target,
        problem_type=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def store_metadata(summary: DatasetSummary) -> Path:
    settings = get_settings()
    path = settings.upload_root / summary.dataset_id / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_metadata(dataset_id: str) -> DatasetSummary | None:
    settings = get_settings()
    path = settings.upload_root / dataset_id / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("problem_type"):
        data["problem_type"] = ProblemType(data["problem_type"])
    return DatasetSummary(**data)
