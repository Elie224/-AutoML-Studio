"""Identifier column detector.

Identifies columns that look like identifiers and should be excluded from
EDA, preprocessing, training, and clustering. Detection is based on:
  * name heuristics (id, _id, uuid, code, ref, tweet_id, …)
  * uniqueness ratio == 1.0 (every row has a different value)
  * monotonic large integers or hash-like strings
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


_ID_NAME_PATTERNS = (
    re.compile(r"(^|_)id($|_)", re.IGNORECASE),
    re.compile(r"_id$", re.IGNORECASE),
    re.compile(r"^(uuid|guid|hash|token|key|sku|code|ref|reference)$", re.IGNORECASE),
    re.compile(r"(tweet|user|post|comment|message|document|record|order|invoice|transaction|tracking)_?id", re.IGNORECASE),
    re.compile(r"^(index|rowid|rownum)$", re.IGNORECASE),
)


@dataclass(frozen=True)
class IDColumn:
    name: str
    reason: str


def is_identifier_name(name: str) -> bool:
    """Return True when `name` matches one of the known identifier patterns."""
    if not isinstance(name, str):
        return False
    candidate = name.strip()
    if not candidate:
        return False
    return any(pattern.search(candidate) for pattern in _ID_NAME_PATTERNS)


def detect_id_columns(df: pd.DataFrame, threshold: float = 0.98) -> list[IDColumn]:
    """Return the list of columns that look like identifiers."""
    detected: list[IDColumn] = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        name_match = is_identifier_name(str(col))
        unique_ratio = series.nunique(dropna=True) / n_rows if n_rows else 0.0

        if name_match and unique_ratio >= threshold:
            detected.append(IDColumn(str(col), "Nom d'identifiant connu et valeurs uniques."))
            continue

        if unique_ratio >= 0.999 and pd.api.types.is_numeric_dtype(series):
            detected.append(IDColumn(str(col), "Colonne numérique avec une valeur unique par ligne."))
            continue

        if unique_ratio >= 0.999 and pd.api.types.is_string_dtype(series) and _looks_like_hash(series):
            detected.append(IDColumn(str(col), "Chaîne de caractères avec une valeur unique par ligne (hash probable)."))
            continue

    return detected


def _looks_like_hash(series: pd.Series, sample: int = 200) -> bool:
    """Detect hash-like string columns (long hex / base64 strings, all unique)."""
    sample_values = series.dropna().astype(str).head(sample)
    if sample_values.empty:
        return False
    long_enough = sample_values.str.len().mean() >= 16
    mostly_alnum = sample_values.str.match(r"^[A-Za-z0-9+/=_\-]+$").mean() >= 0.9
    return bool(long_enough and mostly_alnum)
