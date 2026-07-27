"""General-purpose utilities used across the project."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_name(value: object) -> str:
    """Normalize model or compound names for conservative fallback matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """Hash a JSON-serializable value using canonical key ordering."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_unique_index(frame: pd.DataFrame, name: str) -> None:
    """Raise a clear error when a data matrix index is not unique."""
    duplicates = frame.index[frame.index.duplicated()].unique().tolist()
    if duplicates:
        preview = duplicates[:5]
        raise ValueError(f"{name} contains duplicate row identifiers: {preview}")


def write_json(path: str | Path, value: Any) -> None:
    """Write pretty, deterministic JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
