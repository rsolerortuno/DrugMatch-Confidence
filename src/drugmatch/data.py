"""Data download, manifest creation and schema-tolerant loading."""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from drugmatch.utils import sha256_file, write_json


@dataclass(frozen=True)
class DownloadSpec:
    source: str
    release: str
    name: str
    url: str
    relative_path: str


def stream_download(url: str, destination: str | Path, force: bool = False) -> Path:
    """Download a file atomically using the standard library."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return output
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return output


def download_specs(specs: Iterable[DownloadSpec], root: str | Path, force: bool = False) -> list[dict]:
    """Download a collection and return a reproducible manifest."""
    root_path = Path(root)
    records: list[dict] = []
    for spec in specs:
        path = stream_download(spec.url, root_path / spec.relative_path, force=force)
        records.append(
            {
                "source": spec.source,
                "release": spec.release,
                "name": spec.name,
                "url": spec.url,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_manifest(records: list[dict], path: str | Path) -> None:
    """Write a manifest as JSON."""
    write_json(path, records)


def parse_depmap_download_manifest(path: str | Path, release: str) -> dict[str, str]:
    """Parse the CSV returned by the DepMap bulk-download manifest endpoint."""
    frame = pd.read_csv(path)
    lower = {column.lower(): column for column in frame.columns}
    filename_col = next((lower[k] for k in lower if "file" in k and "name" in k), None)
    release_col = next((lower[k] for k in lower if "release" in k), None)
    url_col = next((lower[k] for k in lower if "url" in k), None)
    if not filename_col or not url_col:
        raise ValueError(f"Could not identify filename and URL columns in {path}")
    selected = frame.copy()
    if release_col:
        selected = selected[selected[release_col].astype(str).eq(release)]
    return dict(zip(selected[filename_col].astype(str), selected[url_col].astype(str), strict=False))


def _truthy_mask(series: pd.Series) -> pd.Series:
    """Normalize common boolean encodings used by DepMap releases."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"yes", "true", "1", "y", "t"})
    )


def load_prism_response(path: str | Path) -> pd.DataFrame:
    """Load and minimally standardize PRISM secondary-screen curve parameters."""
    frame = pd.read_csv(path, low_memory=False)
    if {"model_id", "drug", "auc"}.issubset(frame.columns):
        normalized = frame.dropna(subset=["model_id", "drug", "auc"]).copy()
        normalized["model_id"] = normalized["model_id"].astype(str)
        normalized["drug"] = normalized["drug"].astype(str).str.strip().str.lower()
        return normalized.groupby(["model_id", "drug"], as_index=False)["auc"].median()
    required = {"depmap_id", "name", "auc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Response file must contain either model_id/drug/auc or PRISM depmap_id/name/auc; "
            f"missing PRISM columns: {sorted(missing)}"
        )
    if "passed_str_profiling" in frame:
        frame = frame[_truthy_mask(frame["passed_str_profiling"])]
    frame = frame[frame["depmap_id"].notna() & frame["name"].notna() & frame["auc"].notna()].copy()
    frame["depmap_id"] = frame["depmap_id"].astype(str)
    frame["drug"] = frame["name"].astype(str).str.strip().str.lower()
    aggregation = {
        "auc": "median",
        "r2": "median",
        "moa": "first",
        "target": "first",
        "phase": "first",
    }
    aggregation = {key: value for key, value in aggregation.items() if key in frame.columns}
    return frame.groupby(["depmap_id", "drug"], as_index=False).agg(aggregation)


def load_model_metadata(path: str | Path) -> pd.DataFrame:
    """Load model metadata from current DepMap or legacy PRISM schemas."""
    frame = pd.read_csv(path, low_memory=False)
    id_col = next((c for c in ["ModelID", "model_id", "depmap_id", "DepMap_ID", "row_name"] if c in frame), None)
    if id_col is None:
        raise ValueError("Could not identify a model identifier column")
    frame = frame.rename(columns={id_col: "model_id"})
    frame["model_id"] = frame["model_id"].astype(str)
    lineage_col = next(
        (c for c in ["OncotreeLineage", "primary_tissue", "lineage", "PrimaryDisease"] if c in frame),
        None,
    )
    if lineage_col and lineage_col != "lineage":
        frame = frame.rename(columns={lineage_col: "lineage"})
    elif lineage_col is None:
        frame["lineage"] = "unknown"
    frame["lineage"] = frame["lineage"].fillna("unknown").astype(str).str.strip().str.lower()
    return frame.drop_duplicates("model_id").set_index("model_id", drop=True)


def load_wide_omics(
    path: str | Path,
    model_ids: Sequence[str] | pd.Index | None = None,
    dtype: str = "float32",
) -> pd.DataFrame:
    """Load a wide model-level omics matrix with release-tolerant metadata handling.

    ZIP files containing one CSV are supported directly by pandas. DepMap's Yes/No
    default-profile flags are normalized before duplicate model profiles are removed.
    Numerical values are stored as float32 to keep the real 26Q1 workflow practical on
    ordinary workstations.
    """
    frame = pd.read_csv(path, low_memory=False)
    id_col = next(
        (c for c in ["ModelID", "model_id", "DepMap_ID", "row_name", "Unnamed: 0", "ProfileID"] if c in frame),
        None,
    )
    if id_col is None:
        id_col = frame.columns[0]
    frame = frame.rename(columns={id_col: "model_id"})
    if "IsDefaultEntryForModel" in frame.columns:
        default_mask = _truthy_mask(frame["IsDefaultEntryForModel"])
        if default_mask.any():
            frame = frame[default_mask]
    elif "is_default_entry" in frame.columns:
        default_mask = _truthy_mask(frame["is_default_entry"])
        if default_mask.any():
            frame = frame[default_mask]
    frame["model_id"] = frame["model_id"].astype(str)
    if model_ids is not None:
        requested = set(map(str, model_ids))
        frame = frame[frame["model_id"].isin(requested)]
    metadata_columns = {
        "Unnamed: 0",
        "ProfileID",
        "SequencingID",
        "ModelConditionID",
        "IsDefaultEntryForModel",
        "IsDefaultEntryForMC",
        "is_default_entry",
        "SourceModelCondition",
        "DataType",
    }
    drop_cols = [column for column in frame.columns if column in metadata_columns]
    frame = frame.drop(columns=drop_cols, errors="ignore")
    frame = frame.drop_duplicates("model_id").set_index("model_id")
    if frame.empty:
        raise ValueError(f"No model-level rows were loaded from {path}")
    object_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.astype(dtype, copy=False)


def combine_mutation_matrices(
    damaging: pd.DataFrame | None,
    hotspot: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Combine loss-of-function and activating-hotspot calls without conflating them."""
    matrices: list[pd.DataFrame] = []
    if damaging is not None:
        damaging = damaging.copy()
        damaging.columns = [f"damaging::{column}" for column in damaging.columns]
        matrices.append(damaging)
    if hotspot is not None:
        hotspot = hotspot.copy()
        hotspot.columns = [f"hotspot::{column}" for column in hotspot.columns]
        matrices.append(hotspot)
    if not matrices:
        return None
    common = matrices[0].index
    for matrix in matrices[1:]:
        common = common.intersection(matrix.index)
    return pd.concat([matrix.loc[common] for matrix in matrices], axis=1)
