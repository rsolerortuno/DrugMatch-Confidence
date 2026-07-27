"""Leakage-safe feature assembly and training-only feature selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class OmicsTables:
    expression: pd.DataFrame
    mutations: pd.DataFrame | None
    copy_number: pd.DataFrame | None
    metadata: pd.DataFrame
    signatures: pd.DataFrame | None = None


def _prefix(frame: pd.DataFrame | None, prefix: str) -> pd.DataFrame | None:
    if frame is None:
        return None
    output = frame.copy()
    output.columns = [f"{prefix}::{column}" for column in output.columns]
    return output


def assemble_features(
    tables: OmicsTables, model_ids: list[str] | pd.Index | None = None
) -> pd.DataFrame:
    """Join modality matrices without fitting any data-dependent transformation."""
    expression = _prefix(tables.expression, "expr")
    if expression is None:
        raise AssertionError("Expression data unexpectedly resolved to None")
    frames: list[pd.DataFrame] = [expression]
    for frame, prefix in [
        (tables.mutations, "mut"),
        (tables.copy_number, "cn"),
        (tables.signatures, "sig"),
    ]:
        prefixed = _prefix(frame, prefix)
        if prefixed is not None:
            frames.append(prefixed)
    metadata = tables.metadata.copy()
    if metadata.index.name != "model_id" and "model_id" in metadata:
        metadata = metadata.set_index("model_id")
    lineage = metadata[["lineage"]].copy()
    lineage.columns = ["meta::lineage"]
    frames.append(lineage)
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    if model_ids is not None:
        common = common.intersection(pd.Index(model_ids).astype(str))
    if common.empty:
        raise ValueError("No models share all requested feature modalities")
    return pd.concat([frame.loc[common] for frame in frames], axis=1)


DRUG_BIOLOGY_GENES: dict[str, set[str]] = {
    "trametinib": {"BRAF", "KRAS", "NRAS", "NF1", "MAP2K1", "MAP2K2", "DUSP6", "SPRY2", "EGFR"},
    "afatinib": {
        "EGFR",
        "ERBB2",
        "ERBB3",
        "ERBB4",
        "KRAS",
        "NRAS",
        "BRAF",
        "PIK3CA",
        "PTEN",
        "MET",
    },
    "palbociclib": {
        "RB1",
        "CDKN2A",
        "CDKN2B",
        "CCND1",
        "CCND2",
        "CCND3",
        "CDK4",
        "CDK6",
        "E2F1",
        "TP53",
    },
    "olaparib": {
        "BRCA1",
        "BRCA2",
        "PALB2",
        "RAD51C",
        "RAD51D",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "ARID1A",
    },
    "gemcitabine": {"DCK", "CDA", "RRM1", "RRM2", "SLC29A1", "SLC28A1", "CMPK1", "TYMS", "NT5C2"},
}


def _gene_symbol(column: str) -> str:
    raw = column.split("::")[-1]
    return raw.split(" (")[0].strip().upper()


def _top_variance_columns(frame: pd.DataFrame, columns: list[str], limit: int) -> list[str]:
    if not columns or limit <= 0:
        return []
    variances = frame[columns].var(axis=0, skipna=True).fillna(-np.inf)
    return list(variances.nlargest(min(limit, len(columns))).index)


def select_training_features(
    X_train: pd.DataFrame,
    drug: str,
    max_expression_features: int = 300,
    max_mutation_features: int = 120,
    max_copy_number_features: int = 120,
    max_signature_features: int = 20,
) -> tuple[list[str], pd.DataFrame]:
    """Select a compact raw feature set using training rows only.

    Selection is unsupervised (variance) plus a small mechanism-informed gene list.
    No validation or test outcome is used. The returned manifest records why each
    retained column was selected.
    """
    groups = {
        "expression": ([c for c in X_train if c.startswith("expr::")], max_expression_features),
        "mutations": ([c for c in X_train if c.startswith("mut::")], max_mutation_features),
        "copy_number": ([c for c in X_train if c.startswith("cn::")], max_copy_number_features),
        "signatures": ([c for c in X_train if c.startswith("sig::")], max_signature_features),
    }
    forced_genes = DRUG_BIOLOGY_GENES.get(drug.lower(), set())
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for modality, (columns, limit) in groups.items():
        variance_selected = _top_variance_columns(X_train, columns, limit)
        forced = [column for column in columns if _gene_symbol(column) in forced_genes]
        chosen = list(dict.fromkeys([*variance_selected, *forced]))
        selected.extend(chosen)
        variance_set = set(variance_selected)
        forced_set = set(forced)
        variances = X_train[chosen].var(axis=0, skipna=True) if chosen else pd.Series(dtype=float)
        for column in chosen:
            reason = (
                "variance+biology"
                if column in variance_set and column in forced_set
                else ("biology" if column in forced_set else "variance")
            )
            rows.append(
                {
                    "feature": column,
                    "modality": modality,
                    "gene_symbol": _gene_symbol(column),
                    "selection_reason": reason,
                    "training_variance": float(variances.get(column, np.nan)),
                }
            )
    if "meta::lineage" in X_train:
        selected.append("meta::lineage")
        rows.append(
            {
                "feature": "meta::lineage",
                "modality": "lineage",
                "gene_symbol": "",
                "selection_reason": "required_metadata",
                "training_variance": np.nan,
            }
        )
    selected = list(dict.fromkeys(selected))
    return selected, pd.DataFrame(rows)


def build_preprocessor(
    frame: pd.DataFrame,
    max_expression_features: int = 10_000,
    max_mutation_features: int = 10_000,
    max_copy_number_features: int = 10_000,
    max_signature_features: int = 1_000,
) -> ColumnTransformer:
    """Build preprocessing for an already training-selected compact feature matrix."""
    expr = [column for column in frame if column.startswith("expr::")]
    mut = [column for column in frame if column.startswith("mut::")]
    cn = [column for column in frame if column.startswith("cn::")]
    sig = [column for column in frame if column.startswith("sig::")]
    lineage = [column for column in frame if column == "meta::lineage"]

    def numeric_pipeline(max_features: int, scale: bool) -> Pipeline:
        steps: list[tuple[str, object]] = [
            ("impute", SimpleImputer(strategy="median")),
            ("select", TopVarianceFeatures(max_features=max_features)),
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers: list[tuple[str, object, list[str]]] = []
    if expr:
        transformers.append(("expression", numeric_pipeline(max_expression_features, True), expr))
    if mut:
        transformers.append(("mutations", numeric_pipeline(max_mutation_features, False), mut))
    if cn:
        transformers.append(("copy_number", numeric_pipeline(max_copy_number_features, True), cn))
    if sig:
        transformers.append(("signatures", numeric_pipeline(max_signature_features, True), sig))
    if lineage:
        transformers.append(
            (
                "lineage",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                lineage,
            )
        )
    if not transformers:
        raise ValueError("No supported feature columns were found")
    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=True
    )


class TopVarianceFeatures(BaseEstimator, TransformerMixin):
    """Select the top N columns by training-set variance."""

    def __init__(self, max_features: int = 300):
        self.max_features = max_features

    def fit(self, X: np.ndarray, y: object = None) -> TopVarianceFeatures:
        array = np.asarray(X, dtype=float)
        variances = np.nanvar(array, axis=0)
        order = np.argsort(variances)[::-1]
        count = min(self.max_features, array.shape[1])
        self.indices_ = np.sort(order[:count])
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "indices_"):
            raise RuntimeError("TopVarianceFeatures has not been fitted")
        return np.asarray(X)[:, self.indices_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.indices_] = True
        return self.indices_ if indices else support

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if not hasattr(self, "indices_"):
            raise RuntimeError("TopVarianceFeatures has not been fitted")
        if input_features is None:
            names = np.asarray(
                [f"feature_{index}" for index in range(self.n_features_in_)], dtype=object
            )
        else:
            names = np.asarray(input_features, dtype=object)
        return names[self.indices_]
