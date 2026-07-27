"""Command-line interface for DrugMatch-Confidence."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import typer

from drugmatch.api import DrugMatchPredictor
from drugmatch.audit import audit_drugs
from drugmatch.baselines import evaluate_baselines
from drugmatch.config import load_yaml
from drugmatch.data import (
    DownloadSpec,
    download_specs,
    load_model_metadata,
    load_prism_response,
    parse_depmap_download_manifest,
    write_manifest,
)
from drugmatch.explain import local_explanation
from drugmatch.external import map_gdsc_models, standardize_external_response, validate_external
from drugmatch.reporting import generate_internal_figures
from drugmatch.robustness import feature_stability, leave_one_lineage_out, modality_ablation
from drugmatch.training import train_drug_bundle
from drugmatch.utils import write_json
from drugmatch.workflows import PREFERRED_DRUGS, load_aligned_features, run_prism_lineage_demo, run_synthetic_training

app = typer.Typer(help="DrugMatch-Confidence: trustworthy preclinical drug-response prediction.")
data_app = typer.Typer(help="Download, audit and prepare public data.")
train_app = typer.Typer(help="Train reproducible models.")
evaluate_app = typer.Typer(help="Evaluate trained models.")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")


def _project_root(root: Path | None) -> Path:
    return (root or Path.cwd()).resolve()


def _load_features_from_options(
    expression: Path,
    metadata: Path,
    mutations: Path | None,
    hotspot_mutations: Path | None,
    copy_number: Path | None,
    signatures: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_aligned_features(
        expression,
        metadata,
        mutation_path=mutations,
        copy_number_path=copy_number,
        hotspot_mutation_path=hotspot_mutations,
        signatures_path=signatures,
    )


@app.command("show-config")
def show_config(
    path: Path = typer.Option(Path("configs/project.yaml"), exists=True, readable=True),
) -> None:
    """Print a resolved YAML configuration as JSON."""
    typer.echo(json.dumps(load_yaml(path), indent=2, sort_keys=True))


@data_app.command("download")
def data_download(
    source: str = typer.Option(..., help="prism or depmap"),
    root: Path = typer.Option(Path.cwd(), help="Repository root."),
    manifest_csv: Path | None = typer.Option(None, help="Local DepMap bulk-download manifest."),
    release: str = typer.Option("DepMap Public 26Q1"),
    force: bool = typer.Option(False),
) -> None:
    """Download PRISM directly or DepMap files from a local portal manifest."""
    root = _project_root(root)
    if source.lower() == "prism":
        specs = [
            DownloadSpec("PRISM", "Secondary Screen", "response", "https://ndownloader.figshare.com/files/20237739", "data/raw/prism/secondary_curve_parameters.csv"),
            DownloadSpec("PRISM", "Secondary Screen", "cell_line_info", "https://ndownloader.figshare.com/files/20237769", "data/raw/prism/secondary_cell_line_info.csv"),
            DownloadSpec("PRISM", "Secondary Screen", "treatment_info", "https://ndownloader.figshare.com/files/20237763", "data/raw/prism/secondary_treatment_info.csv"),
        ]
    elif source.lower() == "depmap":
        if manifest_csv is None:
            raise typer.BadParameter("DepMap requires a locally saved bulk-download manifest.")
        urls = parse_depmap_download_manifest(manifest_csv, release)
        required = [
            "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
            "OmicsSomaticMutationsMatrixDamaging.csv",
            "OmicsSomaticMutationsMatrixHotspot.csv",
            "PortalOmicsCNGeneLog2.csv",
            "OmicsGlobalSignatures.csv",
            "Model.csv",
        ]
        missing = [name for name in required if name not in urls]
        if missing:
            raise typer.BadParameter(f"Manifest does not contain required files: {missing}")
        specs = [
            DownloadSpec("DepMap", release, name, urls[name], f"data/raw/depmap/{name}")
            for name in required
        ]
    else:
        raise typer.BadParameter("source must be 'prism' or 'depmap'")
    records = download_specs(specs, root, force=force)
    manifest_path = root / "data" / "raw" / f"{source.lower()}_manifest.json"
    write_manifest(records, manifest_path)
    typer.echo(f"Downloaded {len(records)} files; manifest: {manifest_path}")


@data_app.command("build")
def data_build(
    expression: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    mutations: Path | None = typer.Option(None, exists=True, readable=True),
    hotspot_mutations: Path | None = typer.Option(None, exists=True, readable=True),
    copy_number: Path | None = typer.Option(None, exists=True, readable=True),
    signatures: Path | None = typer.Option(None, exists=True, readable=True),
    output: Path = typer.Option(Path("data/processed/full_omics")),
) -> None:
    """Align public molecular tables and write a stable processed feature matrix."""
    features, model_metadata = _load_features_from_options(
        expression, metadata, mutations, hotspot_mutations, copy_number, signatures
    )
    output.mkdir(parents=True, exist_ok=True)
    features.to_pickle(output / "features.pkl")
    model_metadata.to_csv(output / "metadata.csv", index_label="model_id")
    write_json(
        output / "build_summary.json",
        {
            "n_models": len(features),
            "n_features": features.shape[1],
            "feature_groups": {
                "expression": sum(column.startswith("expr::") for column in features),
                "mutations": sum(column.startswith("mut::") for column in features),
                "copy_number": sum(column.startswith("cn::") for column in features),
                "signatures": sum(column.startswith("sig::") for column in features),
                "lineage": sum(column == "meta::lineage" for column in features),
            },
        },
    )
    typer.echo(f"Wrote {features.shape[0]} models and {features.shape[1]} features to {output}")


@data_app.command("audit")
def data_audit(
    response: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("reports/data_audit/drug_audit.csv")),
) -> None:
    """Audit PRISM drug coverage without training a model."""
    response_frame = load_prism_response(response).rename(columns={"depmap_id": "model_id"})
    metadata_frame = load_model_metadata(metadata)
    report = audit_drugs(response_frame, metadata_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    typer.echo(f"Wrote {len(report)} audited drugs to {output}")


@train_app.command("synthetic-demo")
def train_synthetic_demo(
    root: Path = typer.Option(Path.cwd(), help="Repository root."),
    seed: int = typer.Option(42),
) -> None:
    outcomes = run_synthetic_training(_project_root(root), seed=seed)
    typer.echo(json.dumps([asdict(item) for item in outcomes], indent=2))


@train_app.command("prism-lineage-demo")
def train_prism_demo(
    response: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    root: Path = typer.Option(Path.cwd(), help="Repository root."),
    seed: int = typer.Option(42),
) -> None:
    outcomes = run_prism_lineage_demo(_project_root(root), response, metadata, seed=seed)
    typer.echo(json.dumps([asdict(item) for item in outcomes], indent=2))


@train_app.command("full-omics")
def train_full_omics(
    drug: str = typer.Option(...),
    response: Path = typer.Option(..., exists=True, readable=True),
    expression: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    mutations: Path | None = typer.Option(None, exists=True, readable=True),
    hotspot_mutations: Path | None = typer.Option(None, exists=True, readable=True),
    copy_number: Path | None = typer.Option(None, exists=True, readable=True),
    signatures: Path | None = typer.Option(None, exists=True, readable=True),
    output: Path = typer.Option(Path("models/full")),
    data_release: str = typer.Option("User-supplied public release"),
    seed: int = typer.Option(42),
    tune: bool = typer.Option(True),
) -> None:
    """Train one full molecular model from locally available public files."""
    features, model_metadata = _load_features_from_options(
        expression, metadata, mutations, hotspot_mutations, copy_number, signatures
    )
    response_frame = load_prism_response(response).rename(columns={"depmap_id": "model_id"})
    outcome = train_drug_bundle(
        drug,
        features,
        response_frame,
        model_metadata,
        output,
        data_release=data_release,
        seed=seed,
        tune=tune,
    )
    typer.echo(json.dumps(asdict(outcome), indent=2))


@train_app.command("real-release")
def train_real_release(
    response: Path = typer.Option(..., exists=True, readable=True),
    expression: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    mutations: Path = typer.Option(..., exists=True, readable=True),
    hotspot_mutations: Path = typer.Option(..., exists=True, readable=True),
    copy_number: Path = typer.Option(..., exists=True, readable=True),
    signatures: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("models/real/depmap_26q1_prism")),
    report_figures: Path = typer.Option(Path("reports/figures")),
    data_release: str = typer.Option("DepMap Public 26Q1 + PRISM Secondary Screen"),
    seed: int = typer.Option(42),
) -> None:
    """Load real data once, train all five locked MVP drugs and generate figures."""
    features, model_metadata = _load_features_from_options(
        expression, metadata, mutations, hotspot_mutations, copy_number, signatures
    )
    response_frame = load_prism_response(response).rename(columns={"depmap_id": "model_id"})
    outcomes = []
    for drug in PREFERRED_DRUGS:
        outcomes.append(
            train_drug_bundle(
                drug,
                features,
                response_frame,
                model_metadata,
                output,
                data_release=data_release,
                seed=seed,
                tune=True,
            )
        )
    generate_internal_figures(output, report_figures, PREFERRED_DRUGS)
    typer.echo(json.dumps([asdict(item) for item in outcomes], indent=2))


@app.command("predict")
def predict(
    model: Path = typer.Option(..., exists=True, readable=True),
    features: Path = typer.Option(..., exists=True, readable=True),
    output: Path | None = typer.Option(None),
) -> None:
    predictor = DrugMatchPredictor.load(model)
    frame = pd.read_csv(features)
    if "model_id" in frame.columns:
        frame = frame.set_index("model_id")
    result = predictor.predict(frame.iloc[[0]])
    payload = asdict(result)
    text = json.dumps(payload, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote prediction to {output}")
    else:
        typer.echo(text)


@app.command("app")
def launch_app(script: Path = typer.Option(Path("app/streamlit_app.py"), exists=True)) -> None:
    try:
        subprocess.run(["streamlit", "run", str(script)], check=True)
    except FileNotFoundError as error:
        raise typer.BadParameter("Install the app extra: pip install -e '.[app]'") from error


@evaluate_app.command("bundle")
def evaluate_bundle(model: Path = typer.Option(..., exists=True, readable=True)) -> None:
    predictor = DrugMatchPredictor.load(model)
    typer.echo(json.dumps(predictor.bundle.get("metrics", {}), indent=2))


@evaluate_app.command("external")
def evaluate_external_command(
    model: Path = typer.Option(..., exists=True, readable=True),
    features: Path = typer.Option(..., exists=True, readable=True),
    response: Path = typer.Option(..., exists=True, readable=True),
    drug: str = typer.Option(...),
    depmap_metadata: Path | None = typer.Option(None, exists=True, readable=True),
    output: Path = typer.Option(Path("reports/external_validation")),
) -> None:
    """Evaluate a frozen bundle on mapped GDSC2 data."""
    predictor = DrugMatchPredictor.load(model)
    if features.suffix == ".pkl":
        feature_frame = pd.read_pickle(features)
    elif features.suffix == ".parquet":
        try:
            feature_frame = pd.read_parquet(features)
        except ImportError as error:
            raise typer.BadParameter(
                "Reading Parquet requires an optional parquet engine; use the documented features.pkl file instead."
            ) from error
    else:
        feature_frame = pd.read_csv(features, index_col=0)
    external_frame = pd.read_excel(response) if response.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(response)
    if "model_id" not in external_frame and depmap_metadata is not None:
        external_frame, _ = map_gdsc_models(external_frame, load_model_metadata(depmap_metadata))
    standardized = standardize_external_response(external_frame, drug)
    metrics, predictions = validate_external(predictor, feature_frame, standardized)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / f"{drug}_predictions.csv", index=False)
    write_json(output / f"{drug}_metrics.json", metrics)
    typer.echo(json.dumps(metrics, indent=2))


@evaluate_app.command("lineage-holdout")
def evaluate_lineage_holdout_command(
    drug: str = typer.Option(...),
    response: Path = typer.Option(..., exists=True, readable=True),
    expression: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    mutations: Path | None = typer.Option(None, exists=True, readable=True),
    hotspot_mutations: Path | None = typer.Option(None, exists=True, readable=True),
    copy_number: Path | None = typer.Option(None, exists=True, readable=True),
    signatures: Path | None = typer.Option(None, exists=True, readable=True),
    output: Path = typer.Option(Path("reports/internal_validation/lineage_holdout.csv")),
) -> None:
    features, model_metadata = _load_features_from_options(
        expression, metadata, mutations, hotspot_mutations, copy_number, signatures
    )
    response_frame = load_prism_response(response).rename(columns={"depmap_id": "model_id"})
    result = leave_one_lineage_out(features, response_frame, model_metadata, drug)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    typer.echo(f"Wrote {len(result)} lineage-holdout results to {output}")


@evaluate_app.command("ablation")
def evaluate_ablation_command(
    drug: str = typer.Option(...),
    response: Path = typer.Option(..., exists=True, readable=True),
    expression: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    mutations: Path | None = typer.Option(None, exists=True, readable=True),
    hotspot_mutations: Path | None = typer.Option(None, exists=True, readable=True),
    copy_number: Path | None = typer.Option(None, exists=True, readable=True),
    signatures: Path | None = typer.Option(None, exists=True, readable=True),
    output: Path = typer.Option(Path("reports/interpretation/modality_ablation.csv")),
) -> None:
    features, model_metadata = _load_features_from_options(
        expression, metadata, mutations, hotspot_mutations, copy_number, signatures
    )
    response_frame = load_prism_response(response).rename(columns={"depmap_id": "model_id"})
    result = modality_ablation(features, response_frame, model_metadata, drug)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    typer.echo(f"Wrote {len(result)} ablations to {output}")


@evaluate_app.command("feature-stability")
def evaluate_feature_stability_command(
    drug: str = typer.Option(...),
    response: Path = typer.Option(..., exists=True, readable=True),
    expression: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    mutations: Path | None = typer.Option(None, exists=True, readable=True),
    hotspot_mutations: Path | None = typer.Option(None, exists=True, readable=True),
    copy_number: Path | None = typer.Option(None, exists=True, readable=True),
    signatures: Path | None = typer.Option(None, exists=True, readable=True),
    output: Path = typer.Option(Path("reports/interpretation/feature_stability.csv")),
) -> None:
    features, _ = _load_features_from_options(
        expression, metadata, mutations, hotspot_mutations, copy_number, signatures
    )
    response_frame = load_prism_response(response).rename(columns={"depmap_id": "model_id"})
    result = feature_stability(features, response_frame, drug)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    typer.echo(f"Wrote {len(result)} feature-stability rows to {output}")


@train_app.command("baseline")
def train_baseline_command(
    drug: str = typer.Option(...),
    response: Path = typer.Option(..., exists=True, readable=True),
    expression: Path = typer.Option(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., exists=True, readable=True),
    mutations: Path | None = typer.Option(None, exists=True, readable=True),
    hotspot_mutations: Path | None = typer.Option(None, exists=True, readable=True),
    copy_number: Path | None = typer.Option(None, exists=True, readable=True),
    signatures: Path | None = typer.Option(None, exists=True, readable=True),
    output: Path = typer.Option(Path("reports/internal_validation/baselines.json")),
    seed: int = typer.Option(42),
) -> None:
    features, model_metadata = _load_features_from_options(
        expression, metadata, mutations, hotspot_mutations, copy_number, signatures
    )
    response_frame = load_prism_response(response).rename(columns={"depmap_id": "model_id"})
    results = evaluate_baselines(features, response_frame, model_metadata, drug, seed=seed)
    write_json(output, results)
    typer.echo(json.dumps(results, indent=2))


@app.command("explain")
def explain_command(
    model: Path = typer.Option(..., exists=True, readable=True),
    features: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("reports/interpretation/local_explanation.csv")),
) -> None:
    predictor = DrugMatchPredictor.load(model)
    frame = pd.read_csv(features)
    if "model_id" in frame.columns:
        frame = frame.set_index("model_id")
    aligned = predictor._frame(frame.iloc[[0]])
    explanation = local_explanation(predictor.bundle["classification"], aligned, top_n=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    explanation.to_csv(output, index=False)
    typer.echo(f"Wrote local explanation to {output}")
