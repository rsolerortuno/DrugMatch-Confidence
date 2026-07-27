"""Streamlit demonstration for DrugMatch-Confidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from drugmatch import DrugMatchPredictor

st.set_page_config(page_title="DrugMatch-Confidence", page_icon="🧬", layout="wide")
st.title("DrugMatch-Confidence")
st.caption("Preclinical drug-response prediction with calibrated uncertainty and OOD warnings.")

model_root = Path("models/real/depmap_26q1_prism")
if not model_root.exists():
    model_root = Path("models/demo/synthetic_omics")
models = sorted(model_root.glob("*.joblib"))
if not models:
    st.error(
        "No trained models found. Run `drugmatch train real-release` or `drugmatch train synthetic-demo` first."
    )
    st.stop()

selected_model = st.selectbox("Drug model", models, format_func=lambda path: path.stem)
predictor = DrugMatchPredictor.load(selected_model)
reference = predictor.bundle["training_reference"]
mode = st.radio("Input mode", ["Known demonstration sample", "Upload CSV"])

if mode == "Known demonstration sample":
    model_id = st.selectbox("Model", list(reference.index[:100]))
    input_frame = reference.loc[[model_id]]
else:
    upload = st.file_uploader("Upload one-row CSV", type=["csv"])
    if upload is None:
        st.info(
            "The CSV may contain a subset of the expected features; absent numerical features are imputed."
        )
        st.stop()
    input_frame = pd.read_csv(upload)

if st.button("Predict", type="primary"):
    result = predictor.predict(input_frame)
    left, middle, right = st.columns(3)
    left.metric("Predicted class", result.predicted_class)
    middle.metric("Sensitivity probability", f"{result.sensitivity_probability:.1%}")
    right.metric("Confidence", result.confidence)
    st.write(
        f"Continuous-response zone: **{result.response_zone}**; model agreement: **{result.model_agreement}**"
    )
    st.write(
        f"Predicted AUC: **{result.predicted_auc:.3f}** "
        f"({result.interval_lower:.3f} to {result.interval_upper:.3f})"
    )
    st.write(f"Model validation status: **{result.model_status}**")
    st.caption(result.evidence_summary)
    st.write(f"OOD status: **{result.ood_status}**")
    st.write(
        f"Input feature coverage: **{result.feature_coverage:.1%}** "
        f"({result.missing_feature_count} expected numerical features missing)"
    )
    drivers = pd.DataFrame(result.top_drivers)
    if not drivers.empty:
        st.subheader("Top local drivers")
        st.dataframe(drivers, use_container_width=True)
    st.warning(result.disclaimer)
