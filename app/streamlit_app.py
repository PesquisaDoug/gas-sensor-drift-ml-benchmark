from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "artifacts" / "models"

st.set_page_config(page_title="Gas Sensor Drift Benchmark", layout="wide")
st.title("Gas Sensor Drift - Multi-Task ML Benchmark")
st.caption(
    "Classification, concentration regression and clustering under sensor drift. "
    "The app expects the 128 engineered UCI features and does not acquire live sensor measurements."
)

required = [
    RESULTS / "classification_metrics.csv",
    RESULTS / "regression_metrics.csv",
    RESULTS / "classification_drift_comparison.csv",
    RESULTS / "regression_drift_comparison.csv",
    RESULTS / "clustering_metrics.csv",
    RESULTS / "feature_schema.json",
    RESULTS / "experiment_manifest.json",
    MODELS / "best_classification_model.joblib",
    MODELS / "best_regression_model.joblib",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    st.error("Required artifacts are missing. Run the benchmark first.\n\n" + "\n".join(missing))
    st.stop()

classification_metrics = pd.read_csv(RESULTS / "classification_metrics.csv")
regression_metrics = pd.read_csv(RESULTS / "regression_metrics.csv")
classification_drift = pd.read_csv(RESULTS / "classification_drift_comparison.csv")
regression_drift = pd.read_csv(RESULTS / "regression_drift_comparison.csv")
clustering_metrics = pd.read_csv(RESULTS / "clustering_metrics.csv")
schema = json.loads((RESULTS / "feature_schema.json").read_text(encoding="utf-8"))
manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))

classifier = joblib.load(MODELS / "best_classification_model.joblib")
regressor = joblib.load(MODELS / "best_regression_model.joblib")
feature_names = schema["feature_names"]
class_mapping = {int(k): v for k, v in schema["classification_target"]["mapping"].items()}

tabs = st.tabs(
    [
        "Dataset",
        "Classification",
        "Regression",
        "Drift",
        "Clustering",
        "Inference",
        "Reproducibility",
    ]
)

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Measurements", manifest["dataset"]["rows"])
    c2.metric("Features", manifest["dataset"]["features"])
    c3.metric("Batches", manifest["dataset"]["batches"])
    for filename in [
        "01_gas_distribution.png",
        "02_batch_gas_heatmap.png",
        "03_concentration_by_gas.png",
        "04_pca_by_batch.png",
    ]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[1]:
    st.dataframe(
        classification_metrics.sort_values(["protocol", "macro_f1"], ascending=[True, False]),
        use_container_width=True,
    )
    for filename in ["05_classification_random_vs_temporal.png", "06_classification_drift_delta.png"]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[2]:
    st.dataframe(regression_metrics.sort_values(["protocol", "rmse"]), use_container_width=True)
    for filename in ["08_regression_random_vs_temporal.png", "09_regression_drift_delta.png"]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[3]:
    st.dataframe(classification_drift, use_container_width=True)
    st.dataframe(regression_drift, use_container_width=True)

with tabs[4]:
    st.dataframe(clustering_metrics, use_container_width=True)
    for filename in [
        "12_pca_by_gas.png",
        "13_pca_by_batch_clustering_sample.png",
        "14_kmeans_sensitivity.png",
        "15_pca_kmeans_k6.png",
        "16_dbscan_k_distance.png",
        "17_pca_dbscan.png",
    ]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[5]:
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        missing_features = [feature for feature in feature_names if feature not in frame.columns]
        if missing_features:
            st.error(
                "Missing required features: "
                + ", ".join(missing_features[:20])
                + (" ..." if len(missing_features) > 20 else "")
            )
        else:
            X_upload = frame[feature_names].copy()
            gas_pred = classifier.predict(X_upload)
            concentration_pred = regressor.predict(X_upload)
            output = frame.copy()
            output["gas_class_prediction"] = gas_pred
            output["gas_prediction"] = [class_mapping[int(value)] for value in gas_pred]
            output["concentration_prediction_ppmv"] = concentration_pred
            st.success(f"Processed {len(output)} measurements.")
            st.dataframe(output, use_container_width=True)
            st.download_button(
                "Download predictions",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="gas_sensor_predictions.csv",
                mime="text/csv",
            )

with tabs[6]:
    st.json(manifest)
