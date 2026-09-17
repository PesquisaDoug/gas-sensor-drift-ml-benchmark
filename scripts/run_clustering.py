from __future__ import annotations

import argparse

import joblib

from scripts.common import load_context
from src.clustering import (
    clustering_summary,
    dbscan_parameter_search,
    kmeans_sensitivity,
    prepare_cluster_space,
)
from src.visualization import plot_clustering


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    args = parser.parse_args()

    ctx = load_context(args.config, args.profile)
    seed = int(ctx["config"]["random_seed"])
    profile = ctx["profile"]
    sample, scaler, pca, X_pca, X_2d = prepare_cluster_space(
        ctx["df"], seed, int(profile["cluster_sample_size"])
    )
    y_gas = sample["gas_class"].to_numpy()
    y_batch = sample["batch"].to_numpy()

    kmeans_metrics, kmeans_k6, kmeans_labels = kmeans_sensitivity(
        X_pca, y_gas, y_batch, int(profile["silhouette_sample_size"]), seed
    )
    dbscan_metrics, selected_row, dbscan_model, dbscan_labels = dbscan_parameter_search(
        X_pca, y_gas, y_batch, int(profile["silhouette_sample_size"]), seed
    )
    clustering_metrics = clustering_summary(kmeans_metrics, selected_row)

    kmeans_metrics.to_csv(ctx["results_dir"] / "kmeans_sensitivity.csv", index=False)
    dbscan_metrics.to_csv(ctx["results_dir"] / "dbscan_parameter_search.csv", index=False)
    clustering_metrics.to_csv(ctx["results_dir"] / "clustering_metrics.csv", index=False)

    joblib.dump(scaler, ctx["models_dir"] / "clustering_scaler.joblib", compress=3)
    joblib.dump(pca, ctx["models_dir"] / "clustering_pca.joblib", compress=3)
    joblib.dump(kmeans_k6, ctx["models_dir"] / "kmeans_k6.joblib", compress=3)
    joblib.dump(dbscan_model, ctx["models_dir"] / "dbscan.joblib", compress=3)

    plot_clustering(
        X_2d,
        y_gas,
        y_batch,
        kmeans_metrics,
        kmeans_labels,
        selected_row,
        dbscan_labels,
        X_pca,
        ctx["figures_dir"],
    )

    print(f"Clustering sample: {len(sample)}")
    print(f"PCA components retaining >=95% variance: {X_pca.shape[1]}")
    print(f"Selected DBSCAN: {selected_row['configuration']}")


if __name__ == "__main__":
    main()
