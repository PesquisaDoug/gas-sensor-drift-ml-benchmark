from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS, GAS_MAP


def plot_eda(df: pd.DataFrame, figures_dir: Path, seed: int) -> None:
    gas_counts = df["gas"].value_counts().reindex(GAS_MAP.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(gas_counts.index, gas_counts.values)
    ax.set_ylabel("Measurements")
    ax.set_title("Gas class distribution")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_gas_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = pd.crosstab(df["batch"], df["gas"])
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Gas")
    ax.set_ylabel("Batch")
    ax.set_title("Measurements by gas and acquisition batch")
    fig.colorbar(image, ax=ax, label="Measurements")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_batch_gas_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    gas_order = list(GAS_MAP.values())
    box_data = [df.loc[df["gas"] == gas, "concentration_ppmv"].to_numpy() for gas in gas_order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(box_data, tick_labels=gas_order, showfliers=False)
    ax.set_ylabel("Concentration (ppmv)")
    ax.set_title("Concentration distribution by gas")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "03_concentration_by_gas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    eda_sample = df.sample(n=min(4000, len(df)), random_state=seed)
    eda_scaled = StandardScaler().fit_transform(eda_sample[FEATURE_COLUMNS])
    eda_2d = PCA(n_components=2, random_state=seed).fit_transform(eda_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(eda_2d[:, 0], eda_2d[:, 1], c=eda_sample["batch"], s=10, alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Exploratory PCA colored by acquisition batch")
    fig.colorbar(scatter, ax=ax, label="Batch")
    fig.tight_layout()
    fig.savefig(figures_dir / "04_pca_by_batch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_classification_outputs(metrics_df, drift_df, predictions_df, df, protocols, figures_dir: Path) -> None:
    plot_df = metrics_df.pivot(index="model", columns="protocol", values="macro_f1")
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification: random vs later-batch evaluation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_classification_random_vs_temporal.png", dpi=180, bbox_inches="tight")
    plt.close()

    ordered = drift_df.sort_values("delta_macro_f1_temporal_minus_random")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ordered["model"], ordered["delta_macro_f1_temporal_minus_random"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Temporal Macro-F1 - Random Macro-F1")
    ax.set_title("Classification performance shift")
    fig.tight_layout()
    fig.savefig(figures_dir / "06_classification_drift_delta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    temporal_y_true = df.loc[protocols["temporal"]["test_idx"], "gas_class"]
    for model_name in metrics_df["model"].unique():
        subset = predictions_df.query("protocol == 'temporal' and model == @model_name")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            temporal_y_true,
            subset["gas_class_pred"],
            display_labels=list(GAS_MAP.values()),
            xticks_rotation=45,
            values_format="d",
            ax=ax,
        )
        ax.set_title(f"Temporal confusion matrix - {model_name}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"07_confusion_temporal_{model_name.lower()}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_regression_outputs(metrics_df, drift_df, figures_dir: Path) -> None:
    plot_df = metrics_df.pivot(index="model", columns="protocol", values="rmse")
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("RMSE (ppmv)")
    ax.set_title("Concentration regression: random vs later-batch evaluation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "08_regression_random_vs_temporal.png", dpi=180, bbox_inches="tight")
    plt.close()

    ordered = drift_df.sort_values("delta_rmse_temporal_minus_random", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ordered["model"], ordered["delta_rmse_temporal_minus_random"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Temporal RMSE - Random RMSE (ppmv)")
    ax.set_title("Regression error shift")
    fig.tight_layout()
    fig.savefig(figures_dir / "09_regression_drift_delta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_importance(importance_df, title: str, xlabel: str, filename: str, figures_dir: Path) -> None:
    top = importance_df.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_clustering(
    X_cluster_2d,
    y_cluster_gas,
    y_cluster_batch,
    kmeans_metrics,
    kmeans_k6_labels,
    selected_dbscan_row,
    dbscan_labels,
    X_cluster_pca,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for gas_class, gas_name in enumerate(GAS_MAP.values()):
        mask = y_cluster_gas == gas_class
        ax.scatter(X_cluster_2d[mask, 0], X_cluster_2d[mask, 1], s=9, alpha=0.4, label=gas_name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection colored by gas identity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "12_pca_by_gas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_cluster_2d[:, 0], X_cluster_2d[:, 1], c=y_cluster_batch, s=9, alpha=0.45)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection colored by acquisition batch")
    fig.colorbar(scatter, ax=ax, label="Batch")
    fig.tight_layout()
    fig.savefig(figures_dir / "13_pca_by_batch_clustering_sample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kmeans_metrics["n_clusters"], kmeans_metrics["silhouette"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("k-Means sensitivity analysis")
    ax.set_xticks(range(2, 11))
    fig.tight_layout()
    fig.savefig(figures_dir / "14_kmeans_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_cluster_2d[:, 0], X_cluster_2d[:, 1], c=kmeans_k6_labels, s=9, alpha=0.45)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("k-Means clustering (k=6)")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(figures_dir / "15_pca_kmeans_k6.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_min_samples = int(selected_dbscan_row["min_samples"])
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    neighbors = NearestNeighbors(n_neighbors=selected_min_samples, n_jobs=-1)
    neighbors.fit(X_cluster_pca)
    distances, _ = neighbors.kneighbors(X_cluster_pca)
    kth = np.sort(distances[:, -1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kth)
    ax.axhline(float(selected_dbscan_row["eps"]), linestyle="--", label="selected eps")
    ax.set_xlabel("Sorted observations")
    ax.set_ylabel(f"{selected_min_samples}-NN distance")
    ax.set_title("DBSCAN k-distance diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "16_dbscan_k_distance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_label in sorted(np.unique(dbscan_labels)):
        mask = dbscan_labels == cluster_label
        display_name = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}"
        ax.scatter(X_cluster_2d[mask, 0], X_cluster_2d[mask, 1], s=9, alpha=0.45, label=display_name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Selected DBSCAN clustering")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "17_pca_dbscan.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
