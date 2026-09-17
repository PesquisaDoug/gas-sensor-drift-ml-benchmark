from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS


def prepare_cluster_space(df: pd.DataFrame, seed: int, cluster_sample_size: int):
    cluster_n = min(cluster_sample_size, len(df))
    cluster_sample = df.sample(n=cluster_n, random_state=seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_sample[FEATURE_COLUMNS])
    pca = PCA(n_components=0.95, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    pca_2d = PCA(n_components=2, random_state=seed)
    X_2d = pca_2d.fit_transform(X_scaled)
    return cluster_sample, scaler, pca, X_pca, X_2d


def kmeans_sensitivity(
    X_cluster_pca,
    y_cluster_gas,
    y_cluster_batch,
    silhouette_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, KMeans, np.ndarray]:
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(X_cluster_pca)
        rows.append(
            {
                "algorithm": "KMeans",
                "configuration": f"k={k}",
                "n_clusters": k,
                "noise_fraction": 0.0,
                "silhouette": silhouette_score(
                    X_cluster_pca,
                    labels,
                    sample_size=min(silhouette_sample_size, len(labels)),
                    random_state=seed,
                ),
                "davies_bouldin": davies_bouldin_score(X_cluster_pca, labels),
                "calinski_harabasz": calinski_harabasz_score(X_cluster_pca, labels),
                "ari_gas": adjusted_rand_score(y_cluster_gas, labels),
                "nmi_gas": normalized_mutual_info_score(y_cluster_gas, labels),
                "ari_batch": adjusted_rand_score(y_cluster_batch, labels),
                "nmi_batch": normalized_mutual_info_score(y_cluster_batch, labels),
            }
        )
    kmeans_k6 = KMeans(n_clusters=6, n_init=20, random_state=seed)
    labels_k6 = kmeans_k6.fit_predict(X_cluster_pca)
    return pd.DataFrame(rows), kmeans_k6, labels_k6


def evaluate_dbscan(
    X_space,
    labels,
    gas_external,
    batch_external,
    silhouette_sample_size: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))
    noise_fraction = float(np.mean(labels == -1))
    non_noise_n = int(non_noise.sum())
    row = {
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "non_noise_n": non_noise_n,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "ari_gas": adjusted_rand_score(gas_external, labels),
        "nmi_gas": normalized_mutual_info_score(gas_external, labels),
        "ari_batch": adjusted_rand_score(batch_external, labels),
        "nmi_batch": normalized_mutual_info_score(batch_external, labels),
    }
    if n_clusters >= 2 and non_noise_n >= 100:
        X_valid = X_space[non_noise]
        labels_valid = labels[non_noise]
        if len(np.unique(labels_valid)) >= 2:
            row["silhouette"] = silhouette_score(
                X_valid,
                labels_valid,
                sample_size=min(silhouette_sample_size, len(labels_valid)),
                random_state=seed,
            )
            row["davies_bouldin"] = davies_bouldin_score(X_valid, labels_valid)
            row["calinski_harabasz"] = calinski_harabasz_score(X_valid, labels_valid)
    return row


def dbscan_parameter_search(
    X_cluster_pca,
    y_cluster_gas,
    y_cluster_batch,
    silhouette_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, DBSCAN, np.ndarray]:
    rows = []
    for min_samples in [5, 10, 20]:
        neighbors = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
        neighbors.fit(X_cluster_pca)
        distances, _ = neighbors.kneighbors(X_cluster_pca)
        kth = np.sort(distances[:, -1])
        for quantile in [0.80, 0.85, 0.90, 0.95]:
            eps = float(np.quantile(kth, quantile))
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(X_cluster_pca)
            row = evaluate_dbscan(
                X_cluster_pca,
                labels,
                y_cluster_gas,
                y_cluster_batch,
                silhouette_sample_size,
                seed,
            )
            row.update(
                {
                    "algorithm": "DBSCAN",
                    "configuration": f"eps={eps:.6f},min_samples={min_samples}",
                    "eps": eps,
                    "eps_quantile": quantile,
                    "min_samples": min_samples,
                }
            )
            rows.append(row)

    metrics = pd.DataFrame(rows)
    valid = metrics.dropna(subset=["silhouette"]).copy()
    if valid.empty:
        selected = metrics.sort_values(["n_clusters", "noise_fraction"], ascending=[False, True]).iloc[0]
    else:
        selected = valid.sort_values(["silhouette", "noise_fraction"], ascending=[False, True]).iloc[0]

    selected_dbscan = DBSCAN(
        eps=float(selected["eps"]),
        min_samples=int(selected["min_samples"]),
        n_jobs=-1,
    )
    selected_labels = selected_dbscan.fit_predict(X_cluster_pca)
    return metrics, selected, selected_dbscan, selected_labels


def clustering_summary(kmeans_metrics: pd.DataFrame, selected_dbscan_row: pd.Series) -> pd.DataFrame:
    common_columns = [
        "algorithm",
        "configuration",
        "n_clusters",
        "noise_fraction",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari_gas",
        "nmi_gas",
        "ari_batch",
        "nmi_batch",
    ]
    kmeans_primary = kmeans_metrics.loc[kmeans_metrics["configuration"] == "k=6"].copy()
    dbscan_primary = selected_dbscan_row.to_frame().T.copy()
    return pd.concat(
        [kmeans_primary[common_columns], dbscan_primary[common_columns]],
        ignore_index=True,
    )
