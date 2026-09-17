from __future__ import annotations

import numpy as np

from src.clustering import dbscan_parameter_search, kmeans_sensitivity


def test_kmeans_and_dbscan_return_metrics():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 0.2, (30, 3)), rng.normal(3, 0.2, (30, 3))])
    gas = np.array([0] * 30 + [1] * 30)
    batch = np.array([1] * 30 + [2] * 30)
    kmeans_metrics, model, labels = kmeans_sensitivity(X, gas, batch, 50, 42)
    assert "ari_gas" in kmeans_metrics.columns
    assert len(labels) == len(X)
    dbscan_metrics, selected, dbscan, dbscan_labels = dbscan_parameter_search(X, gas, batch, 50, 42)
    assert "nmi_batch" in dbscan_metrics.columns
    assert "silhouette" in selected.index
    assert len(dbscan_labels) == len(X)
