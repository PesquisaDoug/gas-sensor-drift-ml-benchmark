from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def timed_predict(estimator, X_input, repeats: int = 10) -> dict[str, float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        estimator.predict(X_input)
        times.append(time.perf_counter() - start)
    median_seconds = float(np.median(times))
    return {
        "inference_ms_total_median": median_seconds * 1000,
        "inference_us_per_sample_median": median_seconds * 1_000_000 / len(X_input),
    }


def bootstrap_classification(y_true, y_pred, iterations=1000, seed=42, alpha=0.05) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collectors = {"macro_f1": [], "balanced_accuracy": [], "mcc": []}
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        collectors["macro_f1"].append(f1_score(yt, yp, average="macro", zero_division=0))
        collectors["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        collectors["mcc"].append(matthews_corrcoef(yt, yp))
    return _bootstrap_rows(collectors, alpha)


def bootstrap_regression(y_true, y_pred, iterations=1000, seed=42, alpha=0.05) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collectors = {"mae": [], "rmse": [], "r2": []}
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        collectors["mae"].append(mean_absolute_error(yt, yp))
        collectors["rmse"].append(math.sqrt(mean_squared_error(yt, yp)))
        collectors["r2"].append(r2_score(yt, yp))
    return _bootstrap_rows(collectors, alpha)


def _bootstrap_rows(collectors: dict[str, list[float]], alpha: float) -> pd.DataFrame:
    rows = []
    for metric, values in collectors.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "bootstrap_mean": float(arr.mean()),
                "ci_lower": float(np.quantile(arr, alpha / 2)),
                "ci_upper": float(np.quantile(arr, 1 - alpha / 2)),
                "valid_samples": len(arr),
            }
        )
    return pd.DataFrame(rows)
