from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier

from src.evaluation import (
    bootstrap_classification,
    bootstrap_regression,
    classification_metrics,
    regression_metrics,
    timed_predict,
)


def test_classification_metrics_include_macro_f1():
    metrics = classification_metrics([0, 1, 1], [0, 1, 0])
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_regression_metrics_include_rmse():
    metrics = regression_metrics([1.0, 2.0], [1.0, 4.0])
    assert metrics["rmse"] > 0


def test_bootstrap_helpers_and_timed_predict():
    cls_ci = bootstrap_classification([0, 1, 1], [0, 1, 0], iterations=5, seed=42)
    reg_ci = bootstrap_regression([1.0, 2.0], [1.5, 2.5], iterations=5, seed=42)
    assert set(cls_ci["metric"]) == {"macro_f1", "balanced_accuracy", "mcc"}
    assert set(reg_ci["metric"]) == {"mae", "rmse", "r2"}
    estimator = DummyClassifier(strategy="prior").fit(np.zeros((4, 2)), [0, 0, 1, 1])
    timing = timed_predict(estimator, np.zeros((4, 2)), repeats=2)
    assert timing["inference_us_per_sample_median"] >= 0
