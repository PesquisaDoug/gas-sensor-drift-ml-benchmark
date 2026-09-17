from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.data import FEATURE_COLUMNS
from src.preprocessing import get_feature_matrix


def test_classifier_and_regressor_inference_contract():
    frame = pd.DataFrame([{feature: 0.0 for feature in FEATURE_COLUMNS} for _ in range(4)])
    X = get_feature_matrix(frame)
    classifier = DummyClassifier(strategy="constant", constant=0).fit(X, [0, 1, 2, 3])
    regressor = DummyRegressor(strategy="median").fit(X, [1.0, 2.0, 3.0, 4.0])
    gas_pred = classifier.predict(X)
    concentration_pred = regressor.predict(X)
    assert set(gas_pred).issubset(set(range(6)))
    assert concentration_pred.dtype.kind in {"f", "i"}
