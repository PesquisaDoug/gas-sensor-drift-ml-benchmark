from __future__ import annotations

import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS
from src.preprocessing import get_feature_matrix


def test_feature_order_is_preserved():
    frame = pd.DataFrame([{feature: i for i, feature in enumerate(FEATURE_COLUMNS)}])
    X = get_feature_matrix(frame)
    assert list(X.columns) == FEATURE_COLUMNS


def test_missing_feature_raises_readable_error():
    frame = pd.DataFrame([{feature: 0.0 for feature in FEATURE_COLUMNS[:-1]}])
    with pytest.raises(ValueError, match="Missing required features"):
        get_feature_matrix(frame)
