from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS


def get_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [feature for feature in FEATURE_COLUMNS if feature not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    return frame[FEATURE_COLUMNS].copy()
