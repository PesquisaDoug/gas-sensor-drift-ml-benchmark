from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split

from src.data import FEATURE_COLUMNS, TARGET_METADATA_COLUMNS


def make_protocols(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    all_indices = np.arange(len(df))
    random_cfg = config["protocols"]["random"]
    random_train_idx, random_test_idx = train_test_split(
        all_indices,
        test_size=float(random_cfg["test_size"]),
        stratify=df[random_cfg["stratify"]],
        random_state=int(random_cfg["random_state"]),
    )

    temporal_cfg = config["protocols"]["temporal"]
    temporal_train_idx = df.index[df["batch"].isin(temporal_cfg["train_batches"])].to_numpy()
    temporal_test_idx = df.index[df["batch"].isin(temporal_cfg["test_batches"])].to_numpy()

    return {
        "random": {"train_idx": random_train_idx, "test_idx": random_test_idx},
        "temporal": {"train_idx": temporal_train_idx, "test_idx": temporal_test_idx},
    }


def protocol_summary(df: pd.DataFrame, protocols: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for name, split in protocols.items():
        train = df.loc[split["train_idx"]]
        test = df.loc[split["test_idx"]]
        rows.append(
            {
                "protocol": name,
                "train_n": len(train),
                "test_n": len(test),
                "train_batches": ",".join(map(str, sorted(train["batch"].unique()))),
                "test_batches": ",".join(map(str, sorted(test["batch"].unique()))),
                "train_gases": train["gas"].nunique(),
                "test_gases": test["gas"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(
    df: pd.DataFrame,
    protocols: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    random_overlap = set(protocols["random"]["train_idx"]).intersection(protocols["random"]["test_idx"])
    temporal_train = df.loc[protocols["temporal"]["train_idx"], "batch"]
    temporal_test = df.loc[protocols["temporal"]["test_idx"], "batch"]
    temporal_train_batches = set(map(int, temporal_train.unique()))
    temporal_test_batches = set(map(int, temporal_test.unique()))
    expected_train = set(map(int, config["protocols"]["temporal"]["train_batches"]))
    expected_test = set(map(int, config["protocols"]["temporal"]["test_batches"]))
    leaked_features = TARGET_METADATA_COLUMNS.intersection(FEATURE_COLUMNS)

    audit = {
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns_valid": len(FEATURE_COLUMNS) == 128 and not leaked_features,
        "excluded_metadata": sorted(TARGET_METADATA_COLUMNS),
        "features_containing_targets_or_metadata": sorted(leaked_features),
        "random_overlap_n": len(random_overlap),
        "temporal_train_batches": sorted(temporal_train_batches),
        "temporal_test_batches": sorted(temporal_test_batches),
        "temporal_train_batches_match_expected": temporal_train_batches == expected_train,
        "temporal_test_batches_match_expected": temporal_test_batches == expected_test,
        "temporal_batch_overlap_n": len(temporal_train_batches.intersection(temporal_test_batches)),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if not audit["feature_columns_valid"]:
        raise ValueError("Feature leakage audit failed")
    if audit["random_overlap_n"] != 0:
        raise ValueError("Random train/test indices overlap")
    if audit["temporal_batch_overlap_n"] != 0:
        raise ValueError("Temporal train/test batches overlap")
    if not audit["temporal_train_batches_match_expected"] or not audit["temporal_test_batches_match_expected"]:
        raise ValueError("Temporal split batches do not match the protocol")
    return audit


def cv_for_protocol(
    task: str,
    protocol_name: str,
    df: pd.DataFrame,
    train_indices: np.ndarray,
    folds: int,
    seed: int,
):
    if protocol_name == "random":
        if task == "classification":
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), None
        return KFold(n_splits=folds, shuffle=True, random_state=seed), None

    groups = df.loc[train_indices, "batch"].to_numpy()
    return GroupKFold(n_splits=min(folds, len(np.unique(groups)))), groups
