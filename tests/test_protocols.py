from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS
from src.protocols import leakage_audit, make_protocols, protocol_summary


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for batch in range(1, 11):
        for gas_class in range(6):
            row = {
                "batch": batch,
                "gas_class_original": gas_class + 1,
                "gas_class": gas_class,
                "gas": str(gas_class),
                "concentration_ppmv": float(gas_class + 1),
            }
            row.update({feature: float(batch + gas_class) for feature in FEATURE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def config():
    return {
        "random_seed": 42,
        "protocols": {
            "random": {"test_size": 0.2, "stratify": "gas_class", "random_state": 42},
            "temporal": {
                "train_batches": [1, 2, 3, 4, 5],
                "test_batches": [6, 7, 8, 9, 10],
            },
        },
    }


def test_random_and_temporal_protocols_are_disjoint():
    df = synthetic_frame()
    protocols = make_protocols(df, config())
    audit = leakage_audit(df, protocols, config())
    assert audit["random_overlap_n"] == 0
    assert audit["temporal_batch_overlap_n"] == 0
    assert audit["temporal_train_batches"] == [1, 2, 3, 4, 5]
    assert audit["temporal_test_batches"] == [6, 7, 8, 9, 10]
    summary = protocol_summary(df, protocols)
    assert set(summary["protocol"]) == {"random", "temporal"}
