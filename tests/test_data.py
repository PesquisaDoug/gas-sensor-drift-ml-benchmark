from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import FEATURE_COLUMNS, GAS_MAP, parse_batch_file, validate_dataset, validate_feature_columns


def test_feature_columns_exclude_targets_and_metadata():
    validate_feature_columns(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 128
    assert "gas_class" not in FEATURE_COLUMNS
    assert "batch" not in FEATURE_COLUMNS


def test_parse_batch_file_preserves_metadata(tmp_path: Path):
    feature_tokens = " ".join(f"{i}:{float(i)}" for i in range(1, 129))
    path = tmp_path / "batch1.dat"
    path.write_text(f"1;10 {feature_tokens}\n", encoding="utf-8")
    frame = parse_batch_file(path, batch_id=1)
    assert frame.shape == (1, 133)
    assert frame.loc[0, "gas_class_original"] == 1
    assert frame.loc[0, "gas_class"] == 0
    assert frame.loc[0, "gas"] == GAS_MAP[1]
    assert frame.loc[0, "concentration_ppmv"] == 10.0
    assert frame.loc[0, "feature_128"] == 128.0


def test_validate_dataset_accepts_expected_shape():
    rows = []
    for original_class, gas in GAS_MAP.items():
        row = {
            "batch": 1,
            "gas_class_original": original_class,
            "gas_class": original_class - 1,
            "gas": gas,
            "concentration_ppmv": 10.0,
        }
        row.update({feature: 1.0 for feature in FEATURE_COLUMNS})
        rows.append(row)
    df = pd.DataFrame(rows)
    config = {"dataset": {"expected_rows": 6, "expected_batches": 1}}
    validate_dataset(df, config)
