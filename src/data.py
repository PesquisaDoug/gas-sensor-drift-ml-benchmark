from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.reproducibility import now_utc, project_root, write_json

GAS_MAP: dict[int, str] = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

TARGET_METADATA_COLUMNS = {
    "gas_class_original",
    "gas_class",
    "gas",
    "concentration_ppmv",
    "batch",
}

FEATURE_COLUMNS = [f"feature_{i:03d}" for i in range(1, 129)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)


def extract_dataset(archive_path: str | Path, extract_dir: str | Path) -> None:
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)


def discover_batch_files(extract_dir: str | Path) -> list[Path]:
    files = sorted(
        Path(extract_dir).rglob("batch*.dat"),
        key=lambda p: int(p.stem.replace("batch", "")),
    )
    if len(files) != 10:
        raise ValueError(f"Expected 10 batch files; found {len(files)}")
    return files


def parse_batch_file(path: str | Path, batch_id: int, n_features: int = 128) -> pd.DataFrame:
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            header = parts[0]
            if ";" not in header:
                raise ValueError(f"Unexpected header at {path.name}:{line_number}")

            class_text, concentration_text = header.split(";", 1)
            gas_original = int(class_text)
            concentration = float(concentration_text)

            if gas_original not in GAS_MAP:
                raise ValueError(f"Unknown gas class {gas_original}")

            features = np.full(n_features, np.nan, dtype=float)
            for token in parts[1:]:
                index_text, value_text = token.split(":", 1)
                index = int(index_text) - 1
                if not 0 <= index < n_features:
                    raise ValueError(f"Invalid feature index {index + 1}")
                features[index] = float(value_text)

            if np.isnan(features).any():
                raise ValueError(f"Missing feature value in {path.name}:{line_number}")

            row = {
                "batch": batch_id,
                "gas_class_original": gas_original,
                "gas_class": gas_original - 1,
                "gas": GAS_MAP[gas_original],
                "concentration_ppmv": concentration,
            }
            row.update({f"feature_{i:03d}": value for i, value in enumerate(features, start=1)})
            rows.append(row)

    return pd.DataFrame(rows)


def validate_feature_columns(feature_columns: list[str] = FEATURE_COLUMNS) -> None:
    if len(feature_columns) != 128:
        raise ValueError(f"Expected 128 feature columns; found {len(feature_columns)}")
    leaked = TARGET_METADATA_COLUMNS.intersection(feature_columns)
    if leaked:
        raise ValueError(f"Targets/metadata present in features: {sorted(leaked)}")


def validate_dataset(df: pd.DataFrame, config: dict[str, Any]) -> None:
    validate_feature_columns(FEATURE_COLUMNS)
    expected_rows = int(config["dataset"]["expected_rows"])
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(df)}")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Unexpected missing values in feature matrix")
    if set(df["gas_class_original"].unique()) != set(GAS_MAP):
        raise ValueError("Unexpected gas_class_original values")
    if set(df["gas_class"].unique()) != set(range(6)):
        raise ValueError("Unexpected zero-based gas_class values")
    expected_batches = list(range(1, int(config["dataset"]["expected_batches"]) + 1))
    if sorted(df["batch"].unique()) != expected_batches:
        raise ValueError("Unexpected batch identifiers")


def create_data_manifest(
    df: pd.DataFrame,
    archive_path: Path,
    batch_files: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_name": config["dataset"]["name"],
        "dataset_id": int(config["dataset"]["id"]),
        "uci_dataset_id": int(config["dataset"]["id"]),
        "doi": config["dataset"]["doi"],
        "canonical_source": config["dataset"]["url"],
        "archive_file": str(archive_path.as_posix()),
        "archive_sha256": sha256_file(archive_path),
        "rows": int(len(df)),
        "features": len(FEATURE_COLUMNS),
        "batches": int(df["batch"].nunique()),
        "gas_mapping_original": {str(k): v for k, v in GAS_MAP.items()},
        "gas_mapping_zero_based": {str(k - 1): v for k, v in GAS_MAP.items()},
        "source_batch_files": [
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "rows": int((df["batch"] == int(path.stem.replace("batch", ""))).sum()),
            }
            for path in batch_files
        ],
        "validated_at_utc": now_utc(),
    }


def load_dataset(
    config: dict[str, Any],
    root: Path | None = None,
    write_manifest: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root or project_root()
    archive_path = root / config["dataset"]["archive_path"]
    extract_dir = root / config["dataset"]["extract_dir"]

    if not archive_path.exists():
        download_file(config["dataset"]["url"], archive_path)
    extract_dataset(archive_path, extract_dir)
    batch_files = discover_batch_files(extract_dir)

    frames = []
    for path in batch_files:
        batch_id = int(path.stem.replace("batch", ""))
        frames.append(parse_batch_file(path, batch_id, len(FEATURE_COLUMNS)))
    df = pd.concat(frames, ignore_index=True)
    validate_dataset(df, config)

    manifest = create_data_manifest(df, archive_path, batch_files, config)
    if write_manifest:
        write_json(root / "data" / "data_manifest.json", manifest)
    return df, manifest


def write_data_quality_outputs(df: pd.DataFrame, results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    quality_summary = pd.DataFrame(
        {
            "value": [
                len(df),
                len(FEATURE_COLUMNS),
                int(df[FEATURE_COLUMNS].isna().sum().sum()),
                int(df.duplicated().sum()),
                int(df["gas"].nunique()),
                int(df["batch"].nunique()),
                float(df["concentration_ppmv"].min()),
                float(df["concentration_ppmv"].max()),
            ]
        },
        index=[
            "rows",
            "features",
            "missing_feature_cells",
            "duplicate_rows",
            "gas_classes",
            "batches",
            "min_concentration_ppmv",
            "max_concentration_ppmv",
        ],
    )
    quality_summary.to_csv(results_dir / "data_quality_summary.csv")
    pd.crosstab(df["batch"], df["gas"]).to_csv(results_dir / "batch_gas_counts.csv")
