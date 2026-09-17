from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data import load_dataset, write_data_quality_outputs
from src.protocols import leakage_audit, make_protocols, protocol_summary
from src.reproducibility import ensure_directories, load_yaml, project_root, write_json


def load_context(config_path: str, profile_name: str) -> dict[str, Any]:
    root = project_root()
    ensure_directories(root)
    config = load_yaml(root / config_path)
    if profile_name not in config["profiles"]:
        raise ValueError(f"Unknown profile {profile_name!r}")
    profile = config["profiles"][profile_name]
    seed = int(config["random_seed"])
    np.random.seed(seed)

    df, data_manifest = load_dataset(config, root=root, write_manifest=True)
    results_dir = root / "results"
    figures_dir = root / "figures"
    models_dir = root / "artifacts" / "models"
    write_data_quality_outputs(df, results_dir)

    protocols = make_protocols(df, config)
    protocol_summary(df, protocols).to_csv(results_dir / "protocol_summary.csv", index=False)
    audit = leakage_audit(df, protocols, config)
    write_json(results_dir / "leakage_audit.json", audit)

    return {
        "root": root,
        "config": config,
        "profile_name": profile_name,
        "profile": profile,
        "df": df,
        "data_manifest": data_manifest,
        "protocols": protocols,
        "audit": audit,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "models_dir": models_dir,
    }
