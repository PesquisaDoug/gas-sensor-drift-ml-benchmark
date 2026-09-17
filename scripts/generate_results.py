from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from scripts.common import load_context
from src.data import FEATURE_COLUMNS, GAS_MAP
from src.drift import classification_drift_table, regression_drift_table
from src.evaluation import bootstrap_classification, bootstrap_regression
from src.reproducibility import get_git_commit, now_utc, package_versions, write_json
from src.training import best_model_from_tuning
from src.visualization import (
    plot_classification_outputs,
    plot_eda,
    plot_importance,
    plot_regression_outputs,
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required output is missing: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    args = parser.parse_args()

    ctx = load_context(args.config, args.profile)
    results = ctx["results_dir"]
    models = ctx["models_dir"]
    figures = ctx["figures_dir"]
    df = ctx["df"]
    seed = int(ctx["config"]["random_seed"])
    profile = ctx["profile"]

    classification_metrics = pd.read_csv(_require(results / "classification_metrics.csv"))
    classification_predictions = pd.read_csv(_require(results / "classification_predictions.csv"))
    classification_tuning = pd.read_csv(_require(results / "classification_tuning_summary.csv"))
    regression_metrics = pd.read_csv(_require(results / "regression_metrics.csv"))
    regression_predictions = pd.read_csv(_require(results / "regression_predictions.csv"))
    regression_tuning = pd.read_csv(_require(results / "regression_tuning_summary.csv"))

    classification_drift = classification_drift_table(classification_metrics)
    regression_drift = regression_drift_table(regression_metrics)
    classification_drift.to_csv(results / "classification_drift_comparison.csv", index=False)
    regression_drift.to_csv(results / "regression_drift_comparison.csv", index=False)

    bootstrap_frames = []
    for protocol_name in ["random", "temporal"]:
        for model_name in classification_metrics["model"].unique():
            subset = classification_predictions.loc[
                (classification_predictions["protocol"] == protocol_name)
                & (classification_predictions["model"] == model_name)
            ]
            ci = bootstrap_classification(
                subset["gas_class_true"],
                subset["gas_class_pred"],
                int(profile["bootstrap_iterations"]),
                seed,
            )
            ci.insert(0, "task", "classification")
            ci.insert(1, "protocol", protocol_name)
            ci.insert(2, "model", model_name)
            bootstrap_frames.append(ci)

        for model_name in regression_metrics["model"].unique():
            subset = regression_predictions.loc[
                (regression_predictions["protocol"] == protocol_name)
                & (regression_predictions["model"] == model_name)
            ]
            ci = bootstrap_regression(
                subset["concentration_true"],
                subset["concentration_pred"],
                int(profile["bootstrap_iterations"]),
                seed,
            )
            ci.insert(0, "task", "regression")
            ci.insert(1, "protocol", protocol_name)
            ci.insert(2, "model", model_name)
            bootstrap_frames.append(ci)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(results / "bootstrap_ci.csv", index=False)

    best_temporal_classifier = best_model_from_tuning(
        classification_tuning, "temporal", "best_cv_macro_f1", True
    )
    best_temporal_regressor = best_model_from_tuning(
        regression_tuning, "temporal", "best_cv_neg_rmse", True
    )
    temporal_idx = ctx["protocols"]["temporal"]["test_idx"]
    importance_n = min(int(profile["importance_sample_size"]), len(temporal_idx))
    importance_indices = (
        pd.Series(temporal_idx).sample(n=importance_n, random_state=seed).to_numpy()
    )
    X_importance = df.loc[importance_indices, FEATURE_COLUMNS]

    classifier = joblib.load(models / f"classification_temporal_{best_temporal_classifier}.joblib")
    cls_perm = permutation_importance(
        classifier,
        X_importance,
        df.loc[importance_indices, "gas_class"],
        scoring="f1_macro",
        n_repeats=int(profile["importance_repeats"]),
        random_state=seed,
        n_jobs=-1,
    )
    classification_importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": cls_perm.importances_mean,
                "importance_std": cls_perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    classification_importance.to_csv(results / "classification_permutation_importance.csv", index=False)

    regressor = joblib.load(models / f"regression_temporal_{best_temporal_regressor}.joblib")
    reg_perm = permutation_importance(
        regressor,
        X_importance,
        df.loc[importance_indices, "concentration_ppmv"],
        scoring="neg_root_mean_squared_error",
        n_repeats=int(profile["importance_repeats"]),
        random_state=seed,
        n_jobs=-1,
    )
    regression_importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": reg_perm.importances_mean,
                "importance_std": reg_perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    regression_importance.to_csv(results / "regression_permutation_importance.csv", index=False)

    best_random_classifier = best_model_from_tuning(
        classification_tuning, "random", "best_cv_macro_f1", True
    )
    best_random_regressor = best_model_from_tuning(
        regression_tuning, "random", "best_cv_neg_rmse", True
    )
    shutil.copy2(
        models / f"classification_random_{best_random_classifier}.joblib",
        models / "best_classification_model.joblib",
    )
    shutil.copy2(
        models / f"regression_random_{best_random_regressor}.joblib",
        models / "best_regression_model.joblib",
    )

    plot_eda(df, figures, seed)
    plot_classification_outputs(
        classification_metrics,
        classification_drift,
        classification_predictions,
        df,
        ctx["protocols"],
        figures,
    )
    plot_regression_outputs(regression_metrics, regression_drift, figures)
    plot_importance(
        classification_importance,
        f"Temporal classification importance - {best_temporal_classifier}",
        "Mean decrease in Macro-F1",
        "10_classification_permutation_importance.png",
        figures,
    )
    plot_importance(
        regression_importance,
        f"Temporal regression importance - {best_temporal_regressor}",
        "Increase in RMSE after permutation (ppmv)",
        "11_regression_permutation_importance.png",
        figures,
    )

    feature_schema = {
        "feature_names": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "classification_target": {
            "column": "gas_class",
            "mapping": {str(k - 1): v for k, v in GAS_MAP.items()},
        },
        "regression_target": {"column": "concentration_ppmv", "unit": "ppmv"},
        "drift_metadata": {"column": "batch", "values": list(range(1, 11))},
    }
    write_json(results / "feature_schema.json", feature_schema)

    selected_dbscan = None
    pca_components = None
    clustering_path = results / "clustering_metrics.csv"
    dbscan_path = results / "dbscan_parameter_search.csv"
    if dbscan_path.exists():
        dbscan = pd.read_csv(dbscan_path)
        selected_dbscan = (
            dbscan.dropna(subset=["silhouette"])
            .sort_values(["silhouette", "noise_fraction"], ascending=[False, True])
            .head(1)
            .to_dict("records")
        )
        selected_dbscan = selected_dbscan[0] if selected_dbscan else None
    pca_model = models / "clustering_pca.joblib"
    if pca_model.exists():
        pca_components = int(joblib.load(pca_model).n_components_)

    experiment_manifest = {
        "experiment_name": "gas_sensor_drift_multitask_benchmark",
        "dataset": ctx["data_manifest"],
        "seed": seed,
        "profile": ctx["profile_name"],
        "random_test_size": float(ctx["config"]["protocols"]["random"]["test_size"]),
        "temporal_train_batches": ctx["config"]["protocols"]["temporal"]["train_batches"],
        "temporal_test_batches": ctx["config"]["protocols"]["temporal"]["test_batches"],
        "tuning_iterations": int(profile["tuning_iterations"]),
        "tuning_cv_folds": int(ctx["config"]["cross_validation"]["tuning_folds"]),
        "bootstrap_iterations": int(profile["bootstrap_iterations"]),
        "cluster_sample_size": int(profile["cluster_sample_size"]),
        "silhouette_sample_size": int(profile["silhouette_sample_size"]),
        "importance_sample_size": int(profile["importance_sample_size"]),
        "importance_repeats": int(profile["importance_repeats"]),
        "best_demo_classifier": best_random_classifier,
        "best_demo_regressor": best_random_regressor,
        "best_temporal_classifier_by_training_cv": best_temporal_classifier,
        "best_temporal_regressor_by_training_cv": best_temporal_regressor,
        "selected_dbscan": selected_dbscan,
        "pca_components_95_variance": pca_components,
        "git_commit": get_git_commit(ctx["root"]),
        "packages": package_versions(),
        "completed_at_utc": now_utc(),
        "leakage_audit": ctx["audit"],
    }
    write_json(results / "experiment_manifest.json", experiment_manifest)

    paper_summary = {
        "title": "Classification, Regression, and Clustering under Sensor Drift: A Reproducible Machine Learning Benchmark",
        "dataset": "UCI Gas Sensor Array Drift at Different Concentrations",
        "doi": ctx["config"]["dataset"]["doi"],
        "main_results": {
            "classification": "results/classification_metrics.csv",
            "regression": "results/regression_metrics.csv",
            "classification_drift": "results/classification_drift_comparison.csv",
            "regression_drift": "results/regression_drift_comparison.csv",
            "bootstrap": "results/bootstrap_ci.csv",
            "clustering": "results/clustering_metrics.csv",
        },
        "figures_directory": "figures/",
    }
    write_json(results / "paper_summary.json", paper_summary)

    if clustering_path.exists():
        print(pd.read_csv(clustering_path))
    print("Demo classifier:", best_random_classifier)
    print("Demo regressor:", best_random_regressor)
    print("Generated aggregate results.")


if __name__ == "__main__":
    main()
