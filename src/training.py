from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

from src.data import FEATURE_COLUMNS, GAS_MAP
from src.evaluation import timed_predict
from src.protocols import cv_for_protocol


def save_estimator(estimator, path: str | Path) -> float:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path, compress=3)
    return path.stat().st_size / (1024**2)


def run_supervised_benchmark(
    *,
    task: str,
    df: pd.DataFrame,
    protocols: dict,
    estimators: dict,
    search_spaces: dict,
    metrics_fn: Callable,
    target_column: str,
    scoring: str,
    config: dict[str, Any],
    profile: dict[str, Any],
    models_dir: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    tuning_rows = []
    folds = int(config["cross_validation"]["tuning_folds"])
    seed = int(config["random_seed"])
    n_jobs = int(config["resources"]["search_n_jobs"])
    n_iter = int(profile["tuning_iterations"])
    repeats = int(profile["inference_repeats"])

    for protocol_name, split in protocols.items():
        print(f"\n=== {task.upper()}: {protocol_name.upper()} ===", flush=True)
        train_idx, test_idx = split["train_idx"], split["test_idx"]
        X_train = df.loc[train_idx, FEATURE_COLUMNS]
        y_train = df.loc[train_idx, target_column]
        X_test = df.loc[test_idx, FEATURE_COLUMNS]
        y_test = df.loc[test_idx, target_column]
        cv, groups = cv_for_protocol(task, protocol_name, df, train_idx, folds, seed)

        for model_name, base_estimator in estimators.items():
            print(f"Training {model_name}", flush=True)
            if model_name == "Dummy":
                estimator = clone(base_estimator)
                start = time.perf_counter()
                estimator.fit(X_train, y_train)
                fit_time = time.perf_counter() - start
                best_cv_score = float("nan")
                best_params = {}
            else:
                search = RandomizedSearchCV(
                    base_estimator,
                    search_spaces[model_name],
                    n_iter=n_iter,
                    scoring=scoring,
                    cv=cv,
                    random_state=seed,
                    n_jobs=n_jobs,
                    refit=True,
                )
                start = time.perf_counter()
                if groups is None:
                    search.fit(X_train, y_train)
                else:
                    search.fit(X_train, y_train, groups=groups)
                fit_time = time.perf_counter() - start
                estimator = search.best_estimator_
                best_cv_score = float(search.best_score_)
                best_params = search.best_params_

            y_pred = estimator.predict(X_test)
            model_filename = f"{task}_{protocol_name}_{model_name}.joblib"
            row = metrics_fn(y_test, y_pred)
            row.update(
                {
                    "protocol": protocol_name,
                    "model": model_name,
                    "training_seconds": fit_time,
                    "model_size_mb": save_estimator(estimator, models_dir / model_filename),
                    **timed_predict(estimator, X_test, repeats),
                }
            )
            metric_rows.append(row)

            score_name = "best_cv_macro_f1" if task == "classification" else "best_cv_neg_rmse"
            tuning_rows.append(
                {
                    "protocol": protocol_name,
                    "model": model_name,
                    score_name: best_cv_score,
                    "best_params": json.dumps(best_params, default=str),
                }
            )

            if task == "classification":
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "row_index": test_idx,
                            "protocol": protocol_name,
                            "model": model_name,
                            "batch": df.loc[test_idx, "batch"].to_numpy(),
                            "gas_true": df.loc[test_idx, "gas"].to_numpy(),
                            "gas_class_true": y_test.to_numpy(),
                            "gas_class_pred": y_pred,
                            "gas_pred": [GAS_MAP[int(v) + 1] for v in y_pred],
                        }
                    )
                )
            else:
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "row_index": test_idx,
                            "protocol": protocol_name,
                            "model": model_name,
                            "batch": df.loc[test_idx, "batch"].to_numpy(),
                            "gas": df.loc[test_idx, "gas"].to_numpy(),
                            "concentration_true": y_test.to_numpy(),
                            "concentration_pred": y_pred,
                            "residual": y_test.to_numpy() - y_pred,
                        }
                    )
                )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    tuning_df = pd.DataFrame(tuning_rows)
    metrics_df.to_csv(results_dir / f"{task}_metrics.csv", index=False)
    predictions_df.to_csv(results_dir / f"{task}_predictions.csv", index=False)
    tuning_df.to_csv(results_dir / f"{task}_tuning_summary.csv", index=False)
    return metrics_df, predictions_df, tuning_df


def best_model_from_tuning(tuning_df: pd.DataFrame, protocol: str, score_column: str, maximize: bool = True) -> str:
    subset = tuning_df.loc[
        (tuning_df["protocol"] == protocol) & (tuning_df["model"] != "Dummy")
    ].dropna(subset=[score_column])
    if subset.empty:
        raise ValueError("No tuned non-dummy models available.")
    return subset.sort_values(score_column, ascending=not maximize).iloc[0]["model"]
