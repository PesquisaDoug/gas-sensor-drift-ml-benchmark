from __future__ import annotations

from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import xgboost as xgb


def classification_models(seed: int = 42) -> dict:
    return {
        "Dummy": DummyClassifier(strategy="prior", random_state=seed),
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            random_state=seed,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
        ),
        "SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(probability=True, random_state=seed)),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(random_state=seed, early_stopping=True, max_iter=400),
                ),
            ]
        ),
    }


def classification_spaces() -> dict:
    return {
        "RandomForest": {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 12, 24],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.08, 0.15],
            "subsample": [0.75, 1.0],
            "colsample_bytree": [0.75, 1.0],
        },
        "SVM": {
            "model__C": loguniform(1e-2, 1e2),
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"],
        },
        "MLP": {
            "model__hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": loguniform(1e-5, 1e-2),
            "model__learning_rate_init": loguniform(1e-4, 1e-2),
        },
    }


def regression_models(seed: int = 42) -> dict:
    return {
        "Dummy": DummyRegressor(strategy="median"),
        "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(
            random_state=seed,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
        ),
        "SVR": Pipeline([("scale", StandardScaler()), ("model", SVR())]),
        "MLP": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(random_state=seed, early_stopping=True, max_iter=500),
                ),
            ]
        ),
    }


def regression_spaces() -> dict:
    return {
        "RandomForest": {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 12, 24],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.7, 1.0],
        },
        "XGBoost": {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.08, 0.15],
            "subsample": [0.75, 1.0],
            "colsample_bytree": [0.75, 1.0],
        },
        "SVR": {
            "model__C": loguniform(1e-2, 1e3),
            "model__epsilon": [0.1, 1.0, 5.0, 10.0],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf"],
        },
        "MLP": {
            "model__hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": loguniform(1e-5, 1e-2),
            "model__learning_rate_init": loguniform(1e-4, 1e-2),
        },
    }
