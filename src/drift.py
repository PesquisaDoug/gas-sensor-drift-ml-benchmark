from __future__ import annotations

import pandas as pd


def classification_drift_table(classification_metrics_df: pd.DataFrame) -> pd.DataFrame:
    random_cls = classification_metrics_df.query("protocol == 'random'").set_index("model")
    temporal_cls = classification_metrics_df.query("protocol == 'temporal'").set_index("model")
    return pd.DataFrame(
        {
            "random_macro_f1": random_cls["macro_f1"],
            "temporal_macro_f1": temporal_cls["macro_f1"],
            "delta_macro_f1_temporal_minus_random": temporal_cls["macro_f1"] - random_cls["macro_f1"],
            "random_balanced_accuracy": random_cls["balanced_accuracy"],
            "temporal_balanced_accuracy": temporal_cls["balanced_accuracy"],
            "delta_balanced_accuracy": temporal_cls["balanced_accuracy"] - random_cls["balanced_accuracy"],
        }
    ).reset_index()


def regression_drift_table(regression_metrics_df: pd.DataFrame) -> pd.DataFrame:
    random_reg = regression_metrics_df.query("protocol == 'random'").set_index("model")
    temporal_reg = regression_metrics_df.query("protocol == 'temporal'").set_index("model")
    return pd.DataFrame(
        {
            "random_rmse": random_reg["rmse"],
            "temporal_rmse": temporal_reg["rmse"],
            "delta_rmse_temporal_minus_random": temporal_reg["rmse"] - random_reg["rmse"],
            "random_mae": random_reg["mae"],
            "temporal_mae": temporal_reg["mae"],
            "delta_mae_temporal_minus_random": temporal_reg["mae"] - random_reg["mae"],
            "random_r2": random_reg["r2"],
            "temporal_r2": temporal_reg["r2"],
            "delta_r2_temporal_minus_random": temporal_reg["r2"] - random_reg["r2"],
        }
    ).reset_index()
