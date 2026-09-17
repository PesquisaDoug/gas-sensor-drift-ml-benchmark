from __future__ import annotations

import argparse

from src.evaluation import regression_metrics
from src.models import regression_models, regression_spaces
from src.training import run_supervised_benchmark
from scripts.common import load_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    args = parser.parse_args()

    ctx = load_context(args.config, args.profile)
    run_supervised_benchmark(
        task="regression",
        df=ctx["df"],
        protocols=ctx["protocols"],
        estimators=regression_models(int(ctx["config"]["random_seed"])),
        search_spaces=regression_spaces(),
        metrics_fn=regression_metrics,
        target_column="concentration_ppmv",
        scoring="neg_root_mean_squared_error",
        config=ctx["config"],
        profile=ctx["profile"],
        models_dir=ctx["models_dir"],
        results_dir=ctx["results_dir"],
    )


if __name__ == "__main__":
    main()
