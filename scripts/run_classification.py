from __future__ import annotations

import argparse

from src.evaluation import classification_metrics
from src.models import classification_models, classification_spaces
from src.training import run_supervised_benchmark
from scripts.common import load_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", default="quick", choices=["quick", "full"])
    args = parser.parse_args()

    ctx = load_context(args.config, args.profile)
    run_supervised_benchmark(
        task="classification",
        df=ctx["df"],
        protocols=ctx["protocols"],
        estimators=classification_models(int(ctx["config"]["random_seed"])),
        search_spaces=classification_spaces(),
        metrics_fn=classification_metrics,
        target_column="gas_class",
        scoring="f1_macro",
        config=ctx["config"],
        profile=ctx["profile"],
        models_dir=ctx["models_dir"],
        results_dir=ctx["results_dir"],
    )


if __name__ == "__main__":
    main()
