from __future__ import annotations
from pathlib import Path

from analyzer import analyze_tunability
from config import (
    DEFAULT_DATASET_USAGES,
    DEFAULT_SAMPLING_METHODS,
    ML_MODELS,
    OPENML_DATASETS,
    N_TRIALS,
)
from optimizer import MultiDatasetHyperparameterOptimization
from visualizer import plot_comparative_convergence, plot_convergence


def main() -> None:

    Path("results").mkdir(parents=True, exist_ok=True)

    for dataset_usage in DEFAULT_DATASET_USAGES:

        Path(f"results/usage_{dataset_usage}").mkdir(parents=True, exist_ok=True)

        optimizer = MultiDatasetHyperparameterOptimization(
            openml_ids=OPENML_DATASETS,
            ml_models=ML_MODELS,
            sampling_methods=DEFAULT_SAMPLING_METHODS,
            dataset_usage_percent=dataset_usage,
        )

        print(f"Running optimization for dataset usage {dataset_usage}%...")
        optimizer.run_complete_analysis(n_trials=N_TRIALS)

        print("Generating convergence plots...")
        for metric in ("auc", "accuracy"):
            plot_convergence(
                optimizer,
                save_path=f"results/usage_{dataset_usage}/convergence_plot_{metric}.png",
                metric=metric,
            )
            plot_comparative_convergence(
                optimizer,
                save_path=f"results/usage_{dataset_usage}/comparative_convergence_{metric}.png",
                metric=metric,
            )

        tunability_df = analyze_tunability(optimizer, save_path=f"results/usage_{dataset_usage}/tunability_analysis.csv")
        print("\nTunability Analysis:")
        print(tunability_df)

        optimizer.save_results(f"results/usage_{dataset_usage}/optimization_results.pkl")
        print("Results saved successfully!")


if __name__ == "__main__":
    main()
