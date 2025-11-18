from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from optimizer import MultiDatasetHyperparameterOptimization


def cumulative_best(scores: list[float]) -> list[float]:
    best_so_far = float("-inf")
    best_scores: list[float] = []
    for score in scores:
        best_so_far = max(best_so_far, score)
        best_scores.append(best_so_far)
    return best_scores


def mean_metric(trial_info: dict[str, object], metric: str) -> float:
    key = "mean_auc" if metric == "auc" else "mean_accuracy"
    value = trial_info.get(key) if isinstance(trial_info, dict) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def dataset_metric(trial_info: dict[str, object], dataset_id: str, metric: str) -> float:
    dataset_results = trial_info.get("dataset_results") if isinstance(trial_info, dict) else None
    if not isinstance(dataset_results, dict):
        return 0.0
    dataset_entry = dataset_results.get(dataset_id)
    if not isinstance(dataset_entry, dict):
        return 0.0
    value = dataset_entry.get(metric)
    return float(value) if isinstance(value, (int, float)) else 0.0


def plot_convergence(
    optimizer: MultiDatasetHyperparameterOptimization,
    model_name: Optional[str] = None,
    sampling_method: Optional[str] = None,
    save_path: Optional[str] = None,
    metric: str = "auc",
    show: bool = False,
) -> None:
    if not hasattr(optimizer, "complete_results"):
        print("No results available. Run optimization first.")
        return

    metric = metric.lower()
    if metric not in {"auc", "accuracy"}:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'auc' or 'accuracy'.")

    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    models_to_plot = [model_name] if model_name else list(optimizer.ml_models.keys())
    methods_to_plot = (
        [sampling_method] if sampling_method else optimizer.sampling_methods
    )

    n_plots = len(models_to_plot) * len(methods_to_plot)
    if n_plots == 0:
        print("No data to plot")
        return

    fig = plt.figure(figsize=(6, 5 * n_plots))
    gs = GridSpec(n_plots, 1, figure=fig)

    plot_idx = 0
    for model_name_item in models_to_plot:
        for sampling_method_item in methods_to_plot:
            result_key = (
                f"{model_name_item}_{sampling_method_item}_"
                f"{optimizer.dataset_usage_percent}"
            )
            if result_key not in optimizer.results:
                continue

            result = optimizer.results[result_key]
            trial_results = result["trial_results"]
            if not trial_results:
                continue

            ax = fig.add_subplot(gs[plot_idx])
            trials = [trial_info["trial_number"] for trial_info in trial_results]
            mean_scores = [mean_metric(trial_info, metric) for trial_info in trial_results]
            best_mean_scores = cumulative_best(mean_scores)
            ax.plot(
                trials,
                best_mean_scores,
                "k-",
                linewidth=3,
                label=f"Best-so-far {metric.upper()}",
                alpha=0.9,
            )

            dataset_ids = list(optimizer.datasets.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_ids)))
            for idx, dataset_id in enumerate(dataset_ids):
                dataset_scores = [
                    dataset_metric(trial_info, str(dataset_id), metric)
                    for trial_info in trial_results
                ]

                best_dataset_scores = cumulative_best(dataset_scores)
                ax.plot(
                    trials,
                    best_dataset_scores,
                    "--",
                    color=colors[idx],
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"Dataset {dataset_id} Best-so-far",
                )

            ax.set_title(
                f"{model_name_item.upper()} - {sampling_method_item.upper()} Sampling\n"
                f"Dataset Usage: {optimizer.dataset_usage_percent}%",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Iteration Number", fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            best_scores = result.get("best_scores") if isinstance(result, dict) else None
            best_score = (
                float(best_scores.get(metric))
                if isinstance(best_scores, dict) and isinstance(best_scores.get(metric), (int, float))
                else None
            )
            best_iter = int(np.argmax(best_mean_scores))
            if best_score is not None:
                ax.axhline(y=best_score, color="red", linestyle=":", alpha=0.8)
            ax.axvline(x=best_iter, color="red", linestyle=":", alpha=0.8)

            plot_idx += 1

    if plot_idx == 0:
        print("No valid data to plot")
        plt.close(fig)
        return

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()


def plot_comparative_convergence(
    optimizer: MultiDatasetHyperparameterOptimization,
    save_path: Optional[str] = None,
    metric: str = "auc",
    show: bool = False,
) -> None:
    if not hasattr(optimizer, "complete_results"):
        print("No results available. Run optimization first.")
        return

    metric = metric.lower()
    if metric not in {"auc", "accuracy"}:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'auc' or 'accuracy'.")

    valid_models: list[str] = []
    for model_name in optimizer.ml_models:
        for sampling_method in optimizer.sampling_methods:
            key = f"{model_name}_{sampling_method}_{optimizer.dataset_usage_percent}"
            if key in optimizer.results and optimizer.results[key]["trial_results"]:
                valid_models.append(model_name)
                break

    if not valid_models:
        print("No valid results to plot")
        return

    n_models = len(valid_models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    else:
        axes = list(axes)

    sampling_colors = {"bayesian": "blue", "random": "orange"}

    plot_idx = 0
    for model_name in valid_models:
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        for sampling_method in optimizer.sampling_methods:
            key = f"{model_name}_{sampling_method}_{optimizer.dataset_usage_percent}"
            if key not in optimizer.results:
                continue

            result = optimizer.results[key]
            trials = [trial_info["trial_number"] for trial_info in result["trial_results"]]
            if not trials:
                continue
            mean_scores = [
                mean_metric(trial_info, metric)
                for trial_info in result["trial_results"]
            ]
            cumulative_scores = cumulative_best(mean_scores)
            ax.plot(
                trials,
                cumulative_scores,
                color=sampling_colors.get(sampling_method, "gray"),
                linewidth=2,
                label=f"{sampling_method.upper()} Best-so-far {metric.upper()}",
            )

        ax.set_title(f"{model_name.upper()}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Iteration Number", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1

    for index in range(plot_idx, len(axes)):
        fig.delaxes(axes[index])

    plt.suptitle(
        f"Comparative Convergence Analysis ({metric.upper()})\nDataset Usage: {optimizer.dataset_usage_percent}%",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparative plot saved to {save_path}")
    
    if show:
        plt.show()
