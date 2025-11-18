from __future__ import annotations
from typing import Optional

import pandas as pd

from optimizer import MultiDatasetHyperparameterOptimization


def analyze_tunability(
    optimizer: MultiDatasetHyperparameterOptimization,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    if not hasattr(optimizer, "complete_results"):
        print("No results available. Run optimization first.")
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model_name, model_results in optimizer.complete_results.items():
        for sampling_method, result in model_results.items():
            trial_results = result["trial_results"]
            if not trial_results:
                continue
            best_overall_trial = max(
                trial_results,
                key=lambda trial_info: trial_info.get("mean_metrics", {}).get("auc", trial_info.get("mean_score", 0.0)),
            )

            dataset_rows: list[dict[str, object]] = []
            for dataset_id in optimizer.datasets.keys():
                dataset_key = str(dataset_id)

                reference_results = best_overall_trial["dataset_results"].get(dataset_key)
                if not reference_results:
                    continue
                reference_auc = reference_results.get("auc")
                if reference_auc is None:
                    continue

                dataset_scores = [
                    trial_info["dataset_results"].get(dataset_key, {}).get("auc")
                    for trial_info in trial_results
                    if dataset_key in trial_info["dataset_results"]
                ]
                if not dataset_scores:
                    continue

                dataset_scores = [score for score in dataset_scores if score is not None]
                if not dataset_scores:
                    continue

                best_dataset_auc = max(dataset_scores)

                reference_risk = 1.0 - float(reference_auc)
                best_dataset_risk = 1.0 - float(best_dataset_auc)
                tunability_risk_diff = reference_risk - best_dataset_risk

                dataset_rows.append(
                    {
                        "model": model_name,
                        "sampling_method": sampling_method,
                        "dataset_id": dataset_id,
                        "dataset_usage_percent": optimizer.dataset_usage_percent,
                        "reference_auc": float(reference_auc),
                        "reference_risk": reference_risk,
                        "best_dataset_auc": float(best_dataset_auc),
                        "best_dataset_risk": best_dataset_risk,
                        "tunability_risk_diff": tunability_risk_diff,
                        "n_trials": len(trial_results),
                    }
                )

            rows.extend(dataset_rows)
            
    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False)

    return df