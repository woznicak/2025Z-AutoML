from __future__ import annotations

import math
import pickle
import warnings
from typing import Any, Dict
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import DatasetDict, build_dataset_info, load_datasets

warnings.filterwarnings("ignore")

class MultiDatasetHyperparameterOptimization:
    def __init__(
        self,
        openml_ids: list[int],
        ml_models: Dict[str, Any],
        sampling_methods: list[str],
        dataset_usage_percent: float,
    ) -> None:
        self.openml_ids = openml_ids
        self.ml_models = ml_models
        self.sampling_methods = sampling_methods
        self.dataset_usage_percent = dataset_usage_percent

        self.datasets: DatasetDict = load_datasets(
            openml_ids, dataset_usage_percent
        )
        self.results: Dict[str, Dict[str, Any]] = {}
        self.trial_results: Dict[str, list[dict[str, Any]]] = {}

    def _create_model_pipeline(self, model_name: str, params: Dict[str, Any]) -> Pipeline:

        model_class = self.ml_models[model_name]["class"]
        model_params = params.copy()
        if model_name == "svm":
            model_params = {**model_params, "probability": True}

        if model_name in ["svm", "knn"]:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", model_class(**model_params)),
            ])
        return Pipeline([("model", model_class(**model_params))])

    def _build_grid_search_space(self, model_name: str, n_trials: int) -> Dict[str, list[Any]]:

        search_space: Dict[str, list[Any]] = {}
        model_config = self.ml_models[model_name]["hyperparameters"]
        
        n_params = len(model_config)
        if n_params == 0:
            return search_space
        
        # Calculate how many values per parameter to roughly match n_trials
        # For n_params dimensions with k values each: k^n_params ≈ n_trials
        # So k ≈ n_trials^(1/n_params)
        values_per_param = max(2, math.ceil(n_trials ** (1.0 / n_params)))

        for param_name, param_config in model_config.items():
            if param_config["type"] == "categorical":
                search_space[param_name] = list(param_config["choices"])
            elif param_config["type"] == "int":
                if "choices" in param_config:
                    search_space[param_name] = list(param_config["choices"])
                else:
                    low = param_config["low"]
                    high = param_config["high"]
                    step = max(1, (high - low) // (values_per_param - 1))
                    search_space[param_name] = list(range(low, high + 1, step))[:values_per_param]
            elif param_config["type"] == "float":
                if "choices" in param_config:
                    search_space[param_name] = list(param_config["choices"])
                else:
                    if param_config.get("log", False):
                        values = np.logspace(
                            np.log10(param_config["low"]),
                            np.log10(param_config["high"]),
                            values_per_param
                        )
                    else:
                        values = np.linspace(
                            param_config["low"],
                            param_config["high"],
                            values_per_param
                        )
                    search_space[param_name] = [float(value) for value in values]

        return search_space

    def _objective(self, trial: optuna.Trial, model_name: str, study_key: str) -> float:
        model_config = self.ml_models[model_name]
        params: Dict[str, Any] = {}

        for param_name, param_config in model_config["hyperparameters"].items():
            if param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )

        dataset_results: Dict[str, Dict[str, float]] = {}
        auc_scores: list[float] = []
        accuracy_scores: list[float] = []

        try:
            for dataset_id, (X, y) in self.datasets.items():
                pipeline = self._create_model_pipeline(model_name, params)
                cv_results = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=5,
                    scoring={
                        "accuracy": "accuracy",
                        "auc": "roc_auc",
                    },
                    n_jobs=-1,
                    error_score=np.nan,
                )

                dataset_accuracy = float(np.nanmean(cv_results["test_accuracy"]))
                dataset_auc = float(np.nanmean(cv_results["test_auc"]))

                if np.isnan(dataset_accuracy):
                    dataset_accuracy = 0.0
                if np.isnan(dataset_auc):
                    dataset_auc = 0.0

                accuracy_scores.append(dataset_accuracy)
                auc_scores.append(dataset_auc)
                dataset_results[str(dataset_id)] = {
                    "accuracy": dataset_accuracy,
                    "auc": dataset_auc,
                }
                trial.set_user_attr(f"dataset_{dataset_id}_accuracy", dataset_accuracy)
                trial.set_user_attr(f"dataset_{dataset_id}_auc", dataset_auc)

            mean_accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
            mean_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

            trial_data = {
                "trial_number": trial.number,
                "params": params.copy(),
                "mean_score": mean_auc,
                "mean_auc": mean_auc,
                "mean_accuracy": mean_accuracy,
                "mean_metrics": {
                    "auc": mean_auc,
                    "accuracy": mean_accuracy,
                },
                "dataset_results": dataset_results.copy(),
                "dataset_usage_percent": self.dataset_usage_percent,
                "datetime": pd.Timestamp.now(),
            }

            if study_key not in self.trial_results:
                self.trial_results[study_key] = []
            self.trial_results[study_key].append(trial_data)

            trial.set_user_attr("dataset_results", dataset_results)
            trial.set_user_attr("mean_score", mean_auc)
            trial.set_user_attr("mean_auc", mean_auc)
            trial.set_user_attr("mean_accuracy", mean_accuracy)
            trial.set_user_attr(
                "dataset_usage_percent", self.dataset_usage_percent
            )

            return mean_auc
        except Exception as exc:
            print(f"Error in trial {trial.number}: {exc}")
            trial_data = {
                "trial_number": trial.number,
                "params": params.copy(),
                "mean_score": 0.0,
                "mean_auc": 0.0,
                "mean_accuracy": 0.0,
                "mean_metrics": {
                    "auc": 0.0,
                    "accuracy": 0.0,
                },
                "dataset_results": {
                    str(dataset_id): {"accuracy": 0.0, "auc": 0.0}
                    for dataset_id in self.datasets.keys()
                },
                "dataset_usage_percent": self.dataset_usage_percent,
                "datetime": pd.Timestamp.now(),
                "error": str(exc),
            }
            if study_key not in self.trial_results:
                self.trial_results[study_key] = []
            self.trial_results[study_key].append(trial_data)
            return 0.0

    def optimize_model(
        self, model_name: str, sampling_method: str, n_trials: int
    ) -> Dict[str, Any]:
        print(
            f"Optimizing {model_name} using {sampling_method} sampling "
            f"(dataset usage: {self.dataset_usage_percent}%)..."
        )

        if sampling_method == "bayesian":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif sampling_method == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampling_method == "grid":
            search_space = self._build_grid_search_space(model_name, n_trials)
            sampler = optuna.samplers.GridSampler(search_space, seed=42)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study_key = f"{model_name}_{sampling_method}_{self.dataset_usage_percent}"

        study.optimize(
            lambda trial: self._objective(trial, model_name, study_key),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        trial_results = self.trial_results.get(study_key, [])
        best_accuracy = max(
            (trial_info.get("mean_accuracy") for trial_info in trial_results),
            default=0.0,
        )
        best_scores = {"auc": study.best_value, "accuracy": best_accuracy}
        result = {
            "study": study,
            "trial_results": trial_results,
            "best_params": study.best_params,
            "best_score": study.best_value,
            "best_scores": best_scores,
            "model_name": model_name,
            "sampling_method": sampling_method,
            "dataset_usage_percent": self.dataset_usage_percent,
            "optimization_metric": "auc",
        }
        self.results[study_key] = result
        return result

    def run_complete_analysis(self, n_trials: int) -> Dict[str, Dict[str, Any]]:
        all_results: Dict[str, Dict[str, Any]] = {}
        for model_name in self.ml_models:
            model_results: Dict[str, Any] = {}
            for sampling_method in self.sampling_methods:
                model_results[sampling_method] = self.optimize_model(
                    model_name, sampling_method, n_trials
                )
            all_results[model_name] = model_results
        self.complete_results = all_results
        return all_results

    def save_results(self, filename: str) -> None:
        with open(filename, "wb") as file_handle:
            pickle.dump(
                {
                    "results": self.results,
                    "trial_results": self.trial_results,
                    "complete_results": getattr(self, "complete_results", None),
                    "datasets": self.datasets,
                    "dataset_info": build_dataset_info(self.datasets, self.dataset_usage_percent),
                    "config": {
                        "openml_ids": self.openml_ids,
                        "ml_models": self.ml_models,
                        "sampling_methods": self.sampling_methods,
                        "dataset_usage_percent": self.dataset_usage_percent,
                    },
                },
                file_handle,
            )
