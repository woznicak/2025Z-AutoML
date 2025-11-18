from typing import Dict, Tuple

import numpy as np
import openml
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DatasetDict = Dict[int, Tuple[np.ndarray, np.ndarray]]

def reduce_dataset_size(
    X: np.ndarray,
    y: np.ndarray,
    dataset_usage_percent: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_usage_percent >= 100.0:
        return X, y

    n_samples = len(X)
    target_samples = int(n_samples * (dataset_usage_percent / 100.0))
    if target_samples >= n_samples:
        return X, y

    X_reduced, _, y_reduced, _ = train_test_split(
        X,
        y,
        train_size=target_samples,
        stratify=y,
        random_state=random_state,
    )
    print(
        f"Reduced dataset from {n_samples} to {len(X_reduced)} samples "
        f"({dataset_usage_percent}%)"
    )
    return X_reduced, y_reduced

def load_datasets(openml_ids: list[int], dataset_usage_percent: float) -> DatasetDict:
    datasets: DatasetDict = {}
    for dataset_id in openml_ids:
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe",
            )

            for column in X.select_dtypes(include=["category", "object"]):
                encoder = LabelEncoder()
                X[column] = encoder.fit_transform(X[column].astype(str))

            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            if isinstance(y, pd.Series):
                y = y.to_numpy()
            if y.dtype.kind in "OUS":
                y = LabelEncoder().fit_transform(y)

            X = X.to_numpy(dtype=np.float64, copy=True)
            y = np.asarray(y)

            if dataset_usage_percent < 100.0:
                X, y = reduce_dataset_size(X, y, dataset_usage_percent)

            datasets[dataset_id] = (X, y)
            print(
                f"Loaded dataset {dataset_id} with shape {X.shape} "
                f"({dataset_usage_percent}% of original)"
            )
        except Exception as exc:
            print(f"Error loading dataset {dataset_id}: {exc}")
    return datasets


def build_dataset_info(
    datasets: DatasetDict, dataset_usage_percent: float
) -> list[dict[str, int | float]]:
    info: list[dict[str, int | float]] = []
    for dataset_id, (X, y) in datasets.items():
        info.append(
            {
                "dataset_id": dataset_id,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "dataset_usage_percent": dataset_usage_percent,
            }
        )
    return info
