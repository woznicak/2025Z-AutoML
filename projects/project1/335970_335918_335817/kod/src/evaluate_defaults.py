import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from preprocessing import preprocess_data
from models import get_xgboost, get_lightgbm, get_catboost

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="sklearn.preprocessing._encoders")


DATASETS = {
    'adult': ('src/data/adult.csv', 'target', 'adult'),
    'kr_vs_kp': ('src/data/kr_vs_kp.csv', 'target', 'kr_vs_kp'),
    'mushroom': ('src/data/mushroom.csv', 'target', 'mushroom'),
    'magic_telescope': ('src/data/magic_telescope.csv', 'target', 'magic_telescope'),
    'numerai': ('src/data/numerai.csv', 'target', 'numerai')
}


MODELS = {
    'xgboost': get_xgboost,
    'lightgbm': get_lightgbm,
    'catboost': get_catboost
}


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset: {file_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def evaluate_default_model(df, target_column, model_name, model_func, dataset_name, cv=3):
    X, y, col_trans = preprocess_data(df, target_column)
    estimator, _ = model_func()

    pipeline = Pipeline([
        ('preprocessing', col_trans),
        ('model', estimator)
    ])

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    print(
        f"Evaluating default {model_name} on '{dataset_name}' ({cv}-fold CV)...")

    scoring = ['roc_auc', 'accuracy', 'f1']
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        verbose=0
    )

    results = {
        "dataset": dataset_name,
        "algorithm": model_name,
        "mean_test_roc_auc": np.mean(scores["test_roc_auc"]),
        "std_test_roc_auc": np.std(scores["test_roc_auc"]),
        "mean_test_accuracy": np.mean(scores["test_accuracy"]),
        "std_test_accuracy": np.std(scores["test_accuracy"]),
        "mean_test_f1": np.mean(scores["test_f1"]),
        "std_test_f1": np.std(scores["test_f1"]),
    }

    output_dir = "results_defaults"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/{dataset_name}_{model_name}_defaults.csv"
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}\n")

    return results


def main():
    all_results = []

    for dataset_name, (file_path, target_col, dataset_alias) in DATASETS.items():
        df = load_dataset(file_path)
        if df is None:
            continue

        for model_name, model_func in MODELS.items():
            res = evaluate_default_model(
                df, target_col, model_name, model_func, dataset_alias)
            all_results.append(res)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results_defaults/defaults_summary.csv", index=False)
    print("All default evaluations complete.")
    print(df_results)


if __name__ == "__main__":
    main()
