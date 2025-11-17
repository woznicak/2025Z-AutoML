from preprocessing import preprocess_data
from models import get_xgboost, get_lightgbm, get_catboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        module="sklearn.preprocessing._encoders")

DATASETS = {
    'adult': ('data/adult.csv', 'target', 'adult'),
    'kr_vs_kp': ('data/kr_vs_kp.csv', 'target', 'kr_vs_kp'),
    'mushroom': ('data/mushroom.csv', 'target', 'mushroom'),
    'magic_telescope': ('data/magic_telescope.csv', 'target', 'magic_telescope'),
    'numerai': ('data/numerai.csv', 'target', 'numerai')
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


def run_random_search(df, target_column, model_name, model_func, dataset_name, n_iter=50, cv=3):
    X, y, col_trans = preprocess_data(df, target_column)
    estimator, rs_space = model_func()

    pipeline = Pipeline([
        ('preprocessing', col_trans),
        ('model', estimator)
    ])

    rs_space_prefixed = {f'model__{k}': v for k, v in rs_space.items()}

    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rs_space_prefixed,
        n_iter=n_iter,
        cv=cv,
        scoring=['roc_auc', 'accuracy', 'f1'],
        refit='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    try:
        print(
            f"Starting Random Search for {model_name} on dataset '{dataset_name}'")
        rs.fit(X, y)
        print(
            f"Completed Random Search for {model_name} on dataset '{dataset_name}'")
    except Exception as e:
        print(
            f"RandomizedSearchCV failed for {model_name} on dataset '{dataset_name}': {e}")
        return None

    os.makedirs('results', exist_ok=True)

    results_df = pd.DataFrame(rs.cv_results_)
    results_file = f'results/{dataset_name}_{model_name}_randomsearch.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    print(f"Best ROC AUC: {rs.best_score_:.4f}")
    print(f"Best params: {rs.best_params_}\n")

    return rs


def main():
    for dataset_name, (file_path, target_col, dataset_alias) in DATASETS.items():
        df = load_dataset(file_path)
        if df is None:
            continue

        for model_name, model_func in MODELS.items():
            run_random_search(df, target_col, model_name,
                              model_func, dataset_alias)


if __name__ == "__main__":
    main()
