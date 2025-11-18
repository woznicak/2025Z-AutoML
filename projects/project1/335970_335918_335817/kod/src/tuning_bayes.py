import optuna
from preprocessing import preprocess_data
from models import get_xgboost, get_lightgbm, get_catboost
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DATASETS = {
    'adult': ('C:\\Users\\agama\\OneDrive\\Dokumenty\\Projects\\AuML\\automl_project1\\src\\data\\adult.csv', 'target', 'adult'),
    'kr_vs_kp': ('C:\\Users\\agama\\OneDrive\\Dokumenty\\Projects\\AuML\\automl_project1\\src\\data\\kr_vs_kp.csv', 'target', 'kr_vs_kp'),
    'mushroom': ('C:\\Users\\agama\\OneDrive\\Dokumenty\\Projects\\AuML\\automl_project1\\src\\data\\mushroom.csv', 'target', 'mushroom'),
    'magic_telescope': ('C:\\Users\\agama\\OneDrive\\Dokumenty\\Projects\\AuML\\automl_project1\\src\\data\\magic_telescope.csv', 'target', 'magic_telescope'),
    'numerai': ('C:\\Users\\agama\\OneDrive\\Dokumenty\\Projects\\AuML\\automl_project1\\src\\data\\numerai.csv', 'target', 'numerai')
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


def run_bayes_search(df, target_column, model_name, model_func, dataset_name, n_trials=50, cv_splits=3):
    X, y, col_trans = preprocess_data(df, target_column)
    base_estimator, param_space = model_func()

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    def objective(trial):
        params = {}
        for param_name, dist in param_space.items():
            if hasattr(dist, "rvs"):
                val = dist.rvs()
                params[param_name] = val
                trial.set_user_attr(param_name, val)

        estimator = base_estimator.set_params(**params)

        pipeline = Pipeline([
            ('preprocessing', col_trans),
            ('model', estimator)
        ])

        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean_score = scores.mean()
        return mean_score

    study_name = f"{dataset_name}_{model_name}_bayesopt"
    print(
        f"\nStarting Optuna optimization for {model_name} on dataset '{dataset_name}'")

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(
        f"Completed optimization for {model_name} on dataset '{dataset_name}'")
    print(f"Best ROC AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}\n")

    os.makedirs('results', exist_ok=True)
    results_df = study.trials_dataframe(
        attrs=("number", "value", "user_attrs", "state"))
    results_file = f'results/{dataset_name}_{model_name}_bayes.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    return study


def main():
    for dataset_name, (file_path, target_col, dataset_alias) in DATASETS.items():
        df = load_dataset(file_path)
        if df is None:
            continue

        for model_name, model_func in MODELS.items():
            run_bayes_search(df, target_col, model_name,
                             model_func, dataset_alias)


if __name__ == "__main__":
    main()
