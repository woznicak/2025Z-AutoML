import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_cummax(file_path, score_column):
    df = pd.read_csv(file_path)
    df['max_value'] = df[score_column].cummax()
    return df['max_value'].values

DATASETS = [
    'adult',
    'kr_vs_kp',
    'mushroom',
    'magic_telescope',
    'numerai'
]

MODELS = [
    'xgboost',
    'lightgbm',
    'catboost'
]

MODEL_COLORS = {
    'xgboost': 'blue',
    'lightgbm': 'red',
    'catboost': 'black'
}

def main():
    base_path = 'C:\\Users\\agama\\OneDrive\\Dokumenty\\Projects\\AuML\\automl_project1\\results\\'
    save_base_path = 'C:\\Users\\agama\\OneDrive\\Dokumenty\\Projects\\AuML\\automl_project1\\score_plots\\'

    for dataset in DATASETS:
        plt.figure(figsize=(10, 6))

        for model in MODELS:
            bayes_file = f"{base_path}{dataset}_{model}_bayes.csv"
            random_file = f"{base_path}{dataset}_{model}_randomsearch.csv"

            bayes_vals = load_and_cummax(bayes_file, 'value')
            random_vals = load_and_cummax(random_file, 'mean_test_roc_auc')

            x_bayes = np.arange(1, len(bayes_vals) + 1)
            x_random = np.arange(1, len(random_vals) + 1)

            plt.plot(
                x_bayes,
                bayes_vals,
                color=MODEL_COLORS[model],
                linestyle='-',
                label=f'{model} (Bayes)'
            )

            plt.plot(
                x_random,
                random_vals,
                color=MODEL_COLORS[model],
                linestyle='--',
                label=f'{model} (Random)'
            )

        plt.xlabel('Iteration')
        plt.ylabel('Max ROC AUC Score')
        plt.title(f'Max ROC AUC Score over Iterations – {dataset}')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        save_path = f"{save_base_path}{dataset}.png"
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    main()
