import os
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV 
import matplotlib.pyplot as plt
from numpy import maximum
import re
import ast
import math
from scipy.stats import chi2_contingency

def load_and_split(path, target_name, test_size=0.2, random_state=42):
    data = pd.read_csv(path)
    X=  data.drop(columns=[target_name])
    y = data[target_name]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def tune(model, param_random_grid, param_bayes_grid, X_train, y_train, name, scoring='roc_auc', n_iter_random=50, n_iter_bayes=50):
    random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_random_grid,
    n_iter=n_iter_random,
    cv=3, 
    scoring=scoring,
    verbose=1,
    random_state=42,
    n_jobs=-1 
    )

    random_search.fit(X_train, y_train)    
    
    opt = BayesSearchCV(
    estimator=model,
    search_spaces=param_bayes_grid,
    n_iter=n_iter_bayes,
    cv=3,
    scoring=scoring,      
    n_jobs=-1,
    random_state=42,
    refit=True,
    verbose=0,
    )
    
    opt.fit(X_train, y_train)
    # zapis wyników do csv pod nazwą metody szukania (random, bayes), modelu i zbioru danych
    random_search_results = pd.DataFrame(random_search.cv_results_)
    bayes_search_results = pd.DataFrame(opt.cv_results_)

    os.makedirs("results_tunning", exist_ok=True)
    random_search_results.to_csv(f"results_tunning/random_search_results_{model.__class__.__name__}_{name}.csv", index=False)
    bayes_search_results.to_csv(f"results_tunning/bayes_search_results_{model.__class__.__name__}_{name}.csv", index=False)
    return random_search, opt

def tune_for_each_data(model, param_random_grid, param_bayes_grid, list_X_train, list_y_train, scoring='roc_auc'):
    results_rs = []
    results_bs = []
    scores_rs = []
    all_results_df = []

    for i in range(len(list_X_train)):
        k = f'dataset_{i}'
        rs, bs = tune(model, param_random_grid, param_bayes_grid, list_X_train[i], list_y_train[i], k, scoring)

        results_rs.append(rs)
        results_bs.append(bs)

        rs_df = pd.DataFrame(rs.cv_results_)
        params_df = pd.DataFrame(rs_df['params'].tolist())
        params_df['mean_test_score'] = rs_df['mean_test_score']
        params_df['dataset'] = f'dataset_{i+1}'
        
        all_results_df.append(params_df)
        scores_rs.append(rs_df['mean_test_score'])

    full_results_df = pd.concat(all_results_df, ignore_index=True)

    return results_rs, results_bs, scores_rs, full_results_df

def plots_iterations(rs, bs, model_name, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    random_iter = len(rs['mean_test_score'])
    bayes_iter = len(bs['mean_test_score'])

    rs_best = list(rs['mean_test_score'])
    bs_best = list(bs['mean_test_score'])
    rs_max_score = maximum.accumulate(rs["mean_test_score"])
    bs_max_score = maximum.accumulate(bs["mean_test_score"])

    axes[0].plot(range(1, random_iter + 1), rs_best, label='RandomizedSearchCV', marker='o')    
    axes[0].plot(range(1, random_iter + 1), rs_max_score, color="red", linestyle="--", label="Running best")
    axes[0].set_title('RandomizedSearchCV')
    axes[0].set_xlabel('Liczba iteracji')
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(range(1, bayes_iter + 1), bs_best, label='BayesSearchCV', marker='x')
    axes[1].plot(range(1, bayes_iter + 1), bs_max_score, color="red", linestyle="--", label="Running best")
    axes[1].set_title('BayesSearchCV')
    axes[1].set_xlabel('Liczba iteracji')
    axes[1].legend()
    axes[1].grid()

    axes[0].set_ylabel('Średni wynik testu')

    plt.suptitle(f"Porównanie postępu wyników dla RandomSearch i BayesSearch ({model_name}, {dataset_name})", fontsize=14)
    plt.tight_layout()
    plt.show()
    rs_best_param = rs['rank_test_score'].idxmin()
    bs_best_param = bs['rank_test_score'].idxmin()
    
    print('Najlepsze hiperparametry random:', rs['params'][rs_best_param])
    print('Najlepszy wynik random:', rs['mean_test_score'][rs_best_param])
    print('Najlepsze hiperparametry bayes:', bs['params'][bs_best_param])
    print('Najlepszy wynik bayes:', bs['mean_test_score'][bs_best_param])


def search_default(model_name):
    pattern = re.compile(rf'random_search_results_{model_name}_dataset_(\d+)\.csv')
    df_list = []

    for filename in os.listdir('results_tunning'):
        if match := pattern.match(filename):
            results_rs = pd.read_csv(os.path.join('results_tunning', filename))
            df_subset = results_rs[['params', 'mean_test_score']].copy()
            df_list.append(df_subset)
            results_rs = results_rs.dropna(subset=['mean_test_score'])

    if not df_list:
        print(f"Nie znaleziono plików dla modelu {model_name}")
        return None

    for i, df in enumerate(df_list):
        df.rename(columns={'mean_test_score': f'mean_test_score{i}'}, inplace=True)

    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], how='inner', on='params')

    score_cols = [col for col in merged_df.columns if col.startswith('mean_test_score')]
    merged_df['mean_test_score'] = merged_df[score_cols].mean(axis=1)
    merged_df = merged_df.drop(columns=score_cols)

    merged_df.sort_values('mean_test_score', ascending=False, inplace=True)

    best_params = merged_df.iloc[0]['params']
    best_score = merged_df.iloc[0]['mean_test_score']

    print(f'Dla modelu {model_name}:')
    print('Najlepsze hiperparametry domyślne:', best_params)
    print('Najlepszy (średni) wynik domyślny:', best_score)

    return best_params, best_score, merged_df


def get_best_result(mode, model_name, df_number):
    if mode == 'bayes':
        df = pd.read_csv(os.path.join('results_tunning', f'bayes_search_results_{model_name}_dataset_{df_number}.csv'))
    else:
        df = pd.read_csv(os.path.join('results_tunning', f'random_search_results_{model_name}_dataset_{df_number}.csv'))
        
    df_clean = df.dropna(subset=['mean_test_score']).copy()

    df_clean = df_clean.sort_values('mean_test_score', ascending=False)

    best_params = df_clean.iloc[0]['params']
    best_score = df_clean.iloc[0]['mean_test_score']

    print(f'Dla modelu {model_name} i ramki danych {df_number}:')
    print('Najlepsze hiperparametry:', best_params)
    print('Najlepszy wynik:', best_score)

    return model_name, best_params, best_score

def to_dict(params_list):
    params_dict = {}

    for param_pair in params_list:
        # Dla każdego elementu "key: value"
        key, value_str = param_pair.split(': ', 1)
        key = key.strip("'\"")  # Usuń cudzysłowy

        # Usuń np.float64() z value
        value_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', value_str)

        # Parsuj value
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str

        params_dict[key] = value
    return params_dict


def test(model, list_X_train, list_y_train, list_X_test, list_y_test, best_params):
    
    classification_reports = {}
    test_auc = {}
    for i in range(len(list_X_train)):
        X_train = list_X_train[i]
        y_train = list_y_train[i]
        X_test = list_X_test[i]
        y_test = list_y_test[i]

        model = model.__class__(**best_params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        test_auc[i] = roc_auc_score(y_test, y_pred_proba)
        print(f'Dataset {i}: Test AUC: {test_auc[i]}')
        classification_reports[i] = classification_report(y_test, y_pred)
        print(classification_reports[i])
        
    return test_auc, classification_reports


def plot_target_distribution_by_groups(data, target_col):
    group_cols = [col for col in data.columns if col != target_col and data[col].nunique() <= 10]
    
    n_plots = len(group_cols)
    n_cols = 3  
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()
    
    for i, group_col in enumerate(group_cols):
        grouped = data.groupby(group_col)[target_col].value_counts(normalize=True).unstack()
        print(f"\nRozkład klas w grupach '{group_col}':\n", grouped)
        
        grouped.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'w grupie {group_col}')
        axes[i].set_ylabel('Proporcja')
        
        contingency_table = pd.crosstab(data[group_col], data[target_col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"Chi2 dla '{group_col}': {chi2}, p-value: {p}")
        if p < 0.05:
            print(f"Istnieje statystycznie istotna różnica w rozkładzie klas między grupami w kolumnie '{group_col}' (może występować sampling bias).")
        else:
            print(f"Brak istotnych różnic w rozkładzie klas między grupami w kolumnie '{group_col}'.")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Proporcje klas w różnych grupach', y=1.02)
    plt.tight_layout()
    plt.show()
