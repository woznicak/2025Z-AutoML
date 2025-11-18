import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

RESULTS_FILE = r"results/usage_100/optimization_results.pkl"
ALGORITHM = "random_forest"
PARAM_X = "n_estimators"
PARAM_Y = "min_samples_split"

with open(RESULTS_FILE, "rb") as f:
    data = pickle.load(f)
trial_results = data["trial_results"]

algorithm_keys = [key for key in trial_results.keys() if key.startswith(ALGORITHM)]
fig, axes = plt.subplots(1, len(algorithm_keys), figsize=(5*len(algorithm_keys), 5))

if len(algorithm_keys) == 1:
    axes = [axes]

for idx, study_key in enumerate(algorithm_keys):
    trials = trial_results[study_key]

    param_x_values = []
    param_y_values = []
    auc_scores = []
    trial_counts = []

    for trial in trials:
        params = trial.get("params", {})
        auc = trial.get("mean_auc", 0.0)

        param_x = params.get(PARAM_X)
        param_y = params.get(PARAM_Y)

        if param_x is not None and param_y is not None:
            param_x_values.append(param_x)
            param_y_values.append(param_y)
            auc_scores.append(auc)
            trial_counts.append(1)

    x_range = max(param_x_values) - min(param_x_values)
    y_range = max(param_y_values) - min(param_y_values)

    np.random.seed(42)
    param_x_jittered = [x + np.random.normal(0, x_range * 0.02) for x in param_x_values]
    param_y_jittered = [y + np.random.normal(0, y_range * 0.02) for y in param_y_values]

    param_x_values = param_x_jittered
    param_y_values = param_y_jittered

    auc_array = np.array(auc_scores)
    norm = Normalize(vmin=auc_array.min(), vmax=auc_array.max())
    cmap = plt.cm.get_cmap('RdYlGn')

    ax = axes[idx]

    scatter = ax.scatter(
        param_x_values,
        param_y_values,
        c=auc_scores,
        cmap=cmap,
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        norm=norm
    )

    ax.set_xlabel(PARAM_X, fontsize=11)
    ax.set_ylabel(PARAM_Y, fontsize=11)

    sampling_method = study_key.split('_')[2]

    title_suffix = f"({len(trials)} trials)"
    
    ax.set_title(f'{ALGORITHM.replace("_", " ").title()} - {sampling_method.capitalize()} Sampling\n{title_suffix}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean AUC', fontsize=10)

plt.tight_layout()
filename = f'results/parameter_exploration/{ALGORITHM}_parameter_exploration_{PARAM_X}_vs_{PARAM_Y}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as '{filename}'")
plt.show()
