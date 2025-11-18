from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint, loguniform
from skopt.space import Real, Integer


def get_xgboost():
    estimator = XGBClassifier(
        tree_method="hist",
        eval_metric="auc",
        random_state=42
    )

    hp_space = {
        'learning_rate': loguniform(1e-3, 0.3),
        'n_estimators': randint(100, 2000),
        'max_depth': randint(2, 12),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': loguniform(1e-8, 10)
    }

    return estimator, hp_space


def get_lightgbm():
    estimator = LGBMClassifier(
        objective='binary',
        verbosity=-1,
        random_state=42,
        n_jobs=-1
    )
    hp_space = {
        'learning_rate': loguniform(1e-3, 0.3),
        'n_estimators': randint(100, 2000),
        'max_depth': randint(2, 12),
        'subsample': uniform(0.5, 0.5),
        'feature_fraction': uniform(0.5, 0.5),
        'num_leaves': randint(16, 256)
    }

    return estimator, hp_space


def get_catboost():
    estimator = CatBoostClassifier(
        verbose=0,
        random_state=42
    )

    hp_space = {
        'learning_rate': loguniform(1e-3, 0.3),
        'n_estimators': randint(100, 2000),
        'max_depth': randint(2, 12),
        'subsample': uniform(0.5, 0.5),
        'colsample_bylevel': uniform(0.5, 0.5),
        'l2_leaf_reg': loguniform(0.1, 10)
    }

    return estimator, hp_space
